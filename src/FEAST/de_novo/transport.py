"""Shared spatial optimal-transport backbone for FEAST simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ._ot_transport import sinkhorn_transport
from .quantile_field import quantiles_to_normal_scores, transport_latent_scores


DEFAULT_MAX_TRANSPORT_PAIRS = 25_000_000
TRANSPORT_MEMORY_BYTES_PER_PAIR = 64


@dataclass
class TransportConfig:
    """Configuration for the shared reference-to-target OT field transport."""

    epsilon: float = 0.05
    sinkhorn_iter: int = 1000
    sinkhorn_tol: float = 1e-5
    unbalanced_transport: bool = True
    reg_m: float = 5.0
    transport_nonconvergence: str = "raise"
    geometry_weight: float = 1.0
    boundary_weight: float = 0.25
    assignment_randomness: float = 0.0
    latent_clip_eps: float = 1e-6
    gene_chunk_size: int = 512
    max_transport_pairs: Optional[int] = DEFAULT_MAX_TRANSPORT_PAIRS


@dataclass
class TransportResult:
    """Transported latent field and auditable solver diagnostics."""

    latent_scores: np.ndarray
    diagnostics: Dict[str, Any]


def max_pairs_from_memory_budget(memory_budget_gb: Optional[float]) -> Optional[int]:
    """Convert an optional memory budget to a dense source-target pair cap."""

    if memory_budget_gb is None:
        return None
    budget = float(memory_budget_gb)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("memory_budget_gb must be a positive finite number.")
    return max(1, int((budget * 1024**3) // TRANSPORT_MEMORY_BYTES_PER_PAIR))


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Center and scale coordinates independently along each spatial axis."""

    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 2:
        raise ValueError("coordinates must be a 2D array.")
    if coords.shape[0] == 0:
        return coords.copy()
    center = coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    scale[scale <= 1e-6] = 1.0
    return (coords - center) / scale


def transport_reference_field(
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    source_quantiles: np.ndarray,
    *,
    source_boundary: Optional[np.ndarray] = None,
    target_boundary: Optional[np.ndarray] = None,
    config: Optional[TransportConfig] = None,
    random_seed: Optional[int] = 0,
) -> TransportResult:
    """Transport a reference quantile field onto target coordinates.

    Dense and blocked execution use the same cost, solver, latent-score
    transport, and assignment-randomness definition. Blocking partitions only
    the target columns while retaining the full reference support.
    """

    cfg = config or TransportConfig()
    _validate_config(cfg)

    source_coords = np.asarray(source_coordinates, dtype=np.float32)
    target_coords = np.asarray(target_coordinates, dtype=np.float32)
    quantiles = np.asarray(source_quantiles, dtype=np.float32)
    if source_coords.ndim != 2 or target_coords.ndim != 2:
        raise ValueError("source_coordinates and target_coordinates must be 2D arrays.")
    if not np.all(np.isfinite(source_coords)) or not np.all(np.isfinite(target_coords)):
        raise ValueError("source and target coordinates must be finite.")
    if source_coords.shape[1] != target_coords.shape[1]:
        raise ValueError("source and target coordinate dimensionality must match.")
    if quantiles.ndim != 2 or quantiles.shape[0] != source_coords.shape[0]:
        raise ValueError("source_quantiles rows must match source_coordinates rows.")
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("source_quantiles must be finite.")
    if np.any(quantiles < 0.0) or np.any(quantiles > 1.0):
        raise ValueError("source_quantiles must lie in [0, 1].")
    if source_coords.shape[0] == 0:
        raise ValueError("source_coordinates must contain at least one spot.")

    source_boundary_arr = _boundary_array(source_boundary, source_coords.shape[0], "source_boundary")
    target_boundary_arr = _boundary_array(target_boundary, target_coords.shape[0], "target_boundary")
    n_target = int(target_coords.shape[0])
    n_genes = int(quantiles.shape[1])
    if n_target == 0:
        return TransportResult(
            latent_scores=np.zeros((0, n_genes), dtype=np.float32),
            diagnostics=_aggregate_diagnostics([], cfg, blocked=False),
        )

    source_scores = quantiles_to_normal_scores(
        quantiles,
        clip_eps=float(cfg.latent_clip_eps),
    ).astype(np.float32, copy=False)
    target_blocks = list(
        _target_blocks(
            source_coords.shape[0],
            target_coords,
            cfg.max_transport_pairs,
        )
    )
    transported = np.zeros((n_target, n_genes), dtype=np.float32)
    block_diagnostics = []
    for block_index, target_indices in enumerate(target_blocks):
        plan, diagnostics = _solve_transport_plan(
            source_coords,
            target_coords[target_indices],
            source_boundary_arr,
            target_boundary_arr[target_indices],
            cfg,
        )
        for start in range(0, n_genes, max(1, int(cfg.gene_chunk_size))):
            end = min(start + max(1, int(cfg.gene_chunk_size)), n_genes)
            transported[target_indices, start:end] = transport_latent_scores(
                plan,
                source_scores[:, start:end],
            ).astype(np.float32, copy=False)
        block_diagnostics.append(
            {
                **diagnostics,
                "block_index": int(block_index),
                "source_spots": int(source_coords.shape[0]),
                "target_spots": int(target_indices.size),
                "transport_mass": float(plan.sum(dtype=np.float64)),
            }
        )

    randomness = float(cfg.assignment_randomness)
    if randomness > 0.0:
        rng = np.random.default_rng(random_seed)
        sampled_indices = rng.integers(0, source_scores.shape[0], size=n_target)
        transported = (
            (1.0 - randomness) * transported
            + randomness * source_scores[sampled_indices, :]
        ).astype(np.float32, copy=False)

    return TransportResult(
        latent_scores=transported,
        diagnostics=_aggregate_diagnostics(
            block_diagnostics,
            cfg,
            blocked=len(target_blocks) > 1,
        ),
    )


def _validate_config(config: TransportConfig) -> None:
    if not 0.0 <= float(config.assignment_randomness) <= 1.0:
        raise ValueError("assignment_randomness must be between 0 and 1 inclusive.")
    if not np.isfinite(float(config.epsilon)) or float(config.epsilon) <= 0.0:
        raise ValueError("epsilon must be positive and finite.")
    if int(config.sinkhorn_iter) < 1:
        raise ValueError("sinkhorn_iter must be a positive integer.")
    if not np.isfinite(float(config.sinkhorn_tol)) or float(config.sinkhorn_tol) <= 0.0:
        raise ValueError("sinkhorn_tol must be positive and finite.")
    if str(config.transport_nonconvergence) not in {"raise", "warn"}:
        raise ValueError("transport_nonconvergence must be 'raise' or 'warn'.")
    if not np.isfinite(float(config.geometry_weight)) or float(config.geometry_weight) < 0.0:
        raise ValueError("geometry_weight must be a non-negative finite number.")
    if not np.isfinite(float(config.boundary_weight)) or float(config.boundary_weight) < 0.0:
        raise ValueError("boundary_weight must be a non-negative finite number.")
    if bool(config.unbalanced_transport) and (
        not np.isfinite(float(config.reg_m)) or float(config.reg_m) <= 0.0
    ):
        raise ValueError("reg_m must be positive and finite for unbalanced transport.")
    if int(config.gene_chunk_size) < 1:
        raise ValueError("gene_chunk_size must be a positive integer.")
    if config.max_transport_pairs is not None and int(config.max_transport_pairs) < 1:
        raise ValueError("max_transport_pairs must be a positive integer or None.")


def _boundary_array(values: Optional[np.ndarray], size: int, name: str) -> np.ndarray:
    if values is None:
        return np.zeros(size, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] != size:
        raise ValueError(f"{name} length must match its coordinate row count.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _target_blocks(
    n_source: int,
    target_coordinates: np.ndarray,
    max_transport_pairs: Optional[int],
):
    n_target = int(target_coordinates.shape[0])
    if max_transport_pairs is None or n_source * n_target <= int(max_transport_pairs):
        yield np.arange(n_target, dtype=np.int64)
        return

    target_block_size = int(max_transport_pairs) // int(n_source)
    if target_block_size < 1:
        raise ValueError(
            "max_transport_pairs is smaller than the number of reference spots; "
            "increase the pair cap or memory budget."
        )
    axis = int(np.argmax(np.ptp(target_coordinates, axis=0)))
    order = np.argsort(target_coordinates[:, axis], kind="mergesort")
    for start in range(0, n_target, target_block_size):
        yield order[start : start + target_block_size]


def _solve_transport_plan(
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
    config: TransportConfig,
):
    source_sq = np.sum(source_coordinates**2, axis=1, keepdims=True)
    target_sq = np.sum(target_coordinates**2, axis=1, keepdims=True).T
    distance_squared = np.maximum(
        source_sq + target_sq - 2.0 * source_coordinates @ target_coordinates.T,
        0.0,
    )
    boundary_cost = np.abs(source_boundary[:, None] - target_boundary[None, :])
    cost = (
        float(config.geometry_weight) * distance_squared
        + float(config.boundary_weight) * boundary_cost
    )
    source_mass = np.full(
        source_coordinates.shape[0],
        1.0 / source_coordinates.shape[0],
        dtype=np.float32,
    )
    target_mass = np.full(
        target_coordinates.shape[0],
        1.0 / target_coordinates.shape[0],
        dtype=np.float32,
    )
    return sinkhorn_transport(
        M=cost,
        a=source_mass,
        b=target_mass,
        reg=float(config.epsilon),
        numItermax=int(config.sinkhorn_iter),
        stopThr=float(config.sinkhorn_tol),
        unbalanced=bool(config.unbalanced_transport),
        reg_m=float(config.reg_m),
        nonconvergence=str(config.transport_nonconvergence),
        return_diagnostics=True,
    )


def _aggregate_diagnostics(
    blocks: list[Dict[str, Any]],
    config: TransportConfig,
    *,
    blocked: bool,
) -> Dict[str, Any]:
    final_errors = [
        float(block["final_error"])
        for block in blocks
        if block.get("final_error") is not None
        and np.isfinite(float(block["final_error"]))
    ]
    iterations = [
        int(block["iterations"])
        for block in blocks
        if block.get("iterations") is not None
    ]
    return {
        "converged": bool(all(bool(block.get("converged", False)) for block in blocks)),
        "iterations": max(iterations) if iterations else 0,
        "final_error": max(final_errors) if final_errors else None,
        "stop_threshold": float(config.sinkhorn_tol),
        "max_iterations": int(config.sinkhorn_iter),
        "unbalanced": bool(config.unbalanced_transport),
        "epsilon": float(config.epsilon),
        "reg_m": float(config.reg_m),
        "geometry_weight": float(config.geometry_weight),
        "boundary_weight": float(config.boundary_weight),
        "assignment_randomness": float(config.assignment_randomness),
        "field_space": "normal_score",
        "blocked": bool(blocked),
        "n_blocks": int(len(blocks)),
        "max_transport_pairs": (
            None
            if config.max_transport_pairs is None
            else int(config.max_transport_pairs)
        ),
        "transport_mass": float(
            sum(float(block.get("transport_mass", 0.0)) for block in blocks)
        ),
        "blocks": blocks,
    }
