"""Z-regularization: smooth expression profiles across adjacent z-levels.

Provides the core quadratic-penalty-based z-smoothing primitives extracted from
``02_3d_stack/run.py`` so that both the 2D-conditional-generator benchmark and
the 3D-transfer pipeline share a single source of truth for the regularization
math.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_factor, cho_solve




def z_penalty_matrix(
    z_values: np.ndarray,
    lambda_1: float = 1.0,
    lambda_2: float = 0.5,
) -> np.ndarray:
    """Build a quadratic penalty matrix from z-spacings.

    The penalty combines a **first-derivative** term (penalises large adjacent
    gaps) and a **second-derivative** term (penalises curvature).  The returned
    matrix ``P`` is such that ``x.T @ P @ x`` is the total penalty for a
    profile vector ``x``.

    Parameters
    ----------
    z_values : (n_nodes,) ndarray
        Sorted z-coordinates of the nodes.
    lambda_1 : float
        Weight of the first-derivative (adjacent-gap) penalty.
    lambda_2 : float
        Weight of the second-derivative (curvature) penalty.

    Returns
    -------
    penalty : (n_nodes, n_nodes) ndarray
        Quadratic penalty matrix (symmetric positive semi-definite).
    """
    z_values = np.asarray(z_values, dtype=np.float64)
    n_nodes = int(z_values.size)
    penalty = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if n_nodes < 2:
        return penalty

    if float(lambda_1) > 0.0:
        for idx in range(n_nodes - 1):
            dz = float(z_values[idx + 1] - z_values[idx])
            if dz <= 0.0:
                continue
            row = np.zeros(n_nodes, dtype=np.float64)
            row[idx] = -1.0 / dz
            row[idx + 1] = 1.0 / dz
            penalty += float(lambda_1) * np.outer(row, row)

    if float(lambda_2) > 0.0 and n_nodes >= 3:
        for idx in range(1, n_nodes - 1):
            left_dz = float(z_values[idx] - z_values[idx - 1])
            right_dz = float(z_values[idx + 1] - z_values[idx])
            span = float(z_values[idx + 1] - z_values[idx - 1])
            if left_dz <= 0.0 or right_dz <= 0.0 or span <= 0.0:
                continue
            row = np.zeros(n_nodes, dtype=np.float64)
            row[idx - 1] = 2.0 / (span * left_dz)
            row[idx] = -2.0 / span * (1.0 / right_dz + 1.0 / left_dz)
            row[idx + 1] = 2.0 / (span * right_dz)
            penalty += float(lambda_2) * np.outer(row, row)
    return penalty




def class_anchor_weight(n_spots: int, multiplier: float = 1.0) -> float:
    """Per-class anchor weight based on the number of supporting spots.

    ``multiplier * max(1.0, log1p(n_spots))``
    """
    return float(multiplier) * max(1.0, math.log1p(max(0, int(n_spots))))




def regularize_mean_profiles(
    y_matrix: np.ndarray,
    node_weights: np.ndarray,
    z_values: np.ndarray,
    lambda_1: float = 1.0,
    lambda_2: float = 0.5,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Solve for smooth expression profiles via quadratic-penalty regression.

    Parameters
    ----------
    y_matrix : (n_nodes, n_features) ndarray
        Log1p-transformed mean expression per node.  Nodes with zero weight
        may contain arbitrary values (they are ignored by the diagonal
        weighting).
    node_weights : (n_nodes,) ndarray
        Per-node anchor weight.  Larger values pull the solution closer to
        the observed ``y_matrix`` value at that node.
    z_values : (n_nodes,) ndarray
        Sorted z-coordinate of each node.
    lambda_1 : float
        First-derivative penalty strength (see :func:`z_penalty_matrix`).
    lambda_2 : float
        Second-derivative penalty strength (see :func:`z_penalty_matrix`).
    ridge : float
        Small ridge added to the diagonal of the system to guarantee positive
        definiteness.

    Returns
    -------
    solved_mean : (n_nodes, n_features) ndarray
        Regularized mean expression in **original** space (``expm1`` has been
        applied).
    """
    penalty = z_penalty_matrix(z_values, lambda_1, lambda_2)
    weights = np.asarray(node_weights, dtype=np.float64)
    system = penalty + np.diag(weights + float(ridge))
    rhs = weights[:, None] * np.asarray(y_matrix, dtype=np.float64)
    try:
        factor = cho_factor(system, lower=True, check_finite=False)
        solved_log = cho_solve(factor, rhs, check_finite=False)
    except (LinAlgError, ValueError):
        solved_log = np.linalg.solve(system, rhs)
    return np.expm1(np.clip(solved_log, 0.0, None))




def calibrate_counts_to_regularized_means(
    *,
    counts: np.ndarray,
    labels: np.ndarray,
    gene_names: Sequence[str],
    target_regularized: pd.DataFrame,
) -> np.ndarray:
    """Scale per-spot counts so that per-class per-gene means match targets.

    Parameters
    ----------
    counts : (n_spots, n_genes) ndarray
        Dense count matrix (float, will be cast internally).
    labels : (n_spots,) ndarray
        Class label for each spot.
    gene_names : sequence of str
        Gene identifiers, order must match columns of ``counts``.
    target_regularized : pd.DataFrame
        DataFrame with at least columns ``class``, ``gene``, ``regularized_mean``.
        Rows whose ``regularized_mean`` is non-finite or negative are skipped.

    Returns
    -------
    calibrated : (n_spots, n_genes) ndarray (int32)
        Rescaled integer count matrix.
    """
    out = np.asarray(counts, dtype=np.float64).copy()
    labels = np.asarray(labels).astype(str)
    gene_index = pd.Index(list(map(str, gene_names)))
    for class_name, group in target_regularized.groupby("class", sort=False):
        mask = labels == str(class_name)
        if not np.any(mask):
            continue
        desired = group.set_index("gene")["regularized_mean"].reindex(gene_index).to_numpy(dtype=np.float64)
        valid = np.isfinite(desired) & (desired >= 0.0)
        if not np.any(valid):
            continue
        class_counts = out[mask, :]
        current = class_counts.mean(axis=0)
        scale = np.ones_like(current)
        positive = valid & (current > 0.0)
        scale[positive] = desired[positive] / current[positive]
        class_counts[:, positive] *= scale[positive]

        zero_to_positive = valid & (current <= 0.0) & (desired > 0.0)
        if np.any(zero_to_positive):
            class_counts[:, zero_to_positive] = desired[zero_to_positive]
        out[mask, :] = class_counts
    return np.rint(np.clip(out, 0.0, None)).astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Pearson helper (package-private)
# ---------------------------------------------------------------------------


def _safe_pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson correlation with safe handling of constant / all-nan vectors."""
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_arr) & np.isfinite(right_arr)
    if int(valid.sum()) < 2:
        return float("nan")
    left_arr = left_arr[valid]
    right_arr = right_arr[valid]
    left_centered = left_arr - left_arr.mean()
    right_centered = right_arr - right_arr.mean()
    denominator = math.sqrt(float(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)




def compute_z_coherence(
    records: pd.DataFrame,
    *,
    left_col: str = "generated_mean",
    right_col: str = "target_mean",
) -> pd.DataFrame:
    """Compute per-gene per-class z-coherence (Pearson correlation across z-levels).

    Parameters
    ----------
    records : pd.DataFrame
        Must contain columns ``class``, ``gene``, ``target_z``, and the two
        value columns named by *left_col* and *right_col*.
    left_col : str
        Column name for the first series (e.g. generated means).
    right_col : str
        Column name for the second series (e.g. target means).

    Returns
    -------
    pd.DataFrame
        Columns: ``class``, ``gene``, ``n_slices``, ``z_min``, ``z_max``,
        ``z_coherence``.
    """
    columns = ["class", "gene", "n_slices", "z_min", "z_max", "z_coherence"]
    if records.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (class_name, gene), group in records.groupby(["class", "gene"], sort=True):
        group = group.sort_values("target_z")
        if len(group) < 3:
            continue
        corr = _safe_pearson(group[left_col].to_numpy(dtype=float), group[right_col].to_numpy(dtype=float))
        if not np.isfinite(corr):
            continue
        rows.append(
            {
                "class": str(class_name),
                "gene": str(gene),
                "n_slices": int(len(group)),
                "z_min": float(group["target_z"].min()),
                "z_max": float(group["target_z"].max()),
                "z_coherence": float(corr),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_z_coherence_frame(
    z_coherence_df: pd.DataFrame,
    *,
    prefix: str,
) -> dict[str, Any]:
    """Summary statistics from a z-coherence DataFrame.

    Parameters
    ----------
    z_coherence_df : pd.DataFrame
        Output of :func:`compute_z_coherence`.
    prefix : str
        Key prefix for the returned dictionary entries.

    Returns
    -------
    dict
        Keys: ``{prefix}_median``, ``{prefix}_mean``, ``{prefix}_n_pairs``,
        ``{prefix}_by_class``.
    """
    if z_coherence_df.empty:
        return {
            f"{prefix}_median": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_n_pairs": 0,
            f"{prefix}_by_class": {},
        }
    z_values = z_coherence_df["z_coherence"].to_numpy(dtype=float)
    finite = z_values[np.isfinite(z_values)]

    def _safe_nanmedian(values: np.ndarray) -> float:
        arr = values[np.isfinite(values)]
        if arr.size == 0:
            return float("nan")
        return float(np.median(arr))

    return {
        f"{prefix}_median": float(np.median(finite)) if finite.size else float("nan"),
        f"{prefix}_mean": float(np.mean(finite)) if finite.size else float("nan"),
        f"{prefix}_n_pairs": int(len(z_coherence_df)),
        f"{prefix}_by_class": {
            str(class_name): _safe_nanmedian(group["z_coherence"].to_numpy(dtype=float))
            for class_name, group in z_coherence_df.groupby("class", sort=True)
        },
    }
