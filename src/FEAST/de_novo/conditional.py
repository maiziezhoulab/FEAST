from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, replace
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..FEAST_core.count_decoding import decode_counts_by_spatial_intensity
from ._metadata import (
    encode_blueprint_h5ad_metadata,
    encode_feast_h5ad_metadata,
    records_by_label_to_h5ad_uns,
)
from .core import SliceBlueprint, active_mask_metadata, assign_generated_coordinates, load_blueprint
from .pattern import diffuse_quantile_map
from .quantile_field import (
    combine_weighted_arrays,
    rank_normalize_by_scope,
    reference_conflict_score,
    resolve_auto_rank_scope,
    should_store_quantiles,
    validate_quantile_field_config,
    weighted_stats_log_space,
    QuantileFieldConfig,
)
from .transport import TransportConfig, normalize_coordinates, transport_reference_field


@dataclass
class ReferenceFitConfig:
    min_gene_spots: int = 20
    min_gene_mean: float = 0.1
    max_gene_zero_prop: float = 0.98
    boundary_neighbors: int = 6
    coordinate_scale: Optional[Sequence[float]] = None


@dataclass
class SimulationConfig(TransportConfig):
    """Reference-conditioned generation config built on shared OT settings."""

    reference_weight_eta: float = 4.0
    boundary_multiplier: float = 1.1
    diffusion_level: float = 0.0
    boundary_softness: float = 0.0
    coordinate_scale: Optional[Sequence[float]] = None
    verbose: bool = False
    quantile_field_mode: str = "auto"
    rank_scope: str = "auto"
    target_parameter_mode: str = "reference_weighted_log"
    tie_policy: str = "stable_ordinal"
    tie_jitter_scale: float = 1e-9
    min_rank_scope_size: int = 20
    store_latent_scores: bool = False
    store_quantiles: Any = "auto"
    max_stored_quantile_elements: int = 50_000_000
    reference_conflict_policy: str = "average"
    program_noise_scale: float = 0.0
    program_normalization: str = "zscore"


def _quantile_field_config(config: SimulationConfig) -> QuantileFieldConfig:
    return validate_quantile_field_config(
        QuantileFieldConfig(
            mode=str(config.quantile_field_mode),
            rank_scope=str(config.rank_scope),
            target_parameter_mode=str(config.target_parameter_mode),
            tie_policy=str(config.tie_policy),
            latent_clip_eps=float(config.latent_clip_eps),
            tie_jitter_scale=float(config.tie_jitter_scale),
            min_rank_scope_size=int(config.min_rank_scope_size),
            gene_chunk_size=int(config.gene_chunk_size),
            store_latent_scores=bool(config.store_latent_scores),
            store_quantiles=config.store_quantiles,
            max_stored_quantile_elements=int(config.max_stored_quantile_elements),
            reference_conflict_policy=str(config.reference_conflict_policy),
            program_noise_scale=float(config.program_noise_scale),
            program_normalization=str(config.program_normalization),
        )
    )


def _simulation_config(config: Optional[TransportConfig]) -> SimulationConfig:
    if config is None:
        return SimulationConfig()
    if isinstance(config, SimulationConfig):
        return config
    if not isinstance(config, TransportConfig):
        raise TypeError("config must be a TransportConfig or SimulationConfig.")
    values = {
        item.name: getattr(config, item.name)
        for item in dataclass_fields(TransportConfig)
    }
    return SimulationConfig(**values)


@dataclass
class ReferenceLabelData:
    label: str
    coordinates: np.ndarray
    normalized_coordinates: np.ndarray
    boundary_scores: np.ndarray
    quantiles: np.ndarray
    stats: pd.DataFrame


@dataclass
class ReferenceSliceData:
    reference_name: str
    adata: ad.AnnData
    labels: Dict[str, ReferenceLabelData]


@dataclass
class SimulationReference:
    gene_names: List[str]
    label_key: str
    references: List[ReferenceSliceData]
    coordinate_dim: int
    fit_config: ReferenceFitConfig = field(default_factory=ReferenceFitConfig)
    reference_metadata: Dict[str, Any] = field(default_factory=dict)


def fit_reference(
    reference_slices: Union[ad.AnnData, Sequence[ad.AnnData]],
    label_key: str,
    config: Optional[ReferenceFitConfig] = None,
) -> SimulationReference:
    fit_cfg = config or ReferenceFitConfig()
    slices = _normalize_reference_slices(reference_slices)
    common_genes = _common_gene_names(slices)
    if not common_genes:
        raise ValueError("reference_slices must share at least one gene.")

    filtered_genes = _filter_common_genes(slices, common_genes, fit_cfg)
    if not filtered_genes:
        raise ValueError("No genes remain after conditional reference filtering.")

    references: List[ReferenceSliceData] = []
    label_set: set[str] = set()
    coordinate_dim: Optional[int] = None
    for idx, adata in enumerate(slices):
        if label_key not in adata.obs:
            raise KeyError(f"Reference slice {idx} is missing label_key '{label_key}'.")
        subset = adata[:, filtered_genes].copy()
        if "counts" not in subset.layers:
            subset.layers["counts"] = _to_dense_matrix(subset.X).astype(np.float32, copy=False)
        labels = _label_series(subset, label_key)
        coords = _coordinates(subset)
        if coordinate_dim is None:
            coordinate_dim = int(coords.shape[1])
        elif int(coords.shape[1]) != coordinate_dim:
            raise ValueError("Reference slices must use the same coordinate dimensionality.")
        scaled_coords = _apply_coordinate_scale(coords, fit_cfg.coordinate_scale)
        boundary_scores = _boundary_scores(scaled_coords, labels.to_numpy(), fit_cfg.boundary_neighbors)

        label_map: Dict[str, ReferenceLabelData] = {}
        for label in sorted(labels.unique()):
            mask = labels.to_numpy() == label
            label_coords = coords[mask]
            label_scaled_coords = scaled_coords[mask]
            label_counts = _layer_counts_matrix(subset, mask)
            label_quantiles = _calculate_quantiles(label_counts)
            label_map[label] = ReferenceLabelData(
                label=label,
                coordinates=label_coords,
                normalized_coordinates=normalize_coordinates(label_scaled_coords),
                boundary_scores=boundary_scores[mask],
                quantiles=label_quantiles,
                stats=_stats_dataframe(label_counts, filtered_genes),
            )
            label_set.add(label)

        references.append(
            ReferenceSliceData(
                reference_name=_reference_name(subset, idx),
                adata=subset,
                labels=label_map,
            )
        )

    return SimulationReference(
        gene_names=list(filtered_genes),
        label_key=label_key,
        references=references,
        coordinate_dim=int(coordinate_dim or 2),
        fit_config=fit_cfg,
        reference_metadata={
            "n_reference_slices": len(references),
            "n_reference_genes": len(filtered_genes),
            "labels": sorted(label_set),
            "coordinate_dim": int(coordinate_dim or 2),
            "coordinate_scale": None if fit_cfg.coordinate_scale is None else [float(v) for v in fit_cfg.coordinate_scale],
        },
    )


def simulate_from_reference(
    model: SimulationReference,
    target_blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str, Path],
    parameter_cloud: Optional[Union[pd.DataFrame, Mapping[str, Any]]] = None,
    config: Optional[TransportConfig] = None,
    random_seed: int = 0,
    reference_weights: Optional[Mapping[str, float]] = None,
    marginal_model: Optional[str] = None,
    *,
    count_model_params: Optional[Mapping[str, dict]] = None,
    group_reference_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    count_seed: Optional[int] = None,
) -> ad.AnnData:
    gen_cfg = _simulation_config(config)
    resolved_marginal_model = (
        str(marginal_model)
        if marginal_model is not None
        else ("core" if count_model_params is not None else "parameter_cloud" if parameter_cloud is not None else "empirical_reference")
    )
    if resolved_marginal_model not in {"empirical_reference", "parameter_cloud", "core"}:
        raise ValueError(
            "marginal_model must be 'empirical_reference' or 'parameter_cloud'."
        )
    if resolved_marginal_model == "empirical_reference" and parameter_cloud is not None:
        raise ValueError(
            "parameter_cloud cannot be supplied when marginal_model='empirical_reference'."
        )
    if resolved_marginal_model == "core" and count_model_params is None:
        raise ValueError("core generation requires group-indexed count_model_params")
    if count_model_params is not None and resolved_marginal_model != "core":
        raise ValueError("count_model_params requires marginal_model='core'")
    q_cfg = _quantile_field_config(gen_cfg)
    if q_cfg.mode == "auto":
        quantile_field_mode = "latent_reference"
    elif q_cfg.mode == "latent_reference":
        quantile_field_mode = q_cfg.mode
    else:
        raise ValueError(
            "Reference-conditioned generation supports quantile_field_mode "
            "'auto' or 'latent_reference'."
        )
    original_blueprint = _load_target_blueprint(target_blueprint, label_key=model.label_key)
    mask_metadata = active_mask_metadata(original_blueprint)
    blueprint = original_blueprint.active_subset()
    target_labels = _blueprint_labels(blueprint)
    target_coords_raw = np.asarray(blueprint.coordinates, dtype=float)
    if target_coords_raw.shape[1] != int(model.coordinate_dim):
        raise ValueError(
            "Target blueprint coordinate dimensionality does not match the reference model "
            f"({target_coords_raw.shape[1]} != {model.coordinate_dim})."
        )
    coordinate_scale = gen_cfg.coordinate_scale if gen_cfg.coordinate_scale is not None else model.fit_config.coordinate_scale
    target_coords = _apply_coordinate_scale(target_coords_raw, coordinate_scale)
    target_boundary_scores = _boundary_scores(
        target_coords,
        target_labels,
        model.fit_config.boundary_neighbors,
    )

    n_spots = blueprint.n_spots
    n_genes = len(model.gene_names)
    quantiles = np.zeros((n_spots, n_genes), dtype=np.float32)
    counts = np.zeros((n_spots, n_genes), dtype=np.int32)
    label_weights_out: Dict[str, Dict[str, float]] = {}
    label_cloud_out: Dict[str, Dict[str, Any]] = {}
    label_clouds: Dict[str, pd.DataFrame] = {}
    transport_diagnostics: Dict[str, List[Dict[str, Any]]] = {}
    quantile_field_labels: Dict[str, Dict[str, Any]] = {}
    combined_latent_scores = np.zeros((n_spots, n_genes), dtype=np.float32)
    latent_scores_store = np.zeros((n_spots, n_genes), dtype=np.float32) if q_cfg.store_latent_scores else None

    unique_labels = sorted(set(target_labels.tolist()))
    aligned_count_models = {}
    if count_model_params is not None:
        for label in unique_labels:
            params = count_model_params[label]
            if list(params['genes'].values()) != list(model.gene_names):
                raise ValueError('precomputed count-model gene order differs from reference genes')
            # Keep caller-owned parameters unchanged. Both metadata and decoding
            # consume this same gene-aligned table, without changing its values.
            stats = params['target_stats']
            if 'gene_id' in stats.columns:
                stats = stats.set_index('gene_id')
            aligned_count_models[label] = {
                **params, 'target_stats': _align_stats_frame(stats, model.gene_names)}
    transport_seconds = 0.0
    decode_seconds = 0.0
    count_diagnostics = {}
    transport_index = 0
    for label in unique_labels:
        target_mask = target_labels == label
        target_indices = np.where(target_mask)[0]
        target_coords_label = target_coords[target_mask]
        target_boundary_label = target_boundary_scores[target_mask]
        target_coords_norm = normalize_coordinates(target_coords_label)

        eligible_refs = [ref for ref in model.references if label in ref.labels]
        if not eligible_refs:
            raise ValueError(f"No reference slice contains target label '{label}'.")

        applicable_weights = (group_reference_weights[label]
                              if group_reference_weights is not None else reference_weights)
        if applicable_weights is not None:
            eligible_refs = [ref for ref in eligible_refs
                             if applicable_weights.get(ref.reference_name, 0.0) > 0]
        if applicable_weights is None:
            ref_weights = _reference_weights_for_label(
                eligible_refs,
                label,
                target_coords_norm,
                target_boundary_label,
                gen_cfg.reference_weight_eta,
            )
        else:
            ref_weights = _fixed_reference_weights_for_label(eligible_refs, applicable_weights)
        label_weights_out[label] = ref_weights

        transported_parts = []
        label_diags: List[Dict[str, Any]] = []
        part_weights: list[float] = []
        for ref in eligible_refs:
            ref_label = ref.labels[label]
            transport_started = time.perf_counter()
            transport_result = transport_reference_field(
                source_coordinates=ref_label.normalized_coordinates,
                target_coordinates=target_coords_norm,
                source_quantiles=ref_label.quantiles,
                source_boundary=ref_label.boundary_scores,
                target_boundary=target_boundary_label,
                config=gen_cfg,
                random_seed=int(random_seed) + transport_index,
            )
            transport_seconds += time.perf_counter() - transport_started
            transport_index += 1
            part = transport_result.latent_scores
            solve_diagnostics = transport_result.diagnostics
            transported_parts.append(part)
            part_weights.append(float(ref_weights[ref.reference_name]))
            ref_conflict = reference_conflict_score(
                transported_parts,
                part_weights,
            )
            label_diags.append(
                {
                    "reference_name": ref.reference_name,
                    "transport_mass": float(solve_diagnostics["transport_mass"]),
                    "source_spots": int(ref_label.quantiles.shape[0]),
                    "target_spots": int(len(target_indices)),
                    "assignment_randomness": float(gen_cfg.assignment_randomness),
                    "quantile_field_mode": quantile_field_mode,
                    "reference_conflict_score": float(ref_conflict),
                    "transport_converged": bool(solve_diagnostics["converged"]),
                    "transport_iterations": solve_diagnostics["iterations"],
                    "transport_final_error": solve_diagnostics["final_error"],
                    "transport_stop_threshold": solve_diagnostics["stop_threshold"],
                    "transport_max_iterations": solve_diagnostics["max_iterations"],
                    "transport_nonconvergence_policy": str(
                        gen_cfg.transport_nonconvergence
                    ),
                    "transport_blocked": bool(solve_diagnostics["blocked"]),
                    "transport_blocks": int(solve_diagnostics["n_blocks"]),
                    "transport_solver_method": str(solve_diagnostics["solver_method"]),
                    "transport_backend": str(solve_diagnostics["transport_backend"]),
                    "transport_device": str(solve_diagnostics["transport_device"]),
                    "transport_dtype": str(solve_diagnostics["transport_dtype"]),
                }
            )
        transport_diagnostics[label] = label_diags

        conflict = reference_conflict_score(transported_parts, part_weights)
        if q_cfg.reference_conflict_policy == "highest_weight" and conflict >= 0.9:
            chosen_idx = int(np.argmax(np.asarray(part_weights, dtype=float)))
            combined_scores = np.asarray(transported_parts[chosen_idx], dtype=np.float32)
        else:
            combined_scores = combine_weighted_arrays(transported_parts, part_weights)
        combined_latent_scores[target_indices, :] = combined_scores.astype(np.float32, copy=False)
        q_meta = {
            "reference_conflict_score": float(conflict),
            "reference_conflict_policy": str(q_cfg.reference_conflict_policy),
        }
        quantile_field_labels[label] = q_meta

        if count_model_params is not None:
            label_cloud = aligned_count_models[label]['target_stats']
        else:
            label_cloud = _resolve_parameter_cloud(
                parameter_cloud=parameter_cloud,
                label=label,
                gene_names=model.gene_names,
                eligible_refs=eligible_refs,
                reference_weights=ref_weights,
                target_parameter_mode=str(q_cfg.target_parameter_mode),
            )
        label_clouds[label] = label_cloud
        label_cloud_out[label] = {
            "mean_mean": float(label_cloud["mean"].mean()),
            "variance_mean": float(label_cloud["variance"].mean()),
            "zero_prop_mean": float(label_cloud["zero_prop"].mean()),
        }

    global_q_meta: Dict[str, Any] = {}
    resolved_scope = resolve_auto_rank_scope(
        requested_scope=str(q_cfg.rank_scope),
        coordinate_dim=int(target_coords_raw.shape[1]),
        domain_specific=True,
        stack_like=target_coords_raw.shape[1] >= 3 and len(np.unique(target_coords_raw[:, 2])) > 1,
    )
    quantiles, global_q_meta = rank_normalize_by_scope(
        combined_latent_scores,
        labels=target_labels,
        coordinates=target_coords_raw,
        rank_scope=resolved_scope,
        tie_policy=str(q_cfg.tie_policy),
        clip_eps=float(q_cfg.latent_clip_eps),
        random_seed=int(random_seed),
        tie_jitter_scale=float(q_cfg.tie_jitter_scale),
        min_rank_scope_size=int(q_cfg.min_rank_scope_size),
    )
    if latent_scores_store is not None:
        latent_scores_store[:, :] = combined_latent_scores

    quantiles = diffuse_quantile_map(
        quantiles,
        target_coords,
        float(gen_cfg.diffusion_level),
    ).astype(np.float32, copy=False)

    for label_index, label in enumerate(unique_labels):
        target_mask = target_labels == label
        label_q = quantiles[target_mask, :]
        eligible_refs = [ref for ref in model.references if label in ref.labels and label_weights_out[label].get(ref.reference_name, 0.0) > 0]
        decode_started = time.perf_counter()
        if resolved_marginal_model == "empirical_reference":
            decoded = _decode_label_aware_rank_counts(
                quantiles=label_q,
                label=label,
                label_key=model.label_key,
                gene_names=model.gene_names,
                eligible_refs=eligible_refs,
                reference_weights=label_weights_out[label],
            )
        else:
            model_params = (aligned_count_models[label] if count_model_params is not None
                            else _stats_frame_to_model_params(label_clouds[label]))
            count_diagnostics[label] = {}
            decoded = decode_counts_by_spatial_intensity(
                label_q,
                model_params,
                boundary_multiplier=float(gen_cfg.boundary_multiplier),
                reference_X=_reference_counts_for_label(
                    eligible_refs,
                    label,
                    model.label_key,
                ),
                random_seed=int(random_seed if count_seed is None else count_seed) + label_index,
                show_progress=bool(gen_cfg.verbose),
                diagnostics=count_diagnostics[label] if count_model_params is not None else None,
            )
        decode_seconds += time.perf_counter() - decode_started
        counts[target_mask, :] = np.asarray(decoded, dtype=np.int32)

    obs = blueprint.obs.copy()
    if "domain" not in obs:
        obs["domain"] = target_labels
    obs.index = [str(idx) for idx in obs.index]
    global_cloud = _aggregate_label_parameter_clouds(
        label_clouds,
        target_labels,
        model.gene_names,
    )
    var = pd.DataFrame(index=model.gene_names)
    var["target_mean"] = global_cloud["mean"].to_numpy(dtype=np.float64)
    var["target_variance"] = global_cloud["variance"].to_numpy(dtype=np.float64)
    var["target_zero_prop"] = global_cloud["zero_prop"].to_numpy(dtype=np.float64)

    result = ad.AnnData(X=counts, obs=obs, var=var)
    result.layers["counts"] = counts.astype(np.int32, copy=False)
    store_q = should_store_quantiles(
        q_cfg.store_quantiles,
        int(np.prod(quantiles.shape)),
        int(q_cfg.max_stored_quantile_elements),
    )
    if store_q:
        result.layers["feast_quantiles"] = quantiles.astype(np.float32, copy=False)
    if latent_scores_store is not None:
        result.layers["latent_scores"] = latent_scores_store.astype(np.float32, copy=False)
    result.uns["de_novo"] = encode_feast_h5ad_metadata({
        "conditional_generation": True,
        "timings": {"ot": transport_seconds, "decoding": decode_seconds},
        "count_diagnostics": count_diagnostics,
        "label_key": model.label_key,
        "reference_metadata": dict(model.reference_metadata),
        "transport_weights": label_weights_out,
        "transport_diagnostics": records_by_label_to_h5ad_uns(transport_diagnostics),
        "parameter_cloud_summary": label_cloud_out,
        "decode_method": (
            "empirical_rank"
            if resolved_marginal_model == "empirical_reference"
            else "parameter_cloud_spatial_intensity"
        ),
        "marginal_model": resolved_marginal_model,
        "quantile_calibration": "reference_rank",
        "diffusion_level": float(gen_cfg.diffusion_level),
        "boundary_softness": float(gen_cfg.boundary_softness),
        "assignment_randomness": float(gen_cfg.assignment_randomness),
        "mask": mask_metadata,
        "quantile_field": {
            "method_version": "unified_latent_ot_v1",
            "mode": quantile_field_mode,
            "source": "reference_transport",
            "requested_rank_scope": str(q_cfg.rank_scope),
            "resolved_rank_scope": global_q_meta.get("resolved_rank_scope"),
            "rank_scope_metadata": global_q_meta,
            "labels": quantile_field_labels,
            "tie_policy": str(q_cfg.tie_policy),
            "latent_clip_eps": float(q_cfg.latent_clip_eps),
            "tie_jitter_scale": float(q_cfg.tie_jitter_scale),
            "target_parameter_mode": str(q_cfg.target_parameter_mode),
            "reference_conflict_policy": str(q_cfg.reference_conflict_policy),
            "gene_chunk_size": int(q_cfg.gene_chunk_size),
            "store_latent_scores": bool(q_cfg.store_latent_scores),
            "store_quantiles": q_cfg.store_quantiles,
            "quantiles_stored": bool(store_q),
            "random_seed": int(random_seed),
        },
    })
    assign_generated_coordinates(result, target_coords_raw)
    result.uns["target_blueprint"] = encode_blueprint_h5ad_metadata(
        blueprint.to_dict()
    )
    return result


def _normalize_reference_slices(reference_slices: Union[ad.AnnData, Sequence[ad.AnnData]]) -> List[ad.AnnData]:
    if isinstance(reference_slices, ad.AnnData):
        slices = [reference_slices]
    else:
        slices = list(reference_slices)
    if not slices:
        raise ValueError("reference_slices must contain at least one AnnData object.")
    normalized: List[ad.AnnData] = []
    for adata in slices:
        if not isinstance(adata, ad.AnnData):
            raise TypeError("reference_slices must contain AnnData objects.")
        if "spatial_3d" not in adata.obsm and "spatial" not in adata.obsm:
            raise ValueError("Each reference slice must contain obsm['spatial_3d'] or obsm['spatial'].")
        normalized.append(adata)
    return normalized


def _common_gene_names(slices: Sequence[ad.AnnData]) -> List[str]:
    common = set(map(str, slices[0].var_names))
    for adata in slices[1:]:
        common &= set(map(str, adata.var_names))
    return sorted(common)


def _filter_common_genes(
    slices: Sequence[ad.AnnData],
    common_genes: Sequence[str],
    config: ReferenceFitConfig,
) -> List[str]:
    mean_values = []
    zero_props = []
    nonzero_counts = []
    for adata in slices:
        subset = adata[:, list(common_genes)]
        matrix = _counts_matrix(subset)
        nz = np.count_nonzero(matrix, axis=0).astype(float)
        mean_values.append(matrix.mean(axis=0))
        zero_props.append(1.0 - (nz / max(subset.n_obs, 1)))
        nonzero_counts.append(nz)
    mean_avg = np.mean(np.vstack(mean_values), axis=0)
    zero_avg = np.mean(np.vstack(zero_props), axis=0)
    nonzero_max = np.max(np.vstack(nonzero_counts), axis=0)
    keep = (
        (mean_avg >= float(config.min_gene_mean))
        & (zero_avg <= float(config.max_gene_zero_prop))
        & (nonzero_max >= float(config.min_gene_spots))
    )
    return [str(gene) for gene, keep_gene in zip(common_genes, keep) if keep_gene]


def _label_series(adata: ad.AnnData, label_key: str) -> pd.Series:
    return adata.obs[label_key].astype(str)


def _coordinates(adata: ad.AnnData) -> np.ndarray:
    if "spatial_3d" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial_3d"], dtype=float)
    elif "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
    else:
        raise ValueError("AnnData must contain obsm['spatial_3d'] or obsm['spatial'].")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("spatial coordinates must be a 2D array with at least two columns.")
    if coords.shape[1] > 3:
        raise ValueError("spatial coordinates supports only 2D or 3D coordinates in v1.")
    return coords.copy()


def _apply_coordinate_scale(coords: np.ndarray, coordinate_scale: Optional[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if coordinate_scale is None:
        return arr.copy()
    scale = np.asarray(coordinate_scale, dtype=float).reshape(-1)
    if scale.shape[0] != arr.shape[1]:
        raise ValueError(f"coordinate_scale must contain {arr.shape[1]} values.")
    return arr * scale[None, :]


def _to_dense_matrix(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _sparse_gene_stats(matrix, gene_names) -> "pd.DataFrame":
    """Compute per-gene mean, variance, zero_prop from a sparse matrix."""
    from scipy.sparse import issparse
    import pandas as pd
    if issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
        X_sq = matrix.copy()
        X_sq.data **= 2
        variances = np.asarray(X_sq.mean(axis=0)).ravel() - means ** 2
        n_obs = matrix.shape[0]
        nz = matrix.getnnz(axis=0) if hasattr(matrix, 'getnnz') else np.diff(matrix.tocsc().indptr)
        zero_prop = 1.0 - nz / n_obs
        out = pd.DataFrame(index=list(map(str, gene_names)))
        out["mean"] = np.clip(means, 1e-8, None)
        out["variance"] = np.clip(variances, 1e-8, None)
        out["zero_prop"] = np.clip(zero_prop, 0.0, 0.99)
        return out
    mat = np.asarray(matrix, dtype=np.float64)
    out = pd.DataFrame(index=list(map(str, gene_names)))
    out["mean"] = np.clip(mat.mean(axis=0), 1e-8, None)
    out["variance"] = np.clip(mat.var(axis=0), 1e-8, None)
    out["zero_prop"] = np.clip(np.mean(mat <= 0, axis=0), 0.0, 0.99)
    return out


def _counts_matrix(adata: ad.AnnData) -> np.ndarray:
    if "counts" in adata.layers:
        return _to_dense_matrix(adata.layers["counts"])
    return _to_dense_matrix(adata.X)


def _layer_counts_matrix(adata: ad.AnnData, mask: np.ndarray) -> np.ndarray:
    return _counts_matrix(adata)[np.asarray(mask, dtype=bool), :]


def _stats_dataframe(matrix: np.ndarray, gene_names: Sequence[str]) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=np.float64)
    out = pd.DataFrame(index=list(map(str, gene_names)))
    out["mean"] = np.clip(matrix.mean(axis=0), 1e-8, None)
    out["variance"] = np.clip(matrix.var(axis=0), 1e-8, None)
    out["zero_prop"] = np.clip(np.mean(matrix <= 0, axis=0), 0.0, 0.99)
    return out


def _calculate_quantiles(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n, p = matrix.shape
    if n == 0:
        return np.zeros((0, p), dtype=np.float32)
    if n == 1:
        return np.full((1, p), 0.5, dtype=np.float32)
    ranks = np.zeros((n, p), dtype=np.float64)
    for j in range(p):
        order = np.argsort(matrix[:, j], kind="mergesort")
        values = matrix[order, j]
        start = 0
        while start < n:
            end = start + 1
            while end < n and values[end] == values[start]:
                end += 1
            avg_rank = 0.5 * (start + end - 1)
            ranks[order[start:end], j] = (avg_rank + 0.5) / float(n)
            start = end
    return np.clip(ranks, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)


def _boundary_scores(coords: np.ndarray, labels: np.ndarray, n_neighbors: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels).astype(str)
    n_spots = coords.shape[0]
    if n_spots < 2:
        return np.zeros(n_spots, dtype=float)
    k = min(max(1, int(n_neighbors)), n_spots - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)
    scores = np.zeros(n_spots, dtype=float)
    for idx, neighbors in enumerate(indices):
        neighbor_labels = labels[neighbors[1:]]
        scores[idx] = float(np.mean(neighbor_labels != labels[idx]))
    return scores


def _reference_name(adata: ad.AnnData, index: int) -> str:
    if "reference_name" in adata.uns:
        return str(adata.uns["reference_name"])
    if "sample_name" in adata.obs:
        return str(adata.obs["sample_name"].iloc[0])
    if "sce.sample_name" in adata.obs:
        return str(adata.obs["sce.sample_name"].iloc[0])
    return f"reference_{index:03d}"


def _load_target_blueprint(
    source: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str, Path],
    label_key: str,
) -> SliceBlueprint:
    if isinstance(source, ad.AnnData) and label_key in source.obs:
        obs = source.obs.copy()
        obs["domain"] = source.obs[label_key].astype(str).to_numpy()
        return SliceBlueprint(
            coordinates=_coordinates(source),
            grid_type=source.uns.get("grid_type", "generic"),
            domain_map=obs["domain"].to_numpy(),
            technology=source.uns.get("technology"),
            obs=obs,
            metadata={"source": "anndata", "label_key": label_key},
        )
    blueprint = load_blueprint(source)
    if blueprint.domain_map is None:
        raise ValueError("Target blueprint must provide domain labels for conditional generation.")
    return blueprint


def _blueprint_labels(blueprint: SliceBlueprint) -> np.ndarray:
    return np.asarray(blueprint.domain_map).astype(str)


def _geometry_distance(
    source_coords_norm: np.ndarray,
    target_coords_norm: np.ndarray,
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
) -> float:
    source_radii = np.linalg.norm(source_coords_norm, axis=1)
    target_radii = np.linalg.norm(target_coords_norm, axis=1)
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float)
    source_q = np.quantile(source_radii, quantiles)
    target_q = np.quantile(target_radii, quantiles)
    return (
        float(np.mean(np.abs(source_q - target_q)))
        + 0.5 * float(abs(source_boundary.mean() - target_boundary.mean()))
        + 0.1 * float(abs(np.log1p(len(source_coords_norm)) - np.log1p(len(target_coords_norm))))
    )


def _reference_weights_for_label(
    references: Sequence[ReferenceSliceData],
    label: str,
    target_coords_norm: np.ndarray,
    target_boundary_scores: np.ndarray,
    eta: float,
) -> Dict[str, float]:
    scores = []
    names = []
    for ref in references:
        ref_label = ref.labels[label]
        names.append(ref.reference_name)
        scores.append(
            _geometry_distance(
                ref_label.normalized_coordinates,
                target_coords_norm,
                ref_label.boundary_scores,
                target_boundary_scores,
            )
        )
    return _weights_from_geometry_distances(names, scores, eta)


def _weights_from_geometry_distances(names, scores, eta) -> Dict[str, float]:
    scores_arr = np.asarray(scores, dtype=float)
    shifted = scores_arr - np.min(scores_arr)
    weights = np.exp(-float(eta) * shifted)
    weights = weights / max(float(weights.sum()), 1e-8)
    return {name: float(weight) for name, weight in zip(names, weights)}


def _fixed_reference_weights_for_label(
    references: Sequence[ReferenceSliceData],
    reference_weights: Mapping[str, float],
) -> Dict[str, float]:
    weights = {
        ref.reference_name: float(reference_weights.get(ref.reference_name, 0.0))
        for ref in references
    }
    total = float(sum(max(weight, 0.0) for weight in weights.values()))
    if total <= 0.0:
        raise ValueError("reference_weights must assign positive mass to at least one eligible reference.")
    return {name: max(weight, 0.0) / total for name, weight in weights.items()}


def _reference_counts_for_label(
    references: Sequence[ReferenceSliceData],
    label: str,
    label_key: str,
) -> np.ndarray:
    """Collect aligned reference counts used only for decoder boundaries."""

    parts = []
    for ref in references:
        labels = ref.adata.obs[label_key].astype(str).to_numpy()
        mask = labels == str(label)
        if np.any(mask):
            parts.append(_counts_matrix(ref.adata)[mask, :])
    if not parts:
        return np.zeros((0, len(references[0].adata.var_names)), dtype=np.float32)
    return np.vstack(parts).astype(np.float32, copy=False)


def _decode_label_aware_rank_counts(
    quantiles: np.ndarray,
    label: str,
    label_key: str,
    gene_names: Sequence[str],
    eligible_refs: Sequence[ReferenceSliceData],
    reference_weights: Mapping[str, float],
) -> np.ndarray:
    """Decode counts via rank-preserving assignment against reference data."""
    quantiles = np.asarray(quantiles, dtype=np.float64)
    n_spots, n_genes = quantiles.shape
    counts = np.zeros((n_spots, n_genes), dtype=np.int32)
    if n_spots == 0:
        return counts
    q_positions = (np.arange(n_spots, dtype=np.float64) + 0.5) / float(n_spots)

    for gene_idx, gene_name in enumerate(gene_names):
        source_values = []
        source_weights = []
        for ref in eligible_refs:
            adata = ref.adata
            labels = adata.obs[label_key].astype(str).to_numpy()
            mask = labels == str(label)
            if not np.any(mask):
                continue
            try:
                source_gene_idx = int(adata.var_names.get_loc(str(gene_name)))
            except KeyError:
                continue
            matrix = _counts_matrix(adata)
            values = matrix[mask, source_gene_idx].reshape(-1)
            if values.size == 0:
                continue
            weight = float(reference_weights.get(ref.reference_name, 0.0)) / float(values.size)
            source_values.append(values)
            source_weights.append(np.full(values.size, weight, dtype=np.float64))

        if not source_values:
            continue

        values = np.concatenate(source_values)
        weights = np.concatenate(source_weights)
        order = np.argsort(values, kind="mergesort")
        values = values[order]
        weights = weights[order]
        weight_total = float(weights.sum())
        if weight_total <= 0.0:
            weights = np.full(values.size, 1.0 / float(values.size), dtype=np.float64)
        else:
            weights = weights / weight_total
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0

        target_order = np.argsort(quantiles[:, gene_idx], kind="mergesort")
        q = np.empty(n_spots, dtype=np.float64)
        q[target_order] = q_positions
        q = np.clip(q, 1e-12, 1.0)
        selected = np.searchsorted(cdf, q, side="left")
        selected = np.clip(selected, 0, values.size - 1)
        counts[:, gene_idx] = np.rint(values[selected]).astype(np.int32)

    return counts


def _resolve_parameter_cloud(
    parameter_cloud: Optional[Union[pd.DataFrame, Mapping[str, Any]]],
    label: Optional[str],
    gene_names: Sequence[str],
    eligible_refs: Sequence[ReferenceSliceData],
    reference_weights: Mapping[str, float],
    target_parameter_mode: str = "reference_weighted_log",
) -> pd.DataFrame:
    if parameter_cloud is None:
        if str(target_parameter_mode) == "user_supplied":
            raise ValueError("parameter_cloud is required when target_parameter_mode='user_supplied'.")
        return _weighted_label_stats(label, gene_names, eligible_refs, reference_weights, target_parameter_mode)

    if isinstance(parameter_cloud, pd.DataFrame):
        return _normalize_stats_frame(parameter_cloud, gene_names)

    if _looks_like_stats_mapping(parameter_cloud):
        return _normalize_stats_frame(pd.DataFrame(parameter_cloud, index=gene_names), gene_names)

    if _looks_like_serialized_cloud(parameter_cloud):
        stats_payload = parameter_cloud.get("original_stats") or parameter_cloud.get("full_stats")
        return _normalize_stats_frame(pd.DataFrame.from_dict(stats_payload, orient="index"), gene_names)

    if label is not None and label in parameter_cloud:
        return _resolve_parameter_cloud(
            parameter_cloud=parameter_cloud[label],
            label=None,
            gene_names=gene_names,
            eligible_refs=eligible_refs,
            reference_weights=reference_weights,
            target_parameter_mode=target_parameter_mode,
        )
    if "__default__" in parameter_cloud:
        return _resolve_parameter_cloud(
            parameter_cloud=parameter_cloud["__default__"],
            label=None,
            gene_names=gene_names,
            eligible_refs=eligible_refs,
            reference_weights=reference_weights,
            target_parameter_mode=target_parameter_mode,
        )

    raise TypeError("Unsupported parameter_cloud input for conditional de_novo generation.")


def _weighted_label_stats(
    label: Optional[str],
    gene_names: Sequence[str],
    eligible_refs: Sequence[ReferenceSliceData],
    reference_weights: Mapping[str, float],
    target_parameter_mode: str = "reference_weighted_log",
) -> pd.DataFrame:
    if label is None:
        frames = []
        weights = []
        for ref in eligible_refs:
            for ref_label in ref.labels.values():
                frames.append(ref_label.stats)
                weights.append(1.0 / max(len(eligible_refs) * len(ref.labels), 1))
        if str(target_parameter_mode) == "reference_weighted_log":
            return weighted_stats_log_space(frames, np.asarray(weights, dtype=float), gene_names)
        return _weighted_stats(frames, np.asarray(weights, dtype=float), gene_names)

    frames = [ref.labels[label].stats for ref in eligible_refs]
    weights = np.asarray([reference_weights[ref.reference_name] for ref in eligible_refs], dtype=float)
    if str(target_parameter_mode) == "nearest_reference":
        return _normalize_stats_frame(frames[int(np.argmax(weights))], gene_names)
    if str(target_parameter_mode) == "reference_weighted_log":
        return weighted_stats_log_space(frames, weights, gene_names)
    return _weighted_stats(frames, weights, gene_names)


def _weighted_stats(
    frames: Sequence[pd.DataFrame],
    weights: np.ndarray,
    gene_names: Sequence[str],
) -> pd.DataFrame:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / max(float(weights.sum()), 1e-8)
    out = pd.DataFrame(index=list(gene_names))
    out["mean"] = 0.0
    out["variance"] = 0.0
    out["zero_prop"] = 0.0
    for weight, frame in zip(weights, frames):
        aligned = _normalize_stats_frame(frame, gene_names)
        out["mean"] += float(weight) * aligned["mean"].to_numpy(dtype=float)
        out["variance"] += float(weight) * aligned["variance"].to_numpy(dtype=float)
        out["zero_prop"] += float(weight) * aligned["zero_prop"].to_numpy(dtype=float)
    out["mean"] = np.clip(out["mean"], 1e-8, None)
    out["variance"] = np.clip(out["variance"], 1e-8, None)
    out["zero_prop"] = np.clip(out["zero_prop"], 0.0, 0.99)
    return out


def _aggregate_label_parameter_clouds(
    label_clouds: Mapping[str, pd.DataFrame],
    target_labels: np.ndarray,
    gene_names: Sequence[str],
) -> pd.DataFrame:
    """Summarize resolved label clouds over the target label composition."""
    labels = np.asarray(target_labels).astype(str)
    ordered_labels = list(label_clouds)
    weights = np.asarray(
        [np.count_nonzero(labels == label) for label in ordered_labels],
        dtype=np.float64,
    )
    weights /= weights.sum()

    frames = [
        _normalize_stats_frame(label_clouds[label], gene_names)
        for label in ordered_labels
    ]
    means = np.stack(
        [frame["mean"].to_numpy(dtype=np.float64) for frame in frames]
    )
    variances = np.stack(
        [frame["variance"].to_numpy(dtype=np.float64) for frame in frames]
    )
    zero_props = np.stack(
        [frame["zero_prop"].to_numpy(dtype=np.float64) for frame in frames]
    )

    mean = np.sum(weights[:, None] * means, axis=0)
    second_moment = np.sum(
        weights[:, None] * (variances + np.square(means)),
        axis=0,
    )
    return pd.DataFrame(
        {
            "mean": np.clip(mean, 1e-8, None),
            "variance": np.clip(second_moment - np.square(mean), 1e-8, None),
            "zero_prop": np.clip(
                np.sum(weights[:, None] * zero_props, axis=0),
                0.0,
                0.99,
            ),
        },
        index=list(gene_names),
    )


def _align_stats_frame(frame: pd.DataFrame, gene_names: Sequence[str]) -> pd.DataFrame:
    required = {"mean", "variance", "zero_prop"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"parameter cloud is missing required columns: {sorted(missing)}")
    aligned = frame.copy()
    aligned.index = aligned.index.astype(str)
    if not aligned.index.is_unique:
        duplicates = aligned.index[aligned.index.duplicated()].unique().tolist()
        raise ValueError(f'parameter cloud has duplicate genes: {duplicates}')
    missing_genes = [gene for gene in gene_names if gene not in aligned.index]
    if missing_genes:
        raise ValueError(f"parameter cloud is missing genes: {missing_genes}")
    return aligned.loc[list(gene_names), ["mean", "variance", "zero_prop"]].copy()


def _normalize_stats_frame(frame: pd.DataFrame, gene_names: Sequence[str]) -> pd.DataFrame:
    aligned = _align_stats_frame(frame, gene_names)
    aligned["mean"] = np.clip(aligned["mean"].astype(float), 1e-8, None)
    aligned["variance"] = np.clip(aligned["variance"].astype(float), 1e-8, None)
    aligned["zero_prop"] = np.clip(aligned["zero_prop"].astype(float), 0.0, 0.99)
    return aligned


def _looks_like_serialized_cloud(payload: Mapping[str, Any]) -> bool:
    return "original_stats" in payload or "full_stats" in payload


def _looks_like_stats_mapping(payload: Mapping[str, Any]) -> bool:
    return {"mean", "variance", "zero_prop"} <= set(payload.keys())


def _stats_frame_to_model_params(stats: pd.DataFrame) -> dict:
    model_selected = []
    marginal_param1 = []
    for _, row in stats.iterrows():
        mean = max(float(row["mean"]), 1e-8)
        variance = max(float(row["variance"]), 1e-8)
        pi0 = float(np.clip(row["zero_prop"], 0.0, 0.99))
        active_mean = max(mean / max(1.0 - pi0, 1e-8), 1e-8)
        if variance > active_mean + 1e-8:
            r = max(active_mean * active_mean / max(variance - active_mean, 1e-8), 1e-6)
            model_selected.append("ZINB" if pi0 > 1e-8 else "NB")
            marginal_param1.append([pi0, r, active_mean])
        else:
            model_selected.append("ZIP" if pi0 > 1e-8 else "Poisson")
            marginal_param1.append([pi0, 1.0, active_mean])
    return {
        "genes": list(stats.index.astype(str)),
        "model_selected": model_selected,
        "marginal_param1": marginal_param1,
        "target_stats": stats[["mean", "variance", "zero_prop"]].reset_index(drop=True).copy(),
        "parameter_diagnostics": {
            "requested_config": {
                "apply_to_variance": True,
                "apply_to_zero_prop": True,
                "mean_variance_coupling": None,
            }
        },
    }


# ---------------------------------------------------------------------------
# Auto-estimation of assignment_randomness via mini-sweep
# ---------------------------------------------------------------------------


def estimate_assignment_randomness(
    reference: "ad.AnnData | SimulationReference",
    label_key: str | None = None,
    *,
    spatial_key: str = "spatial",
    n_genes: int = 20,
    ar_step: float = 0.1,
    max_ar: float = 0.5,
    n_neighbors: int = 6,
    random_seed: int | None = None,
    transport: TransportConfig | None = None,
) -> float:
    """Estimate assignment_randomness via a quick mini-sweep on the reference.

    Spatially splits the reference in half, runs the full de-novo conditional
    pipeline on a small gene panel at each AR candidate, and returns the value
    that best preserves spatial autocorrelation (Moran's I correlation between
    generated and held-out halves).

    This mirrors the logic of the full cross-slice benchmark but operates on a
    single slice split, with a small gene set, so it completes in seconds.

    Parameters
    ----------
    reference:
        Reference :class:`~anndata.AnnData` or a fitted
        :class:`SimulationReference`.  If an AnnData, *label_key* must
        be provided.
    label_key:
        Column in ``.obs`` with domain/region labels.  Required when
        *reference* is an AnnData.
    spatial_key:
        Key in ``.obsm`` for spatial coordinates (default ``"spatial"``).
    n_genes:
        Number of high-variance genes to use in the mini-sweep.  More
        genes improve stability but increase runtime linearly.
    ar_step:
        Step size between AR candidates.  Tested values are
        ``[0.0, ar_step, 2*ar_step, ..., max_ar]``.  Use a finer step
        (e.g. 0.05) for higher precision at the cost of more OT runs.
    max_ar:
        Maximum AR value to test.  Values above ~0.5 rarely help.
    n_neighbors:
        Number of spatial neighbors for Moran's-I computation.
    random_seed:
        Seed for reproducible results.
    transport:
        Optional transport configuration used for every mini-sweep candidate.

    Returns
    -------
    float
        Estimated assignment_randomness in [0.0, *max_ar*].

    Notes
    -----
    If the reference has fewer than 50 spots, only a single domain label,
    or the spatial split fails to produce two viable halves, the function
    returns 0.0 (pure OT, no randomness).
    """
    # --- resolve input ---
    if isinstance(reference, SimulationReference):
        if label_key is None:
            label_key = reference.label_key
        if not reference.references:
            return 0.0
        # Use the largest reference slice for estimation (most label coverage)
        adata = max(reference.references, key=lambda r: r.adata.n_obs).adata
    else:
        adata = reference
        if label_key is None:
            raise ValueError("label_key is required when reference is an AnnData.")
    if label_key not in adata.obs:
        raise KeyError(f"label_key {label_key!r} not found in adata.obs.")

    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    else:
        coords = _coordinates(adata)
    coords_2d = coords[:, :2]
    n_spots = coords_2d.shape[0]
    if n_spots < 50:
        return 0.0

    labels = adata.obs[label_key].astype(str).to_numpy()
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # --- subsample if too large for quick OT ---
    max_spots = 10000
    if n_spots > max_spots:
        rng = np.random.default_rng(random_seed or 0)
        keep = rng.choice(n_spots, size=max_spots, replace=False)
        adata = adata[keep, :].copy()
        coords_2d = coords_2d[keep, :]
        labels = labels[keep]
        n_spots = max_spots

    # --- select high-variance genes ---
    matrix = _to_dense_matrix(adata.X)
    variances = np.var(matrix, axis=0, ddof=0)
    top_idx = np.argsort(-variances)[: min(n_genes, adata.n_vars)]
    if len(top_idx) < 5:
        return 0.0
    gene_names = [str(adata.var_names[i]) for i in top_idx]
    adata_sub = adata[:, gene_names].copy()

    # --- spatial split into reference / target halves ---
    axis = int(np.argmax(np.ptp(coords_2d, axis=0)))
    median = float(np.median(coords_2d[:, axis]))
    ref_mask = coords_2d[:, axis] <= median
    tgt_mask = ~ref_mask
    if ref_mask.sum() < 20 or tgt_mask.sum() < 20:
        return 0.0

    ref_adata = adata_sub[ref_mask, :].copy()
    tgt_adata = adata_sub[tgt_mask, :].copy()

    # ensure both halves share the same label set
    ref_label_set = set(np.unique(ref_adata.obs[label_key].astype(str)))
    tgt_label_set = set(np.unique(tgt_adata.obs[label_key].astype(str)))
    common_labels = ref_label_set & tgt_label_set
    if len(common_labels) < 2:
        return 0.0

    if common_labels != ref_label_set or common_labels != tgt_label_set:
        r_mask = ref_adata.obs[label_key].astype(str).isin(common_labels).to_numpy()
        t_mask = tgt_adata.obs[label_key].astype(str).isin(common_labels).to_numpy()
        ref_adata = ref_adata[r_mask, :].copy()
        tgt_adata = tgt_adata[t_mask, :].copy()
        if ref_adata.n_obs < 20 or tgt_adata.n_obs < 20:
            return 0.0

    # --- fit reference on the left half ---
    fit_cfg = ReferenceFitConfig(min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0)
    model = fit_reference(ref_adata, label_key=label_key, config=fit_cfg)
    if not model.gene_names:
        return 0.0

    # align target to model genes
    model_genes = list(model.gene_names)
    tgt_aligned = tgt_adata[:, model_genes].copy()

    # --- pre-compute target Moran's I (same for all AR values) ---
    tgt_coords = np.asarray(tgt_aligned.obsm.get(spatial_key, _coordinates(tgt_aligned)), dtype=float)[:, :2]
    tgt_n = tgt_coords.shape[0]
    k = min(n_neighbors + 1, tgt_n)
    tgt_nn = NearestNeighbors(n_neighbors=k).fit(tgt_coords)
    tgt_neigh = tgt_nn.kneighbors(tgt_coords, return_distance=False)[:, 1:]
    tgt_matrix = _to_dense_matrix(tgt_aligned.X)

    tgt_morans = _per_gene_moran(tgt_matrix, tgt_neigh)
    tgt_valid = np.isfinite(tgt_morans)
    if tgt_valid.sum() < 5:
        return 0.0
    tgt_morans = tgt_morans[tgt_valid]
    tgt_matrix = tgt_matrix[:, tgt_valid]
    model_genes = [g for i, g in enumerate(model_genes) if tgt_valid[i]]

    # --- mini-sweep over AR values ---
    blueprint = SliceBlueprint(
        coordinates=tgt_coords,
        domain_map=tgt_aligned.obs[label_key].astype(str).to_numpy(),
    )
    ar_candidates = np.arange(0.0, float(max_ar) + 1e-9, float(ar_step))
    if len(ar_candidates) < 2:
        ar_candidates = np.array([0.0, float(max_ar)])

    best_ar = 0.0
    best_corr = -np.inf
    correlations = []

    base_gen_cfg = _simulation_config(transport)
    for ar in ar_candidates:
        gen_cfg = replace(base_gen_cfg, assignment_randomness=float(ar), verbose=False)
        generated = simulate_from_reference(
            model, blueprint, config=gen_cfg, random_seed=random_seed or 0,
        )
        gen_matrix = _to_dense_matrix(generated.X)
        gen_morans = _per_gene_moran(gen_matrix, tgt_neigh)
        gen_valid = np.isfinite(gen_morans)
        if gen_valid.sum() < 5:
            continue
        g = gen_morans[gen_valid]
        t = tgt_morans[gen_valid]
        if np.std(g) <= 1e-12 or np.std(t) <= 1e-12:
            continue
        corr = float(np.corrcoef(g, t)[0, 1])
        if not np.isfinite(corr):
            continue
        correlations.append((float(ar), corr))
        if corr > best_corr:
            best_corr = corr
            best_ar = float(ar)

    if not correlations:
        return 0.0
    return best_ar


def _per_gene_moran(
    matrix: np.ndarray,
    neigh_idx: np.ndarray,
) -> np.ndarray:
    """Compute per-gene Moran's I on a dense (spots x genes) matrix."""
    n_spots, n_genes = matrix.shape
    out = np.full(n_genes, np.nan, dtype=np.float64)
    for j in range(n_genes):
        x = matrix[:, j].astype(np.float64)
        z = x - np.mean(x)
        denom = float(np.dot(z, z))
        if denom <= 1e-12:
            continue
        num = 0.0
        w_sum = 0.0
        for i, neigh in enumerate(neigh_idx):
            num += float(np.sum(z[i] * z[neigh]))
            w_sum += len(neigh)
        if w_sum > 0:
            out[j] = (n_spots / w_sum) * (num / denom)
    return out
