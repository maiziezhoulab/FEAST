"""Rank-based count decoding for FEAST simulation.

Single pipeline: sample per gene → sort → assign by argsort(quantiles).
Streaming decode avoids materialising the full raw_counts intermediate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

SPATIAL_INTENSITY_FLOOR = 0.02


def _dense_array(X, dtype=None) -> Optional[np.ndarray]:
    if X is None:
        return None
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(X)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _sparse_per_gene_max(X, n_genes: int) -> np.ndarray:
    """Compute per-gene maximum from a potentially sparse matrix."""
    from scipy.sparse import issparse
    if issparse(X):
        return np.asarray(X.max(axis=0).toarray()).ravel()
    return np.max(np.asarray(X, dtype=np.float64), axis=0).reshape(-1)


def _model_type_and_params(model_params: dict, gene_idx: int):
    model_selected = model_params.get("model_selected", [])
    marginal_param1 = model_params.get("marginal_param1", [])
    model_type = model_selected[gene_idx] if gene_idx < len(model_selected) else "Poisson"
    params = marginal_param1[gene_idx] if gene_idx < len(marginal_param1) else [0.0, 1.0, 1.0]
    if not isinstance(params, (list, tuple, np.ndarray)):
        params = [0.0, 1.0, 1.0]
    pi0 = float(params[0]) if len(params) > 0 and np.isfinite(params[0]) else 0.0
    r = float(params[1]) if len(params) > 1 and np.isfinite(params[1]) else 1.0
    mu = float(params[2]) if len(params) > 2 and np.isfinite(params[2]) else 1.0
    return str(model_type), float(np.clip(pi0, 0.0, 1.0)), max(r, 1e-8), max(mu, 1e-8)


def _boundary_per_gene(
    reference_X,
    n_genes: int,
    model_params: dict,
    boundary_multiplier: float,
) -> np.ndarray:
    boundary = np.full(n_genes, np.inf, dtype=np.float64)
    if reference_X is not None and (hasattr(reference_X, 'shape') and reference_X.shape[0] > 0):
        boundary = _sparse_per_gene_max(reference_X, n_genes) * float(boundary_multiplier)
        if boundary.shape[0] != n_genes:
            boundary = np.resize(boundary, n_genes).astype(np.float64)

    for gene_idx in range(n_genes):
        _, _, _, mu = _model_type_and_params(model_params, gene_idx)
        if boundary[gene_idx] < 1.0 and mu > 1e-6:
            boundary[gene_idx] = np.inf
    return boundary


def _sample_gene_counts(model_type: str, pi0: float, r: float, mu: float, n_spots: int, rng) -> np.ndarray:
    if model_type == "Poisson":
        return rng.poisson(mu, size=n_spots)
    if model_type == "NB":
        p = r / (r + mu)
        return rng.negative_binomial(r, np.clip(p, 1e-8, 1.0 - 1e-8), size=n_spots)
    if model_type == "ZIP":
        counts = rng.poisson(mu, size=n_spots)
        counts[rng.random(n_spots) < pi0] = 0
        return counts
    if model_type == "ZINB":
        p = r / (r + mu)
        counts = rng.negative_binomial(r, np.clip(p, 1e-8, 1.0 - 1e-8), size=n_spots)
        counts[rng.random(n_spots) < pi0] = 0
        return counts
    return rng.poisson(mu, size=n_spots)


def _nb_zero_probability(mu, r):
    mu = np.asarray(mu, dtype=np.float64)
    if not np.isfinite(r) or r > 1e12:
        return np.exp(-mu)
    safe_r = max(float(r), 1e-8)
    return (safe_r / (safe_r + np.clip(mu, 0.0, None))) ** safe_r


def _theoretical_gene_stats(model_type: str, pi0: float, r: float, mu: float):
    if model_type == "Poisson":
        return mu, mu, float(np.exp(-mu))
    if model_type == "NB":
        var = mu + (0.0 if not np.isfinite(r) else mu ** 2 / max(r, 1e-8))
        return mu, var, float(_nb_zero_probability(mu, r))
    if model_type == "ZIP":
        mean = (1.0 - pi0) * mu
        var = (1.0 - pi0) * mu * (1.0 + pi0 * mu)
        zero = pi0 + (1.0 - pi0) * np.exp(-mu)
        return mean, var, float(zero)
    if model_type == "ZINB":
        safe_r = max(r, 1e-8)
        mean = (1.0 - pi0) * mu
        var = (1.0 - pi0) * (mu + mu ** 2 / safe_r + pi0 * mu ** 2)
        zero = pi0 + (1.0 - pi0) * _nb_zero_probability(mu, safe_r)
        return mean, var, float(zero)
    return mu, mu, float(np.exp(-mu))


def _target_gene_stats(model_params: dict, gene_idx: int, model_type: str, pi0: float, r: float, mu: float):
    target_stats = model_params.get("target_stats")
    if target_stats is not None and hasattr(target_stats, "iloc"):
        row = target_stats.iloc[gene_idx]
        if {"mean", "variance", "zero_prop"}.issubset(set(row.index)):
            return (
                float(row["mean"]),
                float(row["variance"]),
                float(row["zero_prop"]),
            )
    return _theoretical_gene_stats(model_type, pi0, r, mu)


def _gene_values(X, gene_idx: int, n_spots: int) -> np.ndarray:
    if X is None:
        return np.ones(n_spots, dtype=np.float64)
    col = X[:, gene_idx] if getattr(X, "ndim", 2) == 2 else X
    if hasattr(col, "toarray"):
        col = col.toarray()
    return np.asarray(col, dtype=np.float64).reshape(-1)


def _calibrated_intensity(
    reference_values: np.ndarray,
    target_variance: float,
    pi0: float,
    mu: float,
    *,
    preserve_variance_floor: bool = False,
) -> tuple[np.ndarray, float]:
    values = np.nan_to_num(np.asarray(reference_values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    if values.size == 0 or float(values.mean()) <= 1e-12 or float(values.max()) <= 0.0:
        return np.ones(values.size, dtype=np.float64), float(target_variance)

    raw = values / float(values.mean())
    raw = np.maximum(raw, SPATIAL_INTENSITY_FLOOR)
    raw = raw / float(raw.mean())
    raw_var = float(np.var(raw))
    if raw_var <= 1e-12:
        return np.ones(values.size, dtype=np.float64), float(target_variance)

    one_minus_pi = max(1.0 - float(pi0), 1e-8)
    mu = max(float(mu), 1e-8)
    # Minimum mixture variance at infinite NB dispersion.  Shrink spatial
    # contrast only when the reference field alone would exceed target variance.
    base_var = one_minus_pi * mu + float(pi0) * one_minus_pi * mu ** 2
    spatial_coeff = one_minus_pi * mu ** 2
    if preserve_variance_floor:
        target_variance = max(float(target_variance), base_var + spatial_coeff * raw_var)
    max_spatial_var = max((float(target_variance) - base_var) / max(spatial_coeff, 1e-12), 0.0)
    gamma = min(1.0, float(np.sqrt(max_spatial_var / raw_var))) if max_spatial_var > 0.0 else 0.0
    intensity = 1.0 + gamma * (raw - 1.0)
    intensity = np.clip(intensity, 0.0, None)
    mean_intensity = float(intensity.mean())
    if mean_intensity <= 1e-12:
        return np.ones(values.size, dtype=np.float64), float(target_variance)
    return intensity / mean_intensity, float(target_variance)


def _effective_dispersion(model_type: str, pi0: float, mu: float, target_variance: float, intensity: np.ndarray, fallback_r: float) -> float:
    if model_type not in {"NB", "ZINB"}:
        return np.inf
    one_minus_pi = max(1.0 - float(pi0), 1e-8)
    mu = max(float(mu), 1e-8)
    spatial_var = float(np.var(intensity))
    fixed_var = (
        one_minus_pi * mu
        + float(pi0) * one_minus_pi * mu ** 2 * (1.0 + spatial_var)
        + one_minus_pi ** 2 * mu ** 2 * spatial_var
    )
    numerator = one_minus_pi * mu ** 2 * (1.0 + spatial_var)
    denom = float(target_variance) - fixed_var
    if denom <= 1e-12:
        return np.inf
    r = numerator / denom
    if not np.isfinite(r) or r <= 0.0:
        return fallback_r
    return float(np.clip(r, 1e-8, 1e12))


def _zero_inflation_for_target(model_type: str, pi0: float, target_zero: float, base_zero: np.ndarray) -> float:
    if model_type not in {"ZIP", "ZINB"}:
        return pi0
    avg_base_zero = float(np.mean(base_zero))
    if avg_base_zero >= 1.0 - 1e-12:
        return 0.99
    adjusted = (float(target_zero) - avg_base_zero) / max(1.0 - avg_base_zero, 1e-12)
    return float(np.clip(adjusted, 0.0, 0.99))


def _base_zero_for_model(model_type: str, mu_values: np.ndarray, r: float) -> np.ndarray:
    if model_type in {"Poisson", "ZIP"} or not np.isfinite(r) or r > 1e12:
        return np.exp(-np.clip(mu_values, 0.0, None))
    return _nb_zero_probability(mu_values, r)


def _component_mu_for_overall_mean(model_type: str, target_mean: float, pi0: float) -> float:
    if model_type in {"ZIP", "ZINB"}:
        return max(float(target_mean) / max(1.0 - float(pi0), 1e-8), 1e-8)
    return max(float(target_mean), 1e-8)


def _predicted_zero_rate(
    intensity: np.ndarray,
    model_type: str,
    target_mean: float,
    target_variance: float,
    target_zero: float,
    pi0: float,
    fallback_r: float,
) -> float:
    component_mu = _component_mu_for_overall_mean(model_type, target_mean, pi0)
    r = _effective_dispersion(model_type, pi0, component_mu, target_variance, intensity, fallback_r)
    base_zero = _base_zero_for_model(model_type, component_mu * intensity, r)
    effective_pi0 = _zero_inflation_for_target(model_type, pi0, target_zero, base_zero)
    component_mu = _component_mu_for_overall_mean(model_type, target_mean, effective_pi0)
    r = _effective_dispersion(model_type, effective_pi0, component_mu, target_variance, intensity, fallback_r)
    base_zero = _base_zero_for_model(model_type, component_mu * intensity, r)
    if model_type in {"ZIP", "ZINB"}:
        effective_pi0 = _zero_inflation_for_target(model_type, effective_pi0, target_zero, base_zero)
        return float(effective_pi0 + (1.0 - effective_pi0) * np.mean(base_zero))
    return float(np.mean(base_zero))


def _floor_intensity_for_zero_target(
    intensity: np.ndarray,
    model_type: str,
    target_mean: float,
    target_variance: float,
    target_zero: float,
    pi0: float,
    fallback_r: float,
) -> np.ndarray:
    if not np.isfinite(target_zero) or target_zero >= 0.999:
        return intensity
    current_zero = _predicted_zero_rate(
        intensity, model_type, target_mean, target_variance, target_zero, pi0, fallback_r
    )
    if current_zero <= float(target_zero) + 0.005:
        return intensity

    lo = float(np.min(intensity))
    hi = 1.0
    best = intensity
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        candidate = np.maximum(intensity, mid)
        candidate = candidate / max(float(candidate.mean()), 1e-12)
        zero_rate = _predicted_zero_rate(
            candidate, model_type, target_mean, target_variance, target_zero, pi0, fallback_r
        )
        if zero_rate > float(target_zero) + 0.005:
            lo = mid
        else:
            best = candidate
            hi = mid
    return best


def _sample_gene_counts_with_mean(model_type: str, pi0: float, r: float, mu_values: np.ndarray, rng) -> np.ndarray:
    mu_values = np.clip(np.asarray(mu_values, dtype=np.float64), 0.0, 1e9)
    if model_type in {"Poisson", "ZIP"} or not np.isfinite(r) or r > 1e12:
        counts = rng.poisson(mu_values)
    else:
        safe_r = max(float(r), 1e-8)
        p = safe_r / (safe_r + mu_values)
        counts = rng.negative_binomial(safe_r, np.clip(p, 1e-8, 1.0 - 1e-8))
    if model_type in {"ZIP", "ZINB"} and pi0 > 0.0:
        counts[rng.random(mu_values.shape[0]) < pi0] = 0
    return counts


def generate_count_bag_from_model_params(
    model_params: dict,
    n_spots: int,
    *,
    boundary_multiplier: float = 1.1,
    reference_X=None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Sample an unordered per-gene count bag from fitted FEAST model params.

    For large datasets, prefer decode_counts_by_rank() directly — it streams
    per gene and avoids allocating this full intermediate matrix.
    """
    if "model_selected" not in model_params or "marginal_param1" not in model_params:
        raise ValueError("model_params must contain 'model_selected' and 'marginal_param1'.")

    n_genes = len(model_params["model_selected"])
    rng = np.random if random_seed is None else np.random.default_rng(int(random_seed))
    boundary = _boundary_per_gene(reference_X, n_genes, model_params, boundary_multiplier)
    counts = np.zeros((int(n_spots), n_genes), dtype=np.float32)

    for gene_idx in range(n_genes):
        model_type, pi0, r, mu = _model_type_and_params(model_params, gene_idx)
        gene_counts = _sample_gene_counts(model_type, pi0, r, mu, int(n_spots), rng).astype(np.float32)
        gene_boundary = boundary[gene_idx]
        if np.isfinite(gene_boundary):
            gene_counts = np.minimum(gene_counts, gene_boundary)
        counts[:, gene_idx] = gene_counts

    return counts


def _match_zero_count_and_mean(gene_bag: np.ndarray, target_zero: float, target_mean: float) -> np.ndarray:
    counts = np.asarray(gene_bag, dtype=np.float64).copy()
    n_spots = counts.shape[0]
    if n_spots == 0 or not np.isfinite(target_zero) or not np.isfinite(target_mean):
        return counts

    target_zero_count = int(np.clip(np.rint(float(target_zero) * n_spots), 0, n_spots))
    counts.sort()
    if target_zero_count >= n_spots:
        counts[:] = 0.0
        return counts

    counts[:target_zero_count] = 0.0
    counts[target_zero_count:] = np.maximum(counts[target_zero_count:], 1.0)
    positive_sum = float(counts[target_zero_count:].sum())
    target_sum = max(float(target_mean) * n_spots, 0.0)
    n_positive = n_spots - target_zero_count
    if positive_sum > 1e-12 and target_sum > 0.0 and n_positive > 0:
        target_total = max(int(np.rint(target_sum)), n_positive)
        remaining = target_total - n_positive
        weights = counts[target_zero_count:] / positive_sum
        allocation_float = weights * remaining
        allocation = np.floor(allocation_float).astype(np.int64)
        remainder = int(remaining - int(allocation.sum()))
        if remainder > 0:
            fractions = allocation_float - allocation
            add_idx = np.argsort(fractions)[-remainder:]
            allocation[add_idx] += 1
        counts[target_zero_count:] = 1.0 + allocation.astype(np.float64)
    counts.sort()
    return counts


def _reference_preservation_policy(model_params: dict) -> tuple[bool, bool]:
    diagnostics = model_params.get("parameter_diagnostics", {})
    requested_config = diagnostics.get("requested_config")
    if requested_config is None:
        return True, True
    if not isinstance(requested_config, dict):
        return False, False

    variance_changed = bool(requested_config.get("apply_to_variance", False))
    variance_changed = variance_changed or requested_config.get("mean_variance_coupling") is not None
    zero_changed = bool(requested_config.get("apply_to_zero_prop", False))
    return not variance_changed, not zero_changed


def decode_counts_by_spatial_intensity(
    intensity: np.ndarray,
    model_params: dict,
    *,
    boundary_multiplier: float = 1.1,
    reference_X=None,
    random_seed: Optional[int] = None,
    show_progress: bool = False,
    diagnostics: Optional[dict] = None,
) -> np.ndarray:
    """Decode counts from spot-specific reference intensity fields.

    For each gene, FEAST builds a heterogeneous count bag from spot-specific
    means, then rank-couples the sorted bag to the sorted reference intensity.
    This keeps synthetic counts generative while preserving reference-conditioned
    magnitude contrast better than an iid global count bag.
    """
    if "model_selected" not in model_params or "marginal_param1" not in model_params:
        raise ValueError("model_params must contain 'model_selected' and 'marginal_param1'.")

    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.ndim != 2:
        raise ValueError("intensity must be a 2D array.")

    n_spots, n_genes = intensity.shape
    if n_spots == 0:
        return np.zeros((0, n_genes), dtype=np.int32)

    rng = np.random if random_seed is None else np.random.default_rng(int(random_seed))
    boundary = _boundary_per_gene(reference_X, n_genes, model_params, boundary_multiplier)
    final_counts = np.zeros((n_spots, n_genes), dtype=np.float32)
    preserve_variance_floor, preserve_reference_zero = _reference_preservation_policy(model_params)

    iterator = range(n_genes)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    if diagnostics is not None:
        diagnostics.update(intensity_variance_ratio=[], clipped_positions=[], generated_mean=[], generated_variance=[], generated_zero_prop=[])
    for gene_idx in iterator:
        model_type, pi0, r, mu = _model_type_and_params(model_params, gene_idx)
        target_mean, target_var, target_zero = _target_gene_stats(
            model_params, gene_idx, model_type, pi0, r, mu
        )
        if not np.isfinite(target_var) or target_var <= 0.0:
            _, target_var, target_zero = _theoretical_gene_stats(model_type, pi0, r, mu)

        ref_values = _gene_values(intensity, gene_idx, n_spots)
        if preserve_reference_zero:
            ref_zero_source = reference_X if reference_X is not None else intensity
            zero_ref = _gene_values(ref_zero_source, gene_idx, n_spots)
            target_zero = float(np.mean(zero_ref <= 0.0))
        component_mu = _component_mu_for_overall_mean(model_type, target_mean, pi0)

        spatial_intensity, effective_target_var = _calibrated_intensity(
            ref_values,
            target_var,
            pi0,
            component_mu,
            preserve_variance_floor=preserve_variance_floor,
        )
        spatial_intensity = _floor_intensity_for_zero_target(
            spatial_intensity,
            model_type,
            target_mean,
            effective_target_var,
            target_zero,
            pi0,
            r,
        )

        initial_r = _effective_dispersion(
            model_type, pi0, component_mu, effective_target_var, spatial_intensity, r
        )
        initial_mu_values = component_mu * spatial_intensity
        if model_type == "ZIP":
            base_zero = np.exp(-initial_mu_values)
        elif model_type == "ZINB":
            base_zero = _nb_zero_probability(initial_mu_values, initial_r)
        else:
            base_zero = np.zeros(n_spots, dtype=np.float64)
        effective_pi0 = _zero_inflation_for_target(model_type, pi0, target_zero, base_zero)
        component_mu = _component_mu_for_overall_mean(model_type, target_mean, effective_pi0)
        effective_r = _effective_dispersion(
            model_type, effective_pi0, component_mu, effective_target_var, spatial_intensity, r
        )
        mu_values = component_mu * spatial_intensity

        gene_bag = _sample_gene_counts_with_mean(
            model_type, effective_pi0, effective_r, mu_values, rng
        ).astype(np.float32)
        gene_boundary = boundary[gene_idx]
        clipped_positions = int(np.sum(gene_bag > gene_boundary))
        if np.isfinite(gene_boundary):
            gene_bag = np.minimum(gene_bag, gene_boundary)
        gene_bag.sort()
        gene_bag = _match_zero_count_and_mean(gene_bag, target_zero, target_mean)
        if np.isfinite(gene_boundary):
            gene_bag = np.minimum(gene_bag, gene_boundary)
            gene_bag.sort()

        if diagnostics is not None:
            raw = np.maximum(ref_values / max(float(ref_values.mean()), 1e-12), SPATIAL_INTENSITY_FLOOR)
            raw /= max(float(raw.mean()), 1e-12)
            raw_var = float(np.var(raw))
            diagnostics['intensity_variance_ratio'].append(float(np.var(spatial_intensity)) / raw_var if raw_var > 0 else 1.0)
            diagnostics['clipped_positions'].append(clipped_positions)
            realized = np.rint(gene_bag)
            diagnostics['generated_mean'].append(float(realized.mean()))
            diagnostics['generated_variance'].append(float(realized.var()))
            diagnostics['generated_zero_prop'].append(float(np.mean(realized == 0)))
        spot_rank_order = np.argsort(ref_values, kind="mergesort")
        final_counts[spot_rank_order, gene_idx] = gene_bag

    return np.rint(final_counts).astype(np.int32)


def decode_counts_by_rank(
    quantiles: np.ndarray,
    model_params: dict,
    *,
    spot_weights: Optional[np.ndarray] = None,
    boundary_multiplier: float = 1.1,
    reference_X=None,
    random_seed: Optional[int] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Decode counts from rank-ordered quantile positions.

    For each gene, samples a count bag, sorts, and assigns to spots by
    argsort(quantiles[:, gene]) — then discards the bag before moving to
    the next gene.  Avoids the full (n_spots × n_genes) raw_counts
    intermediate that would double peak memory for large datasets.
    """
    if "model_selected" not in model_params or "marginal_param1" not in model_params:
        raise ValueError("model_params must contain 'model_selected' and 'marginal_param1'.")

    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim != 2:
        raise ValueError("quantiles must be a 2D array.")

    n_spots, n_genes = quantiles.shape
    if n_spots == 0:
        return np.zeros((0, n_genes), dtype=np.int32)

    rng = np.random if random_seed is None else np.random.default_rng(int(random_seed))
    boundary = _boundary_per_gene(reference_X, n_genes, model_params, boundary_multiplier)

    if spot_weights is not None:
        weights = np.asarray(spot_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != n_spots:
            raise ValueError(f"spot_weights length {weights.shape[0]} does not match n_spots {n_spots}.")
        weights = np.clip(weights, 1e-8, None)
        weights = weights / np.sum(weights)
    else:
        weights = None

    q_positions = np.linspace(0.0, 1.0, n_spots, dtype=np.float64)
    final_counts = np.zeros((n_spots, n_genes), dtype=np.float32)

    iterator = range(n_genes)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    for gene_idx in iterator:
        model_type, pi0, r, mu = _model_type_and_params(model_params, gene_idx)
        gene_bag = _sample_gene_counts(model_type, pi0, r, mu, n_spots, rng).astype(np.float32)
        gene_boundary = boundary[gene_idx]
        if np.isfinite(gene_boundary):
            gene_bag = np.minimum(gene_bag, gene_boundary)
        gene_bag.sort()

        spot_rank_order = np.argsort(quantiles[:, gene_idx])
        if weights is None:
            final_counts[spot_rank_order, gene_idx] = gene_bag
        else:
            w_ordered = weights[spot_rank_order]
            cum_w = np.cumsum(w_ordered)
            if cum_w[-1] <= 0:
                final_counts[spot_rank_order, gene_idx] = gene_bag
            else:
                q_w = (cum_w - 0.5 * w_ordered) / cum_w[-1]
                final_counts[spot_rank_order, gene_idx] = np.interp(q_w, q_positions, gene_bag)

    return np.rint(final_counts).astype(np.int32)
