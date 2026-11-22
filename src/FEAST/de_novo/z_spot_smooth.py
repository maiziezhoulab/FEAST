"""Spot-level z-trajectory regularization via cross-z local weighted linear regression.

For each spot, fits a local linear model of expression vs. z-coordinate using
bilateral-weighted neighbours from the same and adjacent z-slices.  The
predicted expression at the spot's own z-level becomes the smoothed value,
which is then soft-blended with the original count.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors


_GENE_CHUNK_SIZE = 512



def _dense_from_adata(adata: ad.AnnData, layer: Optional[str] = None) -> np.ndarray:
    """Extract dense count matrix from an AnnData object."""
    src = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    if hasattr(src, "toarray"):
        return np.asarray(src.toarray(), dtype=np.float64)
    if hasattr(src, "todense"):
        return np.asarray(src.todense(), dtype=np.float64)
    return np.asarray(src, dtype=np.float64)


def _robust_std(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Robust standard deviation estimate via IQR / 1.349."""
    q25, q75 = np.percentile(arr, [25, 75], axis=axis)
    iqr = q75 - q25
    return iqr / 1.349


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; returns NaN when denominator is zero or too few points."""
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    xc = x[valid] - x[valid].mean()
    yc = y[valid] - y[valid].mean()
    denom = math.sqrt(float((xc @ xc) * (yc @ yc)))
    if denom <= 0.0:
        return float("nan")
    return float(xc @ yc / denom)


def _resolve_genes(
    z_slices: Dict[float, ad.AnnData],
    sorted_z: List[float],
    genes: Optional[Sequence[str]],
) -> List[str]:
    """Resolve gene list from user input or intersection across slices."""
    if genes is not None:
        return list(genes)
    gene_set = set(str(g) for g in z_slices[sorted_z[0]].var_names)
    for z in sorted_z[1:]:
        gene_set &= set(str(g) for g in z_slices[z].var_names)
    return sorted(gene_set)


def _align_expression(
    adata: ad.AnnData,
    gene_list: List[str],
    layer: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return (n, len(gene_list)) dense array and mapping gene->orig_col.

    Columns not present in *adata* are filled with zeros.
    """
    raw = _dense_from_adata(adata, layer)
    var_names = [str(g) for g in adata.var_names]
    gene_to_col = {g: i for i, g in enumerate(var_names)}
    aligned = np.zeros((raw.shape[0], len(gene_list)), dtype=np.float64)
    for gi, g in enumerate(gene_list):
        if g in gene_to_col:
            aligned[:, gi] = raw[:, gene_to_col[g]]
    return aligned, gene_to_col


def _extract_spatial_z(
    adata: ad.AnnData, fallback_z: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (spatial_xy, per_spot_z_vals)."""
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"obsm['spatial'] must be (n, >=2), got {coords.shape}")
    if "z" in adata.obs:
        spot_z = np.asarray(adata.obs["z"], dtype=np.float64)
    else:
        spot_z = np.full(coords.shape[0], float(fallback_z), dtype=np.float64)
    return coords[:, :2], spot_z


def _estimate_median_nn_dist(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance among *coords* (excluding self)."""
    n = coords.shape[0]
    if n < 2:
        return 1.0
    nn = NearestNeighbors(n_neighbors=min(11, n), metric="euclidean")
    nn.fit(coords)
    dists, _ = nn.kneighbors(coords)
    col = 1 if dists.shape[1] > 1 else 0
    return float(np.median(dists[:, col]))




def _smooth_one_slice(
    *,
    z_i: float,
    slice_idx: int,
    sorted_z: List[float],
    n_i: int,
    log1p_i: np.ndarray,
    raw_i: np.ndarray,
    coords_i_norm: np.ndarray,
    spot_z_i_norm: np.ndarray,
    z_to_log1p: Dict[float, np.ndarray],
    z_to_coords_norm: Dict[float, np.ndarray],
    z_to_spot_z_norm: Dict[float, np.ndarray],
    z_to_nn: Dict[float, NearestNeighbors],
    n_spots: Dict[float, int],
    k_xy: int,
    k_z: int,
    xy_sigma2: float,
    z_sigma2: float,
    blend_lambda: float,
    ridge: float,
    n_genes: int,
) -> np.ndarray:
    """Smooth spots in a single z-slice.

    Returns blended (n_i, n_genes) count matrix (float32).
    """

    # --- Determine adjacent slices ---
    adjacent_z: List[float] = []
    if slice_idx > 0:
        adjacent_z.append(sorted_z[slice_idx - 1])
    if slice_idx < len(sorted_z) - 1:
        adjacent_z.append(sorted_z[slice_idx + 1])

    n_adj = len(adjacent_z)
    k_z_each = max(1, (k_z + n_adj - 1) // n_adj) if n_adj > 0 else 0

    # --- Same-slice neighbours ---
    nn_self = z_to_nn[z_i]
    if k_xy > 0 and n_i > 1:
        s_dists, s_idxs = nn_self.kneighbors(
            coords_i_norm, n_neighbors=min(k_xy + 1, n_i)
        )
        same_dists = s_dists[:, 1:]   # skip self
        same_idxs = s_idxs[:, 1:]
    else:
        same_dists = np.zeros((n_i, 0), dtype=np.float64)
        same_idxs = np.zeros((n_i, 0), dtype=np.int64)

    # --- Cross-z neighbours ---
    cross_parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for z_adj in adjacent_z:
        n_adj_spots = n_spots[z_adj]
        k_eff = min(k_z_each, n_adj_spots)
        if k_eff < 1:
            continue
        nn_adj = z_to_nn[z_adj]
        c_dists, c_idxs = nn_adj.kneighbors(coords_i_norm, n_neighbors=k_eff)
        adj_z_vals = z_to_spot_z_norm[z_adj][c_idxs]
        z_deltas = adj_z_vals - spot_z_i_norm[:, None]
        cross_parts.append((c_dists, c_idxs, z_deltas, z_adj))

    # --- Assemble all neighbours ---
    dist_parts = [same_dists] + [p[0] for p in cross_parts]
    zdelta_parts = [np.zeros_like(same_dists)] + [p[2] for p in cross_parts]

    all_dists = np.concatenate(dist_parts, axis=1)
    all_z_deltas = np.concatenate(zdelta_parts, axis=1)
    n_neigh = all_dists.shape[1]

    if n_neigh == 0:
        return raw_i.copy()

    # --- Bilateral weights ---
    log_w = -(all_dists * all_dists) / xy_sigma2 - (all_z_deltas * all_z_deltas) / z_sigma2
    log_w = np.clip(log_w, -50.0, 50.0)
    weights = np.exp(log_w)  # (n_i, N)

    # --- Per-spot WLS aggregation ---
    S0 = weights.sum(axis=1)
    S1 = (weights * all_z_deltas).sum(axis=1)
    S2 = (weights * all_z_deltas * all_z_deltas).sum(axis=1)

    A00 = S0 + ridge
    A11 = S2 + ridge
    A01 = S1
    det = A00 * A11 - A01 * A01

    valid = np.isfinite(det) & (det > 1e-30)
    coef_b0 = np.where(valid, A11 / det, 0.0)
    coef_b1 = np.where(valid, -A01 / det, 0.0)

    # --- Build source batches: (src_z, index_array) ---
    source_batches: List[Tuple[float, np.ndarray]] = []
    if same_idxs.shape[1] > 0:
        source_batches.append((z_i, same_idxs))
    for (_, idxs, _, z_adj) in cross_parts:
        if idxs.shape[1] > 0:
            source_batches.append((z_adj, idxs))

    # --- Process genes in chunks ---
    smoothed_log1p = np.zeros_like(log1p_i)

    for g_start in range(0, n_genes, _GENE_CHUNK_SIZE):
        g_end = min(g_start + _GENE_CHUNK_SIZE, n_genes)
        chunk_size = g_end - g_start

        b0_chunk = np.zeros((n_i, chunk_size), dtype=np.float64)
        b1_chunk = np.zeros((n_i, chunk_size), dtype=np.float64)

        col = 0
        for src_z, idxs in source_batches:
            n_neigh_batch = idxs.shape[1]
            if n_neigh_batch == 0:
                continue
            neigh_expr = z_to_log1p[src_z][idxs, g_start:g_end]
            w_batch = weights[:, col : col + n_neigh_batch]
            zt_batch = all_z_deltas[:, col : col + n_neigh_batch]

            b0_chunk += (w_batch[:, :, None] * neigh_expr).sum(axis=1)
            b1_chunk += (w_batch[:, :, None] * zt_batch[:, :, None] * neigh_expr).sum(axis=1)
            col += n_neigh_batch

        y_hat = coef_b0[:, None] * b0_chunk + coef_b1[:, None] * b1_chunk

        invalid_mask = ~valid
        y_hat[invalid_mask, :] = log1p_i[invalid_mask, g_start:g_end]

        smoothed_log1p[:, g_start:g_end] = y_hat

    # --- Soft blend ---
    smoothed_counts = np.expm1(np.clip(smoothed_log1p, 0.0, None))
    blended = (1.0 - blend_lambda) * raw_i + blend_lambda * smoothed_counts
    return np.rint(np.clip(blended, 0.0, None)).astype(np.float32)




def smooth_cross_z_spots(
    z_slices: Dict[float, ad.AnnData],
    *,
    k_xy: int = 10,
    k_z: int = 5,
    xy_sigma: Optional[float] = None,
    z_sigma: Optional[float] = None,
    blend_lambda: float = 0.3,
    ridge: float = 1e-3,
    genes: Optional[Sequence[str]] = None,
    layer: str = "counts",
    progress: bool = True,
) -> Dict[float, ad.AnnData]:
    """Apply cross-z spot-level trajectory smoothing.

    For each spot, fits a local linear model of expression vs. z-coordinate
    using bilateral-weighted neighbours from the same and adjacent z-slices.
    The predicted expression at the spot's own z-level becomes the smoothed
    value, which is then soft-blended with the original count.

    Parameters
    ----------
    z_slices : dict of float -> AnnData
        Slices keyed by z-coordinate.  Each AnnData must have spatial
        coordinates in ``.obsm['spatial']``.
    k_xy : int
        Number of same-slice spatial neighbours per spot.  Default 10.
    k_z : int
        Number of cross-z neighbours per spot drawn from adjacent z-slices.
        Default 5.
    xy_sigma : float or None
        Spatial bandwidth for bilateral weight.  If *None* (default),
        computed as ``0.15 * median_NN_distance`` in normalised space.
    z_sigma : float or None
        Z-axis bandwidth for bilateral weight.  If *None* (default),
        uses 1.0 in unitless (median-normalised) z-space.
    blend_lambda : float
        Soft-blend fraction.  0 = original counts, 1 = fully smoothed.
        Default 0.3.
    ridge : float
        Ridge penalty added to the diagonal of the WLS normal equations.
        Default 1e-3.
    genes : sequence of str or None
        Genes to smooth.  If *None*, uses the intersection of ``.var_names``
        across all slices.
    layer : str
        Layer key for input counts.  Default ``"counts"``.
    progress : bool
        If True, print per-slice progress information.

    Returns
    -------
    dict of float -> AnnData
        New AnnData objects with smoothed counts in ``.X``.  Original AnnData
        objects are NOT modified in-place.  Only genes present in the resolved
        gene list are modified; all original genes are preserved in the output.
    """
    # --- Validate ---
    sorted_z = sorted(z_slices.keys())
    n_slices = len(sorted_z)
    if n_slices == 0:
        return {}
    if n_slices == 1:
        return {sorted_z[0]: z_slices[sorted_z[0]].copy()}

    # --- Resolve genes ---
    gene_list = _resolve_genes(z_slices, sorted_z, genes)
    n_genes = len(gene_list)
    if n_genes == 0:
        raise ValueError("No genes to smooth (empty intersection or gene list).")

    # --- Extract data from all slices ---
    z_to_coords: Dict[float, np.ndarray] = {}
    z_to_spot_z: Dict[float, np.ndarray] = {}
    z_to_raw: Dict[float, np.ndarray] = {}
    z_to_log1p: Dict[float, np.ndarray] = {}
    z_to_gene_map: Dict[float, Dict[str, int]] = {}
    n_spots: Dict[float, int] = {}

    for z in sorted_z:
        adata = z_slices[z]
        coords, spot_z = _extract_spatial_z(adata, float(z))
        raw_aligned, gene_map = _align_expression(adata, gene_list, layer)
        z_to_coords[z] = coords
        z_to_spot_z[z] = spot_z
        z_to_raw[z] = raw_aligned
        z_to_log1p[z] = np.log1p(raw_aligned)
        z_to_gene_map[z] = gene_map
        n_spots[z] = raw_aligned.shape[0]

    # --- Coordinate normalisation ---
    all_coords = np.concatenate(list(z_to_coords.values()), axis=0)
    xy_scale = _robust_std(all_coords, axis=0)
    xy_scale = np.where(xy_scale < 1e-12, 1.0, xy_scale)

    z_gaps = np.diff(np.array(sorted_z, dtype=np.float64))
    median_z_gap = float(np.median(z_gaps)) if len(z_gaps) > 0 else 1.0
    if median_z_gap < 1e-12:
        median_z_gap = 1.0

    z_to_coords_norm: Dict[float, np.ndarray] = {
        z: coords / xy_scale for z, coords in z_to_coords.items()
    }
    z_to_spot_z_norm: Dict[float, np.ndarray] = {
        z: sz / median_z_gap for z, sz in z_to_spot_z.items()
    }

    # --- Auto-tune sigma ---
    if xy_sigma is None:
        sample_n = min(5000, int(all_coords.shape[0]))
        rng_local = np.random.default_rng(42)
        idx = rng_local.choice(int(all_coords.shape[0]), size=sample_n, replace=False)
        sample_coords = all_coords[idx] / xy_scale
        xy_sigma_val = 0.15 * _estimate_median_nn_dist(sample_coords)
    else:
        xy_sigma_val = float(xy_sigma) / float(np.mean(xy_scale))

    if z_sigma is None:
        z_sigma_val = 1.0
    else:
        z_sigma_val = float(z_sigma) / median_z_gap

    xy_sigma2 = 2.0 * xy_sigma_val * xy_sigma_val
    z_sigma2 = 2.0 * z_sigma_val * z_sigma_val

    # --- Build NN indices ---
    max_nn = max(k_xy + 1, k_z) + 5
    z_to_nn: Dict[float, NearestNeighbors] = {}
    for z in sorted_z:
        nn = NearestNeighbors(n_neighbors=min(max_nn, n_spots[z]), metric="euclidean")
        nn.fit(z_to_coords_norm[z])
        z_to_nn[z] = nn

    # --- Smooth each slice ---
    result: Dict[float, ad.AnnData] = {}

    for slice_idx, z_i in enumerate(sorted_z):
        if progress:
            print(
                f"  Smoothing z={z_i:.4f} [{slice_idx + 1}/{n_slices}]",
                flush=True,
            )

        blended = _smooth_one_slice(
            z_i=z_i,
            slice_idx=slice_idx,
            sorted_z=sorted_z,
            n_i=n_spots[z_i],
            log1p_i=z_to_log1p[z_i],
            raw_i=z_to_raw[z_i],
            coords_i_norm=z_to_coords_norm[z_i],
            spot_z_i_norm=z_to_spot_z_norm[z_i],
            z_to_log1p=z_to_log1p,
            z_to_coords_norm=z_to_coords_norm,
            z_to_spot_z_norm=z_to_spot_z_norm,
            z_to_nn=z_to_nn,
            n_spots=n_spots,
            k_xy=k_xy,
            k_z=k_z,
            xy_sigma2=xy_sigma2,
            z_sigma2=z_sigma2,
            blend_lambda=blend_lambda,
            ridge=ridge,
            n_genes=n_genes,
        )

        # --- Build result AnnData ---
        orig = z_slices[z_i]
        new_adata = orig.copy()
        for gi, g in enumerate(gene_list):
            if g in z_to_gene_map[z_i]:
                orig_col = z_to_gene_map[z_i][g]
                new_adata.X[:, orig_col] = blended[:, gi]

        result[z_i] = new_adata

    return result




def compute_spot_z_autocorrelation(
    z_slices: Dict[float, ad.AnnData],
    *,
    k_z: int = 5,
    genes: Optional[Sequence[str]] = None,
    layer: str = "counts",
) -> float:
    """Compute per-spot z-autocorrelation.

    For each spot, computes the Pearson correlation (across genes) between
    the spot's own expression and the distance-weighted average of its
    cross-z neighbours, then returns the mean correlation across all spots.

    Boundary slices use only the single adjacent slice available.

    Parameters
    ----------
    z_slices : dict of float -> AnnData
        Slices keyed by z-coordinate.
    k_z : int
        Number of cross-z neighbours per spot (default 5).
    genes : sequence of str or None
        Genes to include.  If *None*, uses the intersection across slices.
    layer : str
        Layer key for input counts.  Default ``"counts"``.

    Returns
    -------
    float
        Mean per-spot z-autocorrelation.  Spots with no valid neighbours or
        too few varying genes contribute NaN and are excluded.
    """
    sorted_z = sorted(z_slices.keys())
    if len(sorted_z) < 2:
        return float("nan")

    gene_list = _resolve_genes(z_slices, sorted_z, genes)
    if len(gene_list) == 0:
        return float("nan")

    z_to_coords: Dict[float, np.ndarray] = {}
    z_to_log1p: Dict[float, np.ndarray] = {}
    n_spots: Dict[float, int] = {}

    for z in sorted_z:
        adata = z_slices[z]
        coords, _ = _extract_spatial_z(adata, float(z))
        raw_aligned, _ = _align_expression(adata, gene_list, layer)
        z_to_coords[z] = coords
        z_to_log1p[z] = np.log1p(raw_aligned)
        n_spots[z] = raw_aligned.shape[0]

    all_coords = np.concatenate(list(z_to_coords.values()), axis=0)
    xy_scale = _robust_std(all_coords, axis=0)
    xy_scale = np.where(xy_scale < 1e-12, 1.0, xy_scale)

    z_gaps = np.diff(np.array(sorted_z, dtype=np.float64))
    median_z_gap = float(np.median(z_gaps)) if len(z_gaps) > 0 else 1.0
    if median_z_gap < 1e-12:
        median_z_gap = 1.0

    z_to_coords_norm = {z: c / xy_scale for z, c in z_to_coords.items()}

    z_to_nn: Dict[float, NearestNeighbors] = {}
    for z in sorted_z:
        nn = NearestNeighbors(
            n_neighbors=min(max(k_z, 3), n_spots[z]), metric="euclidean"
        )
        nn.fit(z_to_coords_norm[z])
        z_to_nn[z] = nn

    sample_n = min(5000, int(all_coords.shape[0]))
    rng_local = np.random.default_rng(42)
    idx = rng_local.choice(int(all_coords.shape[0]), size=sample_n, replace=False)
    sample_coords = all_coords[idx] / xy_scale
    xy_sigma_val = 0.15 * _estimate_median_nn_dist(sample_coords)
    xy_sigma2 = 2.0 * xy_sigma_val * xy_sigma_val
    z_sigma2 = 2.0  # z_sigma = 1.0 unitless; 2 * 1^2 = 2

    correlations: List[float] = []

    for slice_idx, z_i in enumerate(sorted_z):
        n_i = n_spots[z_i]
        coords_i = z_to_coords_norm[z_i]
        expr_i = z_to_log1p[z_i]

        adjacent_z: List[float] = []
        if slice_idx > 0:
            adjacent_z.append(sorted_z[slice_idx - 1])
        if slice_idx < len(sorted_z) - 1:
            adjacent_z.append(sorted_z[slice_idx + 1])

        n_adj = len(adjacent_z)
        k_z_each = max(1, (k_z + n_adj - 1) // n_adj) if n_adj > 0 else 0

        neigh_dists: List[np.ndarray] = []
        neigh_idxs: List[np.ndarray] = []
        neigh_z_deltas: List[np.ndarray] = []
        neigh_src: List[float] = []

        z_i_norm = float(z_i) / median_z_gap

        for z_adj in adjacent_z:
            n_adj_spots = n_spots[z_adj]
            k_eff = min(k_z_each, n_adj_spots)
            if k_eff < 1:
                continue
            nn_adj = z_to_nn[z_adj]
            c_dists, c_idxs = nn_adj.kneighbors(coords_i, n_neighbors=k_eff)
            neigh_dists.append(c_dists)
            neigh_idxs.append(c_idxs)
            neigh_src.append(z_adj)
            z_adj_norm = float(z_adj) / median_z_gap
            neigh_z_deltas.append(
                np.full(c_dists.shape, z_adj_norm - z_i_norm, dtype=np.float64)
            )

        if not neigh_dists:
            continue

        all_dists = np.concatenate(neigh_dists, axis=1)
        all_z_deltas = np.concatenate(neigh_z_deltas, axis=1)

        log_w = (
            -(all_dists * all_dists) / xy_sigma2
            - (all_z_deltas * all_z_deltas) / z_sigma2
        )
        log_w = np.clip(log_w, -50.0, 50.0)
        weights = np.exp(log_w)

        w_sum = weights.sum(axis=1, keepdims=True)
        w_norm = np.where(w_sum > 0, weights / w_sum, 1.0 / weights.shape[1])

        n_genes_total = expr_i.shape[1]
        y_bar = np.zeros((n_i, n_genes_total), dtype=np.float64)

        col = 0
        for bi in range(len(neigh_dists)):
            n_batch = neigh_idxs[bi].shape[1]
            w_batch = w_norm[:, col : col + n_batch]
            src_z = neigh_src[bi]
            neigh_expr_all = z_to_log1p[src_z][neigh_idxs[bi], :]
            y_bar += (w_batch[:, :, None] * neigh_expr_all).sum(axis=1)
            col += n_batch

        for si in range(n_i):
            if not np.any(np.isfinite(w_norm[si])):
                continue
            corr = _safe_pearson(expr_i[si, :], y_bar[si, :])
            if np.isfinite(corr):
                correlations.append(corr)

    if not correlations:
        return float("nan")
    return float(np.mean(correlations))
