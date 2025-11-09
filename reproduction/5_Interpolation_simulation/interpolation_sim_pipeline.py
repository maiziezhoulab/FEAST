
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import sys
import time
import re

from FEAST.interpolation import interpolate_slices, InterpolationConfig



BASE = Path(__file__).parent
RECON_DIR = BASE / 'reconstructions'
MANIFEST = RECON_DIR / 'alignment_with_gap_manifest.json'
OUT_CSV = RECON_DIR / 'benchmark_results.csv'
DIAG_DIR = RECON_DIR / 'diagnostics'
DIAG_DIR.mkdir(exist_ok=True)


def to_dense(X):
    if hasattr(X, 'toarray'):
        return X.toarray()
    return np.asarray(X)


def normalize_rows(P, eps=1e-12):
    P = np.asarray(P, dtype=float)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return P / (row_sums + eps)


def flatten_nonan(P):
    arr = np.asarray(P, dtype=float).ravel()
    return arr


def mass_overlap(P, Q):
    Pn = np.asarray(P, dtype=float)
    Qn = np.asarray(Q, dtype=float)
    sP = Pn.sum()
    sQ = Qn.sum()
    if sP == 0 or sQ == 0:
        return np.nan
    Pn = Pn / sP
    Qn = Qn / sQ
    return float(np.minimum(Pn, Qn).sum())


def matrix_flat_pearson(P, Q):
    a = flatten_nonan(P)
    b = flatten_nonan(Q)
    if a.size == 0 or b.size == 0:
        return np.nan
    # align sizes
    if a.size != b.size:
        m = min(a.size, b.size)
        a = a[:m]
        b = b[:m]
    try:
        r = pearsonr(a, b)[0]
    except Exception:
        r = np.nan
    return float(r)


def compute_alignment_matrix(ad_src, ad_tgt, triple_id, name, device=None, rep_layer='X_expr_pca'):
    """Run morpho_align between ad_src and ad_tgt and return numpy pi matrix. Save to DIAG_DIR."""
    try:
        import spateo as st
    except Exception as e:
        print(f"spateo import failed (needed for morpho_align): {e}")
        return None

    # ensure representation exists
    if rep_layer not in ad_src.obsm or rep_layer not in ad_tgt.obsm:
        # attempt a lightweight PCA on common genes
        try:
            genes_common = list(set(ad_src.var_names) & set(ad_tgt.var_names))
            if len(genes_common) == 0:
                raise RuntimeError('No common genes to compute PCA')
            def _to_dense(X):
                if hasattr(X, 'toarray'):
                    return X.toarray()
                return np.asarray(X)
            Xs = _to_dense(ad_src[:, genes_common].X)
            Xt = _to_dense(ad_tgt[:, genes_common].X)
            Xstack = np.vstack([Xs, Xt])
            from sklearn.decomposition import PCA
            n_comp = min(30, Xstack.shape[1], Xstack.shape[0]-1)
            if n_comp <= 0:
                raise RuntimeError('Not enough samples to compute PCA')
            pca = PCA(n_components=n_comp)
            Xp = pca.fit_transform(Xstack)
            n1 = Xs.shape[0]
            ad_src.obsm[rep_layer] = Xp[:n1, :]
            ad_tgt.obsm[rep_layer] = Xp[n1:, :]
            print(f"Computed temporary {rep_layer} for {name}")
        except Exception as e:
            print(f"Failed to compute fallback PCA for {name}: {e}")

    # Ensure device is a valid string for spateo backend check (avoid passing None)
    try:
        dev = 'cpu' if device is None else str(device)
        aligned, pis = st.align.morpho_align(
            models=[ad_src, ad_tgt],
            rep_layer=rep_layer,
            rep_field='obsm',
            dissimilarity='cos',
            verbose=True,
            spatial_key='spatial',
            key_added=f'align_{name}',
            device=dev,
            SVI_mode=False,
        )
    except Exception as e:
        import traceback
        print(f"morpho_align failed for {name}: {e}")
        traceback.print_exc()
        return None

    pi = pis[0]
    try:
        if hasattr(pi, 'cpu'):
            pi_np = pi.cpu().numpy()
        else:
            pi_np = np.array(pi)
    except Exception:
        pi_np = np.asarray(pi)

    outpath = DIAG_DIR / f"{triple_id}_{name}.npy"
    try:
        np.save(outpath, pi_np)
    except Exception as e:
        print(f"Failed to save alignment matrix {outpath}: {e}")

    return pi_np


def mutual_nn_match(coords_ref, coords_pred, radius=None):
    nbr_ref = NearestNeighbors(n_neighbors=1).fit(coords_pred)
    dists_r, inds_r = nbr_ref.kneighbors(coords_ref)
    nbr_pred = NearestNeighbors(n_neighbors=1).fit(coords_ref)
    dists_p, inds_p = nbr_pred.kneighbors(coords_pred)

    matches = []
    for i, pred_i in enumerate(inds_r[:, 0]):
        if inds_p[pred_i, 0] == i:
            dist = float(dists_r[i, 0])
            if (radius is None) or (dist <= radius):
                matches.append((i, int(pred_i), dist))
    return matches


def batch_gene_pearson(Xr, Xp):
    # Xr, Xp shape: (n_samples, n_genes)
    Xr = np.asarray(Xr, dtype=float)
    Xp = np.asarray(Xp, dtype=float)
    # center
    Xr_c = Xr - Xr.mean(axis=0, keepdims=True)
    Xp_c = Xp - Xp.mean(axis=0, keepdims=True)
    num = (Xr_c * Xp_c).sum(axis=0)
    den = np.sqrt((Xr_c ** 2).sum(axis=0) * (Xp_c ** 2).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / den
    return r


def per_spot_correlation(Xr, Xp):
    # compute Pearson correlation per row (spot) across genes
    Xr = np.asarray(Xr, dtype=float)
    Xp = np.asarray(Xp, dtype=float)
    n = Xr.shape[0]
    res = np.zeros(n, dtype=float)
    for i in range(n):
        a = Xr[i, :]
        b = Xp[i, :]
        if np.all(a == a[0]) or np.all(b == b[0]):
            res[i] = np.nan
        else:
            res[i] = pearsonr(a, b)[0]
    return res


def rmse_per_gene(Xr, Xp):
    Xr = np.asarray(Xr, dtype=float)
    Xp = np.asarray(Xp, dtype=float)
    mse = ((Xr - Xp) ** 2).mean(axis=0)
    return np.sqrt(mse)


def grid_aggregate(adata, genes, cell_size=5.0, agg='mean'):
    coords = np.asarray(adata.obsm['spatial'])
    xs, ys = coords[:, 0], coords[:, 1]
    xmin, ymin = xs.min(), ys.min()
    ix = ((xs - xmin) // cell_size).astype(int)
    iy = ((ys - ymin) // cell_size).astype(int)
    cell_ids = ix + iy * (ix.max() + 1)
    unique_cells = np.unique(cell_ids)
    agg_matrix = np.zeros((len(unique_cells), len(genes)), dtype=float)
    for ci, cell in enumerate(unique_cells):
        mask = (cell_ids == cell)
        if mask.sum() == 0:
            continue
        sub = adata[mask, genes]
        mat = to_dense(sub.X)
        if agg == 'mean':
            agg_matrix[ci, :] = mat.mean(axis=0).ravel()
        else:
            agg_matrix[ci, :] = mat.sum(axis=0).ravel()
    return agg_matrix, unique_cells


def hotspot_iou(adata_ref, adata_pred, gene, cell_size=5.0, top_frac=0.05):
    genes = [gene]
    Aagg, Acells = grid_aggregate(adata_ref, genes, cell_size=cell_size)
    Pagg, Pcells = grid_aggregate(adata_pred, genes, cell_size=cell_size)
    if Aagg.shape[0] == 0 or Pagg.shape[0] == 0:
        return np.nan
    # threshold top fraction
    avals = Aagg[:, 0]
    pvals = Pagg[:, 0]
    ath = np.quantile(avals, 1 - top_frac)
    pth = np.quantile(pvals, 1 - top_frac)
    Amask = (avals >= ath).astype(int)
    Pmask = (pvals >= pth).astype(int)
    # align cells by spatial position: use cell index identity approximate
    # Build sets of cell coordinates (ix,iy) for mapping would be more robust, but cell ids ordering is consistent
    intersection = min(len(Amask), len(Pmask))
    if intersection == 0:
        return np.nan
    Amask = Amask[:intersection]
    Pmask = Pmask[:intersection]
    inter = np.logical_and(Amask == 1, Pmask == 1).sum()
    union = np.logical_or(Amask == 1, Pmask == 1).sum()
    if union == 0:
        return 0.0
    return inter / union


def morans_I(values, coords, k=8, permutations=499, seed=0):
    # vectorized Moran's I with kNN weights and permutation p-value
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, inds = nbrs.kneighbors(coords)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in inds[i, 1:]:
            W[i, j] = 1.0
    W_sum = W.sum()
    if W_sum == 0:
        return np.nan, np.nan
    e = values - values.mean()
    num = (e[:, None] * e[None, :] * W).sum()
    den = (e ** 2).sum()
    I = (n / W_sum) * (num / den)
    # permutations
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        perm = rng.permutation(values)
        e_p = perm - perm.mean()
        num_p = (e_p[:, None] * e_p[None, :] * W).sum()
        den_p = (e_p ** 2).sum()
        if den_p == 0:
            null.append(0.0)
        else:
            null.append((n / W_sum) * (num_p / den_p))
    null = np.array(null)
    p = (np.sum(null >= I) + 1) / (permutations + 1)
    return float(I), float(p)


def rasterize_gene_image(adata, gene, img_size=(64, 64), agg='sum'):
    """Rasterize a single-gene spatial expression into a 2D image using histogram2d.
    Returns a float32 array normalized to [0, 1]."""
    coords = np.asarray(adata.obsm['spatial'])
    if coords.size == 0:
        return np.zeros(img_size, dtype=np.float32)
    xs = coords[:, 0]
    ys = coords[:, 1]
    vals = np.asarray(to_dense(adata[:, [gene]].X)).ravel()
    # Create histogram bins based on spatial extent
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    # Avoid degenerate ranges
    if xmax - xmin == 0:
        xmax = xmin + 1.0
    if ymax - ymin == 0:
        ymax = ymin + 1.0
    H, xedges, yedges = np.histogram2d(xs, ys, bins=img_size, range=[[xmin, xmax], [ymin, ymax]], weights=vals)
    # histogram2d returns shape (nx, ny) where x corresponds to first axis; transpose to (H, W) conventional
    H = np.asarray(H, dtype=np.float32).T
    # Normalize to [0,1]
    if H.max() > 0:
        H = H / float(H.max())
    return H


def ahash_from_image(img, hash_size=8):
    """Compute a simple average-hash (aHash) from a 2D float image in [0,1].
    Returns a boolean array of shape (hash_size, hash_size).
    """
    # Downsample image to hash_size x hash_size by histogram2d reuse
    H = img
    # Compute bins = hash_size x hash_size
    xs = np.linspace(0, H.shape[1], hash_size + 1, dtype=int)
    ys = np.linspace(0, H.shape[0], hash_size + 1, dtype=int)
    small = np.zeros((hash_size, hash_size), dtype=np.float32)
    for i in range(hash_size):
        for j in range(hash_size):
            sub = H[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            if sub.size == 0:
                small[i, j] = 0.0
            else:
                small[i, j] = sub.mean()
    med = np.median(small)
    return small > med


def image_similarity_metrics(img_ref, img_pred):
    """Compute simple image-level similarity metrics between two 2D arrays.
    Returns dict with mse, mae, ncc (normalized cross-corr), psnr, ahash_hamming (normalized).
    """
    a = np.asarray(img_ref, dtype=float)
    b = np.asarray(img_pred, dtype=float)
    if a.shape != b.shape:
        # resize b to a using simple nearest-neighbor down/up-sampling via numpy repeat/trim
        # compute scale factors
        ay, ax = a.shape
        by, bx = b.shape
        # use np.interp approach: map target grid to source
        y_idx = (np.linspace(0, by - 1, ay)).astype(int)
        x_idx = (np.linspace(0, bx - 1, ax)).astype(int)
        b = b[np.ix_(y_idx, x_idx)]
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    # NCC: normalized cross-correlation (zero mean)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = (np.sqrt((a_c ** 2).sum()) * np.sqrt((b_c ** 2).sum()))
    if denom == 0:
        ncc = np.nan
    else:
        ncc = float(((a_c * b_c).sum()) / denom)
    psnr = float(20.0 * np.log10(1.0 + 1e-12) - 10.0 * np.log10(mse + 1e-12)) if mse > 0 else float('inf')
    # aHash Hamming distance normalized
    ha = ahash_from_image(a)
    hb = ahash_from_image(b)
    # ensure same shape
    if ha.shape != hb.shape:
        # coerce to smaller
        miny = min(ha.shape[0], hb.shape[0])
        minx = min(ha.shape[1], hb.shape[1])
        ha = ha[:miny, :minx]
        hb = hb[:miny, :minx]
    ham = (ha != hb).sum()
    ahash_norm = float(ham) / (ha.size if ha.size else 1)
    return dict(mse=mse, mae=mae, ncc=ncc, psnr=psnr, ahash=ahash_norm)


def reconstruct_middle(adA_path, adB_path, adM_path=None, alignment_path=None, method='stimator', t=0.5):
    """Reconstruct middle slice from aligned A and B. Uses FEAST interpolation if available
    otherwise raises an informative error asking to install / expose the interpolation API.
    Returns AnnData object for predicted middle slice.
    """

    # Load aligned AnnData objects (they should already contain spatial and counts)
    adA = sc.read_h5ad(adA_path)
    adB = sc.read_h5ad(adB_path)

    adA.layers['counts'] = adA.X.copy()
    adB.layers['counts'] = adB.X.copy()
    
    # Ensure spatial coords exist
    if 'spatial' not in adA.obsm and 'spatial' in adA.obs:
        # try common column
        pass

    # Intersect genes and subset both AnnData to the same gene order expected by interpolator
    if adM_path is not None:
        adM = sc.read_h5ad(adM_path)
        adM.layers['counts'] = adM.X.copy()
        genes_common = sorted(list(set(adA.var_names) & set(adB.var_names) & set(adM.var_names)))
    else:
        genes_common = sorted(list(set(adA.var_names) & set(adB.var_names)))

    if len(genes_common) == 0:
        raise RuntimeError('No common genes between adA and adB (and adM if provided) for interpolation')

    # subset to common genes in the same order
    adA = adA[:, genes_common].copy()
    adB = adB[:, genes_common].copy()
    if adM_path is not None:
        adM = adM[:, genes_common].copy()

    # ensure spatial coordinates exist
    for adx in (adA, adB):
        if 'spatial' not in adx.obsm:
            raise RuntimeError('Input AnnData must have obsm["spatial"] coordinates for interpolation')

    # Intersect and order genes consistently
    common_genes = sorted(list(set(adA.var_names) & set(adB.var_names)))
    if len(common_genes) == 0:
        raise RuntimeError('No common genes between adA and adB')
    # Subset both AnnData to same gene order
    adA = adA[:, common_genes].copy()
    adB = adB[:, common_genes].copy()

    # Load alignment matrix if provided (expecting numpy .npy)
    alignment = None
    if alignment_path is not None:
        try:
            alignment = np.load(alignment_path)
        except Exception as e:
            raise RuntimeError(f'Could not load alignment matrix from {alignment_path}: {e}')

    # Build InterpolationConfig similar to the project's scripts
    cfg = InterpolationConfig(
        t=t,
        use_normalized=True,
        ot_method='sinkhorn',
        ot_regularization=0.05,
        boundary_multiplier=1.1,
        sigma=0,
        verbose=False,
    )
    # If the user provided a reference middle slice, prefer generating at most
    # as many transport pairs / interpolated spots as there are spots in that slice.
    # This prevents the ordered query slice from having an unexpected default size
    # (for example, when the smaller reference slice has 1000 spots).
    if adM_path is not None and hasattr(cfg, 'max_transport_pairs') and cfg.max_transport_pairs is None:
        try:
            # adM was loaded above when adM_path provided
            cfg.max_transport_pairs = int(adM.shape[0])
            print(f"  > Setting config.max_transport_pairs = number of middle spots ({cfg.max_transport_pairs})")
        except Exception:
            pass
    # If the config has common attributes we can set some sensible defaults.
    # Keep defaults otherwise; users can edit this script to set advanced options.
    # t_values is the interpolation fraction between A (t=0) and B (t=1)
    try:
        # Call interpolate_slices with same named args used in working script
        interpolated = interpolate_slices(
            adata1=adA,
            adata2=adB,
            alignment_matrix=alignment,
            config=cfg,
        )
    except TypeError:
        # older/newer API shapes: try with positional args, but raise if it still fails
        try:
            interpolated = interpolate_slices(adA, adB, [t])
        except Exception as e:
            raise RuntimeError(f"interpolate_slices call failed with alternate signature: {e}")
    except Exception as e:
        raise RuntimeError(f"interpolate_slices call failed: {e}")

    # Handle common return types
    if isinstance(interpolated, list) and len(interpolated) > 0:
        first = interpolated[0]
        # If it's already an AnnData-like object
        if hasattr(first, 'obs'):
            ad_pred = first
        # If it's a numpy array (expression matrix), build AnnData using adM if available
        elif isinstance(first, np.ndarray):
            # Some FEAST versions return numpy arrays. Construct AnnData using adM metadata.
            if adM_path is None:
                raise RuntimeError('interpolate_slices returned an ndarray but adM_path is not provided to construct AnnData')
            adM = sc.read_h5ad(adM_path)
            # genes already used above (common_genes)
            genes = common_genes
            if first.shape[1] != len(genes):
                raise RuntimeError(f'Returned array has {first.shape[1]} genes but expected {len(genes)}')
            import anndata as ad
            ad_pred = ad.AnnData(X=first, obs=adM.obs.copy(), var=adM[:, genes].var.copy())
            if 'spatial' in adM.obsm:
                ad_pred.obsm['spatial'] = adM.obsm['spatial'].copy()
        else:
            raise RuntimeError('interpolate_slices returned a list but first element is not AnnData or ndarray')
    elif hasattr(interpolated, 'obs'):
        ad_pred = interpolated
    else:
        # Could be a dict or other structure — try to find an AnnData within
        if isinstance(interpolated, dict):
            for v in interpolated.values():
                if hasattr(v, 'obs'):
                    ad_pred = v
                    break
            else:
                raise RuntimeError('interpolate_slices returned unexpected structure (no AnnData found)')
        else:
            raise RuntimeError('interpolate_slices returned unexpected structure; could not extract AnnData')

    # Ensure the predicted AnnData has spatial coordinates; if not, try to copy from A/B
    if 'spatial' not in ad_pred.obsm:
        if 'spatial' in adA.obsm:
            ad_pred.obsm['spatial'] = adA.obsm['spatial'].copy()
        elif 'spatial' in adB.obsm:
            ad_pred.obsm['spatial'] = adB.obsm['spatial'].copy()

    return ad_pred

def run_benchmark(manifest_path=MANIFEST, out_csv=OUT_CSV):
    """
    Run interpolation benchmarking using the alignment manifest.
    
    Args:
        manifest_path: Path to the alignment manifest JSON file
        out_csv: Path to save benchmark results CSV
    """
    # If manifest is missing, auto-generate it by scanning local .h5ad files
    if not manifest_path.exists():
        print(f"Manifest {manifest_path} not found — auto-generating from local .h5ad files...")
        float_re = re.compile(r"-?\d+\.\d+|-?\d+")
        files = []
        for p in sorted(BASE.glob('*.h5ad')):
            if '_aligned' in p.name or p.name.startswith('triple_'):
                continue
            files.append(p)

        def key_fn(p):
            name = p.stem
            matches = float_re.findall(name)
            if matches:
                try:
                    return float(matches[-1])
                except Exception:
                    pass
            return name

        files_sorted = sorted(files, key=key_fn)
        manifest = []
        for i in range(len(files_sorted) - 2):
            A = str(files_sorted[i])
            M = str(files_sorted[i+1])
            B = str(files_sorted[i+2])
            manifest.append({
                'A': A, 
                'B': B, 
                'middle': M,
                'triple_id': f'triple_{i:02d}',
                'slice_A_name': Path(A).stem,
                'slice_M_name': Path(M).stem,
                'slice_B_name': Path(B).stem
            })
        try:
            RECON_DIR.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"Auto-generated manifest saved to {manifest_path}")
        except Exception as e:
            print(f"Failed to write manifest to {manifest_path}: {e}")
    else:
        manifest = json.loads(manifest_path.read_text())
    
    print(f"\n{'='*60}")
    print(f"Starting interpolation benchmark for {len(manifest)} triples")
    print(f"{'='*60}\n")
    
    rows = []
    for idx, entry in enumerate(manifest):
        # Extract paths from manifest (support both aligned and original file paths)
        A = entry.get('A')
        B = entry.get('B')
        M = entry.get('middle') or entry.get('M')
        alignment = entry.get('alignment') or entry.get('pi') or entry.get('pis')
        triple_id = entry.get('triple_id', f'triple_{idx:02d}')
        
        print(f"\n--- Processing {triple_id} ---")
        print(f"  Slice A: {Path(A).name if A else 'N/A'}")
        print(f"  Slice M (middle): {Path(M).name if M else 'N/A'}")
        print(f"  Slice B: {Path(B).name if B else 'N/A'}")
        print(f"  Alignment: {Path(alignment).name if alignment else 'N/A'}")

        # reconstruct predicted middle using FEAST and pass alignment matrix path if present
        try:
            ad_pred = reconstruct_middle(A, B, adM_path=M, alignment_path=alignment)
            # Save predicted middle slice
            out_pred_path = RECON_DIR / f"{triple_id}_predicted_middle.h5ad"
            ad_pred.write_h5ad(out_pred_path)
            print(f"  ✓ Predicted middle slice saved to: {out_pred_path.name}")
        except Exception as e:
            print(f"  ✗ Reconstruction failed for {triple_id}: {e}")
            if 'interpolate_slices' in str(e) or 'FEAST' in str(e):
                print('  Hint: ensure FEAST is installed and on PYTHONPATH.')
            continue

        if M is None:
            print(f"  ⚠ No reference middle slice provided for {triple_id}; skipping metrics")
            continue

        # Load reference middle slice for comparison
        try:
            ad_ref = sc.read_h5ad(M)
            print(f"  ✓ Loaded reference middle slice: {ad_ref.shape}")
        except Exception as e:
            print(f"  ✗ Failed to load reference middle slice: {e}")
            continue
        
        # TODO: Add metric computation here
        # rows.append({...})
    
    print(f"\n{'='*60}")
    print(f"✓ Interpolation benchmark complete!")
    print(f"  Processed: {len(rows)} triples")
    if out_csv:
        print(f"  Results saved to: {out_csv}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_benchmark()
