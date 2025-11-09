"""
Alignment with Gap: Step 1 of Interpolation Benchmarking Pipeline

Performs spatial alignment between non-consecutive tissue slices (A and B),
skipping the middle slice (M). Results are used by interpolation_sim_pipeline.py
to reconstruct and evaluate the skipped slice.

Usage:
    python alignment_with_gap.py [--data-dir PATH] [--device cuda|cpu]
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import spateo as st
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

os.makedirs(os.path.expanduser('~/.cache/keops2.3'), exist_ok=True)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_paths import get_merfish_slices, check_dataset_exists

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'reconstructions'
OUTPUT_DIR.mkdir(exist_ok=True)


def find_slice_files(folder):
    """Find and sort slice files, excluding outputs and aligned files."""
    all_files = sorted(folder.glob('*.h5ad'))
    files = []
    for p in all_files:
        if str(p).startswith(str(OUTPUT_DIR)):
            continue
        if '_aligned' in p.name or '_predicted' in p.name or 'manifest' in p.name.lower():
            continue
        files.append(p)

    float_re = re.compile(r"-?\d+\.\d+|-?\d+")
    def key_fn(p):
        name = p.stem
        matches = float_re.findall(name)
        if matches:
            try:
                return float(matches[-1])
            except:
                pass
        for sep in ("_", "-"):
            for part in name.split(sep):
                try:
                    return float(part)
                except:
                    continue
        return name

    return sorted(files, key=key_fn)


def safe_write(adata, path):
    """Write AnnData object, converting integer keys to strings."""
    def fix_dict_keys(obj):
        if isinstance(obj, dict):
            return {str(k): fix_dict_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_dict_keys(x) for x in obj]
        return obj
    
    if hasattr(adata, 'uns') and isinstance(adata.uns, dict):
        adata.uns = fix_dict_keys(adata.uns)
    adata.write_h5ad(path)


def align_pair(path_A, path_B, out_prefix, device='cuda'):
    """Align a pair of slices skipping the middle slice."""
    print(f"\nAligning: {Path(path_A).stem} → {Path(path_B).stem}")
    adA = sc.read_h5ad(path_A)
    adB = sc.read_h5ad(path_B)

    for ad in (adA, adB):
        if 'counts' not in ad.layers:
            ad.layers['counts'] = ad.X.copy()
        sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)
    
    genes_common = sorted(list(set(adA.var_names) & set(adB.var_names)))
    if len(genes_common) == 0:
        print("  ✗ No common genes found")
        return None

    genes_use = genes_common
    if 'highly_variable' not in adA.var:
        sc.pp.highly_variable_genes(adA, n_top_genes=2000)
    if 'highly_variable' not in adB.var:
        sc.pp.highly_variable_genes(adB, n_top_genes=2000)
    
    hvg_common = list(set(adA.var_names[adA.var['highly_variable']]) & 
                      set(adB.var_names[adB.var['highly_variable']]) & 
                      set(genes_common))
    if len(hvg_common) >= 50:
        genes_use = sorted(hvg_common)

    to_dense = lambda X: X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    XA = to_dense(adA[:, genes_use].X)
    XB = to_dense(adB[:, genes_use].X)
    Xstack = np.vstack([XA, XB])
    pca = PCA(n_components=min(30, Xstack.shape[1], Xstack.shape[0]-1))
    Xp = pca.fit_transform(Xstack)
    
    nA = XA.shape[0]
    adA.obsm['X_expr_pca'] = Xp[:nA, :]
    adB.obsm['X_expr_pca'] = Xp[nA:, :]
    print(f"  ✓ PCA on {len(genes_use)} genes → {adA.obsm['X_expr_pca'].shape[1]} components")

    aligned_slices, pis = st.align.morpho_align(
        models=[adA, adB],
        rep_layer='X_expr_pca',
        rep_field='obsm',
        dissimilarity='cos',
        verbose=False,
        spatial_key='spatial',
        key_added='align_spatial',
        device=device,
        SVI_mode=False,
    )

    adA_aligned, adB_aligned = aligned_slices
    pi = pis[0]
    pi_np = pi.cpu().numpy() if hasattr(pi, 'cpu') else np.array(pi)

    outA = OUTPUT_DIR / f"{out_prefix}_A_aligned.h5ad"
    outB = OUTPUT_DIR / f"{out_prefix}_B_aligned.h5ad"
    outpi = OUTPUT_DIR / f"{out_prefix}_alignment.npy"
    
    safe_write(adA_aligned, outA)
    safe_write(adB_aligned, outB)
    np.save(outpi, pi_np)

    print(f"Saved aligned files:\n  A: {outA}\n  B: {outB}\n  Alignment: {outpi}")
    return {'A': str(outA), 'B': str(outB), 'pi': str(outpi), 'alignment': str(outpi)}


def main(data_dir=None, device='cuda'):
    """
    Main function to align consecutive slices with gaps.
    
    Args:
        data_dir: Directory containing input slice files (default: use MERFISH slices 5-9)
        device: Device for alignment ('cuda' or 'cpu')
    """
    # Use provided data directory or default to MERFISH slices
    if data_dir is None:
        print("Using default MERFISH slices (005-009) from repository data/")
        try:
            files = get_merfish_slices(5, 9)
            for f in files:
                check_dataset_exists(f)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nAlternatively, place your slice files in this directory and run with:")
            print(f"  python {Path(__file__).name} --data-dir .")
            return
    else:
        data_dir = Path(data_dir)
        files = find_slice_files(data_dir)
    
    print(f"\nFound {len(files)} slice files:")
    for f in files:
        print(f'  - {f.name}')

    if len(files) < 3:
        print("\nError: Need at least 3 consecutive slices to create alignment triples")
        return

    # Process consecutive triples (A, M, B)
    results = []
    for i in range(len(files) - 2):
        A = files[i]
        M = files[i+1]
        B = files[i+2]
        out_prefix = f"triple_{i:02d}"
        print(f"\n{'='*60}")
        print(f"Processing triple {i}: {A.stem} -> [{M.stem}] -> {B.stem}")
        print(f"{'='*60}")
        res = align_pair(A, B, out_prefix, device=device)
        if res is not None:
            # Add middle slice path and metadata
            res['middle'] = str(M)
            res['triple_id'] = out_prefix
            res['slice_A_name'] = A.stem
            res['slice_M_name'] = M.stem
            res['slice_B_name'] = B.stem
            results.append(res)

    # Save manifest
    manifest_path = OUTPUT_DIR / 'alignment_with_gap_manifest.json'
    with open(manifest_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Alignment complete! Processed {len(results)}/{len(files)-2} triples")
    print(f"✓ Manifest saved to: {manifest_path}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Align consecutive spatial transcriptomics slices with gaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default MERFISH slices from repository data/
  python alignment_with_gap.py
  
  # Use custom data directory
  python alignment_with_gap.py --data-dir /path/to/slices
  
  # Use CPU instead of GPU
  python alignment_with_gap.py --device cpu
        """
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing slice .h5ad files (default: use MERFISH_005-009)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for alignment computation (default: cuda)')
    args = parser.parse_args()
    
    main(data_dir=args.data_dir, device=args.device)
