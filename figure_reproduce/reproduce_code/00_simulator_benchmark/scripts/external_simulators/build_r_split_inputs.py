"""Convert prepared .h5ad inputs to R simulator split format.

Produces per-alias directories under R_split/ with:
  cellinfo.csv    — spot IDs, label column, library size
  geneinfo.csv    — gene IDs
  sparse_matrix.mtx — integer counts in Matrix Market format
  spatial.csv     — x, y coordinates
"""
from __future__ import annotations
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
LABEL_COLUMNS = ['ground_truth', 'cell_type', 'annotation', 'region', 'cluster']

def _resolve_label_column(adata: ad.AnnData) -> str | None:
    for col in LABEL_COLUMNS:
        if col in adata.obs.columns:
            return col
    return None

def _as_integer_counts(adata: ad.AnnData) -> np.ndarray:
    if 'counts' in adata.layers:
        mat = adata.layers['counts']
    else:
        mat = adata.X
    if sp.issparse(mat):
        mat = mat.tocsc()
    if hasattr(mat, 'toarray'):
        dense = mat.toarray()
    else:
        dense = np.asarray(mat)
    if not np.allclose(dense, np.round(dense), atol=1e-06):
        raise ValueError('Count matrix contains non-integer values. Ensure raw integer counts are used.')
    return np.round(dense).astype(np.int64)

def build_r_split(input_dir: Path, output_dir: Path, manifest_path: Path) -> int:
    h5ad_files = sorted(input_dir.glob('*.h5ad'))
    if not h5ad_files:
        print(f'ERROR: no .h5ad files found in {input_dir}')
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h5ad_path in h5ad_files:
        alias = h5ad_path.stem
        sample_dir = output_dir / alias
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f'  {alias} ...', end=' ')
        adata = ad.read_h5ad(h5ad_path)
        label_col = _resolve_label_column(adata)
        counts = _as_integer_counts(adata)
        spatial = adata.obsm['spatial']
        if counts.shape[0] != adata.n_obs or counts.shape[1] != adata.n_vars:
            print('ERROR: count matrix shape mismatch')
            rows.append({'alias': alias, 'n_spots': adata.n_obs, 'n_genes': adata.n_vars, 'status': 'failed'})
            continue
        cellinfo = pd.DataFrame(index=adata.obs_names)
        cellinfo['spot_id'] = adata.obs_names
        if label_col:
            cellinfo['label'] = adata.obs[label_col].values
        cellinfo['library_size'] = counts.sum(axis=1)
        cellinfo.to_csv(sample_dir / 'cellinfo.csv', index=False)
        geneinfo = pd.DataFrame({'gene_id': adata.var_names})
        geneinfo.to_csv(sample_dir / 'geneinfo.csv', index=False)
        sparse_mat = sp.csc_matrix(counts.T)
        scipy.io.mmwrite(str(sample_dir / 'sparse_matrix.mtx'), sparse_mat)
        spatial_df = pd.DataFrame(spatial, columns=['x', 'y'])
        spatial_df.insert(0, 'spot_id', adata.obs_names)
        spatial_df.to_csv(sample_dir / 'spatial.csv', index=False)
        row = {'alias': alias, 'n_spots': adata.n_obs, 'n_genes': adata.n_vars, 'label_column': label_col or 'none', 'count_min': int(counts.min()), 'count_max': int(counts.max()), 'status': 'valid'}
        rows.append(row)
        print(f'{adata.n_obs} x {adata.n_vars} OK')
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    print(f'\nR_split manifest written to {manifest_path} ({len(manifest)} rows)')
    return 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing prepared .h5ad files')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory for R_split samples')
    parser.add_argument('--manifest', type=Path, required=True, help='Path for r_split_manifest.csv')
    args = parser.parse_args()
    return build_r_split(args.input_dir, args.output_dir, args.manifest)
if __name__ == '__main__':
    raise SystemExit(main())
