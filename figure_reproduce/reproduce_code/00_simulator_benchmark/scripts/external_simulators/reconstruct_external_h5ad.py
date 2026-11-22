"""Convert recovered simulator Matrix Market intermediates to AnnData."""
from __future__ import annotations
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

def _read_gene_names(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f'Gene table is empty: {path}')
    for column in ('gene', 'genes', 'gene_name', 'Gene', 'GeneSymbol'):
        if column in frame.columns:
            values = frame[column].astype(str).tolist()
            return values
    candidate_columns = [column for column in frame.columns if not str(column).startswith('Unnamed')]
    column = candidate_columns[0] if candidate_columns else frame.columns[0]
    return frame[column].astype(str).tolist()

def _read_location_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f'Location table is empty: {path}')
    unnamed = [column for column in frame.columns if str(column).startswith('Unnamed')]
    if unnamed and len(frame.columns) > len(unnamed):
        frame = frame.drop(columns=unnamed)
    return frame

def _choose_location_columns(frame: pd.DataFrame) -> tuple[str, str]:
    lower_to_column = {str(column).lower(): column for column in frame.columns}
    for x_name, y_name in (('x', 'y'), ('spatial_x', 'spatial_y'), ('array_row', 'array_col'), ('row', 'col')):
        if x_name in lower_to_column and y_name in lower_to_column:
            return (lower_to_column[x_name], lower_to_column[y_name])
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if len(numeric) < 2:
        raise ValueError('Location table needs x/y columns or at least two numeric columns.')
    return (numeric[0], numeric[1])

def _choose_obs_names(frame: pd.DataFrame, n_obs: int) -> list[str]:
    for column in ('label', 'Cell', 'cell', 'barcode', 'spot', 'spot_id'):
        if column in frame.columns and len(frame[column]) == n_obs:
            return frame[column].astype(str).tolist()
    return [f'spot_{idx}' for idx in range(n_obs)]

def convert(count_path: Path, gene_path: Path, location_path: Path, save_path: Path) -> None:
    counts = scipy.io.mmread(count_path)
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    else:
        counts = counts.tocsr()
    genes = _read_gene_names(gene_path)
    locations = _read_location_table(location_path)
    if counts.shape == (len(genes), len(locations)):
        matrix = counts.T.tocsr()
    elif counts.shape == (len(locations), len(genes)):
        matrix = counts.tocsr()
    else:
        raise ValueError(f'Count matrix shape does not match genes and locations: matrix={counts.shape}, genes={len(genes)}, locations={len(locations)}')
    obs_names = _choose_obs_names(locations, matrix.shape[0])
    x_col, y_col = _choose_location_columns(locations)
    spatial = locations[[x_col, y_col]].to_numpy(dtype=float)
    obs = locations.copy()
    obs.index = pd.Index(obs_names, name=None)
    var = pd.DataFrame(index=pd.Index(genes, name=None))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm['spatial'] = spatial
    adata.layers['counts'] = matrix.copy()
    adata.uns['recovered_external_simulator'] = {'count_path': str(count_path), 'gene_path': str(gene_path), 'location_path': str(location_path)}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(save_path, compression='gzip')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count_path', required=True, type=Path)
    parser.add_argument('--gene_path', required=True, type=Path)
    parser.add_argument('--location_path', required=True, type=Path)
    parser.add_argument('--save_adata_path', required=True, type=Path)
    args = parser.parse_args()
    convert(args.count_path, args.gene_path, args.location_path, args.save_adata_path)
if __name__ == '__main__':
    main()
