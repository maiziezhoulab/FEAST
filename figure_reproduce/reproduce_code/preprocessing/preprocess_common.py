from __future__ import annotations
import argparse
import json
import logging
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[2] / 'Datasets' / 'Raw'
DEFAULT_PROCESSED_ROOT = Path(__file__).resolve().parents[2] / 'Datasets' / 'Processed'
DEFAULT_SEED = 2026

def add_common_arguments(parser: argparse.ArgumentParser, dataset_key: str) -> None:
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument('--processed-root', type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument('--dataset-key', default=dataset_key)
    parser.add_argument('--samples', nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--min-counts', type=float, default=1.0)
    parser.add_argument('--min-genes', type=int, default=1)
    parser.add_argument('--min-cells-per-gene', type=int, default=3)
    parser.add_argument('--max-pct-mt', type=float, default=None)
    parser.add_argument('--allow-missing-labels', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

def dataset_paths(processed_root: Path, dataset_key: str) -> dict[str, Path]:
    root = processed_root / dataset_key
    return {'root': root, 'h5ad': root / 'h5ad', 'manifest': root / 'manifest', 'logs': root / 'logs'}

def ensure_processed_dirs(processed_root: Path, dataset_key: str) -> dict[str, Path]:
    paths = dataset_paths(processed_root, dataset_key)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def setup_logging(log_dir: Path, dataset_key: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'preprocess_{dataset_key}_{stamp}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    return log_path

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def environment_summary() -> str:
    return json.dumps({'python': sys.version.split()[0], 'executable': sys.executable, 'platform': platform.platform(), 'conda_prefix': os.environ.get('CONDA_PREFIX', '')}, sort_keys=True)

def read_raw_manifest(raw_dataset_dir: Path) -> pd.DataFrame:
    manifest = raw_dataset_dir / 'manifest' / 'raw_urls.tsv'
    if not manifest.exists():
        raise FileNotFoundError(f'Missing raw manifest: {manifest}')
    return pd.read_csv(manifest, sep='\t')

def filter_requested_samples(sample_ids: Sequence[str], requested: Sequence[str] | None) -> list[str]:
    sample_ids = list(sample_ids)
    if not requested:
        return sample_ids
    requested_set = set(requested)
    missing = sorted(requested_set.difference(sample_ids))
    if missing:
        raise ValueError(f'Requested samples not found: {missing}')
    return [sid for sid in sample_ids if sid in requested_set]

def fail_if_output_exists(path: Path, overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        raise FileExistsError(f'Output exists; pass --overwrite to replace it: {path}')

def unique_index(values: Iterable[object], prefix: str) -> pd.Index:
    index = pd.Index([str(v) for v in values], dtype='object')
    if index.has_duplicates:
        counts: dict[str, int] = {}
        fixed: list[str] = []
        for value in index:
            counts[value] = counts.get(value, 0) + 1
            fixed.append(value if counts[value] == 1 else f'{value}-{counts[value]}')
        index = pd.Index(fixed, dtype='object')
    if index.empty:
        raise ValueError(f'{prefix} index is empty.')
    return index

def _data_values(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.data
    return np.asarray(matrix).ravel()

def is_integer_compatible(matrix, atol: float=1e-06) -> bool:
    values = _data_values(matrix)
    if values.size == 0:
        return True
    finite = np.isfinite(values)
    if not finite.all():
        return False
    return np.allclose(values, np.rint(values), atol=atol)

def to_csr_counts(matrix, *, dtype=np.int32):
    if not is_integer_compatible(matrix):
        raise ValueError('Count matrix is not integer-compatible.')
    if sparse.issparse(matrix):
        out = matrix.tocsr(copy=True)
        out.data = np.rint(out.data).astype(dtype, copy=False)
        out.eliminate_zeros()
        return out
    arr = np.rint(np.asarray(matrix)).astype(dtype, copy=False)
    return sparse.csr_matrix(arr)

def add_counts_layer(adata: ad.AnnData, counts, *, replace_x: bool=True) -> None:
    counts_csr = to_csr_counts(counts)
    if replace_x:
        adata.X = counts_csr
    adata.layers['counts'] = adata.X if replace_x else counts_csr

def add_var_metadata(adata: ad.AnnData, *, gene_ids: Sequence[object] | None=None, gene_symbols: Sequence[object] | None=None, feature_type: str='Gene Expression', species: str | None=None) -> None:
    if gene_ids is None:
        gene_ids = adata.var_names.astype(str)
    if gene_symbols is None:
        if 'gene_symbols' in adata.var:
            gene_symbols = adata.var['gene_symbols'].astype(str).to_numpy()
        elif 'gene_symbol' in adata.var:
            gene_symbols = adata.var['gene_symbol'].astype(str).to_numpy()
        else:
            gene_symbols = adata.var_names.astype(str)
    adata.var['gene_id'] = pd.Index(gene_ids, dtype='object').astype(str).to_numpy()
    adata.var['gene_symbol'] = pd.Index(gene_symbols, dtype='object').astype(str).to_numpy()
    adata.var['feature_type'] = feature_type
    if species:
        adata.var['species'] = species
    symbols = pd.Index(adata.var['gene_symbol'].astype(str))
    adata.var['mt'] = np.asarray(symbols.str.upper().str.startswith('MT-'), dtype=bool)
    adata.var_names = unique_index(adata.var['gene_symbol'].astype(str), 'gene')

def add_standard_obs(adata: ad.AnnData, *, dataset_key: str, sample_id: str, platform: str, species: str, tissue: str, condition: str='unknown', donor_id: str='unknown', replicate_id: str='unknown', section_id: str='unknown') -> None:
    obs = adata.obs
    obs['dataset_key'] = dataset_key
    obs['sample_id'] = sample_id
    obs['platform'] = platform
    obs['species'] = species
    obs['tissue'] = tissue
    obs['condition'] = condition
    obs['donor_id'] = donor_id
    obs['replicate_id'] = replicate_id
    obs['section_id'] = section_id

def set_labels(adata: ad.AnnData, *, region, cell_type, annotation, ground_truth, label_source: str) -> None:
    adata.obs['region'] = pd.Series(region, index=adata.obs_names).astype('category')
    adata.obs['cell_type'] = pd.Series(cell_type, index=adata.obs_names).astype('category')
    adata.obs['annotation'] = pd.Series(annotation, index=adata.obs_names).astype('category')
    adata.obs['ground_truth'] = pd.Series(ground_truth, index=adata.obs_names).astype('category')
    adata.obs['label_source'] = label_source

def set_spatial(adata: ad.AnnData, xy: np.ndarray, xyz: np.ndarray | None=None) -> None:
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (adata.n_obs, 2):
        raise ValueError(f'Expected spatial shape {(adata.n_obs, 2)}, found {xy.shape}')
    if not np.isfinite(xy).all():
        raise ValueError('Spatial coordinates contain non-finite values.')
    adata.obsm['spatial'] = xy
    adata.obs['x'] = xy[:, 0]
    adata.obs['y'] = xy[:, 1]
    if xyz is not None:
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (adata.n_obs, 3):
            raise ValueError(f'Expected spatial_3d shape {(adata.n_obs, 3)}, found {xyz.shape}')
        if not np.isfinite(xyz).all():
            raise ValueError('3D spatial coordinates contain non-finite values.')
        adata.obsm['spatial_3d'] = xyz
        adata.obs['z'] = xyz[:, 2]

def compute_qc_fields(adata: ad.AnnData, *, min_counts: float, min_genes: int, min_cells_per_gene: int, max_pct_mt: float | None) -> tuple[np.ndarray, np.ndarray]:
    X = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    if sparse.issparse(X):
        total_counts = np.asarray(X.sum(axis=1)).ravel()
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
        n_cells_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        arr = np.asarray(X)
        total_counts = arr.sum(axis=1)
        n_genes = (arr > 0).sum(axis=1)
        n_cells_gene = (arr > 0).sum(axis=0)
    adata.obs['total_counts'] = total_counts
    adata.obs['n_genes_by_counts'] = n_genes
    if 'mt' in adata.var and bool(adata.var['mt'].any()):
        mt_mask = adata.var['mt'].to_numpy(dtype=bool)
        mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel() if sparse.issparse(X) else np.asarray(X[:, mt_mask]).sum(axis=1)
        adata.obs['pct_counts_mt'] = np.divide(mt_counts, total_counts, out=np.zeros_like(total_counts, dtype=float), where=total_counts > 0) * 100.0
    elif 'pct_counts_mt' not in adata.obs:
        adata.obs['pct_counts_mt'] = np.nan
    obs_keep = (total_counts >= min_counts) & (n_genes >= min_genes)
    if max_pct_mt is not None and 'pct_counts_mt' in adata.obs:
        pct_mt = pd.to_numeric(adata.obs['pct_counts_mt'], errors='coerce').to_numpy()
        obs_keep &= np.isnan(pct_mt) | (pct_mt <= max_pct_mt)
    var_keep = n_cells_gene >= min_cells_per_gene
    adata.obs['qc_pass'] = obs_keep
    return (obs_keep, var_keep)

def apply_qc(adata: ad.AnnData, *, min_counts: float, min_genes: int, min_cells_per_gene: int, max_pct_mt: float | None) -> tuple[ad.AnnData, dict[str, int]]:
    n_obs_before, n_vars_before = (adata.n_obs, adata.n_vars)
    obs_keep, var_keep = compute_qc_fields(adata, min_counts=min_counts, min_genes=min_genes, min_cells_per_gene=min_cells_per_gene, max_pct_mt=max_pct_mt)
    filtered = adata[obs_keep, var_keep].copy()
    stats = {'n_obs_before_qc': int(n_obs_before), 'n_vars_before_qc': int(n_vars_before), 'n_obs': int(filtered.n_obs), 'n_vars': int(filtered.n_vars)}
    return (filtered, stats)

def preprocessing_uns(*, dataset_key: str, sample_id: str, input_files: Sequence[Path], counts_source: str, coordinate_source: str, label_source: str, seed: int, script_path: Path, extra: dict | None=None) -> dict:
    payload = {'dataset_key': dataset_key, 'sample_id': sample_id, 'input_files': [str(p) for p in input_files], 'counts_source': counts_source, 'coordinate_source': coordinate_source, 'label_source': label_source, 'seed': int(seed), 'created_at': now_iso(), 'script': str(script_path), 'environment': json.loads(environment_summary())}
    if extra:
        payload.update(extra)
    return payload

def validate_processed_adata(adata: ad.AnnData) -> None:
    required_obs = ['ground_truth', 'region', 'cell_type', 'label_source']
    for key in required_obs:
        if key not in adata.obs:
            raise ValueError(f'Missing obs[{key!r}]')
    if 'spatial' not in adata.obsm:
        raise ValueError("Missing obsm['spatial']")
    if 'counts' not in adata.layers:
        raise ValueError("Missing layers['counts']")
    if adata.X.shape != adata.layers['counts'].shape:
        raise ValueError("X and layers['counts'] shape mismatch")
    if adata.obsm['spatial'].shape != (adata.n_obs, 2):
        raise ValueError("obsm['spatial'] has wrong shape")
    if adata.obs_names.has_duplicates:
        raise ValueError('Duplicated obs_names')
    if adata.var_names.has_duplicates:
        raise ValueError('Duplicated var_names')
    if not is_integer_compatible(adata.layers['counts']):
        raise ValueError("layers['counts'] is not integer-compatible")
    values = _data_values(adata.layers['counts'])
    if values.size and np.nanmin(values) < 0:
        raise ValueError("layers['counts'] contains negative values")

def write_processed(adata: ad.AnnData, output_path: Path, *, overwrite: bool, compression: str='gzip') -> None:
    fail_if_output_exists(output_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_processed_adata(adata)
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    if tmp_path.exists():
        tmp_path.unlink()
    adata.write_h5ad(tmp_path, compression=compression)
    tmp_path.replace(output_path)

def append_manifest_rows(manifest_dir: Path, rows: list[dict], filename: str='processed_h5ad.tsv') -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / filename
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, sep='\t')
        df = pd.concat([old, df], ignore_index=True)
        if {'dataset_key', 'sample_id', 'output_h5ad'}.issubset(df.columns):
            df = df.drop_duplicates(['dataset_key', 'sample_id', 'output_h5ad'], keep='last')
    df.to_csv(path, sep='\t', index=False)
    return path

def manifest_row(*, dataset_key: str, sample_id: str, input_files: Sequence[Path], output_h5ad: Path, stats: dict[str, int], label_source: str, coordinate_source: str, counts_source: str, seed: int, script_path: Path) -> dict:
    return {'dataset_key': dataset_key, 'sample_id': sample_id, 'input_files': ';'.join((str(p) for p in input_files)), 'output_h5ad': str(output_h5ad), 'n_obs': stats.get('n_obs'), 'n_vars': stats.get('n_vars'), 'n_obs_before_qc': stats.get('n_obs_before_qc'), 'n_vars_before_qc': stats.get('n_vars_before_qc'), 'label_source': label_source, 'coordinate_source': coordinate_source, 'counts_source': counts_source, 'seed': int(seed), 'created_at': now_iso(), 'script': str(script_path), 'environment': environment_summary()}

def copy_or_link_raw_reference(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        try:
            os.symlink(src, dest)
        except OSError:
            shutil.copy2(src, dest)
    return dest

def print_inventory(items: Sequence[dict]) -> None:
    print(json.dumps(list(items), indent=2, sort_keys=True))
