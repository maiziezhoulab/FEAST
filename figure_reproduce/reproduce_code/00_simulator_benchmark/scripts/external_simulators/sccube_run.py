"""Recovered scCube runner for Figure 2 single-slice simulation."""
from __future__ import annotations
import argparse
import inspect
import random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
DEFAULT_MAX_REFERENCE_DISTANCE_BYTES = 256 * 1024 ** 3
SPATIAL_BIN_COLUMN = '__sccube_spatial_bin__'

def _find_sccube_class() -> type[Any]:
    import scCube
    candidates = [getattr(scCube, 'scCube', None), getattr(scCube, 'ScCube', None), getattr(scCube, 'SCCube', None)]
    for candidate in candidates:
        if inspect.isclass(candidate):
            return candidate
    raise ImportError('Could not find a scCube model class in the scCube package.')

def _construct_model(model_class: type[Any], adata: Any, cell_key: str) -> Any:
    attempts = [{'adata': adata, 'celltype_key': cell_key}, {'adata': adata, 'cell_key': cell_key}, {'adata': adata, 'cluster_key': cell_key}, {'data': adata, 'celltype_key': cell_key}, {'data': adata, 'cell_key': cell_key}, {}]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            if kwargs:
                return model_class(**kwargs)
            return model_class(adata, cell_key)
        except TypeError as exc:
            last_error = exc
    raise TypeError(f'Could not construct scCube model: {last_error}')

def _call_with_compatible_kwargs(method: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(method)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters or any((param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()))}
    return method(**accepted)

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _dense_distance_bytes(n_left: int, n_right: int) -> int:
    return int(n_left) * int(n_right) * np.dtype(np.float64).itemsize

def _add_spatial_bin_labels(adata: Any, grid: int, column: str=SPATIAL_BIN_COLUMN) -> str:
    spatial = np.asarray(adata.obsm['spatial'])
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("spatial bin labels require two-dimensional obsm['spatial']")
    n_bins = max(2, min(int(grid), int(np.sqrt(max(adata.n_obs, 1)))))
    x_rank = pd.Series(spatial[:, 0], index=adata.obs_names).rank(method='first')
    y_rank = pd.Series(spatial[:, 1], index=adata.obs_names).rank(method='first')
    x_bin = pd.qcut(x_rank, q=n_bins, labels=False, duplicates='drop')
    y_bin = pd.qcut(y_rank, q=n_bins, labels=False, duplicates='drop')
    labels = x_bin.astype(str).str.zfill(2) + '_' + y_bin.astype(str).str.zfill(2)
    adata.obs[column] = labels.astype(str).to_numpy()
    return column

def _max_reference_block_bytes(adata: Any, generate_sc_meta: pd.DataFrame, cell_key: str) -> tuple[int, str]:
    real_counts = adata.obs[cell_key].astype(str).value_counts(dropna=False)
    if 'Cell_type' in generate_sc_meta.columns:
        generated_counts = generate_sc_meta['Cell_type'].astype(str).value_counts(dropna=False)
    else:
        generated_counts = pd.Series([len(generate_sc_meta)], index=[str(real_counts.index[0]) if len(real_counts) else 'all'])
    max_bytes = 0
    max_label = ''
    for label, real_n in real_counts.items():
        generated_n = int(generated_counts.get(str(label), 0))
        block_bytes = _dense_distance_bytes(int(real_n), generated_n)
        if block_bytes > max_bytes:
            max_bytes = block_bytes
            max_label = str(label)
    return (max_bytes, max_label)

def _ensure_cell_column(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    if 'Cell' not in meta.columns:
        meta['Cell'] = meta.index.astype(str)
    meta.index = meta['Cell'].astype(str)
    return meta

def _to_dataframe(data: Any, gene_names: list[str]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if sp.issparse(data):
        data = data.toarray()
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f'Generated expression must be 2-D, got {array.shape}')
    if array.shape[1] == len(gene_names):
        return pd.DataFrame(array, columns=gene_names)
    if array.shape[0] == len(gene_names):
        return pd.DataFrame(array.T, columns=gene_names)
    return pd.DataFrame(array)

def run_sccube(h5ad_file: Path, output_dir: Path, cell_key: str, *, seed: int, epoch_num: int, batch_size: int, max_reference_distance_bytes: int, spatial_bin_grid: int, oversized_reference_action: str) -> None:
    _set_seed(seed)
    adata_ref = sc.read_h5ad(h5ad_file)
    if cell_key not in adata_ref.obs:
        raise KeyError(f'{cell_key!r} is not present in adata.obs')
    if 'spatial' not in adata_ref.obsm:
        raise KeyError("'spatial' is not present in adata.obsm")
    label_strategy = 'input_label'
    cell_key_used = cell_key
    input_reference_bytes = _dense_distance_bytes(adata_ref.n_obs, adata_ref.n_obs)
    input_label_count = int(adata_ref.obs[cell_key].nunique(dropna=False))
    if input_label_count < 2 and input_reference_bytes > max_reference_distance_bytes and (spatial_bin_grid > 1):
        cell_key_used = _add_spatial_bin_labels(adata_ref, spatial_bin_grid)
        label_strategy = f'spatial_quantile_bins_{spatial_bin_grid}x{spatial_bin_grid}'
        print(f'scCube: input label has one level; using spatial bins for reference transfer ({cell_key_used}).')
    model_class = _find_sccube_class()
    model = model_class()
    if not hasattr(model, 'train_vae_and_generate_cell'):
        raise AttributeError('scCube model lacks train_vae_and_generate_cell')
    if not hasattr(model, 'generate_pattern_reference'):
        raise AttributeError('scCube model lacks generate_pattern_reference')
    if hasattr(adata_ref.X, 'toarray'):
        adata_ref.X = adata_ref.X.toarray()
    if 'Cell' not in adata_ref.obs.columns:
        adata_ref.obs['Cell'] = adata_ref.obs_names.astype(str)
    try:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'
    generate_sc_meta, generate_sc_data = _call_with_compatible_kwargs(model.train_vae_and_generate_cell, sc_adata=adata_ref, cell_key='Cell', celltype_key=cell_key_used, batch_size=batch_size, epoch_num=epoch_num, used_device=device, save_model=False)
    if 'x' not in adata_ref.obs.columns:
        adata_ref.obs['x'] = adata_ref.obsm['spatial'][:, 0]
    if 'y' not in adata_ref.obs.columns:
        adata_ref.obs['y'] = adata_ref.obsm['spatial'][:, 1]
    if not isinstance(generate_sc_meta, pd.DataFrame):
        generate_sc_meta = pd.DataFrame(generate_sc_meta)
    total_ot_bytes = _dense_distance_bytes(adata_ref.n_obs, generate_sc_meta.shape[0])
    max_block_bytes, max_block_label = _max_reference_block_bytes(adata_ref, generate_sc_meta, cell_key_used)
    reference_transfer = 'generate_pattern_reference'
    if total_ot_bytes > max_reference_distance_bytes:
        message = f'scCube reference transfer would allocate a dense OT distance matrix of about {total_ot_bytes / 1024 ** 3:.2f} GiB ({adata_ref.n_obs} reference × {generate_sc_meta.shape[0]} generated spots) — too large. Preserving reference coordinate order instead.'
        if oversized_reference_action == 'error':
            raise MemoryError(message)
        print(message)
        reference_transfer = 'preserve_reference_coordinate_order'
    elif max_block_bytes > max_reference_distance_bytes:
        message = f'scCube reference transfer would allocate a dense OT distance block of about {max_block_bytes / 1024 ** 3:.2f} GiB for label {max_block_label!r}.'
        if oversized_reference_action == 'error':
            raise MemoryError(message)
        if oversized_reference_action != 'preserve':
            raise ValueError(f"oversized_reference_action must be 'preserve' or 'error', got {oversized_reference_action!r}")
        print(f'{message} Preserving reference coordinate order instead.')
        reference_transfer = 'preserve_reference_coordinate_order'
    else:
        generate_sc_data, generate_sc_meta = _call_with_compatible_kwargs(model.generate_pattern_reference, sc_adata=adata_ref, generate_sc_data=generate_sc_data, generate_sc_meta=generate_sc_meta, celltype_key=cell_key_used, spatial_key=['x', 'y'])
    print(getattr(generate_sc_data, 'shape', None))
    print(getattr(generate_sc_meta, 'shape', None))
    generate_sc_meta = _ensure_cell_column(generate_sc_meta)
    generated = _to_dataframe(generate_sc_data, adata_ref.var_names.astype(str).tolist())
    if generated.shape[0] != generate_sc_meta.shape[0] and generated.shape[1] == generate_sc_meta.shape[0]:
        generated = generated.T
    generated.index = generate_sc_meta.index
    if generated.shape[1] == adata_ref.n_vars:
        generated.columns = adata_ref.var_names.astype(str)
    sim = sc.AnnData(X=sp.csr_matrix(generated.to_numpy()), obs=generate_sc_meta)
    sim.var_names = generated.columns.astype(str)
    if {'x', 'y'}.issubset(generate_sc_meta.columns):
        sim.obsm['spatial'] = generate_sc_meta[['x', 'y']].to_numpy(dtype=float)
    elif {'spatial_x', 'spatial_y'}.issubset(generate_sc_meta.columns):
        sim.obsm['spatial'] = generate_sc_meta[['spatial_x', 'spatial_y']].to_numpy(dtype=float)
    else:
        sim.obsm['spatial'] = np.asarray(adata_ref.obsm['spatial'][:sim.n_obs], dtype=float)
    sim.layers['counts'] = sim.X.copy()
    sim.uns['recovered_external_simulator'] = {'tool': 'scCube', 'input_h5ad': str(h5ad_file), 'cell_key_requested': cell_key, 'cell_key_used': cell_key_used, 'input_label_count': input_label_count, 'label_strategy': label_strategy, 'seed': seed, 'epoch_num': epoch_num, 'batch_size': batch_size, 'device': device, 'reference_transfer': reference_transfer, 'max_reference_distance_bytes': max_reference_distance_bytes, 'estimated_input_reference_distance_bytes': input_reference_bytes, 'estimated_max_reference_block_bytes': max_block_bytes, 'estimated_max_reference_block_label': max_block_label}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    sim.write_h5ad(output_dir, compression='gzip')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad_file', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--cell_key', default='CellType')
    parser.add_argument('--seed', default=2026, type=int)
    parser.add_argument('--epoch-num', default=10000, type=int)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--max-reference-distance-bytes', default=DEFAULT_MAX_REFERENCE_DISTANCE_BYTES, type=int, help='Maximum dense float64 OT distance block allowed for scCube generate_pattern_reference.')
    parser.add_argument('--spatial-bin-grid', default=16, type=int, help='When a large input has a one-level label column, use this quantile grid to create pseudo-labels before scCube reference transfer.')
    parser.add_argument('--oversized-reference-action', choices=('preserve', 'error'), default='preserve', help='Action if a reference-transfer OT block still exceeds the memory guard after optional spatial binning.')
    args = parser.parse_args()
    run_sccube(args.h5ad_file, args.output_dir, args.cell_key, seed=args.seed, epoch_num=args.epoch_num, batch_size=args.batch_size, max_reference_distance_bytes=args.max_reference_distance_bytes, spatial_bin_grid=args.spatial_bin_grid, oversized_reference_action=args.oversized_reference_action)
if __name__ == '__main__':
    main()
