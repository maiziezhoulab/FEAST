"""Rerun scCube for Slide-seq with identity-preserving spatial transfer.

scCube's VAE is unconditional: its ``celltype_key`` is used only to decide how
many generated profiles to draw and to carry labels into the generated
metadata.  Giving every reference spot its own source-ID label therefore lets
us restore the generated profile to the exact source spot after scCube's
internal shuffle.  This avoids the non-converged dense EMD/argmax transfer that
created duplicate coordinates in the historical Slide-seq output.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
SOURCE_ID_KEY = '__sccube_source_reference_id__'

def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def generated_expression_by_cell(generated_data: object, generated_meta: pd.DataFrame, gene_names: pd.Index) -> pd.DataFrame:
    """Return generated expression as genes × generated-cell IDs."""
    if not isinstance(generated_data, pd.DataFrame):
        array = np.asarray(generated_data)
        if array.shape == (len(generated_meta), len(gene_names)):
            generated_data = pd.DataFrame(array.T, index=gene_names)
        elif array.shape == (len(gene_names), len(generated_meta)):
            generated_data = pd.DataFrame(array, index=gene_names)
        else:
            raise ValueError(f'Unexpected generated expression shape: {array.shape}')
    data = generated_data.copy()
    cells = generated_meta['Cell'].astype(str).tolist()
    genes = gene_names.astype(str).tolist()
    if data.shape == (len(genes), len(cells)):
        if data.columns.astype(str).tolist() != cells:
            raise ValueError('Generated expression columns are not in metadata Cell order')
        data.index = genes
        data.columns = cells
    elif data.shape == (len(cells), len(genes)):
        if data.index.astype(str).tolist() != cells:
            raise ValueError('Generated expression rows are not in metadata Cell order')
        data.index = cells
        data.columns = genes
        data = data.T
    else:
        raise ValueError(f'Unexpected generated expression shape: {data.shape}')
    return data

def run(args: argparse.Namespace) -> None:
    import scCube
    import torch
    if args.output_h5ad.exists() or args.provenance_json.exists():
        raise FileExistsError('Refusing to overwrite an existing rerun artifact')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this publication rerun')
    set_seed(args.seed)
    reference = ad.read_h5ad(args.input_h5ad)
    reference_ids = reference.obs_names.astype(str)
    gene_names = reference.var_names.astype(str)
    if not reference_ids.is_unique or not gene_names.is_unique:
        raise ValueError('Reference spot and gene identifiers must be unique')
    spatial = np.asarray(reference.obsm['spatial'], dtype=np.float64)
    if spatial.shape != (reference.n_obs, 2):
        raise ValueError(f'Expected 2-D spatial coordinates, got {spatial.shape}')
    if np.unique(spatial, axis=0).shape[0] != reference.n_obs:
        raise ValueError('Reference spatial coordinates are not one-to-one')
    if sp.issparse(reference.X):
        reference.X = reference.X.toarray()
    reference.obs['Cell'] = reference_ids.to_numpy()
    reference.obs[SOURCE_ID_KEY] = reference_ids.to_numpy()
    model = scCube.scCube()
    generated_meta, generated_data = model.train_vae_and_generate_cell(sc_adata=reference, cell_key='Cell', celltype_key=SOURCE_ID_KEY, batch_size=args.batch_size, epoch_num=args.epoch_num, used_device='cuda:0', save_model=False)
    if not isinstance(generated_meta, pd.DataFrame):
        generated_meta = pd.DataFrame(generated_meta)
    required = {'Cell', 'Cell_type'}
    if not required.issubset(generated_meta.columns):
        raise ValueError(f'Generated metadata lacks columns: {required - set(generated_meta)}')
    generated_meta = generated_meta.copy()
    generated_meta['Cell'] = generated_meta['Cell'].astype(str)
    generated_meta['Cell_type'] = generated_meta['Cell_type'].astype(str)
    if not generated_meta['Cell'].is_unique:
        raise ValueError('scCube generated cell IDs are not unique')
    if not generated_meta['Cell_type'].is_unique:
        raise ValueError('scCube did not preserve unique source IDs')
    if set(generated_meta['Cell_type']) != set(reference_ids):
        raise ValueError('Generated source-ID support does not equal reference spot support')
    data = generated_expression_by_cell(generated_data, generated_meta, gene_names)
    if not data.columns.is_unique or set(data.columns) != set(generated_meta['Cell']):
        raise ValueError('Generated expression columns do not match generated cell IDs')
    cell_for_source = generated_meta.set_index('Cell_type')['Cell']
    ordered_cells = cell_for_source.loc[reference_ids].tolist()
    ordered = data.loc[gene_names, ordered_cells].T.to_numpy(dtype=np.float32, copy=True)
    if ordered.shape != reference.shape:
        raise ValueError(f'Output shape {ordered.shape} differs from reference {reference.shape}')
    if not np.isfinite(ordered).all() or np.min(ordered) < 0:
        raise ValueError('Generated expression must be finite and nonnegative')
    obs = reference.obs.drop(columns=[SOURCE_ID_KEY]).copy()
    obs.index = reference_ids
    obs['sccube_generated_cell'] = cell_for_source.loc[reference_ids].to_numpy()
    obs['sccube_source_reference_id'] = reference_ids.to_numpy()
    output = ad.AnnData(X=sp.csr_matrix(ordered), obs=obs, var=reference.var.copy())
    output.obs_names = reference_ids
    output.var_names = gene_names
    output.obsm['spatial'] = spatial.copy()
    output.layers['counts'] = output.X.copy()
    identity_map = pd.DataFrame({'source_reference_id': reference_ids, 'sccube_generated_cell': cell_for_source.loc[reference_ids].to_numpy(), 'spatial_x': spatial[:, 0], 'spatial_y': spatial[:, 1]})
    args.identity_map_csv.parent.mkdir(parents=True, exist_ok=True)
    identity_map.to_csv(args.identity_map_csv, index=False)
    pip_freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True).stdout.splitlines()
    environment = {'python_executable': sys.executable, 'python_version': platform.python_version(), 'platform': platform.platform(), 'pip_freeze': pip_freeze}
    args.environment_json.parent.mkdir(parents=True, exist_ok=True)
    args.environment_json.write_text(json.dumps(environment, indent=2) + '\n')
    matrix = output.X.tocsr()
    library_sizes = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(library_sizes <= 0):
        raise ValueError('Generated output contains zero-library spots')
    nonzero_min = float(matrix.data.min()) if matrix.nnz else 0.0
    nonzero_max = float(matrix.data.max()) if matrix.nnz else 0.0
    script_path = Path(__file__).resolve()
    provenance = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'configuration_id': 'study00-sccube-slideseq001-identity-v1', 'method': 'scCube', 'method_version': package_version('scCube'), 'input_h5ad': str(args.input_h5ad.resolve()), 'input_artifact_id': 'study00-reference-Slideseq_001', 'input_shape': list(reference.shape), 'seed': args.seed, 'epoch_num': args.epoch_num, 'batch_size': args.batch_size, 'device': 'cuda:0', 'gpu': torch.cuda.get_device_name(0), 'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda, 'python_version': platform.python_version(), 'numpy_version': package_version('numpy'), 'scipy_version': package_version('scipy'), 'anndata_version': package_version('anndata'), 'scanpy_version': package_version('scanpy'), 'feast_version': args.feast_version, 'external_method_environment_supported_by_feast': False, 'external_method_environment_note': 'External scCube environment; not the supported FEAST validation environment.', 'runner': str(script_path), 'identity_map_csv': str(args.identity_map_csv.resolve()), 'environment_json': str(args.environment_json.resolve()), 'reference_transfer': 'unique_source_id_exact_coordinate_restore', 'cell_key_requested': 'ground_truth', 'cell_key_used': SOURCE_ID_KEY, 'source_label_count': int(reference.n_obs), 'generated_values': 'continuous nonnegative expression values', 'solver_diagnostics': {'dense_ot_called': False, 'generated_spots': int(output.n_obs), 'generated_genes': int(output.n_vars), 'unique_source_ids': int(output.obs['sccube_source_reference_id'].nunique()), 'unique_spatial_coordinates': int(np.unique(output.obsm['spatial'], axis=0).shape[0]), 'exact_reference_id_order': bool(output.obs_names.equals(reference_ids)), 'exact_reference_gene_order': bool(output.var_names.equals(gene_names)), 'exact_reference_spatial_order': bool(np.array_equal(output.obsm['spatial'], spatial))}, 'matrix_diagnostics': {'format': 'csr', 'dtype': str(matrix.dtype), 'nnz': int(matrix.nnz), 'nonzero_min': nonzero_min, 'nonzero_max': nonzero_max, 'library_min': float(library_sizes.min()), 'library_max': float(library_sizes.max())}}
    output.uns['publication_rerun'] = provenance
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output_h5ad.with_suffix(args.output_h5ad.suffix + '.partial')
    output.write_h5ad(partial, compression='gzip')
    os.replace(partial, args.output_h5ad)
    provenance['output_h5ad'] = str(args.output_h5ad.resolve())
    args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_json.write_text(json.dumps(provenance, indent=2) + '\n')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-h5ad', required=True, type=Path)
    parser.add_argument('--output-h5ad', required=True, type=Path)
    parser.add_argument('--provenance-json', required=True, type=Path)
    parser.add_argument('--identity-map-csv', required=True, type=Path)
    parser.add_argument('--environment-json', required=True, type=Path)
    parser.add_argument('--feast-repo', type=Path, default=Path(__file__).resolve().parents[2] / 'FEAST')
    parser.add_argument('--feast-version', default='1.0.2')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epoch-num', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=512)
    return parser.parse_args()
if __name__ == '__main__':
    run(parse_args())
