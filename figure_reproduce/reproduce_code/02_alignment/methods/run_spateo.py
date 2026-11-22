"""Run one Spateo alignment job and record validated method diagnostics."""
from __future__ import annotations
import argparse
import json
import platform
import random
import time
import traceback
from pathlib import Path
import numpy as np
import scanpy as sc
import yaml

def fix_uns_keys(value):
    if isinstance(value, dict):
        return {str(key): fix_uns_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [fix_uns_keys(item) for item in value]
    return value

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--moving', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--configuration-id', required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        import spateo as st
        import torch
        config = yaml.safe_load(args.config.read_text())['spateo']
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested for Spateo but is not available')
        torch.cuda.manual_seed_all(args.seed)
        reference = sc.read_h5ad(args.reference)
        moving = sc.read_h5ad(args.moving)
        if not reference.obs_names.is_unique or not moving.obs_names.is_unique:
            raise RuntimeError('spot identifiers must be unique')
        moving_names = moving.obs_names.copy()
        moving_vars = moving.var_names.copy()
        moving_coords = np.asarray(moving.obsm['spatial'], dtype=np.float64).copy()
        if not np.isfinite(moving_coords).all():
            raise RuntimeError('moving coordinates are non-finite')

        def preprocess(adata):
            adata.var_names_make_unique()
            sc.pp.filter_genes(adata, min_cells=int(config['min_cells']))
            adata.layers['counts'] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        preprocess(reference)
        preprocess(moving)
        rep = str(config['representation'])
        st.align.group_pca([reference, moving], pca_key=rep)
        aligned, mappings = st.align.morpho_align(models=[reference, moving], rep_layer=rep, rep_field='obsm', dissimilarity=str(config['dissimilarity']), verbose=False, spatial_key='spatial', key_added='spatial_aligned', device=str(config['device']), max_iter=int(config['max_iter']), SVI_mode=bool(config['svi_mode']), iter_key_added=None, return_mapping=True)
        aligned_moving = aligned[1]
        mapping = np.asarray(mappings[0], dtype=np.float64)
        aligned_coords = np.asarray(aligned_moving.obsm['spatial_aligned'], dtype=np.float64)
        if not aligned_moving.obs_names.equals(moving_names):
            raise RuntimeError('Spateo changed moving spot support/order')
        if mapping.shape != (reference.n_obs, moving.n_obs):
            raise RuntimeError(f'unexpected mapping shape: {mapping.shape}')
        negative_tolerance = float(config['mapping_negative_tolerance'])
        mapping_mass = float(mapping.sum())
        if not np.isfinite(mapping).all() or not np.isfinite(aligned_coords).all():
            raise RuntimeError('Spateo produced non-finite output')
        if float(mapping.min()) < -negative_tolerance or mapping_mass <= 0:
            raise RuntimeError('Spateo produced a negative or zero-mass mapping')
        output_adata = sc.read_h5ad(args.moving)
        if not output_adata.obs_names.equals(moving_names):
            raise RuntimeError('moving input changed while the method was running')
        if not output_adata.var_names.equals(moving_vars):
            raise RuntimeError('moving gene order changed while the method was running')
        output_adata.obsm['spatial_aligned'] = aligned_coords
        output_adata.uns = fix_uns_keys(output_adata.uns)
        aligned_path = args.output_dir / 'aligned.h5ad'
        mapping_path = args.output_dir / 'mapping_matrix.npy'
        coordinates_path = args.output_dir / 'aligned_coordinates.csv'
        output_adata.write_h5ad(aligned_path, compression='gzip')
        np.save(mapping_path, mapping)
        import pandas as pd
        pd.DataFrame({'spot_barcode': moving_names.astype(str), 'x_moving': moving_coords[:, 0], 'y_moving': moving_coords[:, 1], 'x_aligned': aligned_coords[:, 0], 'y_aligned': aligned_coords[:, 1]}).to_csv(coordinates_path, index=False)
        diagnostics = {'status': 'completed', 'method': 'spateo', 'spateo_version': str(getattr(st, '__version__', 'unknown')), 'python_version': platform.python_version(), 'numpy_version': np.__version__, 'torch_version': torch.__version__, 'cuda_available': True, 'cuda_device_name': torch.cuda.get_device_name(0), 'device_requested': str(config['device']), 'feast_imported_in_method_worker': False, 'feast_numpy_support_range': '>=1.24,<2', 'numpy_supported_by_feast': bool(np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.24.0') and np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0')), 'external_environment_limitation': 'This external Spateo worker uses a NumPy version unsupported by FEAST; FEAST is not imported or executed in this process.' if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('2.0.0') else None, 'mapping_shape': list(mapping.shape), 'mapping_all_finite': True, 'mapping_minimum': float(mapping.min()), 'mapping_maximum': float(mapping.max()), 'mapping_mass': mapping_mass, 'mapping_negative_tolerance': negative_tolerance, 'spot_support_preserved': True, 'gene_order_preserved': True, 'rng_policy': 'Python, NumPy, Torch, and all CUDA generators seeded from public_seed', 'iteration_schedule': {'kind': 'fixed', 'max_iter': int(config['max_iter']), 'svi_mode': bool(config['svi_mode']), 'completed_return': True}, 'convergence_claim': False, 'convergence_disposition': 'Spateo morpho_align exposes a fixed iteration schedule without a positive convergence flag; this artifact is evaluated by identity, geometry, mapping, and article-metric validation only.'}
        (args.output_dir / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        transform = {'configuration_id': args.configuration_id, 'method': 'spateo', 'status': 'ok', 'public_seed': args.seed, 'runtime_seconds': round(time.monotonic() - started, 3), 'parameters': config, 'feast': {'version': output_adata.uns['feast_reproduce']['feast_version']}, 'input_artifacts': {}, 'source_artifacts': {}, 'output_artifacts': {path.name: None for path in (aligned_path, mapping_path, coordinates_path)}, 'solver_diagnostics': diagnostics}
        (args.output_dir / 'transform.json').write_text(json.dumps(transform, indent=2) + '\n')
        return 0
    except Exception as exc:
        (args.output_dir / 'FAILED.json').write_text(json.dumps({'status': 'failed_noncanonical', 'configuration_id': args.configuration_id, 'error': repr(exc), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
