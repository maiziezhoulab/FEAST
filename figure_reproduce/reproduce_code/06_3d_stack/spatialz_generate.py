"""Generate native SpatialZ slices from the existing Study 06 reference splits.

Run with the SpatialZ Python 3.9 environment. Only the two reference H5ADs
are opened; target z and seed come from the existing plan.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
import anndata as ad
import numpy as np
import scipy.sparse as sp
import yaml
from run import density_for_gap, input_path, select_targets
from workflow import load_config, read_json, write_json

def load_study(feast_root):
    config = load_config(feast_root / 'run_config.yaml')
    return (config, read_json(feast_root / 'plan.json'))

def target_path(output_root, row, prefix):
    return output_root / 'targets' / row['density_name'] / f"{prefix}.{int(row['target_slice']):03d}"

def reference_data(path, dataset, scale):
    source = ad.read_h5ad(path)
    matrix = source.layers[dataset['counts_layer']]
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'{path}: reference counts must be finite and nonnegative')
    reference = ad.AnnData(X=values.astype(np.float64), obs=source.obs[[dataset['label_key']]].copy(), var=source.var.copy())
    xy = np.asarray(source.obsm[dataset['spatial_key']], dtype=np.float64)
    if xy.shape != (source.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError(f'{path}: expected finite XY coordinates')
    reference.obsm['spatial'] = xy * scale
    if reference.obs[dataset['label_key']].isna().any():
        raise ValueError(f'{path}: reference class labels are missing')
    reference.obs[dataset['label_key']] = reference.obs[dataset['label_key']].astype(str).astype('category')
    return reference

@contextmanager
def bounded_mender_workers(workers):
    """Use upstream MENDER's ordered map and concatenation with a managed pool.

SpatialZ requests 200 processes for three slices. This changes only the
process count/lifetime, and restores the installed method after the call.
    """
    import MENDER
    from multiprocessing import Pool
    original = MENDER.MENDER.run_representation_mp

    def representation(self, mp=200, group_norm=False):
        self.group_norm = group_norm
        with Pool(processes=workers) as pool:
            results = pool.map(self.mp_helper, np.arange(len(self.batch_list)))
        self.adata_MENDER = results[0].concatenate(results[1:])
        self.adata_MENDER_dump = self.adata_MENDER.copy()
    MENDER.MENDER.run_representation_mp = representation
    try:
        yield
    finally:
        MENDER.MENDER.run_representation_mp = original

def generate(row, density, plan, study, recipe, device, workers):
    import SpatialZ
    dataset = study['dataset']
    lower_id, upper_id = (int(row['lower_ref_slice']), int(row['upper_ref_slice']))
    pool = set(map(int, density['reference_pool']))
    if not {lower_id, upper_id} <= pool or int(row['target_slice']) in pool:
        raise ValueError('SpatialZ inputs must be retained references bracketing a held-out target')
    z0, z1, z = (float(row[key]) for key in ('z0', 'z1', 'target_z'))
    if not z0 < z < z1:
        raise ValueError('target z must lie strictly between its references')
    alpha = (z1 - z) / (z1 - z0)
    scale = float(recipe['coordinates']['input_to_working_scale'])
    lower = reference_data(input_path(plan, study, lower_id), dataset, scale)
    upper = reference_data(input_path(plan, study, upper_id), dataset, scale)
    if not lower.var_names.equals(upper.var_names) or lower.n_vars != int(dataset['expected_genes']):
        raise ValueError('reference gene panels/order must match the full Study 06 panel')
    settings = dict(recipe['generation'])
    expected_cells = int((alpha * lower.n_obs + (1 - alpha) * upper.n_obs) * settings['n_mag'])
    if settings['n_cell'] is not None:
        raise ValueError('this benchmark infers cell count from reference slices')
    np.random.seed(int(row['seed']))
    with bounded_mender_workers(workers):
        generated = SpatialZ.Generate_spatialz(lower, upper, adata1_id=f'ref{lower_id}', adata2_id=f'ref{upper_id}', alpha=alpha, device=device, seed=int(row['seed']), cell_type_key=dataset['label_key'], verbose=True, **settings)
    values = np.asarray(generated.X)
    xy = np.asarray(generated.obsm['spatial'], dtype=np.float64) / scale
    if generated.n_obs != expected_cells or not generated.var_names.equals(lower.var_names):
        raise ValueError('SpatialZ returned an unexpected cell count or gene panel')
    if not np.isfinite(values).all() or (values < 0).any() or (not np.isfinite(xy).all()):
        raise ValueError('SpatialZ returned nonfinite or negative expression, or invalid coordinates')
    label_key = dataset['label_key']
    reference_labels = set(lower.obs[label_key].astype(str)) | set(upper.obs[label_key].astype(str))
    if generated.obs[label_key].isna().any() or not set(generated.obs[label_key].astype(str)) <= reference_labels:
        raise ValueError('SpatialZ returned labels unsupported by the references')
    generated.obs_names = [f"spatialz_{row['density_name']}_{int(row['target_slice']):03d}_{i}" for i in range(generated.n_obs)]
    generated.obs['z'] = z
    generated.obs['slice_id'] = int(row['target_slice'])
    generated.obsm['spatial'] = xy
    generated.obsm['spatial_3d'] = np.column_stack((xy, np.full(generated.n_obs, z)))
    generated.layers['counts'] = sp.csr_matrix(values)
    generated.X = generated.layers['counts'].copy()
    return (generated, {'alpha': alpha, 'lower_reference_cells': lower.n_obs, 'upper_reference_cells': upper.n_obs, 'n_generated_cells': generated.n_obs, 'n_genes': generated.n_vars, 'all_zero_cells': int(np.sum(values.sum(axis=1) == 0))})

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feast-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config_spatialz.yaml'))
    parser.add_argument('--gap', type=int, choices=(3, 5, 10), required=True)
    parser.add_argument('--target-id', type=int, nargs='+')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--device', default='cuda:0', help='Use cpu explicitly on a CPU host')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--mender-workers', type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    import SpatialZ
    import torch
    from threadpoolctl import threadpool_limits
    if args.threads < 1 or args.mender_workers < 1:
        raise ValueError('thread and process counts must be positive')
    if args.device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is unavailable; use a GPU host or explicitly select --device cpu')
    torch.set_num_threads(args.threads)
    study, plan = load_study(args.feast_root)
    recipe = yaml.safe_load(args.config.read_text())
    _, density = density_for_gap(plan, args.gap)
    rows = select_targets(density['targets'], args.target_id, args.shard_index, args.shard_count)
    source = Path(SpatialZ.__file__).resolve()
    for row in rows:
        destination = target_path(args.output_root, row, study['dataset']['filename_prefix'])
        destination.mkdir(parents=True, exist_ok=False)
        record = {'method': 'spatialz', 'status': 'running', 'recipe': recipe, 'entry_point': str(Path(__file__).resolve()), 'feast_root': str(args.feast_root.resolve()), 'input_data_dir': plan['data_dir'], 'target': {key: row[key] for key in ('density_name', 'gap', 'target_index', 'target_slice', 'target_z', 'lower_ref_slice', 'upper_ref_slice', 'z0', 'z1', 'seed')}, 'source': str(source), 'python': platform.python_version(), 'torch': torch.__version__, 'device': args.device, 'threads': args.threads, 'mender_workers': args.mender_workers, 'target_fields_used': ['target_z'], 'target_h5ad_opened': False, 'started_utc': datetime.now(timezone.utc).isoformat()}
        write_json(destination / 'run.json', record)
        start = time.monotonic()
        try:
            with threadpool_limits(limits=args.threads):
                generated, info = generate(row, density, plan, study, recipe, args.device, args.mender_workers)
            generated.uns['spatialz_benchmark'] = json.dumps({**record, **info, 'status': 'completed'}, sort_keys=True)
            generated.write_h5ad(destination / 'generated.h5ad', compression='gzip')
            record.update(info, status='completed', elapsed_seconds=time.monotonic() - start)
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - start)
            write_json(destination / 'run.json', record)
            raise
        write_json(destination / 'run.json', record)
        print(f"Completed {destination}: {info['n_generated_cells']} cells", flush=True)
if __name__ == '__main__':
    main()
