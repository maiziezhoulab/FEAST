"""Run one Bioconductor spacexr/RCTD full-mode deconvolution job."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml

def integer_counts(matrix, label: str) -> np.ndarray:
    values = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or values.min(initial=0) < 0:
        raise RuntimeError(f'{label} has invalid count values')
    if np.max(np.abs(values - np.rint(values)), initial=0) > 1e-06:
        raise RuntimeError(f'{label} is not raw integer count data')
    return np.rint(values).astype(np.int64)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell-type-key', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--rscript', type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.output.with_name(args.output.stem + '_metadata.json')
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = yaml.safe_load(args.config.read_text())['rctd']
        spatial = ad.read_h5ad(args.input)
        reference = ad.read_h5ad(args.reference)
        if args.cell_type_key not in reference.obs:
            raise RuntimeError(f'reference lacks {args.cell_type_key!r}')
        if not spatial.var_names.equals(reference.var_names):
            raise RuntimeError('simulation/reference gene identity or order differs')
        spatial_counts = integer_counts(spatial.X, 'spatial simulation')
        reference_counts = integer_counts(reference.X, 'reference')
        full_spots = pd.Index([f'spot_{index}' for index in range(spatial.n_obs)])
        positive_spots = full_spots[spatial_counts.sum(axis=1) > 0]
        labels = reference.obs[args.cell_type_key].astype(str)
        label_counts = labels.value_counts()
        retained_types = label_counts[label_counts >= int(config['min_cells_per_type'])].index
        keep = labels.isin(retained_types).to_numpy()
        if retained_types.empty or keep.sum() == 0:
            raise RuntimeError('no reference cell types pass the declared filter')
        retained_labels = labels[keep]
        original_types = list(pd.unique(retained_labels))
        type_to_code = {name: f'ct_{index:04d}' for index, name in enumerate(original_types)}
        code_to_type = {code: name for name, code in type_to_code.items()}
        with tempfile.TemporaryDirectory(prefix='rctd_pub_') as temporary:
            work = Path(temporary)
            pd.DataFrame(spatial_counts.T, index=spatial.var_names, columns=full_spots).to_csv(work / 'counts.csv')
            coordinates = np.asarray(spatial.obsm['spatial'], dtype=np.float64)
            pd.DataFrame({'x': coordinates[:, 0], 'y': coordinates[:, 1], 'nUMI': spatial_counts.sum(axis=1)}, index=full_spots).to_csv(work / 'coordinates.csv')
            ref_columns = [f'ref_{index}' for index in range(int(keep.sum()))]
            pd.DataFrame(reference_counts[keep].T, index=reference.var_names, columns=ref_columns).to_csv(work / 'reference.csv')
            pd.Series([type_to_code[value] for value in retained_labels], index=ref_columns, name='cell_type').to_csv(work / 'cell_types.csv')
            stdout_path = args.output.with_name(args.output.stem + '_stdout.log')
            stderr_path = args.output.with_name(args.output.stem + '_stderr.log')
            command = [str(args.rscript), str(Path(__file__).with_name('run_rctd_backend.R')), '--counts', str(work / 'counts.csv'), '--coords', str(work / 'coordinates.csv'), '--reference', str(work / 'reference.csv'), '--cell-types', str(work / 'cell_types.csv'), '--output', str(args.output), '--cache-dir', str(config['cache_dir']), '--max-cores', str(config['max_cores']), '--seed', str(args.seed)]
            cache_dir = str(config['cache_dir'])
            environment = {**os.environ, 'RCTD_CACHE_DIR': cache_dir, 'XDG_CACHE_HOME': cache_dir, 'R_USER_CACHE_DIR': cache_dir}
            environment.setdefault('R_LIBS_USER', str(Path(cache_dir) / 'R_libs'))
            completed = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=int(config['timeout_seconds']))
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f'RCTD backend exited {completed.returncode}')
        prediction = pd.read_csv(args.output, index_col=0)
        prediction.index = prediction.index.astype(str)
        prediction.rename(columns=code_to_type, inplace=True)
        if not prediction.index.equals(positive_spots):
            raise RuntimeError('RCTD output does not have exact positive-library spot order')
        if not prediction.columns.is_unique or set(prediction.columns) != set(original_types):
            raise RuntimeError('RCTD output cell-type support is invalid')
        values = prediction.to_numpy(np.float64)
        if not np.isfinite(values).all() or values.min(initial=0) < -1e-06:
            raise RuntimeError('RCTD output contains invalid weights')
        if np.max(np.abs(values.sum(axis=1) - 1), initial=0) > 1e-05:
            raise RuntimeError('RCTD output rows do not sum to one')
        prediction.to_csv(args.output)
        runtime_versions = {}
        for line in completed.stdout.splitlines():
            if line.startswith('runtime::') and '=' in line:
                name, value = line.removeprefix('runtime::').split('=', 1)
                runtime_versions[name] = value
        if not {'R', 'spacexr', 'SummarizedExperiment', 'SpatialExperiment'}.issubset(runtime_versions):
            raise RuntimeError('RCTD backend did not report its runtime versions')
        metadata = {'status': 'validated_success', 'method': 'rctd', 'mode': str(config['mode']), 'public_seed': args.seed, 'input_path': str(args.input.resolve()), 'reference_path': str(args.reference.resolve()), 'output_path': str(args.output.resolve()), 'stdout_path': str(stdout_path.resolve()), 'stderr_path': str(stderr_path.resolve()), 'n_input_spots': spatial.n_obs, 'n_output_spots': len(prediction), 'n_zero_library_spots_excluded': int(spatial.n_obs - len(prediction)), 'n_retained_cell_types': len(original_types), 'min_cells_per_type': int(config['min_cells_per_type']), 'rng_policy': 'R set.seed(public_seed) before createRctd/runRctd', 'environment': runtime_versions, 'solver_diagnostics': {'backend_return_code': completed.returncode, 'weights_all_finite': True, 'weights_nonnegative_with_tolerance': True, 'weights_rows_normalized': True, 'mode': str(config['mode']), 'convergence_claim': False, 'disposition': 'spacexr returned a completed full-mode weights assay; the API does not expose a separate positive convergence flag'}}
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        return 0
    except Exception as exc:
        metadata_path.write_text(json.dumps({'status': 'failed_noncanonical', 'method': 'rctd', 'error': repr(exc), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
