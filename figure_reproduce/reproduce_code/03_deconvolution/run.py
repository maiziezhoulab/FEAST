"""Generate declared FEAST inputs and run selected deconvolution jobs."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from manifests import read_manifest, write_manifest
import yaml
from FEAST import __version__ as feast_version
from FEAST import simulate
from FEAST.deconvolution import aggregate_gene_expression, assign_original_spots_to_grid, calculate_cell_type_proportions_for_lowres, create_low_resolution_grid, filter_grid_to_tissue_shape
DEFAULT_CONFIGURATION_ID = 'study03-fresh-same-expression-v3'

def dense_integer(matrix, label: str) -> np.ndarray:
    values = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or values.min(initial=0) < 0:
        raise RuntimeError(f'{label} contains negative or non-finite values')
    if np.max(np.abs(values - np.rint(values)), initial=0) > 1e-06:
        raise RuntimeError(f'{label} is not raw integer count data')
    return np.rint(values).astype(np.int64)

def simulate_pair(reference: ad.AnnData, resolution: float, config: dict, configuration_id: str) -> tuple[ad.AnnData, ad.AnnData, np.ndarray]:
    settings = config['simulation']
    high_resolution = simulate(reference=reference, seed=int(config['seed']), parameter_mode=str(settings['parameter_mode']), spatial_mode=str(settings['spatial_mode']), assignment_solver=str(settings['assignment_solver']), assignment_blocks=bool(settings['assignment_blocks']), clip_overshoot_factor=float(settings['clip_overshoot_factor']), ppf_method=str(settings['ppf_method']), beta_n_jobs=int(settings['beta_n_jobs']), beta_early_stopping_patience=int(settings['beta_early_stopping_patience']), convert_n_jobs=int(settings['convert_n_jobs']), verbose=True)
    diagnostics = high_resolution.uns.get('simulation_diagnostics', {})
    assignment_diagnostics = diagnostics.get('copula_rank_diagnostics', {})
    if diagnostics.get('spatial_mode') != 'reference_rank':
        raise RuntimeError('FEAST returned the wrong spatial mode')
    if assignment_diagnostics.get('assignment_blocks') is not False:
        raise RuntimeError('FEAST did not use exact global assignment')
    expression = dense_integer(high_resolution.X, 'FEAST high-resolution simulation')
    coordinates = np.asarray(high_resolution.obsm['spatial'], dtype=np.float64)
    grid = create_low_resolution_grid(coordinates, downsampling_factor=resolution, grid_type=str(settings['grid_type']))
    grid = filter_grid_to_tissue_shape(coordinates, grid, alpha=float(settings['alpha']))
    assignments = assign_original_spots_to_grid(coordinates, grid)
    assignment_indices = np.asarray(assignments, dtype=np.int64)
    if assignment_indices.shape != (high_resolution.n_obs,) or assignment_indices.min(initial=0) < 0 or assignment_indices.max(initial=-1) >= len(grid):
        raise RuntimeError('invalid high-resolution-to-grid assignment vector')
    cells_per_location = np.bincount(assignment_indices, minlength=len(grid)).astype(np.int64)
    if int(cells_per_location.sum()) != high_resolution.n_obs:
        raise RuntimeError('aggregated cell-count accounting is incomplete')
    aggregated = aggregate_gene_expression(expression, assignments, len(grid))
    aggregated = dense_integer(aggregated, 'aggregated FEAST simulation')
    if not np.array_equal(aggregated.sum(axis=0), expression.sum(axis=0)):
        raise RuntimeError('hexagonal aggregation changed per-gene totals')
    proportions = calculate_cell_type_proportions_for_lowres(high_resolution.obs[str(config['cell_type_key'])].astype(str).to_numpy(), assignments, len(grid))
    result = ad.AnnData(X=aggregated, var=high_resolution.var.copy())
    result.obs_names = pd.Index([str(index) for index in range(result.n_obs)])
    result.obs['n_source_cells'] = cells_per_location
    result.obsm['spatial'] = grid
    result.obsm['cell_type_proportions'] = proportions.to_numpy(np.float64)
    result.uns['cell_type_names'] = proportions.columns.astype(str).to_numpy()
    result.uns['feast_reproduce'] = {'configuration_id': configuration_id, 'feast_version': str(feast_version), 'public_seed': int(config['seed']), 'parameter_mode': str(settings['parameter_mode']), 'spatial_mode': str(settings['spatial_mode']), 'assignment_solver': str(settings['assignment_solver']), 'assignment_blocks': bool(settings['assignment_blocks']), 'clip_overshoot_factor': float(settings['clip_overshoot_factor']), 'downsampling_factor': float(resolution), 'grid_type': str(settings['grid_type']), 'alpha': float(settings['alpha']), 'solver_diagnostics': {'status': 'completed', 'assignment_method': assignment_diagnostics.get('assignment_method'), 'assignment_blocks': False, 'assignment_solver_requested': 'scipy', 'model_selection_counts': diagnostics.get('model_selection_counts', {}), 'aggregated_source_cell_count': int(cells_per_location.sum()), 'mean_source_cells_per_location': float(cells_per_location.mean())}}
    return result, high_resolution, assignment_indices

def run_method(command: list[str], log_path: Path, cwd: Path | None = None) -> subprocess.CompletedProcess:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', buffering=1) as log:
        log.write(shlex.join(command) + '\n')
        completed = subprocess.run(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONUNBUFFERED': '1'},
        )
        log.write(f'\n--- RUNTIME_SECONDS ---\n{time.monotonic() - started:.3f}\n')
    return completed

def record_path(record: dict, key: str) -> Path | None:
    value = record.get(key)
    return Path(value) if isinstance(value, str) and value else None

def simulation_record_is_valid(record: dict, public_seed: int, orchestration_runner: Path, configuration_id: str) -> bool:
    simulation = record_path(record, 'simulation_path')
    truth = record_path(record, 'truth_path')
    cells = record_path(record, 'cell_level_path')
    assignments = record_path(record, 'cell_assignment_path')
    if cells is None or assignments is None or not cells.is_file() or not assignments.is_file():
        return False
    if not (record.get('status') == 'ok' and record.get('configuration_id') == configuration_id and (record.get('feast_version') == str(feast_version)) and (record.get('public_seed') == public_seed) and (record.get('orchestration_runner_path') == str(orchestration_runner.resolve())) and (simulation is not None) and (truth is not None) and simulation.is_file() and truth.is_file()):
        return False
    simulation_data = None
    try:
        simulation_data = ad.read_h5ad(simulation, backed='r')
        embedded = simulation_data.uns.get('feast_reproduce', {})
        return bool(embedded.get('configuration_id') == configuration_id and embedded.get('feast_version') == str(feast_version) and (int(embedded.get('public_seed', -1)) == public_seed) and (embedded.get('orchestration_runner_path') == str(orchestration_runner.resolve())))
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if simulation_data is not None and simulation_data.isbacked:
            simulation_data.file.close()

def method_record_is_valid(record: dict, public_seed: int, method: str, method_python: Path, rscript: Path | None, orchestration_runner: Path, expected_runner_log: Path, configuration_id: str) -> bool:
    output = record_path(record, 'output_path')
    metadata = record_path(record, 'metadata_path')
    runner_log = record_path(record, 'runner_log_path')
    if not (record.get('status') == 'ok' and record.get('configuration_id') == f'{configuration_id}-{method}' and (record.get('feast_version') == str(feast_version)) and (record.get('public_seed') == public_seed) and (record.get('orchestration_runner_path') == str(orchestration_runner.resolve())) and (record.get('method_python') == str(method_python.resolve())) and (method != 'rctd' or record.get('rscript') == str(rscript.resolve())) and (output is not None) and (metadata is not None) and (runner_log is not None) and (runner_log.resolve() == expected_runner_log.resolve()) and output.is_file() and metadata.is_file() and runner_log.is_file()):
        return False
    try:
        details = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(details.get('status') == 'validated_success' and details.get('method') == method and (details.get('public_seed') == public_seed))

def archive_paths(root: Path, label: str, paths: list[Path], reason: str, orchestration_runner: Path) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    if len({path.name for path in existing}) != len(existing):
        raise RuntimeError('cannot archive artifacts with duplicate file names')
    non_files = [path for path in existing if not path.is_file()]
    if non_files:
        raise RuntimeError(f'refusing to archive non-file artifact: {non_files[0]}')
    archived_utc = datetime.now(timezone.utc)
    stamp = archived_utc.strftime('%Y%m%d_%H%M%S_%f')
    destination = root / 'failures' / f'{label}__{stamp}'
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for path in existing:
        old_path = path.resolve()
        new_path = destination / path.name
        shutil.move(str(path), str(new_path))
        artifacts.append({'disposition': 'superseded_noncanonical', 'reason': reason, 'old_path': str(old_path), 'new_path': str(new_path.resolve())})
    (destination / 'archive_manifest.json').write_text(json.dumps({'status': 'superseded_noncanonical', 'label': label, 'reason': reason, 'archived_utc': archived_utc.isoformat(), 'orchestration_runner_path': str(orchestration_runner.resolve()), 'artifacts': artifacts}, indent=2) + '\n')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rscript', type=Path)
    parser.add_argument('--cell2location-python', type=Path)
    parser.add_argument('--rctd-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'Zenodo/inputs/03_deconvolution/config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--simulate-only', action='store_true', help='write FEAST simulations and ground-truth proportions, but do not start a method job')
    parser.add_argument('--methods', choices=('rctd', 'cell2location'), nargs='+', default=('rctd', 'cell2location'), help='method jobs to run after simulation (default: both)')
    parser.add_argument('--only-resolution', type=float, action='append', help='run only declared resolution(s); useful for a memory-isolated resume')
    args = parser.parse_args()
    if not args.simulate_only:
        if args.rscript is None:
            parser.error('--rscript is required unless --simulate-only is used')
        if args.cell2location_python is None:
            parser.error('--cell2location-python is required unless --simulate-only is used')
    config = yaml.safe_load(args.config.read_text())
    configuration_id = str(config.get('configuration_id', DEFAULT_CONFIGURATION_ID))
    if not configuration_id.strip():
        raise RuntimeError('configuration_id must be a non-empty string')
    rctd_min_cells = int(config['rctd']['min_cells_per_type'])
    cell2location_min_cells = int(config['cell2location'].get('min_cells_per_type', rctd_min_cells))
    if cell2location_min_cells != rctd_min_cells:
        raise RuntimeError('RCTD and Cell2location must use the same min_cells_per_type threshold')
    orchestration_runner = Path(__file__).resolve()
    if str(feast_version) != str(config['feast_version']):
        raise RuntimeError(f'unexpected FEAST version: {feast_version}')
    references = {slice_id: args.input_dir / str(config['reference_pattern']).format(slice_id=slice_id) for slice_id in config['slices']}
    input_manifest_path = args.input_dir / 'input_files.csv'
    required_paths = [*references.values()]
    if not args.simulate_only:
        required_paths.extend([args.rscript, args.cell2location_python, args.rctd_python])
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    declared_resolutions = [float(resolution) for resolution in config['resolutions']]
    selected_resolutions = declared_resolutions
    if args.only_resolution:
        requested = [float(resolution) for resolution in args.only_resolution]
        unknown = [resolution for resolution in requested if not any((np.isclose(resolution, declared) for declared in declared_resolutions))]
        if unknown:
            raise RuntimeError(f'requested undeclared resolution(s): {unknown}')
        selected_resolutions = [resolution for resolution in declared_resolutions if any((np.isclose(resolution, requested_resolution) for requested_resolution in requested))]
    pairs = [(slice_id, resolution) for slice_id in config['slices'] for resolution in selected_resolutions]
    if not pairs or len(set(pairs)) != len(pairs):
        raise RuntimeError('configuration must declare a non-empty set of unique simulation pairs')
    n_method_jobs = 0 if args.simulate_only else len(pairs) * len(args.methods)
    if args.dry_run:
        print(json.dumps({'configuration_id': configuration_id, 'cell_type_key': str(config['cell_type_key']), 'min_cells_per_type': rctd_min_cells, 'simulations': len(pairs), 'method_jobs': n_method_jobs, 'pairs': pairs, 'simulate_only': bool(args.simulate_only), 'methods': list(args.methods)}, indent=2))
        return 0
    if args.output_dir.exists() and (not args.resume):
        raise FileExistsError(f'refusing to overwrite output root: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulation_manifest_path = args.output_dir / 'simulation_manifest.csv'
    method_manifest_path = args.output_dir / 'method_manifest.csv'
    previous_simulations: dict[str, dict] = {}
    previous_methods: dict[tuple[str, str], dict] = {}
    if args.resume and simulation_manifest_path.is_file():
        previous_simulations = {record['pair_id']: record for record in read_manifest(simulation_manifest_path).to_dict('records')}
    if args.resume and method_manifest_path.is_file():
        previous_methods = {(record['pair_id'], record['method']): record for record in read_manifest(method_manifest_path).to_dict('records')}
    pair_order = [f'{slice_id}__resolution_{resolution:g}' for slice_id, resolution in pairs]
    manifest_pair_order = list(dict.fromkeys([*previous_simulations, *pair_order]))

    def write_simulation_manifest(current: list[dict]) -> None:
        merged = dict(previous_simulations)
        merged.update({record['pair_id']: record for record in current})
        write_manifest(pd.DataFrame([merged[key] for key in manifest_pair_order if key in merged]), simulation_manifest_path)
    method_order = list(dict.fromkeys([*previous_methods, *((pair_id, method) for pair_id in pair_order for method in args.methods)]))

    def write_method_manifest(current: list[dict]) -> None:
        merged = dict(previous_methods)
        merged.update({(record['pair_id'], record['method']): record for record in current})
        write_manifest(pd.DataFrame([merged[key] for key in method_order if key in merged]), method_manifest_path)
    simulation_rows = []
    for slice_id, resolution in pairs:
        reference_path = references[slice_id]
        pair_id = f'{slice_id}__resolution_{resolution:g}'
        simulation_path = args.output_dir / 'simulations' / slice_id / f'resolution_{resolution:g}.h5ad'
        truth_path = args.output_dir / 'truth' / slice_id / f'resolution_{resolution:g}_proportions.csv'
        cell_level_path = simulation_path.with_name('cell_level.h5ad')
        assignment_path = truth_path.with_name(f'resolution_{resolution:g}_cell_assignments.csv.gz')
        old = previous_simulations.get(pair_id, {})
        if args.resume and simulation_record_is_valid(old, int(config['seed']), orchestration_runner, configuration_id):
            simulation_rows.append(old)
            write_simulation_manifest(simulation_rows)
            print(f'SIMULATED {pair_id}: skipped verified', flush=True)
            continue
        archive_paths(args.output_dir, f'simulation__{pair_id}', [simulation_path, truth_path, assignment_path], 'existing simulation artifacts lacked a resume record matching the current runner, configuration, and inputs', orchestration_runner)
        reference = ad.read_h5ad(reference_path)
        if config['cell_type_key'] not in reference.obs:
            raise RuntimeError(f"{slice_id}: missing {config['cell_type_key']!r}")
        dense_integer(reference.X, f'reference {slice_id}')
        if not reference.var_names.is_unique or not reference.obs_names.is_unique:
            raise RuntimeError(f'{slice_id}: reference IDs are not unique')
        simulated, cells, indices = simulate_pair(reference, resolution, config, configuration_id)
        simulation_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        if not cells.obs_names.equals(reference.obs_names) or not cells.var_names.equals(reference.var_names):
            raise RuntimeError(f'{slice_id}: cell-level spot or gene order differs from the reference')
        if cell_level_path.is_file():
            saved = ad.read_h5ad(cell_level_path)
            if not (saved.obs_names.equals(cells.obs_names) and saved.var_names.equals(cells.var_names)
                    and np.array_equal(saved.obsm['spatial'], cells.obsm['spatial'])
                    and saved.obs[config['cell_type_key']].equals(cells.obs[config['cell_type_key']])
                    and (csr_matrix(saved.X) != csr_matrix(cells.X)).nnz == 0):
                raise RuntimeError(f'{slice_id}: FEAST cell-level draws differ across resolution jobs')
        else:
            cells.uns['feast_reproduce'] = {
                'configuration_id': configuration_id, 'feast_version': str(feast_version),
                'public_seed': int(config['seed']), 'reference_path': str(reference_path.resolve()),
            }
            temporary = cell_level_path.with_suffix('.pending.h5ad')
            cells.write_h5ad(temporary, compression='gzip')
            temporary.replace(cell_level_path)
        pd.DataFrame({
            'cell_barcode': cells.obs_names,
            'spot_barcode': [f'spot_{index}' for index in indices],
        }).to_csv(assignment_path, index=False)
        simulated.uns['feast_reproduce']['config_path'] = str(args.config.resolve())
        simulated.uns['feast_reproduce']['input_manifest_path'] = str(input_manifest_path.resolve())
        simulated.uns['feast_reproduce']['reference_path'] = str(reference_path.resolve())
        simulated.uns['feast_reproduce']['orchestration_runner_path'] = str(orchestration_runner)
        simulated.write_h5ad(simulation_path, compression='gzip')
        truth = pd.DataFrame(simulated.obsm['cell_type_proportions'], index=[f'spot_{index}' for index in range(simulated.n_obs)], columns=pd.Index(simulated.uns['cell_type_names']).astype(str))
        truth.to_csv(truth_path)
        simulation_rows.append({'pair_id': pair_id, 'slice_id': slice_id, 'resolution': resolution, 'configuration_id': configuration_id, 'public_seed': int(config['seed']), 'feast_version': str(feast_version), 'orchestration_runner_path': str(orchestration_runner), 'reference_path': str(reference_path.resolve()), 'simulation_path': str(simulation_path.resolve()), 'truth_path': str(truth_path.resolve()), 'cell_level_path': str(cell_level_path.resolve()), 'cell_assignment_path': str(assignment_path.resolve()), 'n_spots': simulated.n_obs, 'n_genes': simulated.n_vars, 'n_cell_types': truth.shape[1], 'status': 'ok'})
        write_simulation_manifest(simulation_rows)
        print(f'SIMULATED {pair_id}', flush=True)
    if args.simulate_only:
        simulation_provenance = {'configuration_id': configuration_id, 'generated_utc': datetime.now(timezone.utc).isoformat(), 'run_stage': 'simulation_only', 'feast_version': str(feast_version), 'config_path': str(args.config.resolve()), 'input_manifest_path': str(input_manifest_path.resolve()), 'orchestration_runner_path': str(orchestration_runner), 'reference_inputs': {slice_id: {'path': str(path.resolve())} for slice_id, path in references.items()}, 'simulation_manifest_path': str(simulation_manifest_path.resolve()), 'public_seed': int(config['seed']), 'simulations': len(pairs), 'method_jobs_started': 0, 'planned_methods': list(args.methods), 'same_expression_by_construction': True}
        (args.output_dir / 'simulation_provenance.json').write_text(json.dumps(simulation_provenance, indent=2) + '\n')
        return 0
    method_rows = []
    rctd_script = orchestration_runner.parent / 'methods' / 'run_rctd.py'
    c2l_script = orchestration_runner.parent / 'methods' / 'run_cell2location.py'
    for row in simulation_rows:
        pair_id, slice_id, resolution = (row['pair_id'], row['slice_id'], row['resolution'])
        candidates = {'rctd': (args.rctd_python, rctd_script, args.output_dir / 'methods' / 'rctd' / slice_id / f'resolution_{resolution:g}_proportions.csv'), 'cell2location': (args.cell2location_python, c2l_script, args.output_dir / 'methods' / 'cell2location' / slice_id / f'resolution_{resolution:g}_proportions.csv')}
        jobs = [(method, *candidates[method]) for method in args.methods]
        for method, interpreter, script, output in jobs:
            output.parent.mkdir(parents=True, exist_ok=True)
            metadata = output.with_name(output.stem + '_metadata.json')
            log_path = args.output_dir / 'logs' / f'{pair_id}__{method}.log'
            old = previous_methods.get((pair_id, method), {})
            if args.resume and method_record_is_valid(old, int(config['seed']), method, interpreter, args.rscript if method == 'rctd' else None, orchestration_runner, log_path, configuration_id):
                method_rows.append(old)
                write_method_manifest(method_rows)
                print(f'METHOD {pair_id} {method}: skipped verified', flush=True)
                continue
            archive_paths(args.output_dir, f'method__{pair_id}__{method}', [output, metadata, log_path, output.with_name(output.stem + '_stdout.log'), output.with_name(output.stem + '_stderr.log')], 'existing method artifacts lacked a resume record matching the current runner, configuration, inputs, method source, and log', orchestration_runner)
            command = [str(interpreter.absolute()), str(script), '--input', row['simulation_path'], '--reference', row['reference_path'], '--output', str(output.resolve()), '--cell-type-key', str(config['cell_type_key']), '--seed', str(config['seed']), '--config', str(args.config.resolve())]
            if method == 'rctd':
                command.extend(['--rscript', str(args.rscript.absolute())])
            else:
                command.extend(['--min-cells-per-type', str(cell2location_min_cells)])
                if bool(config['cell2location'].get('positive_library_only', False)):
                    command.append('--positive-library-only')
            print(f'RUNNING {pair_id} {method}', flush=True)
            worker_dir = args.config.resolve().parent if method == 'rctd' else None
            completed = run_method(command, log_path, cwd=worker_dir)
            record = {'pair_id': pair_id, 'slice_id': slice_id, 'resolution': resolution, 'method': method, 'configuration_id': f'{configuration_id}-{method}', 'public_seed': int(config['seed']), 'feast_version': row['feast_version'], 'orchestration_runner_path': str(orchestration_runner), 'method_python': str(interpreter.resolve()), 'rscript': str(args.rscript.resolve()) if method == 'rctd' else '', 'simulation_path': row['simulation_path'], 'reference_path': row['reference_path'], 'output_path': str(output.resolve()) if output.exists() else '', 'metadata_path': str(metadata.resolve()) if metadata.exists() else '', 'runner_log_path': str(log_path.resolve()), 'status': 'ok' if completed.returncode == 0 else 'failed', 'return_code': completed.returncode}
            if record['status'] == 'ok' and (not method_record_is_valid(record, int(config['seed']), method, interpreter, args.rscript if method == 'rctd' else None, orchestration_runner, log_path, configuration_id)):
                record['status'] = 'failed_validation'
            method_rows.append(record)
            write_method_manifest(method_rows)
            print(f"METHOD {pair_id} {method}: {method_rows[-1]['status']}", flush=True)
            if record['status'] != 'ok':
                return completed.returncode or 1
    provenance = {'configuration_id': configuration_id, 'generated_utc': datetime.now(timezone.utc).isoformat(), 'feast_version': str(feast_version), 'config_path': str(args.config.resolve()), 'input_manifest_path': str(input_manifest_path.resolve()), 'orchestration_runner_path': str(orchestration_runner), 'reference_inputs': {slice_id: {'path': str(path.resolve())} for slice_id, path in references.items()}, 'simulation_manifest_path': str(simulation_manifest_path.resolve()), 'method_manifest_path': str(method_manifest_path.resolve()), 'public_seed': int(config['seed']), 'simulations': len(pairs), 'method_jobs': n_method_jobs, 'methods': list(args.methods), 'resume_mode': bool(args.resume), 'same_expression_by_construction': True, 'external_method_limitation': 'External method environments may use NumPy versions unsupported by FEAST; FEAST is not imported or executed in those method workers.'}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
