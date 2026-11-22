"""Run one clustering method sequentially over all 84 fixed-panel inputs."""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
METHOD_SCRIPTS = {'GraphST': 'graphst.py', 'STAGATE_mclust': 'stagate_mclust.py', 'Leiden_unsupervised': 'leiden_unsupervised.py'}
METHOD_CONFIG_KEYS = {'GraphST': 'graphst', 'STAGATE_mclust': 'stagate', 'Leiden_unsupervised': 'leiden'}

def canonical_method_parameters(config: dict, method: str) -> str:
    """Return the exact configured method parameters in a stable representation."""
    return json.dumps(config['methods'][METHOD_CONFIG_KEYS[method]], sort_keys=True, separators=(',', ':'))

def _is_true(value: Any) -> bool:
    return value is True or str(value).casefold() == 'true'

def _int_matches(value: Any, expected: Any) -> bool:
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False

def _int_matches_positive(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False

def method_metadata_is_valid(metadata_path: Path, method: str, settings: dict, public_seed: int, inherited_ld_library_path: str='') -> tuple[bool, str]:
    """Validate the method-specific success, parameter, and accelerator contract."""
    if not metadata_path.is_file():
        return (False, 'metadata.json is missing')
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return (False, f'metadata.json is unreadable: {exc}')
    try:
        metadata_seed = int(metadata.get('public_seed', -1))
    except (TypeError, ValueError):
        metadata_seed = -1
    if metadata.get('status') != 'validated_success':
        return (False, 'method did not report validated_success')
    if metadata.get('method') != method:
        return (False, 'method identity does not match')
    if metadata_seed != int(public_seed):
        return (False, 'public seed does not match')
    environment = metadata.get('environment')
    if not isinstance(environment, dict):
        return (False, 'method environment metadata is missing')
    if not environment.get('executable') or not environment.get('prefix'):
        return (False, 'method interpreter identity is incomplete')
    if method == 'GraphST':
        if metadata.get('device_requested') != str(settings['device']):
            return (False, 'GraphST requested device does not match')
        if not _int_matches(metadata.get('epochs'), settings['epochs']):
            return (False, 'GraphST epoch count does not match')
        if not _int_matches(metadata.get('radius'), settings['radius']):
            return (False, 'GraphST radius does not match')
        if not _is_true(metadata.get('cuda_available')):
            return (False, 'GraphST did not report CUDA availability')
        if not _is_true(metadata.get('cuda_execution_verified')):
            return (False, 'GraphST CUDA execution was not verified')
        if not str(metadata.get('actual_device', '')).startswith('cuda'):
            return (False, 'GraphST actual device was not CUDA')
        if not _int_matches_positive(metadata.get('cuda_peak_memory_bytes')):
            return (False, 'GraphST lacks positive CUDA allocation evidence')
    elif method == 'STAGATE_mclust':
        if not _int_matches(metadata.get('rad_cutoff'), settings['rad_cutoff']):
            return (False, 'STAGATE radius cutoff does not match')
        if not _int_matches(metadata.get('n_epochs'), settings['n_epochs']):
            return (False, 'STAGATE epoch count does not match')
        if metadata.get('accelerator_requested') != 'cuda':
            return (False, 'STAGATE did not request CUDA')
        if metadata.get('actual_accelerator') != 'cuda':
            return (False, 'STAGATE actual accelerator was not CUDA')
        if not _is_true(metadata.get('cuda_execution_verified')):
            return (False, 'STAGATE CUDA execution was not verified')
        if not metadata.get('gpu_devices') or not metadata.get('logical_gpu_devices'):
            return (False, 'STAGATE GPU device evidence is incomplete')
        gpu_memory = metadata.get('gpu_memory_info_bytes', {})
        if not isinstance(gpu_memory, dict) or not _int_matches_positive(gpu_memory.get('peak')):
            return (False, 'STAGATE lacks positive CUDA allocation evidence')
        if environment.get('ld_library_path') != inherited_ld_library_path:
            return (False, 'STAGATE inherited LD_LIBRARY_PATH does not match')
    else:
        expected_resolutions = [float(value) for value in settings['resolutions']]
        try:
            actual_resolutions = [float(value) for value in metadata.get('resolutions_requested', [])]
        except (TypeError, ValueError):
            return (False, 'Leiden requested resolutions are malformed')
        if metadata.get('selection_uses_ground_truth') is not False:
            return (False, 'Leiden selection used ground-truth labels')
        if actual_resolutions != expected_resolutions:
            return (False, 'Leiden resolution grid does not match')
        if not _int_matches(metadata.get('n_pcs_requested'), settings['n_pcs']):
            return (False, 'Leiden PCA setting does not match')
        if not _int_matches(metadata.get('n_neighbors'), settings['n_neighbors']):
            return (False, 'Leiden neighbor setting does not match')
        sweep = metadata.get('sweep')
        if not isinstance(sweep, list) or len(sweep) != len(expected_resolutions):
            return (False, 'Leiden sweep is incomplete')
        forbidden = {'ari', 'nmi', 'ami', 'ground_truth', 'adjusted_rand_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score'}
        if any((forbidden.intersection(entry) for entry in sweep if isinstance(entry, dict))):
            return (False, 'Leiden sweep contains label-informed fields')
        try:
            sweep_resolutions = [float(entry['resolution']) for entry in sweep]
            modularities = [float(entry['modularity_gamma1']) for entry in sweep]
            selected_resolution = float(metadata['selected_resolution'])
        except (KeyError, TypeError, ValueError):
            return (False, 'Leiden sweep diagnostics are malformed')
        if sweep_resolutions != expected_resolutions:
            return (False, 'Leiden sweep resolutions do not match')
        if not all((math.isfinite(value) for value in modularities)):
            return (False, 'Leiden sweep contains non-finite modularity')
        selected_index = max(range(len(sweep)), key=lambda index: (modularities[index], -sweep_resolutions[index]))
        if selected_resolution != sweep_resolutions[selected_index]:
            return (False, 'Leiden selected resolution violates the declared rule')
    return (True, '')

def completed_output_is_valid(output: Path, record: dict, *, config: dict, method: str, method_parameters_json: str, python_invocation: str, python_resolved: str, inherited_ld_library_path: str) -> bool:
    required = (output / 'metadata.json', output / 'clusters.csv', output / 'result.h5ad')
    try:
        seed_matches = int(record.get('public_seed', -1)) == int(config['public_seed'])
    except (TypeError, ValueError):
        seed_matches = False
    expected_fields = {'configuration_id': config['configuration_id'], 'method': method, 'method_parameters_json': method_parameters_json, 'method_python': python_invocation, 'method_python_resolved': python_resolved, 'feast_version': str(config['feast']['version']), 'method_status': 'validated_success'}
    if method == 'STAGATE_mclust':
        expected_fields.update({'ld_library_path': inherited_ld_library_path})
    if not (record and record.get('status') == 'ok' and seed_matches and all((record.get(field) == expected for field, expected in expected_fields.items())) and all((path.is_file() for path in required))):
        return False
    if method in {'GraphST', 'STAGATE_mclust'} and (not _is_true(record.get('cuda_execution_verified'))):
        return False
    metadata_valid, _ = method_metadata_is_valid(output / 'metadata.json', method, config['methods'][METHOD_CONFIG_KEYS[method]], int(config['public_seed']), inherited_ld_library_path)
    return metadata_valid

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--panel-dir', type=Path, required=True)
    parser.add_argument('--method', choices=METHOD_SCRIPTS, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    runner_path = Path(__file__).resolve()
    script = (Path(__file__).parent / 'methods' / METHOD_SCRIPTS[args.method]).resolve()
    settings = config['methods'][METHOD_CONFIG_KEYS[args.method]]
    method_parameters_json = canonical_method_parameters(config, args.method)
    manifest = pd.read_csv(args.manifest, dtype={'slice_id': str, 'simulation_id': str})
    provenance = json.loads((args.panel_dir / 'provenance.json').read_text())
    if provenance['feast_version'] != str(config['feast']['version']):
        raise RuntimeError('fixed panels have a different FEAST version from the configuration')
    if len(manifest) != 84 or manifest[['slice_id', 'simulation_id']].duplicated().any() or (not manifest.status.eq('ok').all()):
        raise RuntimeError('fixed-panel manifest must contain 84 unique jobs')
    if not args.python.is_file() and (not args.dry_run):
        raise FileNotFoundError(args.python)
    if args.dry_run:
        for row in manifest.itertuples(index=False):
            print(f'{args.method} {row.slice_id}/{row.simulation_id}')
        print(f'84 fresh {args.method} jobs')
        return 0
    python_invocation_path = args.python.absolute()
    python_resolved_path = args.python.resolve(strict=True)
    python_invocation = str(python_invocation_path)
    python_resolved = str(python_resolved_path)
    inherited_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '') if args.method == 'STAGATE_mclust' else ''
    method_root = args.output_dir / args.method
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if method_root.exists() and (not args.resume):
        raise FileExistsError(method_root)
    method_root.mkdir(parents=True, exist_ok=True)
    manifest_path = method_root / 'run_manifest.csv'
    previous: dict[tuple[str, str], dict] = {}
    if args.resume and manifest_path.is_file():
        previous_table = pd.read_csv(manifest_path, dtype={'slice_id': str, 'simulation_id': str}, keep_default_na=False)
        previous = {(record['slice_id'], record['simulation_id']): record for record in previous_table.to_dict('records')}
    rows = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        source = args.panel_dir / row.file_path
        output = method_root / row.slice_id / row.simulation_id
        old = previous.get((row.slice_id, row.simulation_id), {})
        if args.resume and completed_output_is_valid(output, old, config=config, method=args.method, method_parameters_json=method_parameters_json, python_invocation=python_invocation, python_resolved=python_resolved, inherited_ld_library_path=inherited_ld_library_path):
            rows.append(old)
            print(f'[{index}/84] {args.method} {row.slice_id}/{row.simulation_id}: skipped verified', flush=True)
            continue
        if output.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            archived = args.output_dir / 'failures' / args.method / row.slice_id / f'{row.simulation_id}__{stamp}'
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(archived))
        output.mkdir(parents=True, exist_ok=False)
        command = [python_invocation, str(script), '--input', str(source), '--output-dir', str(output), '--seed', str(config['public_seed'])]
        if args.method == 'GraphST':
            command += ['--device', str(settings['device']), '--epochs', str(settings['epochs']), '--radius', str(settings['radius'])]
        elif args.method == 'STAGATE_mclust':
            command += ['--rad-cutoff', str(settings['rad_cutoff']), '--n-epochs', str(settings['n_epochs'])]
        else:
            command += ['--resolutions', ','.join(map(str, settings['resolutions'])), '--n-pcs', str(settings['n_pcs']), '--n-neighbors', str(settings['n_neighbors'])]
        started = time.time()
        completed = subprocess.run(command, text=True, capture_output=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
        runner_log = output / 'runner.log'
        runner_log.write_text(shlex.join(command) + '\n' + completed.stdout + '\n--- STDERR ---\n' + completed.stderr)
        required_outputs = (output / 'metadata.json', output / 'clusters.csv', output / 'result.h5ad')
        files_complete = all((path.is_file() for path in required_outputs))
        metadata_valid, metadata_error = method_metadata_is_valid(output / 'metadata.json', args.method, settings, int(config['public_seed']), inherited_ld_library_path)
        ok = completed.returncode == 0 and files_complete and metadata_valid
        metadata: dict[str, Any] = {}
        if (output / 'metadata.json').is_file():
            try:
                metadata = json.loads((output / 'metadata.json').read_text())
            except json.JSONDecodeError:
                pass
        validation_error = ''
        if completed.returncode != 0:
            validation_error = f'method process returned {completed.returncode}'
        elif not files_complete:
            validation_error = 'one or more required output artifacts are missing'
        elif not metadata_valid:
            validation_error = metadata_error
        rows.append({'configuration_id': config['configuration_id'], 'slice_id': row.slice_id, 'simulation_id': row.simulation_id, 'method': args.method, 'method_parameters_json': method_parameters_json, 'method_python': python_invocation, 'method_python_resolved': python_resolved, **({'ld_library_path': inherited_ld_library_path} if args.method == 'STAGATE_mclust' else {}), 'public_seed': config['public_seed'], 'feast_version': str(config['feast']['version']), 'method_status': metadata.get('status', ''), 'cuda_execution_verified': metadata.get('cuda_execution_verified', ''), 'returncode': completed.returncode, 'elapsed_seconds': round(time.time() - started, 2), 'validation_error': validation_error, 'status': 'ok' if ok else 'failed'})
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"[{index}/84] {args.method} {row.slice_id}/{row.simulation_id}: {rows[-1]['status']}", flush=True)
    provenance = {'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'method': args.method, 'method_parameters': settings, 'method_parameters_json': method_parameters_json, 'public_seed': config['public_seed'], 'runner_source': str(runner_path), 'wrapper_source': str(script), 'method_python': python_invocation, 'method_python_resolved': python_resolved, 'feast_version': provenance['feast_version'], 'sequential_execution': True, 'resume_mode': bool(args.resume), 'job_count': len(rows), 'successful_jobs': sum((row['status'] == 'ok' for row in rows))}
    if args.method == 'STAGATE_mclust':
        provenance.update({'ld_library_path': inherited_ld_library_path})
    if args.method == 'GraphST':
        provenance['rng'] = {'training_seed': int(config['public_seed']), 'clustering_pca_seed': 42, 'mclust_seed': 2020}
    (method_root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 0 if all((row['status'] == 'ok' for row in rows)) else 1
if __name__ == '__main__':
    raise SystemExit(main())
