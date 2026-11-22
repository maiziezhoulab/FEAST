"""Run the 28 Spateo fixed-plate alignment jobs."""
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
import numpy as np
import pandas as pd
import yaml

def expected_artifact_names(method: str) -> set[str]:
    common = {'aligned.h5ad', 'aligned_coordinates.csv', 'diagnostics.json', 'transform.json'}
    return common | {'mapping_matrix.npy'}

def artifact_manifest_is_valid(output: Path, method: str, serialized_artifacts: object) -> bool:
    try:
        artifacts = serialized_artifacts if isinstance(serialized_artifacts, list) else json.loads(serialized_artifacts)
    except (TypeError, json.JSONDecodeError):
        return False
    expected_names = expected_artifact_names(method)
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_names):
        return False
    expected_paths = {name: (output / name).resolve() for name in expected_names}
    listed_names: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            return False
        path = output / item['path']
        name = path.name
        if name in listed_names or name not in expected_paths or path.resolve() != expected_paths[name] or (not path.is_file()):
            return False
        listed_names.add(name)
    try:
        actual_names = {path.name for path in output.iterdir() if path.is_file() and path.name != 'runner.log'}
    except OSError:
        return False
    runner_log = output / 'runner.log'
    return bool(listed_names == expected_names and actual_names == expected_names and runner_log.is_file())

def method_matrix_is_valid(output: Path, job: dict, diagnostics: dict, config: dict) -> bool:
    matrix_path = output / 'mapping_matrix.npy'
    try:
        matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
        expected_shape = (int(job['reference_n_spots']), int(job['moving_n_spots']))
        if matrix.ndim != 2 or matrix.shape != expected_shape or matrix.size == 0 or (not np.isfinite(matrix).all()):
            return False
        total_mass = float(np.sum(matrix, dtype=np.float64))
        minimum = float(np.min(matrix))
    except (OSError, TypeError, ValueError):
        return False
    if not np.isfinite(total_mass) or total_mass <= 0:
        return False
    negative_tolerance = float(config['spateo']['mapping_negative_tolerance'])
    try:
        return bool(minimum >= -negative_tolerance and diagnostics.get('mapping_shape') == list(matrix.shape) and (diagnostics.get('mapping_all_finite') is True) and np.isclose(float(diagnostics.get('mapping_minimum')), minimum, rtol=1e-12, atol=1e-12) and np.isclose(float(diagnostics.get('mapping_mass')), total_mass, rtol=1e-12, atol=1e-12) and (float(diagnostics.get('mapping_negative_tolerance')) == negative_tolerance))
    except (TypeError, ValueError):
        return False
    coupling = diagnostics.get('coupling', {})
    try:
        return bool(coupling.get('shape') == list(matrix.shape) and np.isclose(float(coupling.get('mass')), total_mass, rtol=1e-12, atol=1e-12))
    except (TypeError, ValueError):
        return False

def load_jobs(rotation_dir: Path, reference: Path, methods: list[str], config: dict) -> list[dict]:
    manifest_path = rotation_dir / 'rotation_manifest.csv'
    manifest = pd.read_csv(manifest_path)
    required = {'alteration', 'angle_degrees', 'output_path', 'n_spots', 'input_n_spots', 'retained_n_spots', 'dropped_n_spots', 'retained_fraction'}
    if not required.issubset(manifest.columns):
        raise RuntimeError('rotation manifest is missing required support metadata')
    expected_rotations = {(alteration, float(angle)) for alteration in config['alterations'] for angle in config['angles']}
    observed_rotations = set(zip(manifest['alteration'], manifest['angle_degrees'].astype(float)))
    if len(manifest) != len(expected_rotations) or manifest[['alteration', 'angle_degrees']].duplicated().any() or observed_rotations != expected_rotations:
        raise RuntimeError('rotation manifest does not match the configured inputs')
    jobs = []
    for row in manifest.to_dict('records'):
        moving = rotation_dir / row['output_path']
        if not moving.is_file():
            raise RuntimeError(f'rotation artifact is missing: {moving}')
        reference_n_spots = int(row['input_n_spots'])
        moving_n_spots = int(row['retained_n_spots'])
        dropped_n_spots = int(row['dropped_n_spots'])
        if not (reference_n_spots > moving_n_spots > 0 and int(row['n_spots']) == moving_n_spots and (dropped_n_spots == reference_n_spots - moving_n_spots) and np.isclose(float(row['retained_fraction']), moving_n_spots / reference_n_spots)):
            raise RuntimeError(f'invalid retained support metadata: {moving}')
        for method in methods:
            jobs.append({'job_id': f"{row['alteration']}__{method}__angle_{row['angle_degrees']:g}", 'method': method, 'alteration': row['alteration'], 'angle_degrees': float(row['angle_degrees']), 'reference': str(reference.resolve()), 'moving': str(moving.resolve()), 'reference_n_spots': reference_n_spots, 'moving_n_spots': moving_n_spots, 'dropped_n_spots': dropped_n_spots, 'retained_fraction': float(row['retained_fraction']), 'feast_version': row['feast_version']})
    if len(jobs) != len(expected_rotations) * len(methods):
        raise RuntimeError('alignment job matrix is incomplete')
    return jobs

def completed_job_is_valid(output: Path, job: dict, record: dict, config: dict, method_python: Path, public_seed: int) -> bool:
    if not record or record.get('status') != 'ok':
        return False
    expected_configuration_id = f"study02-{job['method']}-{job['alteration']}-{job['angle_degrees']:g}"
    if not (record.get('configuration_id') == expected_configuration_id and record.get('method_python') == str(method_python.resolve()) and (record.get('output_dir') == str(output.resolve())) and (record.get('public_seed') == public_seed) and (record.get('feast_version') == job['feast_version'])):
        return False
    if not artifact_manifest_is_valid(output, job['method'], record.get('artifacts', '[]')):
        return False
    try:
        transform = json.loads((output / 'transform.json').read_text())
        diagnostics = json.loads((output / 'diagnostics.json').read_text())
    except (OSError, json.JSONDecodeError):
        return False
    inputs = transform.get('input_artifacts', {})
    sources = transform.get('source_artifacts', {})
    method_script = Path(__file__).parent / 'methods' / 'run_spateo.py'
    if not (transform.get('status') == 'ok' and transform.get('configuration_id') == expected_configuration_id and (transform.get('public_seed') == public_seed) and method_matrix_is_valid(output, job, diagnostics, config)):
        return False
    return bool(diagnostics.get('status') == 'completed' and diagnostics.get('cuda_available') is True and (diagnostics.get('mapping_all_finite') is True) and (diagnostics.get('convergence_claim') is False) and (diagnostics.get('iteration_schedule', {}).get('completed_return') is True))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--rotation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--spateo-python', type=Path)
    parser.add_argument('--methods', nargs='+', choices=['spateo'], default=['spateo'])
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'Zenodo/inputs/02_alignment' / 'config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    config = yaml.safe_load(args.config.read_text())
    public_seed = int(config['seed'])
    input_manifest = pd.read_csv(args.config.with_name('input_files.csv'))
    if len(input_manifest) != 1:
        raise RuntimeError('alignment input manifest must contain one reference record')
    rotation_provenance_path = args.rotation_dir / 'provenance.json'
    rotation_manifest_path = args.rotation_dir / 'rotation_manifest.csv'
    rotation_provenance = json.loads(rotation_provenance_path.read_text())
    if not int(rotation_provenance.get('public_seed', -1)) == public_seed:
        raise RuntimeError('rotation provenance does not match the current reference/config')
    interpreters = {'spateo': args.spateo_python}
    for method in args.methods:
        executable = interpreters[method]
        if executable is None or not executable.is_file():
            raise FileNotFoundError(f'--{method}-python must name an existing executable')
    all_jobs = load_jobs(args.rotation_dir, args.reference, ['spateo'], config)
    jobs = [job for job in all_jobs if job['method'] in set(args.methods)]
    if args.dry_run:
        print(json.dumps({'jobs': len(jobs), 'methods': args.methods}, indent=2))
        return 0
    if args.output_dir.exists() and (not args.resume):
        raise FileExistsError(f'refusing to overwrite output root: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'method_manifest.csv'
    previous: dict[str, dict] = {}
    path_fields = ('reference', 'moving', 'output_dir')
    if args.resume and manifest_path.is_file():
        previous = {record['job_id']: record for record in pd.read_csv(manifest_path).to_dict('records')}
        for record in previous.values():
            for field in path_fields:
                record[field] = str((args.output_dir / record[field]).resolve())
    results: list[dict] = []
    all_jobs_by_id = {job['job_id']: job for job in all_jobs}
    retained_previous = []
    for record in previous.values():
        if record.get('method') in set(args.methods):
            continue
        job = all_jobs_by_id.get(record.get('job_id'))
        if job is None or not completed_job_is_valid(args.output_dir / job['alteration'] / job['method'] / f"angle_{job['angle_degrees']:g}", job, record, config, interpreters[job['method']], public_seed):
            raise RuntimeError('subset resume cannot retain an unverified row from the other method; resume both methods with the current config and interpreters')
        retained_previous.append(record)

    def write_manifest() -> None:
        table = pd.DataFrame([*retained_previous, *results])
        if not table.empty:
            table = table.sort_values('job_id')
            for field in path_fields:
                table[field] = table[field].map(lambda value: os.path.relpath(value, args.output_dir))
        table.to_csv(manifest_path, index=False)
    method_scripts = {'spateo': Path(__file__).parent / 'methods' / 'run_spateo.py'}
    for index, job in enumerate(jobs, start=1):
        method = job['method']
        output = args.output_dir / job['alteration'] / method / f"angle_{job['angle_degrees']:g}"
        old = previous.get(job['job_id'], {})
        if args.resume and completed_job_is_valid(output, job, old, config, interpreters[method], public_seed):
            results.append(old)
            write_manifest()
            print(f"[{index}/{len(jobs)}] {job['job_id']}: skipped verified", flush=True)
            continue
        if output.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            archived = args.output_dir / 'failures' / f"{job['job_id']}__{stamp}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(archived))
        command = [str(interpreters[method]), str(method_scripts[method]), '--reference', job['reference'], '--moving', job['moving'], '--output-dir', str(output), '--seed', str(public_seed), '--configuration-id', f"study02-{method}-{job['alteration']}-{job['angle_degrees']:g}", '--config', str(args.config)]
        started = time.monotonic()
        completed = subprocess.run(command, text=True, capture_output=True, env={**os.environ, 'PYTHONHASHSEED': '0'})
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / 'runner.log'
        log_path.write_text(shlex.join(command) + '\n' + completed.stdout + '\n--- STDERR ---\n' + completed.stderr)
        artifacts = []
        for path in sorted(output.iterdir()):
            if path.is_file() and path != log_path:
                artifacts.append({'path': path.name})
        status = 'ok' if completed.returncode == 0 else 'failed'
        record = {**job, 'configuration_id': command[command.index('--configuration-id') + 1], 'public_seed': public_seed, 'output_dir': str(output.resolve()), 'status': status, 'return_code': completed.returncode, 'runtime_seconds': round(time.monotonic() - started, 3), 'method_python': str(interpreters[method].resolve()), 'artifacts': json.dumps(artifacts, sort_keys=True)}
        if status == 'ok' and (not completed_job_is_valid(output, job, record, config, interpreters[method], public_seed)):
            record['status'] = 'failed_validation'
        results.append(record)
        write_manifest()
        print(f"[{index}/{len(jobs)}] {job['job_id']}: {record['status']}", flush=True)
        if record['status'] != 'ok' and (not args.continue_on_error):
            return completed.returncode or 1
    return 0 if all((row['status'] == 'ok' for row in results)) else 1
if __name__ == '__main__':
    raise SystemExit(main())
