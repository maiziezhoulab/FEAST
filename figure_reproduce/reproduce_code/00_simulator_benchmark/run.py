"""Generate the fresh Study 00 FEAST simulations."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml
import pandas as pd
LABEL_COLUMNS = ('ground_truth', 'cell_type', 'annotation', 'region', 'cluster')

def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    feast = config['feast']
    required = {'spatial_mode': 'reference_rank', 'assignment_solver': 'scipy', 'assignment_blocks': False}
    for key, expected in required.items():
        if feast.get(key) != expected:
            raise ValueError(f'Study 00 requires feast.{key}={expected!r}')
    if len(config['samples']) != 12 or len(set(config['samples'])) != 12:
        raise ValueError('Study 00 must declare 12 unique samples')
    return config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample', action='append', help='Run only this declared sample; repeat to select more than one.')
    parser.add_argument('--feast-version', default='1.0.2', help='Expected installed FEAST version; the recorded Study 00 default is 1.0.2.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    runner_path = Path(__file__).resolve()
    selected = set(args.sample or config['samples'])
    unknown = selected - set(config['samples'])
    if unknown:
        raise ValueError(f'unknown samples: {sorted(unknown)}')
    jobs = [(sample, args.reference_dir / f'{sample}.h5ad') for sample in config['samples'] if sample in selected]
    if args.dry_run:
        for index, (sample, source) in enumerate(jobs, start=1):
            print(f'{index:02d}/{len(jobs)} {sample}: {source}')
        print(f'{len(jobs)} fresh FEAST reference-rank jobs')
        return 0
    missing = [str(path) for _, path in jobs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing reference inputs: {missing}')
    inputs = pd.read_csv(Path(__file__).parent / 'data/input_files.csv')
    declared_samples = set(inputs.loc[inputs.role == 'reference', 'sample'].astype(str))
    if declared_samples != set(config['samples']):
        raise RuntimeError('reference input manifest does not match the configured samples')
    if args.output_dir.exists() and (not args.resume):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'simulation_manifest.csv'
    previous: dict[str, dict] = {}
    if args.resume and manifest_path.is_file():
        previous = {record['sample']: record for record in pd.read_csv(manifest_path).to_dict('records')}
    import anndata as ad
    import FEAST
    if str(FEAST.__version__) != args.feast_version:
        raise RuntimeError(f'unexpected FEAST version: {FEAST.__version__}')
    feast_config = config['feast']
    legacy_options = {'n_jobs': int(feast_config['n_jobs']), 'use_heuristic_search': False} if str(FEAST.__version__) == '1.0.2' else {}
    rows: list[dict] = []
    for index, (sample, source) in enumerate(jobs, start=1):
        started = time.time()
        output = Path('simulations') / f'{sample}.h5ad'
        output_path = args.output_dir / output
        print(f'[{index}/{len(jobs)}] {sample}', flush=True)
        old = previous.get(sample, {})
        if args.resume and old.get('status') == 'ok' and (old.get('configuration_id') == config['configuration_id']) and (old.get('public_seed') == config['public_seed']) and (old.get('feast_version') == str(FEAST.__version__)) and (old.get('spatial_mode') == 'reference_rank') and (old.get('assignment_solver') == 'scipy') and (old.get('assignment_blocks') in (False, 'False')) and output_path.is_file():
            rows.append(old)
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            print('  skipped verified', flush=True)
            continue
        if output_path.exists():
            failure = args.output_dir / 'failures' / f'{sample}__{int(time.time())}.h5ad'
            failure.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output_path), str(failure))
        try:
            reference = ad.read_h5ad(source)
            annotation_key = next((name for name in LABEL_COLUMNS if name in reference.obs), None)
            simulated = FEAST.simulate(reference, seed=int(config['public_seed']), alteration=None, parameter_mode=feast_config['parameter_mode'], spatial_mode='reference_rank', annotation_key=annotation_key, assignment_method=feast_config['assignment_method'], assignment_solver='scipy', assignment_blocks=False, ppf_method=feast_config['ppf_method'], beta_n_jobs=int(feast_config['beta_n_jobs']), beta_early_stopping_patience=int(feast_config['beta_early_stopping_patience']), convert_n_jobs=int(feast_config['convert_n_jobs']), clip_overshoot_factor=0.0, boundary_multiplier=1.1, verbose=True, **legacy_options)
            diagnostics = simulated.uns.get('simulation_diagnostics', {})
            assignment_diagnostics = diagnostics.get('copula_rank_diagnostics', {})
            if diagnostics.get('spatial_mode') != 'reference_rank':
                raise RuntimeError('FEAST returned the wrong spatial mode')
            if assignment_diagnostics.get('assignment_blocks') is not False:
                raise RuntimeError('FEAST did not use exact global assignment')
            simulated.uns['publication_provenance'] = {'configuration_id': str(config['configuration_id']), 'feast_version': str(FEAST.__version__), 'public_seed': int(config['public_seed']), 'spatial_mode': 'reference_rank', 'assignment_solver': 'scipy', 'assignment_blocks': False, 'solver_diagnostics': {'status': 'completed', 'assignment_method': assignment_diagnostics.get('assignment_method'), 'assignment_blocks': False, 'assignment_solver_requested': 'scipy', 'model_selection_counts': diagnostics.get('model_selection_counts', {})}, 'input_artifact_id': f'study00:reference:{sample}'}
            output_path.parent.mkdir(parents=True, exist_ok=True)
            simulated.write_h5ad(output_path, compression='gzip')
            row = {'configuration_id': config['configuration_id'], 'sample': sample, 'simulator': feast_config['simulator_name'], 'file_path': output.as_posix(), 'n_obs': simulated.n_obs, 'n_vars': simulated.n_vars, 'public_seed': config['public_seed'], 'feast_version': FEAST.__version__, 'spatial_mode': 'reference_rank', 'assignment_solver': 'scipy', 'assignment_blocks': False, 'solver_status': 'completed', 'elapsed_seconds': round(time.time() - started, 2), 'status': 'ok'}
        except Exception as exc:
            row = {'configuration_id': config['configuration_id'], 'sample': sample, 'simulator': feast_config['simulator_name'], 'file_path': output.as_posix(), 'public_seed': config['public_seed'], 'elapsed_seconds': round(time.time() - started, 2), 'status': f'failed: {type(exc).__name__}: {exc}'}
        rows.append(row)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        if row['status'] != 'ok':
            print(row['status'], flush=True)
    provenance = {'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'public_seed': config['public_seed'], 'feast_version': str(FEAST.__version__), 'feast_import_path': str(Path(FEAST.__file__).resolve()), 'runner_source': str(runner_path), 'job_count': len(rows), 'successful_jobs': sum((row['status'] == 'ok' for row in rows)), 'resume_mode': bool(args.resume)}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 0 if all((row['status'] == 'ok' for row in rows)) else 1
if __name__ == '__main__':
    raise SystemExit(main())
