"""Generate all fresh non-OT Study 01 FEAST simulations."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from multiprocessing import get_context
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml

def slug(prefix: str, value: float) -> str:
    return f'{prefix}_{value:.2f}'.replace('.', '_').replace('_00', '')

def build_alterations(config: dict) -> list[dict]:
    from FEAST import Alteration
    settings = config['alterations']
    coupling = settings['mean'].get('variance_coupling')
    rows = []
    for value in settings['mean']['levels']:
        if float(value) != 1.0:
            rows.append({'simulation_id': slug('mean', value), 'alteration_type': 'mean', 'value': value, 'alteration': Alteration.mean_only(fold_change=value, variance_coupling=coupling)})
    rho = settings['variance'].get('heterogeneity_scale', 1.0)
    for value in settings['variance']['levels']:
        if float(value) != 1.0:
            rows.append({'simulation_id': slug('variance', value), 'alteration_type': 'variance', 'value': value, 'alteration': Alteration.variance_only(fold_change=value, dispersion=rho)})
    for value in settings['variance_heterogeneity']['levels']:
        if float(value) != 1.0:
            rows.append({'simulation_id': slug('var_hetero', value), 'alteration_type': 'variance_heterogeneity', 'value': value, 'alteration': Alteration.variance_heterogeneity(rho=value)})
    for value in settings['sparsity_logit_shift']['levels']:
        if float(value) != 0.0:
            sign = 'pos' if value > 0 else 'neg'
            rows.append({'simulation_id': slug(f'sparsity_{sign}', abs(value)), 'alteration_type': 'sparsity_logit_shift', 'value': value, 'alteration': Alteration.sparsity_logit(delta=value)})
    for name, values in settings['combined'].items():
        rows.append({'simulation_id': f'combined_{name}', 'alteration_type': 'combined', 'value': None, 'alteration': Alteration.comprehensive(mean_fc=values.get('mean', 1.0), var_fc=values.get('variance', 1.0), delta_z=values.get('sparsity', 0.0), mean_var_coupling=coupling)})
    rows.append({'simulation_id': 'baseline', 'alteration_type': 'baseline', 'value': 1.0, 'alteration': Alteration()})
    if len(rows) != 27:
        raise RuntimeError(f'expected 27 conditions, found {len(rows)}')
    return rows

def run_simulation(output_dir, slice_id, source, alteration, config):
    import anndata as ad
    import FEAST

    started = time.monotonic()
    simulation_id = alteration['simulation_id']
    relative = Path('simulations') / slice_id / f'{simulation_id}.h5ad'
    output = output_dir / relative
    if output.exists():
        failure = output_dir / 'failures' / slice_id / f'{simulation_id}__{time.time_ns()}.h5ad'
        failure.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(failure))
    log = output_dir / 'logs' / slice_id / f'{simulation_id}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    settings = config['feast']
    try:
        with log.open('w', buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            reference = ad.read_h5ad(source)
            simulated = FEAST.simulate(
                reference,
                seed=int(config['public_seed']),
                alteration=alteration['alteration'],
                parameter_mode=settings['parameter_mode'],
                spatial_mode='reference_rank',
                annotation_key=config['annotation_key'],
                assignment_method=settings['assignment_method'],
                assignment_solver='scipy',
                assignment_blocks=False,
                use_distributional_alteration=settings['use_distributional_alteration'],
                ppf_method=settings['ppf_method'],
                beta_n_jobs=int(settings['beta_n_jobs']),
                beta_early_stopping_patience=int(settings['beta_early_stopping_patience']),
                convert_n_jobs=int(settings['convert_n_jobs']),
                clip_overshoot_factor=0.0,
                boundary_multiplier=1.1,
                verbose=True,
            )
            diagnostics = simulated.uns.get('simulation_diagnostics', {})
            assignment = diagnostics.get('copula_rank_diagnostics', {})
            if diagnostics.get('spatial_mode') != 'reference_rank':
                raise RuntimeError('FEAST returned the wrong spatial mode')
            if assignment.get('assignment_blocks') is not False:
                raise RuntimeError('FEAST did not use exact global assignment')
            simulated.uns['publication_provenance'] = {
                'configuration_id': config['configuration_id'],
                'feast_version': str(FEAST.__version__),
                'public_seed': int(config['public_seed']),
                'spatial_mode': 'reference_rank',
                'assignment_solver': 'scipy',
                'assignment_blocks': False,
                'solver_diagnostics': {
                    'status': 'completed',
                    'assignment_method': assignment['assignment_method'],
                    'assignment_blocks': False,
                    'assignment_solver_requested': 'scipy',
                    'model_selection_counts': diagnostics.get('model_selection_counts', {}),
                },
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            simulated.write_h5ad(output, compression='gzip')
        status = 'ok'
    except Exception as exc:
        status = f'failed: {type(exc).__name__}: {exc}'

    return {
        'configuration_id': config['configuration_id'],
        'slice_id': slice_id,
        'simulation_id': simulation_id,
        'alteration_type': alteration['alteration_type'],
        'alteration_value': alteration['value'],
        'file_path': relative.as_posix(),
        'public_seed': config['public_seed'],
        'feast_version': str(FEAST.__version__),
        'spatial_mode': 'reference_rank',
        'assignment_solver': 'scipy',
        'assignment_blocks': False,
        'solver_status': 'completed' if status == 'ok' else 'failed',
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'status': status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', type=int, default=1, help='maximum concurrent independent FEAST simulations')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    config = yaml.safe_load(args.config.read_text())
    runner_path = Path(__file__).resolve()
    feast_config = config['feast']
    if feast_config.get('spatial_mode') != 'reference_rank' or feast_config.get('assignment_solver') != 'scipy' or feast_config.get('assignment_blocks') is not False:
        raise ValueError('Study 01 requires reference-rank global SciPy assignment')
    if args.dry_run:
        alterations = build_alterations(config)
        for slice_id in config['slices']:
            for alteration in alterations:
                print(f"{slice_id}/{alteration['simulation_id']}")
        print('81 fresh FEAST jobs')
        return 0
    import FEAST
    if str(FEAST.__version__) != str(feast_config['version']):
        raise RuntimeError(f'unexpected FEAST version: {FEAST.__version__}')
    alterations = build_alterations(config)
    sources = {str(slice_id): args.input_dir / f'{slice_id}.h5ad' for slice_id in config['slices']}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing source slices: {missing}')
    if args.output_dir.exists() and (not args.resume):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'simulation_manifest.csv'
    previous: dict[tuple[str, str], dict] = {}
    if args.resume and manifest_path.is_file():
        previous_table = pd.read_csv(manifest_path, dtype={'slice_id': str, 'simulation_id': str})
        previous = {(record['slice_id'], record['simulation_id']): record for record in previous_table.to_dict('records')}
    jobs = {
        (slice_id, alteration['simulation_id']): (source, alteration)
        for slice_id, source in sources.items()
        for alteration in alterations
    }
    rows = {}
    pending = {}
    for key, (source, alteration) in jobs.items():
        old = previous.get(key, {})
        if (
            old.get('status') == 'ok'
            and old.get('configuration_id') == config['configuration_id']
            and old.get('public_seed') == config['public_seed']
            and old.get('feast_version') == str(FEAST.__version__)
            and old.get('spatial_mode') == 'reference_rank'
            and old.get('assignment_solver') == 'scipy'
            and old.get('assignment_blocks') in (False, 'False')
            and (args.output_dir / old['file_path']).is_file()
        ):
            rows[key] = old
        else:
            pending[key] = (args.output_dir, key[0], source, alteration, config)

    def save_result(key, row):
        rows[key] = row
        pending_manifest = manifest_path.with_suffix('.tmp')
        pd.DataFrame(rows[job] for job in jobs if job in rows).to_csv(pending_manifest, index=False)
        pending_manifest.replace(manifest_path)
        print(f"[{len(rows)}/{len(jobs)}] {'/'.join(key)}: {row['status']}", flush=True)

    print(f'{len(rows)} verified and {len(pending)} pending FEAST simulations', flush=True)
    with ExitStack() as executor_context:
        if args.jobs == 1:
            for key, call in pending.items():
                save_result(key, run_simulation(*call))
        elif pending:
            pool = executor_context.enter_context(
                ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context('spawn'))
            )
            futures = {pool.submit(run_simulation, *call): key for key, call in pending.items()}
            for future in as_completed(futures):
                save_result(futures[future], future.result())
        rows = list(rows.values())
        provenance = {
            'configuration_id': config['configuration_id'],
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'public_seed': config['public_seed'],
            'feast_version': str(FEAST.__version__),
            'runner_source': str(runner_path),
            'resume_mode': bool(args.resume),
            'concurrent_jobs': args.jobs,
            'job_count': len(rows),
            'successful_jobs': sum(row['status'] == 'ok' for row in rows),
        }
        provenance_path = args.output_dir / 'provenance.json'
        temporary = provenance_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(provenance, indent=2) + '\n')
        temporary.replace(provenance_path)
    return 0 if all((row['status'] == 'ok' for row in rows)) else 1
if __name__ == '__main__':
    raise SystemExit(main())
