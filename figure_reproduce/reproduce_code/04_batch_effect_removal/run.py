"""Generate the Study 04 ladder, fixed panel, and method outputs.

The workflow writes only below a fresh output root.  Long method stages are
resumable: a cell is skipped only after its recorded provenance and semantic
checks validate; an incomplete or invalid cell is moved under ``failures/`` first.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
STUDY_ROOT = Path(__file__).resolve().parent
METHOD_ARTIFACTS = {'GraphST': ('metadata.json', 'result.h5ad', 'embeddings.npz', 'graphst_checkpoint.pt'), 'STAMP': ('metadata.json', 'result.h5ad', 'embeddings.npz', 'cell_by_topic.csv', 'stamp_params.pt'), 'scVI': ('metadata.json', 'result.h5ad', 'embeddings.npz', 'model/model.pt')}

def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()

def load_config(path: Path) -> dict:
    path = path.resolve()
    with path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    required = {'configuration_id', 'legacy_study_id', 'public_seed', 'paths', 'simulation', 'fixed_panel', 'methods'}
    missing = required - set(config)
    if missing:
        raise ValueError(f'config is missing fields: {sorted(missing)}')
    if str(config['legacy_study_id']) != '08':
        raise ValueError("legacy_study_id must remain '08'")
    simulation = config['simulation']
    if simulation['modes'] != ['shift_only', 'diagonal_affine']:
        raise ValueError('the declared two-mode ladder changed')
    expected_alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    if [float(value) for value in simulation['alpha_levels']] != expected_alphas:
        raise ValueError('the declared alpha ladder changed')
    if set(config['methods']) != set(METHOD_ARTIFACTS):
        raise ValueError('exactly GraphST, STAMP, and scVI must be configured')
    config['_config_path'] = path
    config['_feast_repo'] = resolve_path(path, config['paths']['feast_repo'])
    config['_input_manifest'] = resolve_path(path, config['paths']['input_manifest'])
    config['_output_dir'] = resolve_path(path, config['paths']['output_dir'])
    return config

def feast_identity(config: dict) -> dict[str, str]:
    import FEAST
    version = str(FEAST.__version__)
    import_path = Path(FEAST.__file__).resolve()
    expected_path = config['_feast_repo'] / 'src/FEAST/__init__.py'
    if version != str(config['feast_version']) or import_path != expected_path:
        raise RuntimeError(f'unexpected FEAST import: {version} from {import_path}')
    return {'version': version, 'import_path': str(import_path)}

def raw_input_paths(config: dict) -> tuple[Path, Path]:
    simulation = config['simulation']
    manifest = config['_input_manifest']
    files = pd.read_csv(manifest, dtype={'slice_id': str}).set_index('slice_id')['relative_path']
    return ((manifest.parent / files[simulation['reference_slice']]).resolve(),
            (manifest.parent / files[simulation['deformation_fit_slice']]).resolve())

def validate_count_matrix(matrix, label: str) -> None:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError(f'{label} contains non-finite values')
    if np.any(values < 0):
        raise ValueError(f'{label} contains negative values')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f'{label} is not integral')

def simulate_ladder(config: dict, output_root: Path) -> Path:
    """Generate two six-cell batch ladders plus their alpha-zero baselines."""
    import anndata
    import FEAST
    import scanpy as sc
    feast = feast_identity(config)
    if output_root.exists():
        raise FileExistsError(f'refusing to reuse output root: {output_root}')
    reference_path, fit_query_path = raw_input_paths(config)
    for path in (reference_path, fit_query_path):
        if not path.is_file():
            raise FileNotFoundError(f'required input is missing: {path}')
    output_root.mkdir(parents=True)
    simulation_root = output_root / 'simulations'
    simulation_root.mkdir()
    reference = sc.read_h5ad(reference_path)
    fit_query = sc.read_h5ad(fit_query_path)
    validate_count_matrix(reference.X, 'reference X')
    validate_count_matrix(fit_query.X, 'deformation-fit query X')
    if not reference.obs_names.is_unique or not reference.var_names.is_unique:
        raise ValueError('reference spot and gene identifiers must be unique')
    if 'spatial' not in reference.obsm:
        raise ValueError("reference is missing obsm['spatial']")
    characterized = FEAST.characterize_batch(reference, fit_query, name='study04_characterized_batch')
    deformations = {'shift_only': FEAST.BatchDeformation(D=np.ones(3, dtype=float), b=np.asarray(characterized.b, dtype=float).copy(), name='shift_only'), 'diagonal_affine': FEAST.BatchDeformation(D=np.asarray(characterized.D, dtype=float).copy(), b=np.asarray(characterized.b, dtype=float).copy(), name='diagonal_affine')}
    seed = int(config['public_seed'])
    rows: list[dict] = []
    output_records: list[dict] = []
    for mode in config['simulation']['modes']:
        deformation = deformations[mode]
        mode_root = simulation_root / mode
        mode_root.mkdir()
        for alpha_value in config['simulation']['alpha_levels']:
            alpha = float(alpha_value)
            simulated = FEAST.simulate_batch_effect(reference, D=np.asarray(deformation.D, dtype=float), b=np.asarray(deformation.b, dtype=float), alpha=alpha, random_seed=seed)
            validate_count_matrix(simulated.X, f'{mode}/alpha_{alpha:.2f}')
            if not simulated.obs_names.equals(reference.obs_names):
                raise RuntimeError('FEAST changed spot identity/order')
            if not simulated.var_names.equals(reference.var_names):
                raise RuntimeError('FEAST changed gene identity/order')
            if not np.array_equal(simulated.obsm['spatial'], reference.obsm['spatial']):
                raise RuntimeError('FEAST changed spatial geometry')
            simulated.uns['publication_reproduction'] = {'configuration_id': str(config['configuration_id']), 'study_id': '04', 'legacy_study_id': '08', 'public_seed': seed, 'feast_version': feast['version'], 'deformation_mode': mode, 'alpha': alpha, 'alpha_stratum': 'interpolation' if alpha <= 1.0 else 'extrapolation'}
            output_path = mode_root / f'alpha_{alpha:.2f}.h5ad'
            simulated.write_h5ad(output_path, compression='gzip')
            rows.append({'configuration_id': config['configuration_id'], 'study_id': '04', 'legacy_study_id': '08', 'mode': mode, 'alpha': alpha, 'alpha_stratum': 'interpolation' if alpha <= 1.0 else 'extrapolation', 'file': output_path.relative_to(output_root).as_posix(), 'n_spots': int(simulated.n_obs), 'n_genes': int(simulated.n_vars), 'public_seed': seed, 'matrix_all_finite': True, 'matrix_nonnegative': True, 'matrix_integral': True, 'spot_order_exact': True, 'gene_order_exact': True, 'spatial_exact': True})
            output_records.append({'path': str(output_path.resolve())})
            print(f'simulated {mode}/alpha_{alpha:.2f}', flush=True)
    manifest = pd.DataFrame(rows)
    manifest_path = simulation_root / 'manifest.csv'
    manifest.to_csv(manifest_path, index=False)
    deformation_diagnostics = {mode: {'D': np.asarray(value.D, dtype=float).tolist(), 'b': np.asarray(value.b, dtype=float).tolist()} for mode, value in deformations.items()}
    deformation_diagnostics['characterized_ols'] = {}
    for name in ('r2', 'residual_std', 'n_genes'):
        value = getattr(characterized, name, None)
        if value is not None:
            deformation_diagnostics['characterized_ols'][name] = np.asarray(value).tolist() if name != 'n_genes' else int(value)
    provenance = {'schema_version': 1, 'configuration_id': config['configuration_id'], 'study_id': '04', 'legacy_study_id': '08', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'feast': feast, 'public_seed': seed, 'rng_policy': 'FEAST simulate_batch_effect random_seed', 'inputs': [{'role': 'clean_reference', 'path': str(reference_path.resolve())}, {'role': 'deformation_fit_query', 'path': str(fit_query_path.resolve())}, {'role': 'configuration', 'path': str(config['_config_path'])}], 'deformation_diagnostics': deformation_diagnostics, 'solver_diagnostics': {'status': 'completed', 'comparison_conditions': 12, 'alpha_zero_baselines': 2, 'written_h5ad': len(rows), 'all_spot_gene_geometry_invariants_exact': True, 'all_matrices_finite_nonnegative_integral': True}, 'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scanpy': sc.__version__, 'anndata': anndata.__version__}, 'sources': [{'path': str(Path(__file__).resolve())}], 'outputs': output_records, 'manifest': {'path': str(manifest_path.resolve())}}
    (simulation_root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return simulation_root

def sparse_exact(left, right) -> bool:
    if sp.issparse(left) and sp.issparse(right):
        return bool((left != right).nnz == 0)
    left_array = left.toarray() if sp.issparse(left) else np.asarray(left)
    right_array = right.toarray() if sp.issparse(right) else np.asarray(right)
    return bool(np.array_equal(left_array, right_array))

def build_fixed_panel(config: dict, output_root: Path) -> Path:
    """Select one raw-reference Seurat-v3 panel and audit ladder support."""
    import scanpy as sc
    panel_root = output_root / 'panel'
    if panel_root.exists():
        raise FileExistsError(f'refusing to overwrite fixed panel: {panel_root}')
    simulation_root = output_root / 'simulations'
    manifest_path = simulation_root / 'manifest.csv'
    reference_path, _ = raw_input_paths(config)
    if not manifest_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError('simulation manifest and raw reference are required')
    source = sc.read_h5ad(reference_path)
    source_layer = str(config['fixed_panel']['source_layer'])
    if source_layer not in source.layers:
        raise ValueError(f'reference is missing layers[{source_layer!r}]')
    if not sparse_exact(source.X, source.layers[source_layer]):
        raise ValueError('reference X and the declared raw-count layer differ')
    validate_count_matrix(source.layers[source_layer], 'panel source counts')
    n_genes = int(config['fixed_panel']['n_genes'])
    selected = source.copy()
    sc.pp.highly_variable_genes(selected, n_top_genes=n_genes, flavor=str(config['fixed_panel']['flavor']), layer=source_layer, inplace=True)
    mask = selected.var['highly_variable'].astype(bool).to_numpy()
    if int(mask.sum()) != n_genes:
        raise RuntimeError(f'selected {int(mask.sum())} genes; expected {n_genes}')
    source_indices = np.flatnonzero(mask)
    genes = source.var_names[source_indices].astype(str).tolist()
    if len(set(genes)) != len(genes):
        raise RuntimeError('selected panel contains duplicate genes')
    ladder = pd.read_csv(manifest_path)
    if len(ladder) != 14:
        raise ValueError(f'expected 14 ladder files including baselines; found {len(ladder)}')
    support_rows = []
    for row in ladder.itertuples(index=False):
        path = output_root / row.file
        if not path.is_file():
            raise ValueError(f'ladder file is missing: {path}')
        candidate = sc.read_h5ad(path, backed='r')
        try:
            exact_support = all((gene in candidate.var_names for gene in genes))
            exact_order = candidate.var_names.equals(source.var_names)
            n_obs, n_vars = candidate.shape
        finally:
            candidate.file.close()
        if not exact_support or not exact_order:
            raise ValueError(f'ladder gene contract failed: {path}')
        support_rows.append({'mode': row.mode, 'alpha': float(row.alpha), 'path': path.relative_to(output_root).as_posix(), 'n_spots': int(n_obs), 'n_genes': int(n_vars), 'panel_support_exact': exact_support, 'full_gene_order_exact': exact_order})
    panel_root.mkdir(parents=True)
    panel_path = panel_root / 'panel.csv'
    genes_path = panel_root / 'genes.txt'
    support_path = panel_root / 'support.csv'
    pd.DataFrame({'panel_order': np.arange(n_genes, dtype=int), 'gene': genes, 'source_var_index': source_indices, 'seurat_v3_rank': selected.var.iloc[source_indices]['highly_variable_rank'].to_numpy(dtype=float)}).to_csv(panel_path, index=False)
    genes_path.write_text('\n'.join(genes) + '\n', encoding='utf-8')
    pd.DataFrame(support_rows).to_csv(support_path, index=False)
    outputs = [panel_path, genes_path, support_path]
    provenance = {'schema_version': 1, 'configuration_id': f"{config['configuration_id']}__fixed_panel", 'study_id': '04', 'legacy_study_id': '08', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'feast': feast_identity(config), 'public_seed': int(config['public_seed']), 'selection': {'source': str(reference_path.resolve()), 'source_layer': source_layer, 'flavor': config['fixed_panel']['flavor'], 'n_genes': n_genes, 'ordering': 'raw_reference_var_order'}, 'solver_diagnostics': {'status': 'completed', 'selected_gene_count': len(genes), 'unique_gene_count': len(set(genes)), 'ladder_input_count': len(support_rows), 'all_panel_support_exact': True, 'all_full_gene_order_exact': True, 'all_manifest_paths_present': True}, 'inputs': [{'path': str(reference_path.resolve())}, {'path': str(manifest_path.resolve())}], 'sources': [{'path': str(Path(__file__).resolve())}, {'path': str(config['_config_path'])}], 'outputs': [{'path': str(path.resolve())} for path in outputs]}
    provenance_path = panel_root / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return panel_root

def candidate_is_valid(candidate: Path, method: str, candidate_id: str, config: dict) -> tuple[bool, str]:
    for relative in METHOD_ARTIFACTS[method]:
        if not (candidate / relative).is_file():
            return (False, f'missing {relative}')
    try:
        metadata = json.loads((candidate / 'metadata.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return (False, f'invalid metadata: {error}')
    provenance = metadata.get('provenance', {})
    if provenance.get('feast', {}).get('version') != str(config['feast_version']):
        return (False, 'candidate upstream FEAST version mismatch')
    context = provenance.get('candidate_context', {})
    if provenance.get('configuration_id') != candidate_id:
        return (False, 'candidate configuration ID mismatch')
    if context.get('parent_configuration_id') != config['configuration_id']:
        return (False, 'parent configuration ID mismatch')
    if context.get('method') != method:
        return (False, 'method context mismatch')
    if int(provenance.get('public_seed', -1)) != int(config['public_seed']):
        return (False, 'public seed mismatch')
    diagnostics = provenance.get('solver_diagnostics', {})
    if diagnostics.get('status') != 'completed':
        return (False, 'method diagnostics are not complete')
    if diagnostics.get('cuda_execution_verified') is not True or not str(diagnostics.get('actual_device', '')).startswith('cuda'):
        return (False, 'candidate does not prove actual CUDA execution')
    runtime_controls = metadata.get('runtime_controls', {})
    if runtime_controls.get('pip_check_status') != 'ok':
        return (False, 'candidate environment lacks a successful pip check')
    method_environment = metadata.get('execution_environment', {}) if method == 'scVI' else metadata.get('environment', {})
    numpy_disposition = method_environment.get('numpy_supported_by_feast')
    if numpy_disposition not in {True, False}:
        return (False, 'candidate lacks an explicit FEAST NumPy support disposition')
    if numpy_disposition is False and 'unsupported' not in str(method_environment.get('external_environment_limitation', '')):
        return (False, 'unsupported external NumPy is not explicitly reported')
    if method in {'GraphST', 'STAMP'} and diagnostics.get('cross_batch_edge_count') != 0:
        return (False, 'cross-batch graph edge found')
    for collection in ('inputs', 'outputs', 'sources'):
        records = provenance.get(collection, [])
        if not records:
            return (False, f'empty provenance collection: {collection}')
        for record in records:
            path = Path(record.get('path', ''))
            if not path.is_file():
                return (False, f'provenance path is missing: {path}')
    try:
        with np.load(candidate / 'embeddings.npz', allow_pickle=True) as payload:
            embedding = np.asarray(payload['embedding'])
        if embedding.ndim != 2 or not np.isfinite(embedding).all():
            return (False, 'embedding is not a finite matrix')
    except Exception as error:
        return (False, f'invalid embedding: {error}')
    return (True, 'validated completed candidate')

def archive_invalid(candidate: Path, failure_root: Path, reason: str) -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = failure_root / f'{candidate.parent.parent.name}__{candidate.parent.name}__{candidate.name}__{stamp}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists():
        candidate.replace(destination)
    else:
        destination.mkdir()
    (destination / 'DISPOSITION.txt').write_text(reason + '\n', encoding='utf-8')
    return destination

def format_cli_value(value: object) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)

def run_methods(config: dict, output_root: Path, selected_method: str | None=None) -> int:
    simulation_root = output_root / 'simulations'
    manifest_path = simulation_root / 'manifest.csv'
    panel_path = output_root / 'panel' / 'panel.csv'
    panel_provenance = output_root / 'panel' / 'provenance.json'
    for path in (manifest_path, panel_path, panel_provenance):
        if not path.is_file():
            raise FileNotFoundError(f'required prepared input is missing: {path}')
    manifest = pd.read_csv(manifest_path)
    methods = config['methods']
    if selected_method is not None:
        if selected_method not in methods:
            raise ValueError(f'unknown method: {selected_method}')
        methods = {selected_method: methods[selected_method]}
    seed = int(config['public_seed'])
    method_root = output_root / 'methods'
    failure_root = output_root / 'failures'
    rows = []
    failures = 0
    for method, method_config in methods.items():
        python = resolve_path(config['_config_path'], method_config['python'])
        script = resolve_path(config['_config_path'], method_config['script'])
        if not python.is_file() or not script.is_file():
            raise FileNotFoundError(f'{method} environment or wrapper is missing')
        pip_check = subprocess.run([str(python), '-m', 'pip', 'check'], capture_output=True, text=True)
        if pip_check.returncode != 0:
            details = (pip_check.stdout + '\n' + pip_check.stderr).strip()
            raise RuntimeError(f'{method} environment failed pip check: {details}')
        environment = os.environ.copy()
        environment.update({'PYTHONNOUSERSITE': '1', 'PYTHONHASHSEED': str(seed), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONIOENCODING': 'utf-8', 'LC_ALL': 'C.UTF-8', 'FEAST_REPO': str(config['_feast_repo']), 'FEAST_REPRODUCE_PIP_CHECK': 'ok'})
        for key, value in method_config.get('environment', {}).items():
            environment[str(key)] = str(resolve_path(config['_config_path'], value))
        for mode in config['simulation']['modes']:
            reference_row = manifest[(manifest['mode'] == mode) & (manifest['alpha'] == 0.0)]
            if len(reference_row) != 1:
                raise ValueError(f'missing unique baseline for {mode}')
            reference = output_root / reference_row.iloc[0]['file']
            for alpha_value in config['simulation']['alpha_levels']:
                alpha = float(alpha_value)
                if alpha == 0.0:
                    continue
                query_row = manifest[(manifest['mode'] == mode) & (manifest['alpha'] == alpha)]
                if len(query_row) != 1:
                    raise ValueError(f'missing unique input for {mode}/alpha_{alpha:.2f}')
                query = output_root / query_row.iloc[0]['file']
                candidate_id = f"{config['configuration_id']}__{method}__{mode}__alpha_{alpha:.2f}"
                candidate = method_root / method / mode / f'alpha_{alpha:.2f}'
                if candidate.exists():
                    valid, reason = candidate_is_valid(candidate, method, candidate_id, config)
                    if valid:
                        print(f'skip verified {method}/{mode}/alpha_{alpha:.2f}')
                        log = output_root / 'logs' / method / mode / f'alpha_{alpha:.2f}.log'
                        retry_log = log.with_name(f'{log.stem}_retry.log')
                        if retry_log.is_file():
                            log = retry_log
                        rows.append({'method': method, 'mode': mode, 'alpha': alpha, 'status': 'skipped_verified',
                                     'candidate_id': candidate_id, 'output_dir': candidate.relative_to(output_root).as_posix(),
                                     'log': log.relative_to(output_root).as_posix()})
                        continue
                    archived = archive_invalid(candidate, failure_root, reason)
                    print(f'archived invalid attempt: {archived}')
                command = [str(python), str(script), '--reference', str(reference), '--query', str(query), '--output-dir', str(candidate), '--panel-file', str(panel_path), '--panel-provenance', str(panel_provenance), '--seed', str(seed), '--config-id', candidate_id, '--parent-config-id', str(config['configuration_id']), '--deformation-mode', mode, '--alpha', str(alpha), '--config-path', str(config['_config_path'])]
                for name, value in method_config.get('kwargs', {}).items():
                    command.extend([f"--{name.replace('_', '-')}", format_cli_value(value)])
                candidate.parent.mkdir(parents=True, exist_ok=True)
                log_root = output_root / 'logs' / method / mode
                log_root.mkdir(parents=True, exist_ok=True)
                log_path = log_root / f'alpha_{alpha:.2f}.log'
                print(f'run {method}/{mode}/alpha_{alpha:.2f}', flush=True)
                with log_path.open('w', encoding='utf-8') as log_handle:
                    log_handle.write(shlex.join(command) + '\n')
                    log_handle.flush()
                    result = subprocess.run(command, cwd=STUDY_ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                valid, reason = candidate_is_valid(candidate, method, candidate_id, config)
                if result.returncode != 0 or not valid:
                    failures += 1
                    reason = f'exit={result.returncode}; {reason}'
                    archived = archive_invalid(candidate, failure_root, reason)
                    status = 'failed'
                    output_value = archived.relative_to(output_root).as_posix()
                    print(f'FAILED {method}/{mode}/alpha_{alpha:.2f}: {reason}')
                else:
                    status = 'complete'
                    output_value = candidate.relative_to(output_root).as_posix()
                rows.append({'method': method, 'mode': mode, 'alpha': alpha, 'status': status, 'candidate_id': candidate_id, 'output_dir': output_value, 'log': log_path.relative_to(output_root).as_posix()})
    run_manifest = output_root / 'method_run_manifest.csv'
    current = pd.DataFrame(rows)
    if selected_method is not None and run_manifest.is_file():
        previous = pd.read_csv(run_manifest)
        previous = previous[previous['method'] != selected_method]
        current = pd.concat([previous, current], ignore_index=True)
    if not current.empty:
        current = current.sort_values(['method', 'mode', 'alpha'])
    current.to_csv(run_manifest, index=False)
    return failures

def print_plan(config: dict, output_root: Path, selected_method: str | None) -> None:
    reference, query = raw_input_paths(config)
    print(f"configuration: {config['configuration_id']}")
    print('study: 04 (legacy 08)')
    print(f'reference: {reference}')
    print(f'deformation fit: {query}')
    print(f'output: {output_root}')
    print('simulation: 12 nonzero conditions + 2 mode-specific alpha-zero baselines')
    methods = [selected_method] if selected_method else list(config['methods'])
    for method in methods:
        print(f'{method}: 12 jobs on CUDA')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_ROOT / 'config.yaml')
    parser.add_argument('--stage', choices=('all', 'simulate', 'panel', 'methods'), default='all')
    parser.add_argument('--method', choices=tuple(METHOD_ARTIFACTS), default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = args.output_dir.expanduser().resolve() if args.output_dir is not None else config['_output_dir']
    if args.dry_run:
        print_plan(config, output_root, args.method)
        return 0
    os.environ['FEAST_VERSION'] = feast_identity(config)['version']
    if args.stage == 'all':
        simulate_ladder(config, output_root)
        build_fixed_panel(config, output_root)
        return 1 if run_methods(config, output_root, args.method) else 0
    if args.stage == 'simulate':
        simulate_ladder(config, output_root)
        return 0
    if args.stage == 'panel':
        build_fixed_panel(config, output_root)
        return 0
    return 1 if run_methods(config, output_root, args.method) else 0
if __name__ == '__main__':
    raise SystemExit(main())
