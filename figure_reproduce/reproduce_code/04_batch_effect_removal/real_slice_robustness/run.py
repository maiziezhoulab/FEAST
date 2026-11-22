"""Prepare and run the frozen Study 04B production matrix."""
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
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import metrics
import workflow
STUDY_ROOT = Path(__file__).resolve().parent

def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)

def _gene_theta(matrix) -> np.ndarray:
    values = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    means = np.asarray(values.mean(axis=0)).ravel()
    squared = values.copy().astype(np.float64)
    squared.data **= 2
    variances = np.asarray(squared.mean(axis=0)).ravel() - means ** 2
    zero = 1.0 - values.getnnz(axis=0) / values.shape[0]
    means = np.clip(means, 1e-10, None)
    omega = np.clip(variances / means, 1e-10, None)
    zero = np.clip(zero, 1e-10, 1.0 - 1e-10)
    return np.column_stack([np.log(means), np.log(omega), np.log(zero / (1.0 - zero))])

def _matrix_equal(left, right) -> bool:
    if sp.issparse(left) or sp.issparse(right):
        return bool((sp.csr_matrix(left) != sp.csr_matrix(right)).nnz == 0)
    return bool(np.array_equal(left, right))

def _verify_feast_environment(config: dict) -> dict[str, object]:
    import FEAST
    version = str(FEAST.__version__)
    if version != str(config['feast']['version']):
        raise RuntimeError(f"installed FEAST version is {version}; expected {config['feast']['version']}")
    if Path(FEAST.__file__).resolve() != config['_feast_source_repo'] / 'src/FEAST/__init__.py':
        raise RuntimeError(f'unexpected FEAST import: {FEAST.__file__}')
    if np.__version__ != '1.26.4':
        raise RuntimeError(f'simulation NumPy is {np.__version__}; expected 1.26.4')
    if Path(sys.executable).resolve() != config['_feast_python']:
        raise RuntimeError(f"prepare must use {config['_feast_python']}; observed {Path(sys.executable).resolve()}")
    pip_check = subprocess.run([str(config['_feast_python']), '-m', 'pip', 'check'], capture_output=True, text=True)
    if pip_check.returncode != 0:
        raise RuntimeError(f'FEAST environment failed pip check: {pip_check.stdout}{pip_check.stderr}')
    return {'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()), 'numpy': np.__version__, 'anndata': ad.__version__, 'scanpy': sc.__version__, 'feast': version, 'feast_import_path': str(Path(FEAST.__file__).resolve()), 'pip_check': 'ok'}

def _validate_raw_adata(adata: ad.AnnData, *, slice_id: str, config: dict) -> None:
    workflow.validate_count_matrix(adata.X, f'raw {slice_id} X')
    if 'counts' not in adata.layers or not _matrix_equal(adata.X, adata.layers['counts']):
        raise ValueError(f'raw {slice_id} X and counts layer differ')
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f'raw {slice_id} identifiers are not unique')
    for key in (config['inputs']['layer_key'], 'donor_id', 'sample_id'):
        if key not in adata.obs:
            raise ValueError(f'raw {slice_id} lacks obs[{key!r}]')
    if config['inputs']['spatial_key'] not in adata.obsm:
        raise ValueError(f'raw {slice_id} lacks spatial coordinates')
    if set(adata.obs['sample_id'].astype(str)) != {slice_id}:
        raise ValueError(f'raw {slice_id} sample identity changed')
    if set(adata.obs['donor_id'].astype(str)) != {config['inputs']['donor_id']}:
        raise ValueError(f'raw {slice_id} donor identity changed')
    if not np.isfinite(np.asarray(adata.obsm[config['inputs']['spatial_key']])).all():
        raise ValueError(f'raw {slice_id} spatial coordinates are non-finite')

def _build_panel(reference: ad.AnnData, query: ad.AnnData, *, config: dict, output_root: Path, raw_paths: tuple[Path, Path]) -> tuple[list[str], list[Path]]:
    panel_root = output_root / 'panel'
    panel_root.mkdir()
    common = reference.var_names.intersection(query.var_names, sort=False)
    if len(common) != 17924:
        raise ValueError(f'common-gene support changed: {len(common)}')
    selected = reference[:, common].copy()
    sc.pp.highly_variable_genes(selected, n_top_genes=int(config['fixed_panel']['n_genes']), flavor=str(config['fixed_panel']['flavor']), layer=str(config['fixed_panel']['source_layer']), inplace=True)
    mask = selected.var['highly_variable'].astype(bool).to_numpy()
    indices = np.flatnonzero(mask)
    if len(indices) != int(config['fixed_panel']['n_genes']):
        raise RuntimeError('fixed panel selection did not return exactly 2,000 genes')
    genes = selected.var_names[indices].astype(str).tolist()
    panel = pd.DataFrame({'panel_order': np.arange(len(genes), dtype=int), 'gene': genes, 'source_common_var_index': indices, 'seurat_v3_rank': selected.var.iloc[indices]['highly_variable_rank'].to_numpy()})
    panel_path = panel_root / 'panel.csv'
    genes_path = panel_root / 'genes.txt'
    panel.to_csv(panel_path, index=False)
    genes_path.write_text('\n'.join(genes) + '\n', encoding='utf-8')
    provenance = {'schema_version': 1, 'configuration_id': f"{config['configuration_id']}__fixed_panel", 'selection': {'source_slice': str(config['inputs']['reference_slice']), 'source_layer': config['fixed_panel']['source_layer'], 'common_gene_count': len(common), 'n_genes': len(genes), 'flavor': config['fixed_panel']['flavor']}, 'inputs': [{'path': str(path.resolve())} for path in raw_paths], 'outputs': [{'path': str(panel_path.resolve())}, {'path': str(genes_path.resolve())}]}
    provenance_path = panel_root / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return (genes, [panel_path, genes_path, provenance_path])

def _input_embedding(reference: ad.AnnData, query: ad.AnnData, genes: list[str], *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = sp.csr_matrix(reference[:, genes].X, dtype=np.float64)
    right = sp.csr_matrix(query[:, genes].X, dtype=np.float64)
    matrix = sp.vstack([left, right], format='csr')
    libraries = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.zeros_like(libraries, dtype=np.float64)
    nonzero = libraries > 0
    scale[nonzero] = 10000.0 / libraries[nonzero]
    matrix = matrix.multiply(scale[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    embedding = PCA(n_components=10, svd_solver='randomized', random_state=seed).fit_transform(matrix.toarray())
    batch = np.array(['ref'] * reference.n_obs + ['query'] * query.n_obs)
    layer_key = 'ground_truth'
    layers = np.concatenate([reference.obs[layer_key].astype(str).to_numpy(), query.obs[layer_key].astype(str).to_numpy()])
    source_ids = np.concatenate([reference.obs_names.astype(str), query.obs_names.astype(str)])
    spatial = np.vstack([reference.obsm['spatial'], query.obsm['spatial']])
    return (embedding, batch, layers, source_ids, spatial)

def _layer_pseudobulk_correlation(raw_query: ad.AnnData, candidate: ad.AnnData, genes: list[str]) -> float:
    layers = sorted(set(raw_query.obs['ground_truth'].astype(str)))
    values = []
    for layer in layers:
        raw_keep = raw_query.obs['ground_truth'].astype(str).to_numpy() == layer
        candidate_keep = candidate.obs['ground_truth'].astype(str).to_numpy() == layer
        raw_mean = np.asarray(raw_query[raw_keep, genes].X.mean(axis=0)).ravel()
        candidate_mean = np.asarray(candidate[candidate_keep, genes].X.mean(axis=0)).ravel()
        values.append(float(np.corrcoef(raw_mean, candidate_mean)[0, 1]))
    return float(np.mean(values))

def prepare(config: dict, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f'refusing to reuse output root: {output_root}')
    runtime = _verify_feast_environment(config)
    manifest = config['_input_manifest']
    raw_paths = pd.read_csv(manifest, dtype={'slice_id': str}).set_index('slice_id')['relative_path']
    reference_path = (manifest.parent / raw_paths[config['inputs']['reference_slice']]).resolve()
    query_path = (manifest.parent / raw_paths[config['inputs']['query_slice']]).resolve()
    reference_source = sc.read_h5ad(reference_path)
    query_source = sc.read_h5ad(query_path)
    reference_id = str(config['inputs']['reference_slice'])
    query_id = str(config['inputs']['query_slice'])
    _validate_raw_adata(reference_source, slice_id=reference_id, config=config)
    _validate_raw_adata(query_source, slice_id=query_id, config=config)
    reference = workflow.prefix_spot_ids(reference_source, reference_id)
    query = workflow.prefix_spot_ids(query_source, query_id)
    if len(set(reference_source.obs_names) & set(query_source.obs_names)) != 3036:
        raise ValueError('raw Visium barcode-overlap audit changed')
    output_root.mkdir(parents=True)
    input_root = output_root / 'inputs'
    input_root.mkdir()
    reference_prepared = input_root / 'reference_151675_raw.h5ad'
    query_prepared = input_root / 'query_151676_raw.h5ad'
    reference.write_h5ad(reference_prepared, compression='gzip')
    query.write_h5ad(query_prepared, compression='gzip')
    genes, panel_outputs = _build_panel(reference, query, config=config, output_root=output_root, raw_paths=(reference_path, query_path))
    import FEAST
    simulation_paths: dict[str, Path] = {'raw': query_prepared}
    simulation_records = []
    simulation_outputs: list[Path] = []
    D = np.asarray(config['deformation']['D'], dtype=float)
    b = np.asarray(config['deformation']['b'], dtype=float)
    for row in config['conditions']:
        condition = str(row['name'])
        if row.get('alpha') is None:
            continue
        alpha = float(row['alpha'])
        simulated = FEAST.simulate_batch_effect(query, D=D, b=b, alpha=alpha, random_seed=int(config['feast']['simulation_seed']))
        workflow.validate_count_matrix(simulated.X, condition)
        if not simulated.obs_names.equals(query.obs_names):
            raise RuntimeError(f'{condition} changed query spot identity/order')
        if not simulated.var_names.equals(query.var_names):
            raise RuntimeError(f'{condition} changed query gene identity/order')
        if not np.array_equal(simulated.obsm['spatial'], query.obsm['spatial']):
            raise RuntimeError(f'{condition} changed query spatial coordinates')
        if not simulated.obs['ground_truth'].astype(str).equals(query.obs['ground_truth'].astype(str)):
            raise RuntimeError(f'{condition} changed query layer annotations')
        simulated.uns['study04b'] = {'configuration_id': config['configuration_id'], 'condition': condition, 'role': row['role'], 'alpha': alpha, 'alpha_stratum': 'interpolation' if alpha <= 1 else 'extrapolation', 'simulation_seed': int(config['feast']['simulation_seed']), 'deformation_name': config['deformation']['name']}
        path = input_root / f'query_151676_{condition}.h5ad'
        simulated.write_h5ad(path, compression='gzip')
        simulation_paths[condition] = path
        simulation_outputs.append(path)
        simulation_records.append({'condition': condition, 'alpha': alpha, 'role': row['role'], 'path': path.relative_to(output_root).as_posix(), 'n_spots': simulated.n_obs, 'n_genes': simulated.n_vars})
    simulation_manifest = input_root / 'simulation_manifest.csv'
    pd.DataFrame(simulation_records).to_csv(simulation_manifest, index=False)
    rows = []
    for condition in config['conditions']:
        name = str(condition['name'])
        query_condition_path = simulation_paths[name]
        rows.append({'condition': name, 'alpha': condition.get('alpha'), 'role': condition['role'], 'reference_path': reference_prepared.relative_to(output_root).as_posix(), 'query_path': query_condition_path.relative_to(output_root).as_posix(), 'n_reference': reference.n_obs, 'n_query': query.n_obs})
    condition_manifest = input_root / 'condition_manifest.csv'
    pd.DataFrame(rows).to_csv(condition_manifest, index=False)
    baseline_root = output_root / 'baselines'
    baseline_root.mkdir()
    baseline_rows = []
    query_objects: dict[str, ad.AnnData] = {'raw': query}
    for condition in config['conditions']:
        name = str(condition['name'])
        candidate_query = query if name == 'raw' else sc.read_h5ad(simulation_paths[name])
        query_objects[name] = candidate_query
        embedding, batch, layers, source_ids, spatial = _input_embedding(reference, candidate_query, genes, seed=int(config['analysis']['scoring_seed']))
        path = baseline_root / f'{name}.npz'
        np.savez_compressed(path, embedding=embedding, batch_labels=batch, layer_labels=layers, source_spot_ids=source_ids, spatial=spatial, n_reference=np.array(reference.n_obs), n_query=np.array(candidate_query.n_obs))
        row = {'condition': name, 'path': path.relative_to(output_root).as_posix(), 'n_reference': reference.n_obs, 'n_query': candidate_query.n_obs, 'fixed_panel_zero_reference_rows': int(np.count_nonzero(np.asarray(reference[:, genes].X.sum(axis=1)).ravel() == 0)), 'fixed_panel_zero_query_rows': int(np.count_nonzero(np.asarray(candidate_query[:, genes].X.sum(axis=1)).ravel() == 0)), **metrics.full_metric_row(embedding, baseline=embedding, spatial=spatial, batch=batch, layers=layers, k=int(config['analysis']['n_neighbors']), seed=int(config['analysis']['scoring_seed']))}
        row['layer_pseudobulk_correlation_to_raw_query'] = 1.0 if name == 'raw' else _layer_pseudobulk_correlation(query, candidate_query, genes)
        raw_libraries = np.asarray(query[:, genes].X.sum(axis=1)).ravel()
        candidate_libraries = np.asarray(candidate_query[:, genes].X.sum(axis=1)).ravel()
        row['query_library_size_spearman_to_raw'] = float(spearmanr(raw_libraries, candidate_libraries).statistic)
        baseline_rows.append(row)
    baseline_manifest = baseline_root / 'manifest.csv'
    pd.DataFrame(baseline_rows).to_csv(baseline_manifest, index=False)
    raw_theta = _gene_theta(query[:, genes].X)
    sim0_theta = _gene_theta(query_objects['sim_0.00'][:, genes].X)
    sim1_theta = _gene_theta(query_objects['sim_1.00'][:, genes].X)
    resampling_rmse = np.sqrt(np.mean((raw_theta - sim0_theta) ** 2, axis=0))
    added_rmse = np.sqrt(np.mean((sim0_theta - sim1_theta) ** 2, axis=0))
    if not np.all(added_rmse > resampling_rmse):
        raise RuntimeError(f'added alpha-1 deformation is not distinguishable from alpha-0 resampling in every theta dimension: resampling={resampling_rmse}, added={added_rmse}')
    baseline_table = pd.DataFrame(baseline_rows).set_index('condition')
    if not baseline_table.loc['sim_1.00', 'conditional_mmd2_macro'] > baseline_table.loc['sim_0.00', 'conditional_mmd2_macro']:
        raise RuntimeError('uncorrected conditional MMD did not increase at alpha 1')
    if baseline_table.loc['sim_1.00', 'layer_pseudobulk_correlation_to_raw_query'] < 0.9:
        raise RuntimeError('alpha-1 query layer pseudobulk correlation fell below 0.9')
    qualification = {'configuration_id': config['configuration_id'], 'status': 'passed', 'theta_dimensions': ['log_mu', 'log_omega', 'logit_pi0'], 'raw_to_sim0_theta_rmse': resampling_rmse.tolist(), 'sim0_to_sim1_theta_rmse': added_rmse.tolist(), 'added_deformation_exceeds_resampling_each_dimension': True, 'sim1_conditional_mmd_exceeds_sim0': True, 'sim1_layer_pseudobulk_correlation_to_raw_query': float(baseline_table.loc['sim_1.00', 'layer_pseudobulk_correlation_to_raw_query']), 'identity_geometry_labels_exact': True, 'cross_section_spot_pairing_declared': False}
    qualification_path = input_root / 'qualification.json'
    qualification_path.write_text(json.dumps(qualification, indent=2) + '\n', encoding='utf-8')
    outputs = [reference_prepared, query_prepared, simulation_manifest, condition_manifest, baseline_manifest, qualification_path, *simulation_outputs, *panel_outputs, *[output_root / row['path'] for row in baseline_rows]]
    provenance = {'schema_version': 1, 'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime, 'inputs': [{'path': str(path.resolve())} for path in (reference_path, query_path, config['_input_manifest'])], 'sources': [{'path': str(path.resolve())} for path in (Path(__file__), STUDY_ROOT / 'workflow.py', STUDY_ROOT / 'metrics.py', config['_config_path'])], 'outputs': [{'path': str(path.resolve())} for path in outputs], 'diagnostics': {'status': 'completed', 'condition_count': len(rows), 'production_job_count': len(workflow.production_cells(config)), 'common_gene_count': 17924, 'fixed_panel_gene_count': len(genes), 'raw_barcode_overlap_not_spot_pairing': 3036, 'qualification': qualification}}
    provenance_path = output_root / 'prepare_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'prepared {len(rows)} conditions and {len(genes)} fixed genes at {output_root}')

def _prepared_manifest_row(manifest: pd.DataFrame, condition: str) -> tuple[Path, Path, float]:
    row = manifest[manifest['condition'] == condition]
    if len(row) != 1:
        raise ValueError(f'missing unique condition: {condition}')
    item = row.iloc[0]
    alpha = 0.0 if pd.isna(item.alpha) else float(item.alpha)
    reference = Path(item.reference_path)
    query = Path(item.query_path)
    return (reference, query, alpha)

def run_methods(config: dict, output_root: Path, *, selected_method: str | None, selected_condition: str | None, selected_seed: int | None) -> int:
    prepare_provenance = output_root / 'prepare_provenance.json'
    if not prepare_provenance.is_file():
        raise FileNotFoundError('prepare stage has not completed')
    qualification = json.loads((output_root / 'inputs' / 'qualification.json').read_text(encoding='utf-8'))
    if qualification.get('status') != 'passed':
        raise RuntimeError('prepared-input qualification did not pass')
    manifest = workflow.read_condition_manifest(output_root)
    panel_path = output_root / 'panel' / 'panel.csv'
    panel_provenance = output_root / 'panel' / 'provenance.json'
    pinned_feast_repo = config['_feast_source_repo']
    cells = workflow.production_cells(config)
    if selected_method is not None:
        cells = [cell for cell in cells if cell[1] == selected_method]
    if selected_condition is not None:
        cells = [cell for cell in cells if cell[0] == selected_condition]
    if selected_seed is not None:
        cells = [cell for cell in cells if cell[2] == selected_seed]
    if not cells:
        raise ValueError('method selection produced no jobs')
    pip_status = {}
    for method in sorted({cell[1] for cell in cells}):
        method_config = config['methods'][method]
        python = workflow.resolve_path(config['_config_path'], method_config['python'])
        result = subprocess.run([str(python), '-m', 'pip', 'check'], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'{method} environment failed pip check: {result.stdout}{result.stderr}')
        pip_status[method] = 'ok'
    run_rows = []
    failures = 0
    for condition, method, seed in cells:
        reference, query, alpha = _prepared_manifest_row(manifest, condition)
        candidate = workflow.candidate_path(output_root, condition, method, seed)
        expected_id = workflow.candidate_id(config, condition, method, seed)
        if candidate.exists():
            valid, reason = workflow.candidate_is_valid(candidate, config=config, output_root=output_root, condition=condition, method=method, seed=seed)
            if valid:
                print(f'skip verified {method}/{condition}/seed_{seed}', flush=True)
                run_rows.append({'condition': condition, 'method': method, 'method_seed': seed, 'status': 'skipped_verified', 'candidate_id': expected_id, 'output_dir': candidate.relative_to(output_root).as_posix()})
                continue
            archived = workflow.archive_invalid(candidate, output_root / 'failures', reason)
            print(f'archived invalid attempt: {archived}')
        method_config = config['methods'][method]
        python = workflow.resolve_path(config['_config_path'], method_config['python'])
        script = workflow.resolve_path(config['_config_path'], method_config['script'])
        command = [str(python), str(script), '--reference', str(reference), '--query', str(query), '--output-dir', str(candidate), '--panel-file', str(panel_path), '--panel-provenance', str(panel_provenance), '--seed', str(seed), '--config-id', expected_id, '--parent-config-id', str(config['configuration_id']), '--deformation-mode', condition, '--alpha', str(alpha), '--config-path', str(config['_config_path'])]
        for name, value in method_config.get('kwargs', {}).items():
            command.extend([f"--{name.replace('_', '-')}", workflow.format_cli_value(value)])
        environment = os.environ.copy()
        environment.update({'PYTHONNOUSERSITE': '1', 'PYTHONHASHSEED': str(seed), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONIOENCODING': 'utf-8', 'LC_ALL': 'C.UTF-8', 'FEAST_REPO': str(pinned_feast_repo), 'FEAST_REPRODUCE_PIP_CHECK': pip_status[method], 'MPLCONFIGDIR': '/tmp/matplotlib', 'XDG_CACHE_HOME': '/tmp'})
        for key, value in method_config.get('environment', {}).items():
            environment[str(key)] = str(workflow.resolve_path(config['_config_path'], value))
        candidate.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_root / 'logs' / method / condition / f'seed_{seed}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'run {method}/{condition}/seed_{seed}', flush=True)
        with log_path.open('w', encoding='utf-8') as log_handle:
            log_handle.write(shlex.join(command) + '\n')
            log_handle.flush()
            result = subprocess.run(command, cwd=STUDY_ROOT.parent, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        valid, reason = workflow.candidate_is_valid(candidate, config=config, output_root=output_root, condition=condition, method=method, seed=seed)
        if result.returncode != 0 or not valid:
            failures += 1
            reason = f'exit={result.returncode}; {reason}'
            archived = workflow.archive_invalid(candidate, output_root / 'failures', reason)
            status = 'failed'
            output_value = archived.relative_to(output_root).as_posix()
            print(f'FAILED {method}/{condition}/seed_{seed}: {reason}')
        else:
            status = 'complete'
            output_value = candidate.relative_to(output_root).as_posix()
        run_rows.append({'condition': condition, 'method': method, 'method_seed': seed, 'status': status, 'candidate_id': expected_id, 'output_dir': output_value, 'log': log_path.relative_to(output_root).as_posix()})
    run_manifest = output_root / 'method_run_manifest.csv'
    current = pd.DataFrame(run_rows)
    if run_manifest.is_file():
        previous = pd.read_csv(run_manifest)
        keys = set(zip(current['condition'], current['method'], current['method_seed'].astype(int)))
        previous = previous[~previous.apply(lambda row: (row.condition, row.method, int(row.method_seed)) in keys, axis=1)]
        current = pd.concat([previous, current], ignore_index=True)
    current.sort_values(['method', 'condition', 'method_seed']).to_csv(run_manifest, index=False)
    return failures

def print_plan(config: dict, output_root: Path) -> None:
    print(f"configuration: {config['configuration_id']}")
    print(f'output: {output_root}')
    print('reference: raw 151675')
    print('query conditions: raw, sim_0.00, sim_0.50, sim_1.00, sim_1.50')
    print('methods: GraphST, STAMP, scVI')
    print('method seeds: 42, 43, 44')
    print(f'production jobs: {len(workflow.production_cells(config))}')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_ROOT / 'config.yaml')
    parser.add_argument('--stage', choices=('prepare', 'methods', 'all'), default='all')
    parser.add_argument('--method', choices=tuple(workflow.METHOD_ARTIFACTS), default=None)
    parser.add_argument('--condition', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    output_root = args.output_dir.expanduser().resolve() if args.output_dir is not None else config['_output_dir']
    if args.condition is not None and args.condition not in workflow.condition_lookup(config):
        parser.error(f'unknown condition: {args.condition}')
    if args.seed is not None and args.seed not in config['analysis']['method_seeds']:
        parser.error(f"seed must be one of {config['analysis']['method_seeds']}")
    if args.dry_run:
        print_plan(config, output_root)
        return 0
    if args.stage in {'prepare', 'all'}:
        prepare(config, output_root)
    if args.stage in {'methods', 'all'}:
        os.environ['FEAST_VERSION'] = str(_verify_feast_environment(config)['feast'])
        failures = run_methods(config, output_root, selected_method=args.method, selected_condition=args.condition, selected_seed=args.seed)
        return 1 if failures else 0
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
