"""Validate one or all fresh Study 05 candidates and optional scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from workflow import Job, STUDY_DIR, dataset_config, declared_jobs, direction_id, expected_job_path, input_artifact_id, inspect_inputs, job_id, prepare_job, require_declared_job, validate_generated
DLPFC_DONOR_BY_SLICE = {'151670': 'Br5595', '151675': 'Br8100', '151676': 'Br8100'}
STRATIFICATION_CONTRACT = {'dlpfc_donor_by_slice': DLPFC_DONOR_BY_SLICE, 'mask_half': 'within_slice', 'merfish_cross_slice': 'donor_not_declared'}

def donor_stratum(job: Job) -> str:
    if job.mode == 'mask_half':
        return 'within_slice'
    if job.dataset != 'dlpfc':
        return 'donor_not_declared'
    return 'within_donor' if DLPFC_DONOR_BY_SLICE[job.source] == DLPFC_DONOR_BY_SLICE[job.target] else 'cross_donor'

def _safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    if int(finite.sum()) < 2:
        return float('nan')
    left = first[finite] - np.mean(first[finite])
    right = second[finite] - np.mean(second[finite])
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator > 0.0 else float('nan')

def recompute_panel_summary(per_gene: pd.DataFrame) -> dict[str, float]:
    """Independently reconstruct the six declared metrics from per-gene rows."""
    return {'mean_corr': _safe_pearson(np.log1p(per_gene['generated_mean']), np.log1p(per_gene['target_mean'])), 'var_corr': _safe_pearson(per_gene['generated_variance'], per_gene['target_variance']), 'moran_corr': _safe_pearson(per_gene['generated_moran_i'], per_gene['target_moran_i']), 'zero_ks': float(ks_2samp(per_gene['generated_zero_prop'], per_gene['target_zero_prop']).statistic), 'median_gene_pearson': float(np.nanmedian(per_gene['pearson'])), 'median_gene_spearman': float(np.nanmedian(per_gene['spearman']))}

def _single_job(args: argparse.Namespace, config: dict) -> Job:
    if args.mode == 'cross_slice':
        if args.source is None or args.target is None or args.assignment_randomness is None:
            raise ValueError('cross_slice requires --source, --target, and --assignment-randomness')
        if args.slice is not None:
            raise ValueError('cross_slice does not accept --slice')
        job = Job('cross_slice', str(args.dataset), str(args.source), str(args.target), float(args.assignment_randomness))
    elif args.mode == 'mask_half':
        if args.slice is None or args.source is not None or args.target is not None:
            raise ValueError('mask_half requires --slice and does not accept source/target')
        randomness = float(config['mask_half']['assignment_randomness'])
        if args.assignment_randomness is not None and (not np.isclose(float(args.assignment_randomness), randomness, rtol=0.0, atol=1e-12)):
            raise ValueError('mask_half assignment randomness differs from the frozen value')
        job = Job('mask_half', str(args.dataset), str(args.slice), str(args.slice), randomness)
    else:
        raise ValueError('provide --all or a declared --mode')
    require_declared_job(config, job)
    return job

def jobs_from_args(args: argparse.Namespace, config: dict) -> list[Job]:
    if args.all:
        if any((value is not None for value in (args.mode, args.dataset, args.source, args.target, args.slice, args.assignment_randomness))):
            raise ValueError('--all cannot be combined with a single-job selector')
        return declared_jobs(config)
    if args.dataset is None:
        raise ValueError('single-job validation requires --dataset')
    return [_single_job(args, config)]

def validate_job(directory: Path, job: Job, config: dict, paths: dict[tuple[str, str], Path], panels: dict[str, list[str]]) -> dict:
    expected_files = {'generated.h5ad', 'transport_diagnostics.csv', 'provenance.json'}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected_files:
        raise RuntimeError(f'candidate artifact set is incomplete: {directory}')
    provenance = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    generated_path = directory / 'generated.h5ad'
    diagnostics_path = directory / 'transport_diagnostics.csv'
    expected_id = job_id(config, job)
    if provenance.get('status') != 'ok' or provenance.get('configuration_id') != expected_id:
        raise RuntimeError(f'provenance identity mismatch: {directory}')
    expected_scalars = {'study_configuration_id': config['configuration_id'], 'mode': job.mode, 'dataset': job.dataset, 'annotation_key': dataset_config(config, job.dataset)['annotation_key'], 'feast_version': str(config['required_feast_version']), 'source': job.source, 'target': job.target, 'target_expression_policy': config['target_expression_policy']}
    for key, value in expected_scalars.items():
        if provenance.get(key) != value:
            raise RuntimeError(f'provenance field mismatch for {key}: {directory}')
    if int(provenance.get('public_seed', -1)) != int(config['public_seed']):
        raise RuntimeError('public seed mismatch')
    if not np.isclose(float(provenance.get('assignment_randomness', np.nan)), job.assignment_randomness):
        raise RuntimeError('assignment-randomness mismatch')
    if provenance.get('transport') != config['transport']:
        raise RuntimeError('transport configuration provenance mismatch')
    if provenance.get('target_expression_loaded_by_generation') is not False:
        raise RuntimeError('generation provenance does not deny target-expression loading')
    if provenance.get('identity_verified_before_obs_name_rebinding') is not True:
        raise RuntimeError('generation identity was not verified before obs-name rebinding')
    numerical_change = provenance.get('numerical_change', {})
    if not (numerical_change.get('field') == 'sinkhorn_iter' and int(numerical_change.get('historical', -1)) == 200 and (int(numerical_change.get('fresh', -1)) == int(config['transport']['sinkhorn_iter'])) and (numerical_change.get('classification') == 'declared_solver_configuration_change')):
        raise RuntimeError('declared solver-change provenance mismatch')
    prepared = prepare_job(config, paths, panels, job)
    target = prepared.target_contract
    genes = prepared.genes
    support = prepared.support
    if provenance.get('support') != support:
        raise RuntimeError('source-only support provenance mismatch')
    generated = ad.read_h5ad(generated_path)
    records = validate_generated(generated, target, genes, config, job.dataset)
    diagnostics = pd.read_csv(diagnostics_path, dtype={'label': str})
    if len(diagnostics) != len(records) or set(diagnostics['label']) != {str(row['label']) for row in records}:
        raise RuntimeError('standalone transport diagnostics differ from embedded support')
    embedded = generated.uns.get('feast_reproduce', {})
    if not (embedded.get('configuration_id') == expected_id and embedded.get('study_configuration_id') == config['configuration_id'] and (embedded.get('mode') == job.mode) and (embedded.get('dataset') == job.dataset) and (embedded.get('annotation_key') == dataset_config(config, job.dataset)['annotation_key']) and (int(embedded.get('public_seed', -1)) == int(config['public_seed'])) and (embedded.get('target_expression_policy') == config['target_expression_policy']) and (not bool(embedded.get('target_expression_loaded_by_generation', True))) and bool(embedded.get('identity_verified_before_obs_name_rebinding', False)) and (int(embedded.get('positive_transport_records', -1)) == len(records))):
        raise RuntimeError(f'embedded H5AD provenance mismatch: {directory}')
    return {'configuration_id': expected_id, 'mode': job.mode, 'dataset': job.dataset, 'direction': direction_id(job), 'source': job.source, 'target': job.target, 'assignment_randomness': job.assignment_randomness, 'primary_setting': bool(np.isclose(job.assignment_randomness, float(config['primary_assignment_randomness']))), 'target_candidate_spots': support['target_candidate_spots'], 'target_retained_spots': support['target_retained_spots'], 'target_retained_fraction': support['target_retained_fraction'], 'reference_zero_gene_count': support['reference_zero_gene_count'], 'positive_transport_records': len(records), 'valid': True}
SUMMARY_COLUMNS = {'job_id', 'mode', 'dataset', 'direction', 'source', 'target', 'assignment_randomness', 'primary_setting', 'donor_stratum', 'n_target_spots', 'target_retained_fraction', 'n_genes', 'n_reference_observed_genes', 'mean_corr', 'var_corr', 'moran_corr', 'zero_ks', 'median_gene_pearson', 'median_gene_spearman', 'observed_mean_corr', 'observed_var_corr', 'observed_moran_corr', 'observed_zero_ks', 'observed_median_gene_pearson', 'observed_median_gene_spearman', 'decode_method', 'seed'}
GENE_COLUMNS = {'job_id', 'mode', 'dataset', 'direction', 'source', 'target', 'assignment_randomness', 'primary_setting', 'donor_stratum', 'reference_observed', 'gene', 'pearson', 'spearman', 'generated_mean', 'target_mean', 'generated_variance', 'target_variance', 'generated_zero_prop', 'target_zero_prop', 'generated_moran_i', 'target_moran_i'}

def validate_score_root(score_dir: Path, generation_dir: Path, config: dict, paths: dict[tuple[str, str], Path], panels: dict[str, list[str]]) -> dict:
    expected_files = {'per_gene_metrics.csv.gz', 'summary.csv', 'support_audit.csv', 'provenance.json'}
    if not score_dir.is_dir() or {path.name for path in score_dir.iterdir()} != expected_files:
        raise RuntimeError('score artifact set is incomplete')
    provenance = json.loads((score_dir / 'provenance.json').read_text(encoding='utf-8'))
    expected_panels = {dataset: {'n_genes': len(genes)} for dataset, genes in panels.items()}
    if not (provenance.get('status') == 'ok' and provenance.get('configuration_id') == config['scoring']['configuration_id'] and (provenance.get('study_configuration_id') == config['configuration_id']) and (provenance.get('feast_version') == str(config['required_feast_version'])) and (int(provenance.get('public_seed', -1)) == int(config['public_seed'])) and (provenance.get('target_expression_policy') == config['target_expression_policy']) and (provenance.get('ordered_gene_panels') == expected_panels) and (provenance.get('metric_contract') == config['scoring']) and (provenance.get('stratification_contract') == STRATIFICATION_CONTRACT)):
        raise RuntimeError('score provenance contract mismatch')
    if not int(provenance.get('generation_validation', {}).get('validated_jobs', -1)) == int(config['expected_jobs']):
        raise RuntimeError('score source/validation binding mismatch')
    jobs = declared_jobs(config)
    expected_generation_keys = {job_id(config, job) for job in jobs}
    generation_inputs = provenance.get('generation_inputs', {})
    if set(generation_inputs) != expected_generation_keys:
        raise RuntimeError('score provenance lacks the exact 45 generation inputs')
    summary = pd.read_csv(score_dir / 'summary.csv', dtype={'source': str, 'target': str})
    per_gene = pd.read_csv(score_dir / 'per_gene_metrics.csv.gz', dtype={'source': str, 'target': str, 'gene': str})
    support = pd.read_csv(score_dir / 'support_audit.csv', dtype={'source': str, 'target': str})
    expected_ids = {job_id(config, job) for job in jobs}
    expected_gene_rows = sum((len(panels[job.dataset]) for job in jobs))
    if not (set(summary.columns) == SUMMARY_COLUMNS and len(summary) == int(config['expected_jobs']) and summary['job_id'].is_unique and (set(summary['job_id']) == expected_ids) and (set(per_gene.columns) == GENE_COLUMNS) and (len(per_gene) == expected_gene_rows) and (len(support) == int(config['expected_jobs'])) and support['job_id'].is_unique and (set(support['job_id']) == expected_ids)):
        raise RuntimeError('score tables do not have the declared schema/coverage')
    summary_metrics = ['mean_corr', 'var_corr', 'moran_corr', 'zero_ks', 'median_gene_pearson', 'median_gene_spearman', 'observed_mean_corr', 'observed_var_corr', 'observed_moran_corr', 'observed_zero_ks', 'observed_median_gene_pearson', 'observed_median_gene_spearman']
    if not np.isfinite(summary[summary_metrics].to_numpy(dtype=float)).all():
        raise RuntimeError('score summary contains non-finite headline metrics')
    for job in jobs:
        rows = per_gene[per_gene['job_id'] == job_id(config, job)]
        if rows['gene'].tolist() != panels[job.dataset]:
            raise RuntimeError(f'per-gene score support/order mismatch: {job}')
        provenance_path = expected_job_path(generation_dir, job) / 'provenance.json'
        generation_provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        support_row = support[support['job_id'] == job_id(config, job)].iloc[0]
        summary_row = summary[summary['job_id'] == job_id(config, job)].iloc[0]
        expected_stratum = donor_stratum(job)
        if not (str(summary_row['donor_stratum']) == expected_stratum and set(rows['donor_stratum'].astype(str)) == {expected_stratum} and (str(support_row['donor_stratum']) == expected_stratum)):
            raise RuntimeError(f'score donor stratum mismatch: {job}')
        expected_observed = int(generation_provenance['support']['reference_observed_gene_count'])
        if int(rows['reference_observed'].sum()) != expected_observed:
            raise RuntimeError(f'reference-observed score support mismatch: {job}')
        if not (int(summary_row['n_genes']) == len(panels[job.dataset]) and int(summary_row['n_reference_observed_genes']) == expected_observed and (int(summary_row['seed']) == int(config['public_seed']))):
            raise RuntimeError(f'score summary denominator mismatch: {job}')
        recomputed = recompute_panel_summary(rows)
        observed = recompute_panel_summary(rows[rows['reference_observed']])
        for metric, value in recomputed.items():
            if not np.isclose(float(summary_row[metric]), value, rtol=1e-12, atol=1e-12):
                raise RuntimeError(f'full-panel summary recomputation mismatch: {job}/{metric}')
        for metric, value in observed.items():
            if not np.isclose(float(summary_row[f'observed_{metric}']), value, rtol=1e-12, atol=1e-12):
                raise RuntimeError(f'reference-observed summary recomputation mismatch: {job}/{metric}')
    return {'configuration_id': config['scoring']['configuration_id'], 'summary_rows': len(summary), 'per_gene_rows': len(per_gene), 'support_rows': len(support), 'valid': True}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--mode', choices=('cross_slice', 'mask_half'))
    parser.add_argument('--dataset', choices=('dlpfc', 'merfish'))
    parser.add_argument('--source')
    parser.add_argument('--target')
    parser.add_argument('--slice')
    parser.add_argument('--assignment-randomness', type=float)
    parser.add_argument('--config', type=Path, default=STUDY_DIR / 'config.yaml')
    parser.add_argument('--manifest', type=Path, default=STUDY_DIR / 'data' / 'input_files.csv')
    parser.add_argument('--input-dir', type=Path, default=STUDY_DIR / 'data' / 'local')
    parser.add_argument('--output-dir', type=Path, default=STUDY_DIR / 'outputs' / 'final')
    parser.add_argument('--scores-dir', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    config, paths, panels = inspect_inputs(args.config, args.manifest, args.input_dir)
    jobs = jobs_from_args(args, config)
    if args.all:
        expected_h5ads = {expected_job_path(args.output_dir, job) / 'generated.h5ad' for job in jobs}
        actual_h5ads = set(args.output_dir.rglob('generated.h5ad')) if args.output_dir.exists() else set()
        if actual_h5ads != expected_h5ads:
            raise RuntimeError('outputs/final does not contain the exact declared 45 H5ADs')
    rows = [validate_job(expected_job_path(args.output_dir, job), job, config, paths, panels) for job in jobs]
    score_result = None
    if args.scores_dir is not None:
        if not args.all:
            raise ValueError('--scores-dir requires --all generation validation')
        score_result = validate_score_root(args.scores_dir, args.output_dir, config, paths, panels)
    report = {'status': 'ok', 'configuration_id': config['configuration_id'], 'validated_jobs': len(rows), 'expected_full_matrix': int(config['expected_jobs']), 'jobs': rows, 'scores': score_result}
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
