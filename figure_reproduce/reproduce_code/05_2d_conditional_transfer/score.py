"""Score all 45 validated Study 05 candidates against held-out expression."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import ks_2samp, rankdata
from sklearn.neighbors import NearestNeighbors
from validate import validate_job, validate_score_root
from workflow import STUDY_DIR, declared_jobs, direction_id, expected_job_path, input_artifact_id, inspect_inputs, job_id
DLPFC_DONOR_BY_SLICE = {'151670': 'Br5595', '151675': 'Br8100', '151676': 'Br8100'}
STRATIFICATION_CONTRACT = {'dlpfc_donor_by_slice': DLPFC_DONOR_BY_SLICE, 'mask_half': 'within_slice', 'merfish_cross_slice': 'donor_not_declared'}

def donor_stratum(job) -> str:
    if job.mode == 'mask_half':
        return 'within_slice'
    if job.dataset != 'dlpfc':
        return 'donor_not_declared'
    source_donor = DLPFC_DONOR_BY_SLICE[job.source]
    target_donor = DLPFC_DONOR_BY_SLICE[job.target]
    return 'within_donor' if source_donor == target_donor else 'cross_donor'

def neighbor_indices(coords: np.ndarray, neighbors_including_self: int) -> list[np.ndarray]:
    """Reproduce the historical six-nearest-neighbor construction."""
    coordinates = np.asarray(coords, dtype=float)
    n_spots = coordinates.shape[0]
    if n_spots < 2:
        return []
    n_neighbors = min(int(neighbors_including_self), n_spots)
    raw = NearestNeighbors(n_neighbors=n_neighbors).fit(coordinates).kneighbors(coordinates, return_distance=False)
    return [row[row != index].astype(int, copy=False) for index, row in enumerate(raw)]

def moran_i(values: np.ndarray, neighbors: list[np.ndarray]) -> float:
    """Historical scalar Moran's I definition, retained for contract tests."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size < 3 or not neighbors:
        return float('nan')
    centered = vector - np.mean(vector)
    denominator = float(np.sum(centered * centered))
    if denominator <= 0.0:
        return float('nan')
    numerator = 0.0
    weight_sum = 0.0
    for index, adjacent in enumerate(neighbors):
        if len(adjacent) == 0:
            continue
        numerator += float(np.sum(centered[index] * centered[adjacent]))
        weight_sum += float(len(adjacent))
    if weight_sum <= 0.0:
        return float('nan')
    return float(vector.size / weight_sum * (numerator / denominator))

def _matrix_chunk(matrix: Any, start: int, stop: int) -> np.ndarray:
    chunk = matrix[:, start:stop]
    if sparse.issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk, dtype=np.float64)

def _pearson_columns(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_centered = first - np.mean(first, axis=0, keepdims=True)
    second_centered = second - np.mean(second, axis=0, keepdims=True)
    numerator = np.sum(first_centered * second_centered, axis=0)
    denominator = np.sqrt(np.sum(first_centered * first_centered, axis=0) * np.sum(second_centered * second_centered, axis=0))
    result = np.full(first.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result

def _moran_columns(matrix: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    if matrix.shape[0] < 3 or not neighbors:
        return np.full(matrix.shape[1], np.nan, dtype=np.float64)
    rows = np.concatenate([np.full(len(adjacent), index, dtype=np.int64) for index, adjacent in enumerate(neighbors)])
    columns = np.concatenate(neighbors).astype(np.int64, copy=False)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(centered[rows] * centered[columns], axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = matrix.shape[0] / float(len(rows)) * numerator[valid] / denominator[valid]
    return result

def compute_per_gene_metrics(generated: ad.AnnData, target: ad.AnnData, neighbors: list[np.ndarray], chunk_size: int) -> pd.DataFrame:
    generated_matrix = generated.layers['counts']
    target_matrix = target.layers['counts'] if 'counts' in target.layers else target.X
    columns: dict[str, np.ndarray] = {name: np.full(generated.n_vars, np.nan, dtype=np.float64) for name in ('pearson', 'spearman', 'generated_mean', 'target_mean', 'generated_variance', 'target_variance', 'generated_zero_prop', 'target_zero_prop', 'generated_moran_i', 'target_moran_i')}
    for start in range(0, generated.n_vars, int(chunk_size)):
        stop = min(start + int(chunk_size), generated.n_vars)
        simulated = _matrix_chunk(generated_matrix, start, stop)
        observed = _matrix_chunk(target_matrix, start, stop)
        columns['pearson'][start:stop] = _pearson_columns(simulated, observed)
        columns['spearman'][start:stop] = _pearson_columns(rankdata(simulated, method='average', axis=0), rankdata(observed, method='average', axis=0))
        columns['generated_mean'][start:stop] = np.mean(simulated, axis=0)
        columns['target_mean'][start:stop] = np.mean(observed, axis=0)
        columns['generated_variance'][start:stop] = np.var(simulated, axis=0, ddof=0)
        columns['target_variance'][start:stop] = np.var(observed, axis=0, ddof=0)
        columns['generated_zero_prop'][start:stop] = np.mean(simulated <= 0, axis=0)
        columns['target_zero_prop'][start:stop] = np.mean(observed <= 0, axis=0)
        columns['generated_moran_i'][start:stop] = _moran_columns(simulated, neighbors)
        columns['target_moran_i'][start:stop] = _moran_columns(observed, neighbors)
    return pd.DataFrame({'gene': generated.var_names.astype(str), **columns})

def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return float('nan')
    value = _pearson_columns(first[finite, None], second[finite, None])[0]
    return float(value)

def panel_summary(per_gene: pd.DataFrame) -> dict[str, float]:
    return {'mean_corr': safe_pearson(np.log1p(per_gene['generated_mean']), np.log1p(per_gene['target_mean'])), 'var_corr': safe_pearson(per_gene['generated_variance'], per_gene['target_variance']), 'moran_corr': safe_pearson(per_gene['generated_moran_i'], per_gene['target_moran_i']), 'zero_ks': float(ks_2samp(per_gene['generated_zero_prop'], per_gene['target_zero_prop']).statistic), 'median_gene_pearson': float(np.nanmedian(per_gene['pearson'])), 'median_gene_spearman': float(np.nanmedian(per_gene['spearman']))}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_DIR / 'config.yaml')
    parser.add_argument('--manifest', type=Path, default=STUDY_DIR / 'data' / 'input_files.csv')
    parser.add_argument('--input-dir', type=Path, default=STUDY_DIR / 'data' / 'local')
    parser.add_argument('--generation-dir', type=Path, default=STUDY_DIR / 'outputs' / 'final')
    parser.add_argument('--output-dir', type=Path, default=STUDY_DIR / 'outputs' / 'scores')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'fresh-score contract forbids replacing {args.output_dir}')
    config, paths, panels = inspect_inputs(args.config, args.manifest, args.input_dir)
    if not (int(config['scoring']['variance_ddof']) == 0 and config['scoring']['zero_definition'] == 'counts_less_than_or_equal_to_zero'):
        raise RuntimeError('unsupported change to the historical metric definition')
    jobs = declared_jobs(config)
    expected_h5ads = {expected_job_path(args.generation_dir, job) / 'generated.h5ad' for job in jobs}
    actual_h5ads = set(args.generation_dir.rglob('generated.h5ad')) if args.generation_dir.exists() else set()
    if actual_h5ads != expected_h5ads:
        raise RuntimeError('generation root is not the exact declared 45-H5AD matrix')
    validation_rows = [validate_job(expected_job_path(args.generation_dir, job), job, config, paths, panels) for job in jobs]
    if len(validation_rows) != int(config['expected_jobs']) or not all((row['valid'] for row in validation_rows)):
        raise RuntimeError('the exact 45 generation candidates did not validate')
    work_root = args.output_dir.parent / '.work'
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{config['scoring']['configuration_id']}-", dir=work_root))
    per_gene_path = work_dir / 'per_gene_metrics.csv.gz'
    summary_path = work_dir / 'summary.csv'
    support_path = work_dir / 'support_audit.csv'
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    summary_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    generation_inputs: dict[str, dict[str, str]] = {}
    first_write = True
    targets = {key: ad.read_h5ad(path) for key, path in paths.items()}
    for job in jobs:
        candidate_dir = expected_job_path(args.generation_dir, job)
        generated_path = candidate_dir / 'generated.h5ad'
        generation_provenance = json.loads((candidate_dir / 'provenance.json').read_text(encoding='utf-8'))
        generated = ad.read_h5ad(generated_path)
        genes = panels[job.dataset]
        target_full = targets[job.dataset, job.target]
        target = target_full[generated.obs_names.astype(str).tolist(), genes].copy()
        if list(map(str, target.obs_names)) != list(map(str, generated.obs_names)):
            raise RuntimeError(f'evaluation target identity mismatch: {job}')
        neighbors = neighbor_indices(target.obsm['spatial'], int(config['scoring']['neighbors_including_self']))
        per_gene = compute_per_gene_metrics(generated, target, neighbors, int(config['scoring']['gene_chunk_size']))
        support = generation_provenance['support']
        zero_genes = set(map(str, support['reference_zero_genes']))
        per_gene.insert(0, 'reference_observed', ~per_gene['gene'].astype(str).isin(zero_genes))
        primary = bool(np.isclose(job.assignment_randomness, float(config['primary_assignment_randomness'])))
        identity = [('donor_stratum', donor_stratum(job)), ('primary_setting', primary), ('assignment_randomness', job.assignment_randomness), ('target', job.target), ('source', job.source), ('direction', direction_id(job)), ('dataset', job.dataset), ('mode', job.mode), ('job_id', job_id(config, job))]
        for name, value in identity:
            per_gene.insert(0, name, value)
        per_gene.to_csv(per_gene_path, mode='w' if first_write else 'a', header=first_write, index=False)
        first_write = False
        observed = per_gene[per_gene['reference_observed']]
        observed_summary = {f'observed_{name}': value for name, value in panel_summary(observed).items()}
        summary_rows.append({'job_id': job_id(config, job), 'mode': job.mode, 'dataset': job.dataset, 'direction': direction_id(job), 'source': job.source, 'target': job.target, 'assignment_randomness': job.assignment_randomness, 'primary_setting': primary, 'donor_stratum': donor_stratum(job), 'n_target_spots': target.n_obs, 'target_retained_fraction': support['target_retained_fraction'], 'n_genes': len(genes), 'n_reference_observed_genes': int(support['reference_observed_gene_count']), **panel_summary(per_gene), **observed_summary, 'decode_method': generated.uns.get('de_novo', {}).get('decode_method', ''), 'seed': int(config['public_seed'])})
        split = support.get('mask_split') or {}
        support_rows.append({'job_id': job_id(config, job), 'mode': job.mode, 'dataset': job.dataset, 'direction': direction_id(job), 'source': job.source, 'target': job.target, 'assignment_randomness': job.assignment_randomness, 'donor_stratum': donor_stratum(job), 'source_candidate_spots': support['source_candidate_spots'], 'source_retained_spots': support['source_retained_spots'], 'target_candidate_spots': support['target_candidate_spots'], 'target_retained_spots': support['target_retained_spots'], 'target_retained_fraction': support['target_retained_fraction'], 'min_source_spots_per_label': support['min_source_spots_per_label'], 'eligible_label_count': len(support['eligible_labels']), 'eligible_labels_json': json.dumps(support['eligible_labels']), 'source_label_counts_json': json.dumps(support['source_label_counts'], sort_keys=True), 'excluded_target_label_counts_json': json.dumps(support['excluded_target_label_counts'], sort_keys=True), 'dataset_gene_panel_count': support['dataset_gene_panel_count'], 'reference_observed_gene_count': support['reference_observed_gene_count'], 'reference_zero_gene_count': support['reference_zero_gene_count'], 'mask_axis': split.get('axis', ''), 'mask_threshold': split.get('threshold', '')})
        key = job_id(config, job)
        generation_inputs[key] = {'prediction': str(generated_path.relative_to(args.generation_dir))}
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(support_rows).to_csv(support_path, index=False)
    provenance = {'status': 'ok', 'configuration_id': config['scoring']['configuration_id'], 'study_configuration_id': config['configuration_id'], 'feast_version': str(config['required_feast_version']), 'public_seed': int(config['public_seed']), 'target_expression_policy': config['target_expression_policy'], 'evaluation_input_artifacts': {f'{dataset}/{slice_id}': {'artifact_id': input_artifact_id(config, dataset, slice_id)} for (dataset, slice_id), path in paths.items()}, 'ordered_gene_panels': {dataset: {'n_genes': len(genes)} for dataset, genes in panels.items()}, 'generation_inputs': generation_inputs, 'generation_validation': {'validated_jobs': len(validation_rows)}, 'sources': {}, 'metric_contract': config['scoring'], 'stratification_contract': STRATIFICATION_CONTRACT, 'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'anndata': ad.__version__}, 'started_at': started_at.isoformat(), 'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started, 'outputs': {'per_gene_metrics.csv.gz': None, 'summary.csv': None, 'support_audit.csv': None}}
    (work_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    validate_score_root(work_dir, args.generation_dir, config, paths, panels)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.output_dir.exists():
        raise FileExistsError(f'fresh-score contract forbids replacing {args.output_dir}')
    os.replace(work_dir, args.output_dir)
    print(json.dumps({'status': 'ok', 'output_dir': str(args.output_dir)}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
