"""Summarize validated full-axis Study 07 coverage and z continuity."""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import workflow

def finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float('nan')
    x = left[valid]
    y = right[valid]
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def adjacent_metrics(previous: dict, current: dict) -> dict:
    shared_regions = sorted(set(previous['region_means']) & set(current['region_means']))
    previous_region = np.concatenate([previous['region_means'][key] for key in shared_regions]) if shared_regions else np.array([], dtype=np.float64)
    current_region = np.concatenate([current['region_means'][key] for key in shared_regions]) if shared_regions else np.array([], dtype=np.float64)
    return {'lower_z_index': int(previous['z_index']), 'upper_z_index': int(current['z_index']), 'lower_z': float(previous['z_world']), 'upper_z': float(current['z_world']), 'midpoint_z': float((previous['z_world'] + current['z_world']) / 2.0), 'shared_regions': len(shared_regions), 'gene_mean_pearson': finite_pearson(previous['gene_mean'], current['gene_mean']), 'log1p_gene_mean_pearson': finite_pearson(np.log1p(previous['gene_mean']), np.log1p(current['gene_mean'])), 'region_gene_mean_pearson': finite_pearson(previous_region, current_region), 'log1p_region_gene_mean_pearson': finite_pearson(np.log1p(previous_region), np.log1p(current_region)), 'mean_absolute_log1p_gene_change': float(np.mean(np.abs(np.log1p(current['gene_mean']) - np.log1p(previous['gene_mean']))))}

def solver_summary(diagnostics: dict) -> tuple[int, float]:
    if not bool(diagnostics.get('all_converged')):
        raise ValueError('Study 07 slice lacks positive convergence evidence')
    return (int(diagnostics['transport_records']), float(diagnostics['max_final_error']))

def summarize_slice(path: Path, chunk_size: int) -> tuple[dict, list[dict], dict]:
    result = ad.read_h5ad(path, backed='r')
    try:
        n_obs, n_genes = result.shape
        regions = result.obs['region'].astype(str).to_numpy()
        unique_regions = sorted(set(regions))
        gene_sum = np.zeros(n_genes, dtype=np.float64)
        gene_nonzero = np.zeros(n_genes, dtype=np.int64)
        library_sizes = np.empty(n_obs, dtype=np.float64)
        detected_genes = np.empty(n_obs, dtype=np.int32)
        region_sum = {region: np.zeros(n_genes, dtype=np.float64) for region in unique_regions}
        region_count = {region: 0 for region in unique_regions}
        for start in range(0, n_obs, chunk_size):
            stop = min(start + chunk_size, n_obs)
            values = np.asarray(result.X[start:stop, :], dtype=np.float64)
            gene_sum += values.sum(axis=0)
            positive = values > 0
            gene_nonzero += positive.sum(axis=0)
            library_sizes[start:stop] = values.sum(axis=1)
            detected_genes[start:stop] = positive.sum(axis=1)
            chunk_regions = regions[start:stop]
            for region in sorted(set(chunk_regions)):
                mask = chunk_regions == region
                region_sum[region] += values[mask].sum(axis=0)
                region_count[region] += int(mask.sum())
        provenance = result.uns['publication_reproduction']
        diagnostics = provenance['solver_diagnostics']
        transport_records, max_final_error = solver_summary(diagnostics)
        z_index = int(provenance['z_index'])
        z_world = float(provenance['z_world'])
        gene_mean = gene_sum / float(n_obs)
        region_means = {region: region_sum[region] / float(region_count[region]) for region in unique_regions}
        row = {'age': str(provenance['age']), 'z_index': z_index, 'z_world': z_world, 'n_spots': n_obs, 'n_genes': n_genes, 'n_regions': len(unique_regions), 'mean_library_size': float(np.mean(library_sizes)), 'median_library_size': float(np.median(library_sizes)), 'mean_detected_genes': float(np.mean(detected_genes)), 'median_detected_genes': float(np.median(detected_genes)), 'overall_zero_fraction': float(1.0 - gene_nonzero.sum() / float(n_obs * n_genes)), 'transport_records': transport_records, 'max_final_error': max_final_error}
        coverage = [{'age': row['age'], 'z_index': z_index, 'z_world': z_world, 'region': region, 'n_spots': region_count[region], 'spot_fraction': region_count[region] / float(n_obs)} for region in unique_regions]
        state = {'z_index': z_index, 'z_world': z_world, 'gene_mean': gene_mean, 'region_means': region_means}
        return (row, coverage, state)
    finally:
        result.file.close()

def require_validation(config: dict) -> dict:
    path = config['_final_dir'] / 'validation.json'
    if not path.is_file():
        raise FileNotFoundError('complete Study 07 validation is required before evaluation')
    decision = json.loads(path.read_text(encoding='utf-8'))
    if decision.get('status') != 'validated' or len(decision.get('ages', [])) != 2:
        raise ValueError('Study 07 validation decision is incomplete')
    if not all((row.get('all_transport_records_converged') for row in decision['ages'])):
        raise ValueError('Study 07 validation lacks positive convergence evidence')
    return decision

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=workflow.DEFAULT_CONFIG)
    parser.add_argument('--chunk-size', type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError('chunk size must be positive')
    config = workflow.load_config(args.config)
    decision = require_validation(config)
    evaluation_dir = config['_final_dir'] / 'evaluation'
    staging = config['_work_dir'] / 'evaluating-final'
    if evaluation_dir.exists() or staging.exists():
        raise FileExistsError('fresh-only Study 07 evaluation path already exists')
    staging.mkdir(parents=True)
    slice_rows: list[dict] = []
    coverage_rows: list[dict] = []
    adjacent_rows: list[dict] = []
    age_rows: list[dict] = []
    for age in workflow.AGE_ORDER:
        manifest = pd.read_csv(config['_final_dir'] / age / 'manifest.csv').sort_values('z_index')
        previous = None
        age_slices: list[dict] = []
        age_adjacent: list[dict] = []
        for row in manifest.itertuples(index=False):
            path = config['_final_dir'] / age / str(row.filename)
            summary, coverage, state = summarize_slice(path, args.chunk_size)
            slice_rows.append(summary)
            age_slices.append(summary)
            coverage_rows.extend(coverage)
            if previous is not None:
                adjacent = {'age': age, **adjacent_metrics(previous, state)}
                adjacent_rows.append(adjacent)
                age_adjacent.append(adjacent)
            previous = state
        age_rows.append({'age': age, 'n_z_levels': len(age_slices), 'n_spots': sum((row['n_spots'] for row in age_slices)), 'z_start': age_slices[0]['z_world'], 'z_end': age_slices[-1]['z_world'], 'n_genes': age_slices[0]['n_genes'], 'transport_records': sum((row['transport_records'] for row in age_slices)), 'max_final_error': max((row['max_final_error'] for row in age_slices)), 'median_adjacent_gene_mean_pearson': float(np.nanmedian([row['gene_mean_pearson'] for row in age_adjacent])), 'median_adjacent_region_gene_mean_pearson': float(np.nanmedian([row['region_gene_mean_pearson'] for row in age_adjacent])), 'minimum_adjacent_gene_mean_pearson': float(np.nanmin([row['gene_mean_pearson'] for row in age_adjacent])), 'minimum_adjacent_region_gene_mean_pearson': float(np.nanmin([row['region_gene_mean_pearson'] for row in age_adjacent]))})
    outputs = {'slice_metrics.csv': staging / 'slice_metrics.csv', 'region_coverage.csv': staging / 'region_coverage.csv', 'adjacent_z_continuity.csv': staging / 'adjacent_z_continuity.csv', 'age_summary.csv': staging / 'age_summary.csv'}
    pd.DataFrame(slice_rows).sort_values(['age', 'z_index']).to_csv(outputs['slice_metrics.csv'], index=False)
    pd.DataFrame(coverage_rows).sort_values(['age', 'z_index', 'region']).to_csv(outputs['region_coverage.csv'], index=False)
    pd.DataFrame(adjacent_rows).sort_values(['age', 'lower_z_index']).to_csv(outputs['adjacent_z_continuity.csv'], index=False)
    pd.DataFrame(age_rows).sort_values('age').to_csv(outputs['age_summary.csv'], index=False)
    provenance = {'schema_version': 1, 'configuration_id': config['configuration_id'], 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'publication_canonical': False, 'interpretation': 'expression-free_target_descriptive_coverage_and_continuity_only', 'accuracy_claim_authorized': False, 'composite_score': 'prohibited', 'validation_status': decision['status'], 'final_manifests': {age: None for age in workflow.AGE_ORDER}, 'outputs': {name: None for name, path in outputs.items()}}
    (staging / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(staging, evaluation_dir)
    print(f'wrote Study 07 descriptive evaluation to {evaluation_dir}')
if __name__ == '__main__':
    main()
