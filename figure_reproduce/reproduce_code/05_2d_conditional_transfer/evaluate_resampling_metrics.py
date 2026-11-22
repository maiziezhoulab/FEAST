"""Compare FEAST with conditional resampling using distribution and Moran metrics."""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wasserstein_distance
from conditional_resampling_baseline import conditional_donor_indices, dense_rows_chunk, selected_jobs
from score import _moran_columns, donor_stratum, neighbor_indices, safe_pearson
from workflow import STUDY_DIR, Job, direction_id, expected_job_path, inspect_inputs, prepare_job
HEADLINE_METRICS = ('conditional_expression_wasserstein', 'conditional_zero_fraction_wasserstein', 'moran_profile_correlation', 'residual_moran_profile_correlation')
TASK_KEYS = ('mode', 'dataset', 'direction', 'source', 'target', 'donor_stratum')

def row_statistics(matrix: Any, rows: np.ndarray | slice) -> tuple[np.ndarray, np.ndarray]:
    selected = matrix[rows, :]
    library = np.asarray(selected.sum(axis=1), dtype=np.float64).reshape(-1)
    if sparse.issparse(selected):
        detected = np.asarray((selected > 0).sum(axis=1), dtype=np.float64).reshape(-1)
    else:
        detected = np.count_nonzero(np.asarray(selected) > 0, axis=1).astype(float)
    return (library, detected)

def log_normalize(values: np.ndarray, library: np.ndarray) -> np.ndarray:
    scale = np.divide(10000.0, library, out=np.zeros_like(library, dtype=np.float64), where=library > 0)
    return np.log1p(values * scale[:, None])

def residualize(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    label_strings = np.asarray(labels, dtype=str)
    for label in sorted(set(label_strings.tolist())):
        mask = label_strings == label
        result[mask] -= result[mask].mean(axis=0, keepdims=True)
    return result

def evaluate_job(config: dict[str, Any], paths: dict[tuple[str, str], Path], panels: dict[str, list[str]], job: Job, targets: dict[tuple[str, str], ad.AnnData], rngs: list[np.random.Generator], seeds: list[int], feast_root: Path) -> list[dict[str, Any]]:
    prepared = prepare_job(config, paths, panels, job)
    annotation_key = str(config['datasets'][job.dataset]['annotation_key'])
    genes = prepared.genes
    target_full = targets[job.dataset, job.target]
    target = target_full[prepared.target_contract.obs_names.astype(str).tolist(), genes].copy()
    feast_path = expected_job_path(feast_root, job) / 'generated.h5ad'
    feast = ad.read_h5ad(feast_path)
    if list(map(str, feast.obs_names)) != list(map(str, target.obs_names)):
        raise ValueError(f'FEAST target spot order differs for {direction_id(job)}')
    if list(map(str, feast.var_names)) != genes:
        raise ValueError(f'FEAST gene order differs for {direction_id(job)}')
    target_labels = target.obs[annotation_key].astype(str).to_numpy()
    source_labels = prepared.reference.obs[annotation_key].astype(str).to_numpy()
    source_matrix = prepared.reference.layers['counts'] if 'counts' in prepared.reference.layers else prepared.reference.X
    target_matrix = target.layers['counts'] if 'counts' in target.layers else target.X
    feast_matrix = feast.layers['counts'] if 'counts' in feast.layers else feast.X
    sampled_rows = [conditional_donor_indices(source_labels, target_labels, rng) for rng in rngs]
    candidates: list[dict[str, Any]] = [{'method': 'feast', 'replicate': -1, 'seed': np.nan, 'matrix': feast_matrix, 'rows': slice(None)}]
    candidates.extend(({'method': 'conditional_whole_spot_resampling', 'replicate': replicate, 'seed': seed, 'matrix': source_matrix, 'rows': rows} for replicate, (seed, rows) in enumerate(zip(seeds, sampled_rows, strict=True))))
    target_library, target_detected = row_statistics(target_matrix, slice(None))
    for candidate in candidates:
        library, detected = row_statistics(candidate['matrix'], candidate['rows'])
        candidate['library'] = library
        candidate['detected'] = detected
        candidate['expression_w1'] = np.zeros(len(genes), dtype=np.float64)
        candidate['moran'] = np.full(len(genes), np.nan, dtype=np.float64)
        candidate['residual_moran'] = np.full(len(genes), np.nan, dtype=np.float64)
        candidate['zero_by_label'] = {}
    labels = sorted(set(target_labels.tolist()))
    label_indices = {label: np.flatnonzero(target_labels == label) for label in labels}
    label_weights = {label: indices.size / target_labels.size for label, indices in label_indices.items()}
    target_zero_by_label = {label: np.empty(len(genes), dtype=np.float64) for label in labels}
    for candidate in candidates:
        candidate['zero_by_label'] = {label: np.empty(len(genes), dtype=np.float64) for label in labels}
    neighbors = neighbor_indices(target.obsm['spatial'], int(config['scoring']['neighbors_including_self']))
    chunk_size = int(config['scoring']['gene_chunk_size'])
    target_moran = np.full(len(genes), np.nan, dtype=np.float64)
    target_residual_moran = np.full(len(genes), np.nan, dtype=np.float64)
    for start in range(0, len(genes), chunk_size):
        stop = min(start + chunk_size, len(genes))
        observed = dense_rows_chunk(target_matrix, slice(None), start, stop)
        normalized_observed = log_normalize(observed, target_library)
        observed_sorted = {label: np.sort(normalized_observed[indices], axis=0) for label, indices in label_indices.items()}
        for label, indices in label_indices.items():
            target_zero_by_label[label][start:stop] = np.mean(observed[indices] <= 0, axis=0)
        target_moran[start:stop] = _moran_columns(observed, neighbors)
        target_residual_moran[start:stop] = _moran_columns(residualize(observed, target_labels), neighbors)
        for candidate in candidates:
            generated = dense_rows_chunk(candidate['matrix'], candidate['rows'], start, stop)
            normalized_generated = log_normalize(generated, candidate['library'])
            gene_w1 = np.zeros(stop - start, dtype=np.float64)
            for label, indices in label_indices.items():
                generated_sorted = np.sort(normalized_generated[indices], axis=0)
                gene_w1 += label_weights[label] * np.mean(np.abs(generated_sorted - observed_sorted[label]), axis=0)
                candidate['zero_by_label'][label][start:stop] = np.mean(generated[indices] <= 0, axis=0)
            candidate['expression_w1'][start:stop] = gene_w1
            candidate['moran'][start:stop] = _moran_columns(generated, neighbors)
            candidate['residual_moran'][start:stop] = _moran_columns(residualize(generated, target_labels), neighbors)
    identity = {'mode': job.mode, 'dataset': job.dataset, 'direction': direction_id(job), 'source': job.source, 'target': job.target, 'donor_stratum': donor_stratum(job)}
    rows = []
    for candidate in candidates:
        zero_w1 = sum((label_weights[label] * wasserstein_distance(target_zero_by_label[label], candidate['zero_by_label'][label]) for label in labels))
        rows.append({**identity, 'method': candidate['method'], 'replicate': candidate['replicate'], 'seed': candidate['seed'], 'n_target_spots': int(target.n_obs), 'n_genes': len(genes), 'n_labels': len(labels), 'conditional_expression_wasserstein': float(np.median(candidate['expression_w1'])), 'conditional_zero_fraction_wasserstein': float(zero_w1), 'moran_profile_correlation': safe_pearson(target_moran, candidate['moran']), 'residual_moran_profile_correlation': safe_pearson(target_residual_moran, candidate['residual_moran']), 'library_size_wasserstein': float(wasserstein_distance(target_library, candidate['library'])), 'detected_genes_wasserstein': float(wasserstein_distance(target_detected, candidate['detected'])), 'target_zero_library_spots': int(np.sum(target_library <= 0)), 'generated_zero_library_spots': int(np.sum(candidate['library'] <= 0))})
    return rows

def summarize_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame['method'] == 'conditional_whole_spot_resampling']
    rows = []
    for keys, group in baseline.groupby(list(TASK_KEYS), sort=False):
        row = dict(zip(TASK_KEYS, keys, strict=True))
        row['replicates'] = len(group)
        for metric in HEADLINE_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.mean(values))
            row[f'{metric}_sd'] = float(np.std(values, ddof=1))
            row[f'{metric}_p05'] = float(np.quantile(values, 0.05))
            row[f'{metric}_p95'] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)

def comparison_frame(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    feast = metrics[metrics['method'] == 'feast']
    merged = summary.merge(feast[list(TASK_KEYS) + list(HEADLINE_METRICS)], on=list(TASK_KEYS), validate='one_to_one')
    rows = []
    for record in merged.to_dict('records'):
        for metric in HEADLINE_METRICS:
            rows.append({**{key: record[key] for key in TASK_KEYS}, 'metric': metric, 'baseline_mean': record[f'{metric}_mean'], 'baseline_sd': record[f'{metric}_sd'], 'baseline_p05': record[f'{metric}_p05'], 'baseline_p95': record[f'{metric}_p95'], 'feast': record[metric], 'feast_minus_baseline_mean': record[metric] - record[f'{metric}_mean']})
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_DIR / 'config.yaml')
    parser.add_argument('--manifest', type=Path, default=STUDY_DIR / 'data' / 'input_files.csv')
    parser.add_argument('--input-dir', type=Path, default=STUDY_DIR / 'data' / 'local')
    parser.add_argument('--feast-root', type=Path, default=STUDY_DIR / 'outputs' / 'final')
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=STUDY_DIR / 'outputs' / 'baselines' / 'conditional_metric_comparison')
    args = parser.parse_args()
    if args.replicates != 10:
        raise ValueError('the approved comparison requires exactly 10 replicates')
    if args.output_dir.exists():
        raise FileExistsError(f'candidate metric output already exists: {args.output_dir}')
    config, paths, panels = inspect_inputs(args.config, args.manifest, args.input_dir)
    jobs = selected_jobs(config)
    seeds = [int(config['public_seed']) + index for index in range(args.replicates)]
    rngs = [np.random.default_rng(seed) for seed in seeds]
    targets = {key: ad.read_h5ad(path) for key, path in paths.items()}
    work_root = args.output_dir.parent / '.work'
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='study05-metric-comparison-', dir=work_root))
    try:
        rows = []
        for job in jobs:
            rows.extend(evaluate_job(config, paths, panels, job, targets, rngs, seeds, args.feast_root))
        metrics = pd.DataFrame(rows)
        summary = summarize_baseline(metrics)
        comparison = comparison_frame(metrics, summary)
        metrics.to_csv(work_dir / 'method_metrics.csv', index=False)
        summary.to_csv(work_dir / 'baseline_summary.csv', index=False)
        comparison.to_csv(work_dir / 'feast_comparison.csv', index=False)
        provenance = {'status': 'complete', 'configuration_id': config['configuration_id'], 'methods': ['feast', 'conditional_whole_spot_resampling'], 'replicates': args.replicates, 'seeds': seeds, 'normalization': 'per_spot_total_10000_then_log1p_for_expression_wasserstein_only', 'conditional_aggregation': 'target_spot_fraction_weighted_labels_then_median_genes', 'moran_graph': 'existing_study05_neighbors_including_self_contract', 'residualization': 'separate_within_label_gene_means_for_target_and_candidate', 'target_expression_access': 'evaluation_only', 'headline_metrics': list(HEADLINE_METRICS), 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        (work_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_dir, args.output_dir)
    except Exception:
        raise
    print(f'wrote {len(metrics)} Study 05 method evaluations to {args.output_dir}')
if __name__ == '__main__':
    main()
