"""Evaluate label-conditioned whole-spot resampling for Study 05.

Each target spot draws one source spot with replacement from the same
conditional label.  The complete count vector is copied, preserving empirical
sparsity, library size, gene covariance, and within-label heterogeneity while
deliberately ignoring target spatial coordinates during generation.
"""
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
from scipy.stats import rankdata
from score import _moran_columns, _pearson_columns, donor_stratum, neighbor_indices, panel_summary
from workflow import STUDY_DIR, Job, declared_jobs, direction_id, inspect_inputs, prepare_job
METRICS = ('mean_corr', 'var_corr', 'moran_corr', 'zero_ks', 'median_gene_pearson', 'median_gene_spearman')
IDENTITY_COLUMNS = ('mode', 'dataset', 'direction', 'source', 'target', 'donor_stratum')

def dense_rows_chunk(matrix: Any, row_indices: np.ndarray | slice, start: int, stop: int) -> np.ndarray:
    values = matrix[row_indices, start:stop]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)

def conditional_donor_indices(source_labels: np.ndarray, target_labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one same-label source row for every target row."""
    source = np.asarray(source_labels, dtype=str)
    target = np.asarray(target_labels, dtype=str)
    sampled = np.empty(target.size, dtype=np.int64)
    for label in sorted(set(target.tolist())):
        source_indices = np.flatnonzero(source == label)
        if source_indices.size == 0:
            raise ValueError(f'target label lacks source support: {label}')
        target_indices = np.flatnonzero(target == label)
        sampled[target_indices] = rng.choice(source_indices, size=target_indices.size, replace=True)
    return sampled

def selected_jobs(config: dict[str, Any]) -> list[Job]:
    primary = float(config['primary_assignment_randomness'])
    return [job for job in declared_jobs(config) if job.mode == 'mask_half' or np.isclose(job.assignment_randomness, primary)]

def evaluate_job(config: dict[str, Any], paths: dict[tuple[str, str], Path], panels: dict[str, list[str]], job: Job, targets: dict[tuple[str, str], ad.AnnData], rngs: list[np.random.Generator], seeds: list[int]) -> list[dict[str, Any]]:
    prepared = prepare_job(config, paths, panels, job)
    annotation_key = str(config['datasets'][job.dataset]['annotation_key'])
    genes = prepared.genes
    target_full = targets[job.dataset, job.target]
    target = target_full[prepared.target_contract.obs_names.astype(str).tolist(), genes].copy()
    target_labels = target.obs[annotation_key].astype(str).to_numpy()
    source_labels = prepared.reference.obs[annotation_key].astype(str).to_numpy()
    missing = sorted(set(target_labels) - set(source_labels))
    if missing:
        raise ValueError(f'target labels lack source support: {missing}')
    source_matrix = prepared.reference.layers['counts'] if 'counts' in prepared.reference.layers else prepared.reference.X
    target_matrix = target.layers['counts'] if 'counts' in target.layers else target.X
    sampled_rows = [conditional_donor_indices(source_labels, target_labels, rng) for rng in rngs]
    neighbors = neighbor_indices(target.obsm['spatial'], int(config['scoring']['neighbors_including_self']))
    chunk_size = int(config['scoring']['gene_chunk_size'])
    column_names = ('pearson', 'spearman', 'generated_mean', 'target_mean', 'generated_variance', 'target_variance', 'generated_zero_prop', 'target_zero_prop', 'generated_moran_i', 'target_moran_i')
    columns_by_replicate = [{name: np.full(len(genes), np.nan, dtype=np.float64) for name in column_names} for _ in rngs]
    for start in range(0, len(genes), chunk_size):
        stop = min(start + chunk_size, len(genes))
        observed = dense_rows_chunk(target_matrix, slice(None), start, stop)
        observed_rank = rankdata(observed, method='average', axis=0)
        target_mean = observed.mean(axis=0)
        target_variance = observed.var(axis=0)
        target_zero = np.mean(observed <= 0, axis=0)
        target_moran = _moran_columns(observed, neighbors)
        for replicate, donor_rows in enumerate(sampled_rows):
            generated = dense_rows_chunk(source_matrix, donor_rows, start, stop)
            columns = columns_by_replicate[replicate]
            columns['pearson'][start:stop] = _pearson_columns(generated, observed)
            columns['spearman'][start:stop] = _pearson_columns(rankdata(generated, method='average', axis=0), observed_rank)
            columns['generated_mean'][start:stop] = generated.mean(axis=0)
            columns['target_mean'][start:stop] = target_mean
            columns['generated_variance'][start:stop] = generated.var(axis=0)
            columns['target_variance'][start:stop] = target_variance
            columns['generated_zero_prop'][start:stop] = np.mean(generated <= 0, axis=0)
            columns['target_zero_prop'][start:stop] = target_zero
            columns['generated_moran_i'][start:stop] = _moran_columns(generated, neighbors)
            columns['target_moran_i'][start:stop] = target_moran
    identity = {'method': 'conditional_whole_spot_resampling', 'mode': job.mode, 'dataset': job.dataset, 'direction': direction_id(job), 'source': job.source, 'target': job.target, 'donor_stratum': donor_stratum(job)}
    zero_genes = set(map(str, prepared.support['reference_zero_genes']))
    reference_observed = ~pd.Series(genes).isin(zero_genes).to_numpy()
    summaries = []
    for replicate, (seed, columns) in enumerate(zip(seeds, columns_by_replicate, strict=True)):
        per_gene = pd.DataFrame({'gene': genes, **columns})
        observed_summary = {f'observed_{name}': value for name, value in panel_summary(per_gene[reference_observed]).items()}
        summaries.append({**identity, 'replicate': replicate, 'seed': seed, 'n_target_spots': int(target.n_obs), 'target_retained_fraction': float(prepared.support['target_retained_fraction']), 'n_genes': len(genes), 'n_reference_observed_genes': int(prepared.support['reference_observed_gene_count']), **panel_summary(per_gene), **observed_summary})
    return summaries

def aggregate_replicates(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in summary.groupby(list(IDENTITY_COLUMNS), sort=False):
        row = dict(zip(IDENTITY_COLUMNS, keys, strict=True))
        first = frame.iloc[0]
        for name in ('n_target_spots', 'target_retained_fraction', 'n_genes', 'n_reference_observed_genes'):
            row[name] = first[name]
        row['replicates'] = len(frame)
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.nanmean(values))
            row[f'{metric}_sd'] = float(np.nanstd(values, ddof=1))
            row[f'{metric}_p05'] = float(np.nanquantile(values, 0.05))
            row[f'{metric}_p95'] = float(np.nanquantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)

def feast_comparison(aggregate: pd.DataFrame, feast_path: Path) -> pd.DataFrame:
    feast = pd.read_csv(feast_path)
    feast = feast[feast['primary_setting'].astype(bool)].copy()
    keys = list(IDENTITY_COLUMNS)
    merged = aggregate.merge(feast[keys + list(METRICS)], on=keys, validate='one_to_one')
    rows = []
    for record in merged.to_dict('records'):
        for metric in METRICS:
            baseline_mean = float(record[f'{metric}_mean'])
            rows.append({**{key: record[key] for key in keys}, 'metric': metric, 'baseline_mean': baseline_mean, 'baseline_sd': float(record[f'{metric}_sd']), 'baseline_p05': float(record[f'{metric}_p05']), 'baseline_p95': float(record[f'{metric}_p95']), 'feast': float(record[metric]), 'feast_minus_baseline_mean': float(record[metric]) - baseline_mean})
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_DIR / 'config.yaml')
    parser.add_argument('--manifest', type=Path, default=STUDY_DIR / 'data' / 'input_files.csv')
    parser.add_argument('--input-dir', type=Path, default=STUDY_DIR / 'data' / 'local')
    parser.add_argument('--feast-summary', type=Path, default=STUDY_DIR / 'outputs' / 'scores' / 'summary.csv')
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=STUDY_DIR / 'outputs' / 'baselines' / 'conditional_whole_spot_resampling')
    args = parser.parse_args()
    if args.replicates < 2:
        raise ValueError('at least two replicates are required to estimate uncertainty')
    if args.output_dir.exists():
        raise FileExistsError(f'baseline output already exists: {args.output_dir}')
    config, paths, panels = inspect_inputs(args.config, args.manifest, args.input_dir)
    jobs = selected_jobs(config)
    seeds = [int(config['public_seed']) + index for index in range(args.replicates)]
    rngs = [np.random.default_rng(seed) for seed in seeds]
    targets = {key: ad.read_h5ad(path) for key, path in paths.items()}
    work_root = STUDY_DIR / '.work'
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix='study05-resampling-', dir=work_root))
    try:
        summaries = []
        for job in jobs:
            summaries.extend(evaluate_job(config, paths, panels, job, targets, rngs, seeds))
        replicate_frame = pd.DataFrame(summaries)
        aggregate_frame = aggregate_replicates(replicate_frame)
        replicate_frame.to_csv(work_dir / 'replicate_summary.csv', index=False)
        aggregate_frame.to_csv(work_dir / 'aggregate_summary.csv', index=False)
        feast_comparison(aggregate_frame, args.feast_summary).to_csv(work_dir / 'feast_comparison.csv', index=False)
        provenance = {'status': 'complete', 'configuration_id': config['configuration_id'], 'method': 'same_label_whole_source_spot_resampling_with_replacement', 'replicates': args.replicates, 'seeds': seeds, 'spatial_coordinates_used_for_generation': False, 'target_expression_access': 'evaluation_only', 'compared_feast_setting': float(config['primary_assignment_randomness']), 'jobs': len(jobs), 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        (work_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_dir, args.output_dir)
    except Exception:
        raise
    print(f'wrote {len(summaries)} Study 05 resampling evaluations to {args.output_dir}')
if __name__ == '__main__':
    main()
