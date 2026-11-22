"""Evaluate five-reference whole-spot resampling against completed Study 06 targets.

Reuse the established conditional Wasserstein/Moran metrics and ten-repeat design.
Only target labels enter generation; target expression is evaluation-only.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from aggregate import METRICS, class_means, continuity_row, counts, metric_row, require_validation, safe_pearson, study00_graph, study00_metric_row, study00_moran, z_coherence_rows
from conditional_resampling_baseline import prepare_sampling, resample_spots
from run import input_path
from workflow import load_config, read_json
HEADLINE_METRICS = ('conditional_expression_wasserstein', 'conditional_zero_fraction_wasserstein', 'moran_profile_correlation', 'residual_moran_profile_correlation')
TARGET_KEYS = ('density_name', 'gap', 'target_index', 'target_slice', 'target_z')

def log_normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    library = values.sum(axis=1)
    detected = np.count_nonzero(values > 0, axis=1).astype(float)
    scale = np.divide(10000.0, library, out=np.zeros_like(library, dtype=np.float64), where=library > 0)
    return (np.log1p(values * scale[:, None]), library, detected)

def residualize(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    label_strings = np.asarray(labels, dtype=str)
    for label in sorted(set(label_strings.tolist())):
        mask = label_strings == label
        result[mask] -= result[mask].mean(axis=0, keepdims=True)
    return result

def prepare_conditionals(observed: np.ndarray, labels: np.ndarray):
    normalized, library, detected = log_normalize(observed)
    groups = []
    for label in sorted(set(map(str, labels))):
        indices = np.flatnonzero(labels == label)
        groups.append((indices, indices.size / len(labels), np.sort(normalized[indices], axis=0), np.mean(observed[indices] <= 0, axis=0)))
    return (groups, library, detected)

def conditional_distances(observed: np.ndarray, generated: np.ndarray, labels: np.ndarray, prepared=None) -> tuple[float, float, float, float]:
    groups, target_library, target_detected = prepare_conditionals(observed, labels) if prepared is None else prepared
    normalized_generated, generated_library, generated_detected = log_normalize(generated)
    gene_w1 = np.zeros(observed.shape[1], dtype=np.float64)
    zero_w1 = 0.0
    for indices, weight, target_sorted, target_zero in groups:
        generated_sorted = np.sort(normalized_generated[indices], axis=0)
        gene_w1 += weight * np.mean(np.abs(target_sorted - generated_sorted), axis=0)
        generated_zero = np.mean(generated[indices] <= 0, axis=0)
        zero_w1 += weight * wasserstein_distance(target_zero, generated_zero)
    return (float(np.median(gene_w1)), float(zero_w1), float(wasserstein_distance(target_library, generated_library)), float(wasserstein_distance(target_detected, generated_detected)))

def spatial_profiles(observed: np.ndarray, generated: np.ndarray, labels: np.ndarray, weights: Any, gene_indices: np.ndarray, target_moran: np.ndarray, target_residual_moran: np.ndarray) -> tuple[float, float]:
    generated_moran = study00_moran(generated, gene_indices, weights)
    generated_residual_moran = study00_moran(residualize(generated[:, gene_indices], labels), np.arange(len(gene_indices)), weights)
    return (safe_pearson(target_moran, generated_moran), safe_pearson(target_residual_moran, generated_residual_moran))

def summarize_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame['method'] == 'conditional_whole_spot_resampling']
    rows = []
    for keys, group in baseline.groupby(list(TARGET_KEYS), sort=False):
        row = dict(zip(TARGET_KEYS, keys, strict=True))
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
    merged = summary.merge(feast[list(TARGET_KEYS) + list(HEADLINE_METRICS)], on=list(TARGET_KEYS), validate='one_to_one')
    rows = []
    for record in merged.to_dict('records'):
        for metric in HEADLINE_METRICS:
            rows.append({**{key: record[key] for key in TARGET_KEYS}, 'metric': metric, 'baseline_mean': record[f'{metric}_mean'], 'baseline_sd': record[f'{metric}_sd'], 'baseline_p05': record[f'{metric}_p05'], 'baseline_p95': record[f'{metric}_p95'], 'feast': record[metric], 'feast_minus_baseline_mean': record[metric] - record[f'{metric}_mean']})
    return pd.DataFrame(rows)

def evaluate_target(row, density, plan, config, output_root, rngs, seeds, feast_atomic):
    dataset = config['dataset']
    target = ad.read_h5ad(input_path(plan, config, int(row['target_slice'])))
    target_name = f"{dataset['filename_prefix']}.{int(row['target_slice']):03d}"
    feast = ad.read_h5ad(output_root / 'targets' / row['density_name'] / target_name / 'generated.h5ad')
    if not target.obs_names.equals(feast.obs_names) or not target.var_names.equals(feast.var_names):
        raise ValueError(f'{target_name}: FEAST and target row/gene order differ')
    labels = target.obs[dataset['label_key']].astype(str).to_numpy()
    genes = list(map(str, target.var_names))
    groups, support = prepare_sampling(row, density, plan['data_dir'], dataset['filename_prefix'], dataset['label_key'], dataset['counts_layer'], labels, genes)
    observed = counts(target, dataset['counts_layer'])
    coordinates = np.asarray(target.obsm[dataset['spatial_key']], dtype=float)
    target_means = class_means(observed, labels)
    prepared = prepare_conditionals(observed, labels)
    graph = study00_graph(coordinates, neighbors=6)
    top = np.argsort(observed.mean(axis=0), kind='stable')[-min(300, observed.shape[1]):][::-1]
    target_moran = study00_moran(observed, top, graph)
    target_residual_moran = study00_moran(residualize(observed[:, top], labels), np.arange(len(top)), graph)
    records, sampled_means = ([], [])
    for replicate in range(-1, len(rngs)):
        is_feast = replicate == -1
        generated = counts(feast) if is_feast else resample_spots(groups, observed.shape, rngs[replicate])
        distances = conditional_distances(observed, generated, labels, prepared)
        moran, residual_moran = spatial_profiles(observed, generated, labels, graph, top, target_moran, target_residual_moran)
        atomic = dict(feast_atomic) if is_feast else {**metric_row(generated, observed), **study00_metric_row(observed, generated), 'moran_i_correlation': moran}
        records.append({**{key: row[key] for key in TARGET_KEYS}, 'method': 'feast' if is_feast else 'conditional_whole_spot_resampling', 'replicate': replicate, 'seed': int(row['seed']) if is_feast else seeds[replicate], 'n_target_spots': int(target.n_obs), 'n_genes': len(genes), 'n_labels': len(set(labels)), 'conditional_expression_wasserstein': distances[0], 'conditional_zero_fraction_wasserstein': distances[1], 'library_size_wasserstein': distances[2], 'detected_genes_wasserstein': distances[3], 'moran_profile_correlation': moran, 'residual_moran_profile_correlation': residual_moran, **atomic})
        if not is_feast:
            sampled_means.append(class_means(generated, labels))
        del generated
    return (records, target_means, sampled_means, genes, support)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--replicates', type=int, default=10)
    args = parser.parse_args()
    if args.replicates != 10:
        raise ValueError('the established comparison uses ten resampling repeats')
    root = args.output_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / 'baselines' / 'five_reference_conditional_resampling'
    if output.exists():
        raise FileExistsError(output)
    validation = require_validation(root)
    config = load_config(root / 'run_config.yaml')
    plan = read_json(root / 'plan.json')
    dataset = config['dataset']
    feast_metrics = pd.read_csv(root / 'evaluation/target_metrics.csv')
    atomic_lookup = {(int(r['gap']), int(r['target_slice'])): {k: r[k] for k in METRICS} for r in feast_metrics.to_dict('records')}
    seeds = [int(config['public_seed']) + index for index in range(args.replicates)]
    rngs = [np.random.default_rng(seed) for seed in seeds]
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.five-reference-resampling-', dir=output.parent))
    records, adjacent_records, coherence_records, support_records = ([], [], [], [])
    processed = 0
    for density_name, density in sorted(plan['densities'].items(), key=lambda item: int(item[1]['spec']['gap'])):
        gap = int(density['spec']['gap'])
        target_rows = {int(r['target_slice']): r for r in density['targets']}
        references = set(map(int, density['reference_pool']))
        trajectories = [{} for _ in rngs]
        previous = [None for _ in rngs]
        previous_slice = previous_z = None
        genes = None
        for slice_id in map(int, plan['available_slices']):
            if slice_id in references:
                data = ad.read_h5ad(input_path(plan, config, slice_id))
                genes = list(map(str, data.var_names))
                labels = data.obs[dataset['label_key']].astype(str).to_numpy()
                target_means = class_means(counts(data, dataset['counts_layer']), labels)
                means = [target_means for _ in rngs]
                z = float(data.obs[dataset['z_key']].iloc[0])
                del data
            else:
                row = target_rows[slice_id]
                started = time.perf_counter()
                local, target_means, means, observed_genes, support = evaluate_target(row, density, plan, config, root, rngs, seeds, atomic_lookup[gap, slice_id])
                if genes is not None and genes != observed_genes:
                    raise ValueError('gene order changed within the reconstructed stack')
                genes = observed_genes
                z = float(row['target_z'])
                records.extend(local)
                support_records.extend(({**{key: row[key] for key in TARGET_KEYS}, **entry} for entry in support))
                processed += 1
                pd.DataFrame(local).to_csv(work / 'method_metrics.csv', mode='a', header=processed == 1, index=False)
                print(f'[{processed}/{len(validation)}] gap {gap}, target {slice_id:03d}: FEAST + 10 repeats in {time.perf_counter() - started:.1f}s', flush=True)
            for replicate, (seed, current) in enumerate(zip(seeds, means, strict=True)):
                for label in sorted(target_means):
                    trajectories[replicate].setdefault(label, []).append({'z': z, 'baseline': current[label], 'target': target_means[label]})
                if previous[replicate] is not None:
                    adjacent_records.append({'density_name': density_name, 'gap': gap, 'replicate': replicate, 'seed': seed, 'lower_slice': previous_slice, 'upper_slice': slice_id, 'lower_z': previous_z, 'upper_z': z, **continuity_row(previous[replicate], current)})
                previous[replicate] = current
            previous_slice, previous_z = (slice_id, z)
        for replicate, (seed, trajectory) in enumerate(zip(seeds, trajectories, strict=True)):
            rows = z_coherence_rows(trajectory, genes, 'baseline', 'target', int(config['evaluation']['real_continuity_baseline']['minimum_z_levels']))
            values = np.array([r['z_coherence'] for r in rows])
            coherence_records.append({'density_name': density_name, 'gap': gap, 'replicate': replicate, 'seed': seed, 'class_gene_pairs': len(values), 'mean_z_coherence': float(np.mean(values)), 'median_z_coherence': float(np.median(values))})
        print(f'Gap {gap}: all target metrics and full-stack continuity complete.', flush=True)
    metrics = pd.DataFrame(records)
    if processed != len(validation) or len(metrics) != len(validation) * (args.replicates + 1):
        raise ValueError('incomplete matched target/repeat coverage')
    summary = summarize_baseline(metrics)
    summary.to_csv(work / 'baseline_summary.csv', index=False)
    comparison_frame(metrics, summary).to_csv(work / 'feast_comparison.csv', index=False)
    baseline = metrics[metrics['method'] == 'conditional_whole_spot_resampling']
    atomic = baseline.groupby(list(TARGET_KEYS))[list(METRICS)].agg(['mean', 'std'])
    atomic.columns = ['_'.join(column) for column in atomic.columns]
    atomic.reset_index().to_csv(work / 'atomic_baseline_summary.csv', index=False)
    pd.DataFrame(adjacent_records).to_csv(work / 'adjacent_z_continuity.csv', index=False)
    pd.DataFrame(coherence_records).to_csv(work / 'z_coherence_summary.csv', index=False)
    pd.DataFrame(support_records).to_json(work / 'reference_support.json', orient='records', indent=2)
    provenance = {'status': 'complete', 'configuration_id': config['configuration_id'], 'method': 'same_original_class_whole_spot_resampling_from_five_weighted_references', 'reference_selection': 'primary_reference_slices and reference_weights from the completed FEAST plan', 'class_weight_policy': 'renormalize primary z weights over references containing the original class', 'unsupported_class_policy': 'use the explicitly declared same-class supporting donor only', 'replicates': args.replicates, 'seeds': seeds, 'normalization': 'per_spot_total_10000_then_log1p_for_expression_wasserstein_only', 'conditional_aggregation': 'target_spot_fraction_weighted_labels_then_median_genes', 'moran_graph': 'study00_target_knn_6_top_300_target_mean_genes', 'residualization': 'separate_within_label_gene_means_for_target_and_candidate', 'target_expression_access': 'evaluation_only', 'target_coordinates_used_for_generation': False, 'conditioning_labels': 'original class labels; no FEAST modeling-group merge', 'continuity_population': 'all 144 positions; retained real anchors copied into each baseline stack', 'headline_metrics': list(HEADLINE_METRICS), 'atomic_metrics': list(METRICS), 'validated_feast_targets': len(validation), 'baseline_target_evaluations': len(baseline), 'input_root': str(root), 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    (work / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    os.replace(work, output)
    print(f'Completed {processed} targets × 10 resampling repeats: {output}', flush=True)
if __name__ == '__main__':
    main()
