"""Compare native SpatialZ and FEAST slices without matching individual cells.

Run in the Study 06 Python 3.11 environment. Evaluate an explicit target subset
for pilots; omitting selectors requires all 336 SpatialZ targets.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from aggregate import counts, safe_pearson, study00_gene_stats, study00_graph, study00_log_corr, study00_moran, require_validation
from evaluate_resampling_metrics import log_normalize, residualize
from run import input_path
from spatialz_generate import load_study, target_path
from workflow import read_json, write_json
KEYS = ['density_name', 'gap', 'target_index', 'target_slice', 'target_z']
METRICS = ['input_mean_corr', 'input_variance_corr', 'gene_zero_fraction_pearson', 'gene_zero_fraction_wasserstein', 'relative_error_mean', 'expression_wasserstein', 'library_size_wasserstein', 'detected_genes_wasserstein', 'conditional_expression_wasserstein', 'conditional_zero_fraction_wasserstein', 'moran_profile_correlation', 'residual_moran_profile_correlation', 'cell_count_relative_error', 'class_proportion_total_variation', 'missing_target_class_mass', 'extra_generated_class_mass']

def gene_distances(left, right):
    """Equal-mass empirical W1 per gene, including unequal numbers of cells."""
    return np.array([wasserstein_distance(left[:, g], right[:, g]) for g in range(left.shape[1])])

def slice_arrays(adata, dataset):
    values = counts(adata, dataset['counts_layer'])
    xy = np.asarray(adata.obsm[dataset['spatial_key']], dtype=float)
    labels = adata.obs[dataset['label_key']]
    if values.ndim != 2 or adata.n_obs <= 6:
        raise ValueError('slice requires a count matrix and more than six cells for its spatial graph')
    if (values < 0).any() or not np.isfinite(values).all():
        raise ValueError('expression must be finite and nonnegative')
    if xy.shape != (adata.n_obs, 2) or not np.isfinite(xy).all() or labels.isna().any():
        raise ValueError('slice coordinates or class labels are invalid')
    return (values, xy, labels.astype(str).to_numpy())

def prepare_observed(adata, dataset):
    values, xy, labels = slice_arrays(adata, dataset)
    normalized, library, detected = log_normalize(values)
    stats = study00_gene_stats(values)
    top = np.argsort(stats[0], kind='stable')[-min(300, adata.n_vars):][::-1]
    graph = study00_graph(xy, neighbors=6)
    return dict(values=values, labels=labels, normalized=normalized, library=library, detected=detected, stats=stats, top=top, moran=study00_moran(values, top, graph), residual_moran=study00_moran(residualize(values[:, top], labels), np.arange(len(top)), graph))

def evaluate_slice(observed, candidate, dataset, genes):
    """Use separate class memberships and spatial graphs for each cell set."""
    values, xy, labels = slice_arrays(candidate, dataset)
    real, real_labels = (observed['values'], observed['labels'])
    if values.shape[1] != real.shape[1]:
        raise ValueError('gene panels differ')
    normalized, library, detected = log_normalize(values)
    mean, variance, zero = study00_gene_stats(values)
    real_mean, real_variance, real_zero = observed['stats']
    expression_w1 = gene_distances(observed['normalized'], normalized)
    graph = study00_graph(xy, neighbors=6)
    top = observed['top']
    moran = study00_moran(values, top, graph)
    residual_moran = study00_moran(residualize(values[:, top], labels), np.arange(len(top)), graph)
    metrics = {'n_target_cells': len(real), 'n_generated_cells': len(values), 'n_genes': len(genes), 'input_mean_corr': study00_log_corr(real_mean, mean), 'input_variance_corr': study00_log_corr(real_variance, variance), 'gene_zero_fraction_pearson': safe_pearson(real_zero, zero), 'gene_zero_fraction_wasserstein': float(wasserstein_distance(real_zero, zero)), 'relative_error_mean': float(np.mean(np.abs(real_mean - mean) / (real_mean + 1e-10))), 'expression_wasserstein': float(np.median(expression_w1)), 'library_size_wasserstein': float(wasserstein_distance(observed['library'], library)), 'detected_genes_wasserstein': float(wasserstein_distance(observed['detected'], detected)), 'moran_profile_correlation': safe_pearson(observed['moran'], moran), 'residual_moran_profile_correlation': safe_pearson(observed['residual_moran'], residual_moran), 'moran_valid_genes': int(np.sum(np.isfinite(observed['moran']) & np.isfinite(moran))), 'residual_moran_valid_genes': int(np.sum(np.isfinite(observed['residual_moran']) & np.isfinite(residual_moran))), 'cell_count_relative_error': abs(len(values) - len(real)) / len(real)}
    conditional_w1 = np.zeros(real.shape[1])
    conditional_zero = 0.0
    class_rows = []
    for label in sorted(set(real_labels) | set(labels)):
        mask_real, mask_generated = (real_labels == label, labels == label)
        nr, ng = (int(mask_real.sum()), int(mask_generated.sum()))
        row = {'class': label, 'n_target_cells': nr, 'n_generated_cells': ng, 'target_proportion': nr / len(real), 'generated_proportion': ng / len(values), 'expression_wasserstein': np.nan, 'zero_fraction_wasserstein': np.nan, 'support': 'shared' if nr and ng else 'missing_generated' if nr else 'extra_generated'}
        if nr and ng:
            distance = gene_distances(observed['normalized'][mask_real], normalized[mask_generated])
            zero_distance = wasserstein_distance((real[mask_real] == 0).mean(axis=0), (values[mask_generated] == 0).mean(axis=0))
            conditional_w1 += row['target_proportion'] * distance
            conditional_zero += row['target_proportion'] * zero_distance
            row.update(expression_wasserstein=float(np.median(distance)), zero_fraction_wasserstein=float(zero_distance))
        class_rows.append(row)
    missing_mass = sum((r['target_proportion'] for r in class_rows if r['support'] == 'missing_generated'))
    metrics.update({'class_proportion_total_variation': 0.5 * sum((abs(r['target_proportion'] - r['generated_proportion']) for r in class_rows)), 'missing_target_class_mass': missing_mass, 'extra_generated_class_mass': sum((r['generated_proportion'] for r in class_rows if r['support'] == 'extra_generated')), 'conditional_expression_wasserstein': float(np.median(conditional_w1)) if missing_mass == 0 else np.nan, 'conditional_zero_fraction_wasserstein': float(conditional_zero) if missing_mass == 0 else np.nan})
    profiles = pd.DataFrame({'gene': genes, 'target_mean': real_mean, 'generated_mean': mean, 'target_variance': real_variance, 'generated_variance': variance, 'target_zero_fraction': real_zero, 'generated_zero_fraction': zero, 'expression_wasserstein': expression_w1})
    for name, profile in [('target_moran', observed['moran']), ('generated_moran', moran), ('target_residual_moran', observed['residual_moran']), ('generated_residual_moran', residual_moran)]:
        profiles[name] = np.nan
        profiles.loc[top, name] = profile
    return (metrics, pd.DataFrame(class_rows), profiles)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feast-root', type=Path, required=True)
    parser.add_argument('--spatialz-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh evaluation directory')
    parser.add_argument('--gap', type=int, choices=(3, 5, 10), nargs='+')
    parser.add_argument('--target-id', type=int, nargs='+')
    return parser.parse_args()

def main():
    args = parse_args()
    study, plan = load_study(args.feast_root)
    require_validation(args.feast_root)
    rows = [r for d in plan['densities'].values() for r in d['targets'] if (args.gap is None or int(r['gap']) in args.gap) and (args.target_id is None or int(r['target_slice']) in args.target_id)]
    if not rows or (args.target_id is not None and set(args.target_id) - {int(r['target_slice']) for r in rows}):
        raise ValueError('requested targets are not held out in the selected densities')
    prefix, dataset = (study['dataset']['filename_prefix'], study['dataset'])
    run_records = []
    for row in rows:
        folder = target_path(args.spatialz_root, row, prefix)
        record = read_json(folder / 'run.json')
        if record['status'] != 'completed' or not (folder / 'generated.h5ad').is_file():
            raise ValueError(f'{folder}: SpatialZ generation has not completed')
        if Path(record['feast_root']).resolve() != args.feast_root.resolve():
            raise ValueError(f'{folder}: SpatialZ used a different Study 06 plan')
        if any((record['target'][key] != row[key] for key in record['target'])):
            raise ValueError(f'{folder}: SpatialZ target metadata differs from the selected plan')
        run_records.append(record)
    if any((r['recipe'] != run_records[0]['recipe'] for r in run_records)):
        raise ValueError('selected SpatialZ targets use different recipes')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'gene_profiles').mkdir()
    records, classes = ([], [])
    for row in rows:
        target = ad.read_h5ad(input_path(plan, study, int(row['target_slice'])))
        prepared = prepare_observed(target, dataset)
        keys = {key: row[key] for key in KEYS}
        for method, root in [('feast', args.feast_root), ('spatialz', args.spatialz_root)]:
            candidate = ad.read_h5ad(target_path(root, row, prefix) / 'generated.h5ad')
            if not candidate.var_names.equals(target.var_names):
                raise ValueError(f'{method}: generated and target gene identities/order differ')
            metrics, class_rows, profiles = evaluate_slice(prepared, candidate, dataset, list(target.var_names))
            records.append({**keys, 'method': method, **metrics})
            for key, value in {**keys, 'method': method}.items():
                class_rows[key] = value
            classes.append(class_rows)
            profiles.to_csv(args.output_dir / 'gene_profiles' / f"{row['density_name']}_{int(row['target_slice']):03d}_{method}.csv", index=False)
        print(f"Evaluated {row['density_name']} slice {row['target_slice']}", flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(args.output_dir / 'metrics.csv', index=False)
    pd.concat(classes, ignore_index=True).to_csv(args.output_dir / 'class_metrics.csv', index=False)
    summary = []
    for (density, method), group in metrics.groupby(['density_name', 'method'], sort=False):
        for metric in METRICS:
            finite = group.loc[np.isfinite(group[metric]), metric]
            summary.append({'density_name': density, 'method': method, 'metric': metric, 'n_targets': len(group), 'n_defined': len(finite), 'mean': finite.mean(), 'median': finite.median()})
    pd.DataFrame(summary).to_csv(args.output_dir / 'summary.csv', index=False)
    matched = metrics[metrics.method == 'feast'].merge(metrics[metrics.method == 'spatialz'], on=KEYS, suffixes=('_feast', '_spatialz'), validate='one_to_one')
    comparisons = []
    for row in matched.to_dict('records'):
        for metric in METRICS:
            comparisons.append({**{key: row[key] for key in KEYS}, 'metric': metric, 'feast': row[f'{metric}_feast'], 'spatialz': row[f'{metric}_spatialz'], 'spatialz_minus_feast': row[f'{metric}_spatialz'] - row[f'{metric}_feast']})
    pd.DataFrame(comparisons).to_csv(args.output_dir / 'comparison.csv', index=False)
    write_json(args.output_dir / 'evaluation.json', {'status': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat(), 'entry_point': str(Path(__file__).resolve()), 'feast_root': str(args.feast_root.resolve()), 'spatialz_root': str(args.spatialz_root.resolve()), 'targets_evaluated': len(rows), 'full_336_target_benchmark': len(rows) == 336, 'recipe': run_records[0]['recipe'], 'metrics': METRICS, 'expression_distribution_scale': 'per-cell counts / library * 10000, then log1p', 'moran_scale': 'raw counts; own directed row-normalized 6-NN graph; target top 300 mean-expression genes', 'missing_class_policy': 'per-class distances undefined; full conditional score undefined if any target class absent', 'feast_information': 'five primary references plus declared donors; observed target XY and class labels', 'spatialz_information': 'two bracketing references; target z only; native generated cells and labels', 'targets_are_biological_replicates': False})
if __name__ == '__main__':
    main()
