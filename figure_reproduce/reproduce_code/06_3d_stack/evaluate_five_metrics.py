"""Complete the six-metric benchmark using the historical baseline RNG order.

Run in the Study 06 precision environment. Existing simulations and evaluation
artifacts are read only; --output-dir must be a new directory.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from aggregate import require_validation, study00_log_corr
from conditional_resampling_baseline import prepare_sampling, resample_spots
from evaluate_resampling_metrics import log_normalize
from regional_metrics import read_data, region_plan, center_classes, correlation, WIDTHS
STUDY = Path(__file__).resolve().parent
DEFAULT_FEAST = STUDY / 'outputs/reconstruction'
DEFAULT_SPATIALZ = STUDY / 'outputs/spatialz'
DEFAULT_OUTPUT = STUDY / 'outputs/benchmark'
METHODS = ('feast', 'resampling', 'spatialz')
SLICE_METRICS = ('input_mean_corr', 'input_variance_corr', 'expression_wasserstein', 'residual_moran_profile_correlation', 'region_mae')
KEYS = ['density_name', 'gap', 'target_index', 'target_slice', 'target_z']

def positive_sorted_columns(values):
    """Store sorted positive tails; zero atoms need no sorting or storage."""
    return [np.sort(column[column > 0]) for column in np.ascontiguousarray(values.T)]

def equal_count_w1(observed_tails, candidate, verify=False):
    """Exact empirical W1 for nonnegative samples with equal cell counts."""
    distances = []
    for left, right in zip(observed_tails, positive_sorted_columns(candidate), strict=True):
        if len(left) > len(right):
            left, right = (right, left)
        offset = len(right) - len(left)
        distances.append((right[:offset].sum() + np.abs(left - right[offset:]).sum()) / len(candidate))
    result = np.asarray(distances)
    if verify:
        for g in np.linspace(0, candidate.shape[1] - 1, 12).round().astype(int):
            left = np.pad(observed_tails[g], (len(candidate) - len(observed_tails[g]), 0))
            np.testing.assert_allclose(result[g], wasserstein_distance(left, candidate[:, g]), rtol=1e-12, atol=1e-12)
    return result

def prepare_regions(real, feast, spatialz):
    """Cache observed aggregation without changing the trial definitions."""
    result = {}
    for width in WIDTHS:
        for conditional in (False, True):
            spec = region_plan(real, feast, spatialz, width, conditional)
            keep = spec['shared']
            a = spec['matrices'][0] @ real['x']
            weights = spec['counts'][0, keep]
            weights = weights / weights.sum()
            means = a[keep] / spec['counts'][0, keep, None]
            classes = spec['keys'][keep, 2] if conditional else np.zeros(keep.sum())
            spec.update(observed_sums=a, observed_means=means, weights=weights, classes=classes, observed_centered=center_classes(means, weights, classes))
            result[width, conditional] = spec
    return result

def score_regions(candidate, specs, index):
    rows = []
    for width in WIDTHS:
        row = {'bin_width_mm': width}
        for conditional in (False, True):
            spec = specs[width, conditional]
            b = spec['matrices'][index] @ candidate
            keep = spec['shared']
            br = b[keep] / spec['counts'][index, keep, None]
            ar, w = (spec['observed_means'], spec['weights'])
            ac = spec['observed_centered']
            bc = center_classes(br, w, spec['classes'])
            prefix = 'class_region' if conditional else 'region'
            metrics = {f'{prefix}_mae': np.sum(w[:, None] * np.abs(ar - br), axis=0), f'{prefix}_centered_mae': np.sum(w[:, None] * np.abs(ac - bc), axis=0), f'{prefix}_pattern_pearson': correlation(ac, bc, w)}
            if not conditional:
                a = spec['observed_sums']
                at, bt = (a.sum(axis=0), b.sum(axis=0))
                defined = (at > 0) & (bt > 0)
                tv = np.full(a.shape[1], np.nan)
                tv[defined] = 0.5 * np.abs(a[:, defined] / at[defined] - b[:, defined] / bt[defined]).sum(axis=0)
                metrics['expression_mass_tv'] = tv
                counts = spec['counts']
                row['cell_density_tv'] = 0.5 * np.abs(counts[0] / counts[0].sum() - counts[index] / counts[index].sum()).sum()
            row[f'{prefix}_observed_cell_coverage'] = spec['coverage']
            row[f'{prefix}_n_shared'] = int(keep.sum())
            for name, values in metrics.items():
                finite = values[np.isfinite(values)]
                row[name] = float(np.median(finite)) if len(finite) else np.nan
                row[name + '_n_defined'] = len(finite)
        rows.append(row)
    return rows

def append_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)

def summary_rows(data, metrics, population, unit):
    rows = []
    for (gap, method), frame in data.groupby(['gap', 'method']):
        for metric in metrics:
            values = frame[metric]
            finite = values[np.isfinite(values)]
            rows.append(dict(population=population, gap=gap, method=method, metric=metric, unit=unit, n_total=len(values), n_defined=len(finite), median=finite.median(), q25=finite.quantile(0.25), q75=finite.quantile(0.75)))
    return rows

def trajectories(plan, means, common, output):
    rows = []
    for density, arm in plan['densities'].items():
        gap = int(arm['spec']['gap'])
        ordered = sorted(arm['targets'], key=lambda r: r['target_z'])
        for population in ('common_targets', 'full_arm'):
            selected = [r for r in ordered if population == 'full_arm' or r['target_slice'] in common]
            observed = np.stack([np.log1p(means[gap, r['target_slice']]['observed']) for r in selected])
            z = np.array([r['target_z'] for r in selected])
            assert np.all(np.diff(z) > 0)
            for method, replicate in [('feast', -1), ('spatialz', -1)] + [('resampling', i) for i in range(10)]:
                key = method if replicate == -1 else f'resampling_{replicate}'
                candidate = np.stack([np.log1p(means[gap, r['target_slice']][key]) for r in selected])
                difference = candidate - observed
                rows.append(pd.DataFrame(dict(population=population, gap=gap, method=method, replicate=replicate, gene=plan['gene_names'], n_targets=len(selected), trajectory_pearson=correlation(observed, candidate), trajectory_mae=np.abs(difference).mean(axis=0), z_slope_mae=np.abs(np.diff(difference, axis=0) / np.diff(z)[:, None]).mean(axis=0))))
    frame = pd.concat(rows, ignore_index=True)
    frame['shared_defined'] = False
    common_mask = frame.population == 'common_targets'
    valid = frame[common_mask].groupby('gene').trajectory_pearson.apply(lambda v: np.isfinite(v).all())
    frame.loc[common_mask, 'shared_defined'] = frame.loc[common_mask, 'gene'].map(valid)
    for gap in (3, 5, 10):
        mask = (frame.population == 'full_arm') & (frame.gap == gap)
        valid = frame[mask].groupby('gene').trajectory_pearson.apply(lambda v: np.isfinite(v).all())
        frame.loc[mask, 'shared_defined'] = frame.loc[mask, 'gene'].map(valid)
    frame.to_csv(output / 'trajectory_repeat_metrics.csv.gz', index=False)
    keys = ['population', 'gap', 'method', 'gene', 'n_targets', 'shared_defined']
    metrics = ['trajectory_pearson', 'trajectory_mae', 'z_slope_mae']
    averaged = frame.groupby(keys, as_index=False)[metrics].mean()
    averaged.to_csv(output / 'trajectory_gene_metrics.csv', index=False)
    return averaged

def evaluate(args):
    plan = json.loads((args.feast_root / 'plan.json').read_text())
    require_validation(args.feast_root)
    current = pd.read_csv(args.spatialz_root / 'evaluation/metrics.csv')
    history_path = args.feast_root / 'baselines/five_reference_conditional_resampling/method_metrics.csv'
    history = pd.read_csv(history_path)
    historical = history[history.method == 'conditional_whole_spot_resampling'].set_index(['gap', 'target_slice', 'replicate'])
    assert len(historical) == 3360 and historical.index.is_unique
    native = current.set_index(['gap', 'target_slice', 'method'])
    assert len(native) == 672 and native.index.is_unique
    common = sorted(set.intersection(*(set(a['target_pool']) for a in plan['densities'].values())))
    assert len(common) == 76
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), entry_point=str(Path(__file__).resolve()), feast_root=str(args.feast_root.resolve()), spatialz_root=str(args.spatialz_root.resolve()), historical_baseline=str(history_path.resolve()), common_targets=common, full_target_counts={3: 95, 5: 114, 10: 127}, genes=len(plan['gene_names']), resampling_seeds=list(range(2026, 2036)), resampling_rng='Continuous streams across gaps 3/5/10, then available-slice order; historical replay', region_widths_mm=WIDTHS, main_region_width_mm=0.2, region_origin=[0, 0], min_cells_per_region=5, region_weighting='observed cell counts on common supported regions', normalization='counts/library*10000 then log1p for W1 and regions; raw-count means then log1p for trajectories', mean_correlation='existing study00_log_corr: paired means >1e-10, log1p Pearson, existing <10-gene behavior', trajectory_population='held-out targets only; no retained anchors', trajectory_support='main: common finite genes across gaps/methods/repeats; full-arm: common finite genes within gap', presented_metrics=[*SLICE_METRICS, 'trajectory_pearson'], study00_shared_source=str(STUDY.parent / '00_simulator_benchmark/score.py'), variance_correlation='Study 00 raw-count population variance followed by paired-positive log1p Pearson; input_variance_corr', resampling_aggregation='mean of per-repeat scores per target/gene, then median and IQR across targets/genes', input_asymmetry='FEAST/resampling: target XY/classes and five references plus donors; SpatialZ: two references and inferred cells', threads=2, biological_replicate_inference=False)
    status_path = args.output_dir / 'evaluation.json'
    status_path.write_text(json.dumps(metadata, indent=2))
    rngs = [np.random.default_rng(seed) for seed in range(2026, 2036)]
    means, slice_rows, regional_rows, parity = ({}, [], [], [])
    for density, arm in sorted(plan['densities'].items(), key=lambda item: int(item[1]['spec']['gap'])):
        gap = int(arm['spec']['gap'])
        lookup = {int(r['target_slice']): r for r in arm['targets']}
        for sid in map(int, plan['available_slices']):
            if sid not in lookup:
                continue
            row = lookup[sid]
            start = time.monotonic()
            name = f'Zhuang-ABCA-1.{sid:03d}'
            genes = plan['gene_names']
            real = read_data(Path(plan['data_dir']) / f'{name}.h5ad', genes)
            feast = read_data(args.feast_root / 'targets' / density / name / 'generated.h5ad', genes)
            spatialz = read_data(args.spatialz_root / 'targets' / density / name / 'generated.h5ad', genes)
            for key in ('ids', 'xy', 'labels'):
                assert np.array_equal(real[key], feast[key]), f'FEAST target {key} mismatch'
            profiles = {m: pd.read_csv(args.spatialz_root / 'evaluation/gene_profiles' / f'{density}_{sid:03d}_{m}.csv') for m in ('feast', 'spatialz')}
            for table in profiles.values():
                assert table.gene.tolist() == genes
            np.testing.assert_array_equal(profiles['feast'].target_mean, profiles['spatialz'].target_mean)
            raw_means = dict(observed=profiles['feast'].target_mean.to_numpy(), feast=profiles['feast'].generated_mean.to_numpy(), spatialz=profiles['spatialz'].generated_mean.to_numpy())
            specs = prepare_regions(real, feast, spatialz)
            tails = positive_sorted_columns(real['x'])
            groups, _ = prepare_sampling(row, arm, plan['data_dir'], 'Zhuang-ABCA-1', 'class', 'counts', real['labels'], genes)
            local_slices, local_regions = ([], [])
            keys = {k: row[k] for k in KEYS}
            for method, replicate in [('feast', -1), ('spatialz', -1)] + [('resampling', i) for i in range(10)]:
                if method == 'resampling':
                    generated = resample_spots(groups, real['x'].shape, rngs[replicate])
                    raw_mean = generated.mean(axis=0)
                    raw_means[f'resampling_{replicate}'] = raw_mean
                    expected = historical.loc[gap, sid, replicate]
                    replay_mean = study00_log_corr(raw_means['observed'], raw_mean)
                    np.testing.assert_allclose(replay_mean, expected.input_mean_corr, rtol=1e-09, atol=1e-10)
                    normalized, _, _ = log_normalize(generated)
                    del generated
                    w1 = equal_count_w1(tails, normalized, verify=replicate == 0)
                    metrics = {name: expected[name] for name in current.columns if name in expected.index and name not in KEYS + ['method']}
                    metrics['expression_wasserstein'] = float(np.median(w1))
                    candidate, index = (normalized, 0)
                    parity.append(dict(gap=gap, target_slice=sid, replicate=replicate, historical_input_mean_corr=float(expected.input_mean_corr), replay_input_mean_corr=replay_mean, absolute_difference=abs(replay_mean - expected.input_mean_corr)))
                else:
                    metrics = native.loc[gap, sid, method].to_dict()
                    metrics = {k: v for k, v in metrics.items() if k not in KEYS}
                    candidate, index = (feast['x'], 1) if method == 'feast' else (spatialz['x'], 2)
                regional = score_regions(candidate, specs, index)
                primary = next((r for r in regional if r['bin_width_mm'] == 0.2))
                local_slices.append({**keys, 'method': method, 'replicate': replicate, **metrics, 'region_mae': primary['region_mae']})
                local_regions.extend(({**keys, 'method': method, 'replicate': replicate, **r} for r in regional))
            columns = KEYS + ['method', 'replicate'] + [c for c in current.columns if c not in KEYS + ['method']] + ['region_mae']
            local_frame = pd.DataFrame(local_slices).reindex(columns=columns)
            path = args.output_dir / 'slice_repeat_metrics.csv'
            local_frame.to_csv(path, mode='a', header=not path.exists(), index=False)
            append_csv(local_regions, args.output_dir / 'regional_repeat_metrics.csv')
            slice_rows.extend(local_slices)
            regional_rows.extend(local_regions)
            means[gap, sid] = raw_means
            print(f'[{len(means)}/336] gap {gap}, slice {sid}: {time.monotonic() - start:.1f}s; 10 replay checks passed', flush=True)
    slices = pd.DataFrame(slice_rows)
    assert len(slices) == 336 * 12 and (not slices.duplicated(['gap', 'target_slice', 'method', 'replicate']).any())
    np.savez_compressed(args.output_dir / 'slice_gene_means.npz', genes=np.array(plan['gene_names']), **{f'gap{g}_slice{s}_{m}': v for (g, s), data in means.items() for m, v in data.items()})
    pd.DataFrame(parity).to_csv(args.output_dir / 'baseline_replay_checks.csv', index=False)
    keys = KEYS + ['method']
    numeric = [c for c in slices.select_dtypes(include='number') if c not in KEYS + ['replicate']]
    target = slices.groupby(keys, as_index=False)[numeric].mean()
    target['common_target'] = target.target_slice.isin(common)
    assert target.groupby(['gap', 'method']).size().to_dict() == {(g, m): n for g, n in [(3, 95), (5, 114), (10, 127)] for m in METHODS}
    target.to_csv(args.output_dir / 'target_metrics.csv', index=False)
    regional = pd.DataFrame(regional_rows)
    keys = KEYS + ['method', 'bin_width_mm']
    numeric = [c for c in regional.select_dtypes(include='number') if c not in keys + ['replicate']]
    regional = regional.groupby(keys, as_index=False)[numeric].mean()
    regional['common_target'] = regional.target_slice.isin(common)
    regional.to_csv(args.output_dir / 'regional_target_metrics.csv', index=False)
    trajectory = trajectories(plan, means, common, args.output_dir)
    summaries = []
    for population, selected in [('common_targets', target[target.common_target]), ('full_arm', target)]:
        summaries.extend(summary_rows(selected, SLICE_METRICS, population, 'target_slice'))
        genes = trajectory[(trajectory.population == population) & trajectory.shared_defined]
        summaries.extend(summary_rows(genes, ['trajectory_pearson'], population, 'gene'))
    pd.DataFrame(summaries).to_csv(args.output_dir / 'summary.csv', index=False)
    metadata.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(), targets_evaluated=len(means), baseline_replay_checks=len(parity), max_replay_difference=max((p['absolute_difference'] for p in parity)), w1_scipy_checks=336 * 12, trajectory_defined_main_genes=int(trajectory[(trajectory.population == 'common_targets') & trajectory.shared_defined].gene.nunique()))
    status_path.write_text(json.dumps(metadata, indent=2))
    print('Completed:', args.output_dir, flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feast-root', type=Path, default=DEFAULT_FEAST)
    parser.add_argument('--spatialz-root', type=Path, default=DEFAULT_SPATIALZ)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        evaluate(args)
if __name__ == '__main__':
    main()
