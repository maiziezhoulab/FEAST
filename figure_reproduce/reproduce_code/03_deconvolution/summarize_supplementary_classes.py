"""Descriptive class breakdown of the saved Study 03 predictions; no model runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

from manifests import read_manifest
from scipy.stats import pearsonr
from score import read_table, with_other

def summarize(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulations = read_manifest(run_dir / 'simulation_manifest.csv')
    methods = read_manifest(run_dir / 'method_manifest.csv')
    audit = pd.read_csv(run_dir / 'scores/support_audit.csv', dtype={'slice_id': str})
    existing = pd.read_csv(run_dir / 'scores/deconvolution_scores.csv')
    rows, support_rows = ([], [])
    for sim in simulations.itertuples(index=False):
        support = audit.loc[audit.pair_id.eq(sim.pair_id)].iloc[0]
        retained = pd.Index(json.loads(support.retained_cell_class_order_json))
        truth = read_table(Path(sim.truth_path), 'truth')
        predictions, paths = ({}, {})
        for method in ('rctd', 'cell2location'):
            entry = methods.loc[methods.pair_id.eq(sim.pair_id) & methods.method.eq(method)].iloc[0]
            assert entry.status == 'ok'
            paths[method] = entry.output_path
            predictions[method] = read_table(Path(entry.output_path), method)
            assert set(predictions[method].columns) == set(retained)
        positive = predictions['rctd'].index
        assert positive.is_unique and len(positive) == support.n_positive_spots
        assert truth.index.equals(predictions['cell2location'].index)
        assert len(truth) == support.n_full_spots
        assert np.allclose(predictions['cell2location'].drop(index=positive).sum(axis=1), 0)
        assert np.allclose(truth.loc[positive].sum(axis=1), 1)
        target = with_other(truth.loc[positive], retained)
        for method, frame in predictions.items():
            assert np.allclose(frame.loc[positive].sum(axis=1), 1, atol=1e-05)
            predicted = with_other(frame.loc[positive], retained)
            correlations, squared_errors = ([], [])
            for i, name in enumerate([*retained, 'Other']):
                left, right = (target[:, i], predicted[:, i])
                is_other = name == 'Other'
                constant = np.std(left) == 0 or np.std(right) == 0
                correlation = np.nan if is_other or constant else float(pearsonr(left, right).statistic)
                mse = float(np.mean((left - right) ** 2))
                correlations.append(correlation)
                squared_errors.append(mse)
                status = 'structural_other_not_estimated' if is_other else 'constant_vector' if constant else 'defined'
                rows.append(dict(panel_id='C', pair_id=sim.pair_id, slice_id=sim.slice_id, resolution=sim.resolution, method=method, cell_class=name, class_order=i, is_other=is_other, n_scored_spots=len(positive), truth_mean_proportion=float(left.mean()), pearson=correlation, pearson_display=correlation, pearson_status=status, rmse=np.sqrt(mse), source_path=paths[method], truth_path=sim.truth_path))
            original = existing.loc[existing.pair_id.eq(sim.pair_id) & existing.method.eq(method)].iloc[0]
            np.testing.assert_allclose(np.nanmean(correlations), original.pearson_mean, atol=1e-10)
            np.testing.assert_allclose(np.sqrt(np.mean(squared_errors)), original.rmse, atol=1e-10)
        data = ad.read_h5ad(sim.simulation_path, backed='r')
        try:
            names = data.obs_names.astype(str)
            assert names.equals(pd.Index([str(i) for i in range(data.n_obs)]))
            scored_names = pd.Index(['spot_' + name for name in names])
            assert scored_names.equals(truth.index)
            counts_by_spot = pd.Series(data.obs['n_source_cells'].to_numpy(), index=scored_names)
            assert counts_by_spot.index.is_unique and positive.isin(counts_by_spot.index).all()
            counts = counts_by_spot.loc[positive].to_numpy()
        finally:
            data.file.close()
        support_rows.append(dict(panel_id='C', pair_id=sim.pair_id, slice_id=sim.slice_id, resolution=sim.resolution, n_full_spots=int(support.n_full_spots), n_scored_spots=len(positive), n_zero_library_spots=int(support.n_zero_library_spots), n_retained_cell_classes=len(retained), cells_per_scored_spot_mean=float(counts.mean()), cells_per_scored_spot_median=float(np.median(counts)), cells_per_scored_spot_q1=float(np.quantile(counts, 0.25)), cells_per_scored_spot_q3=float(np.quantile(counts, 0.75)), truth_other_mean_proportion=float(target[:, -1].mean()), denominator='positive_library_spots_spot_weighted', spot_identity_mapping='score.py: verified sequential obs name i mapped to spot_i; then identifier join', simulation_path=sim.simulation_path, truth_path=sim.truth_path))
    classes, support = (pd.DataFrame(rows), pd.DataFrame(support_rows))
    assert len(classes) == 156 and len(support) == 6
    assert not classes.duplicated(['pair_id', 'method', 'cell_class']).any()
    return (classes, support)

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=root / '03_deconvolution/outputs/cell_class')
    parser.add_argument('--output-dir', type=Path, default=root / '03_deconvolution/outputs/supplementary_diagnostics_S2')
    args = parser.parse_args()
    classes, support = summarize(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes.to_csv(args.output_dir / 'deconvolution_classes.csv', index=False)
    support.to_csv(args.output_dir / 'deconvolution_support.csv', index=False)
    print(f'Saved {len(classes)} category rows and {len(support)} support rows to {args.output_dir}; all 12 aggregate scores reconciled.')
if __name__ == '__main__':
    main()
