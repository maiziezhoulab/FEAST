"""Independently validate Study 03 cell-class scores and their input lineage."""
from __future__ import annotations
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

from manifests import read_manifest
import yaml
from scipy.special import rel_entr
from scipy.stats import pearsonr
METHODS = ('rctd', 'cell2location')
METRICS = ('jsd_mean', 'jsd_median', 'pearson_mean', 'pearson_median', 'rmse')

def require_path(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f'missing {label}: {path}')

def composition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    frame = frame.apply(pd.to_numeric, errors='raise').astype(np.float64)
    if frame.empty or not frame.index.is_unique or (not frame.columns.is_unique) or (not np.isfinite(frame.to_numpy()).all()) or (frame.to_numpy().min(initial=0) < -1e-10):
        raise RuntimeError(f'invalid composition: {path}')
    return frame.clip(lower=0)

def libraries(matrix) -> np.ndarray:
    if hasattr(matrix, 'toarray'):
        data = np.asarray(matrix.data, dtype=np.float64)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        values = np.asarray(matrix, dtype=np.float64)
        data, totals = (values.ravel(), values.sum(axis=1))
    if not np.isfinite(data).all() or data.min(initial=0) < 0 or np.max(np.abs(data - np.rint(data)), initial=0) > 1e-06:
        raise RuntimeError('simulation is not finite raw integer count data')
    return totals

def add_other(frame: pd.DataFrame, retained: pd.Index) -> np.ndarray:
    values = frame.loc[:, retained].to_numpy(dtype=np.float64)
    result = np.column_stack([values, np.clip(1 - values.sum(axis=1), 0, None)])
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals[:, 0] <= 0):
        raise RuntimeError('zero-mass scored composition')
    return result / totals

def recompute(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    midpoint = (truth + prediction) / 2
    divergence = 0.5 * (rel_entr(truth, midpoint).sum(axis=1) + rel_entr(prediction, midpoint).sum(axis=1)) / np.log(2)
    correlations = []
    # The last column is Other, which contributes to divergence and RMSE.
    for index in range(truth.shape[1] - 1):
        left, right = (truth[:, index], prediction[:, index])
        correlations.append(np.nan if np.std(left) == 0 or np.std(right) == 0 else float(pearsonr(left, right).statistic))
    result = {'jsd_mean': float(divergence.mean()), 'jsd_median': float(np.median(divergence)), 'pearson_mean': float(np.nanmean(correlations)), 'pearson_median': float(np.nanmedian(correlations)), 'rmse': float(np.sqrt(np.mean((truth - prediction) ** 2)))}
    if not all((math.isfinite(value) for value in result.values())):
        raise RuntimeError('independent scoring produced a non-finite result')
    return result

def one(frame: pd.DataFrame, pair_id: str, method: str | None=None) -> pd.Series:
    selected = frame[frame['pair_id'].astype(str) == pair_id]
    if method is not None:
        selected = selected[selected['method'].astype(str) == method]
    if len(selected) != 1:
        raise RuntimeError(f'expected one row for {pair_id}, {method}')
    return selected.iloc[0]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scores-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    config = yaml.safe_load(args.config.read_text())
    if config.get('cell_type_key') != 'cell_class' or int(config['rctd']['min_cells_per_type']) != int(config['cell2location']['min_cells_per_type']):
        raise RuntimeError('configuration is not the exact matched cell_class design')
    minimum = int(config['rctd']['min_cells_per_type'])
    simulations_path = args.run_dir / 'simulation_manifest.csv'
    methods_path = args.run_dir / 'method_manifest.csv'
    scores_path = args.scores_dir / 'deconvolution_scores.csv'
    support_path = args.scores_dir / 'support_audit.csv'
    lineage_path = args.scores_dir / 'input_lineage.csv'
    provenance_path = args.scores_dir / 'scoring_provenance.json'
    simulations = read_manifest(simulations_path)
    methods = read_manifest(methods_path)
    scores = pd.read_csv(scores_path, dtype={'slice_id': str})
    support = pd.read_csv(support_path, dtype={'slice_id': str})
    lineage = pd.read_csv(lineage_path)
    provenance = json.loads(provenance_path.read_text())
    expected_pairs = {f'{slice_id}__resolution_{float(resolution):g}' for slice_id in config['slices'] for resolution in config['resolutions']}
    expected_score_keys = {(pair, method) for pair in expected_pairs for method in METHODS}
    observed_score_keys = set(zip(scores['pair_id'], scores['method'], strict=True))
    if len(simulations) != 6 or len(methods) != 12 or len(scores) != 12 or (len(support) != 6) or (set(simulations['pair_id']) != expected_pairs) or (set(methods['pair_id']) != expected_pairs) or (set(support['pair_id']) != expected_pairs) or (observed_score_keys != expected_score_keys) or methods.duplicated(['pair_id', 'method']).any() or scores.duplicated(['pair_id', 'method']).any() or (not simulations['status'].eq('ok').all()) or (not methods['status'].eq('ok').all()) or (not scores['status'].eq('ok').all()):
        raise RuntimeError('artifacts do not form the exact successful registered matrix')
    for output_name in provenance['outputs']:
        require_path(args.scores_dir / output_name, output_name)
    for assertion in ('method_class_sets_exactly_equal', 'rctd_positive_spot_order_exact', 'cell2location_full_spot_order_exact', 'cell2location_zero_rows_zero', 'other_residual_included'):
        if not support[assertion].astype(bool).all():
            raise RuntimeError(f'recorded support assertion is false: {assertion}')
    for row in lineage.itertuples(index=False):
        require_path(args.run_dir / row.path, f'lineage {row.pair_id} {row.artifact}')
    recomputed_rows = []
    recomputed_support = []
    for simulation_row in simulations.itertuples(index=False):
        simulation_record = pd.Series(simulation_row._asdict())
        pair_id = str(simulation_record['pair_id'])
        simulation_path = Path(str(simulation_record['simulation_path']))
        truth_path = Path(str(simulation_record['truth_path']))
        reference_path = Path(str(simulation_record['reference_path']))
        for path in (simulation_path, truth_path, reference_path):
            require_path(path, f'{pair_id} source')
        simulation = ad.read_h5ad(simulation_path, backed='r')
        try:
            totals = libraries(simulation.X)
            spots = pd.Index([f'spot_{index}' for index in range(simulation.n_obs)])
            truth_names = pd.Index(simulation.uns['cell_type_names']).astype(str)
            embedded = dict(simulation.uns['feast_reproduce'])
        finally:
            simulation.file.close()
        if embedded.get('configuration_id') != config['configuration_id']:
            raise RuntimeError(f'embedded configuration differs: {pair_id}')
        positive = spots[totals > 0]
        zero = spots[totals <= 0]
        reference = ad.read_h5ad(reference_path, backed='r')
        try:
            counts = reference.obs['cell_class'].astype(str).value_counts()
        finally:
            reference.file.close()
        retained_set = set(counts[counts >= minimum].index.astype(str))
        retained = truth_names[truth_names.isin(retained_set)]
        if set(retained) != retained_set:
            raise RuntimeError(f'reference/truth retained support differs: {pair_id}')
        truth = composition(truth_path)
        if not truth.index.equals(spots) or not truth.columns.equals(truth_names):
            raise RuntimeError(f'truth identity differs: {pair_id}')
        truth_scored = add_other(truth.loc[positive], retained)
        for method in METHODS:
            method_row = one(methods, pair_id, method)
            output_path = Path(str(method_row['output_path']))
            metadata_path = Path(str(method_row['metadata_path']))
            require_path(output_path, f'{pair_id} {method}')
            require_path(metadata_path, f'{pair_id} {method} metadata')
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('status') != 'validated_success' or metadata.get('method') != method:
                raise RuntimeError(f'method provenance is not closed: {pair_id} {method}')
            prediction = composition(output_path)
            expected_index = positive if method == 'rctd' else spots
            if not prediction.index.equals(expected_index) or set(prediction.columns) != retained_set or (not np.allclose(prediction.loc[positive].sum(axis=1), 1, atol=1e-05)):
                raise RuntimeError(f'prediction support differs: {pair_id} {method}')
            if method == 'rctd':
                execution_valid = int(metadata.get('min_cells_per_type', -1)) == minimum
            else:
                label_filter = metadata.get('reference_label_filter', {})
                execution_valid = bool(metadata.get('actual_accelerator') == 'cuda' and metadata.get('cuda_execution_verified') is True and (label_filter.get('cell_type_key') == 'cell_class') and (int(label_filter.get('min_cells_per_type', -1)) == minimum) and (set(label_filter.get('retained_cell_types', [])) == retained_set) and np.allclose(prediction.loc[zero].sum(axis=1), 0, atol=1e-12))
            if not execution_valid:
                raise RuntimeError(f'execution contract differs: {pair_id} {method}')
            recomputed_rows.append({'pair_id': pair_id, 'method': method, **recompute(truth_scored, add_other(prediction.loc[positive, retained], retained))})
        recomputed_support.append({'pair_id': pair_id, 'n_full_spots': len(spots), 'n_positive_spots': len(positive), 'n_zero_library_spots': len(zero), 'n_truth_cell_classes': len(truth_names), 'n_retained_cell_classes': len(retained)})
    recomputed = pd.DataFrame(recomputed_rows)
    merged = scores.merge(recomputed, on=['pair_id', 'method'], suffixes=('_recorded', '_recomputed'), validate='one_to_one')
    for metric in METRICS:
        if not np.allclose(merged[f'{metric}_recorded'], merged[f'{metric}_recomputed'], atol=1e-11, rtol=1e-11):
            raise RuntimeError(f'recorded {metric} does not independently reproduce')
    support_check = support.merge(pd.DataFrame(recomputed_support), on='pair_id', suffixes=('_recorded', '_recomputed'))
    for field in ('n_full_spots', 'n_positive_spots', 'n_zero_library_spots', 'n_truth_cell_classes', 'n_retained_cell_classes'):
        if not support_check[f'{field}_recorded'].equals(support_check[f'{field}_recomputed']):
            raise RuntimeError(f'support audit field does not reproduce: {field}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame([{'status': 'validated', 'configuration_id': config['configuration_id'], 'annotation_key': 'cell_class', 'min_cells_per_class': minimum, 'n_pairs': 6, 'n_method_jobs': 12, 'all_metrics_independently_reproduced': True, 'exact_method_support_verified': True, 'cuda_execution_verified_for_all_cell2location_jobs': True, 'validated_utc': datetime.now(timezone.utc).isoformat()}])
    result.to_csv(args.output, index=False)
    print(f'Saved: {args.output}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
