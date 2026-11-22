"""Score the versioned Study 03 cell-class rerun on exact shared support."""
from __future__ import annotations
import argparse
import json
import math
import os
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
        raise FileNotFoundError(f'missing {label}: {path}')

def read_table(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.empty or not frame.index.is_unique or (not frame.columns.is_unique):
        raise RuntimeError(f'{label} has invalid row or column support')
    frame = frame.apply(pd.to_numeric, errors='raise').astype(np.float64)
    values = frame.to_numpy()
    if not np.isfinite(values).all() or values.min(initial=0) < -1e-10:
        raise RuntimeError(f'{label} contains invalid proportions')
    return frame.clip(lower=0)

def integer_libraries(matrix, label: str) -> np.ndarray:
    if hasattr(matrix, 'toarray'):
        data = np.asarray(matrix.data, dtype=np.float64)
        libraries = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        values = np.asarray(matrix, dtype=np.float64)
        data, libraries = (values.ravel(), values.sum(axis=1))
    if not np.isfinite(data).all() or data.min(initial=0) < 0:
        raise RuntimeError(f'{label} contains invalid counts')
    if np.max(np.abs(data - np.rint(data)), initial=0) > 1e-06:
        raise RuntimeError(f'{label} is not raw integer count data')
    return libraries

def with_other(frame: pd.DataFrame, retained: pd.Index) -> np.ndarray:
    values = frame.loc[:, retained].to_numpy(np.float64)
    residual = np.clip(1.0 - values.sum(axis=1), 0.0, None)
    result = np.column_stack([values, residual])
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals[:, 0] <= 0):
        raise RuntimeError('a scored composition has zero total mass')
    return result / totals

def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    midpoint = 0.5 * (truth + prediction)
    jsd = 0.5 * (rel_entr(truth, midpoint).sum(axis=1) + rel_entr(prediction, midpoint).sum(axis=1)) / np.log(2)
    correlations = []
    # Other is the final compositional residual and has no fitted prediction.
    for column in range(truth.shape[1] - 1):
        left, right = (truth[:, column], prediction[:, column])
        correlations.append(np.nan if np.std(left) == 0 or np.std(right) == 0 else float(pearsonr(left, right).statistic))
    result = {'jsd_mean': float(jsd.mean()), 'jsd_median': float(np.median(jsd)), 'pearson_mean': float(np.nanmean(correlations)), 'pearson_median': float(np.nanmedian(correlations)), 'rmse': float(np.sqrt(np.mean((truth - prediction) ** 2)))}
    if not all((math.isfinite(value) for value in result.values())):
        raise RuntimeError('scoring produced a non-finite metric')
    return result

def one_row(frame: pd.DataFrame, **criteria: object) -> pd.Series:
    selected = frame
    for column, value in criteria.items():
        selected = selected[selected[column].astype(str) == str(value)]
    if len(selected) != 1:
        raise RuntimeError(f'expected one row for {criteria}, found {len(selected)}')
    return selected.iloc[0]

def validated_method(row: pd.Series, simulation_row: pd.Series, method: str) -> tuple[Path, Path, dict[str, object]]:
    if str(row['status']) != 'ok':
        raise RuntimeError(f"method row is not successful: {row['pair_id']} {method}")
    output = Path(str(row['output_path']))
    metadata_path = Path(str(row['metadata_path']))
    require_path(output, f"{row['pair_id']} {method} output")
    require_path(metadata_path, f"{row['pair_id']} {method} metadata")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('status') != 'validated_success' or metadata.get('method') != method:
        raise RuntimeError(f"method provenance is not closed: {row['pair_id']} {method}")
    return (output, metadata_path, metadata)

def score_pair(simulation_row: pd.Series, method_rows: pd.DataFrame, config: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, str]]]:
    pair_id = str(simulation_row['pair_id'])
    slice_id = str(simulation_row['slice_id'])
    resolution = float(simulation_row['resolution'])
    simulation_path = Path(str(simulation_row['simulation_path']))
    truth_path = Path(str(simulation_row['truth_path']))
    reference_path = Path(str(simulation_row['reference_path']))
    for path, label in ((simulation_path, 'simulation'), (truth_path, 'truth'), (reference_path, 'reference')):
        require_path(path, f'{pair_id} {label}')
    simulation = ad.read_h5ad(simulation_path, backed='r')
    try:
        libraries = integer_libraries(simulation.X, f'{pair_id} simulation')
        full_spots = pd.Index([f'spot_{index}' for index in range(simulation.n_obs)])
        truth_names = pd.Index(simulation.uns['cell_type_names']).astype(str)
        embedded = dict(simulation.uns['feast_reproduce'])
    finally:
        simulation.file.close()
    if embedded.get('configuration_id') != str(config['configuration_id']):
        raise RuntimeError(f'wrong embedded configuration: {pair_id}')
    positive_spots = full_spots[libraries > 0]
    zero_spots = full_spots[libraries <= 0]
    reference = ad.read_h5ad(reference_path, backed='r')
    try:
        labels = reference.obs[str(config['cell_type_key'])].astype(str)
        counts = labels.value_counts()
    finally:
        reference.file.close()
    minimum = int(config['rctd']['min_cells_per_type'])
    retained_set = set(counts[counts >= minimum].index.astype(str))
    retained = truth_names[truth_names.isin(retained_set)]
    if not retained_set or set(retained) != retained_set:
        raise RuntimeError(f'retained reference support differs from truth: {pair_id}')
    truth = read_table(truth_path, f'{pair_id} truth')
    if not truth.index.equals(full_spots) or not truth.columns.equals(truth_names):
        raise RuntimeError(f'truth identity/order differs: {pair_id}')
    if not np.allclose(truth.loc[positive_spots].sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError(f'positive truth rows are not normalized: {pair_id}')
    predictions: dict[str, pd.DataFrame] = {}
    metadata_paths: dict[str, Path] = {}
    for method in METHODS:
        row = one_row(method_rows, pair_id=pair_id, method=method)
        output, metadata_path, metadata = validated_method(row, simulation_row, method)
        prediction = read_table(output, f'{pair_id} {method}')
        expected_index = positive_spots if method == 'rctd' else full_spots
        if not prediction.index.equals(expected_index):
            raise RuntimeError(f'spot identity/order differs: {pair_id} {method}')
        if set(prediction.columns) != retained_set:
            raise RuntimeError(f'retained class support differs: {pair_id} {method}')
        prediction = prediction.loc[:, retained]
        if not np.allclose(prediction.loc[positive_spots].sum(axis=1), 1.0, atol=1e-05):
            raise RuntimeError(f'positive rows are not normalized: {pair_id} {method}')
        if method == 'rctd':
            if int(metadata.get('min_cells_per_type', -1)) != minimum:
                raise RuntimeError(f'RCTD threshold differs: {pair_id}')
        else:
            label_filter = metadata.get('reference_label_filter', {})
            if metadata.get('actual_accelerator') != 'cuda' or metadata.get('cuda_execution_verified') is not True or label_filter.get('cell_type_key') != 'cell_class' or (int(label_filter.get('min_cells_per_type', -1)) != minimum) or (set(label_filter.get('retained_cell_types', [])) != retained_set) or (not np.allclose(prediction.loc[zero_spots].sum(axis=1), 0.0, atol=1e-12)):
                raise RuntimeError(f'Cell2location execution/support contract differs: {pair_id}')
        predictions[method] = prediction
        metadata_paths[method] = metadata_path
    truth_scored = with_other(truth.loc[positive_spots], retained)
    rows = []
    for method in METHODS:
        prediction = predictions[method].loc[positive_spots]
        rows.append({'method': method, 'pair_id': pair_id, 'slice_id': slice_id, 'resolution': resolution, 'status': 'ok', 'comparison_scope': 'same_expression_exact_positive_spots_exact_retained_cell_classes_plus_other', 'annotation_key': 'cell_class', 'min_cells_per_class': minimum, 'n_input_spots': len(full_spots), 'n_scored_spots': len(positive_spots), 'n_zero_library_spots_excluded': len(zero_spots), 'n_truth_cell_classes': len(truth_names), 'n_retained_cell_classes': len(retained), 'n_scored_categories_including_other': len(retained) + 1, **metrics(truth_scored, with_other(prediction, retained))})
    audit = {'pair_id': pair_id, 'slice_id': slice_id, 'resolution': resolution, 'annotation_key': 'cell_class', 'min_cells_per_class': minimum, 'n_full_spots': len(full_spots), 'n_positive_spots': len(positive_spots), 'n_zero_library_spots': len(zero_spots), 'n_truth_cell_classes': len(truth_names), 'n_retained_cell_classes': len(retained), 'method_class_sets_exactly_equal': True, 'rctd_positive_spot_order_exact': True, 'cell2location_full_spot_order_exact': True, 'cell2location_zero_rows_zero': True, 'other_residual_included': True, 'retained_cell_class_order_json': json.dumps(list(retained))}
    lineage = []
    for role, path in (('simulation', simulation_path), ('truth', truth_path), ('reference', reference_path), ('rctd', Path(one_row(method_rows, pair_id=pair_id, method='rctd')['output_path'])), ('rctd_metadata', metadata_paths['rctd']), ('cell2location', Path(one_row(method_rows, pair_id=pair_id, method='cell2location')['output_path'])), ('cell2location_metadata', metadata_paths['cell2location'])):
        lineage.append({'pair_id': pair_id, 'artifact': role, 'path': str(path.resolve())})
    return (rows, audit, lineage)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    config = yaml.safe_load(args.config.read_text())
    if config.get('cell_type_key') != 'cell_class':
        raise RuntimeError('cell-class scorer requires cell_type_key: cell_class')
    if int(config['rctd']['min_cells_per_type']) != int(config['cell2location']['min_cells_per_type']):
        raise RuntimeError('method thresholds differ')
    simulations_path = args.run_dir / 'simulation_manifest.csv'
    methods_path = args.run_dir / 'method_manifest.csv'
    simulations = read_manifest(simulations_path)
    methods = read_manifest(methods_path)
    expected_pairs = {f'{slice_id}__resolution_{float(resolution):g}' for slice_id in config['slices'] for resolution in config['resolutions']}
    if len(simulations) != 6 or set(simulations['pair_id']) != expected_pairs or (not simulations['status'].eq('ok').all()) or (len(methods) != 12) or (set(methods['pair_id']) != expected_pairs) or (set(methods['method']) != set(METHODS)) or methods.duplicated(['pair_id', 'method']).any() or (not methods['status'].eq('ok').all()):
        raise RuntimeError('run manifests do not contain the exact successful 6-pair/12-job matrix')
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    lineage: list[dict[str, str]] = []
    for simulation_row in simulations.sort_values(['slice_id', 'resolution']).itertuples(index=False):
        result_rows, audit, result_lineage = score_pair(pd.Series(simulation_row._asdict()), methods, config)
        rows.extend(result_rows)
        audits.append(audit)
        lineage.extend(result_lineage)
    args.output_dir.mkdir(parents=True)
    scores = pd.DataFrame(rows).sort_values(['slice_id', 'resolution', 'method'])
    summary = scores.groupby('method', as_index=False).agg(n_pairs=('pair_id', 'size'), jsd_mean=('jsd_mean', 'mean'), pearson_mean=('pearson_mean', 'mean'), rmse=('rmse', 'mean'))
    output_paths = {'scores': args.output_dir / 'deconvolution_scores.csv', 'summary': args.output_dir / 'deconvolution_summary.csv', 'support': args.output_dir / 'support_audit.csv', 'lineage': args.output_dir / 'input_lineage.csv'}
    scores.to_csv(output_paths['scores'], index=False, float_format='%.12g')
    summary.to_csv(output_paths['summary'], index=False, float_format='%.12g')
    pd.DataFrame(audits).sort_values(['slice_id', 'resolution']).to_csv(output_paths['support'], index=False)
    lineage_table = pd.DataFrame(lineage)
    lineage_table['path'] = lineage_table['path'].map(lambda value: os.path.relpath(value, args.run_dir))
    lineage_table.to_csv(output_paths['lineage'], index=False)
    provenance = {'schema_version': 1, 'analysis': 'study03_cell_class_exact_matched_support', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'configuration_id': str(config['configuration_id']), 'comparison': 'same simulation and positive spots; exact slice-specific retained cell_class support for both methods; rare-class truth mass represented as Other', 'metric_definition': {'jsd': 'base-2 Jensen-Shannon divergence per spot', 'pearson': 'Pearson correlation across positive-library spots for each fitted class. The mean and median summarize defined fitted-class correlations.', 'rmse': 'sqrt(mean squared error across all scored spot-category entries)'}, 'sources': {'scorer': {'path': str(Path(__file__).resolve())}, 'config': {'path': str(args.config.resolve())}, 'simulation_manifest': {'path': str(simulations_path.resolve())}, 'method_manifest': {'path': str(methods_path.resolve())}}, 'outputs': {path.name: None for path in output_paths.values()}, 'aggregate_method_ranking_authorized': False}
    provenance_path = args.output_dir / 'scoring_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    for path in (*output_paths.values(), provenance_path):
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
