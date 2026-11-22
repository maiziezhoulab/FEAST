"""Compute alignment metrics for the fresh Spateo fixed-plate run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
from metric_utils import angular_error, mapped_label_metrics, mean_row_correlation, ordered_reference_positions, partial_transport_metrics, recovered_rotation

def expected_artifact_names(method: str) -> set[str]:
    common = {'aligned.h5ad', 'aligned_coordinates.csv', 'diagnostics.json', 'transform.json'}
    return common | {'mapping_matrix.npy'}

def artifact_manifest_is_valid(output: Path, job: object) -> bool:
    try:
        artifacts = json.loads(job.artifacts)
    except (TypeError, json.JSONDecodeError):
        return False
    expected_names = expected_artifact_names(job.method)
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_names):
        return False
    expected_paths = {name: (output / name).resolve() for name in expected_names}
    listed_names: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            return False
        path = output / item['path']
        name = path.name
        if name in listed_names or name not in expected_paths or path.resolve() != expected_paths[name] or (not path.is_file()):
            return False
        listed_names.add(name)
    try:
        actual_names = {path.name for path in output.iterdir() if path.is_file() and path.name != 'runner.log'}
    except OSError:
        return False
    runner_log = output / 'runner.log'
    return bool(listed_names == expected_names and actual_names == expected_names and runner_log.is_file())

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--rotation-dir', type=Path, required=True)
    parser.add_argument('--methods-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=('spateo',), default=('spateo',), help='Method rows to score from the manifest.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'Zenodo/inputs/02_alignment' / 'config.yaml')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = yaml.safe_load(args.config.read_text())
    rotations = pd.read_csv(args.rotation_dir / 'rotation_manifest.csv')
    full_method_manifest = pd.read_csv(args.methods_dir / 'method_manifest.csv')
    selected_methods = tuple(dict.fromkeys(args.methods))
    methods = full_method_manifest.loc[full_method_manifest['method'].isin(selected_methods)].copy()
    expected = {(alteration, float(angle), method) for alteration in config['alterations'] for angle in config['angles'] for method in selected_methods}
    observed = set(zip(methods['alteration'], methods['angle_degrees'].astype(float), methods['method']))
    if len(methods) != len(expected) or methods['job_id'].duplicated().any() or observed != expected or (set(methods['status']) != {'ok'}) or (not methods['public_seed'].eq(int(config['seed'])).all()):
        raise RuntimeError('method manifest does not match the successful configured jobs')
    reference = ad.read_h5ad(args.reference)
    if not reference.obs_names.is_unique:
        raise RuntimeError('reference spot IDs are not unique')
    reference_coords = np.asarray(reference.obsm['spatial'], dtype=np.float64)
    tree = cKDTree(reference_coords)
    nearest_distances, _ = tree.query(reference_coords, k=2)
    spot_spacing = float(np.median(nearest_distances[:, 1]))
    if not np.isfinite(spot_spacing) or spot_spacing <= 0:
        raise RuntimeError('reference spot spacing is invalid')
    expected_rotations = {(alteration, float(angle)) for alteration in config['alterations'] for angle in config['angles']}
    observed_rotations = set(zip(rotations['alteration'], rotations['angle_degrees'].astype(float)))
    if len(rotations) != len(expected_rotations) or rotations[['alteration', 'angle_degrees']].duplicated().any() or observed_rotations != expected_rotations:
        raise RuntimeError('rotation manifest does not match the configured inputs')
    rotation_lookup = {(row.alteration, float(row.angle_degrees)): row for row in rotations.itertuples(index=False)}
    rows = []
    for job in methods.itertuples(index=False):
        angle = float(job.angle_degrees)
        rotation = rotation_lookup[job.alteration, angle]
        moving_path = args.rotation_dir / rotation.output_path
        if (
            job.configuration_id != f'study02-{job.method}-{rotation.alteration}-{angle:g}'
            or job.job_id != f'{rotation.alteration}__{job.method}__angle_{angle:g}'
            or (args.methods_dir / job.reference).resolve() != args.reference.resolve()
            or (args.methods_dir / job.moving).resolve() != moving_path.resolve()
            or job.feast_version != rotation.feast_version
            or int(job.public_seed) != int(rotation.public_seed)
        ):
            raise RuntimeError(f'job provenance differs from the registered rotation/reference: {job.job_id}')
        moving = ad.read_h5ad(moving_path)
        if not moving.obs_names.is_unique:
            raise RuntimeError(f'moving spot IDs are not unique: {job.job_id}')
        try:
            reference_positions = ordered_reference_positions(reference.obs_names, moving.obs_names)
        except ValueError as error:
            raise RuntimeError(f'moving spots are not an ordered reference subset: {job.job_id}') from error
        if not (int(job.reference_n_spots) == reference.n_obs and int(job.moving_n_spots) == moving.n_obs and (int(job.dropped_n_spots) == reference.n_obs - moving.n_obs) and np.isclose(float(job.retained_fraction), moving.n_obs / reference.n_obs)):
            raise RuntimeError(f'support metadata differs: {job.job_id}')
        common = reference.var_names.intersection(moving.var_names, sort=False)
        if common.empty:
            raise RuntimeError(f'no exact genes shared: {job.job_id}')
        output_dir = args.methods_dir / job.output_dir
        if not artifact_manifest_is_valid(output_dir, job):
            raise RuntimeError(f'artifact manifest or runner log is invalid: {job.job_id}')
        coordinates = pd.read_csv(output_dir / 'aligned_coordinates.csv')
        if not pd.Index(coordinates['spot_barcode'].astype(str)).equals(moving.obs_names):
            raise RuntimeError(f'coordinate spot order differs: {job.job_id}')
        aligned = coordinates[['x_aligned', 'y_aligned']].to_numpy(np.float64)
        moving_coords = np.asarray(moving.obsm['spatial'], dtype=np.float64)
        matched_reference_coords = reference_coords[reference_positions]
        if not np.isfinite(aligned).all():
            raise RuntimeError(f'non-finite aligned coordinates: {job.job_id}')
        distances = np.linalg.norm(aligned - matched_reference_coords, axis=1)
        nn_distances, nn_indices = tree.query(aligned, k=1)
        coordinate_nn_spot_accuracy = float(np.mean(reference.obs_names.to_numpy()[nn_indices] == moving.obs_names.to_numpy()))
        coordinate_nn_region_accuracy = np.nan
        if 'ground_truth' in reference.obs and 'ground_truth' in moving.obs:
            coordinate_nn_region_accuracy = float(np.mean(reference.obs['ground_truth'].astype(str).to_numpy()[nn_indices] == moving.obs['ground_truth'].astype(str).to_numpy()))
        oracle_ge_correlation = mean_row_correlation(reference[reference_positions, common].X, moving[:, common].X)
        moving_to_aligned = recovered_rotation(moving_coords, aligned)
        residual = recovered_rotation(aligned, matched_reference_coords)
        matrix_path = output_dir / ('mapping_matrix.npy' if job.method == 'spateo' else 'coupling_matrix.npy')
        matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
        if matrix.shape != (reference.n_obs, moving.n_obs) or matrix.size == 0 or (not np.isfinite(matrix).all()):
            raise RuntimeError(f'invalid method matrix: {job.job_id}')
        negative_tolerance = float(config['spateo']['mapping_negative_tolerance']) if job.method == 'spateo' else 1e-12
        transport, active_moving, moving_to_reference = partial_transport_metrics(matrix, reference_positions, support_tolerance=1e-12, negative_tolerance=negative_tolerance)
        transport_mass = transport['transport_total_mass']
        transport_minimum = transport['transport_minimum']
        transport_predicted_ge_correlation = np.nan
        if active_moving.any():
            transport_predicted_ge_correlation = mean_row_correlation(reference[moving_to_reference[active_moving], common].X, moving[active_moving, common].X)
        label_metrics = {'transport_moving_region_recovery_rate': np.nan, 'transport_moving_conditional_region_accuracy': np.nan, 'transport_label_transfer_ari': np.nan}
        coordinate_nn_label_transfer_ari = np.nan
        if 'ground_truth' in reference.obs and 'ground_truth' in moving.obs:
            reference_labels = reference.obs['ground_truth'].astype(str).to_numpy()
            moving_labels = moving.obs['ground_truth'].astype(str).to_numpy()
            label_metrics = mapped_label_metrics(reference_labels, moving_labels, active_moving, moving_to_reference)
            coordinate_nn_label_transfer_ari = float(adjusted_rand_score(moving_labels, reference_labels[nn_indices]))
        coordinate_nn_predicted_ge_correlation = mean_row_correlation(reference[nn_indices, common].X, moving[:, common].X)
        diagnostics = json.loads((output_dir / 'diagnostics.json').read_text())
        if job.method == 'spateo':
            matrix_evidence_valid = bool(transport_minimum >= -negative_tolerance and diagnostics.get('mapping_shape') == list(matrix.shape) and (diagnostics.get('mapping_all_finite') is True) and np.isclose(float(diagnostics.get('mapping_minimum', np.nan)), transport_minimum, rtol=1e-12, atol=1e-12) and np.isclose(float(diagnostics.get('mapping_mass', np.nan)), transport_mass, rtol=1e-12, atol=1e-12) and (float(diagnostics.get('mapping_negative_tolerance', np.nan)) == negative_tolerance))
        else:
            coupling = diagnostics.get('coupling', {})
            matrix_evidence_valid = bool(coupling.get('shape') == list(matrix.shape) and np.isclose(float(coupling.get('mass', np.nan)), transport_mass, rtol=1e-12, atol=1e-12))
        solver_ok = diagnostics.get('status') == 'completed' and diagnostics.get('cuda_available') is True and (diagnostics.get('mapping_all_finite') is True) and (diagnostics.get('spot_support_preserved') is True) and (diagnostics.get('gene_order_preserved') is True) and (diagnostics.get('convergence_claim') is False) and (diagnostics.get('iteration_schedule', {}).get('kind') == 'fixed') and (diagnostics.get('iteration_schedule', {}).get('completed_return') is True) and matrix_evidence_valid if job.method == 'spateo' else diagnostics.get('status') == 'converged' and diagnostics.get('positive_convergence_evidence') is True and matrix_evidence_valid
        rows.append({'job_id': job.job_id, 'configuration_id': job.configuration_id, 'method': job.method, 'alteration': job.alteration, 'angle_degrees': float(job.angle_degrees), 'status': 'ok' if solver_ok else 'invalid', 'canonical_candidate': bool(solver_ok), 'solver_evidence_disposition': 'fixed_schedule_complete_no_convergence_claim' if job.method == 'spateo' else 'positive_convergence_evidence', 'n_aligned': len(aligned), 'n_reference_spots': int(reference.n_obs), 'n_moving_spots': int(moving.n_obs), 'retained_fraction': float(moving.n_obs / reference.n_obs), 'n_exact_join_genes': len(common), 'reference_median_spot_spacing': spot_spacing, 'mean_spatial_error': float(distances.mean()), 'mean_spatial_error_spot_units': float(distances.mean() / spot_spacing), 'median_spatial_error': float(np.median(distances)), 'rmse_spatial_error': float(np.sqrt(np.mean(distances ** 2))), 'max_spatial_error': float(distances.max()), 'coordinate_nn_spot_accuracy': coordinate_nn_spot_accuracy, 'coordinate_nn_region_accuracy': coordinate_nn_region_accuracy, 'coordinate_nn_label_transfer_ari': coordinate_nn_label_transfer_ari, 'coordinate_nn_mean_distance': float(np.mean(nn_distances)), 'oracle_ge_correlation': oracle_ge_correlation, 'coordinate_nn_predicted_pair_ge_correlation': coordinate_nn_predicted_ge_correlation, 'coordinate_nn_ge_correlation_gap': float(oracle_ge_correlation - coordinate_nn_predicted_ge_correlation), 'transport_predicted_pair_ge_correlation': transport_predicted_ge_correlation, 'transport_ge_correlation_gap': float(oracle_ge_correlation - transport_predicted_ge_correlation), 'recovered_moving_to_aligned_rotation': moving_to_aligned, 'expected_moving_to_aligned_rotation': -float(job.angle_degrees), 'rotation_recovery_error': angular_error(moving_to_aligned, -float(job.angle_degrees)), 'residual_aligned_to_target_rotation': residual, **transport, **label_metrics})
    results = pd.DataFrame(rows)
    if len(results) != len(expected) or not results['canonical_candidate'].all():
        raise RuntimeError('one or more alignment rows failed the score-time evidence gate')
    summary = results.groupby(['method', 'alteration'], as_index=False).agg(n_rows=('job_id', 'size'), mean_spatial_error_spot_units=('mean_spatial_error_spot_units', 'mean'), mean_coordinate_nn_spot_accuracy=('coordinate_nn_spot_accuracy', 'mean'), mean_coordinate_nn_region_accuracy=('coordinate_nn_region_accuracy', 'mean'), mean_coordinate_nn_predicted_pair_ge_correlation=('coordinate_nn_predicted_pair_ge_correlation', 'mean'), mean_moving_transport_coverage=('moving_transport_coverage', 'mean'), mean_transport_moving_exact_spot_recovery_rate=('transport_moving_exact_spot_recovery_rate', 'mean'), mean_ground_truth_transport_mass_fraction=('ground_truth_transport_mass_fraction', 'mean'), mean_transport_moving_region_recovery_rate=('transport_moving_region_recovery_rate', 'mean'), mean_oracle_ge_correlation=('oracle_ge_correlation', 'mean'), mean_rotation_recovery_error=('rotation_recovery_error', 'mean'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output_dir / 'alignment_metrics.csv'
    summary_path = args.output_dir / 'alignment_summary.csv'
    results.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    (args.output_dir / 'provenance.json').write_text(json.dumps({'configuration_id': 'study02-fixed-plate-alignment-score-v5', 'metric_schema_version': 3, 'methods': list(selected_methods), 'feast_versions': sorted(set(methods['feast_version'].astype(str))), 'region_key': 'ground_truth', 'outputs': {metrics_path.name: None, summary_path.name: None}}, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
