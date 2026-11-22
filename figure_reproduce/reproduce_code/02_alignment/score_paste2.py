"""Score PASTE2 aligned coordinates plus auxiliary partial-transport metrics."""
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
METRIC_SCHEMA_VERSION = 3
SUPPORT_TOLERANCE = 1e-12
NEGATIVE_TOLERANCE = 1e-12
EXPECTED_ARTIFACTS = {'aligned.h5ad', 'aligned_coordinates.csv', 'coupling_matrix.npy', 'diagnostics.json', 'transform.json'}

def artifacts_are_valid(job: object, output: Path) -> bool:
    try:
        artifacts = json.loads(job.artifacts)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(artifacts, list) or len(artifacts) != len(EXPECTED_ARTIFACTS):
        return False
    listed: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            return False
        path = output / item['path']
        if path.name in listed or path.name not in EXPECTED_ARTIFACTS or path.resolve() != (output / path.name).resolve() or (not path.is_file()):
            return False
        listed.add(path.name)
    actual = {path.name for path in output.iterdir() if path.is_file()}
    return listed == EXPECTED_ARTIFACTS and actual == EXPECTED_ARTIFACTS

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--rotation-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'Zenodo/inputs/02_alignment' / 'config.yaml')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = yaml.safe_load(args.config.read_text())
    expected = {(alteration, float(angle)) for alteration in config['alterations'] for angle in config['angles']}
    rotations = pd.read_csv(args.rotation_dir / 'rotation_manifest.csv')
    manifest = pd.read_csv(args.run_dir / 'method_manifest.csv')
    observed_rotations = set(zip(rotations['alteration'], rotations['angle_degrees'].astype(float)))
    observed_jobs = set(zip(manifest['alteration'], manifest['angle_degrees'].astype(float)))
    if len(rotations) != len(expected) or rotations[['alteration', 'angle_degrees']].duplicated().any() or observed_rotations != expected:
        raise RuntimeError('rotation manifest does not cover the configured grid')
    if len(manifest) != len(expected) or manifest['job_id'].duplicated().any() or observed_jobs != expected or (set(manifest['method']) != {'paste2'}) or (not manifest['status'].eq('ok').all()):
        raise RuntimeError('PASTE2 run is incomplete or duplicated')
    rotation_lookup = {(row.alteration, float(row.angle_degrees)): row for row in rotations.itertuples(index=False)}
    reference = ad.read_h5ad(args.reference)
    if not reference.obs_names.is_unique:
        raise RuntimeError('reference spot IDs are not unique')
    reference_index = pd.Index(reference.obs_names)
    reference_xy = np.asarray(reference.obsm['spatial'], dtype=np.float64)
    if not np.isfinite(reference_xy).all():
        raise RuntimeError('reference coordinates are non-finite')
    tree = cKDTree(reference_xy)
    nearest_distances, _ = tree.query(reference_xy, k=2)
    spot_spacing = float(np.median(nearest_distances[:, 1]))
    if not np.isfinite(spot_spacing) or spot_spacing <= 0:
        raise RuntimeError('reference spot spacing is invalid')
    rows: list[dict[str, object]] = []
    for job in manifest.sort_values(['alteration', 'angle_degrees']).itertuples(index=False):
        key = (job.alteration, float(job.angle_degrees))
        rotation = rotation_lookup[key]
        if (
            job.configuration_id != f'study02-paste2-{rotation.alteration}-{float(rotation.angle_degrees):g}'
            or job.feast_version != rotation.feast_version
            or int(job.public_seed) != int(rotation.public_seed)
            or int(job.public_seed) != int(config['seed'])
        ):
            raise RuntimeError(f'PASTE2 seed, FEAST version or configuration differs for {job.job_id}')
        moving_path = args.run_dir / job.moving
        output = args.run_dir / job.output_dir
        runner_log = output.parent / f'angle_{float(job.angle_degrees):g}.runner.log'
        if not ((args.run_dir / job.reference).resolve() == args.reference.resolve() and moving_path.resolve() == (args.rotation_dir / rotation.output_path).resolve() and np.isclose(float(job.retained_fraction), rotation.retained_fraction) and runner_log.is_file() and artifacts_are_valid(job, output)):
            raise RuntimeError(f'PASTE2 provenance failed for {job.job_id}')
        moving = ad.read_h5ad(moving_path)
        aligned_adata = ad.read_h5ad(output / 'aligned.h5ad')
        if not moving.obs_names.is_unique:
            raise RuntimeError(f'moving spot IDs are not unique: {job.job_id}')
        try:
            reference_positions = ordered_reference_positions(reference.obs_names, moving.obs_names)
        except ValueError as error:
            raise RuntimeError(f'invalid retained support: {job.job_id}') from error
        if not (aligned_adata.obs_names.equals(moving.obs_names) and aligned_adata.var_names.equals(moving.var_names) and (int(rotation.retained_n_spots) == moving.n_obs) and (int(rotation.input_n_spots) == reference.n_obs)):
            raise RuntimeError(f'aligned support differs: {job.job_id}')
        coordinates = pd.read_csv(output / 'aligned_coordinates.csv')
        if not pd.Index(coordinates['spot_barcode'].astype(str)).equals(moving.obs_names):
            raise RuntimeError(f'coordinate spot order differs: {job.job_id}')
        aligned = coordinates[['x_aligned', 'y_aligned']].to_numpy(np.float64)
        stored_aligned = np.asarray(aligned_adata.obsm['spatial_aligned'], dtype=np.float64)[:, :2]
        if not np.isfinite(aligned).all() or not np.allclose(aligned, stored_aligned, rtol=0.0, atol=1e-10):
            raise RuntimeError(f'aligned coordinates are invalid: {job.job_id}')
        moving_xy = np.asarray(moving.obsm['spatial'], dtype=np.float64)[:, :2]
        target_xy = reference_xy[reference_positions]
        spatial_errors = np.linalg.norm(aligned - target_xy, axis=1)
        nn_distances, nn_indices = tree.query(aligned, k=1)
        coordinate_nn_spot_accuracy = float(np.mean(reference.obs_names.to_numpy()[nn_indices] == moving.obs_names.to_numpy()))
        common = reference.var_names.intersection(moving.var_names, sort=False)
        if common.empty:
            raise RuntimeError(f'no exact genes shared: {job.job_id}')
        oracle_ge_correlation = mean_row_correlation(reference[reference_positions, common].X, moving[:, common].X)
        matrix = np.load(output / 'coupling_matrix.npy', mmap_mode='r', allow_pickle=False)
        transport, active_moving, moving_to_reference = partial_transport_metrics(matrix, reference_positions, support_tolerance=SUPPORT_TOLERANCE, negative_tolerance=NEGATIVE_TOLERANCE)
        diagnostics = json.loads((output / 'diagnostics.json').read_text())
        transform = json.loads((output / 'transform.json').read_text())
        coupling = diagnostics.get('coupling', {})
        solver = diagnostics.get('solver', {})
        solver_ok = bool(diagnostics.get('status') == 'converged' and diagnostics.get('positive_convergence_evidence') is True and (coupling.get('shape') == list(matrix.shape)) and np.isclose(float(coupling.get('mass', np.nan)), transport['transport_total_mass'], rtol=1e-12, atol=1e-12) and (float(coupling.get('row_constraint_max_excess', np.inf)) <= 1e-08) and (float(coupling.get('column_constraint_max_excess', np.inf)) <= 1e-08) and (int(solver.get('iterations', 0)) < int(solver.get('num_iter_max', 0))) and (transform.get('status') == 'ok') and (transform.get('method') == 'paste2'))
        if not solver_ok:
            raise RuntimeError(f'PASTE2 solver evidence failed: {job.job_id}')
        transport_predicted_ge_correlation = np.nan
        if active_moving.any():
            transport_predicted_ge_correlation = mean_row_correlation(reference[moving_to_reference[active_moving], common].X, moving[active_moving, common].X)
        label_metrics = {'transport_moving_region_recovery_rate': np.nan, 'transport_moving_conditional_region_accuracy': np.nan, 'transport_label_transfer_ari': np.nan}
        coordinate_nn_region_accuracy = np.nan
        coordinate_nn_label_transfer_ari = np.nan
        if 'ground_truth' in reference.obs and 'ground_truth' in moving.obs:
            reference_labels = reference.obs['ground_truth'].astype(str).to_numpy()
            moving_labels = moving.obs['ground_truth'].astype(str).to_numpy()
            label_metrics = mapped_label_metrics(reference_labels, moving_labels, active_moving, moving_to_reference)
            coordinate_nn_region_accuracy = float(np.mean(reference_labels[nn_indices] == moving_labels))
            coordinate_nn_label_transfer_ari = float(adjusted_rand_score(moving_labels, reference_labels[nn_indices]))
        coordinate_nn_predicted_ge_correlation = mean_row_correlation(reference[nn_indices, common].X, moving[:, common].X)
        moving_to_aligned = recovered_rotation(moving_xy, aligned)
        residual = recovered_rotation(aligned, target_xy)
        rows.append({'job_id': job.job_id, 'method': 'paste2', 'alteration': job.alteration, 'angle_degrees': float(job.angle_degrees), 'status': 'ok', 'canonical_candidate': True, 'n_aligned': int(aligned_adata.n_obs), 'n_reference_spots': int(reference.n_obs), 'n_moving_spots': int(moving.n_obs), 'retained_fraction': float(job.retained_fraction), 'n_exact_join_genes': int(len(common)), 'reference_median_spot_spacing': spot_spacing, 'mean_spatial_error': float(spatial_errors.mean()), 'mean_spatial_error_spot_units': float(spatial_errors.mean() / spot_spacing), 'median_spatial_error': float(np.median(spatial_errors)), 'rmse_spatial_error': float(np.sqrt(np.mean(spatial_errors ** 2))), 'max_spatial_error': float(spatial_errors.max()), 'coordinate_nn_spot_accuracy': coordinate_nn_spot_accuracy, 'coordinate_nn_region_accuracy': coordinate_nn_region_accuracy, 'coordinate_nn_label_transfer_ari': coordinate_nn_label_transfer_ari, 'coordinate_nn_mean_distance': float(nn_distances.mean()), 'oracle_ge_correlation': oracle_ge_correlation, 'coordinate_nn_predicted_pair_ge_correlation': coordinate_nn_predicted_ge_correlation, 'coordinate_nn_ge_correlation_gap': float(oracle_ge_correlation - coordinate_nn_predicted_ge_correlation), 'transport_predicted_pair_ge_correlation': transport_predicted_ge_correlation, 'transport_ge_correlation_gap': float(oracle_ge_correlation - transport_predicted_ge_correlation), 'recovered_moving_to_aligned_rotation': moving_to_aligned, 'expected_moving_to_aligned_rotation': -float(job.angle_degrees), 'rotation_recovery_error': angular_error(moving_to_aligned, -float(job.angle_degrees)), 'residual_aligned_to_target_rotation': residual, **transport, **label_metrics})
    metrics = pd.DataFrame(rows).sort_values(['alteration', 'angle_degrees'])
    if len(metrics) != len(expected) or not metrics['canonical_candidate'].all():
        raise RuntimeError('PASTE2 score table is incomplete')
    summary = metrics.groupby(['method', 'alteration'], as_index=False).agg(n_rows=('job_id', 'size'), mean_spatial_error_spot_units=('mean_spatial_error_spot_units', 'mean'), mean_rotation_recovery_error=('rotation_recovery_error', 'mean'), mean_coordinate_nn_spot_accuracy=('coordinate_nn_spot_accuracy', 'mean'), mean_coordinate_nn_region_accuracy=('coordinate_nn_region_accuracy', 'mean'), mean_coordinate_nn_predicted_pair_ge_correlation=('coordinate_nn_predicted_pair_ge_correlation', 'mean'), mean_moving_transport_coverage=('moving_transport_coverage', 'mean'), mean_transport_moving_exact_spot_recovery_rate=('transport_moving_exact_spot_recovery_rate', 'mean'), mean_ground_truth_transport_mass_fraction=('ground_truth_transport_mass_fraction', 'mean'), mean_transport_moving_region_recovery_rate=('transport_moving_region_recovery_rate', 'mean'), mean_oracle_ge_correlation=('oracle_ge_correlation', 'mean'))
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / 'alignment_metrics.csv'
    summary_path = args.output_dir / 'alignment_summary.csv'
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    provenance_path = args.output_dir / 'provenance.json'
    provenance_path.write_text(json.dumps({'metric_schema_version': METRIC_SCHEMA_VERSION, 'configuration_id': 'study02-paste2-aligned-coordinate-score-v3', 'support_tolerance': SUPPORT_TOLERANCE, 'negative_tolerance': NEGATIVE_TOLERANCE, 'region_key': 'ground_truth', 'outputs': {metrics_path.name: None, summary_path.name: None}}, indent=2) + '\n')
    print(metrics_path)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
