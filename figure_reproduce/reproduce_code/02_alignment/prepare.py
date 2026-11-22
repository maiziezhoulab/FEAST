"""Create the 28 fixed-plate alignment inputs in a fresh directory."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
from FEAST import Alteration, __version__ as feast_version, simulate
from FEAST.alignment import apply_spatial_transform, rotate_spatial
CONFIGURATION_ID = 'study02-fixed-plate-rotation-v3'

def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'Zenodo/inputs/02_alignment' / 'config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    if str(feast_version) != str(config['feast_version']):
        raise RuntimeError(f'unexpected FEAST version: {feast_version}')
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite output root: {args.output_dir}')
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    input_manifest = pd.read_csv(args.config.with_name('input_files.csv'))
    if len(input_manifest) != 1:
        raise RuntimeError('alignment reference does not match its input manifest')
    raw_reference = ad.read_h5ad(args.reference)
    reference_coords = np.asarray(raw_reference.obsm['spatial'], dtype=np.float64)
    if not raw_reference.obs_names.is_unique or not np.isfinite(reference_coords).all():
        raise RuntimeError('reference spot IDs/coordinates are invalid')
    tolerance = float(config['geometry_tolerance'])
    seed = int(config['seed'])
    rotation_config = config['rotation']
    if rotation_config.get('plate_bounds_source') != 'reference_spatial_min_max' or rotation_config.get('crop_inclusive') is not True:
        raise RuntimeError('unsupported fixed-plate geometry configuration')
    angles = [float(value) for value in config['angles']]
    expected_retained = {float(angle): int(count) for angle, count in rotation_config['expected_retained_spots'].items()}
    if set(expected_retained) != set(angles):
        raise RuntimeError('expected retained counts do not match configured angles')
    plate_bounds = np.stack([reference_coords.min(axis=0), reference_coords.max(axis=0)])
    plate_center = plate_bounds.mean(axis=0)
    plate_half_extent = (plate_bounds[1] - plate_bounds[0]) / 2.0
    pivot_fraction = float(rotation_config['pivot_half_extent_fraction'])
    rotation_center = plate_center + pivot_fraction * plate_half_extent
    if not np.isfinite(rotation_center).all():
        raise RuntimeError('derived rotation center is non-finite')
    rows: list[dict] = []
    geometry_support = {}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base_dir = args.output_dir / 'base_simulations'
    base_dir.mkdir()
    import scanpy as sc
    reference = raw_reference.copy()
    sc.pp.filter_genes(reference, min_cells=int(config['simulation']['min_cells']))
    simulation = config['simulation']
    for alteration in config['alterations']:
        parameters = config['alteration_parameters'][alteration]
        if parameters['type'] == 'baseline':
            alteration_config = Alteration()
        elif parameters['type'] == 'mean':
            alteration_config = Alteration.mean_only(float(parameters['fold_change']))
        elif parameters['type'] == 'variance':
            alteration_config = Alteration.variance_only(float(parameters['fold_change']))
        elif parameters['type'] == 'sparsity_logit':
            alteration_config = Alteration.sparsity_logit(float(parameters['logit_shift']))
        else:
            raise RuntimeError(f"unsupported alteration type: {parameters['type']}")
        source = simulate(reference=reference, alteration=alteration_config, seed=seed, parameter_mode=str(simulation['parameter_mode']), spatial_mode=str(simulation['spatial_mode']), assignment_solver=str(simulation['assignment_solver']), assignment_blocks=bool(simulation['assignment_blocks']), ppf_method=str(simulation['ppf_method']), beta_n_jobs=int(simulation['beta_n_jobs']), beta_early_stopping_patience=int(simulation['beta_early_stopping_patience']), convert_n_jobs=int(simulation['convert_n_jobs']), verbose=True)
        diagnostics = source.uns.get('simulation_diagnostics', {})
        assignment_diagnostics = diagnostics.get('copula_rank_diagnostics', {})
        if diagnostics.get('spatial_mode') != 'reference_rank':
            raise RuntimeError(f'{alteration}: FEAST returned the wrong spatial mode')
        if assignment_diagnostics.get('assignment_blocks') is not False:
            raise RuntimeError(f'{alteration}: FEAST did not use global assignment')
        source_path = base_dir / f'{alteration}.h5ad'
        source.uns['feast_reproduce'] = {'configuration_id': CONFIGURATION_ID, 'feast_version': str(feast_version), 'public_seed': seed, 'alteration': alteration, 'alteration_parameters': parameters, 'parameter_mode': str(simulation['parameter_mode']), 'spatial_mode': str(simulation['spatial_mode']), 'assignment_solver': str(simulation['assignment_solver']), 'assignment_blocks': bool(simulation['assignment_blocks']), 'solver_diagnostics': {'status': 'completed', 'assignment_method': assignment_diagnostics.get('assignment_method'), 'assignment_blocks': False, 'assignment_solver_requested': 'scipy', 'model_selection_counts': diagnostics.get('model_selection_counts', {})}}
        source.write_h5ad(source_path, compression='gzip')
        source_coords = np.asarray(source.obsm['spatial'], dtype=np.float64)
        if not source.obs_names.equals(raw_reference.obs_names):
            raise RuntimeError(f'{alteration}: spot identity/order differs from reference')
        if not source.var_names.is_unique:
            raise RuntimeError(f'{alteration}: gene IDs are not unique')
        geometry_error = float(np.max(np.abs(source_coords - reference_coords)))
        if geometry_error > tolerance:
            raise RuntimeError(f'{alteration}: spatial geometry differs from reference by {geometry_error}')
        for angle_value in angles:
            angle = float(angle_value)
            rotated = rotate_spatial(source, angle, center=rotation_center, plate_bounds=plate_bounds)
            transform = rotated.uns['feast_alignment_transform']
            forward_coords = apply_spatial_transform(source_coords, transform['forward_matrix'])
            expected_mask = np.all((forward_coords >= plate_bounds[0]) & (forward_coords <= plate_bounds[1]), axis=1)
            retained_positions = source.obs_names.get_indexer(rotated.obs_names)
            expected_positions = np.flatnonzero(expected_mask)
            if np.any(retained_positions < 0) or not np.array_equal(retained_positions, expected_positions):
                raise RuntimeError(f'{alteration}/{angle:g}: retained spot support/order is incorrect')
            if angle in geometry_support and not rotated.obs_names.equals(geometry_support[angle]):
                raise RuntimeError(f'{alteration}/{angle:g}: retained barcode order changed across alterations')
            geometry_support[angle] = rotated.obs_names.copy()
            restored = apply_spatial_transform(rotated.obsm['spatial'], transform['inverse_matrix'])
            matched_source_coords = source_coords[retained_positions]
            inverse_error = float(np.max(np.abs(restored - matched_source_coords)))
            if inverse_error > tolerance:
                raise RuntimeError(f'{alteration}/{angle:g}: inverse error {inverse_error}')
            if not rotated.var_names.equals(source.var_names):
                raise RuntimeError(f'{alteration}/{angle:g}: gene order changed')
            retained_n_spots = int(rotated.n_obs)
            dropped_n_spots = int(source.n_obs - retained_n_spots)
            retained_fraction = float(retained_n_spots / source.n_obs)
            if retained_n_spots != expected_retained[angle]:
                raise RuntimeError(f'{alteration}/{angle:g}: expected {expected_retained[angle]} retained spots, got {retained_n_spots}')
            if not np.all((rotated.obsm['spatial'] >= plate_bounds[0]) & (rotated.obsm['spatial'] <= plate_bounds[1])):
                raise RuntimeError(f'{alteration}/{angle:g}: retained coordinates exceed plate bounds')
            metadata_valid = bool(transform.get('plate_bounds_applied') is True and transform.get('preserve_spot_count') is False and (int(transform.get('input_n_spots', -1)) == source.n_obs) and (int(transform.get('retained_n_spots', -1)) == retained_n_spots) and (int(transform.get('dropped_n_spots', -1)) == dropped_n_spots) and np.isclose(float(transform.get('retained_fraction', np.nan)), retained_fraction) and np.allclose(np.asarray(transform.get('center'), dtype=np.float64), rotation_center, rtol=0.0, atol=tolerance) and np.allclose(np.asarray(transform.get('plate_bounds'), dtype=np.float64), plate_bounds, rtol=0.0, atol=tolerance))
            if not metadata_valid:
                raise RuntimeError(f'{alteration}/{angle:g}: FEAST crop metadata is inconsistent')
            output = args.output_dir / f'{alteration}_rotated_{angle:g}.h5ad'
            rotated.uns['feast_reproduce'] = {'configuration_id': CONFIGURATION_ID, 'feast_version': str(feast_version), 'public_seed': seed, 'alteration': alteration, 'angle_degrees': angle, 'inverse_max_abs_error': inverse_error, 'input_n_spots': int(source.n_obs), 'retained_n_spots': retained_n_spots, 'dropped_n_spots': dropped_n_spots, 'retained_fraction': retained_fraction, 'plate_bounds': plate_bounds, 'rotation_center': rotation_center}
            rotated.write_h5ad(output, compression='gzip')
            rows.append({'configuration_id': CONFIGURATION_ID, 'alteration': alteration, 'angle_degrees': angle, 'public_seed': seed, 'feast_version': str(feast_version), 'n_spots': rotated.n_obs, 'input_n_spots': int(source.n_obs), 'retained_n_spots': retained_n_spots, 'dropped_n_spots': dropped_n_spots, 'retained_fraction': retained_fraction, 'n_genes': rotated.n_vars, 'source_path': source_path.relative_to(args.output_dir).as_posix(), 'output_path': output.relative_to(args.output_dir).as_posix(), 'reference_geometry_max_abs_error': geometry_error, 'inverse_max_abs_error': inverse_error, 'spot_order_preserved': True, 'retained_spot_order_preserved': True, 'gene_order_preserved': True, 'plate_x_min': float(plate_bounds[0, 0]), 'plate_y_min': float(plate_bounds[0, 1]), 'plate_x_max': float(plate_bounds[1, 0]), 'plate_y_max': float(plate_bounds[1, 1]), 'rotation_center_x': float(rotation_center[0]), 'rotation_center_y': float(rotation_center[1])})
    expected = len(config['alterations']) * len(config['angles'])
    if len(rows) != expected or expected != 28:
        raise RuntimeError(f'expected 28 rotations, generated {len(rows)}')
    manifest_path = args.output_dir / 'rotation_manifest.csv'
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    provenance = {'configuration_id': CONFIGURATION_ID, 'generated_utc': datetime.now(timezone.utc).isoformat(), 'feast_version': str(feast_version), 'public_seed': seed, 'reference_path': str(args.reference.resolve()), 'completed_rotations': len(rows), 'fresh_base_simulations': len(config['alterations']), 'rotation_geometry': {'plate_bounds_source': 'reference_spatial_min_max', 'plate_bounds': plate_bounds.tolist(), 'plate_center': plate_center.tolist(), 'plate_half_extent': plate_half_extent.tolist(), 'pivot_half_extent_fraction': pivot_fraction, 'rotation_center': rotation_center.tolist(), 'crop_inclusive': True, 'angles': angles, 'reference_n_spots': int(raw_reference.n_obs), 'expected_retained_spots': {f'{angle:g}': expected_retained[angle] for angle in angles}}}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'generated {len(rows)} rotations in {args.output_dir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
