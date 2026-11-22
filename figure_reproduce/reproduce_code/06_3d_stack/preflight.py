"""Freeze Study 06 inputs, target assignments, donors, and AR estimates."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import anndata as ad
from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd
from workflow import STUDY_ROOT, CONFIG_PATH, SliceInfo, build_assignments, choose_donor, density_specs, load_config, representative_reference_ids, feast_runtime, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--calibration-plan', type=Path, help='Reuse the recorded assignment-randomness estimates from an existing plan.')
    parser.add_argument('--plan-only', action='store_true', help='Validate the corrected partition without writing artifacts or running FEAST')
    return parser.parse_args()

def inspect_inputs(data_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[int, SliceInfo], dict[int, set[str]], dict[int, Counter[str]], list[str]]:
    dataset = config['dataset']
    prefix = str(dataset['filename_prefix'])
    all_paths = sorted(data_dir.glob(f'{prefix}.*.h5ad'))
    excluded = {int(item) for item in dataset.get('excluded_conflicting_slices', [])}
    expected_ids = set(range(int(dataset['first_slice']), int(dataset['last_slice']) + 1))
    expected_ids -= {int(item) for item in dataset['intentionally_absent_slices']}
    observed_all_ids = {slice_id_from_path(path, prefix) for path in all_paths}
    unexpected = observed_all_ids - expected_ids - excluded
    if unexpected:
        raise ValueError(f'unexpected source slice IDs: {sorted(unexpected)}')
    paths = [path for path in all_paths if slice_id_from_path(path, prefix) in expected_ids]
    observed_ids = {slice_id_from_path(path, prefix) for path in paths}
    if observed_ids != expected_ids:
        raise ValueError(f'source slice IDs changed; missing={sorted(expected_ids - observed_ids)}, extra={sorted(observed_ids - expected_ids)}')
    if len(paths) != int(dataset['expected_files']):
        raise ValueError(f"expected {dataset['expected_files']} input files, found {len(paths)}")
    records: list[dict[str, Any]] = []
    slice_infos: dict[int, SliceInfo] = {}
    label_sets: dict[int, set[str]] = {}
    label_counts: dict[int, Counter[str]] = {}
    canonical_genes: list[str] | None = None
    for path in paths:
        slice_id = slice_id_from_path(path, prefix)
        with h5py.File(path, 'r') as handle:
            obs = read_elem(handle['obs'])
            var = read_elem(handle['var'])
            adata = ad.AnnData(obs=obs, var=var)
            for key in (dataset['spatial_key'], dataset['spatial_3d_key']):
                adata.obsm[str(key)] = read_elem(handle[f'obsm/{key}'])
            adata.uns['counts_layer_present'] = f"layers/{dataset['counts_layer']}" in handle
        try:
            validate_adata_metadata(adata, path, config)
            genes = list(map(str, adata.var_names))
            if canonical_genes is None:
                canonical_genes = genes
            elif genes != canonical_genes:
                raise ValueError(f'{path.name} gene names/order differ from the first input')
            labels = adata.obs[str(dataset['label_key'])].astype(str).tolist()
            counts = Counter(labels)
            z_values = np.asarray(adata.obs[str(dataset['z_key'])], dtype=float)
            z_value = float(z_values[0])
            spatial = np.asarray(adata.obsm[str(dataset['spatial_key'])], dtype=float)
            spatial_3d = np.asarray(adata.obsm[str(dataset['spatial_3d_key'])], dtype=float)
            records.append({'artifact_id': f'{prefix}.{slice_id:03d}', 'slice_id': slice_id, 'filename': path.name, 'bytes': path.stat().st_size, 'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars), 'z': z_value})
            slice_infos[slice_id] = SliceInfo(slice_id=slice_id, path=path.resolve(), z=z_value, n_obs=int(adata.n_obs), n_vars=int(adata.n_vars))
            label_sets[slice_id] = set(counts)
            label_counts[slice_id] = counts
        finally:
            del adata
    assert canonical_genes is not None
    return (pd.DataFrame(records).sort_values('slice_id'), slice_infos, label_sets, label_counts, canonical_genes)

def validate_adata_metadata(adata: ad.AnnData, path: Path, config: dict[str, Any]) -> None:
    dataset = config['dataset']
    if int(adata.n_vars) != int(dataset['expected_genes']):
        raise ValueError(f"{path.name} expected {dataset['expected_genes']} genes, found {adata.n_vars}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f'{path.name} has non-unique observation or gene names')
    for key in (dataset['label_key'], dataset['z_key']):
        if str(key) not in adata.obs:
            raise KeyError(f'{path.name} is missing obs[{key!r}]')
    for key in (dataset['spatial_key'], dataset['spatial_3d_key']):
        if str(key) not in adata.obsm:
            raise KeyError(f'{path.name} is missing obsm[{key!r}]')
    if not adata.uns.get('counts_layer_present', str(dataset['counts_layer']) in adata.layers):
        raise KeyError(f"{path.name} is missing layers[{dataset['counts_layer']!r}]")
    raw_labels = adata.obs[str(dataset['label_key'])]
    labels = raw_labels.astype(str)
    if raw_labels.isna().any() or (labels.str.len() == 0).any():
        raise ValueError(f'{path.name} contains missing/empty class labels')
    z_values = np.asarray(adata.obs[str(dataset['z_key'])], dtype=float)
    if not np.all(np.isfinite(z_values)) or not np.allclose(z_values, z_values[0], rtol=0.0, atol=1e-08):
        raise ValueError(f'{path.name} has invalid or nonconstant z')
    spatial = np.asarray(adata.obsm[str(dataset['spatial_key'])], dtype=float)
    spatial_3d = np.asarray(adata.obsm[str(dataset['spatial_3d_key'])], dtype=float)
    if spatial.shape != (adata.n_obs, 2) or spatial_3d.shape != (adata.n_obs, 3):
        raise ValueError(f'{path.name} has an invalid spatial coordinate shape')
    if not np.all(np.isfinite(spatial)) or not np.all(np.isfinite(spatial_3d)):
        raise ValueError(f'{path.name} has non-finite spatial coordinates')
    if not np.array_equal(spatial, spatial_3d[:, :2]):
        raise ValueError(f'{path.name} spatial XY differs from spatial_3d XY')
    if not np.allclose(spatial_3d[:, 2], z_values, rtol=0.0, atol=1e-08):
        raise ValueError(f'{path.name} spatial_3d z differs from obs z')

def slice_id_from_path(path: Path, prefix: str) -> int:
    expected = f'{prefix}.'
    if not path.stem.startswith(expected):
        raise ValueError(f'unexpected input filename: {path.name}')
    try:
        return int(path.stem[len(expected):])
    except ValueError as exc:
        raise ValueError(f'cannot parse slice ID from {path.name}') from exc

def plan_density(spec, assignments, reference_ids, slice_infos, label_sets, label_counts, prefix: str, n_references=5) -> list[dict[str, Any]]:
    label_support: dict[str, list[int]] = {}
    for reference_id in reference_ids:
        for label in label_sets[reference_id]:
            label_support.setdefault(label, []).append(reference_id)
    for values in label_support.values():
        values.sort()
    targets: list[dict[str, Any]] = []
    for assignment in assignments:
        from FEAST.de_novo import select_z_references
        primary, weights, bandwidth = select_z_references({item: slice_infos[item].z for item in reference_ids}, assignment.target_z, n_references=n_references)
        bracket_labels = set().union(*(label_sets[item] for item in primary))
        missing_labels = sorted(label_sets[assignment.target_slice] - bracket_labels)
        donors: list[dict[str, Any]] = []
        for label in missing_labels:
            donor_id = choose_donor(label, assignment.target_slice, assignment.target_z, label_support, slice_infos)
            donor_z = float(slice_infos[donor_id].z)
            donors.append({'label': label, 'donor_slice': donor_id, 'donor_z': donor_z, 'side': 'lower' if donor_z <= assignment.target_z else 'upper', 'n_spots': int(label_counts[donor_id][label]), 'artifact_id': f'{prefix}.{donor_id:03d}'})
        row = assignment.to_dict()
        row.update({'target_artifact_id': f'{prefix}.{assignment.target_slice:03d}', 'lower_reference_artifact_id': f'{prefix}.{assignment.lower_ref_slice:03d}', 'upper_reference_artifact_id': f'{prefix}.{assignment.upper_ref_slice:03d}', 'reference_weights': {f'{prefix}.{assignment.lower_ref_slice:03d}': float(1.0 - assignment.tau), f'{prefix}.{assignment.upper_ref_slice:03d}': float(assignment.tau)}, 'donor_support': donors})
        row.update(primary_reference_slices=primary, bandwidth=bandwidth, n_references=n_references, reference_weights={f'{prefix}.{item:03d}': value for item, value in weights.items()})
        targets.append(row)
    return targets

def estimate_density_ar(spec, reference_ids, slice_infos, config) -> dict[str, Any]:
    from FEAST.de_novo import calibrate_local_references
    from run import read_reference, simulation_config
    references = [read_reference(slice_infos[item].path, f"{config['dataset']['filename_prefix']}.{item:03d}", config['dataset']['label_key']) for item in reference_ids]
    z = {ref.uns['reference_name']: slice_infos[item].z for ref, item in zip(references, reference_ids)}
    return calibrate_local_references(references, label_key=config['dataset']['label_key'], n_references=config['generation']['n_references'], reference_z=z, config=simulation_config(config, 0.0, False), random_seed=int(config['public_seed']), cache_path=STUDY_ROOT / config['generation']['parameter_cache'], min_positions=config['generation']['min_positions'], batch_sd=config['generation']['batch_sd'])

def build_plan(config: dict[str, Any], data_dir: Path, slice_infos: dict[int, SliceInfo], label_sets: dict[int, set[str]], label_counts: dict[int, Counter[str]], genes: list[str], *, include_assignment_randomness: bool, calibration_plan: dict[str, Any] | None=None) -> dict[str, Any]:
    if calibration_plan is not None and calibration_plan['gene_names'] != genes:
        raise ValueError('the saved calibration uses a different gene panel or order')
    prefix = str(config['dataset']['filename_prefix'])
    available = sorted(slice_infos, key=lambda item: (float(slice_infos[item].z), int(item)))
    plan: dict[str, Any] = {'configuration_id': config['configuration_id'], 'data_dir': str(data_dir), 'available_slices': available, 'excluded_conflicting_slices': list(map(int, config['dataset'].get('excluded_conflicting_slices', []))), 'intentionally_absent_slices': list(map(int, config['dataset']['intentionally_absent_slices'])), 'gene_names': genes, 'densities': {}}
    for spec in density_specs(config):
        assignments, reference_ids = build_assignments(slice_infos, spec, int(config['public_seed']))
        targets = plan_density(spec, assignments, reference_ids, slice_infos, label_sets, label_counts, prefix, config['generation']['n_references'])
        density: dict[str, Any] = {'spec': {'name': spec.name, 'gap': spec.gap, 'extra_anchor_slices': list(map(int, spec.extra_anchor_slices)), 'expected_references': spec.expected_references, 'expected_targets': spec.expected_targets}, 'reference_pool': reference_ids, 'target_pool': [int(row.target_slice) for row in assignments], 'targets': targets}
        if include_assignment_randomness:
            if calibration_plan is None:
                density['assignment_randomness_preflight'] = estimate_density_ar(spec, reference_ids, slice_infos, config)
            else:
                previous = calibration_plan['densities'][spec.name]
                if previous['reference_pool'] != reference_ids or previous['target_pool'] != density['target_pool']:
                    raise ValueError(f'{spec.name}: the saved calibration uses a different reference/target partition')
                density['assignment_randomness_preflight'] = previous['assignment_randomness_preflight']
        plan['densities'][spec.name] = density
    return plan

def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    config = load_config(config_path)
    if output_dir is not None and output_dir.exists():
        recorded = output_dir / 'run_config.yaml'
        if not recorded.is_file() or recorded.read_text() != config_path.read_text():
            raise ValueError('partial preflight root has a different configuration')
        if (output_dir / 'preflight.json').exists():
            raise FileExistsError('preflight is already complete')
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not args.plan_only and output_dir is None:
        raise ValueError('--output-dir is required unless --plan-only is used')
    manifest, slice_infos, label_sets, label_counts, genes = inspect_inputs(data_dir, config)
    if args.plan_only:
        plan = build_plan(config, data_dir, slice_infos, label_sets, label_counts, genes, include_assignment_randomness=False)
        summary = {'configuration_id': config['configuration_id'], 'available_slices': len(plan['available_slices']), 'excluded_conflicting_slices': plan['excluded_conflicting_slices'], 'intentionally_absent_slices': plan['intentionally_absent_slices'], 'densities': {name: {'real': len(density['reference_pool']), 'simulated': len(density['target_pool']), 'donor_supported_targets': sum((bool(row['donor_support']) for row in density['targets'])), 'reference_pool': density['reference_pool']} for name, density in plan['densities'].items()}}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    assert output_dir is not None
    runtime = feast_runtime(config)
    import FEAST
    if FEAST.__version__ != str(config['feast_version']):
        raise ValueError(f"FEAST version mismatch: expected {config['feast_version']}, observed {FEAST.__version__}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / 'run_config.yaml')
    manifest_path = output_dir / 'input_manifest.csv'
    manifest.to_csv(manifest_path, index=False)
    calibration_plan = json.loads(args.calibration_plan.read_text()) if args.calibration_plan else None
    plan = build_plan(config, data_dir, slice_infos, label_sets, label_counts, genes, include_assignment_randomness=True, calibration_plan=calibration_plan)
    plan.update({'feast_version': FEAST.__version__})
    plan_path = output_dir / 'plan.json'
    write_json(plan_path, plan)
    common = {'configuration_id': config['configuration_id'], 'config_path': str(config_path), 'feast_version': FEAST.__version__, 'feast_module_path': str(Path(FEAST.__file__).resolve()), 'data_dir': str(data_dir), 'expected_outputs': int(config['validation']['expected_outputs']), 'calibration_reestimated': args.calibration_plan is None, 'calibration_plan': str(args.calibration_plan.resolve()) if args.calibration_plan else None}
    write_json(output_dir / 'preflight.json', {**common, 'runtime': runtime, 'status': 'passed'})
    target_count = sum((len(density['targets']) for density in plan['densities'].values()))
    print(f'Preflight passed: {len(manifest)} modeled inputs and {target_count} target assignments frozen in {output_dir}')
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
