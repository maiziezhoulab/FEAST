"""Generate fresh Study 06 targets from one frozen preflight plan."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
import anndata as ad
from anndata.experimental import read_elem
import h5py
import numpy as np
import scipy.sparse as sp
from workflow import load_config, read_json, feast_runtime, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Root created by preflight.py')
    parser.add_argument('--gap', type=int, choices=(3, 5, 10), required=True)
    parser.add_argument('--target-id', type=int, nargs='+', default=None)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def load_frozen_run(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight_path = output_dir / 'preflight.json'
    plan_path = output_dir / 'plan.json'
    config_path = output_dir / 'run_config.yaml'
    manifest_path = output_dir / 'input_manifest.csv'
    for path in (preflight_path, plan_path, config_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f'frozen preflight artifact is missing: {path}')
    preflight = read_json(preflight_path)
    if preflight.get('status') != 'passed':
        raise ValueError('preflight status is not passed')
    config = load_config(config_path)
    plan = read_json(plan_path)
    runtime = feast_runtime(config)
    import FEAST
    if FEAST.__version__ != config['feast_version'] or FEAST.__version__ != preflight['feast_version']:
        raise ValueError('FEAST version changed after preflight')
    return (config, plan, preflight, runtime)

def density_for_gap(plan: Mapping[str, Any], gap: int) -> tuple[str, dict[str, Any]]:
    matches = [(name, density) for name, density in plan['densities'].items() if int(density['spec']['gap']) == int(gap)]
    if len(matches) != 1:
        raise ValueError(f'expected one frozen density for gap {gap}, found {len(matches)}')
    return matches[0]

def select_targets(rows: Sequence[dict[str, Any]], target_ids: Sequence[int] | None, shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('shard contract requires 0 <= shard-index < shard-count')
    selected = [row for row in rows if int(row['target_index']) % shard_count == shard_index]
    if target_ids is not None:
        requested = {int(item) for item in target_ids}
        available = {int(row['target_slice']) for row in rows}
        missing = requested - available
        if missing:
            raise ValueError(f'requested target IDs are not in this density: {sorted(missing)}')
        selected = [row for row in selected if int(row['target_slice']) in requested]
    if not selected:
        raise ValueError('target selection is empty')
    return selected

def input_path(plan: Mapping[str, Any], config: Mapping[str, Any], slice_id: int) -> Path:
    prefix = str(config['dataset']['filename_prefix'])
    return Path(plan['data_dir']) / f'{prefix}.{int(slice_id):03d}.h5ad'

def verify_target_inputs(row: Mapping[str, Any], plan: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [('target_metadata_only_generation_expression_evaluation_only', int(row['target_slice'])), ('lower_reference_expression', int(row['lower_ref_slice'])), ('upper_reference_expression', int(row['upper_ref_slice']))]
    records.extend((('primary_reference_expression', int(item)) for item in row['primary_reference_slices']))
    records.extend((('declared_label_donor_expression', int(item['donor_slice'])) for item in row['donor_support']))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for role, slice_id in records:
        key = (role, slice_id)
        if key in seen:
            continue
        seen.add(key)
        path = input_path(plan, config, slice_id)
        if not path.is_file():
            raise FileNotFoundError(f'input slice {slice_id:03d} is missing: {path}')
        unique.append({'role': role, 'artifact_id': f"{config['dataset']['filename_prefix']}.{slice_id:03d}", 'slice_id': slice_id})
    return unique

def validate_reference_scope(row: Mapping[str, Any], density: Mapping[str, Any]) -> None:
    references = {int(item) for item in density['reference_pool']}
    target = int(row['target_slice'])
    declared = set(map(int, row['primary_reference_slices']))
    declared.update((int(item['donor_slice']) for item in row['donor_support']))
    if target in references:
        raise ValueError(f'target {target:03d} is also present in the retained reference pool')
    if not declared <= references:
        raise ValueError(f'target {target:03d} declares expression outside the retained reference pool')

@lru_cache(maxsize=64)
def read_reference(path: Path, reference_name: str, label_key: str) -> ad.AnnData:
    reference = ad.read_h5ad(path)
    reference.obs = reference.obs.copy()
    reference.obs[label_key] = reference.obs[label_key].astype(str)
    reference.uns['reference_name'] = reference_name
    return reference

def target_metadata(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config['dataset']
    with h5py.File(path, 'r') as handle:
        obs = read_elem(handle['obs'])
        var = read_elem(handle['var'])
        return {'obs_names': list(map(str, obs.index)), 'var_names': list(map(str, var.index)), 'labels': obs[str(dataset['label_key'])].astype(str).to_numpy(), 'z': np.asarray(obs[str(dataset['z_key'])], dtype=float), 'spatial': np.asarray(read_elem(handle[f"obsm/{dataset['spatial_key']}"]), dtype=float), 'spatial_3d': np.asarray(read_elem(handle[f"obsm/{dataset['spatial_3d_key']}"]), dtype=float), 'grid_type': read_elem(handle['uns/grid_type']) if 'uns/grid_type' in handle else 'generic', 'technology': read_elem(handle['uns/technology']) if 'uns/technology' in handle else None}

def make_blueprint(metadata: Mapping[str, Any], label_key: str):
    import pandas as pd
    from FEAST.de_novo import SimulationBlueprint
    labels = np.asarray(metadata['labels'], dtype=str)
    obs = pd.DataFrame({'_target_obs_name': list(metadata['obs_names']), label_key: labels, 'domain': labels})
    return SimulationBlueprint(coordinates=np.asarray(metadata['spatial_3d'], dtype=float), grid_type=metadata['grid_type'], domain_map=labels, technology=metadata['technology'], obs=obs, metadata={'source': 'target_metadata_only', 'label_key': label_key})

def simulation_config(config: Mapping[str, Any], assignment_randomness: float, verbose: bool):
    from FEAST.de_novo import SimulationConfig
    generation = config['generation']
    transport = config['transport']
    return SimulationConfig(epsilon=float(transport['epsilon']), sinkhorn_iter=int(transport['sinkhorn_iter']), sinkhorn_tol=float(transport['sinkhorn_tol']), unbalanced_transport=bool(transport['unbalanced_transport']), reg_m=float(transport['reg_m']), transport_nonconvergence=str(transport['transport_nonconvergence']), sinkhorn_method=str(transport['sinkhorn_method']), transport_backend=str(transport['transport_backend']), transport_device=str(transport['transport_device']), transport_dtype=str(transport['transport_dtype']), geometry_weight=float(transport['geometry_weight']), boundary_weight=float(transport['boundary_weight']), assignment_randomness=float(assignment_randomness), latent_clip_eps=float(transport['latent_clip_eps']), gene_chunk_size=int(transport['gene_chunk_size']), max_transport_pairs=int(transport['max_transport_pairs']), reference_weight_eta=float(generation['reference_weight_eta']), boundary_multiplier=float(generation['boundary_multiplier']), diffusion_level=float(generation['diffusion_level']), boundary_softness=float(generation['boundary_softness']), coordinate_scale=config['reference_fit']['coordinate_scale'], verbose=bool(verbose), quantile_field_mode=str(generation['quantile_field_mode']), rank_scope=str(generation['rank_scope']), target_parameter_mode=str(generation['target_parameter_mode']), tie_policy=str(generation['tie_policy']), tie_jitter_scale=float(generation['tie_jitter_scale']), min_rank_scope_size=int(generation['min_rank_scope_size']), store_latent_scores=bool(generation['store_latent_scores']), store_quantiles=generation['store_quantiles'], reference_conflict_policy=str(generation['reference_conflict_policy']))

def convergence_summary(generated: ad.AnnData, transport: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = generated.uns.get('de_novo', {}).get('transport_diagnostics', {})
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ValueError('generated output lacks transport diagnostics')
    total = 0
    maximum_error = 0.0
    minimum_mass = float('inf')
    for label, records in diagnostics.items():
        count = int(records.get('n_records', 0))
        if count < 1:
            raise ValueError(f'label {label!r} has no transport records')
        required = ('transport_converged', 'transport_final_error', 'transport_stop_threshold', 'transport_max_iterations', 'transport_nonconvergence_policy', 'transport_mass', 'transport_solver_method', 'transport_backend', 'transport_device', 'transport_dtype')
        if any((key not in records for key in required)):
            raise ValueError(f'label {label!r} lacks positive convergence fields')
        for index in range(count):
            converged = str(records['transport_converged'][index]).lower() == 'true'
            error = float(records['transport_final_error'][index])
            tolerance = float(records['transport_stop_threshold'][index])
            policy = str(records['transport_nonconvergence_policy'][index])
            mass = float(records['transport_mass'][index])
            method = str(records['transport_solver_method'][index])
            backend = str(records['transport_backend'][index])
            device = str(records['transport_device'][index])
            dtype = str(records['transport_dtype'][index])
            if not converged or not np.isfinite(error) or (not error < tolerance) or (policy != 'raise') or (not np.isfinite(mass)) or (mass <= 0.0) or (method != str(transport['sinkhorn_method'])) or (backend != str(transport['transport_backend'])) or (device != str(transport['transport_device'])) or (dtype != str(transport['transport_dtype'])):
                raise ValueError(f'label {label!r} transport record {index} lacks positive convergence evidence')
            maximum_error = max(maximum_error, error)
            minimum_mass = min(minimum_mass, mass)
            total += 1
    return {'transport_records': total, 'all_converged': True, 'maximum_final_error': maximum_error, 'minimum_transport_mass': minimum_mass}

def dense_counts(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix)

def validate_generated_in_memory(generated: ad.AnnData, metadata: Mapping[str, Any], row: Mapping[str, Any], transport: Mapping[str, Any]) -> None:
    expected_shape = (len(metadata['obs_names']), len(metadata['var_names']))
    if generated.shape != expected_shape:
        raise ValueError(f'generated shape {generated.shape} differs from target metadata {expected_shape}')
    if list(map(str, generated.obs_names)) != list(metadata['obs_names']):
        raise ValueError('generated observation identity/order changed')
    if list(map(str, generated.var_names)) != list(metadata['var_names']):
        raise ValueError('generated gene identity/order changed')
    if not np.array_equal(generated.obs['class'].astype(str).to_numpy(), metadata['labels']):
        raise ValueError('generated class labels changed')
    if not np.array_equal(np.asarray(generated.obs['z'], dtype=float), metadata['z']):
        raise ValueError('generated z metadata changed')
    if not np.array_equal(np.asarray(generated.obsm['spatial'], dtype=float), metadata['spatial']):
        raise ValueError('generated XY coordinates changed')
    if not np.array_equal(np.asarray(generated.obsm['spatial_3d'], dtype=float), metadata['spatial_3d']):
        raise ValueError('generated 3D coordinates changed')
    counts = dense_counts(generated)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or (not np.array_equal(counts, np.rint(counts))):
        raise ValueError('generated counts must be finite, nonnegative integers')
    x_values = generated.X.toarray() if sp.issparse(generated.X) else np.asarray(generated.X)
    if not np.array_equal(counts, x_values):
        raise ValueError('generated X and counts layer differ')
    if int(generated.uns['study06']['seed']) != int(row['seed']):
        raise ValueError('generated seed metadata changed')
    convergence_summary(generated, transport)
_SPATIAL_FITS = {}

def generate_target(row: dict[str, Any], density: Mapping[str, Any], config: dict[str, Any], plan: dict[str, Any], preflight: dict[str, Any], runtime: dict[str, Any], output_dir: Path, verbose: bool) -> dict[str, Any]:
    from FEAST.de_novo import simulate_local_references
    import FEAST
    prefix = str(config['dataset']['filename_prefix'])
    label_key = str(config['dataset']['label_key'])
    target_name = f"{prefix}.{int(row['target_slice']):03d}"
    final_dir = output_dir / 'targets' / str(row['density_name']) / target_name
    work_dir = output_dir / '.work' / str(row['density_name']) / target_name
    if final_dir.exists():
        generated = ad.read_h5ad(final_dir / 'generated.h5ad')
        metadata = target_metadata(input_path(plan, config, int(row['target_slice'])), config)
        validate_generated_in_memory(generated, metadata, row, config['transport'])
        if generated.uns['study06']['configuration_id'] != config['configuration_id']:
            raise ValueError('resumed target belongs to another configuration')
        return {'target_slice': int(row['target_slice']), 'target_index': int(row['target_index']), 'target_dir': str(final_dir.relative_to(output_dir))}
    if work_dir.exists():
        raise FileExistsError(f'fresh-only target path already exists: {(final_dir if final_dir.exists() else work_dir)}')
    work_dir.mkdir(parents=True)
    lower: ad.AnnData | None = None
    upper: ad.AnnData | None = None
    try:
        validate_reference_scope(row, density)
        input_artifacts = verify_target_inputs(row, plan, config)
        target_meta = target_metadata(input_path(plan, config, int(row['target_slice'])), config)
        references = [read_reference(input_path(plan, config, item), f'{prefix}.{item:03d}', label_key) for item in density['reference_pool']]
        z = {ref.uns['reference_name']: float(np.unique(ref.obs['z'])[0]) for ref in references}
        ar = float(density['assignment_randomness_preflight']['observed_assignment_randomness'])
        generated = simulate_local_references(references, make_blueprint(target_meta, label_key), label_key=label_key, n_references=config['generation']['n_references'], reference_z=z, target_z=float(row['target_z']), config=simulation_config(config, ar, verbose), random_seed=int(row['seed']), parameter_seed=int(config['public_seed']), batch_seed=int(config['public_seed']) + int(row['target_slice']), cache_path=output_dir / 'reference_parameters.sqlite', fit_cache=_SPATIAL_FITS, min_positions=config['generation']['min_positions'], batch_sd=config['generation']['batch_sd'])
        del references
        generated = generated[:, target_meta['var_names']].copy()
        generated.obs_names = list(target_meta['obs_names'])
        if '_target_obs_name' in generated.obs:
            generated.obs.drop(columns=['_target_obs_name'], inplace=True)
        generated.obs[label_key] = np.asarray(target_meta['labels'], dtype=str)
        generated.obs['z'] = np.asarray(target_meta['z'], dtype=float)
        generated.obsm['spatial'] = np.asarray(target_meta['spatial'], dtype=float)
        generated.obsm['spatial_3d'] = np.asarray(target_meta['spatial_3d'], dtype=float)
        generated.uns['study06'] = {'configuration_id': str(config['configuration_id']), 'feast_version': str(config['feast_version']), 'target_slice': int(row['target_slice']), 'target_index': int(row['target_index']), 'density_gap': int(row['gap']), 'density_name': str(row['density_name']), 'seed': int(row['seed']), 'assignment_randomness': ar, 'lower_ref_slice': int(row['lower_ref_slice']), 'upper_ref_slice': int(row['upper_ref_slice']), 'tau': float(row['tau']), 'reference_weights': {str(key): float(value) for key, value in row['reference_weights'].items()}, 'donor_support_json': json.dumps(row['donor_support'], sort_keys=True, separators=(',', ':')), 'target_expression_accessed': False, 'smoothing': False, 'z_regularization': False}
        diagnostics = convergence_summary(generated, config['transport'])
        validate_generated_in_memory(generated, target_meta, row, config['transport'])
        h5ad_path = work_dir / 'generated.h5ad'
        generated.write_h5ad(h5ad_path, compression='gzip')
        restored = ad.read_h5ad(h5ad_path)
        try:
            validate_generated_in_memory(restored, target_meta, row, config['transport'])
            restored_diagnostics = convergence_summary(restored, config['transport'])
            if restored_diagnostics != diagnostics:
                raise ValueError('transport diagnostics changed during H5AD round-trip')
        finally:
            del restored
        provenance = {'status': 'complete', 'configuration_id': config['configuration_id'], 'feast_version': FEAST.__version__, 'public_seed': int(config['public_seed']), 'target_seed': int(row['seed']), 'target_index': int(row['target_index']), 'density_gap': int(row['gap']), 'density_name': str(row['density_name']), 'target_slice': int(row['target_slice']), 'lower_ref_slice': int(row['lower_ref_slice']), 'upper_ref_slice': int(row['upper_ref_slice']), 'tau': float(row['tau']), 'reference_weights': row['reference_weights'], 'donor_support': row['donor_support'], 'assignment_randomness': ar, 'assignment_randomness_preflight': density['assignment_randomness_preflight'], 'target_expression_accessed': False, 'target_observed_fields': config['generation']['target_observed_fields'], 'smoothing': False, 'z_regularization': False, 'transport': config['transport'], 'solver_diagnostics': diagnostics, 'input_artifacts': input_artifacts, 'output': {'filename': 'generated.h5ad'}, 'runner': {'path': Path(__file__).name}, 'python': sys.version, 'platform': platform.platform(), 'runtime': runtime, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(work_dir / 'provenance.json', provenance)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_dir, final_dir)
        return {'target_slice': int(row['target_slice']), 'target_index': int(row['target_index']), 'target_dir': str(final_dir.relative_to(output_dir))}
    except Exception as exc:
        write_json(work_dir / 'FAILED.json', {'status': 'failed', 'target_slice': int(row['target_slice']), 'seed': int(row['seed']), 'error_type': type(exc).__name__, 'error': str(exc), 'failed_at_utc': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        del lower, upper

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config, plan, preflight, runtime = load_frozen_run(output_dir)
    density_name, density = density_for_gap(plan, args.gap)
    targets = select_targets(density['targets'], args.target_id, int(args.shard_index), int(args.shard_count))
    completed: list[dict[str, Any]] = []
    for position, row in enumerate(targets, start=1):
        print(f"[{density_name}] {position}/{len(targets)} target {int(row['target_slice']):03d} seed={int(row['seed'])}", flush=True)
        completed.append(generate_target(row, density, config, plan, preflight, runtime, output_dir, bool(args.verbose)))
    manifest_name = f"gap_{args.gap}_targets_{'-'.join((str(item) for item in args.target_id))}.json" if args.target_id is not None else f'gap_{args.gap}_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json'
    manifest_path = output_dir / 'shard_manifests' / manifest_name
    if manifest_path.exists():
        raise FileExistsError(f'fresh-only shard manifest exists: {manifest_path}')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {'configuration_id': config['configuration_id'], 'density_name': density_name, 'gap': int(args.gap), 'shard_index': int(args.shard_index), 'shard_count': int(args.shard_count), 'targets': completed})
    print(f'Completed {len(completed)} fresh targets; manifest: {manifest_path}')
if __name__ == '__main__':
    main()
