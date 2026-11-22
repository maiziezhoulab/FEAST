"""Independently validate every fresh Study 06 target and its provenance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from workflow import load_config, read_json, feast_runtime, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--canary-target', action='append', default=None, metavar='GAP:SLICE')
    return parser.parse_args()

def load_frozen(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight = read_json(output_dir / 'preflight.json')
    if preflight.get('status') != 'passed':
        raise ValueError('preflight status is not passed')
    config_path = output_dir / 'run_config.yaml'
    plan_path = output_dir / 'plan.json'
    manifest_path = output_dir / 'input_manifest.csv'
    config = load_config(config_path)
    plan = read_json(plan_path)
    runtime = feast_runtime(config)
    return (config, plan, preflight)

def verify_source_files(output_dir: Path, plan: Mapping[str, Any], config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = pd.read_csv(output_dir / 'input_manifest.csv')
    expected_positions = int(config['validation']['expected_positions_per_stack'])
    if len(manifest) != expected_positions or manifest['slice_id'].nunique() != expected_positions:
        raise ValueError(f'input manifest must contain {expected_positions} unique modeled slices')
    by_id: dict[int, dict[str, Any]] = {}
    data_dir = Path(plan['data_dir'])
    for record in manifest.to_dict('records'):
        slice_id = int(record['slice_id'])
        path = data_dir / str(record['filename'])
        if not path.is_file():
            raise FileNotFoundError(f'source input {slice_id:03d} is missing: {path}')
        by_id[slice_id] = record
    return by_id

def validate_plan_contract(plan: Mapping[str, Any], config: Mapping[str, Any], sources: Mapping[int, Mapping[str, Any]]) -> None:
    available = [int(item) for item in plan['available_slices']]
    if available != sorted(sources, key=lambda item: (float(sources[item]['z']), int(item))):
        raise ValueError('frozen available-slice order differs from the modeled source manifest')
    if {1, 2, 3} & set(available) or {75, 94, 112} & set(available):
        raise ValueError('excluded or absent slices entered the corrected modeled population')
    expected_by_gap = {int(row['gap']): row for row in config['densities']}
    for density in plan['densities'].values():
        gap = int(density['spec']['gap'])
        spec = expected_by_gap[gap]
        references = {int(item) for item in density['reference_pool']}
        targets = {int(item) for item in density['target_pool']}
        if len(references) != int(spec['expected_references']) or len(targets) != int(spec['expected_targets']):
            raise ValueError(f'gap {gap} real/simulated counts changed')
        if references & targets or references | targets != set(available):
            raise ValueError(f'gap {gap} is not an exact real/target partition')
        if not {4, 150} <= references:
            raise ValueError(f'gap {gap} lost a real boundary anchor')
        if gap == 10 and 89 not in references:
            raise ValueError('gap 10 lost required class-support anchor 89')
        rows = {int(row['target_slice']): row for row in density['targets']}
        if set(rows) != targets:
            raise ValueError(f'gap {gap} target rows differ from the target pool')
        for target, row in rows.items():
            target_z = float(sources[target]['z'])
            lower = max((item for item in references if float(sources[item]['z']) < target_z), key=lambda item: float(sources[item]['z']))
            upper = min((item for item in references if float(sources[item]['z']) > target_z), key=lambda item: float(sources[item]['z']))
            if (int(row['lower_ref_slice']), int(row['upper_ref_slice'])) != (lower, upper):
                raise ValueError(f'gap {gap} target {target:03d} does not use its nearest real anchors')
            tau = (target_z - float(sources[lower]['z'])) / (float(sources[upper]['z']) - float(sources[lower]['z']))
            if not np.isclose(float(row['tau']), tau, rtol=0.0, atol=1e-15):
                raise ValueError(f'gap {gap} target {target:03d} has an incorrect actual-z tau')
            donors = {int(item['donor_slice']) for item in row['donor_support']}
            if not donors <= references:
                raise ValueError(f'gap {gap} target {target:03d} uses a donor outside the real pool')

def stack_manifest(output_dir: Path, plan: Mapping[str, Any], config: Mapping[str, Any], sources: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    prefix = str(config['dataset']['filename_prefix'])
    records: list[dict[str, Any]] = []
    for density_name, density in plan['densities'].items():
        references = {int(item) for item in density['reference_pool']}
        for stack_index, slice_id in enumerate(map(int, plan['available_slices'])):
            source_kind = 'real' if slice_id in references else 'simulated'
            path = Path(plan['data_dir']) / str(sources[slice_id]['filename']) if source_kind == 'real' else output_dir / 'targets' / density_name / f'{prefix}.{slice_id:03d}' / 'generated.h5ad'
            records.append({'configuration_id': config['configuration_id'], 'density_name': density_name, 'gap': int(density['spec']['gap']), 'stack_index': stack_index, 'slice_id': slice_id, 'z': float(sources[slice_id]['z']), 'source_kind': source_kind, 'path': str(path)})
    frame = pd.DataFrame(records).sort_values(['gap', 'stack_index'])
    expected_positions = int(config['validation']['expected_positions_per_stack'])
    canonical_geometry: pd.DataFrame | None = None
    for _name, group in frame.groupby('density_name'):
        group = group.sort_values('stack_index')
        if len(group) != expected_positions or group['slice_id'].tolist() != list(map(int, plan['available_slices'])):
            raise ValueError('assembled stack does not contain the complete modeled geometry')
        if set(group['source_kind']) != {'real', 'simulated'}:
            raise ValueError('assembled stack must contain both real and simulated slices')
        geometry = group[['slice_id', 'z']].reset_index(drop=True)
        if canonical_geometry is None:
            canonical_geometry = geometry
        elif not geometry.equals(canonical_geometry):
            raise ValueError('assembled stack slice positions or z coordinates differ across density arms')
    return frame

def all_target_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for density in plan['densities'].values() for row in density['targets']]

def select_canary_rows(rows: Sequence[dict[str, Any]], specifications: Sequence[str]) -> list[dict[str, Any]]:
    requested: set[tuple[int, int]] = set()
    for value in specifications:
        try:
            gap_text, slice_text = str(value).split(':', 1)
            requested.add((int(gap_text), int(slice_text)))
        except ValueError as exc:
            raise ValueError(f'invalid canary target {value!r}; expected GAP:SLICE') from exc
    by_key = {(int(row['gap']), int(row['target_slice'])): row for row in rows}
    missing = requested - set(by_key)
    if missing:
        raise ValueError(f'canary targets are absent from the corrected plan: {sorted(missing)}')
    return [by_key[key] for key in sorted(requested)]

def validate_target_directory_set(output_dir: Path, rows: Sequence[Mapping[str, Any]], prefix: str) -> None:
    expected = {Path('targets') / str(row['density_name']) / f"{prefix}.{int(row['target_slice']):03d}" for row in rows}
    targets_root = output_dir / 'targets'
    observed = {path.relative_to(output_dir) for path in targets_root.glob('*/*') if path.is_dir()} if targets_root.is_dir() else set()
    if observed != expected:
        raise ValueError(f'final target directory set changed; missing={sorted(map(str, expected - observed))}, extra={sorted(map(str, observed - expected))}')
    work_root = output_dir / '.work'
    if work_root.exists() and any((path.is_file() for path in work_root.rglob('*'))):
        raise ValueError('production root retains failed or incomplete .work artifacts')

def target_metadata(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config['dataset']
    target = ad.read_h5ad(path, backed='r')
    try:
        return {'shape': (int(target.n_obs), int(target.n_vars)), 'obs_names': list(map(str, target.obs_names)), 'var_names': list(map(str, target.var_names)), 'labels': target.obs[str(dataset['label_key'])].astype(str).to_numpy(), 'z': np.asarray(target.obs[str(dataset['z_key'])], dtype=float), 'spatial': np.asarray(target.obsm[str(dataset['spatial_key'])], dtype=float), 'spatial_3d': np.asarray(target.obsm[str(dataset['spatial_3d_key'])], dtype=float)}
    finally:
        target.file.close()

def dense_matrix(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)

def validate_count_matrix(matrix: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    values = dense_matrix(matrix)
    if values.shape != expected_shape:
        raise ValueError(f'count matrix shape changed: {values.shape} != {expected_shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('count matrix contains non-finite values')
    if np.any(values < 0):
        raise ValueError('count matrix contains negative values')
    if not np.array_equal(values, np.rint(values)):
        raise ValueError('count matrix contains non-integral values')
    return values

def column_values(records: Mapping[str, Any], key: str, count: int) -> list[str]:
    if key not in records:
        raise ValueError(f'transport diagnostics lack {key!r}')
    values = np.asarray(records[key]).reshape(-1).tolist()
    if len(values) != count:
        raise ValueError(f'transport diagnostic {key!r} length changed')
    return [str(value) for value in values]

def validate_transport_diagnostics(diagnostics: Mapping[str, Any], transport: Mapping[str, Any], allowed_references: set[str]) -> tuple[int, float]:
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ValueError('transport diagnostics are missing')
    total = 0
    maximum_error = 0.0
    for label, records in diagnostics.items():
        count = int(records.get('n_records', 0))
        if count < 1 or records.get('format') != 'columnar_records_v1':
            raise ValueError(f'label {label!r} has invalid transport records')
        converged = column_values(records, 'transport_converged', count)
        errors = column_values(records, 'transport_final_error', count)
        tolerances = column_values(records, 'transport_stop_threshold', count)
        iterations = column_values(records, 'transport_iterations', count)
        maxima = column_values(records, 'transport_max_iterations', count)
        policies = column_values(records, 'transport_nonconvergence_policy', count)
        references = column_values(records, 'reference_name', count)
        masses = column_values(records, 'transport_mass', count)
        methods = column_values(records, 'transport_solver_method', count)
        backends = column_values(records, 'transport_backend', count)
        devices = column_values(records, 'transport_device', count)
        dtypes = column_values(records, 'transport_dtype', count)
        for index in range(count):
            error = float(errors[index])
            tolerance = float(tolerances[index])
            iteration = int(iterations[index])
            maximum = int(maxima[index])
            if converged[index].lower() != 'true':
                raise ValueError(f'label {label!r} record {index} is not converged')
            if not np.isfinite(error) or not error < tolerance:
                raise ValueError(f'label {label!r} record {index} lacks a finite sub-tolerance error')
            if tolerance != float(transport['sinkhorn_tol']):
                raise ValueError('recorded Sinkhorn tolerance differs from the frozen config')
            if maximum != int(transport['sinkhorn_iter']) or not 0 <= iteration <= maximum:
                raise ValueError('recorded Sinkhorn iteration contract changed')
            if policies[index] != 'raise':
                raise ValueError('transport nonconvergence policy is not strict')
            if methods[index] != str(transport['sinkhorn_method']):
                raise ValueError('transport solver method differs from the frozen config')
            if backends[index] != str(transport['transport_backend']):
                raise ValueError('transport backend differs from the frozen config')
            if devices[index] != str(transport['transport_device']):
                raise ValueError('transport device differs from the frozen config')
            if dtypes[index] != str(transport['transport_dtype']):
                raise ValueError('transport dtype differs from the frozen config')
            if references[index] not in allowed_references:
                raise ValueError(f'undeclared logical reference {references[index]!r}')
            mass = float(masses[index])
            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError(f'label {label!r} record {index} has nonpositive transport mass')
            maximum_error = max(maximum_error, error)
            total += 1
    return (total, maximum_error)

def expected_input_artifacts(row: Mapping[str, Any], prefix: str) -> set[tuple[str, int]]:
    records = {('target_metadata_only_generation_expression_evaluation_only', int(row['target_slice'])), ('lower_reference_expression', int(row['lower_ref_slice'])), ('upper_reference_expression', int(row['upper_ref_slice']))}
    records.update((('primary_reference_expression', int(item)) for item in row['primary_reference_slices']))
    records.update((('declared_label_donor_expression', int(item['donor_slice'])) for item in row['donor_support']))
    return records

def validate_provenance(provenance: Mapping[str, Any], row: Mapping[str, Any], config: Mapping[str, Any], preflight: Mapping[str, Any], h5ad_path: Path) -> None:
    required_equal = {'status': 'complete', 'configuration_id': config['configuration_id'], 'feast_version': config['feast_version'], 'public_seed': int(config['public_seed']), 'target_seed': int(row['seed']), 'target_index': int(row['target_index']), 'density_gap': int(row['gap']), 'density_name': str(row['density_name']), 'target_slice': int(row['target_slice']), 'lower_ref_slice': int(row['lower_ref_slice']), 'upper_ref_slice': int(row['upper_ref_slice']), 'target_expression_accessed': False, 'smoothing': False, 'z_regularization': False, 'transport': config['transport']}
    for key, expected in required_equal.items():
        if provenance.get(key) != expected:
            raise ValueError(f'provenance field {key!r} changed')
    if provenance.get('reference_weights') != row['reference_weights']:
        raise ValueError('provenance reference weights changed')
    if provenance.get('donor_support') != row['donor_support']:
        raise ValueError('provenance donor support changed')
    if not np.isclose(float(provenance['tau']), float(row['tau']), rtol=0.0, atol=1e-15):
        raise ValueError('provenance tau changed')
    observed_inputs = {(str(item['role']), int(item['slice_id'])) for item in provenance.get('input_artifacts', [])}
    if observed_inputs != expected_input_artifacts(row, str(config['dataset']['filename_prefix'])):
        raise ValueError('provenance input artifact set changed')
    solver = provenance.get('solver_diagnostics', {})
    if solver.get('all_converged') is not True or int(solver.get('transport_records', 0)) < 1:
        raise ValueError('provenance lacks positive solver diagnostics')

def validate_one(output_dir: Path, row: Mapping[str, Any], config: Mapping[str, Any], plan: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    prefix = str(config['dataset']['filename_prefix'])
    target_name = f"{prefix}.{int(row['target_slice']):03d}"
    target_dir = output_dir / 'targets' / str(row['density_name']) / target_name
    if {path.name for path in target_dir.iterdir()} != {'generated.h5ad', 'provenance.json'}:
        raise ValueError(f'{target_dir} must contain only generated.h5ad and provenance.json')
    h5ad_path = target_dir / 'generated.h5ad'
    provenance_path = target_dir / 'provenance.json'
    provenance = read_json(provenance_path)
    validate_provenance(provenance, row, config, preflight, h5ad_path)
    source_path = Path(plan['data_dir']) / f'{target_name}.h5ad'
    target = target_metadata(source_path, config)
    generated = ad.read_h5ad(h5ad_path)
    try:
        if generated.shape != target['shape']:
            raise ValueError(f'{target_name} generated shape changed')
        if list(map(str, generated.obs_names)) != target['obs_names']:
            raise ValueError(f'{target_name} spot identity/order changed')
        if list(map(str, generated.var_names)) != target['var_names']:
            raise ValueError(f'{target_name} gene identity/order changed')
        if not np.array_equal(generated.obs['class'].astype(str).to_numpy(), target['labels']):
            raise ValueError(f'{target_name} class labels changed')
        if not np.array_equal(np.asarray(generated.obs['z'], dtype=float), target['z']):
            raise ValueError(f'{target_name} z values changed')
        if not np.array_equal(np.asarray(generated.obsm['spatial'], dtype=float), target['spatial']):
            raise ValueError(f'{target_name} XY changed')
        if not np.array_equal(np.asarray(generated.obsm['spatial_3d'], dtype=float), target['spatial_3d']):
            raise ValueError(f'{target_name} XYZ changed')
        counts = validate_count_matrix(generated.layers['counts'], target['shape'])
        if not np.array_equal(counts, dense_matrix(generated.X)):
            raise ValueError(f'{target_name} X and counts layer differ')
        study = generated.uns.get('study06', {})
        required_study = {'configuration_id': config['configuration_id'], 'feast_version': config['feast_version'], 'target_slice': int(row['target_slice']), 'target_index': int(row['target_index']), 'density_gap': int(row['gap']), 'density_name': str(row['density_name']), 'seed': int(row['seed']), 'lower_ref_slice': int(row['lower_ref_slice']), 'upper_ref_slice': int(row['upper_ref_slice']), 'target_expression_accessed': False, 'smoothing': False, 'z_regularization': False}
        for key, expected in required_study.items():
            if study.get(key) != expected:
                raise ValueError(f'{target_name} study06 metadata {key!r} changed')
        if json.loads(str(study['donor_support_json'])) != row['donor_support']:
            raise ValueError(f'{target_name} donor support metadata changed')
        weights = {str(key): float(value) for key, value in study['reference_weights'].items()}
        if weights != {str(key): float(value) for key, value in row['reference_weights'].items()}:
            raise ValueError(f'{target_name} reference weights changed')
        local = json.loads(generated.uns['local_generation_json'])
        references = set(row['reference_weights']) | {item['artifact_id'] for item in row['donor_support']}
        if local['n_references'] != config['generation']['n_references']:
            raise ValueError('generated reference count setting differs')
        if set(local['primary_references']) != set(row['reference_weights']):
            raise ValueError('generated primary references differ from planned selection')
        if generated.uns['de_novo']['marginal_model'] != 'core':
            raise ValueError('Study 06 requires full core count generation')
        for group, group_weights in local['group_weights'].items():
            if not set(group_weights) <= references:
                raise ValueError(f'group {group} uses undeclared references')
            actual = set(map(str, generated.uns['de_novo']['transport_diagnostics'][group]['reference_name']))
            if actual != set(group_weights):
                raise ValueError('parameter and spatial reference selections differ')
        records, max_error = validate_transport_diagnostics(generated.uns.get('de_novo', {}).get('transport_diagnostics', {}), config['transport'], references)
    finally:
        del generated
    return {'configuration_id': config['configuration_id'], 'density_name': row['density_name'], 'gap': int(row['gap']), 'target_index': int(row['target_index']), 'target_slice': int(row['target_slice']), 'seed': int(row['seed']), 'n_obs': int(target['shape'][0]), 'n_vars': int(target['shape'][1]), 'donor_labels': len(row['donor_support']), 'transport_records': records, 'maximum_final_error': max_error, 'valid': True}

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config, plan, preflight = load_frozen(output_dir)
    sources = verify_source_files(output_dir, plan, config)
    validate_plan_contract(plan, config, sources)
    rows = all_target_rows(plan)
    if len(rows) != int(config['validation']['expected_outputs']):
        raise ValueError(f"frozen plan no longer contains exactly {config['validation']['expected_outputs']} targets")
    if args.canary_target:
        rows = select_canary_rows(rows, args.canary_target)
        validate_target_directory_set(output_dir, rows, str(config['dataset']['filename_prefix']))
        report_path = args.report.resolve() if args.report else output_dir / 'canary_validation.csv'
        summary_path = report_path.with_name('canary_validation_summary.json')
        if report_path.exists() or summary_path.exists():
            raise FileExistsError('canary validation outputs already exist')
        records = [validate_one(output_dir, row, config, plan, preflight) for row in rows]
        report = pd.DataFrame(records).sort_values(['gap', 'target_index'])
        report.to_csv(report_path, index=False)
        write_json(summary_path, {'configuration_id': config['configuration_id'], 'status': 'passed', 'scope': 'canary_only_not_full_validation', 'validated_outputs': int(len(report)), 'targets': [{'gap': int(row['gap']), 'slice': int(row['target_slice'])} for row in rows], 'all_positive_convergence': True, 'validation_csv': report_path.name})
        print(f'Canary validation passed for {len(report)} corrected outputs: {report_path}')
        return
    validate_target_directory_set(output_dir, rows, str(config['dataset']['filename_prefix']))
    report_path = args.report.resolve() if args.report else output_dir / 'validation.csv'
    summary_path = report_path.with_name('validation_summary.json')
    if report_path.exists() or summary_path.exists():
        raise FileExistsError('validation outputs already exist; validate a fresh root or choose a fresh --report')
    records = [validate_one(output_dir, row, config, plan, preflight) for row in rows]
    report = pd.DataFrame(records).sort_values(['gap', 'target_index'])
    report.to_csv(report_path, index=False)
    stack_path = report_path.with_name('assembled_stack_manifest.csv')
    if stack_path.exists():
        raise FileExistsError(f'assembled stack manifest already exists: {stack_path}')
    stacks = stack_manifest(output_dir, plan, config, sources)
    stacks.to_csv(stack_path, index=False)
    write_json(summary_path, {'configuration_id': config['configuration_id'], 'status': 'passed', 'expected_outputs': int(config['validation']['expected_outputs']), 'validated_outputs': int(len(report)), 'positions_per_stack': int(config['validation']['expected_positions_per_stack']), 'stack_geometries_identical': True, 'assembled_stack_manifest': stack_path.name, 'all_positive_convergence': True, 'validation_csv': report_path.name})
    print(f'Validation passed for {len(report)} fresh Study 06 outputs: {report_path}')
if __name__ == '__main__':
    main()
