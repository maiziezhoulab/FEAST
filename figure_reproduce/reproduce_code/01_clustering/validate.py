"""Validate the complete fresh Study 01 workflow."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import anndata as ad
import pandas as pd
import yaml
from run_methods import METHOD_CONFIG_KEYS, METHOD_SCRIPTS, canonical_method_parameters, method_metadata_is_valid
from run_simulations import build_alterations
METHODS = ('GraphST', 'STAGATE_mclust', 'Leiden_unsupervised')

def _is_true(value: Any) -> bool:
    return value is True or str(value).casefold() == 'true'

def _is_false(value: Any) -> bool:
    return value is False or str(value).casefold() == 'false'

def _keys(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, ...]]:
    return [tuple((str(value) for value in values)) for values in frame[columns].itertuples(index=False, name=None)]

def _require_exact_keys(frame: pd.DataFrame, columns: list[str], expected: set[tuple[str, ...]], description: str) -> None:
    actual_list = _keys(frame, columns)
    actual = set(actual_list)
    if len(actual_list) != len(actual):
        raise RuntimeError(f'{description} contains duplicate keys')
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f'{description} key mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}')

def _require_row_contract(row: Any, expected: dict[str, Any], description: str) -> None:
    for field, value in expected.items():
        actual = getattr(row, field, None)
        if actual != value:
            raise RuntimeError(f'{description} {field} mismatch: {actual!r} != {value!r}')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--simulation-dir', type=Path, required=True)
    parser.add_argument('--panel-dir', type=Path, required=True)
    parser.add_argument('--method-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    feast = config['feast']
    feast_version = str(feast['version'])
    if feast['spatial_mode'] != 'reference_rank' or feast['assignment_solver'] != 'scipy' or feast['assignment_blocks'] is not False:
        raise RuntimeError('invalid FEAST assignment contract')
    slice_ids = {str(value) for value in config['slices']}
    alterations = build_alterations(config)
    simulation_ids = {str(row['simulation_id']) for row in alterations}
    if len(simulation_ids) != 27:
        raise RuntimeError('configuration must define 27 unique simulation IDs')
    expected_simulation_keys = {(slice_id, simulation_id) for slice_id in slice_ids for simulation_id in simulation_ids}
    expected_panel_keys = expected_simulation_keys | {(slice_id, 'real_baseline') for slice_id in slice_ids}
    input_manifest = pd.read_csv(args.raw_dir / 'input_files.csv', dtype={'slice_id': str})
    _require_exact_keys(input_manifest, ['slice_id'], {(slice_id,) for slice_id in slice_ids}, 'source-input manifest')
    for record in input_manifest.itertuples(index=False):
        slice_id = record.slice_id
        source = args.raw_dir / record.relative_path
        data = ad.read_h5ad(source, backed='r')
        try:
            if 'ground_truth' not in data.obs or 'spatial' not in data.obsm:
                raise RuntimeError(f'raw slice lacks labels or spatial coordinates: {slice_id}')
        finally:
            data.file.close()
    simulations = pd.read_csv(args.simulation_dir / 'simulation_manifest.csv', dtype={'slice_id': str, 'simulation_id': str})
    _require_exact_keys(simulations, ['slice_id', 'simulation_id'], expected_simulation_keys, 'simulation manifest')
    if not simulations.status.eq('ok').all():
        raise RuntimeError('all 81 simulation rows must be successful')
    for row in simulations.itertuples(index=False):
        description = f'simulation {row.slice_id}/{row.simulation_id}'
        _require_row_contract(row, {'configuration_id': config['configuration_id'], 'public_seed': config['public_seed'], 'feast_version': feast_version, 'spatial_mode': 'reference_rank', 'assignment_solver': 'scipy', 'solver_status': 'completed'}, description)
        if not _is_false(row.assignment_blocks):
            raise RuntimeError(f'{description} used blocked assignment')
        path = args.simulation_dir / row.file_path
        data = ad.read_h5ad(path, backed='r')
        try:
            provenance = data.uns.get('publication_provenance', {})
            embedded_expected = {'configuration_id': config['configuration_id'], 'feast_version': feast_version, 'public_seed': config['public_seed'], 'spatial_mode': 'reference_rank', 'assignment_solver': 'scipy'}
            for field, value in embedded_expected.items():
                if provenance.get(field) != value:
                    raise RuntimeError(f'simulation provenance {field} mismatch: {path}')
            if not _is_false(provenance.get('assignment_blocks')):
                raise RuntimeError(f'simulation provenance is blocked: {path}')
            diagnostics = provenance.get('solver_diagnostics', {})
            if diagnostics.get('status') != 'completed' or not _is_false(diagnostics.get('assignment_blocks')):
                raise RuntimeError(f'simulation solver diagnostics mismatch: {path}')
        finally:
            data.file.close()
    simulation_provenance_path = args.simulation_dir / 'provenance.json'
    simulation_provenance = json.loads(simulation_provenance_path.read_text())
    simulation_manifest_path = args.simulation_dir / 'simulation_manifest.csv'
    simulation_provenance_expected = {'configuration_id': config['configuration_id'], 'public_seed': config['public_seed'], 'feast_version': feast_version, 'job_count': 81, 'successful_jobs': 81}
    for field, value in simulation_provenance_expected.items():
        if simulation_provenance.get(field) != value:
            raise RuntimeError(f'simulation provenance {field} mismatch')
    panels = pd.read_csv(args.panel_dir / 'fixed_panel_manifest.csv', dtype={'slice_id': str, 'simulation_id': str})
    _require_exact_keys(panels, ['slice_id', 'simulation_id'], expected_panel_keys, 'fixed-panel manifest')
    if not panels.status.eq('ok').all():
        raise RuntimeError('all 84 fixed-panel rows must be successful')
    for row in panels.itertuples(index=False):
        path = args.panel_dir / row.file_path
        data = ad.read_h5ad(path, backed='r')
        try:
            panel = pd.read_csv(args.panel_dir / 'panels' / f'{row.slice_id}.csv').gene_id.astype(str)
            if list(data.var_names.astype(str)) != panel.tolist() or 'log1p' not in data.uns:
                raise RuntimeError(f'fixed-panel contract mismatch: {path}')
            provenance = data.uns.get('publication_reproduction', {})
            if provenance.get('configuration_id') != config['configuration_id'] or provenance.get('public_seed') != config['public_seed'] or provenance.get('feast_version') != feast_version:
                raise RuntimeError(f'fixed-panel provenance mismatch: {path}')
        finally:
            data.file.close()
    for method in METHODS:
        manifest_path = args.method_dir / method / 'run_manifest.csv'
        manifest = pd.read_csv(manifest_path, dtype={'slice_id': str, 'simulation_id': str}, keep_default_na=False)
        _require_exact_keys(manifest, ['slice_id', 'simulation_id'], expected_panel_keys, f'{method} run manifest')
        if not manifest.status.eq('ok').all():
            raise RuntimeError(f'all 84 {method} rows must be successful')
        settings = config['methods'][METHOD_CONFIG_KEYS[method]]
        method_parameters_json = canonical_method_parameters(config, method)
        ld_library_path = ''
        if method == 'STAGATE_mclust':
            unique_library_paths = set(manifest.ld_library_path)
            if len(unique_library_paths) != 1:
                raise RuntimeError('STAGATE used more than one LD_LIBRARY_PATH contract')
            ld_library_path = next(iter(unique_library_paths))
        for row in manifest.itertuples(index=False):
            description = f'{method} {row.slice_id}/{row.simulation_id}'
            _require_row_contract(row, {'configuration_id': config['configuration_id'], 'method': method, 'method_parameters_json': method_parameters_json, 'public_seed': config['public_seed'], 'feast_version': feast_version, 'method_status': 'validated_success', 'returncode': 0}, description)
            invocation_path = Path(row.method_python)
            resolved_path = Path(row.method_python_resolved)
            if not invocation_path.is_file() or invocation_path.resolve() != resolved_path:
                raise RuntimeError(f'{description} interpreter identity mismatch')
            if method in {'GraphST', 'STAGATE_mclust'} and (not _is_true(row.cuda_execution_verified)):
                raise RuntimeError(f'{description} lacks verified CUDA execution')
            output = args.method_dir / method / row.slice_id / row.simulation_id
            metadata_valid, metadata_error = method_metadata_is_valid(output / 'metadata.json', method, settings, int(config['public_seed']), ld_library_path)
            if not metadata_valid:
                raise RuntimeError(f'{description}: {metadata_error}')
            metadata = json.loads((output / 'metadata.json').read_text())
            metadata_executable = Path(metadata['environment']['executable'])
            if metadata_executable.absolute() != invocation_path.absolute() or metadata_executable.resolve() != resolved_path:
                raise RuntimeError(f'{description} metadata interpreter identity mismatch')
        provenance_path = args.method_dir / method / 'provenance.json'
        provenance = json.loads(provenance_path.read_text())
        unique_invocations = set(manifest.method_python)
        unique_resolved = set(manifest.method_python_resolved)
        if not len(unique_invocations) == len(unique_resolved) == 1:
            raise RuntimeError(f'{method} used more than one interpreter identity')
        provenance_expected = {'configuration_id': config['configuration_id'], 'method': method, 'method_parameters_json': method_parameters_json, 'public_seed': config['public_seed'], 'method_python': next(iter(unique_invocations)), 'method_python_resolved': next(iter(unique_resolved)), 'sequential_execution': True, 'job_count': 84, 'successful_jobs': 84}
        if method == 'STAGATE_mclust':
            provenance_expected.update({'ld_library_path': ld_library_path})
        for field, value in provenance_expected.items():
            if provenance.get(field) != value:
                raise RuntimeError(f'{method} provenance {field} mismatch')
    metrics = pd.read_csv(args.report_dir / 'fixed_panel_benchmark_metrics.csv', dtype={'slice_id': str, 'simulation_id': str})
    expected_metric_keys = {(slice_id, simulation_id, method) for slice_id, simulation_id in expected_panel_keys for method in METHODS}
    _require_exact_keys(metrics, ['slice_id', 'simulation_id', 'method'], expected_metric_keys, 'benchmark metric table')
    if not (metrics.feast_version.eq(feast_version).all() and metrics.public_seed.eq(config['public_seed']).all()):
        raise RuntimeError('metric table FEAST version or seed mismatch')
    if any(('composite' in column.casefold() for column in metrics)):
        raise RuntimeError('composite metrics are prohibited')
    print('Study 01 validation: OK (81 simulations, 84 panels, 252 methods)')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
