"""Validate the additive Study 01 extension and export realized alterations."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import anndata as ad
import pandas as pd
import yaml
from run_methods import METHOD_CONFIG_KEYS, METHOD_SCRIPTS, method_metadata_is_valid
EXPECTED_CONDITIONS = {('mean_0_20', 'mean', 0.2), ('mean_5', 'mean', 5.0), ('variance_0_20', 'variance', 0.2), ('variance_5', 'variance', 5.0)}

def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    conditions = {(str(row['simulation_id']), str(row['alteration_type']), float(row['value'])) for row in config['conditions']}
    if conditions != EXPECTED_CONDITIONS:
        raise RuntimeError(f'unexpected extension conditions: {conditions}')
    if [str(value) for value in config['slices']] != ['151508', '151670', '151676']:
        raise RuntimeError('unexpected extension slices')
    return config

def require_jobs(table: pd.DataFrame, expected: set[tuple[str, str]], label: str) -> None:
    keys = list(table[['slice_id', 'simulation_id']].astype(str).itertuples(index=False, name=None))
    if len(keys) != len(expected) or len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError(f'{label} does not contain the 12 unique extension jobs')
    if not table['status'].eq('ok').all():
        raise RuntimeError(f'{label} contains unsuccessful jobs')

def require_path(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f'{label} is missing: {path}')

def diagnostics_row(row: Any, simulated: ad.AnnData) -> dict[str, Any]:
    diagnostics = simulated.uns.get('alteration_diagnostics', {})
    requested = diagnostics.get('requested_config', {})
    target = diagnostics.get('target_stage_achieved_change', {})
    realized = diagnostics.get('realized_stage_achieved_change', {})
    for label, values in (('target', target), ('realized', realized)):
        missing = {'mean', 'variance', 'zero_prop'} - set(values)
        if missing:
            raise RuntimeError(f'{row.slice_id}/{row.simulation_id} lacks {label} diagnostics: {sorted(missing)}')
    intended = str(row.alteration_type)
    off_target = 'variance' if intended == 'mean' else 'mean'
    return {'slice_id': str(row.slice_id), 'simulation_id': str(row.simulation_id), 'alteration_type': intended, 'nominal_fc': float(row.alteration_value), 'requested_mean_fc': float(requested.get('mean_fold_change', 1.0)), 'requested_variance_fc': float(requested.get('variance_fold_change', 1.0)), 'requested_sparsity_logit_shift': float(requested.get('sparsity_logit_shift', 0.0)), 'target_mean_fc': float(target['mean']), 'realized_mean_fc': float(realized['mean']), 'target_variance_fc': float(target['variance']), 'realized_variance_fc': float(realized['variance']), 'target_zero_prop_fc': float(target['zero_prop']), 'realized_zero_prop_fc': float(realized['zero_prop']), 'intended_metric_realized_fc': float(realized[intended]), 'off_target_metric': off_target, 'off_target_metric_realized_fc': float(realized[off_target])}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('expanded_config.yaml'))
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    feast_version = str(config['feast']['version'])
    seed = int(config['public_seed'])
    expected = {(str(slice_id), str(condition['simulation_id'])) for slice_id in config['slices'] for condition in config['conditions']}
    simulation_manifest_path = args.output_root / 'simulations' / 'simulation_manifest.csv'
    simulations = pd.read_csv(simulation_manifest_path, dtype={'slice_id': str, 'simulation_id': str})
    require_jobs(simulations, expected, 'simulation manifest')
    diagnostic_rows: list[dict[str, Any]] = []
    for row in simulations.itertuples(index=False):
        path = args.output_root / 'simulations' / row.file_path
        require_path(path, 'simulation')
        data = ad.read_h5ad(path, backed='r')
        try:
            provenance = data.uns.get('expanded_sensitivity_provenance', {})
            if (
                provenance.get('configuration_id') != config['configuration_id']
                or provenance.get('feast_version') != feast_version or row.feast_version != feast_version
                or int(provenance.get('public_seed', -1)) != seed or int(row.public_seed) != seed
            ):
                raise RuntimeError(f'simulation provenance mismatch: {path}')
            diagnostic_rows.append(diagnostics_row(row, data))
        finally:
            data.file.close()
    panel_manifest_path = args.output_root / 'fixed_panels' / 'fixed_panel_manifest.csv'
    panels = pd.read_csv(panel_manifest_path, dtype={'slice_id': str, 'simulation_id': str})
    require_jobs(panels, expected, 'fixed-panel manifest')
    panel_obs: dict[tuple[str, str], pd.Index] = {}
    for row in panels.itertuples(index=False):
        key = (row.slice_id, row.simulation_id)
        if int(row.n_genes) != 3000 or row.feast_version != feast_version:
            raise RuntimeError(f'fixed-panel contract mismatch: {key}')
        path = args.output_root / 'fixed_panels' / row.file_path
        require_path(path, 'fixed panel')
        data = ad.read_h5ad(path, backed='r')
        try:
            panel_obs[key] = data.obs_names.copy()
        finally:
            data.file.close()
    for method in METHOD_SCRIPTS:
        manifest_path = args.output_root / 'methods' / method / 'run_manifest.csv'
        runs = pd.read_csv(manifest_path, dtype={'slice_id': str, 'simulation_id': str}, keep_default_na=False)
        require_jobs(runs, expected, f'{method} run manifest')
        if not runs.feast_version.eq(feast_version).all() or not runs.public_seed.eq(seed).all():
            raise RuntimeError(f'{method} has an unexpected upstream FEAST version or seed')
        method_provenance = json.loads((args.output_root / 'methods' / method / 'provenance.json').read_text())
        inherited_ld = method_provenance.get('ld_library_path', '')
        for row in runs.itertuples(index=False):
            key = (row.slice_id, row.simulation_id)
            output = args.output_root / 'methods' / method / row.slice_id / row.simulation_id
            for path in (output / 'metadata.json', output / 'clusters.csv', output / 'result.h5ad'):
                require_path(path, f'{method} output')
            valid, error = method_metadata_is_valid(output / 'metadata.json', method, config['methods'][METHOD_CONFIG_KEYS[method]], int(config['public_seed']), inherited_ld)
            if not valid:
                raise RuntimeError(f'{method} metadata invalid for {key}: {error}')
            clusters = pd.read_csv(output / 'clusters.csv', dtype={'spot_barcode': str})
            if not pd.Index(clusters['spot_barcode']).equals(panel_obs[key]):
                raise RuntimeError(f'{method} cluster spot order mismatch: {key}')
            result = ad.read_h5ad(output / 'result.h5ad', backed='r')
            try:
                if not result.obs_names.equals(panel_obs[key]):
                    raise RuntimeError(f'{method} result spot order mismatch: {key}')
            finally:
                result.file.close()
    report = args.output_root / 'report'
    expanded_metrics = report / 'expanded_benchmark_metrics.csv'
    metrics = pd.read_csv(expanded_metrics, dtype={'slice_id': str, 'simulation_id': str})
    metric_keys = list(metrics[['slice_id', 'simulation_id', 'method']].astype(str).itertuples(index=False, name=None))
    expected_metric_keys = {(a, b, method) for a, b in expected for method in METHOD_SCRIPTS}
    if set(metric_keys) != expected_metric_keys or len(metric_keys) != 36 or (not metrics.status.eq('ok').all()):
        raise RuntimeError('expanded benchmark metrics do not contain 36 unique successful rows')
    if not metrics.feast_version.eq(feast_version).all() or not metrics.public_seed.eq(seed).all():
        raise RuntimeError('extended scores have an unexpected FEAST version or seed')
    diagnostics_path = report / 'realized_alteration_diagnostics.csv'
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(['alteration_type', 'nominal_fc', 'slice_id'])
    diagnostics.to_csv(diagnostics_path, index=False)
    validation_path = report / 'validation.json'
    validation = {'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'validator_source': str(Path(__file__).resolve()), 'feast_version': feast_version, 'public_seed': seed, 'simulation_jobs': len(simulations), 'fixed_panel_jobs': len(panels), 'method_jobs': {method: 12 for method in METHOD_SCRIPTS}, 'metric_rows': len(metrics), 'all_status_ok': True}
    validation_path.write_text(json.dumps(validation, indent=2) + '\n')
    print(f"validated {len(simulations)} simulations, {len(panels)} panels, {sum(validation['method_jobs'].values())} method jobs, and {len(metrics)} metric rows")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
