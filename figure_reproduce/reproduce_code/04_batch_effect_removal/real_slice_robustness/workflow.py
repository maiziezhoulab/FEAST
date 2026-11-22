"""Shared contracts for Study 04B real-slice robustness."""
from __future__ import annotations
import json
import subprocess
from datetime import datetime
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
STUDY_ROOT = Path(__file__).resolve().parent
METHOD_ARTIFACTS = {'GraphST': ('metadata.json', 'result.h5ad', 'embeddings.npz', 'graphst_checkpoint.pt'), 'STAMP': ('metadata.json', 'result.h5ad', 'embeddings.npz', 'cell_by_topic.csv', 'stamp_params.pt'), 'scVI': ('metadata.json', 'result.h5ad', 'embeddings.npz', 'model/model.pt')}

def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()

def load_config(path: Path=STUDY_ROOT / 'config.yaml') -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    required = {'configuration_id', 'paths', 'feast', 'inputs', 'deformation', 'conditions', 'fixed_panel', 'analysis', 'methods'}
    missing = required - set(config)
    if missing:
        raise ValueError(f'configuration is missing fields: {sorted(missing)}')
    expected_conditions = {'raw': None, 'sim_0.00': 0.0, 'sim_0.50': 0.5, 'sim_1.00': 1.0, 'sim_1.50': 1.5}
    observed_conditions = {str(row['name']): None if row.get('alpha') is None else float(row['alpha']) for row in config['conditions']}
    if observed_conditions != expected_conditions:
        raise ValueError('the frozen five-condition Study 04B matrix changed')
    if list(map(int, config['analysis']['method_seeds'])) != [42, 43, 44]:
        raise ValueError('the frozen method seeds must remain 42, 43, and 44')
    if set(config['methods']) != set(METHOD_ARTIFACTS):
        raise ValueError('exactly GraphST, STAMP, and scVI must be configured')
    if config['analysis'].get('composite_score') != 'prohibited':
        raise ValueError('Study 04B composite scores must remain prohibited')
    if config['analysis'].get('winner_ranking') != 'prohibited':
        raise ValueError('Study 04B winner rankings must remain prohibited')
    config['_config_path'] = path
    config['_input_manifest'] = resolve_path(path, config['paths']['input_manifest'])
    config['_output_dir'] = resolve_path(path, config['paths']['output_dir'])
    config['_feast_python'] = resolve_path(path, config['paths']['feast_python'])
    config['_feast_source_repo'] = resolve_path(path, config['paths']['feast_source_repo'])
    fit_path = resolve_path(path, config['deformation']['fit_provenance'])
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    if fit['feast']['version'] != str(config['feast']['version']):
        raise ValueError(f'04A fit has a different FEAST version: {fit_path}')
    config['deformation'].update(fit['deformation_diagnostics']['diagonal_affine'])
    return config

def condition_rows(config: dict) -> list[dict]:
    return [dict(row) for row in config['conditions']]

def condition_lookup(config: dict) -> dict[str, dict]:
    return {str(row['name']): dict(row) for row in config['conditions']}

def production_cells(config: dict) -> list[tuple[str, str, int]]:
    return [(str(condition['name']), method, int(seed)) for condition in config['conditions'] for method in config['methods'] for seed in config['analysis']['method_seeds']]

def candidate_id(config: dict, condition: str, method: str, seed: int) -> str:
    return f"{config['configuration_id']}__{method}__{condition}__seed_{int(seed)}"

def candidate_path(output_root: Path, condition: str, method: str, seed: int) -> Path:
    return Path(output_root) / 'methods' / method / condition / f'seed_{int(seed)}'

def validate_count_matrix(matrix, label: str) -> dict[str, object]:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    finite = bool(np.isfinite(values).all())
    nonnegative = bool(np.all(values >= 0))
    integral = bool(np.equal(values, np.floor(values)).all())
    if not finite or not nonnegative or (not integral):
        raise ValueError(f'{label} is not a finite nonnegative integral matrix')
    return {'label': label, 'finite': finite, 'nonnegative': nonnegative, 'integral': integral, 'n_spots': int(matrix.shape[0]), 'n_genes': int(matrix.shape[1])}

def prefix_spot_ids(adata: ad.AnnData, slice_id: str) -> ad.AnnData:
    result = adata.copy()
    barcodes = result.obs_names.astype(str).tolist()
    result.obs['source_barcode'] = barcodes
    result.obs['source_section_id'] = str(slice_id)
    result.obs_names = pd.Index([f'{slice_id}::{barcode}' for barcode in barcodes])
    if not result.obs_names.is_unique:
        raise ValueError(f'prefixed observation IDs are not unique for {slice_id}')
    return result

def read_condition_manifest(output_root: Path) -> pd.DataFrame:
    path = Path(output_root) / 'inputs' / 'condition_manifest.csv'
    if not path.is_file():
        raise FileNotFoundError(f'condition manifest is missing: {path}')
    table = pd.read_csv(path)
    required = {'condition', 'alpha', 'role', 'reference_path', 'query_path', 'n_reference', 'n_query'}
    if required - set(table):
        raise ValueError('condition manifest schema changed')
    for column in ('reference_path', 'query_path'):
        table[column] = table[column].map(lambda value: str((output_root / value).resolve()))
    return table

def read_panel(output_root: Path) -> list[str]:
    path = Path(output_root) / 'panel' / 'panel.csv'
    table = pd.read_csv(path, dtype={'gene': str})
    if list(table.columns[:2]) != ['panel_order', 'gene']:
        raise ValueError('panel must begin with panel_order,gene')
    if table['panel_order'].astype(int).tolist() != list(range(len(table))):
        raise ValueError('panel order is not contiguous')
    genes = table['gene'].astype(str).tolist()
    if len(set(genes)) != len(genes):
        raise ValueError('panel contains duplicate genes')
    return genes

def _expected_condition_inputs(condition_manifest: pd.DataFrame, condition: str) -> tuple[Path, Path, int, int]:
    row = condition_manifest[condition_manifest['condition'] == condition]
    if len(row) != 1:
        raise ValueError(f'missing unique condition manifest row: {condition}')
    item = row.iloc[0]
    return (Path(item.reference_path), Path(item.query_path), int(item.n_reference), int(item.n_query))

def candidate_is_valid(candidate: Path, *, config: dict, output_root: Path, condition: str, method: str, seed: int) -> tuple[bool, str]:
    candidate = Path(candidate)
    for relative in METHOD_ARTIFACTS[method]:
        if not (candidate / relative).is_file():
            return (False, f'missing {relative}')
    try:
        metadata = json.loads((candidate / 'metadata.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return (False, f'invalid metadata: {error}')
    provenance = metadata.get('provenance', {})
    context = provenance.get('candidate_context', {})
    expected_id = candidate_id(config, condition, method, seed)
    condition_row = condition_lookup(config)[condition]
    expected_alpha = 0.0 if condition_row.get('alpha') is None else float(condition_row['alpha'])
    if provenance.get('configuration_id') != expected_id:
        return (False, 'candidate configuration ID mismatch')
    if context.get('parent_configuration_id') != config['configuration_id']:
        return (False, 'parent configuration ID mismatch')
    if context.get('method') != method:
        return (False, 'method context mismatch')
    if context.get('deformation_mode') != condition:
        return (False, 'condition context mismatch')
    if float(context.get('alpha', float('nan'))) != expected_alpha:
        return (False, 'alpha context mismatch')
    if int(provenance.get('public_seed', -1)) != int(seed):
        return (False, 'method seed mismatch')
    if provenance.get('feast', {}).get('version') != str(config['feast']['version']):
        return (False, 'candidate upstream FEAST version mismatch')
    diagnostics = provenance.get('solver_diagnostics', {})
    if diagnostics.get('status') != 'completed':
        return (False, 'method diagnostics are incomplete')
    if diagnostics.get('cuda_execution_verified') is not True:
        return (False, 'candidate lacks verified CUDA execution')
    if not str(diagnostics.get('actual_device', '')).startswith('cuda'):
        return (False, 'candidate device is not CUDA')
    if method in {'GraphST', 'STAMP'} and diagnostics.get('cross_batch_edge_count') != 0:
        return (False, 'cross-section spatial edge found')
    if metadata.get('runtime_controls', {}).get('pip_check_status') != 'ok':
        return (False, 'external environment pip check is not recorded')
    for collection in ('inputs', 'outputs', 'sources'):
        records = provenance.get(collection, [])
        if not records:
            return (False, f'empty provenance collection: {collection}')
        for record in records:
            path = Path(str(record.get('path', '')))
            if not path.is_file():
                return (False, f'provenance path is missing: {path}')
    condition_manifest = read_condition_manifest(output_root)
    ref_path, query_path, n_ref, n_query = _expected_condition_inputs(condition_manifest, condition)
    expected_input_paths = {ref_path.resolve(), query_path.resolve(), (Path(output_root) / 'panel' / 'panel.csv').resolve(), (Path(output_root) / 'panel' / 'provenance.json').resolve()}
    observed_input_paths = {Path(str(row.get('path', ''))).resolve() for row in provenance.get('inputs', [])}
    if observed_input_paths != expected_input_paths:
        return (False, 'candidate input artifact set mismatch')
    try:
        with np.load(candidate / 'embeddings.npz', allow_pickle=True) as payload:
            embedding = np.asarray(payload['embedding'])
            batch_labels = np.asarray(payload['batch_labels']).astype(str)
    except Exception as error:
        return (False, f'invalid embedding artifact: {error}')
    if embedding.shape[0] != n_ref + n_query or embedding.ndim != 2:
        return (False, 'embedding support mismatch')
    if not np.isfinite(embedding).all():
        return (False, 'embedding contains non-finite values')
    expected_batches = np.array(['ref'] * n_ref + ['query'] * n_query)
    if not np.array_equal(batch_labels, expected_batches):
        return (False, 'embedding batch order mismatch')
    try:
        ref = ad.read_h5ad(ref_path, backed='r')
        query = ad.read_h5ad(query_path, backed='r')
        result = ad.read_h5ad(candidate / 'result.h5ad', backed='r')
        try:
            expected_ids = np.concatenate([ref.obs_names.astype(str).to_numpy(), query.obs_names.astype(str).to_numpy()])
            observed_ids = result.obs['source_spot_id'].astype(str).to_numpy()
            observed_batches = result.obs['batch'].astype(str).to_numpy()
            if not np.array_equal(observed_ids, expected_ids):
                return (False, 'result source spot identity/order mismatch')
            if not np.array_equal(observed_batches, expected_batches):
                return (False, 'result batch identity/order mismatch')
            if result.var_names.astype(str).tolist() != read_panel(output_root):
                return (False, 'result fixed-panel gene order mismatch')
            if method == 'scVI' and 'scvi_normalized' not in result.layers:
                return (False, 'scVI result lacks normalized expression')
        finally:
            ref.file.close()
            query.file.close()
            result.file.close()
    except Exception as error:
        return (False, f'invalid result H5AD: {error}')
    graph = metadata.get('spatial_graph', {})
    if method in {'GraphST', 'STAMP'}:
        if graph.get('batch_sizes') != {'query': n_query, 'ref': n_ref}:
            return (False, 'spatial graph batch sizes mismatch')
        if graph.get('cross_batch_edge_count') != 0:
            return (False, 'spatial graph contains cross-section edges')
    return (True, 'validated completed candidate')

def archive_invalid(candidate: Path, failure_root: Path, reason: str) -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = Path(failure_root) / f'{candidate.parent.parent.name}__{candidate.parent.name}__{candidate.name}__{stamp}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists():
        candidate.replace(destination)
    else:
        destination.mkdir()
    (destination / 'DISPOSITION.txt').write_text(reason + '\n', encoding='utf-8')
    return destination

def format_cli_value(value: object) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)
