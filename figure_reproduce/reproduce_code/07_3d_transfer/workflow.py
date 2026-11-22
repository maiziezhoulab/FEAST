"""Shared, non-scientific contracts for the clean Study 07 workflow."""
from __future__ import annotations
import gzip
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import yaml
STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_male_feast106.yaml"
AGE_ORDER = ('E15.5', 'E18.5')
EXPECTED_REGIONS = {'BT', 'CP', 'DorsalVZ', 'GE', 'IZ', 'SeptalVZ', 'Septum', 'VentralVZ'}
NIFTI_DTYPES = {2: np.dtype('u1'), 4: np.dtype('i2'), 8: np.dtype('i4'), 16: np.dtype('f4'), 64: np.dtype('f8'), 256: np.dtype('i1'), 512: np.dtype('u2'), 768: np.dtype('u4')}

def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()

def load_config(path: Path | str=DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    required = {'configuration_id', 'blueprint_configuration_id', 'legacy_study_id', 'public_seed', 'paths', 'reference_fit', 'transport', 'ages', 'blueprint'}
    missing = required - set(config)
    if missing:
        raise ValueError(f'configuration is missing fields: {sorted(missing)}')
    if tuple(config['ages']) != AGE_ORDER:
        raise ValueError(f'ages must be declared in canonical order {AGE_ORDER}')
    fixed = {'E15.5': (-4.12, -0.98, 0.02, 158, 2330927), 'E18.5': (-5.56, -1.54, 0.02, 202, 5213461)}
    for age, contract in fixed.items():
        item = config['ages'][age]
        observed = (float(item['z_start']), float(item['z_end']), float(item['z_step']), int(item['expected_levels']), int(item['expected_spots']))
        if observed != contract:
            raise ValueError(f'{age} full-axis contract changed: {observed} != {contract}')
    solver = config['transport']
    if not (int(solver['sinkhorn_iter']) == 1000 and float(solver['sinkhorn_tol']) == 1e-05 and (solver['transport_nonconvergence'] == 'raise') and (int(solver['max_transport_pairs']) == 25000000) and (solver['sinkhorn_method'] == 'sinkhorn_log') and (solver['transport_backend'] == 'torch') and (solver['transport_device'] == 'cuda:0') and (solver['transport_dtype'] == 'float32')):
        raise ValueError('strict publication OT contract changed')
    for age in AGE_ORDER:
        ar = config['ages'][age]['assignment_randomness']
        if ar.get('mode') == 'approved_estimate':
            if not np.isfinite(float(ar['value'])) or not 0 <= float(ar['value']) <= 0.5:
                raise ValueError('estimated AR must be finite and within 0–0.5')
            if not ar.get('basis'):
                raise ValueError('estimated AR requires its evidence and approval basis')
        elif ar.get('mode') != 'reference_calibration':
            raise ValueError('unknown assignment-randomness mode')
    fit = config['reference_fit']
    fit_contract = (int(fit['expected_genes']), int(fit['min_gene_spots']), float(fit['min_gene_mean']), float(fit['max_gene_zero_prop']))
    if fit_contract != (550, 1, 0.0, 1.0):
        raise ValueError('Study 07 must preserve the historical E15.5 550-gene reference-fit contract')
    config['_config_path'] = config_path
    config['_feast_repo'] = resolve_path(config_path, config['paths']['feast_repo'])
    config['_dataset_root'] = resolve_path(config_path, config['paths']['dataset_root'])
    config['_reference_dir'] = config['_dataset_root'] / config['paths']['reference_dir']
    config['_devccf_dir'] = config['_dataset_root'] / config['paths']['devccf_dir']
    config['_blueprint_dir'] = resolve_path(config_path, config['paths']['blueprint_dir'])
    config['_work_dir'] = resolve_path(config_path, config['paths']['work_dir'])
    config['_final_dir'] = resolve_path(config_path, config['paths']['final_dir'])
    config['_input_manifest'] = resolve_path(config_path, config['paths']['input_manifest']) if 'input_manifest' in config['paths'] else STUDY_ROOT / 'data' / 'input_checksums.csv'
    return config

def expected_z_values(config: dict[str, Any], age: str) -> list[float]:
    item = config['ages'][age]
    return [round(float(item['z_start']) + i * float(item['z_step']), 2) for i in range(int(item['expected_levels']))]

def blueprint_path(config: dict[str, Any], age: str) -> Path:
    return config['_blueprint_dir'] / f'{age}.blueprint.json.gz'

def blueprint_manifest_path(config: dict[str, Any], age: str) -> Path:
    return config['_blueprint_dir'] / f'{age}.manifest.csv'

def blueprint_provenance_path(config: dict[str, Any], age: str) -> Path:
    return config['_blueprint_dir'] / f'{age}.blueprint.provenance.json'

def calibration_path(config: dict[str, Any], age: str='E18.5') -> Path:
    value = config['ages'][age]['assignment_randomness']['calibration_file']
    return resolve_path(config['_config_path'], value)

def calibration_manifest_path(config: dict[str, Any], age: str='E18.5') -> Path:
    path = calibration_path(config, age)
    return path.with_name(f'{path.stem}.manifest.json')

def assignment_randomness_provenance(config: dict[str, Any], age: str) -> dict:
    ar = config['ages'][age]['assignment_randomness']
    if ar['mode'] == 'approved_estimate':
        return {'mode': 'approved_estimate', 'value': float(ar['value']), 'basis': str(ar['basis'])}
    return {'mode': 'reference_calibration',
            'path': str(calibration_path(config, age).relative_to(STUDY_ROOT)),
            'manifest_path': str(calibration_manifest_path(config, age).relative_to(STUDY_ROOT))}

def input_manifest(config: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(config['_input_manifest'], dtype=str)

def input_paths(config: dict[str, Any], age: str | None=None) -> list[tuple[pd.Series, Path]]:
    records = input_manifest(config)
    if age is not None:
        records = records[records['age'].isin([age, 'both'])]
    return [(row, (config['_dataset_root'] / row['relative_path']).resolve()) for _, row in records.iterrows()]

def reference_artifact_contract(config: dict[str, Any], age: str) -> dict[str, dict[str, str]]:
    return {str(row['artifact_id']): {'filename': path.name} for row, path in input_paths(config, age) if row['role'] == 'reference'}

def reference_paths(config: dict[str, Any], age: str) -> list[Path]:
    paths = sorted(config['_reference_dir'].glob(config['ages'][age]['reference_glob']))
    expected = int(config['ages'][age]['expected_references'])
    if len(paths) != expected:
        raise ValueError(f'{age}: expected {expected} references, found {len(paths)}')
    declared = {path for row, path in input_paths(config, age) if row['role'] == 'reference'}
    if set(paths) != declared:
        raise ValueError(f'{age}: reference glob and input manifest disagree')
    return paths

def preflight_inputs(config: dict[str, Any], *, inspect_h5ad: bool=True) -> dict[str, Any]:
    for row, path in input_paths(config):
        if not path.is_file():
            raise FileNotFoundError(f"missing input {row['artifact_id']}: {path}")
    summary: dict[str, Any] = {'inputs': int(len(input_manifest(config))), 'input_manifest': str(config['_input_manifest']), 'ages': {}}
    if not inspect_h5ad:
        return summary
    import anndata as ad
    global_genes: list[str] | None = None
    for age in AGE_ORDER:
        rows = []
        for path in reference_paths(config, age):
            data = ad.read_h5ad(path, backed='r')
            try:
                genes = list(map(str, data.var_names))
                if data.n_vars != int(config['reference_fit']['expected_genes']):
                    raise ValueError(f'{path.name}: expected 550 genes, found {data.n_vars}')
                if not data.obs_names.is_unique or not data.var_names.is_unique:
                    raise ValueError(f'{path.name}: duplicate spot or gene identifiers')
                label_key = config['reference_fit']['label_key']
                if label_key not in data.obs or 'spatial' not in data.obsm:
                    raise ValueError(f'{path.name}: missing region labels or spatial coordinates')
                if global_genes is None:
                    global_genes = genes
                elif genes != global_genes:
                    raise ValueError(f'{path.name}: gene identity/order differs')
                rows.append({'file': path.name, 'n_spots': int(data.n_obs), 'n_genes': int(data.n_vars)})
            finally:
                data.file.close()
        summary['ages'][age] = rows
    return summary

def feast_identity(config: dict[str, Any], *, require_cuda: bool=True) -> dict[str, str]:
    """Record runtime details without source or build-identity verification."""
    import FEAST
    import numpy
    import ot
    import torch
    if sys.version_info[:2] != (3, 11) or str(numpy.__version__) != '1.26.4':
        raise RuntimeError(f'unsupported execution environment: Python {sys.version.split()[0]}, NumPy {numpy.__version__}')
    if str(ot.__version__) != '0.9.7':
        raise RuntimeError(f'unsupported POT version: {ot.__version__}')
    if config.get('feast_version') and str(FEAST.__version__) != str(config['feast_version']):
        raise RuntimeError(f"expected FEAST {config['feast_version']}, found {FEAST.__version__}")
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError('declared CUDA transport device is unavailable')
    branch = subprocess.check_output(
        ['git', '-C', str(config['_feast_repo']), 'branch', '--show-current'], text=True).strip()
    subject = subprocess.check_output(
        ['git', '-C', str(config['_feast_repo']), 'log', '-1', '--format=%s'], text=True).strip()
    commit = subprocess.check_output(
        ['git', '-C', str(config['_feast_repo']), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(
        ['git', '-C', str(config['_feast_repo']), 'status', '--short'], text=True).strip()
    return {'version': str(FEAST.__version__), 'import_path': str(Path(FEAST.__file__).resolve()),
            'git_branch': branch, 'git_subject': subject, 'git_commit': commit, 'git_status': status, 'python': sys.version.split()[0],
            'numpy': str(numpy.__version__), 'pot': str(ot.__version__),
            'torch': str(torch.__version__), 'torch_cuda': str(torch.version.cuda),
            'cuda_available': str(cuda_available).lower(),
            'gpu': str(torch.cuda.get_device_name(0)) if cuda_available else 'not_available'}


def read_nifti_gz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(path, 'rb') as handle:
        raw = handle.read()
    if len(raw) < 352:
        raise ValueError(f'truncated NIfTI input: {path}')
    if struct.unpack('<i', raw[:4])[0] == 348:
        endian = '<'
    elif struct.unpack('>i', raw[:4])[0] == 348:
        endian = '>'
    else:
        raise ValueError(f'not a NIfTI-1 file: {path}')
    dim = struct.unpack(endian + '8h', raw[40:56])
    shape = tuple((int(v) for v in dim[1:1 + int(dim[0])]))
    if len(shape) != 3:
        raise ValueError(f'expected a 3D NIfTI volume, found shape {shape}')
    datatype = int(struct.unpack(endian + 'h', raw[70:72])[0])
    if datatype not in NIFTI_DTYPES:
        raise ValueError(f'unsupported NIfTI datatype {datatype}')
    dtype = NIFTI_DTYPES[datatype].newbyteorder(endian)
    offset = int(round(struct.unpack(endian + 'f', raw[108:112])[0]))
    count = int(np.prod(shape))
    stop = offset + count * dtype.itemsize
    if stop > len(raw):
        raise ValueError(f'truncated NIfTI data: {path}')
    data = np.frombuffer(raw[offset:stop], dtype=dtype, count=count).reshape(shape, order='F')
    slope = float(struct.unpack(endian + 'f', raw[112:116])[0])
    intercept = float(struct.unpack(endian + 'f', raw[116:120])[0])
    if slope not in (0.0, 1.0) or intercept != 0.0:
        data = data.astype(np.float32) * (slope or 1.0) + intercept
    affine = np.eye(4, dtype=np.float64)
    if int(struct.unpack(endian + 'h', raw[254:256])[0]) > 0:
        affine[0] = struct.unpack(endian + '4f', raw[280:296])
        affine[1] = struct.unpack(endian + '4f', raw[296:312])
        affine[2] = struct.unpack(endian + '4f', raw[312:328])
    else:
        pixdim = struct.unpack(endian + '8f', raw[76:108])
        affine[0, 0], affine[1, 1], affine[2, 2] = (pixdim[1] or 1.0, pixdim[2] or 1.0, pixdim[3] or 1.0)
    return (data, affine)

def load_region_schema(path: Path) -> dict[int, dict[str, Any]]:
    frame = pd.read_csv(path, sep='\t')
    required = {'region_id', 'region_label', 'include_in_final_generation'}
    if not required.issubset(frame.columns):
        raise ValueError(f'region schema is missing {sorted(required - set(frame.columns))}')
    result = {}
    for row in frame.itertuples(index=False):
        include = str(row.include_in_final_generation).strip().upper() == 'TRUE'
        result[int(row.region_id)] = {'label': str(row.region_label), 'include': include}
    return result

def make_spot_ids(age: str, entry: dict[str, Any]) -> list[str]:
    age_token = age.lower().replace('.', '_')
    z_index = int(entry['z_index'])
    voxel_j = int(entry['voxel_j'])
    return [f'devccf-{age_token}-z{z_index:03d}-j{voxel_j:03d}-i{int(i):03d}-k{int(k):03d}' for i, k in zip(entry['voxel_i'], entry['voxel_k'])]

def extract_blueprint_payload(data: np.ndarray, affine: np.ndarray, schema: dict[int, dict[str, Any]], age: str) -> dict[str, Any]:
    if data.ndim != 3 or np.asarray(affine).shape != (4, 4):
        raise ValueError('blueprint extraction requires a 3D volume and 4x4 affine')
    unknown = set(map(int, np.unique(data))) - set(schema)
    if unknown:
        raise ValueError(f'volume contains region IDs absent from schema: {sorted(unknown)}')
    slices = []
    for voxel_j in range(data.shape[1]):
        plane = data[:, voxel_j, :]
        keep = np.zeros(plane.shape, dtype=bool)
        for region_id, record in schema.items():
            if region_id != 0 and record['include']:
                keep |= plane == region_id
        voxel_i, voxel_k = np.nonzero(keep)
        if not len(voxel_i):
            continue
        voxel = np.column_stack([voxel_i, np.full(len(voxel_i), voxel_j), voxel_k, np.ones(len(voxel_i))])
        world = voxel @ np.asarray(affine, dtype=np.float64).T
        regions = [schema[int(value)]['label'] for value in plane[voxel_i, voxel_k]]
        slices.append({'z_world': round(float(world[0, 2]), 2), 'voxel_j': int(voxel_j), 'voxel_i': voxel_i.astype(int).tolist(), 'voxel_k': voxel_k.astype(int).tolist(), 'x': world[:, 0].astype(float).tolist(), 'y': world[:, 1].astype(float).tolist(), 'region': regions, 'n_spots': int(len(voxel_i))})
    slices.sort(key=lambda entry: float(entry['z_world']))
    for z_index, entry in enumerate(slices):
        entry['z_index'] = z_index
    return {'metadata': {'schema_version': 1, 'age': age, 'affine': np.asarray(affine, dtype=float).tolist(), 'n_z_levels': len(slices), 'expression_source': 'none', 'spot_id_rule': 'voxel-identity-v1', 'coordinate_system': 'devccf_world', 'excluded_labels': sorted((record['label'] for record in schema.values() if not record['include']))}, 'slices': slices}

def summarize_volume_contract(data: np.ndarray, affine: np.ndarray, schema: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Low-memory regression summary using the extractor's keep/z rules."""
    z_values: list[float] = []
    spot_counts: list[int] = []
    included_ids = [region_id for region_id, record in schema.items() if region_id != 0 and record['include']]
    for voxel_j in range(data.shape[1]):
        plane = data[:, voxel_j, :]
        keep = np.isin(plane, included_ids)
        n_spots = int(keep.sum())
        if not n_spots:
            continue
        first_i, first_k = np.argwhere(keep)[0]
        z_world = np.asarray(affine, dtype=float) @ np.array([first_i, voxel_j, first_k, 1.0])
        z_values.append(round(float(z_world[2]), 2))
        spot_counts.append(n_spots)
    order = np.argsort(z_values)
    ordered_z = [z_values[index] for index in order]
    ordered_counts = [spot_counts[index] for index in order]
    return {'n_z_levels': len(ordered_z), 'z_values': ordered_z, 'spot_counts': ordered_counts, 'n_spots': int(sum(ordered_counts))}

def validate_blueprint_payload(config: dict[str, Any], age: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get('metadata', {}).get('age') != age:
        raise ValueError(f'{age}: blueprint age metadata mismatch')
    if payload.get('metadata', {}).get('expression_source') != 'none':
        raise ValueError(f'{age}: target blueprint must not contain expression')
    entries = payload.get('slices')
    if not isinstance(entries, list):
        raise ValueError(f'{age}: blueprint slices must be a list')
    expected_z = expected_z_values(config, age)
    if len(entries) != len(expected_z):
        raise ValueError(f'{age}: expected {len(expected_z)} levels, found {len(entries)}')
    total_spots = 0
    all_regions: set[str] = set()
    for index, (entry, expected) in enumerate(zip(entries, expected_z)):
        if int(entry.get('z_index', -1)) != index or float(entry['z_world']) != expected:
            raise ValueError(f'{age}: noncontiguous z contract at index {index}')
        n_spots = int(entry['n_spots'])
        fields = ('voxel_i', 'voxel_k', 'x', 'y', 'region')
        if any((len(entry[field]) != n_spots for field in fields)):
            raise ValueError(f'{age} z-index {index}: blueprint arrays differ in length')
        if not np.isfinite(np.asarray(entry['x'], dtype=float)).all() or not np.isfinite(np.asarray(entry['y'], dtype=float)).all():
            raise ValueError(f'{age} z-index {index}: non-finite coordinates')
        voxel_pairs = np.column_stack([entry['voxel_i'], entry['voxel_k']]).astype(np.int64)
        if len(np.unique(voxel_pairs, axis=0)) != n_spots:
            raise ValueError(f'{age} z-index {index}: duplicate deterministic spot IDs')
        all_regions.update(map(str, entry['region']))
        total_spots += n_spots
    if total_spots != int(config['ages'][age]['expected_spots']):
        raise ValueError(f"{age}: expected {config['ages'][age]['expected_spots']} spots, found {total_spots}")
    if all_regions != EXPECTED_REGIONS:
        raise ValueError(f'{age}: region support changed: {sorted(all_regions)}')
    return entries

def write_blueprint(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite blueprint: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            for chunk in json.JSONEncoder(separators=(',', ':'), ensure_ascii=True).iterencode(payload):
                compressed.write(chunk.encode('utf-8'))

def verify_prepared_blueprint(config: dict[str, Any], age: str) -> dict[str, Any]:
    output = blueprint_path(config, age)
    manifest = blueprint_manifest_path(config, age)
    provenance_path = blueprint_provenance_path(config, age)
    for path in (output, manifest, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f'prepared {age} artifact is missing: {path}')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    volume = config['_devccf_dir'] / config['ages'][age]['volume']
    schema = config['_devccf_dir'] / config['blueprint']['region_schema']
    expected = {'blueprint_configuration_id': str(config['blueprint_configuration_id']), 'age': age, 'n_z_levels': int(config['ages'][age]['expected_levels']), 'n_spots': int(config['ages'][age]['expected_spots']), 'expression_source': 'none', 'smoothing': 'none'}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f'prepared {age} provenance differs for {key}')
    return provenance

def load_blueprint(config: dict[str, Any], age: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = blueprint_path(config, age)
    verify_prepared_blueprint(config, age)
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        payload = json.load(handle)
    return (payload, validate_blueprint_payload(config, age, payload))

def blueprint_contract_row(age: str, entry: dict[str, Any]) -> dict[str, Any]:
    ids = make_spot_ids(age, entry)
    xy = np.column_stack([entry['x'], entry['y']])
    xyz = np.column_stack([xy, np.full(len(ids), float(entry['z_world']))])
    return {'age': age, 'z_index': int(entry['z_index']), 'z_world': f"{float(entry['z_world']):.2f}", 'voxel_j': int(entry['voxel_j']), 'n_spots': int(entry['n_spots'])}

def frozen_transport_config(config: dict[str, Any], assignment_randomness: float) -> dict[str, Any]:
    values = config['transport']
    return {'marginal_model': str(values['marginal_model']), 'epsilon': float(values['epsilon']), 'sinkhorn_iter': int(values['sinkhorn_iter']), 'sinkhorn_tol': float(values['sinkhorn_tol']), 'unbalanced_transport': bool(values['unbalanced_transport']), 'reg_m': float(values['reg_m']), 'transport_nonconvergence': str(values['transport_nonconvergence']), 'sinkhorn_method': str(values['sinkhorn_method']), 'transport_backend': str(values['transport_backend']), 'transport_device': str(values['transport_device']), 'transport_dtype': str(values['transport_dtype']), 'geometry_weight': float(values['geometry_weight']), 'boundary_weight': float(values['boundary_weight']), 'assignment_randomness': float(assignment_randomness), 'latent_clip_eps': float(values['latent_clip_eps']), 'gene_chunk_size': int(values['gene_chunk_size']), 'max_transport_pairs': int(values['max_transport_pairs']), 'quantile_field_mode': str(values['quantile_field_mode']), 'rank_scope': str(values['rank_scope']), 'tie_policy': str(values['tie_policy']), 'store_latent_scores': bool(values['store_latent_scores']), 'store_quantiles': bool(values['store_quantiles'])}

def transport_summary(adata: Any, expected_transport: dict[str, Any]) -> dict[str, Any]:
    diagnostics = adata.uns.get('de_novo', {}).get('transport_diagnostics')
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise ValueError('missing FEAST transport diagnostics')
    errors: list[float] = []
    iterations: list[int] = []
    records = 0
    for label, table in diagnostics.items():
        if not isinstance(table, dict) or table.get('format') != 'columnar_records_v1':
            raise ValueError(f'invalid transport diagnostic table for {label}')
        n_records = int(table.get('n_records', -1))
        converged = list(table.get('transport_converged', []))
        final_errors = list(table.get('transport_final_error', []))
        thresholds = list(table.get('transport_stop_threshold', []))
        maximums = list(table.get('transport_max_iterations', []))
        masses = list(table.get('transport_mass', []))
        policies = list(table.get('transport_nonconvergence_policy', []))
        iteration_values = list(table.get('transport_iterations', []))
        methods = list(table.get('transport_solver_method', []))
        backends = list(table.get('transport_backend', []))
        devices = list(table.get('transport_device', []))
        dtypes = list(table.get('transport_dtype', []))
        if not all((len(values) == n_records for values in (converged, final_errors, thresholds, maximums, masses, policies, iteration_values, methods, backends, devices, dtypes))):
            raise ValueError(f'incomplete transport diagnostics for {label}')
        for ok, error, threshold, maximum, mass, policy, n_iter, method, backend, device, dtype in zip(converged, final_errors, thresholds, maximums, masses, policies, iteration_values, methods, backends, devices, dtypes):
            error_value = float(error)
            threshold_value = float(threshold)
            maximum_value = int(float(maximum))
            mass_value = float(mass)
            iteration_value = int(float(n_iter))
            if str(ok).lower() != 'true' or str(policy) != 'raise':
                raise ValueError(f'nonconverged or non-strict transport record for {label}')
            if str(method) != str(expected_transport['sinkhorn_method']):
                raise ValueError(f'transport solver method changed for {label}')
            if str(backend) != str(expected_transport['transport_backend']):
                raise ValueError(f'transport backend changed for {label}')
            if str(device) != str(expected_transport['transport_device']):
                raise ValueError(f'transport device changed for {label}')
            if str(dtype) != str(expected_transport['transport_dtype']):
                raise ValueError(f'transport dtype changed for {label}')
            if threshold_value != float(expected_transport['sinkhorn_tol']):
                raise ValueError(f'transport stop threshold changed for {label}')
            if maximum_value != int(expected_transport['sinkhorn_iter']):
                raise ValueError(f'transport iteration cap changed for {label}')
            if not 0 <= iteration_value <= maximum_value:
                raise ValueError(f'transport iteration count is outside its bound for {label}')
            if not np.isfinite(mass_value) or mass_value <= 0.0:
                raise ValueError(f'transport mass is not positive and finite for {label}')
            if not np.isfinite(error_value) or not error_value < float(expected_transport['sinkhorn_tol']):
                raise ValueError(f'transport lacks positive convergence evidence for {label}')
            errors.append(error_value)
            iterations.append(iteration_value)
        records += n_records
    if records < 1:
        raise ValueError('transport diagnostics contain no solver records')
    return {'transport_records': records, 'all_converged': True, 'max_final_error': max(errors), 'max_iterations_observed': max(iterations)}

def validate_result_identity(adata: Any, age: str, entry: dict[str, Any], genes: list[str], expected_transport: dict[str, Any]) -> dict[str, Any]:
    ids = make_spot_ids(age, entry)
    regions = list(map(str, entry['region']))
    xy = np.column_stack([entry['x'], entry['y']]).astype(np.float64)
    xyz = np.column_stack([xy, np.full(len(ids), float(entry['z_world']))])
    if list(map(str, adata.obs_names)) != ids:
        raise ValueError('spot identity/order differs from blueprint')
    if list(map(str, adata.var_names)) != genes:
        raise ValueError('gene identity/order differs from reference panel')
    if 'region' not in adata.obs or list(adata.obs['region'].astype(str)) != regions:
        raise ValueError('region identity/order differs from blueprint')
    if 'spatial' not in adata.obsm or not np.array_equal(np.asarray(adata.obsm['spatial']), xy):
        raise ValueError('2D coordinates differ from blueprint')
    if 'spatial_3d' not in adata.obsm or not np.array_equal(np.asarray(adata.obsm['spatial_3d']), xyz):
        raise ValueError('3D coordinates differ from blueprint')
    if 'counts' not in adata.layers:
        raise ValueError("output is missing layers['counts']")
    import scipy.sparse as sp
    x_matrix = adata.X
    count_matrix = adata.layers['counts']
    x_dense = x_matrix.toarray() if sp.issparse(x_matrix) else np.asarray(x_matrix)
    count_dense = count_matrix.toarray() if sp.issparse(count_matrix) else np.asarray(count_matrix)
    if not np.array_equal(x_dense, count_dense):
        raise ValueError("X and layers['counts'] differ")
    values = x_dense
    if not np.isfinite(values).all() or np.any(values < 0) or (not np.equal(values, np.floor(values)).all()):
        raise ValueError('generated counts must be finite, nonnegative, and integral')
    return {**blueprint_contract_row(age, entry), **transport_summary(adata, expected_transport)}

def validate_publication_lineage(adata: Any, config: dict[str, Any], age: str, entry: dict[str, Any], expected_transport: dict[str, Any], expected_reference_artifacts: dict[str, dict[str, str]] | None=None) -> None:
    local = json.loads(adata.uns['local_generation_json'])
    if local['n_references'] != config['generation']['n_references']:
        raise ValueError('local reference selection setting differs')
    if adata.uns['de_novo']['marginal_model'] != 'core':
        raise ValueError('Study 07 requires full core generation')
    reference_artifacts = reference_artifact_contract(config, age) if expected_reference_artifacts is None else expected_reference_artifacts
    allowed_names = {Path(item['filename']).stem for item in reference_artifacts.values()}
    for group, weights in local['group_weights'].items():
        names = set(map(str, adata.uns['de_novo']['transport_diagnostics'][group]['reference_name']))
        if names != set(weights):
            raise ValueError('spatial and parameter reference selection differs')
        if not names <= allowed_names:
            raise ValueError(f'{age}: selected references outside declared cohort: {sorted(names - allowed_names)}')
    record = adata.uns.get('publication_reproduction')
    if not isinstance(record, dict):
        raise ValueError('missing publication reproduction lineage')
    if not record.get('feast_version') or not record.get('feast_source'):
        raise ValueError('missing recorded FEAST runtime information')
    if config.get('feast_version') and record['feast_version'] != str(config['feast_version']):
        raise ValueError('publication FEAST version differs from configured version')
    expected_scalars = {'configuration_id': str(config['configuration_id']), 'study_id': '07', 'legacy_study_id': '06', 'age': age, 'z_index': int(entry['z_index']), 'z_world': float(entry['z_world']), 'public_seed': int(config['public_seed']) + int(entry['z_index']), 'assignment_randomness': float(expected_transport['assignment_randomness'])}
    for key, expected in expected_scalars.items():
        observed = record.get(key)
        if isinstance(expected, float):
            if float(observed) != expected:
                raise ValueError(f'publication lineage differs for {key}')
        elif observed != expected:
            raise ValueError(f'publication lineage differs for {key}')
    if record.get('reference_artifacts') != reference_artifacts:
        raise ValueError('publication reference artifact IDs differ')
    if record.get('transport_config') != expected_transport:
        raise ValueError('publication transport configuration differs')
    expected_ar_provenance = assignment_randomness_provenance(config, age)
    if record.get('assignment_randomness_provenance') != expected_ar_provenance:
        raise ValueError('publication assignment-randomness provenance differs')
    observed_solver = record.get('solver_diagnostics', {})
    recomputed = transport_summary(adata, expected_transport)
    for key in ('transport_records', 'all_converged', 'max_final_error', 'max_iterations_observed'):
        if observed_solver.get(key) != recomputed[key]:
            raise ValueError(f'embedded solver diagnostic differs for {key}')
