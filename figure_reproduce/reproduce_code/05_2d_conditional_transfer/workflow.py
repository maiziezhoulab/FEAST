"""Shared input and execution helpers for clean Study 05 scripts."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
STUDY_DIR = Path(__file__).resolve().parent
REQUIRED_MANIFEST_COLUMNS = {'dataset', 'slice_id', 'relative_path', 'artifact_id', 'n_obs', 'n_vars'}

@dataclass(frozen=True)
class Job:
    mode: str
    dataset: str
    source: str
    target: str
    assignment_randomness: float

@dataclass
class PreparedJob:
    reference: ad.AnnData
    target_contract: ad.AnnData
    genes: list[str]
    support: dict[str, Any]

def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise RuntimeError('Study 05 configuration must be a mapping')
    return config

def ar_token(value: float) -> str:
    return f'{float(value):.1f}'.replace('.', '_')

def direction_id(job: Job) -> str:
    if job.mode == 'cross_slice':
        return f'{job.source}_to_{job.target}'
    if job.mode == 'mask_half':
        return f'{job.source}_low_x_to_high_x'
    raise ValueError(f'unknown Study 05 mode: {job.mode}')

def job_id(config: dict[str, Any], job: Job) -> str:
    return f"{config['configuration_id']}-{job.mode}-{job.dataset}-{direction_id(job)}-ar-{job.assignment_randomness:.1f}"

def declared_jobs(config: dict[str, Any]) -> list[Job]:
    jobs: list[Job] = []
    randomness_values = [float(value) for value in config['assignment_randomness']]
    mask_randomness = float(config['mask_half']['assignment_randomness'])
    for dataset, dataset_config in config['datasets'].items():
        for source, target in dataset_config['directions']:
            jobs.extend((Job('cross_slice', str(dataset), str(source), str(target), randomness) for randomness in randomness_values))
        jobs.extend((Job('mask_half', str(dataset), str(slice_id), str(slice_id), mask_randomness) for slice_id in dataset_config['slices']))
    if len(jobs) != int(config['expected_jobs']) or len(set(jobs)) != len(jobs):
        raise RuntimeError('declared Study 05 job matrix differs from expected_jobs')
    cross_count = sum((job.mode == 'cross_slice' for job in jobs))
    mask_count = sum((job.mode == 'mask_half' for job in jobs))
    if cross_count != int(config['expected_cross_slice_jobs']) or mask_count != int(config['expected_mask_half_jobs']):
        raise RuntimeError('declared Study 05 mode counts differ from configuration')
    return jobs

def require_declared_job(config: dict[str, Any], job: Job) -> None:
    if job not in set(declared_jobs(config)):
        raise ValueError(f'undeclared Study 05 job: {job}')

def expected_job_path(output_root: Path, job: Job) -> Path:
    return output_root / job.mode / job.dataset / direction_id(job) / f'ar_{ar_token(job.assignment_randomness)}'

def dataset_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    try:
        result = config['datasets'][dataset]
    except KeyError as exc:
        raise ValueError(f'undeclared Study 05 dataset: {dataset}') from exc
    if not isinstance(result, dict):
        raise RuntimeError(f'invalid dataset configuration: {dataset}')
    return result

def input_artifact_id(config: dict[str, Any], dataset: str, slice_id: str) -> str:
    prefix = str(dataset_config(config, dataset)['artifact_prefix'])
    return f'{prefix}_{slice_id}'

def _validate_count_matrix(matrix: Any, label: str) -> None:
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data, dtype=np.float64)
    else:
        values = np.asarray(matrix, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError(f'{label} contains negative or non-finite values')
    if values.size and np.max(np.abs(values - np.rint(values))) > 1e-06:
        raise RuntimeError(f'{label} is not raw integer count data')

def _validate_slice(adata: ad.AnnData, path: Path, annotation_key: str, check_counts: bool) -> None:
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise RuntimeError(f'{path} has non-unique observation or gene identifiers')
    if annotation_key not in adata.obs or adata.obs[annotation_key].isna().any():
        raise RuntimeError(f'{path} lacks complete {annotation_key!r} labels')
    if 'spatial' not in adata.obsm:
        raise RuntimeError(f"{path} lacks obsm['spatial']")
    coordinates = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError(f'{path} does not contain finite two-dimensional coordinates')
    if check_counts:
        matrix = adata.layers['counts'] if 'counts' in adata.layers else adata.X
        _validate_count_matrix(matrix, str(path))

def inspect_inputs(config_path: Path, manifest_path: Path, input_dir: Path, *, check_counts: bool=False) -> tuple[dict[str, Any], dict[tuple[str, str], Path], dict[str, list[str]]]:
    config = load_config(config_path)
    manifest = pd.read_csv(manifest_path, dtype={'dataset': str, 'slice_id': str, 'artifact_id': str})
    if not REQUIRED_MANIFEST_COLUMNS.issubset(manifest.columns):
        raise RuntimeError('input manifest columns differ from the Study 05 contract')
    expected = {(str(dataset), str(slice_id)) for dataset, entry in config['datasets'].items() for slice_id in entry['slices']}
    observed = set(zip(manifest['dataset'], manifest['slice_id']))
    if len(manifest) != len(expected) or observed != expected:
        raise RuntimeError('input manifest does not contain the exact declared slices')
    if manifest[['dataset', 'slice_id']].duplicated().any():
        raise RuntimeError('input manifest contains duplicate slice identities')
    paths: dict[tuple[str, str], Path] = {}
    gene_sets: dict[str, list[set[str]]] = {dataset: [] for dataset in config['datasets']}
    rows = manifest.set_index(['dataset', 'slice_id'])
    for dataset, slice_id in sorted(expected):
        row = rows.loc[dataset, slice_id]
        path = input_dir / str(row['relative_path'])
        if not path.is_file():
            raise RuntimeError(f'missing input: {dataset}/{slice_id}')
        expected_artifact = input_artifact_id(config, dataset, slice_id)
        if str(row['artifact_id']) != expected_artifact:
            raise RuntimeError(f'input artifact ID mismatch: {dataset}/{slice_id}')
        adata = ad.read_h5ad(path) if check_counts else ad.read_h5ad(path, backed='r')
        try:
            if adata.shape != (int(row['n_obs']), int(row['n_vars'])):
                raise RuntimeError(f'input shape mismatch: {dataset}/{slice_id}')
            annotation_key = str(dataset_config(config, dataset)['annotation_key'])
            _validate_slice(adata, path, annotation_key, check_counts)
            gene_sets[dataset].append(set(map(str, adata.var_names)))
            paths[dataset, slice_id] = path.resolve()
        finally:
            if not adata.isbacked:
                del adata
            else:
                adata.file.close()
    panels: dict[str, list[str]] = {}
    for dataset, sets_for_dataset in gene_sets.items():
        genes = sorted(set.intersection(*sets_for_dataset))
        panel = dataset_config(config, dataset)['gene_panel']
        if len(genes) != int(panel['expected_genes']):
            raise RuntimeError(f'common-gene count differs for dataset {dataset}')
        panels[dataset] = genes
    declared_jobs(config)
    return (config, paths, panels)

def _counts_matrix(adata: ad.AnnData) -> Any:
    return adata.layers['counts'] if 'counts' in adata.layers else adata.X

def _nonzero_per_gene(adata: ad.AnnData) -> np.ndarray:
    matrix = _counts_matrix(adata)
    if sparse.issparse(matrix):
        return np.asarray((matrix != 0).sum(axis=0)).reshape(-1)
    return np.count_nonzero(np.asarray(matrix), axis=0)

def prepare_job(config: dict[str, Any], paths: dict[tuple[str, str], Path], panels: dict[str, list[str]], job: Job) -> PreparedJob:
    """Load source expression only and build an expression-free target contract."""
    require_declared_job(config, job)
    entry = dataset_config(config, job.dataset)
    annotation_key = str(entry['annotation_key'])
    panel = panels[job.dataset]
    source_path = paths[job.dataset, job.source]
    target_path = paths[job.dataset, job.target]
    source_backed = ad.read_h5ad(source_path, backed='r')
    target_backed = source_backed if source_path == target_path else ad.read_h5ad(target_path, backed='r')
    try:
        source_labels = source_backed.obs[annotation_key].astype(str).to_numpy()
        target_labels = target_backed.obs[annotation_key].astype(str).to_numpy()
        source_mask = np.ones(source_backed.n_obs, dtype=bool)
        target_mask = np.ones(target_backed.n_obs, dtype=bool)
        split_record: dict[str, Any] | None = None
        if job.mode == 'mask_half':
            mask_config = config['mask_half']
            axis = int(mask_config['split_axis'])
            quantile = float(mask_config['split_quantile'])
            coordinates = np.asarray(source_backed.obsm['spatial'], dtype=np.float64)
            split = float(np.quantile(coordinates[:, axis], quantile))
            source_mask = coordinates[:, axis] <= split
            target_mask = coordinates[:, axis] > split
            if np.any(source_mask & target_mask) or not np.all(source_mask | target_mask):
                raise RuntimeError('half-slice split is not disjoint and exhaustive')
            split_record = {'axis': axis, 'coordinate': 'x' if axis == 0 else 'y', 'quantile': quantile, 'threshold': split, 'observed_relation': str(mask_config['observed_relation']), 'masked_relation': str(mask_config['masked_relation']), 'direction': str(mask_config['direction'])}
        minimum = int(config['label_support']['min_source_spots'])
        source_counts = {label: int(np.sum(source_mask & (source_labels == label))) for label in sorted(set(source_labels))}
        eligible_labels = sorted((label for label, count in source_counts.items() if count >= minimum))
        if not eligible_labels:
            raise RuntimeError(f'no labels satisfy source support for {job}')
        source_retained = source_mask & np.isin(source_labels, eligible_labels)
        target_retained = target_mask & np.isin(target_labels, eligible_labels)
        if not source_retained.any() or not target_retained.any():
            raise RuntimeError(f'empty retained source or target support for {job}')
        reference_rows = source_backed[source_retained, :].to_memory()
        reference = reference_rows[:, panel].copy()
        coordinate_contract = config['coordinates']
        reference_coordinate_key = str(coordinate_contract['reference_key'])
        target_coordinate_key = str(coordinate_contract['target_key'])
        coordinate_dimensions = int(coordinate_contract['dimensions'])
        removed_reference_keys = []
        for key in map(str, coordinate_contract['remove_reference_keys']):
            if key in reference.obsm:
                del reference.obsm[key]
                removed_reference_keys.append(key)
        reference_coordinates = np.asarray(reference.obsm[reference_coordinate_key], dtype=np.float64)
        if reference_coordinates.shape != (reference.n_obs, coordinate_dimensions):
            raise RuntimeError('reference coordinates differ from the frozen 2D contract')
        nonzero = _nonzero_per_gene(reference)
        genes = list(panel)
        observed_genes = [gene for gene, count in zip(panel, nonzero) if int(count) >= 1]
        zero_genes = [gene for gene, count in zip(panel, nonzero) if int(count) == 0]
        if not observed_genes:
            raise RuntimeError(f'no expressed reference genes remain for {job}')
        reference.uns['reference_name'] = f'{job.dataset}_{job.source}_{job.mode}'
        target_indices = np.flatnonzero(target_retained)
        target_obs = pd.DataFrame({annotation_key: target_labels[target_indices]}, index=pd.Index(target_backed.obs_names.astype(str).to_numpy()[target_indices], name=target_backed.obs_names.name))
        target_contract = ad.AnnData(X=None, obs=target_obs, var=pd.DataFrame(index=pd.Index(genes)))
        target_contract.obsm[target_coordinate_key] = np.asarray(target_backed.obsm[target_coordinate_key], dtype=np.float64)[target_indices]
        if target_contract.obsm[target_coordinate_key].shape != (target_contract.n_obs, coordinate_dimensions):
            raise RuntimeError('target coordinates differ from the frozen 2D contract')
        excluded_target = target_mask & ~np.isin(target_labels, eligible_labels)
        excluded_counts = {label: int(np.sum(excluded_target & (target_labels == label))) for label in sorted(set(target_labels[excluded_target]))}
        source_candidate_ids = source_backed.obs_names.astype(str).to_numpy()[source_mask]
        source_retained_ids = source_backed.obs_names.astype(str).to_numpy()[source_retained]
        target_candidate_ids = target_backed.obs_names.astype(str).to_numpy()[target_mask]
        target_retained_ids = target_contract.obs_names.astype(str).to_numpy()
        support = {'policy': str(config['label_support']['policy']), 'min_source_spots_per_label': minimum, 'source_candidate_spots': int(source_mask.sum()), 'source_retained_spots': int(source_retained.sum()), 'target_candidate_spots': int(target_mask.sum()), 'target_retained_spots': int(target_retained.sum()), 'target_retained_fraction': float(target_retained.sum() / target_mask.sum()), 'eligible_labels': eligible_labels, 'source_label_counts': {label: source_counts[label] for label in eligible_labels}, 'excluded_target_label_counts': excluded_counts, 'dataset_gene_panel_count': len(panel), 'reference_observed_gene_count': len(observed_genes), 'reference_zero_gene_count': len(zero_genes), 'reference_zero_genes': zero_genes, 'coordinate_contract': {'reference_key': reference_coordinate_key, 'target_key': target_coordinate_key, 'dimensions': coordinate_dimensions, 'removed_reference_keys': removed_reference_keys}, 'mask_split': split_record}
        return PreparedJob(reference, target_contract, genes, support)
    finally:
        source_backed.file.close()
        if target_backed is not source_backed:
            target_backed.file.close()

def columnar_transport_records(adata: ad.AnnData) -> list[dict[str, Any]]:
    try:
        payload = adata.uns['de_novo']['transport_diagnostics']
    except (KeyError, TypeError) as exc:
        raise RuntimeError('missing FEAST conditional transport diagnostics') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('transport diagnostics are not grouped by label')
    records: list[dict[str, Any]] = []
    for label, columns in payload.items():
        if not isinstance(columns, dict) or columns.get('format') != 'columnar_records_v1':
            raise RuntimeError(f'invalid transport diagnostics encoding for label {label}')
        n_records = int(columns.get('n_records', -1))
        if n_records < 1:
            raise RuntimeError(f'no transport diagnostics for label {label}')
        keys = [key for key in columns if key not in {'format', 'n_records'}]
        for index in range(n_records):
            record: dict[str, Any] = {'label': str(label)}
            for key in keys:
                values = np.asarray(columns[key]).reshape(-1)
                if len(values) != n_records:
                    raise RuntimeError(f'malformed transport diagnostics column {key}')
                value = values[index]
                record[str(key)] = value.item() if hasattr(value, 'item') else value
            records.append(record)
    return records

def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value) in {'True', 'true'}:
        return True
    if str(value) in {'False', 'false'}:
        return False
    raise RuntimeError(f'invalid boolean solver diagnostic: {value!r}')

def validate_transport_evidence(records: list[dict[str, Any]], transport: dict[str, Any], expected_labels: Iterable[str]) -> None:
    expected = {str(value) for value in expected_labels}
    if len(records) != len(expected) or {row['label'] for row in records} != expected:
        raise RuntimeError('transport diagnostics do not cover each target label exactly once')
    tolerance = float(transport['sinkhorn_tol'])
    max_iterations = int(transport['sinkhorn_iter'])
    for row in records:
        final_error = float(row['transport_final_error'])
        mass = float(row['transport_mass'])
        iterations = int(row['transport_iterations'])
        stop_threshold = float(row['transport_stop_threshold'])
        if not _strict_bool(row['transport_converged']):
            raise RuntimeError(f"transport did not converge for label {row['label']}")
        if row['transport_nonconvergence_policy'] != 'raise':
            raise RuntimeError("transport nonconvergence policy was not 'raise'")
        if not np.isfinite(final_error) or final_error >= tolerance:
            raise RuntimeError(f"transport error exceeds tolerance for label {row['label']}")
        if not np.isfinite(mass) or mass <= 0.0:
            raise RuntimeError(f"transport mass is invalid for label {row['label']}")
        if iterations < 1 or iterations > max_iterations:
            raise RuntimeError(f"transport iteration evidence is invalid for label {row['label']}")
        if not np.isclose(stop_threshold, tolerance, rtol=0.0, atol=1e-15):
            raise RuntimeError('recorded transport tolerance differs from the configuration')
        if int(row['transport_max_iterations']) != max_iterations:
            raise RuntimeError('recorded transport iteration limit differs from the configuration')

def validate_generated(generated: ad.AnnData, target: ad.AnnData, genes: list[str], config: dict[str, Any], dataset: str, *, require_bound_obs_names: bool=True) -> list[dict[str, Any]]:
    annotation_key = str(dataset_config(config, dataset)['annotation_key'])
    spot_id_column = str(config['blueprint']['spot_id_column'])
    allowed_columns = {spot_id_column, annotation_key, 'domain'}
    if generated.shape != (target.n_obs, len(genes)):
        raise RuntimeError('generated shape differs from the exact target/panel support')
    if require_bound_obs_names and list(map(str, generated.obs_names)) != list(map(str, target.obs_names)):
        raise RuntimeError('generated observation identity/order differs from the target')
    if list(map(str, generated.var_names)) != genes:
        raise RuntimeError('generated ordered gene panel differs from the contract')
    if set(map(str, generated.obs.columns)) != allowed_columns:
        raise RuntimeError('generated obs contains undeclared target covariates')
    np.testing.assert_array_equal(generated.obs[spot_id_column].astype(str).to_numpy(), target.obs_names.astype(str).to_numpy(), err_msg='blueprint target IDs differ from the target')
    np.testing.assert_array_equal(generated.obs[annotation_key].astype(str).to_numpy(), target.obs[annotation_key].astype(str).to_numpy(), err_msg='generated labels differ from the target')
    np.testing.assert_array_equal(np.asarray(generated.obsm['spatial'], dtype=np.float64), np.asarray(target.obsm['spatial'], dtype=np.float64), err_msg='generated coordinates differ from the target')
    if 'counts' not in generated.layers:
        raise RuntimeError("generated H5AD lacks layers['counts']")
    for start in range(0, generated.n_obs, 256):
        stop = min(start + 256, generated.n_obs)
        left = generated.X[start:stop]
        right = generated.layers['counts'][start:stop]
        left = left.toarray() if sparse.issparse(left) else np.asarray(left)
        right = right.toarray() if sparse.issparse(right) else np.asarray(right)
        _validate_count_matrix(left, 'generated X')
        if not np.array_equal(left, right):
            raise RuntimeError('generated X and counts layer differ')
    records = columnar_transport_records(generated)
    validate_transport_evidence(records, config['transport'], target.obs[annotation_key].astype(str).unique())
    return records
