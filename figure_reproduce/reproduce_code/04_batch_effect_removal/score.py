"""Score the Study 04A metric and paired-support rows."""
from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
import run as workflow
import metric_support as support
STUDY_ROOT = Path(__file__).resolve().parent
REPRODUCTION_ROOT = STUDY_ROOT.parents[1]
K_NEIGHBORS = 30
MMD_SAMPLE_PER_BATCH = 1000
METRIC_COLUMNS = ('batch_asw', 'batch_mixing_entropy_k30', 'paired_retrieval_top1', 'paired_retrieval_median_rank', 'domain_asw', 'domain_purity_k30', 'centroid_distance', 'covariance_distance', 'mmd2_rbf_biased', 'paired_expression_correlation')
CORE_METRIC_COLUMNS = METRIC_COLUMNS[:-1]
ALL_SPOTS = support.ALL_SPOTS
PAIRED_SUPPORT = support.PAIRED_SUPPORT
STAMP_RETRY_POLICY_ID = 'study04-stamp-numerical-retry-v1'
STAMP_RETRY_POLICY_PATH = 'reproduce_code/04_batch_effect_removal/stamp_numerical_retry_v1.yaml'
STAMP_RETRY_ELIGIBLE = {('shift_only', '1.50'), ('diagonal_affine', '1.50')}
STAMP_RETRY_UNCHANGED_CONTRACT = {'public_seed': 42, 'n_topics': 10, 'n_layers': 1, 'hidden_size': 128, 'mode': 'sgc', 'learning_rate': 0.005, 'max_epochs': 800, 'min_epochs': 100, 'device': 'cuda:0', 'zero_library_strategy': 'raw_zero_likelihood_within_batch_sgc_encoder_v1'}

def alpha_text(value: object) -> str:
    return f'{float(value):.2f}'

def _retry_bound_file(relative_path: object, label: str) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise ValueError(f'STAMP retry {label} path must be repository-relative')
    path = (REPRODUCTION_ROOT / relative).resolve()
    if not path.is_relative_to(REPRODUCTION_ROOT):
        raise ValueError(f'STAMP retry {label} path escapes the reproduction repository')
    if not path.is_file():
        raise ValueError(f'STAMP retry {label} is missing: {path}')
    return path

def validate_stamp_retry(metadata: dict, candidate: Path) -> None:
    """Fail closed unless STAMP used standard patience or the pinned narrow retry."""
    patience = int(metadata.get('patience', -1))
    retry = metadata.get('numerical_retry')
    if patience == 20:
        if retry is not None:
            raise ValueError('standard-patience STAMP candidate carries a retry declaration')
        return
    if patience != 10:
        raise ValueError(f'STAMP patience changed outside the declared contract: {patience}')
    if not isinstance(retry, dict):
        raise ValueError('patience-10 STAMP candidate lacks numerical_retry provenance')
    context = metadata.get('provenance', {}).get('candidate_context', {})
    mode = str(context.get('deformation_mode', ''))
    alpha = alpha_text(context.get('alpha', float('nan')))
    if (mode, alpha) not in STAMP_RETRY_ELIGIBLE:
        raise ValueError(f'STAMP retry is not eligible for {mode}/alpha_{alpha}')
    eligible_condition = retry.get('eligible_condition', {})
    if eligible_condition.get('mode') != mode or alpha_text(eligible_condition.get('alpha', float('nan'))) != alpha:
        raise ValueError('STAMP retry eligible-condition record changed')
    if retry.get('policy_configuration_id') != STAMP_RETRY_POLICY_ID:
        raise ValueError('STAMP retry policy configuration ID changed')
    if retry.get('policy_path') != STAMP_RETRY_POLICY_PATH:
        raise ValueError('STAMP retry policy path changed')
    policy_path = _retry_bound_file(retry.get('policy_path'), 'policy')
    if policy_path != (STUDY_ROOT / 'stamp_numerical_retry_v1.yaml').resolve():
        raise ValueError('STAMP retry policy resolves to the wrong file')
    if retry.get('changed_parameter') != {'name': 'patience', 'standard_value': 20, 'retry_value': 10}:
        raise ValueError('STAMP retry changed-parameter record is not patience 20 -> 10')
    if retry.get('unchanged_contract') != STAMP_RETRY_UNCHANGED_CONTRACT:
        raise ValueError('STAMP retry unchanged-contract record changed')
    reason = str(retry.get('activation_reason', ''))
    if not all((token in reason for token in ('non-finite', 'Pyro', 'patience'))):
        raise ValueError('STAMP retry activation reason is not explicit')
    failure = _retry_bound_file(retry.get('original_failure_path'), 'original failure')
    disposition = _retry_bound_file(retry.get('original_disposition_path'), 'original disposition')
    expected_failure_prefix = f'STAMP__{mode}__alpha_{alpha}__'
    run_dir = candidate.parents[3].resolve()
    if candidate.name != f'alpha_{alpha}' or candidate.parent.name != mode or candidate.parent.parent.name != 'STAMP' or (failure.name != 'FAILED.txt') or (disposition.name != 'DISPOSITION.txt') or (failure.parent != disposition.parent) or (not failure.parent.name.startswith(expected_failure_prefix)) or (failure.parent.parent.resolve() != run_dir / 'failures'):
        raise ValueError('STAMP retry is not bound to its matching preserved failure')
    failure_text = failure.read_text(encoding='utf-8')
    if not all((token in failure_text for token in ('caux', 'tensor([nan]', 'cuda:0'))):
        raise ValueError('STAMP retry failure does not prove the declared Pyro non-finiteness')
    if disposition.read_text(encoding='utf-8').strip() != 'exit=1; missing metadata.json':
        raise ValueError('STAMP retry failure disposition changed')
    expected_actual = {'public_seed': int(metadata.get('seed', -1)), 'n_topics': int(metadata.get('n_topics', -1)), 'n_layers': int(metadata.get('n_layers', -1)), 'hidden_size': int(metadata.get('hidden_size', -1)), 'mode': metadata.get('mode'), 'learning_rate': float(metadata.get('learning_rate', float('nan'))), 'max_epochs': int(metadata.get('max_epochs', -1)), 'min_epochs': int(metadata.get('min_epochs', -1)), 'device': metadata.get('device'), 'zero_library_strategy': metadata.get('zero_library_safeguard', {}).get('strategy')}
    if expected_actual != STAMP_RETRY_UNCHANGED_CONTRACT:
        raise ValueError('STAMP retry candidate differs from its unchanged contract')
    provenance = metadata.get('provenance', {})
    diagnostics = provenance.get('solver_diagnostics', {})
    if int(provenance.get('public_seed', -1)) != 42:
        raise ValueError('STAMP retry provenance seed changed')
    if float(diagnostics.get('learning_rate', float('nan'))) != 0.005 or diagnostics.get('cuda_execution_verified') is not True or diagnostics.get('actual_device') != 'cuda:0':
        raise ValueError('STAMP retry LR/CUDA provenance changed')
    graph = metadata.get('spatial_graph', {})
    graph_contract = {'schema_version': 3, 'construction': 'independent_directed_knn_by_batch', 'distance': 'squared_euclidean', 'self_exclusion': 'exact_row_identity', 'distance_tie_break': 'original_within_batch_spot_order', 'n_neighbors': 6, 'n_spots': 7222, 'n_batches': 2, 'directed_edge_count': 43332, 'cross_batch_edge_count': 0, 'self_edge_count': 0}
    if any((graph.get(key) != value for key, value in graph_contract.items())):
        raise ValueError('STAMP retry graph contract changed')
    expected_inputs = {(run_dir / 'simulations' / mode / 'alpha_0.00.h5ad').resolve(), (run_dir / 'simulations' / mode / f'alpha_{alpha}.h5ad').resolve(), (run_dir / 'panel' / 'panel.csv').resolve(), (run_dir / 'panel' / 'provenance.json').resolve()}
    observed_inputs = {Path(str(item.get('path', ''))).resolve() for item in provenance.get('inputs', [])}
    if observed_inputs != expected_inputs:
        raise ValueError('STAMP retry input artifact set changed')
    policy_sources = [item for item in provenance.get('sources', []) if Path(str(item.get('path', ''))).resolve() == policy_path]
    if len(policy_sources) != 1:
        raise ValueError('STAMP retry provenance does not bind the pinned policy source')
    training = metadata.get('training', {})
    if training.get('early_stopped') is not True or int(training.get('early_stopping_counter', -1)) != 10:
        raise ValueError('STAMP retry did not stop under the declared patience-10 rule')

def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)

def standardize(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale

def neighbors_without_self(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[0] <= k:
        raise ValueError(f'{values.shape[0]} rows cannot support k={k}')
    raw = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values, return_distance=False)
    rows = []
    for index, neighbors in enumerate(raw):
        without_self = neighbors[neighbors != index]
        if without_self.size < k:
            raise RuntimeError('nearest-neighbor result lacks enough non-self rows')
        rows.append(without_self[:k])
    return np.vstack(rows)

def local_batch_entropy(batch_labels: np.ndarray, neighbors: np.ndarray) -> float:
    query_fraction = (batch_labels[neighbors] == 'query').mean(axis=1)
    entropy = np.zeros(query_fraction.size, dtype=np.float64)
    for probability in (query_fraction, 1.0 - query_fraction):
        nonzero = probability > 0
        entropy[nonzero] -= probability[nonzero] * np.log(probability[nonzero])
    return float(np.mean(entropy / np.log(2.0)))

def domain_purity(domain_labels: np.ndarray, valid: np.ndarray, neighbors: np.ndarray) -> float:
    neighbor_labels = domain_labels[neighbors]
    purity = (neighbor_labels == domain_labels[:, None]).mean(axis=1)
    return float(np.mean(purity[valid]))

def paired_retrieval(values: np.ndarray, n_reference: int) -> tuple[float, float]:
    reference = values[:n_reference]
    query = values[n_reference:]
    if reference.shape != query.shape:
        raise ValueError('paired retrieval requires equal reference/query support')
    distances = pairwise_distances(query, reference, metric='euclidean')
    paired = np.diag(distances)
    minimum = distances.min(axis=1)
    tolerance = 1e-12 + 1e-09 * np.maximum(1.0, np.abs(minimum))
    top1 = float(np.mean(paired <= minimum + tolerance))
    ranks = 1 + np.sum(distances < paired[:, None] - tolerance[:, None], axis=1)
    return (top1, float(np.median(ranks)))

def mmd2_rbf_biased(values: np.ndarray, n_reference: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    query_count = values.shape[0] - n_reference
    ref_index = rng.choice(n_reference, min(MMD_SAMPLE_PER_BATCH, n_reference), replace=False)
    query_index = rng.choice(query_count, min(MMD_SAMPLE_PER_BATCH, query_count), replace=False)
    x = values[ref_index]
    y = values[n_reference + query_index]
    pooled = np.vstack([x, y])
    distances = pairwise_distances(pooled, metric='euclidean')
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)
    value = rbf_kernel(x, x, gamma=gamma).mean() + rbf_kernel(y, y, gamma=gamma).mean() - 2.0 * rbf_kernel(x, y, gamma=gamma).mean()
    return (float(max(value, 0.0)), bandwidth)

def paired_expression_correlation(matrix: np.ndarray, n_reference: int, paired_spot_ids: np.ndarray) -> float:
    reference = np.asarray(matrix[:n_reference], dtype=np.float64)
    query = np.asarray(matrix[n_reference:], dtype=np.float64)
    if reference.shape != query.shape:
        raise ValueError('expression correlation requires equal paired support')
    reference -= reference.mean(axis=1, keepdims=True)
    query -= query.mean(axis=1, keepdims=True)
    numerator = np.sum(reference * query, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(query, axis=1)
    zero_variance = denominator <= 0
    if zero_variance.any():
        names = np.asarray(paired_spot_ids, dtype=str)[zero_variance].tolist()
        raise ValueError(f'paired-expression correlation has zero-variance named pairs: {names[:20]}')
    correlations = numerator / denominator
    if not np.isfinite(correlations).all():
        raise ValueError('paired-expression correlation is non-finite')
    return float(np.median(correlations))

def representations(method: str, embedding: np.ndarray, seed: int) -> list[tuple[str, bool, np.ndarray]]:
    if method != 'GraphST':
        representation = 'native_10d_topics' if method == 'STAMP' else 'native_10d_latent'
        return [(representation, True, embedding)]
    return [('pca20_randomized_seed42', True, PCA(n_components=20, svd_solver='randomized', random_state=seed).fit_transform(embedding)), ('pca10_randomized_seed42_sensitivity', False, PCA(n_components=10, svd_solver='randomized', random_state=seed).fit_transform(embedding))]

def metric_row(*, method: str, representation: str, primary: bool, embedding: np.ndarray, batch_labels: np.ndarray, domain_labels: np.ndarray, n_reference: int, expression_correlation: float | None, seed: int) -> dict[str, object]:
    values = standardize(embedding)
    neighbors = neighbors_without_self(values, K_NEIGHBORS)
    domain_text = pd.Series(domain_labels).astype('string').fillna('<NA>').to_numpy()
    valid_domain = ~np.isin(np.char.lower(domain_text.astype(str)), ['nan', 'none', 'unknown', '<na>'])
    if len(np.unique(domain_text[valid_domain])) < 2:
        raise ValueError('fewer than two valid domains remain')
    top1, median_rank = paired_retrieval(values, n_reference)
    mmd2, bandwidth = mmd2_rbf_biased(values, n_reference, seed)
    reference = values[:n_reference]
    query = values[n_reference:]
    return {'method': method, 'representation': representation, 'primary_representation': primary, 'representation_dimensions': int(values.shape[1]), 'batch_asw': float(silhouette_score(values, batch_labels)), 'batch_mixing_entropy_k30': local_batch_entropy(batch_labels, neighbors), 'paired_retrieval_top1': top1, 'paired_retrieval_median_rank': median_rank, 'domain_asw': float(silhouette_score(values[valid_domain], domain_text[valid_domain])), 'domain_purity_k30': domain_purity(domain_text, valid_domain, neighbors), 'centroid_distance': float(np.linalg.norm(reference.mean(axis=0) - query.mean(axis=0))), 'covariance_distance': float(np.linalg.norm(np.cov(reference, rowvar=False) - np.cov(query, rowvar=False), ord='fro') / values.shape[1]), 'mmd2_rbf_biased': mmd2, 'mmd_bandwidth': bandwidth, 'paired_expression_correlation': expression_correlation}

def load_panel(path: Path) -> list[str]:
    table = pd.read_csv(path)
    if list(table.columns[:2]) != ['panel_order', 'gene']:
        raise ValueError('panel must begin with panel_order,gene')
    if table['panel_order'].astype(int).tolist() != list(range(len(table))):
        raise ValueError('panel order is not contiguous')
    genes = table['gene'].astype(str).tolist()
    if len(set(genes)) != len(genes):
        raise ValueError('panel contains duplicate genes')
    return genes

def load_support(reference_path: Path, query_path: Path, genes: list[str], *, mode: str, alpha: float) -> dict[str, object]:
    reference = ad.read_h5ad(reference_path, backed='r')
    query = ad.read_h5ad(query_path, backed='r')
    try:
        if not reference.obs_names.equals(query.obs_names):
            raise ValueError('reference/query spot identity or order differs')
        if any((gene not in reference.var_names or gene not in query.var_names for gene in genes)):
            raise ValueError('fixed-panel support is incomplete')
        reference_matrix = reference[:, genes].X
        query_matrix = query[:, genes].X
        reference_zero = np.asarray(reference_matrix.sum(axis=1)).reshape(-1) == 0
        query_zero = np.asarray(query_matrix.sum(axis=1)).reshape(-1) == 0
        reference_zero_ids = reference.obs_names[reference_zero].astype(str).tolist()
        query_zero_ids = query.obs_names[query_zero].astype(str).tolist()
        if reference_zero_ids:
            raise ValueError(f'reference fixed panel has raw-zero rows for {mode}/{alpha:.2f}: {reference_zero_ids}')
        result = {'n_spots': int(reference.n_obs), 'spot_ids': reference.obs_names.astype(str).to_numpy(copy=True), 'remove_pairs': query_zero, 'query_zero_count': int(query_zero.sum()), 'query_zero_ids': query_zero_ids, 'reference_zero_count': 0, 'reference_zero_ids': reference_zero_ids, 'reference_path': str(reference_path.resolve()), 'query_path': str(query_path.resolve())}
    finally:
        reference.file.close()
        query.file.close()
    return result

def load_candidate(candidate: Path, method: str, candidate_id: str, config: dict, support: dict[str, object]) -> dict[str, object]:
    valid, reason = workflow.candidate_is_valid(candidate, method, candidate_id, config)
    if not valid:
        raise ValueError(f'invalid candidate {candidate}: {reason}')
    metadata_path = candidate / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    provenance = metadata.get('provenance', {})
    diagnostics = provenance.get('solver_diagnostics', {})
    if provenance.get('configuration_id') != candidate_id:
        raise ValueError('candidate provenance configuration ID changed')
    if diagnostics.get('status') != 'completed':
        raise ValueError('candidate solver status is not completed')
    if method == 'STAMP':
        validate_stamp_retry(metadata, candidate)
    with np.load(candidate / 'embeddings.npz', allow_pickle=True) as payload:
        embedding = np.asarray(payload['embedding'], dtype=np.float64)
        batch_labels = np.asarray(payload['batch_labels']).astype(str)
    n_spots = int(support['n_spots'])
    if embedding.shape[0] != 2 * n_spots or not np.isfinite(embedding).all():
        raise ValueError('candidate embedding support or finiteness failed')
    expected_batches = np.array(['ref'] * n_spots + ['query'] * n_spots)
    if not np.array_equal(batch_labels, expected_batches):
        raise ValueError('candidate batch order differs from the paired contract')
    result_path = candidate / 'result.h5ad'
    result = ad.read_h5ad(result_path, backed='r')
    try:
        result_batches = result.obs['batch'].astype(str).to_numpy(copy=True)
        if not np.array_equal(result_batches, expected_batches):
            raise ValueError('method result batch order differs from the paired contract')
        source_ids = result.obs['source_spot_id'].astype(str).to_numpy(copy=True)
        expected_ids = np.concatenate([support['spot_ids'], support['spot_ids']])
        if not np.array_equal(source_ids, expected_ids):
            raise ValueError('method output changed paired spot identity/order')
        domains = result.obs['dlpfc_layer'].to_numpy(copy=True)
        normalized = None
        if method == 'scVI':
            if 'scvi_normalized' not in result.layers:
                raise ValueError('scVI result lacks the declared decoded-expression layer')
            normalized = np.asarray(result.layers['scvi_normalized'])
            if normalized.shape[0] != 2 * n_spots or not np.isfinite(normalized).all():
                raise ValueError('scVI normalized expression is invalid')
            if normalized.shape[1] != int(config['fixed_panel']['n_genes']):
                raise ValueError('scVI normalized expression panel size changed')
            if np.any(normalized < 0):
                raise ValueError('scVI normalized expression contains negative values')
    finally:
        result.file.close()
    artifact_paths = list(workflow.METHOD_ARTIFACTS[method])
    method_environment = metadata.get('execution_environment', {}) if method == 'scVI' else metadata.get('environment', {})
    input_artifact_ids = [str(record.get('artifact_id')) for record in provenance.get('inputs', []) if record.get('artifact_id')]
    return {'embedding': embedding, 'batch_labels': batch_labels, 'domain_labels': domains, 'normalized': normalized, 'candidate_method_environment_json': json.dumps(method_environment, sort_keys=True, separators=(',', ':')), 'candidate_artifact_paths_json': json.dumps(artifact_paths), 'input_artifact_ids_json': json.dumps(sorted(input_artifact_ids)), 'candidate_path': str(candidate.resolve())}

def row_set(*, method: str, mode: str, alpha: float, candidate_id: str, candidate: dict[str, object], support: dict[str, object], support_variant: str, keep: np.ndarray, seed: int) -> list[dict[str, object]]:
    n_full = int(support['n_spots'])
    reference_keep = keep[:n_full]
    query_keep = keep[n_full:]
    n_reference = int(reference_keep.sum())
    if n_reference != int(query_keep.sum()):
        raise ValueError('reference/query reduced supports differ')
    embedding = candidate['embedding'][keep]
    batches = candidate['batch_labels'][keep]
    domains = candidate['domain_labels'][keep]
    paired_spot_ids = np.asarray(support['spot_ids'])[reference_keep]
    expression_correlation = None
    if method == 'scVI':
        expression_correlation = paired_expression_correlation(candidate['normalized'][keep], n_reference, paired_spot_ids)
    removed_pair_count = int(support['query_zero_count']) if support_variant == PAIRED_SUPPORT else 0
    removed_ids = support['query_zero_ids'] if removed_pair_count else []
    rows = []
    for representation, primary, values in representations(method, embedding, seed):
        row = metric_row(method=method, representation=representation, primary=primary, embedding=values, batch_labels=batches, domain_labels=domains, n_reference=n_reference, expression_correlation=expression_correlation, seed=seed)
        row.update({'configuration_id': f"{candidate_id.split('__', 1)[0]}__atomic_metrics", 'study_id': '04', 'legacy_study_id': '08', 'mode': mode, 'alpha': f'{alpha:.2f}', 'alpha_stratum': 'interpolation' if alpha <= 1.0 else 'extrapolation', 'support_variant': support_variant, 'primary_support': support_variant == ALL_SPOTS, 'analysis_role': 'primary' if support_variant == ALL_SPOTS and primary else 'representation_sensitivity' if support_variant == ALL_SPOTS else 'paired_support_sensitivity' if primary else 'representation_and_paired_support_sensitivity', 'removed_paired_spots': removed_pair_count, 'removed_named_pairs': json.dumps(removed_ids, ensure_ascii=True), 'fixed_panel_zero_query_rows': int(support['query_zero_count']), 'fixed_panel_zero_query_spot_ids': json.dumps(support['query_zero_ids'], ensure_ascii=True), 'fixed_panel_zero_reference_rows': int(support['reference_zero_count']), 'fixed_panel_zero_reference_spot_ids': json.dumps(support['reference_zero_ids'], ensure_ascii=True), 'n_reference': n_reference, 'n_query': n_reference, 'method_candidate_id': candidate_id, 'public_seed': seed, 'candidate_method_environment_json': candidate['candidate_method_environment_json'], 'candidate_artifact_paths_json': candidate['candidate_artifact_paths_json'], 'input_artifact_ids_json': candidate['input_artifact_ids_json']})
        rows.append(row)
    return rows

def expected_keys(config: dict, common_cells) -> pd.DataFrame:
    return support.expected_key_frame(modes=config['simulation']['modes'], alpha_levels=config['simulation']['alpha_levels'], common_support_cells=common_cells)

def validate_atomic_table(table: pd.DataFrame, config: dict, zero_query_ids: dict) -> None:
    common_cells = {key for key, barcodes in zero_query_ids.items() if barcodes}
    support.assert_exact_key_matrix(table, expected_keys(config, common_cells), 'fresh atomic table')
    core = table[list(CORE_METRIC_COLUMNS)].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(core.to_numpy(dtype=float)).all():
        raise ValueError('a required core atomic metric is non-finite')
    correlation = pd.to_numeric(table['paired_expression_correlation'], errors='coerce')
    expected_correlation = table['method'].eq('scVI')
    if not np.array_equal(correlation.notna().to_numpy(), expected_correlation.to_numpy()):
        raise ValueError('paired-expression correlation applicability does not match the scVI rows')
    if not np.isfinite(correlation[expected_correlation].to_numpy(dtype=float)).all():
        raise ValueError('a named paired-expression correlation is non-finite')
    bounded = {'batch_asw': (-1.0, 1.0), 'batch_mixing_entropy_k30': (0.0, 1.0), 'paired_retrieval_top1': (0.0, 1.0), 'domain_asw': (-1.0, 1.0), 'domain_purity_k30': (0.0, 1.0), 'paired_expression_correlation': (-1.0, 1.0)}
    tolerance = 1e-12
    for metric, (lower, upper) in bounded.items():
        values = pd.to_numeric(table[metric], errors='coerce').dropna().to_numpy(float)
        if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
            raise ValueError(f'{metric} lies outside [{lower}, {upper}]')
    for metric in ('centroid_distance', 'covariance_distance', 'mmd2_rbf_biased'):
        values = pd.to_numeric(table[metric], errors='coerce').to_numpy(float)
        if np.any(values < -tolerance):
            raise ValueError(f'{metric} contains a negative value')
    median_rank = pd.to_numeric(table['paired_retrieval_median_rank'], errors='coerce')
    n_reference = pd.to_numeric(table['n_reference'], errors='raise')
    if ((median_rank < 1) | (median_rank > n_reference)).any():
        raise ValueError('paired retrieval median rank is outside named reference support')
    bandwidth = pd.to_numeric(table['mmd_bandwidth'], errors='coerce').to_numpy(float)
    if not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0):
        raise ValueError('MMD bandwidth must be finite and positive')
    expected_dimensions = {'pca20_randomized_seed42': 20, 'pca10_randomized_seed42_sensitivity': 10, 'native_10d_topics': 10, 'native_10d_latent': 10}
    for row in table.itertuples(index=False):
        if int(row.representation_dimensions) != expected_dimensions[row.representation]:
            raise ValueError('representation dimension changed')
        is_primary_representation = row.representation in {'pca20_randomized_seed42', 'native_10d_topics', 'native_10d_latent'}
        if bool(row.primary_representation) != is_primary_representation:
            raise ValueError('primary representation flag changed')
        is_all_spots = row.support_variant == ALL_SPOTS
        if bool(row.primary_support) != is_all_spots:
            raise ValueError('primary support flag changed')
        zero_ids = zero_query_ids[row.mode, alpha_text(row.alpha)]
        expected_removed_ids = [] if is_all_spots else zero_ids
        expected_removed = len(expected_removed_ids)
        if int(row.removed_paired_spots) != expected_removed:
            raise ValueError('removed paired-spot count does not match support variant')
        if json.loads(row.removed_named_pairs) != expected_removed_ids:
            raise ValueError('removed named pairs do not match the support contract')
        if int(row.fixed_panel_zero_query_rows) != len(zero_ids):
            raise ValueError('fixed-panel query-zero count changed')
        if json.loads(row.fixed_panel_zero_query_spot_ids) != zero_ids:
            raise ValueError('fixed-panel query-zero IDs changed')
        if int(row.fixed_panel_zero_reference_rows) != 0:
            raise ValueError('reference fixed panel contains a raw-zero row')
        if json.loads(row.fixed_panel_zero_reference_spot_ids) != []:
            raise ValueError('reference fixed-panel zero IDs must be empty')
        expected_n = 3611 - expected_removed
        if int(row.n_reference) != expected_n or int(row.n_query) != expected_n:
            raise ValueError('paired named support count changed')

def build_primary_summary(table: pd.DataFrame) -> pd.DataFrame:
    primary = table[(table['support_variant'] == ALL_SPOTS) & table['primary_representation']]
    summary = primary.groupby(['method', 'mode', 'alpha_stratum', 'representation'], dropna=False)[list(METRIC_COLUMNS)].mean().reset_index()
    if len(summary) != 12:
        raise ValueError(f'primary summary has {len(summary)} rows; expected 12')
    return summary

def build_common_support_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    sensitivity = table[table['support_variant'] == PAIRED_SUPPORT].merge(table[table['support_variant'] == ALL_SPOTS], on=['method', 'mode', 'alpha', 'representation'], how='left', suffixes=('_paired_support', '_all_spots'), validate='one_to_one')
    columns = ['method', 'mode', 'alpha', 'representation', 'removed_paired_spots_paired_support',
               'removed_named_pairs_paired_support', 'n_reference_paired_support', 'n_query_paired_support']
    result = sensitivity[columns].rename(columns=lambda column: column.removesuffix('_paired_support')).copy()
    for metric in METRIC_COLUMNS:
        result[f'{metric}_paired_support'] = sensitivity[f'{metric}_paired_support']
        result[f'{metric}_all_spots'] = sensitivity[f'{metric}_all_spots']
        result[f'delta_{metric}_paired_minus_all'] = sensitivity[f'{metric}_paired_support'] - sensitivity[f'{metric}_all_spots']
    return result.sort_values(['method', 'mode', 'alpha', 'representation']).reset_index(drop=True)

def build_graphst_pca_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    graphst = table[table['method'] == 'GraphST']
    pca20 = graphst[graphst['representation'] == 'pca20_randomized_seed42']
    pca10 = graphst[graphst['representation'] == 'pca10_randomized_seed42_sensitivity']
    sensitivity = pca20.merge(pca10, on=['method', 'mode', 'alpha', 'support_variant'], validate='one_to_one', suffixes=('_pca20', '_pca10'))
    result = sensitivity[['method', 'mode', 'alpha', 'support_variant', 'removed_paired_spots_pca20', 'n_reference_pca20', 'n_query_pca20']].rename(columns={'removed_paired_spots_pca20': 'removed_paired_spots', 'n_reference_pca20': 'n_reference', 'n_query_pca20': 'n_query'})
    for metric in METRIC_COLUMNS:
        result[f'{metric}_pca20'] = sensitivity[f'{metric}_pca20']
        result[f'{metric}_pca10'] = sensitivity[f'{metric}_pca10']
        result[f'delta_{metric}_pca10_minus_pca20'] = sensitivity[f'{metric}_pca10'] - sensitivity[f'{metric}_pca20']
    return result.sort_values(['mode', 'alpha', 'support_variant']).reset_index(drop=True)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_ROOT / 'config.yaml')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    run_dir = args.run_dir.resolve() if args.run_dir else config['_output_dir']
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / 'metrics'
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite metric output: {output_dir}')
    panel_path = run_dir / 'panel' / 'panel.csv'
    panel_provenance_path = run_dir / 'panel' / 'provenance.json'
    simulation_manifest_path = run_dir / 'simulations' / 'manifest.csv'
    simulation_provenance_path = run_dir / 'simulations' / 'provenance.json'
    method_manifest_path = run_dir / 'method_run_manifest.csv'
    for required in (panel_path, panel_provenance_path, simulation_manifest_path, simulation_provenance_path, method_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f'required scoring-lineage artifact is missing: {required}')
    genes = load_panel(panel_path)
    if len(genes) != int(config['fixed_panel']['n_genes']):
        raise ValueError('fixed-panel size differs from the config')
    panel_provenance = json.loads(panel_provenance_path.read_text(encoding='utf-8'))
    if panel_provenance.get('configuration_id') != f"{config['configuration_id']}__fixed_panel":
        raise ValueError('panel provenance configuration ID changed')
    if int(panel_provenance.get('public_seed', -1)) != int(config['public_seed']):
        raise ValueError('panel provenance seed changed')
    simulation_manifest = pd.read_csv(simulation_manifest_path)
    simulation_provenance = json.loads(simulation_provenance_path.read_text(encoding='utf-8'))
    if simulation_provenance.get('configuration_id') != config['configuration_id']:
        raise ValueError('simulation provenance configuration ID changed')
    if int(simulation_provenance.get('public_seed', -1)) != int(config['public_seed']):
        raise ValueError('simulation provenance seed changed')
    expected_simulation_cells = {(str(mode), alpha_text(alpha)) for mode in config['simulation']['modes'] for alpha in config['simulation']['alpha_levels']}
    observed_simulation_cells = {(str(row.mode), alpha_text(row.alpha)) for row in simulation_manifest.itertuples(index=False)}
    if len(simulation_manifest) != 14 or observed_simulation_cells != expected_simulation_cells:
        raise ValueError('simulation manifest does not contain the exact 14-cell ladder')
    if len(observed_simulation_cells) != len(simulation_manifest):
        raise ValueError('simulation manifest contains duplicate cells')
    method_manifest = pd.read_csv(method_manifest_path)
    expected_method_cells = {(method, str(mode), alpha_text(alpha)) for method in workflow.METHOD_ARTIFACTS for mode in config['simulation']['modes'] for alpha in config['simulation']['alpha_levels'] if float(alpha) != 0.0}
    observed_method_cells = {(str(row.method), str(row.mode), alpha_text(row.alpha)) for row in method_manifest.itertuples(index=False)}
    if len(method_manifest) != 36 or observed_method_cells != expected_method_cells:
        raise ValueError('method run manifest does not contain the exact 36-job matrix')
    if len(observed_method_cells) != len(method_manifest):
        raise ValueError('method run manifest contains duplicate jobs')
    if not set(method_manifest['status']).issubset({'complete', 'skipped_verified'}):
        raise ValueError('method run manifest contains a failed or incomplete job')
    seed = int(config['public_seed'])
    if seed != 42:
        raise ValueError('the declared Study 04 metric and PCA seed must remain 42')
    rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    provenance_inputs: list[dict[str, str]] = [{'role': 'simulation_manifest', 'path': str(simulation_manifest_path.resolve())}, {'role': 'simulation_provenance', 'path': str(simulation_provenance_path.resolve())}, {'role': 'fixed_panel', 'path': str(panel_path.resolve())}, {'role': 'fixed_panel_provenance', 'path': str(panel_provenance_path.resolve())}, {'role': 'method_run_manifest', 'path': str(method_manifest_path.resolve())}]
    for row in simulation_manifest.itertuples(index=False):
        path = run_dir / row.file
        provenance_inputs.append({'role': 'simulation_h5ad', 'path': str(path.resolve())})
    zero_query_ids: dict[tuple[str, str], list[str]] = {}
    for mode in config['simulation']['modes']:
        reference_path = run_dir / 'simulations' / mode / 'alpha_0.00.h5ad'
        for alpha_value in config['simulation']['alpha_levels']:
            alpha = float(alpha_value)
            if alpha == 0.0:
                continue
            alpha_key = alpha_text(alpha)
            query_path = run_dir / 'simulations' / mode / f'alpha_{alpha_key}.h5ad'
            support = load_support(reference_path, query_path, genes, mode=mode, alpha=alpha)
            zero_query_ids[mode, alpha_key] = support['query_zero_ids']
            support_rows.append({'mode': mode, 'alpha': alpha_key, 'reference_path': support['reference_path'], 'query_path': support['query_path'], 'n_spots_per_batch': int(support['n_spots']), 'fixed_panel_zero_reference_rows': int(support['reference_zero_count']), 'fixed_panel_zero_reference_spot_ids': json.dumps(support['reference_zero_ids'], ensure_ascii=True), 'fixed_panel_zero_query_rows': int(support['query_zero_count']), 'fixed_panel_zero_query_spot_ids': json.dumps(support['query_zero_ids'], ensure_ascii=True), 'all_spots_removed_paired_spots': 0, 'paired_support_removed_paired_spots': int(support['query_zero_count']), 'paired_support_removed_named_pairs': json.dumps(support['query_zero_ids'], ensure_ascii=True), 'paired_support_n_reference': int(support['n_spots']) - int(support['query_zero_count']), 'paired_support_n_query': int(support['n_spots']) - int(support['query_zero_count'])})
            for method in workflow.METHOD_ARTIFACTS:
                candidate_id = f"{config['configuration_id']}__{method}__{mode}__alpha_{alpha_key}"
                manifest_match = method_manifest[(method_manifest['method'] == method) & (method_manifest['mode'] == mode) & (method_manifest['alpha'].map(alpha_text) == alpha_key)]
                if len(manifest_match) != 1:
                    raise ValueError(f'missing unique method manifest row: {candidate_id}')
                manifest_row = manifest_match.iloc[0]
                if str(manifest_row['candidate_id']) != candidate_id:
                    raise ValueError('method manifest candidate ID changed')
                candidate_path = run_dir / 'methods' / method / mode / f'alpha_{alpha_key}'
                if (run_dir / manifest_row['output_dir']).resolve() != candidate_path.resolve():
                    raise ValueError('method manifest output path changed')
                if pd.notna(manifest_row.get('log')):
                    log_path = run_dir / str(manifest_row['log'])
                    provenance_inputs.append({'role': 'method_log', 'path': str(log_path.resolve())})
                candidate = load_candidate(candidate_path, method, candidate_id, config, support)
                n_spots = int(support['n_spots'])
                rows.extend(row_set(method=method, mode=mode, alpha=alpha, candidate_id=candidate_id, candidate=candidate, support=support, support_variant=ALL_SPOTS, keep=np.ones(2 * n_spots, dtype=bool), seed=seed))
                if support['query_zero_count']:
                    remove = np.asarray(support['remove_pairs'], dtype=bool)
                    paired_keep = np.concatenate([~remove, ~remove])
                    rows.extend(row_set(method=method, mode=mode, alpha=alpha, candidate_id=candidate_id, candidate=candidate, support=support, support_variant=PAIRED_SUPPORT, keep=paired_keep, seed=seed))
                artifact_paths = json.loads(candidate['candidate_artifact_paths_json'])
                for relative in artifact_paths:
                    provenance_inputs.append({'role': f'{method}_candidate_artifact', 'path': str((candidate_path / relative).resolve())})
            print(f'scored {mode}/alpha_{alpha_key}', flush=True)
    table = pd.DataFrame(rows).sort_values(['method', 'mode', 'alpha', 'support_variant', 'representation']).reset_index(drop=True)
    validate_atomic_table(table, config, zero_query_ids)
    summary = build_primary_summary(table)
    sensitivity_deltas = build_common_support_sensitivity(table)
    graphst_pca_sensitivity = build_graphst_pca_sensitivity(table)
    support_audit = pd.DataFrame(support_rows).sort_values(['mode', 'alpha']).reset_index(drop=True)
    if len(support_audit) != 12:
        raise ValueError('support audit must contain exactly 12 simulation cells')
    table_path = output_dir / 'atomic_metrics.csv'
    summary_path = output_dir / 'primary_summary.csv'
    sensitivity_path = output_dir / 'common_support_sensitivity.csv'
    graphst_pca_path = output_dir / 'graphst_pca20_vs_pca10_sensitivity.csv'
    support_audit_path = output_dir / 'support_audit.csv'
    output_dir.mkdir(parents=True)
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    sensitivity_deltas.to_csv(sensitivity_path, index=False)
    graphst_pca_sensitivity.to_csv(graphst_pca_path, index=False)
    support_audit.to_csv(support_audit_path, index=False)
    summary_json = {'schema_version': 1, 'configuration_id': f"{config['configuration_id']}__atomic_metrics", 'study_id': '04', 'legacy_study_id': '08', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(table), 'all_spots_rows': int((table['support_variant'] == ALL_SPOTS).sum()), 'common_support_rows': int((table['support_variant'] == PAIRED_SUPPORT).sum()), 'primary_summary_rows': len(summary), 'graphst_pca20_vs_pca10_sensitivity_rows': len(graphst_pca_sensitivity), 'support_audit_rows': len(support_audit), 'composite_score': 'prohibited_not_computed', 'winner_ranking': 'prohibited_not_computed', 'active_manuscript_claim': None, 'exit_2_guard_triggered': False, 'alpha_strata': {'interpolation': 'alpha <= 1.00', 'extrapolation': 'alpha > 1.00'}, 'decision': 'fresh_atomic_metrics_complete'}
    summary_json_path = output_dir / 'summary.json'
    summary_json_path.write_text(json.dumps(summary_json, indent=2) + '\n', encoding='utf-8')
    output_paths = [table_path, summary_path, sensitivity_path, graphst_pca_path, support_audit_path, summary_json_path]
    score_source = Path(__file__).resolve()
    run_source = STUDY_ROOT / 'run.py'
    config_path = config['_config_path']
    provenance = {'schema_version': 1, 'configuration_id': f"{config['configuration_id']}__atomic_metrics", 'study_id': '04', 'legacy_study_id': '08', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'feast': workflow.feast_identity(config), 'public_seed': seed, 'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'anndata': ad.__version__, 'sklearn': sklearn.__version__}, 'sources': [{'role': 'scorer', 'path': str(score_source)}, {'role': 'workflow_dispatcher', 'path': str(run_source.resolve())}, {'role': 'configuration', 'path': str(config_path)}], 'inputs': provenance_inputs, 'outputs': [{'path': str(path.resolve())} for path in output_paths], 'diagnostics': {'status': 'completed', 'rows': len(table), 'validated_method_candidates': 36, 'simulation_h5ad': 14, 'exact_metric_key_rows': len(table), 'exact_named_support_cells': sum(bool(ids) for ids in zero_query_ids.values()), 'reference_fixed_panel_zero_rows': 0, 'graphst_pca_sensitivity_rows': len(graphst_pca_sensitivity), 'active_manuscript_claim': None, 'exit_2_guard_triggered': False, 'composite_score': 'prohibited_not_computed', 'winner_ranking': 'prohibited_not_computed'}}
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {len(table)} rows to {table_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
