"""Independent validation for Study 04B prepared inputs, candidates, and scores."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import workflow
STUDY_ROOT = Path(__file__).resolve().parent

def v_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale

def v_center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - values.mean(axis=0)

def v_neighbors(values: np.ndarray, k: int) -> np.ndarray:
    raw = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values, return_distance=False)
    rows = []
    for index, neighbors in enumerate(raw):
        keep = neighbors[neighbors != index]
        if len(keep) < k:
            raise RuntimeError('validator neighbor support is incomplete')
        rows.append(keep[:k])
    return np.vstack(rows)

def v_entropy(batch: np.ndarray, neighbors: np.ndarray) -> float:
    batch = np.asarray(batch).astype(str)
    unique = np.unique(batch)
    if len(unique) != 2:
        raise ValueError('validator entropy requires two batches')
    p = (batch[neighbors] == unique[1]).mean(axis=1)
    value = np.zeros(len(p), dtype=np.float64)
    for term in (p, 1.0 - p):
        nonzero = term > 0
        value[nonzero] -= term[nonzero] * np.log(term[nonzero])
    return float(np.mean(value / np.log(2.0)))

def v_mmd(left: np.ndarray, right: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) > 1000:
        x = x[rng.choice(len(x), 1000, replace=False)]
    if len(y) > 1000:
        y = y[rng.choice(len(y), 1000, replace=False)]
    distance = pairwise_distances(np.vstack([x, y]))
    positive = distance[distance > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)
    value = rbf_kernel(x, x, gamma=gamma).mean() + rbf_kernel(y, y, gamma=gamma).mean() - 2 * rbf_kernel(x, y, gamma=gamma).mean()
    return float(max(value, 0.0))

def v_shared_layers(batch: np.ndarray, layers: np.ndarray) -> list[str]:
    batch = np.asarray(batch).astype(str)
    layers = np.asarray(layers).astype(str)
    result = [layer for layer in sorted(np.unique(layers)) if set(batch[layers == layer]) == {'ref', 'query'}]
    if len(result) < 2:
        raise ValueError('validator found fewer than two shared layers')
    return result

def v_knn_jaccard(left: np.ndarray, right: np.ndarray, k: int) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError('validator KNN overlap support differs')
    a = v_neighbors(v_center(left), k)
    b = v_neighbors(v_center(right), k)
    values = []
    for x, y in zip(a, b, strict=True):
        intersection = len(set(x.tolist()) & set(y.tolist()))
        values.append(intersection / (2 * k - intersection))
    return float(np.mean(values))

def validator_full_metric_row(values: np.ndarray, *, baseline: np.ndarray, spatial: np.ndarray, batch: np.ndarray, layers: np.ndarray, k: int, seed: int) -> dict[str, float]:
    z = v_standardize(values)
    batch = np.asarray(batch).astype(str)
    layers = np.asarray(layers).astype(str)
    conditional = []
    shared = v_shared_layers(batch, layers)
    for layer in shared:
        keep = layers == layer
        layer_values = z[keep]
        layer_batch = batch[keep]
        layer_neighbors = v_neighbors(layer_values, min(k, len(layer_values) - 1))
        conditional.append([float(silhouette_score(layer_values, layer_batch)), v_entropy(layer_batch, layer_neighbors), v_mmd(layer_values[layer_batch == 'ref'], layer_values[layer_batch == 'query'], seed)])
    conditional = np.asarray(conditional)
    neighbors = v_neighbors(z, k)
    purity = float(np.mean(layers[neighbors] == layers[:, None]))
    ref = batch == 'ref'
    query = batch == 'query'
    classifier = KNeighborsClassifier(n_neighbors=k).fit(z[ref], layers[ref])
    predicted = classifier.predict(z[query])
    preservation = np.mean([v_knn_jaccard(baseline[batch == label], values[batch == label], k) for label in ('ref', 'query')])
    spatial_overlap = np.mean([v_knn_jaccard(spatial[batch == label], values[batch == label], min(6, k)) for label in ('ref', 'query')])
    return {'conditional_batch_asw_macro': float(conditional[:, 0].mean()), 'conditional_batch_entropy_macro': float(conditional[:, 1].mean()), 'conditional_mmd2_macro': float(conditional[:, 2].mean()), 'shared_layer_count': int(len(shared)), 'layer_asw': float(silhouette_score(z, layers)), 'layer_purity_k': purity, 'cross_layer_neighbor_contamination_k': float(1.0 - purity), 'reference_to_query_layer_macro_f1': float(f1_score(layers[query], predicted, labels=shared, average='macro')), 'within_section_knn_preservation_k': float(preservation), 'spatial_embedding_knn_overlap_k': float(spatial_overlap)}

def v_distance_spearman(left: np.ndarray, right: np.ndarray, *, max_pairs: int, seed: int) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError('validator distance stability support differs')
    n = left.shape[0]
    total = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    if total <= max_pairs:
        row, col = np.triu_indices(n, 1)
    else:
        pairs = set()
        while len(pairs) < max_pairs:
            a = rng.integers(0, n, size=max_pairs)
            b = rng.integers(0, n, size=max_pairs)
            for x, y in zip(a, b, strict=True):
                if x != y:
                    pairs.add((min(int(x), int(y)), max(int(x), int(y))))
                if len(pairs) == max_pairs:
                    break
        array = np.asarray(sorted(pairs), dtype=int)
        row, col = (array[:, 0], array[:, 1])
    x = v_center(left)
    y = v_center(right)
    dx = np.linalg.norm(x[row] - x[col], axis=1)
    dy = np.linalg.norm(y[row] - y[col], axis=1)
    return float(spearmanr(dx, dy).statistic)

def v_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = v_center(left)
    y = v_center(right)
    numerator = np.linalg.norm(x.T @ y, ord='fro') ** 2
    denominator = np.linalg.norm(x.T @ x, ord='fro') * np.linalg.norm(y.T @ y, ord='fro')
    if denominator <= 0:
        raise ValueError('validator CKA denominator is zero')
    return float(numerator / denominator)

def v_stability(values: np.ndarray, anchor: np.ndarray, n_reference: int, config: dict) -> dict[str, float]:
    query = values[n_reference:]
    anchor_query = anchor[n_reference:]
    k = int(config['analysis']['n_neighbors'])
    seed = int(config['analysis']['scoring_seed'])
    return {'query_stability_knn_jaccard_k': v_knn_jaccard(anchor_query, query, k), 'query_stability_distance_spearman': v_distance_spearman(anchor_query, query, max_pairs=int(config['analysis']['pairwise_distance_sample']), seed=seed), 'query_stability_linear_cka': v_cka(anchor_query, query)}

def _baseline(output_root: Path, condition: str) -> dict[str, object]:
    path = output_root / 'baselines' / f'{condition}.npz'
    with np.load(path, allow_pickle=True) as payload:
        return {'embedding': np.asarray(payload['embedding'], dtype=np.float64), 'batch': np.asarray(payload['batch_labels']).astype(str), 'layers': np.asarray(payload['layer_labels']).astype(str), 'spatial': np.asarray(payload['spatial'], dtype=np.float64), 'n_reference': int(payload['n_reference'])}

def _representations(method: str, embedding: np.ndarray, seed: int):
    if method == 'GraphST':
        return [('pca10_randomized_seed42', PCA(n_components=10, svd_solver='randomized', random_state=seed).fit_transform(embedding)), ('pca20_randomized_seed42_sensitivity', PCA(n_components=20, svd_solver='randomized', random_state=seed).fit_transform(embedding))]
    name = 'native_10d_topics' if method == 'STAMP' else 'native_10d_latent'
    return [(name, np.asarray(embedding, dtype=np.float64))]

def validate_prepared(config: dict, output_root: Path) -> dict[str, object]:
    provenance_path = output_root / 'prepare_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance.get('configuration_id') != config['configuration_id']:
        raise ValueError('prepare provenance configuration ID mismatch')
    for collection in ('inputs', 'sources', 'outputs'):
        records = provenance.get(collection, [])
        if not records:
            raise ValueError(f'prepare provenance lacks {collection}')
        for record in records:
            path = Path(record['path'])
            if not path.is_file():
                raise ValueError(f'prepare provenance path is missing: {path}')
    qualification = json.loads((output_root / 'inputs' / 'qualification.json').read_text(encoding='utf-8'))
    if qualification.get('status') != 'passed':
        raise ValueError('prepared input qualification did not pass')
    manifest = workflow.read_condition_manifest(output_root)
    if set(manifest.condition) != set(workflow.condition_lookup(config)) or len(manifest) != 5:
        raise ValueError('prepared condition matrix changed')
    genes = workflow.read_panel(output_root)
    if len(genes) != int(config['fixed_panel']['n_genes']):
        raise ValueError('prepared fixed panel size changed')
    return {'status': 'passed', 'conditions': len(manifest), 'fixed_panel_genes': len(genes), 'provenance_outputs': len(provenance['outputs'])}

def validate_candidates(config: dict, output_root: Path, *, allow_incomplete: bool) -> dict[str, object]:
    complete = []
    missing = []
    for condition, method, seed in workflow.production_cells(config):
        candidate = workflow.candidate_path(output_root, condition, method, seed)
        if not candidate.exists():
            missing.append((condition, method, seed))
            continue
        valid, reason = workflow.candidate_is_valid(candidate, config=config, output_root=output_root, condition=condition, method=method, seed=seed)
        if not valid:
            raise ValueError(f'invalid candidate {method}/{condition}/seed_{seed}: {reason}')
        complete.append((condition, method, seed))
    if missing and (not allow_incomplete):
        raise ValueError(f'production matrix is incomplete: {len(missing)} missing candidates')
    if allow_incomplete and (not complete):
        raise ValueError('no completed candidate is available to validate')
    return {'status': 'passed', 'complete_candidates': len(complete), 'missing_candidates': len(missing), 'allow_incomplete': allow_incomplete}

def _load_embedding(candidate: Path) -> np.ndarray:
    with np.load(candidate / 'embeddings.npz', allow_pickle=True) as payload:
        return np.asarray(payload['embedding'], dtype=np.float64)

def reconstruct_metrics(config: dict, output_root: Path) -> pd.DataFrame:
    scoring_seed = int(config['analysis']['scoring_seed'])
    k = int(config['analysis']['n_neighbors'])
    conditions = workflow.condition_lookup(config)
    baselines = {name: _baseline(output_root, name) for name in conditions}
    values = {}
    for condition, method, seed in workflow.production_cells(config):
        embedding = _load_embedding(workflow.candidate_path(output_root, condition, method, seed))
        for representation, representation_values in _representations(method, embedding, scoring_seed):
            values[condition, method, seed, representation] = representation_values
    rows = []
    for condition in conditions:
        baseline = baselines[condition]
        embedding = baseline['embedding']
        row = {'condition': condition, 'method': 'Uncorrected', 'method_seed': -1, 'representation': 'fixed_panel_log1p_pca10_seed42'}
        row.update(validator_full_metric_row(embedding, baseline=embedding, spatial=baseline['spatial'], batch=baseline['batch'], layers=baseline['layers'], k=k, seed=scoring_seed))
        row.update(v_stability(embedding, baselines['sim_0.00']['embedding'], int(baseline['n_reference']), config))
        rows.append(row)
    for condition, method, seed in workflow.production_cells(config):
        baseline = baselines[condition]
        for representation, embedding in _representations(method, _load_embedding(workflow.candidate_path(output_root, condition, method, seed)), scoring_seed):
            row = {'condition': condition, 'method': method, 'method_seed': seed, 'representation': representation}
            row.update(validator_full_metric_row(embedding, baseline=baseline['embedding'], spatial=baseline['spatial'], batch=baseline['batch'], layers=baseline['layers'], k=k, seed=scoring_seed))
            row.update(v_stability(embedding, values['sim_0.00', method, seed, representation], int(baseline['n_reference']), config))
            rows.append(row)
    return pd.DataFrame(rows)

def validate_scores(config: dict, output_root: Path) -> dict[str, object]:
    metrics_root = output_root / 'metrics'
    table_path = metrics_root / 'atomic_metrics.csv'
    reported = pd.read_csv(table_path)
    reconstructed = reconstruct_metrics(config, output_root)
    keys = ['condition', 'method', 'method_seed', 'representation']
    if reported.duplicated(keys).any() or reconstructed.duplicated(keys).any():
        raise ValueError('metric key rows are not unique')
    if set(map(tuple, reported[keys].to_numpy())) != set(map(tuple, reconstructed[keys].to_numpy())):
        raise ValueError('reported and reconstructed metric keys differ')
    metric_columns = [column for column in reconstructed.columns if column not in keys]
    merged = reported.merge(reconstructed, on=keys, how='inner', suffixes=('_reported', '_reconstructed'), validate='one_to_one')
    max_error = 0.0
    for column in metric_columns:
        observed = merged[f'{column}_reported'].to_numpy(dtype=float)
        expected = merged[f'{column}_reconstructed'].to_numpy(dtype=float)
        error = float(np.nanmax(np.abs(observed - expected)))
        max_error = max(max_error, error)
        if not np.allclose(observed, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f'independent metric reconstruction differs: {column}')
    prohibited = {'paired_retrieval_top1', 'paired_retrieval_median_rank', 'composite_score'}
    if prohibited & set(reported):
        raise ValueError('reported table contains a prohibited metric')
    provenance = json.loads((metrics_root / 'provenance.json').read_text(encoding='utf-8'))
    for collection in ('inputs', 'sources', 'outputs'):
        for record in provenance.get(collection, []):
            path = Path(record['path'])
            if not path.is_file():
                raise ValueError(f'metric provenance path is missing: {path}')
    return {'status': 'passed', 'reported_rows': len(reported), 'independently_reconstructed_rows': len(reconstructed), 'metric_columns_reconstructed': len(metric_columns), 'maximum_absolute_error': max_error}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_ROOT / 'config.yaml')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--stage', choices=('prepared', 'candidates', 'complete'), default='complete')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    output_root = args.run_dir.resolve() if args.run_dir else config['_output_dir']
    report = {'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage, 'prepared': validate_prepared(config, output_root)}
    if args.stage in {'candidates', 'complete'}:
        report['candidates'] = validate_candidates(config, output_root, allow_incomplete=args.allow_incomplete if args.stage == 'candidates' else False)
    if args.stage == 'complete':
        report['scores'] = validate_scores(config, output_root)
        report['decision'] = 'complete_atomic_evidence_author_review_required'
        report_path = args.report or output_root / 'validation' / 'summary.json'
    elif args.stage == 'candidates':
        report_path = args.report or output_root / 'candidate_validation.json'
    else:
        report_path = args.report or output_root / 'prepared_validation.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(f'refusing to overwrite validation report: {report_path}')
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
