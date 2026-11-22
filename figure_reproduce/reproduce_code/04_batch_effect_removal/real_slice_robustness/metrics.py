"""Prespecified non-composite Study 04B metric formulas."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

def standardize(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    scale = result.std(axis=0)
    scale[scale == 0] = 1.0
    return (result - result.mean(axis=0)) / scale

def center(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result - result.mean(axis=0)

def neighbors_without_self(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[0] <= k:
        raise ValueError(f'{values.shape[0]} rows cannot support k={k}')
    raw = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values, return_distance=False)
    rows = []
    for index, neighbors in enumerate(raw):
        without_self = neighbors[neighbors != index]
        if without_self.size < k:
            raise RuntimeError('nearest-neighbor result lacks non-self rows')
        rows.append(without_self[:k])
    return np.vstack(rows)

def local_batch_entropy(batch: np.ndarray, neighbors: np.ndarray) -> float:
    labels = np.asarray(batch).astype(str)
    unique = np.unique(labels)
    if len(unique) != 2:
        raise ValueError('batch entropy requires exactly two batches')
    probability = (labels[neighbors] == unique[1]).mean(axis=1)
    entropy = np.zeros(probability.size, dtype=np.float64)
    for term in (probability, 1.0 - probability):
        nonzero = term > 0
        entropy[nonzero] -= term[nonzero] * np.log(term[nonzero])
    return float(np.mean(entropy / np.log(2.0)))

def mmd2_rbf_biased(left: np.ndarray, right: np.ndarray, *, seed: int, max_per_batch: int=1000) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) > max_per_batch:
        x = x[rng.choice(len(x), max_per_batch, replace=False)]
    if len(y) > max_per_batch:
        y = y[rng.choice(len(y), max_per_batch, replace=False)]
    pooled = np.vstack([x, y])
    distances = pairwise_distances(pooled)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)
    value = rbf_kernel(x, x, gamma=gamma).mean() + rbf_kernel(y, y, gamma=gamma).mean() - 2.0 * rbf_kernel(x, y, gamma=gamma).mean()
    return float(max(value, 0.0))

def _shared_layers(batch: np.ndarray, layers: np.ndarray) -> list[str]:
    batch = np.asarray(batch).astype(str)
    layers = np.asarray(layers).astype(str)
    result = []
    for layer in sorted(np.unique(layers)):
        present = set(batch[layers == layer])
        if present == {'ref', 'query'}:
            result.append(layer)
    if len(result) < 2:
        raise ValueError('fewer than two shared layers remain')
    return result

def conditional_batch_metrics(values: np.ndarray, batch: np.ndarray, layers: np.ndarray, *, k: int, seed: int) -> dict[str, float]:
    values = standardize(values)
    batch = np.asarray(batch).astype(str)
    layers = np.asarray(layers).astype(str)
    rows = []
    for layer in _shared_layers(batch, layers):
        keep = layers == layer
        layer_values = values[keep]
        layer_batch = batch[keep]
        layer_k = min(k, len(layer_values) - 1)
        neighbors = neighbors_without_self(layer_values, layer_k)
        left = layer_values[layer_batch == 'ref']
        right = layer_values[layer_batch == 'query']
        rows.append((float(silhouette_score(layer_values, layer_batch)), local_batch_entropy(layer_batch, neighbors), mmd2_rbf_biased(left, right, seed=seed)))
    array = np.asarray(rows, dtype=np.float64)
    return {'conditional_batch_asw_macro': float(array[:, 0].mean()), 'conditional_batch_entropy_macro': float(array[:, 1].mean()), 'conditional_mmd2_macro': float(array[:, 2].mean()), 'shared_layer_count': int(len(rows))}

def biological_metrics(values: np.ndarray, batch: np.ndarray, layers: np.ndarray, *, k: int) -> dict[str, float]:
    values = standardize(values)
    batch = np.asarray(batch).astype(str)
    layers = np.asarray(layers).astype(str)
    neighbors = neighbors_without_self(values, k)
    purity = float(np.mean(layers[neighbors] == layers[:, None]))
    ref = batch == 'ref'
    query = batch == 'query'
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(values[ref], layers[ref])
    predicted = classifier.predict(values[query])
    shared = _shared_layers(batch, layers)
    return {'layer_asw': float(silhouette_score(values, layers)), 'layer_purity_k': purity, 'cross_layer_neighbor_contamination_k': float(1.0 - purity), 'reference_to_query_layer_macro_f1': float(f1_score(layers[query], predicted, labels=shared, average='macro'))}

def mean_knn_jaccard(left: np.ndarray, right: np.ndarray, *, k: int) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError('KNN overlap requires identical row support')
    left_neighbors = neighbors_without_self(center(left), k)
    right_neighbors = neighbors_without_self(center(right), k)
    scores = []
    for a, b in zip(left_neighbors, right_neighbors, strict=True):
        intersection = len(set(a.tolist()) & set(b.tolist()))
        scores.append(intersection / (2 * k - intersection))
    return float(np.mean(scores))

def within_batch_knn_preservation(baseline: np.ndarray, corrected: np.ndarray, batch: np.ndarray, *, k: int) -> float:
    batch = np.asarray(batch).astype(str)
    values = []
    for label in ('ref', 'query'):
        keep = batch == label
        values.append(mean_knn_jaccard(baseline[keep], corrected[keep], k=k))
    return float(np.mean(values))

def spatial_embedding_knn_overlap(spatial: np.ndarray, embedding: np.ndarray, batch: np.ndarray, *, k: int) -> float:
    batch = np.asarray(batch).astype(str)
    values = []
    for label in ('ref', 'query'):
        keep = batch == label
        values.append(mean_knn_jaccard(spatial[keep], embedding[keep], k=k))
    return float(np.mean(values))

def sampled_pairwise_distance_spearman(left: np.ndarray, right: np.ndarray, *, max_pairs: int, seed: int) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError('distance stability requires identical row support')
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
        pair_array = np.asarray(sorted(pairs), dtype=int)
        row, col = (pair_array[:, 0], pair_array[:, 1])
    left_values = center(left)
    right_values = center(right)
    left_distance = np.linalg.norm(left_values[row] - left_values[col], axis=1)
    right_distance = np.linalg.norm(right_values[row] - right_values[col], axis=1)
    value = spearmanr(left_distance, right_distance).statistic
    return float(value)

def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError('linear CKA requires identical row support')
    x = center(left)
    y = center(right)
    cross = np.linalg.norm(x.T @ y, ord='fro') ** 2
    left_norm = np.linalg.norm(x.T @ x, ord='fro')
    right_norm = np.linalg.norm(y.T @ y, ord='fro')
    denominator = left_norm * right_norm
    if denominator <= 0:
        raise ValueError('linear CKA denominator is zero')
    return float(cross / denominator)

def full_metric_row(values: np.ndarray, *, baseline: np.ndarray, spatial: np.ndarray, batch: np.ndarray, layers: np.ndarray, k: int, seed: int) -> dict[str, float]:
    result = conditional_batch_metrics(values, batch, layers, k=k, seed=seed)
    result.update(biological_metrics(values, batch, layers, k=k))
    result['within_section_knn_preservation_k'] = within_batch_knn_preservation(baseline, values, batch, k=k)
    result['spatial_embedding_knn_overlap_k'] = spatial_embedding_knn_overlap(spatial, values, batch, k=min(6, k))
    return result
