"""Publication-safe spatial graph construction for multi-batch methods."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
DISTANCE_BLOCK_SIZE = 512

def _deterministic_nonself_knn(points: np.ndarray, *, n_neighbors: int) -> np.ndarray:
    """Return exact squared-Euclidean KNN with a declared index tie-break.

    Stable sorting makes the original within-batch row order the secondary key
    whenever squared distances are exactly tied.  Distances are evaluated in
    bounded row blocks so the contract does not require a full n-by-n matrix.
    """
    n_points = points.shape[0]
    result = np.empty((n_points, n_neighbors), dtype=np.intp)
    for start in range(0, n_points, DISTANCE_BLOCK_SIZE):
        stop = min(start + DISTANCE_BLOCK_SIZE, n_points)
        distances = cdist(points[start:stop], points, metric='sqeuclidean')
        local_rows = np.arange(stop - start)
        distances[local_rows, np.arange(start, stop)] = np.inf
        result[start:stop] = np.argsort(distances, axis=1, kind='stable')[:, :n_neighbors]
    return result

def build_within_batch_knn(spatial: np.ndarray, batch_labels: np.ndarray, *, n_neighbors: int) -> tuple[sp.csr_matrix, dict]:
    """Build a directed KNN graph independently within each batch.

    The returned matrix excludes self-edges.  A hard postcondition verifies
    that no edge crosses a batch boundary, even when batches share identical
    coordinate systems.
    """
    spatial = np.asarray(spatial, dtype=np.float64)
    batch_labels = np.asarray(batch_labels)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError('spatial must have shape (n_spots, 2)')
    if spatial.shape[0] != batch_labels.shape[0]:
        raise ValueError('spatial coordinates and batch labels must align')
    if not np.isfinite(spatial).all():
        raise ValueError('spatial coordinates must be finite')
    if n_neighbors < 1:
        raise ValueError('n_neighbors must be positive')
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    batch_sizes: dict[str, int] = {}
    for batch in np.unique(batch_labels):
        batch_index = np.flatnonzero(batch_labels == batch)
        batch_sizes[str(batch)] = int(batch_index.size)
        if batch_index.size <= n_neighbors:
            raise ValueError(f'batch {batch!r} has {batch_index.size} spots; requires more than n_neighbors={n_neighbors}')
        local_index = _deterministic_nonself_knn(spatial[batch_index], n_neighbors=n_neighbors)
        rows.append(np.repeat(batch_index, n_neighbors))
        cols.append(batch_index[local_index.reshape(-1)])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    values = np.ones(row.size, dtype=np.float32)
    directed = sp.csr_matrix((values, (row, col)), shape=(spatial.shape[0], spatial.shape[0]))
    directed.eliminate_zeros()
    edge_rows, edge_cols = directed.nonzero()
    cross_batch_edges = int(np.count_nonzero(batch_labels[edge_rows] != batch_labels[edge_cols]))
    if cross_batch_edges:
        raise RuntimeError(f'constructed {cross_batch_edges} cross-batch edges')
    if directed.diagonal().any():
        raise RuntimeError('constructed unexpected self-edges')
    per_batch = {}
    for batch in np.unique(batch_labels):
        batch_index = np.flatnonzero(batch_labels == batch)
        subgraph = directed[batch_index][:, batch_index].tocsr()
        out_degree = np.diff(subgraph.indptr)
        in_degree = np.asarray(subgraph.sum(axis=0)).ravel()
        per_batch[str(batch)] = {'nodes': int(batch_index.size), 'directed_edges': int(subgraph.nnz), 'weak_components': int(connected_components(subgraph, directed=True, connection='weak', return_labels=False)), 'out_degree_min': int(out_degree.min()), 'out_degree_max': int(out_degree.max()), 'out_degree_mean': float(out_degree.mean()), 'in_degree_min': int(in_degree.min()), 'in_degree_max': int(in_degree.max()), 'in_degree_mean': float(in_degree.mean())}
    metadata = {'schema_version': 3, 'construction': 'independent_directed_knn_by_batch', 'distance': 'squared_euclidean', 'self_exclusion': 'exact_row_identity', 'distance_tie_break': 'original_within_batch_spot_order', 'distance_block_size': DISTANCE_BLOCK_SIZE, 'n_neighbors': int(n_neighbors), 'n_spots': int(spatial.shape[0]), 'n_batches': int(len(batch_sizes)), 'batch_sizes': batch_sizes, 'per_batch': per_batch, 'directed_edge_count': int(directed.nnz), 'cross_batch_edge_count': cross_batch_edges, 'self_edge_count': 0}
    return (directed, metadata)

def graphst_matrices(directed: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return the dense directed and symmetric matrices GraphST expects."""
    interaction = directed.astype(np.float32).toarray()
    adjacency = np.maximum(interaction, interaction.T)
    return (interaction, adjacency)
