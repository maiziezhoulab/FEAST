from pathlib import Path
import json
import sys
import time
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from conditional_resampling_baseline import prepare_sampling, resample_spots
from evaluate_resampling_metrics import log_normalize
WIDTHS = (0.1, 0.2, 0.4)
MIN_BIN = 5

def correlation(a, b, weights=None):
    """Column-wise Pearson correlation, with optional observation weights."""
    a, b = (np.asarray(a, float), np.asarray(b, float))
    w = np.ones(a.shape[0]) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    ac = a - w @ a
    bc = b - w @ b
    numerator = np.sum(w[:, None] * ac * bc, axis=0)
    denominator = np.sqrt(np.sum(w[:, None] * ac ** 2, axis=0) * np.sum(w[:, None] * bc ** 2, axis=0))
    return np.divide(numerator, denominator, out=np.full(a.shape[1], np.nan), where=denominator > 0)

def read_data(path, genes):
    a = ad.read_h5ad(path)
    assert a.var_names.tolist() == genes
    matrix = a.layers['counts']
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    assert np.isfinite(values).all() and (values >= 0).all()
    normalized, _, _ = log_normalize(values)
    return dict(x=normalized, xy=np.asarray(a.obsm['spatial']), labels=a.obs['class'].astype(str).to_numpy(), ids=a.obs_names.to_numpy())

def membership(codes, size):
    return sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(size, len(codes)))

def region_plan(real, feast, spatialz, width, conditional):
    arrays, sizes = ([], [])
    class_names = sorted(set(real['labels']) | set(feast['labels']) | set(spatialz['labels']))
    class_codes = {name: i for i, name in enumerate(class_names)}
    for data in (real, feast, spatialz):
        keys = np.floor(data['xy'] / width).astype(np.int64)
        if conditional:
            keys = np.column_stack((keys, [class_codes[label] for label in data['labels']]))
        arrays.append(keys)
        sizes.append(len(keys))
    keys, inverse = np.unique(np.concatenate(arrays), axis=0, return_inverse=True)
    offsets = np.cumsum([0] + sizes)
    matrices = [membership(inverse[offsets[i]:offsets[i + 1]], len(keys)) for i in range(3)]
    ns = np.stack([np.asarray(m.sum(axis=1)).ravel() for m in matrices])
    shared = np.all(ns >= MIN_BIN, axis=0)
    assert shared.sum() > 1, 'No sufficient common regional support'
    return dict(matrices=matrices, counts=ns, shared=shared, keys=keys, coverage=float(ns[0, shared].sum() / sizes[0]), width=width, conditional=conditional)

def center_classes(means, weights, classes):
    centered = means.copy()
    for label in np.unique(classes):
        use = classes == label
        centered[use] -= np.average(means[use], axis=0, weights=weights[use])
    return centered
