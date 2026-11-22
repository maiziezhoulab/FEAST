"""Run label-free modularity-selected Leiden on one fixed-panel input."""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from output_integrity import atomic_write_h5ad

def weighted_modularity(connectivities, labels: np.ndarray) -> float:
    graph = sp.csr_matrix(connectivities, dtype=np.float64)
    total = float(graph.sum())
    if total <= 0 or graph.shape[0] != len(labels):
        raise ValueError('invalid connectivity graph')
    degree = np.asarray(graph.sum(axis=1)).ravel()
    score = 0.0
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        score += float(graph[members][:, members].sum()) / total - (float(degree[members].sum()) / total) ** 2
    return score

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resolutions', required=True)
    parser.add_argument('--n-pcs', type=int, default=50)
    parser.add_argument('--n-neighbors', type=int, default=15)
    args = parser.parse_args()
    data = sc.read_h5ad(args.input)
    if 'log1p' not in data.uns:
        raise ValueError('input is not marked as log1p-transformed')
    started = time.time()
    n_components = min(args.n_pcs, data.n_obs - 1, data.n_vars - 1)
    sc.pp.pca(data, n_comps=n_components, random_state=args.seed)
    sc.pp.neighbors(data, n_neighbors=args.n_neighbors, n_pcs=n_components, random_state=args.seed)
    sweep, labels_by_resolution = ([], {})
    for resolution in [float(value) for value in args.resolutions.split(',')]:
        key = f'leiden_{resolution:g}'
        sc.tl.leiden(data, resolution=resolution, key_added=key, random_state=args.seed)
        labels = data.obs[key].astype(str).to_numpy(copy=True)
        labels_by_resolution[resolution] = labels
        sweep.append({'resolution': resolution, 'n_clusters': int(pd.Series(labels).nunique()), 'modularity_gamma1': weighted_modularity(data.obsp['connectivities'], labels)})
    selected = max(sweep, key=lambda row: (row['modularity_gamma1'], -row['resolution']))
    labels = labels_by_resolution[selected['resolution']]
    data.obs['predicted_cluster'] = pd.Categorical(labels)
    atomic_write_h5ad(data, args.output_dir / 'result.h5ad')
    pd.DataFrame({'spot_barcode': data.obs_names, 'predicted_cluster': labels}).to_csv(args.output_dir / 'clusters.csv', index=False)
    metadata = {'status': 'validated_success', 'method': 'Leiden_unsupervised', 'public_seed': args.seed, 'selection_rule': 'maximum weighted Newman-Girvan modularity at fixed gamma=1; lower resolution wins exact ties', 'selection_uses_ground_truth': False, 'resolutions_requested': [float(value) for value in args.resolutions.split(',')], 'selected_resolution': selected['resolution'], 'selected_modularity_gamma1': selected['modularity_gamma1'], 'n_pcs_requested': args.n_pcs, 'n_pcs_used': n_components, 'n_neighbors': args.n_neighbors, 'sweep': sweep, 'elapsed_seconds': round(time.time() - started, 2), 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scanpy': sc.__version__, 'executable': str(Path(sys.executable).absolute()), 'prefix': sys.prefix}}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
