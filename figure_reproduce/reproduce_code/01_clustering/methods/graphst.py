"""Run GraphST+mclust on one prepared fixed-panel input."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from output_integrity import atomic_write_h5ad
os.environ.setdefault('R_HOME', str(Path(sys.prefix) / 'lib' / 'R'))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--radius', type=int, default=50)
    args = parser.parse_args()
    import torch
    from GraphST import GraphST as graphst_module
    from GraphST.utils import clustering
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested for GraphST but is not available')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    requested_device = torch.device(args.device)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(requested_device)
    data = sc.read_h5ad(args.input)
    n_clusters = int(data.obs['ground_truth'].astype(str).nunique())
    data.var['highly_variable'] = True
    sc.pp.scale(data, zero_center=False, max_value=10)
    started = time.time()
    model = graphst_module.GraphST(data, device=requested_device, epochs=args.epochs, dim_input=data.n_vars, random_seed=args.seed, datatype='10X')
    result = model.train()
    first_parameter = next(model.model.parameters(), None)
    if first_parameter is None:
        raise RuntimeError('GraphST model exposes no parameters after training')
    actual_device = str(first_parameter.device)
    if args.device.startswith('cuda') and (not actual_device.startswith('cuda')):
        raise RuntimeError(f'GraphST silently used a non-CUDA device: {actual_device}')
    cuda_peak_memory_bytes = int(torch.cuda.max_memory_allocated(requested_device)) if args.device.startswith('cuda') else 0
    if args.device.startswith('cuda') and cuda_peak_memory_bytes <= 0:
        raise RuntimeError('GraphST completed without positive CUDA memory allocation')
    clustering(result, n_clusters=n_clusters, radius=args.radius, method='mclust', refinement=True)
    labels = result.obs['domain'].astype(str).to_numpy()
    result.obs['predicted_cluster'] = pd.Categorical(labels)
    result.uns.clear()
    atomic_write_h5ad(result, args.output_dir / 'result.h5ad')
    pd.DataFrame({'spot_barcode': result.obs_names, 'predicted_cluster': labels}).to_csv(args.output_dir / 'clusters.csv', index=False)
    numpy_supported = bool(np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.24.0') and np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0'))
    metadata = {'status': 'validated_success', 'method': 'GraphST', 'public_seed': args.seed, 'device_requested': args.device, 'actual_device': actual_device, 'cuda_execution_verified': actual_device.startswith('cuda') and cuda_peak_memory_bytes > 0, 'cuda_peak_memory_bytes': cuda_peak_memory_bytes, 'cuda_available': bool(torch.cuda.is_available()), 'cuda_device_name': torch.cuda.get_device_name(requested_device) if torch.cuda.is_available() else None, 'epochs': args.epochs, 'radius': args.radius, 'n_clusters_target': n_clusters, 'n_clusters_actual': int(pd.Series(labels).nunique()), 'elapsed_seconds': round(time.time() - started, 2), 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'executable': str(Path(sys.executable).absolute()), 'prefix': sys.prefix}, 'feast_imported_in_method_worker': False, 'feast_numpy_support_range': '>=1.24,<2', 'numpy_supported_by_feast': numpy_supported, 'external_environment_limitation': None if numpy_supported else 'This external GraphST worker uses a NumPy version unsupported by FEAST; FEAST is not imported or executed in this process.'}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
