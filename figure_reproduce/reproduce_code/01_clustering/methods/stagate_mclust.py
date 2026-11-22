"""Run one single-process STAGATE+mclust fixed-panel job."""
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
import scipy.sparse as sp
import tensorflow as tf
from output_integrity import atomic_write_h5ad
os.environ.setdefault('R_HOME', str(Path(sys.prefix) / 'lib' / 'R'))
tf.compat.v1.disable_eager_execution()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--rad-cutoff', type=int, default=150)
    parser.add_argument('--n-epochs', type=int, default=500)
    args = parser.parse_args()
    inherited_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    import STAGATE
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices:
        raise RuntimeError('CUDA GPU was requested for STAGATE but is not available')
    data = sc.read_h5ad(args.input)
    if not sp.issparse(data.X):
        data.X = sp.csr_matrix(data.X)
    n_clusters = int(data.obs['ground_truth'].astype(str).nunique())
    started = time.time()
    STAGATE.Cal_Spatial_Net(data, rad_cutoff=args.rad_cutoff)
    result = STAGATE.train_STAGATE(data, alpha=0, n_epochs=args.n_epochs, key_added='STAGATE', random_seed=args.seed, save_attention=False, save_loss=False)
    logical_gpu_devices = tf.config.list_logical_devices('GPU')
    if not logical_gpu_devices:
        raise RuntimeError('STAGATE completed without a visible logical CUDA device')
    gpu_memory_info = {key: int(value) for key, value in tf.config.experimental.get_memory_info('GPU:0').items()}
    if gpu_memory_info.get('peak', 0) <= 0:
        raise RuntimeError('STAGATE completed without positive CUDA memory allocation')
    result = STAGATE.mclust_R(result, used_obsm='STAGATE', num_cluster=n_clusters, random_seed=args.seed)
    labels = result.obs['mclust'].astype(str).to_numpy()
    result.obs['predicted_cluster'] = pd.Categorical(labels)
    result.uns.clear()
    atomic_write_h5ad(result, args.output_dir / 'result.h5ad')
    pd.DataFrame({'spot_barcode': result.obs_names, 'predicted_cluster': labels}).to_csv(args.output_dir / 'clusters.csv', index=False)
    numpy_supported = bool(np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.24.0') and np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0'))
    metadata = {'status': 'validated_success', 'method': 'STAGATE_mclust', 'public_seed': args.seed, 'accelerator_requested': 'cuda', 'actual_accelerator': 'cuda', 'cuda_execution_verified': gpu_memory_info.get('peak', 0) > 0, 'gpu_memory_info_bytes': gpu_memory_info, 'gpu_devices': [device.name for device in gpu_devices], 'logical_gpu_devices': [device.name for device in logical_gpu_devices], 'rad_cutoff': args.rad_cutoff, 'n_epochs': args.n_epochs, 'n_clusters_target': n_clusters, 'n_clusters_actual': int(pd.Series(labels).nunique()), 'elapsed_seconds': round(time.time() - started, 2), 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'tensorflow': tf.__version__, 'executable': str(Path(sys.executable).absolute()), 'prefix': sys.prefix, 'ld_library_path': inherited_ld_library_path}, 'feast_imported_in_method_worker': False, 'feast_numpy_support_range': '>=1.24,<2', 'numpy_supported_by_feast': numpy_supported, 'external_environment_limitation': None if numpy_supported else 'This external STAGATE worker uses a NumPy version unsupported by FEAST; FEAST is not imported or executed in this process.'}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
