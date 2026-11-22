"""STAMP batch correction wrapper for FEAST batch-effect benchmarking.

Uses STAMP (Spatial Topic modeling for Annotation and Multi-sample integration)
with SGC mode and categorical_covariate_keys for batch correction.

Usage:
    python methods/_stamp_base.py \\
      --reference alpha_0.00.h5ad --query alpha_0.50.h5ad \\
      --output-dir outputs/methods/STAMP/shift_only/alpha_0.50
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from graph_contract import build_within_batch_knn
from fixed_panel import load_fixed_panel, subset_pair_to_fixed_panel, validate_raw_counts
from provenance import build_provenance
warnings.filterwarnings('ignore')

def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'

def _capture_training_progress(stamp_module):
    """Capture STAMP's displayed epoch losses without changing its optimizer."""
    original_tqdm = stamp_module.tqdm
    state = {'epochs_run': 0, 'displayed_epoch_loss_history': []}

    class CapturingProgress:

        def __init__(self, *args, **kwargs):
            self._progress = original_tqdm(*args, **kwargs)

        def __iter__(self):
            for value in self._progress:
                state['epochs_run'] += 1
                yield value

        def set_description(self, description):
            prefix = 'Epoch Loss:'
            if str(description).startswith(prefix):
                state['displayed_epoch_loss_history'].append(float(str(description)[len(prefix):]))
            return self._progress.set_description(description)

        def __getattr__(self, name):
            return getattr(self._progress, name)
    stamp_module.tqdm = CapturingProgress
    return (original_tqdm, state)

def _validate_saved_param_store(checkpoint_path: Path) -> dict:
    """Round-trip a trusted local Pyro checkpoint under PyTorch 2.6+."""
    import torch
    from pyro.params.param_store import ParamStoreDict
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    store = ParamStoreDict()
    store.set_state(state)
    named_parameters = list(store.named_parameters())
    nonfinite = [str(name) for name, parameter in named_parameters if not bool(torch.isfinite(parameter).all().item())]
    if not named_parameters or nonfinite:
        raise FloatingPointError('round-tripped STAMP checkpoint has missing or non-finite parameters')
    return {'status': 'trusted_local_round_trip_passed', 'loader': "torch.load(map_location='cpu', weights_only=False) + ParamStoreDict.set_state", 'security_scope': 'only the checkpoint written by this same local candidate process', 'parameter_tensors': int(len(named_parameters)), 'parameter_values': int(sum((parameter.numel() for _, parameter in named_parameters))), 'constraint_count': int(len(state.get('constraints', {}))), 'all_parameters_finite': True, 'nonfinite_parameter_names': nonfinite}

def _build_spatial_graph(adata, batch_key='batch', n_neighbors=6):
    """Build an asserted cross-batch-free spatial neighbor graph."""
    adj, metadata = build_within_batch_knn(adata.obsm['spatial'], adata.obs[batch_key].astype(str).to_numpy(), n_neighbors=n_neighbors)
    adata.obsp['spatial_connectivities'] = adj
    adata.uns['spatial_neighbors'] = metadata
    return metadata

def run_stamp_batch_correction(adata_ref: sc.AnnData, adata_query: sc.AnnData, panel_genes: list[str], panel_metadata: dict, n_topics: int=10, n_layers: int=1, hidden_size: int=128, batch_key: str='batch', mode: str='sgc', learning_rate: float=0.01, max_epochs: int=800, min_epochs: int=100, patience: int=20, device: str='cuda:0', seed: int=42) -> tuple[sc.AnnData, dict, object]:
    """Run STAMP batch correction on concatenated reference + query.

    Returns (concatenated AnnData with cell_by_topic, metadata dict).
    """
    import pyro
    import sctm.stamp as stamp_module
    import torch
    from sctm.stamp import STAMP
    if device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested for STAMP but is not available')
    random.seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cuda_device = torch.device(device) if device.startswith('cuda') else None
    cuda_device_index = 0 if cuda_device is not None and cuda_device.index is None else cuda_device.index if cuda_device is not None else None
    if cuda_device_index is not None:
        torch.cuda.set_device(cuda_device_index)
        torch.cuda.init()
        torch.cuda.synchronize(cuda_device_index)
        torch.cuda.reset_peak_memory_stats(cuda_device_index)
    t0 = time.time()
    raw_count_checks = [validate_raw_counts(adata_ref, label='reference'), validate_raw_counts(adata_query, label='query')]
    adata_ref, adata_query = subset_pair_to_fixed_panel(adata_ref, adata_query, panel_genes)
    fixed_panel_input_matrices = {'canonical_representation': 'study04-count-matrix-v1 int64 row-major'}
    adata_ref_copy = adata_ref.copy()
    adata_query_copy = adata_query.copy()
    adata_ref_copy.obs[batch_key] = 'ref'
    adata_query_copy.obs[batch_key] = 'query'
    adata_ref_copy.obs['source_spot_id'] = adata_ref_copy.obs_names.astype(str)
    adata_query_copy.obs['source_spot_id'] = adata_query_copy.obs_names.astype(str)
    adata = sc.concat([adata_ref_copy, adata_query_copy], join='inner', keys=['ref', 'query'], index_unique='__')
    adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    if not adata.obs_names.is_unique:
        raise ValueError('STAMP concatenation did not create unique composite spot IDs')
    if adata.var_names.astype(str).tolist() != panel_genes:
        raise RuntimeError('STAMP concatenation changed the fixed-panel gene order')
    adata_hvg = adata
    graph_meta = _build_spatial_graph(adata_hvg, batch_key=batch_key, n_neighbors=6)
    model = STAMP(adata_hvg, n_topics=n_topics, n_layers=n_layers, hidden_size=hidden_size, categorical_covariate_keys=[batch_key], gene_likelihood='nb', mode=mode, verbose=False)
    original_tqdm, training_state = _capture_training_progress(stamp_module)
    try:
        model.train(learning_rate=learning_rate, max_epochs=max_epochs, min_epochs=min_epochs, patience=patience, device=device)
    finally:
        stamp_module.tqdm = original_tqdm
    finite_parameters = all((bool(torch.isfinite(parameter).all().item()) for parameter in model.model.parameters()))
    pyro_parameters = list(pyro.get_param_store().named_parameters())
    nonfinite_pyro_parameters = [str(name) for name, parameter in pyro_parameters if not bool(torch.isfinite(parameter).all().item())]
    finite_pyro_param_store = bool(pyro_parameters) and (not nonfinite_pyro_parameters)
    displayed_losses = training_state['displayed_epoch_loss_history']
    finite_loss_history = bool(displayed_losses) and bool(np.isfinite(displayed_losses).all())
    early_stopper = getattr(model, 'early_stopper', None)
    early_stopped = bool(early_stopper is not None and early_stopper.counter >= patience)
    if training_state['epochs_run'] < max_epochs and (not early_stopped):
        raise FloatingPointError('STAMP training stopped before max_epochs without satisfying its early-stopping rule')
    if not finite_parameters or not finite_pyro_param_store or (not finite_loss_history):
        raise FloatingPointError('STAMP ended with non-finite parameters or losses')
    actual_device = str(next(model.model.parameters()).device)
    if device.startswith('cuda') and (not actual_device.startswith('cuda')):
        raise RuntimeError(f'STAMP silently used a non-CUDA device: {actual_device}')
    if cuda_device_index is not None:
        torch.cuda.synchronize(cuda_device_index)
        peak_cuda_memory_allocated_bytes = int(torch.cuda.max_memory_allocated(cuda_device_index))
        if peak_cuda_memory_allocated_bytes <= 0:
            raise RuntimeError('STAMP did not record a positive CUDA allocation')
    else:
        peak_cuda_memory_allocated_bytes = 0
    cell_by_topic = model.get_cell_by_topic()
    feature_by_topic = model.get_feature_by_topic()
    topic_values = cell_by_topic.to_numpy()
    feature_values = feature_by_topic.to_numpy()
    topic_row_sums = topic_values.sum(axis=1)
    if not np.isfinite(topic_values).all() or not np.isfinite(feature_values).all():
        raise FloatingPointError('STAMP returned a non-finite topic matrix')
    if np.any(topic_values < 0):
        raise FloatingPointError('STAMP returned negative topic proportions')
    if adata_hvg.var_names.astype(str).tolist() != panel_genes:
        raise RuntimeError('STAMP output changed the fixed-panel gene order')
    adata_hvg.obsm['X_STAMP'] = cell_by_topic.values.astype(np.float32)
    adata_hvg.uns['STAMP_cell_by_topic'] = cell_by_topic
    adata_hvg.uns['STAMP_feature_by_topic'] = feature_by_topic
    elapsed = time.time() - t0
    numpy_supported_by_feast = bool(np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.24.0') and np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0'))
    meta = {'method': 'STAMP', 'gene_selection': 'fixed_raw_reference_panel', 'gene_panel': panel_metadata, 'n_topics': n_topics, 'n_hvgs': len(panel_genes), 'n_layers': n_layers, 'hidden_size': hidden_size, 'mode': mode, 'learning_rate': learning_rate, 'max_epochs': max_epochs, 'min_epochs': min_epochs, 'patience': patience, 'device': device, 'actual_device': actual_device, 'cuda_execution_verified': bool(actual_device.startswith('cuda') and peak_cuda_memory_allocated_bytes > 0), 'peak_cuda_memory_allocated_bytes': peak_cuda_memory_allocated_bytes, 'gene_likelihood': 'nb', 'n_genes_input': raw_count_checks[0]['n_genes'], 'n_genes_hvg': adata_hvg.n_vars, 'n_spots_total': adata_hvg.n_obs, 'seed': seed, 'raw_count_checks': raw_count_checks, 'fixed_panel_input_matrices': fixed_panel_input_matrices, 'spatial_graph': graph_meta, 'training': {'epochs_run': training_state['epochs_run'], 'early_stopped': early_stopped, 'stopping_reason': 'early_stopping' if early_stopped else 'max_epochs', 'displayed_epoch_loss_history': displayed_losses, 'minimum_training_loss': float(early_stopper.min_training_loss), 'early_stopping_counter': int(early_stopper.counter), 'all_losses_finite': finite_loss_history, 'all_model_parameters_finite': finite_parameters, 'all_pyro_param_store_parameters_finite': finite_pyro_param_store, 'pyro_param_store_parameter_tensors': int(len(pyro_parameters)), 'pyro_param_store_parameter_values': int(sum((parameter.numel() for _, parameter in pyro_parameters))), 'nonfinite_pyro_param_store_parameter_names': nonfinite_pyro_parameters, 'topic_min': float(topic_values.min()), 'topic_max': float(topic_values.max()), 'topic_row_sum_min': float(topic_row_sums.min()), 'topic_row_sum_max': float(topic_row_sums.max())}, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scanpy': sc.__version__, 'torch': torch.__version__, 'pyro': pyro.__version__, 'sctm_distribution': _distribution_version('sctm'), 'feast_numpy_support_range': '>=1.24,<2', 'numpy_supported_by_feast': numpy_supported_by_feast, 'external_environment_limitation': None if numpy_supported_by_feast else 'This external STAMP worker uses a NumPy version unsupported by FEAST; FEAST is not imported or executed in this process.'}, 'runtime_controls': {'python_no_user_site': bool(sys.flags.no_user_site), 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'pip_check_status': os.environ.get('FEAST_REPRODUCE_PIP_CHECK')}, 'elapsed_seconds': round(elapsed, 2)}
    return (adata_hvg, meta, model)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panel-file', type=Path, required=True)
    parser.add_argument('--panel-provenance', type=Path, required=True)
    parser.add_argument('--expected-panel-size', type=int, default=2000)
    parser.add_argument('--n-topics', type=int, default=10)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='sgc')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--max-epochs', type=int, default=800)
    parser.add_argument('--min-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config-id', type=str, required=True)
    parser.add_argument('--parent-config-id', type=str, required=True)
    parser.add_argument('--deformation-mode', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--config-path', type=Path, required=True)
    args = parser.parse_args()
    if not args.reference.exists():
        print(f'ERROR: reference not found: {args.reference}', file=sys.stderr)
        return 1
    if not args.query.exists():
        print(f'ERROR: query not found: {args.query}', file=sys.stderr)
        return 1
    if not args.config_path.exists():
        print(f'ERROR: config not found: {args.config_path}', file=sys.stderr)
        return 1
    if not args.panel_provenance.exists():
        print(f'ERROR: panel provenance not found: {args.panel_provenance}', file=sys.stderr)
        return 1
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f'ERROR: refusing nonempty output: {args.output_dir}', file=sys.stderr)
        return 1
    sc.settings.verbosity = 0
    adata_ref = sc.read_h5ad(str(args.reference))
    adata_query = sc.read_h5ad(str(args.query))
    panel_genes, panel_metadata = load_fixed_panel(args.panel_file, expected_n_genes=args.expected_panel_size)
    panel_metadata['provenance_path'] = str(args.panel_provenance.resolve())
    print(f'STAMP: ref={adata_ref.n_obs}x{adata_ref.n_vars}, query={adata_query.n_obs}x{adata_query.n_vars}')
    try:
        adata_out, meta, model = run_stamp_batch_correction(adata_ref, adata_query, panel_genes=panel_genes, panel_metadata=panel_metadata, n_topics=args.n_topics, n_layers=args.n_layers, hidden_size=args.hidden_size, mode=args.mode, learning_rate=args.learning_rate, max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience, device=args.device, seed=args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / 'FAILED.txt'
        if fail_file.exists():
            fail_file.unlink()
        embeddings_path = args.output_dir / 'embeddings.npz'
        topics_path = args.output_dir / 'cell_by_topic.csv'
        result_path = args.output_dir / 'result.h5ad'
        checkpoint_path = args.output_dir / 'stamp_params.pt'
        np.savez(embeddings_path, embedding=adata_out.obsm['X_STAMP'], batch_labels=adata_out.obs['batch'].values)
        adata_out.uns['STAMP_cell_by_topic'].to_csv(topics_path)
        adata_out.write_h5ad(result_path, compression='gzip')
        model.save(checkpoint_path)
        meta['checkpoint_validation'] = _validate_saved_param_store(checkpoint_path)
        meta['source'] = {'path': str(Path(__file__).resolve())}
        meta['provenance'] = build_provenance(configuration_id=args.config_id, seed=args.seed, inputs=[args.reference, args.query, args.panel_file, args.panel_provenance], outputs=[embeddings_path, topics_path, result_path, checkpoint_path], solver_diagnostics={'status': 'completed', 'embedding_all_finite': bool(np.isfinite(adata_out.obsm['X_STAMP']).all()), 'cross_batch_edge_count': meta['spatial_graph']['cross_batch_edge_count'], 'learning_rate': args.learning_rate, 'epochs_run': meta['training']['epochs_run'], 'stopping_reason': meta['training']['stopping_reason'], 'all_losses_finite': meta['training']['all_losses_finite'], 'all_model_parameters_finite': meta['training']['all_model_parameters_finite'], 'all_pyro_param_store_parameters_finite': meta['training']['all_pyro_param_store_parameters_finite'], 'actual_device': meta['actual_device'], 'cuda_execution_verified': meta['actual_device'].startswith('cuda'), **meta['fixed_panel_input_matrices'], 'checkpoint_round_trip_status': meta['checkpoint_validation']['status']}, candidate_context={'parent_configuration_id': args.parent_config_id, 'method': 'STAMP', 'deformation_mode': args.deformation_mode, 'alpha': float(args.alpha), 'alpha_stratum': 'interpolation' if args.alpha <= 1.0 else 'extrapolation'}, sources=[Path(__file__).resolve(), Path(__file__).with_name('fixed_panel.py'), Path(__file__).with_name('graph_contract.py'), Path(__file__).with_name('provenance.py'), Path(__file__).resolve().parents[1] / 'run.py', args.config_path.resolve()])
        with open(args.output_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  -> {adata_out.n_obs} spots, {meta['n_topics']} topics in {meta['elapsed_seconds']:.1f}s")
        return 0
    except Exception as e:
        import traceback
        fail_file = args.output_dir / 'FAILED.txt'
        fail_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_file, 'w') as f:
            f.write(traceback.format_exc())
        print(f'  -> FAILED: {e}', file=sys.stderr)
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
