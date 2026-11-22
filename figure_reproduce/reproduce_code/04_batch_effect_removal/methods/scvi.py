"""scVI batch correction wrapper for FEAST batch-effect benchmarking.

Usage:
    python methods/scvi.py \\
      --reference alpha_0.00.h5ad --query alpha_0.50.h5ad \\
      --output-dir outputs/methods/scVI/shift_only/alpha_0.50
"""
from __future__ import annotations
import argparse
import importlib
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
from fixed_panel import load_fixed_panel, subset_pair_to_fixed_panel, validate_raw_counts
from provenance import build_provenance
warnings.filterwarnings('ignore')

def _import_scvi_tools():
    """Import scvi-tools without resolving this wrapper as ``scvi``."""
    wrapper_path = Path(__file__).resolve()
    wrapper_directory = wrapper_path.parent
    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != wrapper_directory]
        module = importlib.import_module('scvi')
    finally:
        sys.path[:] = original_sys_path
    module_path = Path(module.__file__).resolve()
    if module_path == wrapper_path or not hasattr(module, 'settings'):
        raise ImportError(f'scvi-tools import resolved to an invalid module: {module_path}')
    return module

def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'

def _serialize_history(history) -> tuple[dict, dict]:
    serialized = {}
    summary = {}
    for key, value in history.items():
        frame = value.to_frame() if isinstance(value, pd.Series) else pd.DataFrame(value)
        values = frame.to_numpy()
        numeric = pd.to_numeric(pd.Series(values.ravel()), errors='coerce').to_numpy()
        finite = numeric[np.isfinite(numeric)]
        serialized_values = []
        for row in values:
            serialized_row = []
            for item in row:
                try:
                    numeric_item = float(item)
                except (TypeError, ValueError):
                    numeric_item = np.nan
                serialized_row.append(numeric_item if np.isfinite(numeric_item) else None)
            serialized_values.append(serialized_row)
        serialized[key] = {'columns': [str(column) for column in frame.columns], 'index': [str(index) for index in frame.index], 'values': serialized_values}
        summary[key] = {'rows': int(frame.shape[0]), 'columns': int(frame.shape[1]), 'total_values': int(values.size), 'finite_values': int(finite.size), 'first': float(finite[0]) if finite.size else None, 'last': float(finite[-1]) if finite.size else None, 'minimum': float(finite.min()) if finite.size else None, 'maximum': float(finite.max()) if finite.size else None}
    return (serialized, summary)

def run_scvi_batch_correction(adata_ref: sc.AnnData, adata_query: sc.AnnData, panel_genes: list[str], panel_metadata: dict, n_latent: int=10, n_layers: int=1, n_hidden: int=128, dropout_rate: float=0.1, max_epochs: int=400, learning_rate: float=0.001, seed: int=42, batch_key: str='batch', accelerator: str='auto', devices: int=1) -> tuple[sc.AnnData, dict, object]:
    """Run scVI batch correction on concatenated reference + query.

    Uses scVI defaults: zinb likelihood, 1 hidden layer, 10-dim latent space.
    """
    scvi = _import_scvi_tools()
    import torch
    if accelerator in {'gpu', 'cuda'} and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested for scVI but is not available')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    scvi.settings.seed = seed
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
        raise ValueError('scVI concatenation did not create unique composite spot IDs')
    if adata.var_names.astype(str).tolist() != panel_genes:
        raise RuntimeError('scVI concatenation changed the fixed-panel gene order')
    if hasattr(adata.X, 'toarray'):
        adata.layers['counts'] = adata.X.copy()
    else:
        adata.layers['counts'] = adata.X.astype(np.float32).copy()
    adata_hvg = adata
    scvi.model.SCVI.setup_anndata(adata_hvg, layer='counts', batch_key=batch_key)
    model = scvi.model.SCVI(adata_hvg, n_latent=n_latent, n_layers=n_layers, n_hidden=n_hidden, dropout_rate=dropout_rate, gene_likelihood='zinb')
    model.train(max_epochs=max_epochs, early_stopping=True, accelerator=accelerator, devices=devices, plan_kwargs={'lr': learning_rate}, deterministic=True)
    adata_hvg.obsm['X_scVI'] = model.get_latent_representation()
    adata_hvg.layers['scvi_normalized'] = model.get_normalized_expression(transform_batch='ref', return_numpy=True)
    elapsed = time.time() - t0
    model_history, history_summary = _serialize_history(model.history)
    history_lengths = [item['rows'] for item in history_summary.values()]
    if not history_lengths or max(history_lengths) == 0:
        raise RuntimeError('scVI did not expose a usable training history')
    all_history_values_finite = bool(history_summary) and all((item['finite_values'] == item['total_values'] for item in history_summary.values()))
    if not all_history_values_finite:
        raise FloatingPointError('scVI training history contains non-finite values')
    latent = np.asarray(adata_hvg.obsm['X_scVI'])
    normalized = np.asarray(adata_hvg.layers['scvi_normalized'])
    if not np.isfinite(latent).all() or not np.isfinite(normalized).all():
        raise FloatingPointError('scVI returned non-finite corrected values')
    if np.any(normalized < 0):
        raise FloatingPointError('scVI returned negative normalized expression')
    model_parameters = list(model.module.parameters())
    finite_model_parameters = bool(model_parameters) and all((bool(torch.isfinite(parameter).all().item()) for parameter in model_parameters))
    if not finite_model_parameters:
        raise FloatingPointError('scVI ended with non-finite model parameters')
    state_tensors = [value for value in model.module.state_dict().values() if torch.is_tensor(value)]
    finite_model_state = bool(state_tensors) and all((bool(torch.isfinite(value).all().item()) for value in state_tensors))
    if not finite_model_state:
        raise FloatingPointError('scVI state dictionary contains non-finite tensors')
    if adata_hvg.var_names.astype(str).tolist() != panel_genes:
        raise RuntimeError('scVI output changed the fixed-panel gene order')
    actual_device = str(model.device)
    if accelerator in {'gpu', 'cuda'} and (not actual_device.startswith('cuda')):
        raise RuntimeError(f'scVI silently used a non-CUDA device: {actual_device}')
    numpy_supported_by_feast = bool(np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.24.0') and np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0'))
    early_stopping = {}
    for callback in getattr(model.trainer, 'callbacks', []):
        if 'EarlyStopping' not in callback.__class__.__name__:
            continue
        best_score = getattr(callback, 'best_score', None)
        early_stopping = {'callback': callback.__class__.__name__, 'monitor': getattr(callback, 'monitor', None), 'mode': getattr(callback, 'mode', None), 'patience': getattr(callback, 'patience', None), 'wait_count': getattr(callback, 'wait_count', None), 'stopped_epoch': getattr(callback, 'stopped_epoch', None), 'best_score': float(best_score.detach().cpu()) if best_score is not None else None}
        break
    meta = {'method': 'scVI', 'gene_selection': 'fixed_raw_reference_panel', 'gene_panel': panel_metadata, 'gene_likelihood': 'zinb', 'n_hvgs': len(panel_genes), 'n_latent': n_latent, 'n_layers': n_layers, 'n_hidden': n_hidden, 'dropout_rate': dropout_rate, 'max_epochs': max_epochs, 'learning_rate': learning_rate, 'n_genes_input': raw_count_checks[0]['n_genes'], 'n_genes_hvg': adata_hvg.n_vars, 'n_spots_total': adata_hvg.n_obs, 'seed': seed, 'raw_count_checks': raw_count_checks, 'fixed_panel_input_matrices': fixed_panel_input_matrices, 'normalized_expression_transform_batch': 'ref', 'seed_application': {'python_random': seed, 'numpy': seed, 'torch': seed, 'torch_cuda_all': seed if torch.cuda.is_available() else 'not_available', 'scvi_settings': int(scvi.settings.seed), 'lightning_deterministic': True}, 'training': {'epochs_recorded': int(max(history_lengths)), 'trainer_current_epoch': int(model.trainer.current_epoch), 'trainer_should_stop': bool(model.trainer.should_stop), 'early_stopping': early_stopping, 'history_summary': history_summary, 'all_history_values_finite': all_history_values_finite, 'latent_all_finite': True, 'normalized_expression_all_finite': True, 'normalized_expression_nonnegative': True, 'all_model_parameters_finite': finite_model_parameters, 'all_model_state_tensors_finite': finite_model_state, 'model_parameter_tensors': int(len(model_parameters)), 'model_parameter_values': int(sum((parameter.numel() for parameter in model_parameters))), 'model_state_tensors': int(len(state_tensors)), 'model_state_values': int(sum((value.numel() for value in state_tensors)))}, 'accelerator_requested': accelerator, 'devices_requested': devices, 'actual_device': actual_device, 'execution_environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scanpy': sc.__version__, 'torch': torch.__version__, 'scvi_tools': scvi.__version__, 'scvi_module_path': str(Path(scvi.__file__).resolve()), 'lightning': _distribution_version('lightning'), 'anndata': _distribution_version('anndata'), 'feast_numpy_support_range': '>=1.24,<2', 'numpy_supported_by_feast': numpy_supported_by_feast, 'external_environment_limitation': None if numpy_supported_by_feast else 'This external scVI worker uses a NumPy version unsupported by FEAST; FEAST is not imported or executed in this process.'}, 'runtime_controls': {'python_no_user_site': bool(sys.flags.no_user_site), 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'pip_check_status': os.environ.get('FEAST_REPRODUCE_PIP_CHECK')}, 'elapsed_seconds': round(elapsed, 2), 'model_history': model_history}
    return (adata_hvg, meta, model)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panel-file', type=Path, required=True)
    parser.add_argument('--panel-provenance', type=Path, required=True)
    parser.add_argument('--expected-panel-size', type=int, default=2000)
    parser.add_argument('--n-latent', type=int, default=10)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--max-epochs', type=int, default=400)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
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
    print(f'scVI: ref={adata_ref.n_obs}x{adata_ref.n_vars}, query={adata_query.n_obs}x{adata_query.n_vars}')
    try:
        adata_out, meta, model = run_scvi_batch_correction(adata_ref, adata_query, panel_genes=panel_genes, panel_metadata=panel_metadata, n_latent=args.n_latent, n_layers=args.n_layers, n_hidden=args.n_hidden, dropout_rate=args.dropout_rate, max_epochs=args.max_epochs, learning_rate=args.learning_rate, seed=args.seed, accelerator=args.accelerator, devices=args.devices)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / 'FAILED.txt'
        if fail_file.exists():
            fail_file.unlink()
        ref_mask = adata_out.obs['batch'] == 'ref'
        query_mask = adata_out.obs['batch'] == 'query'
        embeddings_path = args.output_dir / 'embeddings.npz'
        result_path = args.output_dir / 'result.h5ad'
        model_dir = args.output_dir / 'model'
        np.savez(embeddings_path, embedding=adata_out.obsm['X_scVI'], batch_labels=adata_out.obs['batch'].values, ref_mask=ref_mask.values, query_mask=query_mask.values)
        adata_out.write_h5ad(result_path, compression='gzip')
        model.save(str(model_dir), overwrite=False, save_anndata=False)
        meta['source'] = {'path': str(Path(__file__).resolve())}
        model_files = sorted((path for path in model_dir.rglob('*') if path.is_file()))
        meta['provenance'] = build_provenance(configuration_id=args.config_id, seed=args.seed, inputs=[args.reference, args.query, args.panel_file, args.panel_provenance], outputs=[embeddings_path, result_path, *model_files], solver_diagnostics={'status': 'completed', 'embedding_all_finite': bool(np.isfinite(adata_out.obsm['X_scVI']).all()), 'early_stopping': True, 'max_epochs': args.max_epochs, 'learning_rate': args.learning_rate, 'epochs_recorded': meta['training']['epochs_recorded'], 'trainer_should_stop': meta['training']['trainer_should_stop'], 'latent_all_finite': meta['training']['latent_all_finite'], 'normalized_expression_all_finite': meta['training']['normalized_expression_all_finite'], 'all_model_parameters_finite': meta['training']['all_model_parameters_finite'], 'all_model_state_tensors_finite': meta['training']['all_model_state_tensors_finite'], 'all_history_values_finite': meta['training']['all_history_values_finite'], 'actual_device': meta['actual_device'], 'cuda_execution_verified': meta['actual_device'].startswith('cuda'), **meta['fixed_panel_input_matrices']}, candidate_context={'parent_configuration_id': args.parent_config_id, 'method': 'scVI', 'deformation_mode': args.deformation_mode, 'alpha': float(args.alpha), 'alpha_stratum': 'interpolation' if args.alpha <= 1.0 else 'extrapolation'}, sources=[Path(__file__).resolve(), Path(__file__).with_name('fixed_panel.py'), Path(__file__).with_name('provenance.py'), Path(__file__).resolve().parents[1] / 'run.py', args.config_path.resolve(), Path(meta['execution_environment']['scvi_module_path'])])
        with open(args.output_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  -> {adata_out.n_obs} spots, latent={meta['n_latent']}d in {meta['elapsed_seconds']:.1f}s")
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
