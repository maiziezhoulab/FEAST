"""GraphST vertical integration wrapper for FEAST batch-effect benchmarking.

The spatial graph is built independently within each slice and a hard
postcondition rejects any cross-batch edge before training.

Reference: https://deepst-tutorials.readthedocs.io/en/latest/Tutorial%205_Vertical%20Integration.html

Usage:
    python methods/graphst.py \\
      --reference alpha_0.00.h5ad --query alpha_0.50.h5ad \\
      --output-dir outputs/methods/GraphST/shift_only/alpha_0.50
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
from graph_contract import build_within_batch_knn, graphst_matrices
from fixed_panel import load_fixed_panel, subset_pair_to_fixed_panel, validate_raw_counts
from provenance import build_provenance
warnings.filterwarnings('ignore')

def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'

def run_graphst_batch_correction(adata_ref: sc.AnnData, adata_query: sc.AnnData, panel_genes: list[str], panel_metadata: dict, device: str='cuda', epochs: int=600, seed: int=41) -> tuple[sc.AnnData, dict, object]:
    """Run GraphST vertical integration on concatenated reference + query.

    The graph is explicit because the slices use overlapping coordinate frames.
    """
    import torch
    from GraphST import GraphST as graphst_module
    if device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested for GraphST but is not available')
    torch_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch_key = 'batch'
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
        raise ValueError('GraphST concatenation did not create unique composite spot IDs')
    if adata.var_names.astype(str).tolist() != panel_genes:
        raise RuntimeError('GraphST concatenation changed the fixed-panel gene order')
    adata_hvg = adata
    sc.pp.normalize_total(adata_hvg, target_sum=10000.0)
    sc.pp.log1p(adata_hvg)
    adata_hvg.var['highly_variable'] = True
    sc.pp.scale(adata_hvg, zero_center=False, max_value=10)
    directed, graph_meta = build_within_batch_knn(adata_hvg.obsm['spatial'], adata_hvg.obs[batch_key].astype(str).to_numpy(), n_neighbors=3)
    graph_neigh, adjacency = graphst_matrices(directed)
    adata_hvg.obsm['graph_neigh'] = graph_neigh
    adata_hvg.obsm['adj'] = adjacency
    model = graphst_module.GraphST(adata_hvg, device=torch_device, epochs=epochs, dim_input=adata_hvg.n_vars, random_seed=seed, datatype='10X')
    adata_hvg = model.train()
    parameter_tensors = list(model.model.parameters())
    finite_parameters = bool(parameter_tensors) and all((bool(torch.isfinite(parameter).all().item()) for parameter in parameter_tensors))
    optimizer_tensors = [value for state in model.optimizer.state.values() for value in state.values() if torch.is_tensor(value)]
    finite_optimizer_state = bool(optimizer_tensors) and all((bool(torch.isfinite(value).all().item()) for value in optimizer_tensors))
    loss_components = {'feature': float(model.loss_feat.detach().cpu()), 'contrastive_1': float(model.loss_sl_1.detach().cpu()), 'contrastive_2': float(model.loss_sl_2.detach().cpu())}
    loss_components['weighted_total'] = float(model.alpha * loss_components['feature'] + model.beta * (loss_components['contrastive_1'] + loss_components['contrastive_2']))
    if not finite_parameters or not finite_optimizer_state or (not np.isfinite(list(loss_components.values())).all()):
        raise FloatingPointError('GraphST ended with non-finite parameters or losses')
    if 'emb_pca' in adata_hvg.obsm:
        adata_hvg.obsm['X_GraphST'] = adata_hvg.obsm['emb_pca']
        embedding_source_key = 'emb_pca'
    elif 'emb' in adata_hvg.obsm:
        adata_hvg.obsm['X_GraphST'] = adata_hvg.obsm['emb']
        embedding_source_key = 'emb'
    else:
        raise RuntimeError("GraphST training did not produce 'emb' or 'emb_pca' in obsm")
    embedding = np.asarray(adata_hvg.obsm['X_GraphST'])
    if not np.isfinite(embedding).all():
        raise FloatingPointError('GraphST returned a non-finite embedding')
    if adata_hvg.var_names.astype(str).tolist() != panel_genes:
        raise RuntimeError('GraphST output changed the fixed-panel gene order')
    actual_device = str(next(model.model.parameters()).device)
    if device.startswith('cuda') and (not actual_device.startswith('cuda')):
        raise RuntimeError(f'GraphST silently used a non-CUDA device: {actual_device}')
    numpy_supported_by_feast = bool(np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.24.0') and np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0'))
    elapsed = time.time() - t0
    meta = {'method': 'GraphST', 'gene_selection': 'fixed_raw_reference_panel', 'gene_panel': panel_metadata, 'n_hvgs': len(panel_genes), 'epochs': epochs, 'device': str(device), 'n_genes_input': raw_count_checks[0]['n_genes'], 'n_genes_hvg': adata_hvg.n_vars, 'n_spots_total': adata_hvg.n_obs, 'embedding_key': 'X_GraphST', 'embedding_source_key': embedding_source_key, 'embedding_dimensions': int(adata_hvg.obsm['X_GraphST'].shape[1]), 'seed': seed, 'raw_count_checks': raw_count_checks, 'fixed_panel_input_matrices': fixed_panel_input_matrices, 'spatial_graph': graph_meta, 'training': {'epochs_requested': int(epochs), 'epochs_completed': int(epochs), 'all_model_parameters_finite': finite_parameters, 'all_optimizer_state_tensors_finite': finite_optimizer_state, 'model_parameter_tensors': int(len(parameter_tensors)), 'model_parameter_values': int(sum((parameter.numel() for parameter in parameter_tensors))), 'optimizer_state_tensors': int(len(optimizer_tensors)), 'optimizer_state_values': int(sum((value.numel() for value in optimizer_tensors))), 'final_loss_components': loss_components}, 'actual_device': actual_device, 'runtime_controls': {'python_no_user_site': bool(sys.flags.no_user_site), 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'pip_check_status': os.environ.get('FEAST_REPRODUCE_PIP_CHECK')}, 'graphst_import': {'path': str(Path(graphst_module.__file__).resolve())}, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scanpy': sc.__version__, 'torch': torch.__version__, 'graphst': _distribution_version('GraphST'), 'feast_numpy_support_range': '>=1.24,<2', 'numpy_supported_by_feast': numpy_supported_by_feast, 'external_environment_limitation': None if numpy_supported_by_feast else 'This external GraphST worker uses a NumPy version unsupported by FEAST; FEAST is not imported or executed in this process.'}, 'elapsed_seconds': round(elapsed, 2)}
    return (adata_hvg, meta, model)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panel-file', type=Path, required=True)
    parser.add_argument('--panel-provenance', type=Path, required=True)
    parser.add_argument('--expected-panel-size', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--seed', type=int, default=41)
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
    print(f'GraphST: ref={adata_ref.n_obs}x{adata_ref.n_vars}, query={adata_query.n_obs}x{adata_query.n_vars}')
    try:
        adata_out, meta, model = run_graphst_batch_correction(adata_ref, adata_query, panel_genes=panel_genes, panel_metadata=panel_metadata, device=args.device, epochs=args.epochs, seed=args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / 'FAILED.txt'
        if fail_file.exists():
            fail_file.unlink()
        embeddings_path = args.output_dir / 'embeddings.npz'
        result_path = args.output_dir / 'result.h5ad'
        checkpoint_path = args.output_dir / 'graphst_checkpoint.pt'
        np.savez(embeddings_path, embedding=adata_out.obsm['X_GraphST'], batch_labels=adata_out.obs['batch'].values)
        adata_out.write_h5ad(result_path, compression='gzip')
        import torch
        torch.save({'model_state_dict': model.model.state_dict(), 'optimizer_state_dict': model.optimizer.state_dict(), 'epochs_requested': args.epochs, 'seed': args.seed}, checkpoint_path)
        meta['source'] = {'path': str(Path(__file__).resolve())}
        meta['provenance'] = build_provenance(configuration_id=args.config_id, seed=args.seed, inputs=[args.reference, args.query, args.panel_file, args.panel_provenance], outputs=[embeddings_path, result_path, checkpoint_path], solver_diagnostics={'status': 'completed', 'epochs_requested': args.epochs, 'epochs_completed': args.epochs, 'embedding_all_finite': True, 'all_model_parameters_finite': meta['training']['all_model_parameters_finite'], 'all_optimizer_state_tensors_finite': meta['training']['all_optimizer_state_tensors_finite'], 'final_loss_components': meta['training']['final_loss_components'], 'actual_device': meta['actual_device'], 'cuda_execution_verified': meta['actual_device'].startswith('cuda'), **meta['fixed_panel_input_matrices'], 'cross_batch_edge_count': meta['spatial_graph']['cross_batch_edge_count']}, candidate_context={'parent_configuration_id': args.parent_config_id, 'method': 'GraphST', 'deformation_mode': args.deformation_mode, 'alpha': float(args.alpha), 'alpha_stratum': 'interpolation' if args.alpha <= 1.0 else 'extrapolation'}, sources=[Path(__file__).resolve(), Path(__file__).with_name('fixed_panel.py'), Path(__file__).with_name('graph_contract.py'), Path(__file__).with_name('provenance.py'), Path(__file__).resolve().parents[1] / 'run.py', args.config_path.resolve(), Path(meta['graphst_import']['path'])])
        with open(args.output_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  -> {adata_out.n_obs} spots, latent={adata_out.obsm['X_GraphST'].shape[1]}d in {meta['elapsed_seconds']:.1f}s")
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
