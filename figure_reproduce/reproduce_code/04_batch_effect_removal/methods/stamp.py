"""Run STAMP with a declared zero-panel-library encoder safeguard.

The raw fixed-panel count matrix remains the likelihood input.  STAMP's
upstream preprocessing divides every row by its fixed-panel library size;
for a raw-zero row that operation produces NaNs before epoch 1.  This wrapper
changes only those two preprocessing divisions: raw-zero rows remain zero for
background estimation, and the existing within-batch SGC supplies their
encoder features from spatial neighbours.  Positive-library rows use the
upstream implementation unchanged.
"""
from __future__ import annotations
import argparse
import inspect
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import _stamp_base as base
STRATEGY = 'raw_zero_likelihood_within_batch_sgc_encoder_v1'

def _row_sums(matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=1)).reshape(-1)

@contextmanager
def _zero_library_safeguard(state: dict[str, object]):
    """Temporarily make STAMP's two row normalizations zero-safe."""
    import torch
    import sctm.stamp as stamp_module
    original_get_init_bg = stamp_module.get_init_bg
    original_precompute_sgc = stamp_module.precompute_SGC_scipy

    def safe_get_init_bg(data):
        library = data.sum(axis=1, keepdim=True)
        zero = library.squeeze(1) == 0
        zero_count = int(zero.sum().item())
        state['background_zero_library_rows'] = zero_count
        if zero_count == 0:
            state['background_path'] = 'upstream_unchanged'
            return original_get_init_bg(data)
        normalized = torch.zeros_like(data)
        positive = ~zero
        normalized[positive] = data[positive] / library[positive]
        means = torch.mean(normalized, axis=0)
        result = torch.log(means + 1e-15)
        state['background_path'] = 'zero_rows_contribute_zero_mass'
        state['background_all_finite'] = bool(torch.isfinite(result).all().item())
        print('Computing zero-library-safe background frequencies')
        return result

    def safe_precompute_sgc(x, adj, n_layers, mode='sign', add_diag=True, csr=True):
        library = x.sum(dim=1, keepdim=True)
        zero = library.squeeze(1) == 0
        zero_count = int(zero.sum().item())
        state['encoder_zero_library_rows'] = zero_count
        if zero_count == 0:
            state['encoder_path'] = 'upstream_unchanged'
            return original_precompute_sgc(x, adj, n_layers=n_layers, mode=mode, add_diag=add_diag, csr=csr)
        from scipy.sparse import diags, identity
        degree = adj.sum(axis=1)
        if add_diag:
            degree = degree + 2
            adj = adj + identity(n=x.shape[0])
        degree = diags(degree.A1).power(-0.5)
        degree.data[degree.data == np.inf] = 0
        normalized_adj = degree @ adj @ degree
        torch_adj = stamp_module.make_sparse_tensor(normalized_adj)
        median_library = torch.median(library.squeeze(1))
        normalized = torch.zeros_like(x)
        positive = ~zero
        normalized[positive] = x[positive] / library[positive] * median_library
        propagated = [normalized]
        for _ in range(n_layers):
            propagated.append(torch.sparse.mm(torch_adj, propagated[-1]))
        if mode == 'sgc':
            propagated = [propagated[-1]]
        result = torch.log(torch.cat(propagated, dim=1) + 1)
        zero_norms = torch.linalg.vector_norm(result[zero], dim=1)
        state['encoder_path'] = 'zero_rows_receive_within_batch_sgc_signal'
        state['encoder_all_finite'] = bool(torch.isfinite(result).all().item())
        state['zero_row_encoder_norm_min'] = float(zero_norms.min().item())
        state['zero_row_encoder_norm_max'] = float(zero_norms.max().item())
        return result
    stamp_module.get_init_bg = safe_get_init_bg
    stamp_module.precompute_SGC_scipy = safe_precompute_sgc
    try:
        yield stamp_module
    finally:
        stamp_module.get_init_bg = original_get_init_bg
        stamp_module.precompute_SGC_scipy = original_precompute_sgc

def _validate_zero_library_contract(adata_out, model, state: dict[str, object]) -> dict[str, object]:
    import torch
    libraries = _row_sums(adata_out.X)
    zero_positions = np.flatnonzero(libraries == 0)
    model_x = model.train_dataset.tensor_dict['x'].detach().cpu().numpy()
    observed_x = adata_out.X.toarray() if sp.issparse(adata_out.X) else adata_out.X
    likelihood_exact = bool(np.array_equal(model_x, np.asarray(observed_x)))
    if not likelihood_exact:
        raise RuntimeError('STAMP likelihood matrix differs from raw fixed-panel counts')
    if int(state.get('background_zero_library_rows', -1)) != len(zero_positions):
        raise RuntimeError('background zero-row count differs from the raw matrix')
    if int(state.get('encoder_zero_library_rows', -1)) != len(zero_positions):
        raise RuntimeError('encoder zero-row count differs from the raw matrix')
    init_bg = torch.as_tensor(model.init_bg_mean)
    sgc_x = model.train_dataset.tensor_dict['sgc_x']
    if not bool(torch.isfinite(init_bg).all().item()):
        raise FloatingPointError('zero-safe STAMP background remains non-finite')
    if not bool(torch.isfinite(sgc_x).all().item()):
        raise FloatingPointError('zero-safe STAMP encoder input remains non-finite')
    if len(zero_positions) and float(state.get('zero_row_encoder_norm_min', 0.0)) <= 0:
        raise FloatingPointError('a raw-zero row received no spatial encoder signal')
    source_spot_ids = adata_out.obs['source_spot_id'].astype(str).to_numpy()
    batches = adata_out.obs['batch'].astype(str).to_numpy()
    return {'strategy': STRATEGY, 'activated': bool(len(zero_positions)), 'raw_fixed_panel_counts_modified': False, 'likelihood_uses_exact_raw_fixed_panel_counts': likelihood_exact, 'zero_library_row_count': int(len(zero_positions)), 'zero_library_composite_obs_names': adata_out.obs_names[zero_positions].tolist(), 'zero_library_source_spot_ids': source_spot_ids[zero_positions].tolist(), 'zero_library_batches': batches[zero_positions].tolist(), 'zero_library_positions': zero_positions.astype(int).tolist(), 'background_rule': state.get('background_path'), 'encoder_rule': state.get('encoder_path'), 'background_all_finite': bool(torch.isfinite(init_bg).all().item()), 'encoder_all_finite': bool(torch.isfinite(sgc_x).all().item()), 'zero_row_encoder_norm_min': state.get('zero_row_encoder_norm_min'), 'zero_row_encoder_norm_max': state.get('zero_row_encoder_norm_max')}

def _full_input_support_for_zero_rows(adata_ref, adata_query, safeguard: dict[str, object]) -> list[dict[str, object]]:
    """Describe the full-gene support proving each zero is panel-induced."""
    rows = []
    for source_spot_id, batch in zip(safeguard['zero_library_source_spot_ids'], safeguard['zero_library_batches']):
        source = adata_ref if batch == 'ref' else adata_query
        position = source.obs_names.get_loc(source_spot_id)
        values = source.X[position]
        if sp.issparse(values):
            library_size = float(values.sum())
            detected_genes = int(values.count_nonzero())
        else:
            values = np.asarray(values).reshape(-1)
            library_size = float(values.sum())
            detected_genes = int(np.count_nonzero(values))
        if library_size <= 0 or detected_genes <= 0:
            raise ValueError('a fixed-panel zero row is also empty on the full simulation input')
        rows.append({'batch': batch, 'source_spot_id': source_spot_id, 'full_input_n_genes': int(source.n_vars), 'full_input_library_size': library_size, 'full_input_detected_genes': detected_genes, 'fixed_panel_library_size': 0, 'classification': 'panel_induced_zero_library'})
    return rows

def main() -> int:
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
    parser.add_argument('--learning-rate', type=float, default=0.005)
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
    parser.add_argument('--zero-library-strategy', type=str, required=True)
    args = parser.parse_args()
    if args.zero_library_strategy != STRATEGY:
        print('ERROR: undeclared zero-library strategy', file=sys.stderr)
        return 1
    required_inputs = (args.reference, args.query, args.panel_file, args.panel_provenance, args.config_path)
    if not all((path.is_file() for path in required_inputs)):
        print('ERROR: one or more declared inputs do not exist', file=sys.stderr)
        return 1
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f'ERROR: refusing nonempty output: {args.output_dir}', file=sys.stderr)
        return 1
    sc.settings.verbosity = 0
    adata_ref = sc.read_h5ad(args.reference)
    adata_query = sc.read_h5ad(args.query)
    panel_genes, panel_metadata = base.load_fixed_panel(args.panel_file, expected_n_genes=args.expected_panel_size)
    panel_metadata['provenance_path'] = str(args.panel_provenance.resolve())
    print(f'STAMP zero-safe: ref={adata_ref.n_obs}x{adata_ref.n_vars}, query={adata_query.n_obs}x{adata_query.n_vars}')
    try:
        safeguard_state: dict[str, object] = {}
        with _zero_library_safeguard(safeguard_state) as stamp_module:
            adata_out, meta, model = base.run_stamp_batch_correction(adata_ref, adata_query, panel_genes=panel_genes, panel_metadata=panel_metadata, n_topics=args.n_topics, n_layers=args.n_layers, hidden_size=args.hidden_size, mode=args.mode, learning_rate=args.learning_rate, max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience, device=args.device, seed=args.seed)
        safeguard = _validate_zero_library_contract(adata_out, model, safeguard_state)
        safeguard['full_input_support_for_zero_rows'] = _full_input_support_for_zero_rows(adata_ref, adata_query, safeguard)
        meta['zero_library_safeguard'] = safeguard
        args.output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = args.output_dir / 'embeddings.npz'
        topics_path = args.output_dir / 'cell_by_topic.csv'
        result_path = args.output_dir / 'result.h5ad'
        checkpoint_path = args.output_dir / 'stamp_params.pt'
        np.savez(embeddings_path, embedding=adata_out.obsm['X_STAMP'], batch_labels=adata_out.obs['batch'].values)
        adata_out.uns['STAMP_cell_by_topic'].to_csv(topics_path)
        adata_out.write_h5ad(result_path, compression='gzip')
        model.save(checkpoint_path)
        meta['checkpoint_validation'] = base._validate_saved_param_store(checkpoint_path)
        import sctm.data as sctm_data
        import sctm.layers as sctm_layers
        installed_sources = [Path(inspect.getfile(stamp_module)).resolve(), Path(inspect.getfile(sys.modules['sctm.utils'])).resolve(), Path(inspect.getfile(sys.modules['sctm.model'])).resolve(), Path(inspect.getfile(sctm_data)).resolve(), Path(inspect.getfile(sctm_layers)).resolve()]
        meta['zero_library_safeguard']['installed_sctm_source_files'] = [{'path': str(path)} for path in installed_sources]
        meta['source'] = {'path': str(Path(__file__).resolve())}
        meta['provenance'] = base.build_provenance(configuration_id=args.config_id, seed=args.seed, inputs=[args.reference, args.query, args.panel_file, args.panel_provenance], outputs=[embeddings_path, topics_path, result_path, checkpoint_path], solver_diagnostics={'status': 'completed', 'embedding_all_finite': bool(np.isfinite(adata_out.obsm['X_STAMP']).all()), 'actual_device': meta['actual_device'], 'cuda_execution_verified': meta['cuda_execution_verified'], 'peak_cuda_memory_allocated_bytes': meta['peak_cuda_memory_allocated_bytes'], 'cross_batch_edge_count': meta['spatial_graph']['cross_batch_edge_count'], 'learning_rate': args.learning_rate, 'epochs_run': meta['training']['epochs_run'], 'stopping_reason': meta['training']['stopping_reason'], 'all_losses_finite': meta['training']['all_losses_finite'], 'all_model_parameters_finite': meta['training']['all_model_parameters_finite'], 'all_pyro_param_store_parameters_finite': meta['training']['all_pyro_param_store_parameters_finite'], **meta['fixed_panel_input_matrices'], 'checkpoint_round_trip_status': meta['checkpoint_validation']['status'], 'zero_library_strategy': STRATEGY, 'zero_library_row_count': safeguard['zero_library_row_count'], 'raw_fixed_panel_counts_modified': False, 'likelihood_uses_exact_raw_fixed_panel_counts': True, 'background_all_finite': safeguard['background_all_finite'], 'encoder_all_finite': safeguard['encoder_all_finite']}, candidate_context={'parent_configuration_id': args.parent_config_id, 'method': 'STAMP', 'deformation_mode': args.deformation_mode, 'alpha': float(args.alpha), 'alpha_stratum': 'interpolation' if args.alpha <= 1.0 else 'extrapolation'}, sources=[Path(__file__).resolve(), Path(base.__file__).resolve(), Path(base.__file__).with_name('fixed_panel.py'), Path(base.__file__).with_name('graph_contract.py'), Path(base.__file__).with_name('provenance.py'), Path(__file__).resolve().parents[1] / 'run.py', args.config_path.resolve(), *installed_sources])
        (args.output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
        print(f"  -> {adata_out.n_obs} spots, {meta['n_topics']} topics, zero rows={safeguard['zero_library_row_count']} in {meta['elapsed_seconds']:.1f}s")
        return 0
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(f'  -> FAILED: {error}', file=sys.stderr)
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
