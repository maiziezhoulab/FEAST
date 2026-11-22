"""Run one PASTE2 partial-alignment job and save a rigid coordinate recovery.

The study's fixed-plate design supplies the geometric overlap fraction for each
synthetic pair.  PASTE2 uses this fraction as the transported mass ``s``.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy.spatial import distance

def preprocess(adata: ad.AnnData) -> ad.AnnData:
    result = adata.copy()
    result.var_names_make_unique()
    sc.pp.filter_genes(result, min_cells=1)
    sc.pp.normalize_total(result, target_sum=10000.0)
    sc.pp.log1p(result)
    return result

def add_joint_pca(reference: ad.AnnData, moving: ad.AnnData, n_pcs: int) -> tuple[ad.AnnData, ad.AnnData, int, int]:
    common = reference.var_names.intersection(moving.var_names, sort=False)
    if common.empty:
        raise RuntimeError('reference and moving inputs have no shared genes')
    reference = reference[:, common].copy()
    moving = moving[:, common].copy()
    combined = ad.concat([reference, moving], label='slice', keys=['reference', 'moving'], index_unique='-', join='inner')
    used_pcs = min(n_pcs, combined.n_obs - 1, combined.n_vars - 1)
    if used_pcs < 2:
        raise RuntimeError('insufficient dimensions for joint PCA')
    sc.pp.pca(combined, n_comps=used_pcs, svd_solver='arpack')
    values = np.asarray(combined.obsm['X_pca'], dtype=np.float64)
    reference.obsm['X_pca'] = values[:reference.n_obs]
    moving.obsm['X_pca'] = values[reference.n_obs:]
    return (reference, moving, used_pcs, len(common))

def _scale_distance_matrix(values: np.ndarray, scale: float) -> np.ndarray:
    positive = values[values > 0]
    if positive.size == 0 or scale <= 0 or (not np.isfinite(scale)):
        raise RuntimeError('PASTE2 distance normalization is undefined')
    result = values / positive.min()
    result /= result[result > 0].max()
    return result * scale

def partial_pairwise_align(reference: ad.AnnData, moving: ad.AnnData, overlap_fraction: float, alpha: float, max_iterations: int, relative_tolerance: float, absolute_tolerance: float) -> tuple[np.ndarray, dict[str, object]]:
    """Run the official PASTE2 Partial-FGW primitive with retained diagnostics."""
    from paste2 import PASTE2
    if not 0 < overlap_fraction <= 1:
        raise ValueError('overlap_fraction must be in (0, 1]')
    reference_coords = np.asarray(reference.obsm['spatial'], dtype=np.float64)
    moving_coords = np.asarray(moving.obsm['spatial'], dtype=np.float64)
    expression_cost = distance.cdist(np.asarray(reference.obsm['X_pca'], dtype=np.float64), np.asarray(moving.obsm['X_pca'], dtype=np.float64))
    maximum_cost = float(expression_cost.max())
    reference_distances = _scale_distance_matrix(distance.cdist(reference_coords, reference_coords), maximum_cost)
    moving_distances = _scale_distance_matrix(distance.cdist(moving_coords, moving_coords), maximum_cost)
    reference_weights = np.full(reference.n_obs, 1 / reference.n_obs)
    moving_weights = np.full(moving.n_obs, 1 / moving.n_obs)
    coupling, log = PASTE2.partial_fused_gromov_wasserstein(expression_cost, reference_distances, moving_distances, reference_weights, moving_weights, alpha=alpha, m=overlap_fraction, loss_fun='square_loss', armijo=False, log=True, verbose=False, numItermax=max_iterations, stopThr=relative_tolerance, stopThr2=absolute_tolerance)
    coupling = np.asarray(coupling, dtype=np.float64)
    losses = np.asarray(log.get('loss', []), dtype=np.float64)
    if coupling.shape != (reference.n_obs, moving.n_obs):
        raise RuntimeError('PASTE2 returned an unexpected coupling shape')
    if not np.isfinite(coupling).all() or not np.isfinite(losses).all():
        raise RuntimeError('PASTE2 returned non-finite values')
    if coupling.min() < -1e-12:
        raise RuntimeError('PASTE2 returned a materially negative coupling')
    mass = float(coupling.sum())
    if not np.isclose(mass, overlap_fraction, rtol=1e-07, atol=1e-08):
        raise RuntimeError('PASTE2 coupling mass differs from requested overlap')
    row_excess = float(np.maximum(coupling.sum(axis=1) - reference_weights, 0).max())
    column_excess = float(np.maximum(coupling.sum(axis=0) - moving_weights, 0).max())
    if max(row_excess, column_excess) > 1e-08:
        raise RuntimeError('PASTE2 coupling violates partial marginal constraints')
    if losses.size < 2:
        raise RuntimeError('PASTE2 did not provide enough objective values')
    absolute_delta = float(abs(losses[-1] - losses[-2]))
    relative_delta = float(absolute_delta / abs(losses[-1])) if losses[-1] else np.inf
    converged = bool(relative_delta < relative_tolerance or absolute_delta < absolute_tolerance)
    if not converged:
        raise RuntimeError('PASTE2 exhausted its iteration budget without convergence')
    diagnostics: dict[str, object] = {'status': 'converged', 'positive_convergence_evidence': True, 'overlap_fraction': overlap_fraction, 'coupling': {'shape': list(coupling.shape), 'mass': mass, 'minimum': float(coupling.min()), 'row_constraint_max_excess': row_excess, 'column_constraint_max_excess': column_excess}, 'solver': {'iterations': int(losses.size), 'num_iter_max': max_iterations, 'objective': float(log['partial_fgw_cost']), 'final_absolute_loss_delta': absolute_delta, 'final_relative_loss_delta': relative_delta, 'relative_tolerance': relative_tolerance, 'absolute_tolerance': absolute_tolerance}}
    return (coupling, diagnostics)

def weighted_procrustes(reference_coords: np.ndarray, moving_coords: np.ndarray, coupling: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit a rigid transform using a partial coupling of arbitrary total mass."""
    reference_mass = coupling.sum(axis=1)
    moving_mass = coupling.sum(axis=0)
    mass = float(coupling.sum())
    if mass <= 0:
        raise RuntimeError('cannot fit a transform from zero transport mass')
    reference_center = reference_mass @ reference_coords / mass
    moving_center = moving_mass @ moving_coords / mass
    covariance = (moving_coords - moving_center).T @ (coupling.T @ (reference_coords - reference_center))
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    aligned = (moving_coords - moving_center) @ rotation.T + reference_center
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return (aligned, rotation, angle, reference_center, moving_center)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--moving', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--configuration-id', required=True)
    parser.add_argument('--overlap-fraction', type=float, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        config = yaml.safe_load(args.config.read_text())['paste2']
        np.random.seed(args.seed)
        reference_raw = ad.read_h5ad(args.reference)
        moving_raw = ad.read_h5ad(args.moving)
        if not reference_raw.obs_names.is_unique or not moving_raw.obs_names.is_unique:
            raise RuntimeError('spot identifiers must be unique')
        moving_names = moving_raw.obs_names.copy()
        moving_vars = moving_raw.var_names.copy()
        reference, moving, used_pcs, common_genes = add_joint_pca(preprocess(reference_raw), preprocess(moving_raw), int(config['n_pcs']))
        coupling, diagnostics = partial_pairwise_align(reference, moving, args.overlap_fraction, float(config['alpha']), int(config['max_iterations']), float(config['relative_tolerance']), float(config['absolute_tolerance']))
        reference_coords = np.asarray(reference.obsm['spatial'], dtype=np.float64)
        moving_coords = np.asarray(moving.obsm['spatial'], dtype=np.float64)
        aligned_coords, rotation, angle, reference_center, moving_center = weighted_procrustes(reference_coords, moving_coords, coupling)
        if not moving_raw.obs_names.equals(moving_names) or not moving_raw.var_names.equals(moving_vars):
            raise RuntimeError('PASTE2 preprocessing changed source identity/order')
        if not np.isfinite(aligned_coords).all():
            raise RuntimeError('PASTE2 produced non-finite aligned coordinates')
        output_adata = moving_raw.copy()
        output_adata.obsm['spatial_aligned'] = aligned_coords
        paths = {'aligned': args.output_dir / 'aligned.h5ad', 'coupling': args.output_dir / 'coupling_matrix.npy', 'coordinates': args.output_dir / 'aligned_coordinates.csv', 'diagnostics': args.output_dir / 'diagnostics.json', 'transform': args.output_dir / 'transform.json'}
        output_adata.write_h5ad(paths['aligned'], compression='gzip')
        np.save(paths['coupling'], coupling)
        pd.DataFrame({'spot_barcode': moving_names.astype(str), 'x_moving': moving_coords[:, 0], 'y_moving': moving_coords[:, 1], 'x_aligned': aligned_coords[:, 0], 'y_aligned': aligned_coords[:, 1]}).to_csv(paths['coordinates'], index=False)
        diagnostics['method'] = 'paste2'
        diagnostics['runtime_seconds'] = round(time.monotonic() - started, 3)
        paths['diagnostics'].write_text(json.dumps(diagnostics, indent=2) + '\n')
        transform = {'configuration_id': args.configuration_id, 'method': 'paste2', 'implementation': 'paste2==' + importlib.metadata.version('paste2'), 'status': 'ok', 'public_seed': args.seed, 'runtime_seconds': round(time.monotonic() - started, 3), 'parameters': config | {'overlap_fraction': args.overlap_fraction}, 'n_pcs_used': used_pcs, 'common_gene_count': common_genes, 'rotation_angle': angle, 'rotation_matrix': rotation.tolist(), 'reference_center': reference_center.tolist(), 'moving_center': moving_center.tolist(), 'input_artifacts': {}, 'source_artifacts': {}, 'environment': {'python': platform.python_version(), 'numpy': np.__version__}, 'output_artifacts': {path.name: None for path in (paths['aligned'], paths['coupling'], paths['coordinates'], paths['diagnostics'])}}
        paths['transform'].write_text(json.dumps(transform, indent=2) + '\n')
    except Exception:
        import traceback
        (args.output_dir / 'failure.txt').write_text(traceback.format_exc())
        raise
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
