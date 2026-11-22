"""Compute corrected atomic Study 00 simulator metrics."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.neighbors import NearestNeighbors

def gene_stats(matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sp.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float64)
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        squared = matrix.copy()
        squared.data **= 2
        mean_squared = np.asarray(squared.mean(axis=0)).ravel()
        zero = 1.0 - matrix.getnnz(axis=0) / matrix.shape[0]
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        mean = matrix.mean(axis=0)
        mean_squared = np.square(matrix).mean(axis=0)
        zero = np.mean(matrix == 0, axis=0)
    return (mean, np.maximum(mean_squared - mean ** 2, 0.0), np.asarray(zero))

def log_corr(left: np.ndarray, right: np.ndarray) -> float:
    keep = (left > 1e-10) & (right > 1e-10)
    if keep.sum() < 10:
        return 0.0
    value = pearsonr(np.log1p(left[keep]), np.log1p(right[keep])).statistic
    return float(value) if np.isfinite(value) else 0.0

def zero_jaccard(left, right) -> float:
    total = left.shape[0] * left.shape[1]
    left = sp.csr_matrix(left).astype(bool)
    right = sp.csr_matrix(right).astype(bool)
    zero_intersection = total - left.maximum(right).nnz
    zero_union = total - left.multiply(right).nnz
    return 1.0 if zero_union == 0 else float(zero_intersection / zero_union)

def cosine_divergence(left, right) -> float:
    left = sp.csr_matrix(left).astype(np.float64)
    right = sp.csr_matrix(right).astype(np.float64)
    left.data = np.log1p(left.data)
    right.data = np.log1p(right.data)
    dot = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_norm = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
    right_norm = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
    return float(np.median(1.0 - dot / np.maximum(left_norm * right_norm, 1e-12)))

def paired_spot_indices(ref: ad.AnnData, sim: ad.AnnData, coordinate_decimals: int=8) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Find defensible paired support, preferring identifiers over coordinates."""
    diagnostics: dict[str, object] = {'n_reference_spots': int(ref.n_obs), 'n_simulated_spots': int(sim.n_obs), 'spot_pairing': 'unavailable', 'n_coordinate_overlap': np.nan, 'n_reference_unique_coordinates': np.nan, 'n_simulated_unique_coordinates': np.nan, 'n_reference_duplicate_coordinate_rows': np.nan, 'n_simulated_duplicate_coordinate_rows': np.nan, 'pairing_fraction_reference': 0.0, 'pairing_fraction_simulated': 0.0, 'coordinate_decimals': coordinate_decimals, 'coordinate_bijection': False, 'coordinate_max_abs_delta': np.nan, 'pairing_issue': ''}
    if not ref.obs_names.is_unique or not sim.obs_names.is_unique:
        diagnostics['pairing_issue'] = 'nonunique_observation_identifier'
        return (np.array([], dtype=int), np.array([], dtype=int), diagnostics)
    spots = ref.obs_names[ref.obs_names.isin(sim.obs_names)]
    if len(spots):
        reference_indices = ref.obs_names.get_indexer(spots)
        simulated_indices = sim.obs_names.get_indexer(spots)
        pair_map = np.column_stack((reference_indices, simulated_indices)).astype('<i8')
        diagnostics.update({'spot_pairing': 'exact_identifier', 'pairing_fraction_reference': float(len(spots) / ref.n_obs), 'pairing_fraction_simulated': float(len(spots) / sim.n_obs)})
        return (reference_indices, simulated_indices, diagnostics)
    if 'spatial' not in ref.obsm or 'spatial' not in sim.obsm:
        diagnostics.update({'spot_pairing': 'unavailable', 'pairing_issue': 'missing_spatial_coordinates'})
        return (np.array([], dtype=int), np.array([], dtype=int), diagnostics)
    ref_coords = np.asarray(ref.obsm['spatial'], dtype=float)
    sim_coords = np.asarray(sim.obsm['spatial'], dtype=float)
    if ref_coords.ndim != 2 or sim_coords.ndim != 2 or ref_coords.shape[0] != ref.n_obs or (sim_coords.shape[0] != sim.n_obs) or (ref_coords.shape[1] != sim_coords.shape[1]):
        diagnostics.update({'spot_pairing': 'unavailable', 'pairing_issue': 'incompatible_spatial_shapes'})
        return (np.array([], dtype=int), np.array([], dtype=int), diagnostics)
    if not np.isfinite(ref_coords).all() or not np.isfinite(sim_coords).all():
        diagnostics.update({'spot_pairing': 'unavailable', 'pairing_issue': 'nonfinite_spatial_coordinates'})
        return (np.array([], dtype=int), np.array([], dtype=int), diagnostics)
    columns = [f'coordinate_{index}' for index in range(ref_coords.shape[1])]
    ref_frame = pd.DataFrame(np.round(ref_coords, coordinate_decimals), columns=columns)
    sim_frame = pd.DataFrame(np.round(sim_coords, coordinate_decimals), columns=columns)
    ref_frame['reference_index'] = np.arange(ref.n_obs)
    sim_frame['simulated_index'] = np.arange(sim.n_obs)
    ref_duplicates = ref_frame.duplicated(columns, keep=False)
    sim_duplicates = sim_frame.duplicated(columns, keep=False)
    ref_unique = ref_frame[columns].drop_duplicates()
    sim_unique = sim_frame[columns].drop_duplicates()
    unique_overlap = ref_unique.merge(sim_unique, on=columns, how='inner')
    diagnostics.update({'n_coordinate_overlap': int(len(unique_overlap)), 'n_reference_unique_coordinates': int(len(ref_unique)), 'n_simulated_unique_coordinates': int(len(sim_unique)), 'n_reference_duplicate_coordinate_rows': int(ref_duplicates.sum()), 'n_simulated_duplicate_coordinate_rows': int(sim_duplicates.sum()), 'pairing_fraction_reference': float(len(unique_overlap) / ref.n_obs), 'pairing_fraction_simulated': float(len(unique_overlap) / sim.n_obs)})
    if ref_duplicates.any() or sim_duplicates.any():
        diagnostics.update({'spot_pairing': 'unavailable', 'pairing_issue': 'nonunique_spatial_coordinates'})
        return (np.array([], dtype=int), np.array([], dtype=int), diagnostics)
    overlap = ref_frame.merge(sim_frame, on=columns, how='inner', validate='one_to_one')
    if len(overlap) != ref.n_obs or len(overlap) != sim.n_obs:
        diagnostics.update({'spot_pairing': 'unavailable', 'pairing_issue': 'incomplete_spatial_coordinate_bijection'})
        return (np.array([], dtype=int), np.array([], dtype=int), diagnostics)
    overlap.sort_values('reference_index', inplace=True)
    reference_indices = overlap['reference_index'].to_numpy(dtype=int)
    simulated_indices = overlap['simulated_index'].to_numpy(dtype=int)
    coordinate_delta = np.abs(ref_coords[reference_indices] - sim_coords[simulated_indices])
    pair_map = np.column_stack((reference_indices, simulated_indices)).astype('<i8')
    diagnostics.update({'spot_pairing': 'unique_spatial_coordinate', 'pairing_fraction_reference': 1.0, 'pairing_fraction_simulated': 1.0, 'coordinate_bijection': True, 'coordinate_max_abs_delta': float(coordinate_delta.max(initial=0.0))})
    return (reference_indices, simulated_indices, diagnostics)

def graph(coords: np.ndarray, neighbors: int) -> sp.csr_matrix:
    model = NearestNeighbors(n_neighbors=neighbors + 1).fit(coords)
    raw = model.kneighbors(coords, return_distance=False)
    selected = np.vstack([row[row != index][:neighbors] for index, row in enumerate(raw)])
    rows = np.repeat(np.arange(len(coords)), neighbors)
    return sp.csr_matrix((np.full(rows.size, 1.0 / neighbors), (rows, selected.ravel())), shape=(len(coords), len(coords)))

def moran(matrix, indices: np.ndarray, weights: sp.csr_matrix) -> np.ndarray:
    result = []
    for start in range(0, len(indices), 25):
        block = matrix[:, indices[start:start + 25]]
        values = block.toarray() if sp.issparse(block) else np.asarray(block)
        centered = values - values.mean(axis=0)
        denominator = np.square(centered).sum(axis=0)
        numerator = np.sum(centered * (weights @ centered), axis=0)
        result.extend(np.where(denominator > 0, numerator / denominator, np.nan))
    return np.asarray(result)

def pair_metrics(ref: ad.AnnData, sim: ad.AnnData, max_genes: int, neighbors: int, coordinate_decimals: int=8):
    if not ref.var_names.is_unique or not sim.var_names.is_unique:
        raise ValueError('gene identifiers must be unique')
    genes = ref.var_names[ref.var_names.isin(sim.var_names)]
    if genes.empty:
        raise ValueError('no exact shared gene identifiers')
    ri = ref.var_names.get_indexer(genes)
    si = sim.var_names.get_indexer(genes)
    rm, rv, rz = gene_stats(ref.X[:, ri])
    sm, sv, sz = gene_stats(sim.X[:, si])
    ro, so, pairing = paired_spot_indices(ref, sim, coordinate_decimals)
    if len(ro):
        paired_ref = ref.X[ro, :][:, ri]
        paired_sim = sim.X[so, :][:, si]
        jaccard = zero_jaccard(paired_ref, paired_sim)
        cosine = cosine_divergence(paired_ref, paired_sim)
    else:
        jaccard = cosine = np.nan
    top = np.argsort(rm, kind='stable')[-min(max_genes, len(genes)):][::-1]
    panel = genes[top]
    moran_corr = np.nan
    ref_moran = sim_moran = np.full(len(panel), np.nan)
    if len(panel) >= 5 and 'spatial' in ref.obsm and ('spatial' in sim.obsm):
        ref_weights = graph(np.asarray(ref.obsm['spatial']), neighbors)
        sim_weights = ref_weights if ref.obs_names.equals(sim.obs_names) and np.array_equal(ref.obsm['spatial'], sim.obsm['spatial']) else graph(np.asarray(sim.obsm['spatial']), neighbors)
        ref_moran = moran(ref.X, ref.var_names.get_indexer(panel), ref_weights)
        sim_moran = moran(sim.X, sim.var_names.get_indexer(panel), sim_weights)
        valid = np.isfinite(ref_moran) & np.isfinite(sim_moran)
        if valid.sum() >= 5:
            moran_corr = float(pearsonr(ref_moran[valid], sim_moran[valid]).statistic)
    mean_corr = log_corr(rm, sm)
    variance_corr = log_corr(rv, sv)
    metrics = {'identity_flag': bool(len(ro) and mean_corr >= 0.995 and (variance_corr >= 0.95) and (jaccard >= 0.95)), 'n_common_genes': len(genes), 'n_paired_spots': len(ro), **pairing, 'n_moran_genes': len(panel), 'input_mean_corr': mean_corr, 'input_variance_corr': variance_corr, 'cosine_divergence': cosine, 'relative_error_mean': float(np.mean(np.abs(rm - sm) / (rm + 1e-10))), 'zero_mask_jaccard': jaccard, 'gene_zero_fraction_wasserstein': wasserstein_distance(rz, sz), 'gene_mean_wasserstein': wasserstein_distance(rm, sm), 'gene_variance_wasserstein': wasserstein_distance(rv, sv), 'library_size_wasserstein': wasserstein_distance(np.asarray(ref.X[:, ri].sum(axis=1)).ravel(), np.asarray(sim.X[:, si].sum(axis=1)).ravel()), 'moran_i_correlation': moran_corr}
    panels = [{'rank': rank + 1, 'gene_id': str(gene), 'reference_moran_i': ref_moran[rank], 'simulated_moran_i': sim_moran[rank]} for rank, gene in enumerate(panel)]
    return (metrics, panels)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--simulation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample', action='append', help='Score only this declared sample; repeat to select more than one.')
    parser.add_argument('--feast-only', action='store_true', help='Score FEAST without loading external simulator outputs.')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    selected = set(args.sample or config['samples'])
    unknown = selected - set(config['samples'])
    if unknown:
        raise ValueError(f'unknown samples: {sorted(unknown)}')
    samples = [sample for sample in config['samples'] if sample in selected]
    manifest = pd.read_csv(Path(__file__).parent / 'data/input_files.csv')
    simulations = pd.read_csv(args.simulation_dir / 'simulation_manifest.csv')
    if simulations['sample'].duplicated().any():
        raise RuntimeError('simulation manifest contains duplicate samples')
    if not set(simulations['sample']).issubset(config['samples']):
        raise RuntimeError('simulation manifest contains undeclared samples')
    simulations = simulations.loc[simulations['sample'].isin(selected)]
    if set(simulations['sample']) != selected or not simulations['status'].eq('ok').all():
        raise RuntimeError('simulation manifest must contain every selected sample with successful status')
    simulations = simulations.set_index('sample')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, panels = ([], [])
    for sample in samples:
        ref_record = manifest[(manifest['role'] == 'reference') & (manifest['sample'] == sample)].iloc[0]
        ref_path = args.data_root / ref_record.relative_path
        ref = ad.read_h5ad(ref_path)
        feast_path = args.simulation_dir / 'simulations' / f'{sample}.h5ad'
        candidates = [(config['feast']['simulator_name'], feast_path)]
        for simulator in ([] if args.feast_only else config['external_simulators']):
            record = manifest[(manifest['simulator'] == simulator) & (manifest['sample'] == sample)].iloc[0]
            external_path = args.data_root / record.relative_path
            candidates.append((simulator, external_path))
        for simulator, path in candidates:
            sim = ad.read_h5ad(path)
            metric, panel = pair_metrics(ref, sim, int(config['metrics']['max_moran_genes']), int(config['metrics']['spatial_neighbors']), int(config['metrics'].get('coordinate_pairing_decimals', 8)))
            rows.append({'simulator': simulator, 'sample': sample, **metric})
            panels.extend(({'simulator': simulator, 'sample': sample, **item} for item in panel))
            print(f'{sample}: {simulator}', flush=True)
    metrics_path = args.output_dir / 'simulator_quality_metrics.csv'
    panel_path = args.output_dir / 'moran_gene_panel.csv.gz'
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(panels).to_csv(panel_path, index=False)
    provenance = {'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'feast_version': str(simulations['feast_version'].iloc[0]), 'public_seed': config['public_seed'], 'calculation_randomness': 'none', 'rules': 'exact gene joins; exact named spot joins preferred; complete unique spatial-coordinate bijection at the configured precision used only when identifiers do not overlap; ordered reference-derived Moran panel; no composite score', 'row_count': len(rows), 'outputs': {metrics_path.name: None, panel_path.name: None}}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
