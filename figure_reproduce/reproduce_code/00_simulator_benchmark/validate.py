"""Validate Study 00 inputs, fresh simulations, and atomic metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
ROOT = Path(__file__).parent

def validate_inputs(data_root: Path) -> None:
    manifest = pd.read_csv(ROOT / 'data/input_files.csv')
    if len(manifest) != 60 or manifest[['role', 'simulator', 'sample']].duplicated().any():
        raise RuntimeError('input manifest must contain 60 unique files')
    for row in manifest.itertuples(index=False):
        path = data_root / row.relative_path
        if not path.is_file():
            raise RuntimeError(f'missing input: {path}')

def validate_simulations(root: Path) -> None:
    manifest = pd.read_csv(root / 'simulation_manifest.csv')
    if len(manifest) != 12 or manifest['sample'].nunique() != 12 or (not manifest['status'].eq('ok').all()):
        raise RuntimeError('expected 12 unique successful FEAST simulations')
    for row in manifest.itertuples(index=False):
        path = root / row.file_path
        data = ad.read_h5ad(path)
        provenance = data.uns.get('publication_provenance', {})
        if provenance.get('spatial_mode') != 'reference_rank':
            raise RuntimeError(f'wrong spatial mode: {path}')
        if provenance.get('assignment_solver') != 'scipy' or bool(provenance.get('assignment_blocks')):
            raise RuntimeError(f'wrong assignment contract: {path}')
        if int(provenance.get('public_seed', -1)) != 2026:
            raise RuntimeError(f'wrong public seed: {path}')

def validate_metrics(root: Path) -> None:
    metrics = pd.read_csv(root / 'simulator_quality_metrics.csv')
    if len(metrics) != 60 or metrics[['simulator', 'sample']].duplicated().any():
        raise RuntimeError('expected 60 unique simulator/sample metric rows')
    forbidden = [column for column in metrics if 'composite' in column.casefold()]
    if forbidden:
        raise RuntimeError(f'composite metrics are prohibited: {forbidden}')
    allowed_pairing = {'exact_identifier', 'unique_spatial_coordinate', 'unavailable'}
    if not set(metrics['spot_pairing']).issubset(allowed_pairing):
        raise RuntimeError('metric table contains an undeclared spot-pairing method')
    paired = metrics['spot_pairing'] != 'unavailable'
    finite_pairwise = metrics[['cosine_divergence', 'zero_mask_jaccard']].notna().all(axis=1)
    if not (metrics.loc[paired, 'n_paired_spots'] > 0).all() or not finite_pairwise.equals(paired):
        raise RuntimeError('pairwise metric availability does not match paired support')
    if not metrics.loc[~paired, 'n_paired_spots'].eq(0).all():
        raise RuntimeError('unavailable pairwise rows must have zero paired spots')
    panel_path = root / 'moran_gene_panel.csv.gz'
    panel = pd.read_csv(panel_path if panel_path.is_file() else root / 'moran_gene_panel.csv')
    keys = ['simulator', 'sample']
    values = ['reference_moran_i', 'simulated_moran_i']
    required = {*keys, 'rank', 'gene_id', *values}
    if not required.issubset(panel.columns):
        raise RuntimeError('Moran panel is missing required columns')
    summaries = metrics.set_index(keys)
    groups = panel.groupby(keys)
    if set(groups.groups) != set(summaries.index) or panel.duplicated(keys + ['rank']).any():
        raise RuntimeError('Moran panel and metric table must have matching simulator/sample support')
    for pair, group in groups:
        summary = summaries.loc[pair]
        if len(group) != summary.n_moran_genes:
            raise RuntimeError(f'Moran panel gene count differs for {pair}')
        pairs = group[values].to_numpy(dtype=float)
        finite = pairs[np.isfinite(pairs).all(axis=1)]
        correlation = pearsonr(*finite.T).statistic if len(finite) >= 5 else np.nan
        if not np.isclose(correlation, summary.moran_i_correlation, rtol=0, atol=1e-12, equal_nan=True):
            raise RuntimeError(f'Moran scalar and per-gene profile disagree for {pair}')
    audit_path = root / 'pairing_support_audit.csv'
    if audit_path.is_file():
        audit = pd.read_csv(audit_path)
        keys = ['simulator', 'sample']
        if audit.empty or audit[keys].duplicated().any():
            raise RuntimeError('pairing support audit must contain unique affected rows')
        repaired = audit['repair_status'] == 'repaired'
        if not (audit.loc[repaired, 'coordinate_bijection'].astype(bool).all() and audit.loc[repaired, 'pairing_fraction_reference'].eq(1.0).all() and audit.loc[repaired, 'pairing_fraction_simulated'].eq(1.0).all() and None and audit.loc[repaired, 'n_paired_spots'].eq(audit.loc[repaired, 'n_reference_spots']).all() and audit.loc[repaired, 'n_paired_spots'].eq(audit.loc[repaired, 'n_simulated_spots']).all()):
            raise RuntimeError('repaired coordinate support is not a complete bijection')
        if not audit.loc[~repaired, 'n_paired_spots'].eq(0).all():
            raise RuntimeError('unresolved support audit rows must remain unpaired')
        joined = audit.merge(metrics[keys + ['spot_pairing', 'n_paired_spots']], on=keys, validate='one_to_one', suffixes=('_audit', '_metrics'))
        if not (joined['spot_pairing_audit'].eq(joined['spot_pairing_metrics']).all() and joined['n_paired_spots_audit'].eq(joined['n_paired_spots_metrics']).all()):
            raise RuntimeError('pairing support audit and metric table disagree')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('inputs', 'simulations', 'metrics', 'all'), default='all')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--simulation-dir', type=Path)
    parser.add_argument('--metrics-dir', type=Path)
    args = parser.parse_args()
    if args.stage in {'inputs', 'all'}:
        if args.data_root is None:
            parser.error('--data-root is required')
        validate_inputs(args.data_root)
    if args.stage in {'simulations', 'all'}:
        if args.simulation_dir is None:
            parser.error('--simulation-dir is required')
        validate_simulations(args.simulation_dir)
    if args.stage in {'metrics', 'all'}:
        if args.metrics_dir is None:
            parser.error('--metrics-dir is required')
        validate_metrics(args.metrics_dir)
    print(f'Study 00 {args.stage}: OK')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
