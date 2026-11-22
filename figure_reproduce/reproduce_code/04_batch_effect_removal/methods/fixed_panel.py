"""Shared fixed-gene-panel contract for the Study 04 method wrappers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
REQUIRED_COLUMNS = {'panel_order', 'gene'}

def load_fixed_panel(panel_path: Path, *, expected_n_genes: int | None=None) -> tuple[list[str], dict]:
    """Load and strictly validate an ordered fixed-panel CSV."""
    panel_path = Path(panel_path)
    if not panel_path.is_file():
        raise FileNotFoundError(f'fixed panel not found: {panel_path}')
    table = pd.read_csv(panel_path, dtype={'gene': str})
    missing_columns = REQUIRED_COLUMNS - set(table.columns)
    if missing_columns:
        raise ValueError(f'fixed panel is missing columns: {sorted(missing_columns)}')
    if table.empty:
        raise ValueError('fixed panel is empty')
    expected_order = list(range(len(table)))
    try:
        observed_order = table['panel_order'].astype(int).tolist()
    except (TypeError, ValueError) as error:
        raise ValueError('fixed panel_order must contain integers') from error
    if observed_order != expected_order:
        raise ValueError('fixed panel_order must be contiguous and start at zero')
    genes = table['gene'].tolist()
    if any((not gene or gene.strip() != gene for gene in genes)):
        raise ValueError('fixed panel contains an empty or padded gene name')
    if len(set(genes)) != len(genes):
        raise ValueError('fixed panel contains duplicate genes')
    if expected_n_genes is not None and len(genes) != expected_n_genes:
        raise ValueError(f'fixed panel has {len(genes)} genes; expected {expected_n_genes}')
    return (genes, {'path': str(panel_path.resolve()), 'n_genes': len(genes)})

def subset_pair_to_fixed_panel(adata_ref, adata_query, genes: list[str]):
    """Return copies of a reference/query pair on exactly ``genes`` in order."""
    if not adata_ref.var_names.is_unique or not adata_query.var_names.is_unique:
        raise ValueError('reference and query gene names must be unique')
    missing_ref = [gene for gene in genes if gene not in adata_ref.var_names]
    missing_query = [gene for gene in genes if gene not in adata_query.var_names]
    if missing_ref or missing_query:
        raise ValueError(f'fixed-panel support is incomplete: reference_missing={len(missing_ref)}, query_missing={len(missing_query)}')
    ref = adata_ref[:, genes].copy()
    query = adata_query[:, genes].copy()
    if ref.var_names.tolist() != genes or query.var_names.tolist() != genes:
        raise RuntimeError('fixed-panel gene order was not preserved during subsetting')
    return (ref, query)

def validate_raw_counts(adata, *, label: str) -> dict:
    """Fail closed unless ``adata.X`` is a finite nonnegative count matrix."""
    values = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X)
    finite = bool(np.isfinite(values).all())
    nonnegative = bool(np.all(values >= 0))
    integral = bool(np.equal(values, np.floor(values)).all())
    if not finite or not nonnegative or (not integral):
        raise ValueError(f'{label} X is not a finite nonnegative integral count matrix')
    return {'label': label, 'n_spots': int(adata.n_obs), 'n_genes': int(adata.n_vars), 'finite': finite, 'nonnegative': nonnegative, 'integral': integral}
