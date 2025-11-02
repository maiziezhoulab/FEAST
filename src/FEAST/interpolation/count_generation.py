"""
Count Generation from Interpolated Parameters

This module generates gene expression counts from interpolated parameter clouds,
following the same process as single-slice simulation:
1. Convert parameters to count model parameters (ZIP/ZINB/NB/Poisson)
2. Generate counts from those distributions
3. Assign counts to genes using optimal assignment
4. Apply rank-based spatial assignment using the ordered query slice

Uses boundaries based on the maximum range observed in both reference slices.
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse
from typing import Optional

# Import single-slice simulation functions we'll reuse
from ..FEAST_core.parameter_cloud import convert_params_for_new_simulator
from ..FEAST_core.simulator import SpatialSimulator


def calculate_reference_boundaries(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    boundary_multiplier: float = 1.1
) -> pd.DataFrame:
    """
    Calculate gene expression boundaries from two reference slices.
    
    Takes the MAXIMUM count observed for each gene across both slices,
    then applies a multiplier to allow some variation.
    
    CRITICAL: Uses RAW COUNTS from .layers['counts'] if available!
    
    Args:
        adata1: First reference slice
        adata2: Second reference slice
        boundary_multiplier: Multiplier for max boundary (default 1.1 = 110%)
        
    Returns:
        DataFrame with max_count for each common gene
    """
    # Get common genes
    common_genes = adata1.var_names.intersection(adata2.var_names)
    
    # Extract expression matrices - USE RAW COUNTS!
    adata1_subset = adata1[:, common_genes]
    adata2_subset = adata2[:, common_genes]
    
    if 'counts' in adata1_subset.layers:
        X1 = adata1_subset.layers['counts']
        print("  Using raw counts from adata1.layers['counts'] for boundaries")
    else:
        X1 = adata1_subset.X
        print("  WARNING: Using adata1.X for boundaries (may be log-normalized!)")
    
    if 'counts' in adata2_subset.layers:
        X2 = adata2_subset.layers['counts']
        print("  Using raw counts from adata2.layers['counts'] for boundaries")
    else:
        X2 = adata2_subset.X
        print("  WARNING: Using adata2.X for boundaries (may be log-normalized!)")
    
    if issparse(X1):
        X1 = X1.toarray()
    if issparse(X2):
        X2 = X2.toarray()
    
    # Calculate max counts per gene across both slices
    max_counts_slice1 = np.max(X1, axis=0)
    max_counts_slice2 = np.max(X2, axis=0)
    
    # Take the maximum of the two
    max_counts = np.maximum(max_counts_slice1, max_counts_slice2)
    
    # Apply boundary multiplier
    boundaries = max_counts * boundary_multiplier
    
    boundaries_df = pd.DataFrame({
        'max_count': boundaries
    }, index=common_genes)
    
    return boundaries_df


def generate_counts_from_interpolated_parameters(
    parameter_cloud: pd.DataFrame,
    ordered_query_slice: ad.AnnData,
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    boundary_multiplier: float = 1.1,
    sigma: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> ad.AnnData:

    if random_seed is not None:
        np.random.seed(random_seed)
    
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 4: Finalize Interpolated Expression")
        print(f"{'='*70}")
        print(f"  Using distance-weighted expression from ordered query")
        print(f"  Sigma (spatial smoothness): {sigma}")
        print(f"  Note: Expression already in normalized space - no count generation needed!")
    

    stats_df = parameter_cloud.copy()

    stats_df = stats_df.reindex(ordered_query_slice.var_names)


    if verbose:
        print("\n  Converting interpolated parameter cloud into count-model parameters...")

    model_params = convert_params_for_new_simulator(stats_df)

    # Create a SpatialSimulator using the ordered query slice as the spatial reference
    simulator = SpatialSimulator(reference_adata=ordered_query_slice.copy(), model_params=model_params)

    # Generate raw counts per-gene from the fitted/converted marginal distributions
    new_counts = simulator._generate_counts_from_parameters(ordered_query_slice, model_params, verbose=verbose, boundary_multiplier=boundary_multiplier)

    # new_counts is (n_spots, n_genes). We now assign these counts to spatial locations
    # using the ordered_query_slice as a rank-preserving template: for each gene,
    # map the highest generated counts to the spots with highest expression in the ordered query.
    if issparse(ordered_query_slice.X):
        ref_matrix = ordered_query_slice.X.toarray()
    else:
        ref_matrix = ordered_query_slice.X.copy()

    n_spots, n_genes = ref_matrix.shape
    assigned_matrix = np.zeros_like(new_counts, dtype=np.float32)

    if verbose:
        print("\n  Applying rank-based spatial assignment using the ordered query slice...")

    # For each gene, sort reference spots by expression and place generated counts by rank
    for j in range(n_genes):
        try:
            ref_vals = ref_matrix[:, j]
            # Indices of spots sorted descending by reference expression
            ref_order = np.argsort(ref_vals)[::-1]

            # Sort generated counts descending so highest counts go to highest-reference spots
            gene_generated = new_counts[:, j]
            sorted_generated = np.sort(gene_generated)[::-1]

            # If shapes differ (shouldn't), handle safely by truncation or padding with zeros
            L_ref = len(ref_order)
            L_gen = len(sorted_generated)
            if L_gen >= L_ref:
                assigned_matrix[ref_order, j] = sorted_generated[:L_ref]
            else:
                # pad generated values with zeros if fewer
                padded = np.zeros(L_ref, dtype=sorted_generated.dtype)
                padded[:L_gen] = sorted_generated
                assigned_matrix[ref_order, j] = padded
        except Exception as e:
            if verbose:
                print(f"  Warning: rank-assignment failed for gene index {j}: {e}")
            # Fallback: copy generated counts as-is into column
            assigned_matrix[:, j] = new_counts[:, j]

    # Build final AnnData to return
    simulated_adata = ad.AnnData(
        X=assigned_matrix.astype(np.float32),
        obs=ordered_query_slice.obs.copy(),
        var=ordered_query_slice.var.copy(),
        obsm={'spatial': ordered_query_slice.obsm.get('spatial', None).copy() if 'spatial' in ordered_query_slice.obsm else {}}
    )

    simulated_adata.uns['simulation_method'] = 'interpolated_rank_based_assignment'
    simulated_adata.uns['simulation_params'] = {
        'boundary_multiplier': boundary_multiplier,
        'sigma': sigma,
        'note': 'Counts generated from converted parameter cloud and assigned by rank using ordered query slice'
    }

    if verbose:
        X_array = simulated_adata.X.toarray() if issparse(simulated_adata.X) else simulated_adata.X
        print(f"\n  ✓ Interpolated counts generated and rank-assigned")
        print(f"    Shape: {simulated_adata.shape}")
        print(f"    Expression range: [{X_array.min():.2f}, {X_array.max():.2f}]")
        print(f"    Mean expression: {X_array.mean():.4f}")
        print(f"    Sparsity: {(X_array == 0).sum() / X_array.size:.1%}")

    return simulated_adata


def validate_generated_counts(
    counts_adata: ad.AnnData,
    parameter_cloud: Optional[pd.DataFrame] = None,
    ordered_query: Optional[ad.AnnData] = None,
) -> dict:

    """
    Validate generated counts. Backwards-compatible wrapper:
    - If only `counts_adata` is provided, perform basic sanity checks.
    - If `parameter_cloud` and `ordered_query` are also provided, perform
      additional checks comparing shapes and basic rank concordance with
      the ordered query slice.

    Returns a dict with keys: 'valid' (bool), 'issues' (list), and 'metrics' (dict).
    """

    result = {
        'valid': True,
        'issues': [],
        'metrics': {}
    }

    try:
        # Extract count matrix
        X = counts_adata.X.toarray() if issparse(counts_adata.X) else counts_adata.X
    except Exception as e:
        result['valid'] = False
        result['issues'].append(f"Failed to access counts matrix: {e}")
        return result

    # Basic checks: finite and non-negative
    if np.any(~np.isfinite(X)):
        result['valid'] = False
        result['issues'].append("Counts contain NaN or inf values")
        return result

    if np.any(X < 0):
        result['valid'] = False
        result['issues'].append("Counts contain negative values")

    # Basic statistics
    try:
        n_spots = int(getattr(counts_adata, 'n_obs', X.shape[0]))
        n_genes = int(getattr(counts_adata, 'n_vars', X.shape[1]))
    except Exception:
        n_spots, n_genes = X.shape

    result['metrics'] = {
        'n_spots': n_spots,
        'n_genes': n_genes,
        'mean_expression': float(X.mean()),
        'max_expression': float(X.max()),
        'sparsity': float((X == 0).sum() / X.size),
        'total_counts': float(X.sum())
    }

    # Extended validation when ordered_query is provided
    if ordered_query is not None:
        try:
            Xq = ordered_query.X.toarray() if issparse(ordered_query.X) else ordered_query.X
            result['metrics']['ordered_query_shape'] = Xq.shape

            # If gene dims match, compute simple per-gene Spearman rank concordance
            if Xq.shape[1] == X.shape[1] and Xq.shape[1] > 0:
                from scipy.stats import spearmanr
                corrs = []
                for gi in range(X.shape[1]):
                    try:
                        r = spearmanr(X[:, gi], Xq[:, gi]).correlation
                        if np.isfinite(r):
                            corrs.append(r)
                    except Exception:
                        continue
                if corrs:
                    result['metrics']['n_genes_compared'] = len(corrs)
                    result['metrics']['mean_gene_spearman'] = float(np.mean(corrs))
                else:
                    result['metrics']['n_genes_compared'] = 0
                    result['metrics']['mean_gene_spearman'] = np.nan
            else:
                result['metrics']['n_genes_compared'] = 0
                result['metrics']['mean_gene_spearman'] = np.nan
        except Exception as e:
            result['issues'].append(f"Ordered-query comparison failed: {e}")

    # Optionally include parameter_cloud info
    if parameter_cloud is not None:
        try:
            result['metrics']['parameter_cloud_shape'] = tuple(parameter_cloud.shape)
        except Exception:
            pass

    return result
