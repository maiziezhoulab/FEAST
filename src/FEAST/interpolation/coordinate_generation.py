import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse, csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Union


def extract_transport_pairs(
    alignment_matrix: Union[np.ndarray, csr_matrix],
    max_pairs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract high-probability transport pairs from alignment matrix.
    
    Args:
        alignment_matrix: Transport/alignment matrix between two slices
        max_pairs: Maximum number of pairs to extract (default: min of matrix dimensions)
        threshold_percentile: Percentile threshold for pair selection
        
    Returns:
        valid_pairs: Array of (slice1_idx, slice2_idx) pairs
        pair_weights: Transport probabilities for each pair
    """
    # Convert to dense numpy array if sparse
    if issparse(alignment_matrix):
        pi_dense = alignment_matrix.toarray()
    else:
        pi_dense = alignment_matrix.copy()

    n1, n2 = pi_dense.shape
    # Default max_pairs to min(n1, n2) if not provided
    if max_pairs is None:
        max_pairs = min(n1, n2)
    else:
        max_pairs = min(int(max_pairs), min(n1, n2))
    # Use Hungarian assignment to find global one-to-one matching maximizing total probability
    from scipy.optimize import linear_sum_assignment

    pi_clean = np.nan_to_num(pi_dense, nan=0.0, posinf=0.0, neginf=0.0)
    if pi_clean.size == 0 or np.all(pi_clean == 0):
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    cost = -pi_clean.copy()
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception as e:
        print(f"  > Hungarian assignment failed: {e}")
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    probs_assigned = pi_clean[row_ind, col_ind]
    mask = probs_assigned > 0
    if not np.any(mask):
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    row_sel = row_ind[mask]
    col_sel = col_ind[mask]
    prob_sel = probs_assigned[mask]

    # sort by probability and truncate to max_pairs
    order = np.argsort(prob_sel)[::-1][:max_pairs]
    selected_pairs = np.column_stack((row_sel[order].astype(int), col_sel[order].astype(int)))
    selected_weights = prob_sel[order].astype(float)
    print(f"  Using {len(selected_pairs)} one-to-one transport pairs (Hungarian) ")
    return selected_pairs, selected_weights


def generate_intermediate_coordinates_weighted(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    alignment_matrix: Union[np.ndarray, csr_matrix],
    t: float = 0.5,
    max_pairs: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate intermediate spatial coordinates using transport-based interpolation.
    
    Coordinates are computed as weighted midpoints of transport pairs, where
    the weight is determined by the transport probability and interpolation parameter t.
    
    Args:
        adata1: First reference slice
        adata2: Second reference slice
        alignment_matrix: Transport matrix from slice1 to slice2
        t: Interpolation parameter (0=slice1, 1=slice2)
        max_pairs: Maximum transport pairs to use
        
    Returns:
        intermediate_coords: Generated spatial coordinates
        valid_pairs: Transport pairs used
        pair_weights: Weights for each pair
    """
    print(f"\n=== Generating Intermediate Coordinates (t={t:.3f}) ===")
    
    # Get spatial coordinates
    coords1 = adata1.obsm['spatial']
    coords2 = adata2.obsm['spatial']

    # Convert alignment matrix to dense numpy array
    if issparse(alignment_matrix):
        pi_dense = alignment_matrix.toarray()
    else:
        pi_dense = alignment_matrix.copy()

    n1, n2 = pi_dense.shape
    # Default max_pairs to min(n1, n2) if not provided
    if max_pairs is None:
        max_pairs = min(n1, n2)
    # Use the canonical extract_transport_pairs (Hungarian-based) to select pairs
    valid_pairs, pair_weights = extract_transport_pairs(pi_dense, max_pairs=max_pairs)
    if valid_pairs.shape[0] == 0:
        print("  > No valid one-to-one pairs selected")
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    else:
        print(f"  Using {len(valid_pairs)} one-to-one transport pairs (from extract_transport_pairs)")

    # If fewer positive-probability pairs were selected than requested, do NOT
    # augment with synthetic pairs here; rely only on transport-backed pairs.
    n_selected = len(valid_pairs)
    if n_selected < max_pairs:
        print(f"  > Note: requested max_pairs={max_pairs} but only {n_selected} transport-backed pairs were found; proceeding with {n_selected} pairs")

    # Get coordinates for paired spots
    coords1_selected = coords1[valid_pairs[:, 0]]
    coords2_selected = coords2[valid_pairs[:, 1]]
    
    # Linear interpolation weighted by t
    intermediate_coords = (1 - t) * coords1_selected + t * coords2_selected
    
    print(f"  ✓ Generated {len(intermediate_coords)} intermediate coordinates")
    print(f"  Coordinate range: X=[{intermediate_coords[:, 0].min():.2f}, {intermediate_coords[:, 0].max():.2f}], "
          f"Y=[{intermediate_coords[:, 1].min():.2f}, {intermediate_coords[:, 1].max():.2f}]")
    
    return intermediate_coords, valid_pairs, pair_weights


def create_ordered_query_slice(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    intermediate_coords: np.ndarray,
    valid_pairs: np.ndarray,
    pair_weights: np.ndarray,
    t: float = 0.5,
    use_normalized: bool = True,
    layer_key: Optional[str] = None
) -> ad.AnnData:
    """
    Create ordered query slice with distance-weighted expression averaging.
    
    This generates the initial expression estimates for the interpolated slice
    by averaging gene expression from the two reference slices, weighted by:
    1. Transport probability (from alignment matrix)
    2. Distance from interpolation parameter t
    
    IMPORTANT: Uses normalized/logged expression to avoid batch effects!
    
    Args:
        adata1: First reference slice
        adata2: Second reference slice
        intermediate_coords: Generated intermediate coordinates
        valid_pairs: Transport pairs (slice1_idx, slice2_idx)
        pair_weights: Transport probabilities
        t: Interpolation parameter (0=slice1, 1=slice2)
        use_normalized: If True, uses log-normalized data from .X (recommended!)
                       If False, uses raw counts (may cause batch effects)
        layer_key: Optional layer to use instead of .X (e.g., 'log1p_norm')
        
    Returns:
        AnnData object with ordered query slice (in normalized space)
    """
    print(f"\n=== Creating Ordered Query Slice ===")
    
    # Get common genes
    common_genes = adata1.var_names.intersection(adata2.var_names)
    print(f"  Common genes: {len(common_genes)}")
    
    # Extract expression matrices for common genes
    adata1_common = adata1[:, common_genes]
    adata2_common = adata2[:, common_genes]
    
    # Choose expression matrix based on normalization preference
    if use_normalized:
        if layer_key is not None:
            # Use specified layer
            if layer_key not in adata1_common.layers or layer_key not in adata2_common.layers:
                raise ValueError(f"Layer '{layer_key}' not found in both AnnData objects")
            X1 = adata1_common.layers[layer_key]
            X2 = adata2_common.layers[layer_key]
            print(f"  Using normalized expression from layer: {layer_key}")
        else:
            # Use .X which should contain normalized/logged data
            X1 = adata1_common.X
            X2 = adata2_common.X
            print(f"  Using normalized expression from .X (log-normalized recommended)")
    else:
        # Use raw counts (not recommended for interpolation due to batch effects)
        print(f"  WARNING: Using raw counts - may introduce more batch effects!")
        if 'counts' in adata1_common.layers:
            X1 = adata1_common.layers['counts']
            X2 = adata2_common.layers['counts']
        else:
            X1 = adata1_common.X
            X2 = adata2_common.X
    
    # Convert to dense if sparse
    X1 = X1.toarray() if issparse(X1) else np.asarray(X1)
    X2 = X2.toarray() if issparse(X2) else np.asarray(X2)
    
    n_spots = len(intermediate_coords)
    n_genes = len(common_genes)
    
    # Initialize expression matrix
    X_intermediate = np.zeros((n_spots, n_genes), dtype=np.float32)
    
    # For each intermediate spot, compute weighted average expression
    print(f"  Computing distance-weighted expression for {n_spots} spots...")
    
    for i, (idx1, idx2) in enumerate(valid_pairs):
        # Get expression from both reference spots
        expr1 = X1[idx1, :]
        expr2 = X2[idx2, :]
        
        # Distance-based weights: closer to slice1 when t=0, closer to slice2 when t=1
        weight1 = (1 - t) * pair_weights[i]
        weight2 = t * pair_weights[i]
        
        # Normalize weights
        total_weight = weight1 + weight2
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight
        
        # Weighted average expression
        X_intermediate[i, :] = weight1 * expr1 + weight2 * expr2
    
    # Create AnnData object
    ordered_query = ad.AnnData(
        X=X_intermediate,
        var=pd.DataFrame(index=common_genes),
        obsm={'spatial': intermediate_coords}
    )
    
    # Add metadata
    ordered_query.uns['interpolation_t'] = t
    ordered_query.uns['method'] = 'distance_weighted_averaging'
    ordered_query.uns['n_transport_pairs'] = len(valid_pairs)
    ordered_query.uns['expression_space'] = 'normalized' if use_normalized else 'raw'
    ordered_query.uns['layer_used'] = layer_key if layer_key else 'X'
    
    print(f"  ✓ Ordered query slice created: {ordered_query.shape}")
    print(f"  Expression space: {'normalized/logged' if use_normalized else 'raw counts'}")
    print(f"  Expression range: [{X_intermediate.min():.2f}, {X_intermediate.max():.2f}]")
    print(f"  Mean expression: {X_intermediate.mean():.4f}")
    
    return ordered_query


def validate_coordinates(
    coords: np.ndarray,
    reference_coords1: np.ndarray,
    reference_coords2: np.ndarray,
    t: float
) -> dict:
    """
    Validate that intermediate coordinates are reasonable.
    
    Args:
        coords: Intermediate coordinates
        reference_coords1: First reference coordinates
        reference_coords2: Second reference coordinates
        t: Interpolation parameter
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'issues': [],
        'metrics': {}
    }
    
    # Check for NaN or inf
    if np.any(~np.isfinite(coords)):
        result['valid'] = False
        result['issues'].append("Coordinates contain NaN or inf values")
        return result
    
    # Check coordinate ranges
    min_ref = np.minimum(reference_coords1.min(axis=0), reference_coords2.min(axis=0))
    max_ref = np.maximum(reference_coords1.max(axis=0), reference_coords2.max(axis=0))
    
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    
    # Allow 10% margin outside reference range
    margin = 0.1 * (max_ref - min_ref)
    
    if np.any(min_coord < min_ref - margin) or np.any(max_coord > max_ref + margin):
        result['issues'].append("Coordinates significantly outside reference range")
    
    # Store metrics
    result['metrics'] = {
        'n_spots': len(coords),
        'coord_min': min_coord,
        'coord_max': max_coord,
        'reference_min': min_ref,
        'reference_max': max_ref
    }
    
    return result
