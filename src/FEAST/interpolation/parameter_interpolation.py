import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import Optional
import anndata as ad
import ot  # Python Optimal Transport


def calculate_parameter_cloud(adata: ad.AnnData) -> pd.DataFrame:

    from scipy.sparse import issparse
    
    # CRITICAL: Use raw counts if available, not log-normalized data!
    if 'counts' in adata.layers:
        X = adata.layers['counts']
        print("  Using raw counts from .layers['counts'] for parameter calculation")
    else:
        X = adata.X
        print("  WARNING: .layers['counts'] not found, using .X (may be log-normalized!)")
    
    if issparse(X):
        mean_expr = np.ravel(X.mean(axis=0))
        var_expr = np.ravel(X.power(2).mean(axis=0) - np.square(mean_expr))
    else:
        mean_expr = np.mean(X, axis=0)
        var_expr = np.var(X, axis=0)
    
    n_spots = adata.n_obs
    n_zeros = n_spots - np.ravel((X > 0).sum(axis=0))
    zero_prop = n_zeros / n_spots
    
    cloud_df = pd.DataFrame({
        'mean': mean_expr,
        'variance': var_expr,
        'zero_prop': zero_prop
    }, index=adata.var_names)
    
    # Ensure valid values
    cloud_df['mean'] = cloud_df['mean'].clip(lower=1e-10)
    cloud_df['variance'] = cloud_df['variance'].clip(lower=1e-10)
    cloud_df['zero_prop'] = cloud_df['zero_prop'].clip(0, 1)
    
    return cloud_df


def interpolate_parameter_clouds_ot(
    cloud_A: pd.DataFrame,
    cloud_B: pd.DataFrame,
    t: float = 0.5,
    method: str = 'sinkhorn',
    regularization: float = 0.05,
    feature_weights: dict = {'mean': 3.0, 'variance': 1.0, 'zero_prop': 1.0},
    verbose: bool = True
) -> pd.DataFrame:
    """
    Interpolate parameter clouds using real Optimal Transport to infer middle state.
    
    This creates an ANONYMOUS cloud (no gene names) that represents the geometric
    interpolation between the two parameter distributions.
    
    Args:
        cloud_A: Parameter cloud from slice A
        cloud_B: Parameter cloud from slice B
        t: Interpolation parameter (0=A, 1=B)
        method: OT method ('sinkhorn' or 'emd', or 'linear' for simple interpolation)
        regularization: Entropic regularization for Sinkhorn
        feature_weights: Weights for each parameter in transport cost
        verbose: Print progress information
        
    Returns:
        Anonymous interpolated parameter cloud (genes × parameters)
    """
    if verbose:
        print(f"\n=== OT Parameter Cloud Interpolation (t={t:.3f}) ===")
    
    # Ensure clouds have same genes in same order
    common_genes = cloud_A.index.intersection(cloud_B.index)
    if len(common_genes) == 0:
        raise ValueError("No common genes between clouds")
    
    cloud_A = cloud_A.loc[common_genes].sort_index()
    cloud_B = cloud_B.loc[common_genes].sort_index()
    
    if verbose:
        print(f"  Common genes: {len(common_genes)}")
    
    # Extract parameter values
    points_A = cloud_A.values.astype(np.float64)
    points_B = cloud_B.values.astype(np.float64)
    n_genes = len(cloud_A)
    
    # Feature scaling for numerical stability
    scaler = StandardScaler()
    points_A_scaled = scaler.fit_transform(points_A)
    points_B_scaled = scaler.transform(points_B)
    
    # Apply feature weights (default 3,1,1 to emphasize mean differences)
    weight_vector = np.array([
        feature_weights.get('mean', 3.0),
        feature_weights.get('variance', 1.0),
        feature_weights.get('zero_prop', 1.0)
    ])
    
    points_A_weighted = points_A_scaled * weight_vector
    points_B_weighted = points_B_scaled * weight_vector
    
    if verbose:
        print(f"  Feature weights: {dict(zip(['mean', 'variance', 'zero_prop'], weight_vector))}")
    
    # Compute interpolation using OT
    if method in ['sinkhorn', 'emd']:
        anonymous_cloud = _interpolate_with_pot(
            points_A, points_B, points_A_weighted, points_B_weighted,
            t, method, regularization, verbose
        )
    elif method == 'linear':
        if verbose:
            print(f"  Using linear interpolation")
        anonymous_cloud = (1 - t) * points_A + t * points_B
    else:
        raise ValueError(f"Unknown interpolation method: {method}. Use 'sinkhorn', 'emd', or 'linear'.")
    
    # Create anonymous DataFrame (no gene names assigned yet)
    anonymous_df = pd.DataFrame(
        anonymous_cloud,
        columns=['mean', 'variance', 'zero_prop']
    )
    
    # Ensure valid parameter ranges
    anonymous_df['mean'] = anonymous_df['mean'].clip(lower=1e-10)
    anonymous_df['variance'] = anonymous_df['variance'].clip(lower=1e-10)
    anonymous_df['zero_prop'] = anonymous_df['zero_prop'].clip(0, 1)
    
    if verbose:
        print(f"  ✓ Anonymous cloud generated: {anonymous_df.shape}")
        print(f"  Parameter ranges:")
        for col in anonymous_df.columns:
            print(f"    {col}: [{anonymous_df[col].min():.4f}, {anonymous_df[col].max():.4f}]")
    
    return anonymous_df


def _interpolate_with_pot(
    points_A: np.ndarray,
    points_B: np.ndarray,
    points_A_weighted: np.ndarray,
    points_B_weighted: np.ndarray,
    t: float,
    method: str,
    regularization: float,
    verbose: bool
) -> np.ndarray:
    """
    Perform OT-based interpolation using POT library.
    
    Returns:
        Interpolated parameter array
    """
    n_genes = len(points_A)
    
    # Create uniform measures
    mu_a = ot.unif(n_genes)
    mu_b = ot.unif(n_genes)
    
    # Compute cost matrix
    M = ot.dist(points_A_weighted, points_B_weighted, metric='euclidean')
    
    if verbose:
        print(f"  Computing OT plan using {method}...")
    
    # Compute transport plan
    if method == 'sinkhorn':
        transport_plan = ot.sinkhorn(mu_a, mu_b, M, reg=regularization)
    elif method == 'emd':
        transport_plan = ot.emd(mu_a, mu_b, M)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Use transport plan to interpolate
    interpolated = np.zeros_like(points_A)
    
    for gene_a_idx in range(n_genes):
        # Get transport probabilities for this gene
        transport_probs = transport_plan[gene_a_idx, :]
        
        # Normalize
        if transport_probs.sum() > 1e-12:
            transport_probs = transport_probs / transport_probs.sum()
        else:
            transport_probs = np.ones(n_genes) / n_genes
        
        # Compute weighted target in cloud B
        gene_a_start = points_A[gene_a_idx]
        gene_a_target = np.average(points_B, weights=transport_probs, axis=0)
        
        # Linear interpolation along transport trajectory
        interpolated[gene_a_idx] = (1 - t) * gene_a_start + t * gene_a_target
    
    if verbose:
        transport_cost = np.sum(transport_plan * M)
        print(f"  ✓ OT transport cost: {transport_cost:.4f}")
    
    return interpolated


def assign_gene_names_to_anonymous_cloud(
    anonymous_cloud: pd.DataFrame,
    cloud_A: pd.DataFrame,
    cloud_B: pd.DataFrame,
    t: float = 0.5,
    assignment_weights: dict = {'mean': 1.0, 'variance': 1.0, 'zero_prop': 1.0},
    identity_bonus: float = 0.3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Assign gene names to anonymous interpolated cloud using optimal matching.
    
    The assignment is based on minimizing the distance to a "target reference"
    which is the linear interpolation of the two reference clouds.
    
    Args:
        anonymous_cloud: Anonymous interpolated cloud (no gene names)
        cloud_A: Reference cloud from slice A
        cloud_B: Reference cloud from slice B
        t: Interpolation parameter
        assignment_weights: Feature weights for matching
        identity_bonus: Bonus for preserving gene identities (0-1)
        verbose: Print progress
        
    Returns:
        Parameter cloud with assigned gene names
    """
    if verbose:
        print(f"\n=== Assigning Gene Names ===")
    
    # Ensure reference clouds have same genes IN THE SAME ORDER (preserve original order!)
    common_genes = cloud_A.index.intersection(cloud_B.index)
    
    # CRITICAL: Preserve the original gene order from cloud_A, do NOT sort!
    # Sorting would break the correspondence with the input AnnData objects
    cloud_A_ordered = cloud_A.loc[common_genes]  # Keep original order
    cloud_B_ordered = cloud_B.loc[common_genes]  # Match cloud_A's order
    
    # Create target reference (linear interpolation of originals, with preserved order)
    target_cloud = (1 - t) * cloud_A_ordered + t * cloud_B_ordered
    
    if verbose:
        print(f"  Genes to assign: {len(common_genes)}")
        print(f"  Anonymous cloud size: {len(anonymous_cloud)}")
    
    # Scale features for assignment
    scaler = StandardScaler()
    scaler.fit(np.vstack([target_cloud.values, anonymous_cloud.values]))
    
    target_scaled = scaler.transform(target_cloud.values)
    anonymous_scaled = scaler.transform(anonymous_cloud.values)
    
    # Apply feature weights
    weight_vector = np.array([
        assignment_weights.get('mean', 1.0),
        assignment_weights.get('variance', 1.0),
        assignment_weights.get('zero_prop', 1.0)
    ])
    
    # Compute cost matrix
    cost_matrix = cdist(
        target_scaled * weight_vector,
        anonymous_scaled * weight_vector,
        'euclidean'
    )
    
    # Add identity preservation bonus
    if identity_bonus > 0:
        median_cost = np.median(cost_matrix)
        bonus_value = identity_bonus * median_cost
        
        for i in range(min(len(cost_matrix), len(cost_matrix[0]))):
            cost_matrix[i, i] -= bonus_value
        
        if verbose:
            print(f"  Applied identity bonus: {bonus_value:.4f}")
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create final cloud with assigned names
    final_cloud = pd.DataFrame(
        anonymous_cloud.iloc[col_ind].values,
        index=target_cloud.index[row_ind],
        columns=anonymous_cloud.columns
    )
    
    # Calculate assignment quality
    avg_cost = cost_matrix[row_ind, col_ind].mean()
    identity_preservation = (row_ind == col_ind).sum() / len(row_ind)
    
    if verbose:
        print(f"  ✓ Assignment complete")
        print(f"  Average assignment cost: {avg_cost:.4f}")
        print(f"  Identity preservation: {identity_preservation:.1%}")
    
    return final_cloud


def interpolate_and_assign_parameters(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    t: float = 0.5,
    ot_method: str = 'sinkhorn',
    ot_regularization: float = 0.05,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete parameter interpolation pipeline: calculate clouds, interpolate with OT, assign names.
    
    Args:
        adata1: First reference slice
        adata2: Second reference slice  
        t: Interpolation parameter
        ot_method: OT method for interpolation
        ot_regularization: Regularization parameter
        verbose: Print progress
        
    Returns:
        Interpolated parameter cloud with assigned gene names
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Parameter Cloud Interpolation Pipeline (t={t:.3f})")
        print(f"{'='*60}")
    
    # Step 1: Calculate parameter clouds from reference slices
    if verbose:
        print("\nStep 1: Calculating parameter clouds from references...")
    
    cloud_A = calculate_parameter_cloud(adata1)
    cloud_B = calculate_parameter_cloud(adata2)
    
    if verbose:
        print(f"  Cloud A: {cloud_A.shape}")
        print(f"  Cloud B: {cloud_B.shape}")
    
    # Step 2: Interpolate using OT (generates anonymous cloud)
    if verbose:
        print("\nStep 2: Interpolating parameter clouds with OT...")
    
    anonymous_cloud = interpolate_parameter_clouds_ot(
        cloud_A, cloud_B, t=t,
        method=ot_method,
        regularization=ot_regularization,
        verbose=verbose
    )

    # Step 3: Assign gene names using optimal matching (Hungarian)
    if verbose:
        print("\nStep 3: Assigning gene names to interpolated cloud (Hungarian matching)...")

    final_cloud = assign_gene_names_to_anonymous_cloud(
        anonymous_cloud, cloud_A, cloud_B, t=t, verbose=verbose
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Parameter interpolation complete: {final_cloud.shape}")
        print(f"  (gene labels assigned via optimal matching)")
        print(f"{'='*60}")

    return final_cloud
