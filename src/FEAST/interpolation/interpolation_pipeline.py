import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse, csr_matrix
from typing import Optional, Union
from dataclasses import dataclass

# Import from our new modules
from .coordinate_generation import (
    generate_intermediate_coordinates_weighted,
    create_ordered_query_slice,
    validate_coordinates
)

from .parameter_interpolation import (
    interpolate_and_assign_parameters
)

from .count_generation import (
    calculate_reference_boundaries,
    generate_counts_from_interpolated_parameters,
    validate_generated_counts
)


@dataclass
class InterpolationConfig:
    # Interpolation parameter
    t: float = 0.5 
    
    # Coordinate generation
    max_transport_pairs: Optional[int] = None  
    
    # Expression averaging (IMPORTANT for avoiding batch effects!)
    use_normalized: bool = True  # Use normalized/logged expression (RECOMMENDED)
    expression_layer: Optional[str] = None  # Layer to use (None = use .X)
    
    # Parameter cloud interpolation
    ot_method: str = 'sinkhorn'  # 'sinkhorn', 'emd', or 'linear'
    ot_regularization: float = 0.05
    feature_weights: dict = None  # Default: {'mean': 3.0, 'variance': 1.0, 'zero_prop': 1.0}
    identity_bonus: float = 0.3  # For gene name assignment
    
    # Count generation
    boundary_multiplier: float = 1.1  # 110% of max observed
    random_seed: Optional[int] = None
    
    # Sigma parameter for final refinement (like single-slice)
    sigma: float = 0# spatial smoothness (0=perfect preservation, >1=more variation)
    
    # Verbosity
    verbose: bool = True
    
    def __post_init__(self):
        if self.feature_weights is None:
            self.feature_weights = {'mean': 3.0, 'variance': 1.0, 'zero_prop': 1.0}


def interpolate_slices(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    alignment_matrix: Union[np.ndarray, csr_matrix],
    config: Optional[InterpolationConfig] = None,
    **kwargs
) -> ad.AnnData:
    """
    Interpolate between two adjacent spatial transcriptomics slices.
    
    This is the main API function that implements the complete interpolation pipeline:
    
    Pipeline Steps:
    ---------------
    1. Generate intermediate spatial coordinates using transport pairs
    2. Create "ordered query slice" with distance-weighted expression averaging
       (Uses normalized/logged expression by default to avoid batch effects!)
    3. Interpolate parameter clouds using real OT (generates anonymous middle state)
    4. Assign gene names to interpolated parameters based on references
    5. Generate counts from parameters (respecting reference boundaries)
    6. Assign generated counts back to spots using rank-based ordering
    
    Args:
        adata1: First reference slice (must have 'spatial' in obsm)
                CRITICAL: Must have RAW COUNTS in .layers['counts'] for parameter estimation!
                Recommended: Keep normalized data in .X for query slice averaging
        adata2: Second reference slice (must have 'spatial' in obsm)
                CRITICAL: Must have RAW COUNTS in .layers['counts'] for parameter estimation!
                Recommended: Keep normalized data in .X for query slice averaging
        alignment_matrix: Transport/alignment matrix from slice1 to slice2
        config: InterpolationConfig object (or use kwargs)
        **kwargs: Override config parameters (e.g., t=0.5, use_normalized=True)
        
    Returns:
        Interpolated AnnData object with simulated gene expression
        
    Example:
        >>> # CRITICAL: Proper preprocessing for interpolation
        >>> # 1. Save raw counts in .layers['counts']
        >>> slice1.layers['counts'] = slice1.X.copy()
        >>> slice2.layers['counts'] = slice2.X.copy()
        >>> 
        >>> # 2. Normalize for query slice averaging (avoid batch effects)
        >>> sc.pp.normalize_total(slice1, target_sum=1e4)
        >>> sc.pp.log1p(slice1)
        >>> sc.pp.normalize_total(slice2, target_sum=1e4)
        >>> sc.pp.log1p(slice2)
        >>> 
        >>> # 3. Now interpolate (uses .layers['counts'] for parameters, .X for query)
        >>> config = InterpolationConfig(t=0.5, sigma=1.0, use_normalized=True)
        >>> interpolated = interpolate_slices(slice1, slice2, alignment, config)
    """
    # Setup configuration
    if config is None:
        config = InterpolationConfig(**kwargs)
    else:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    t = config.t
    verbose = config.verbose
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"3D Slice Interpolation Pipeline")
        print(f"{'='*70}")
        print(f"Interpolation parameter t: {t:.3f} (0=slice1, 1=slice2)")
        print(f"Reference slices: {adata1.shape} × {adata2.shape}")
        print(f"Alignment matrix: {alignment_matrix.shape}")
        print(f"Expression space: {'normalized/logged' if config.use_normalized else 'raw counts'}")
        print(f"OT method: {config.ot_method}, Regularization: {config.ot_regularization}")
        print(f"Boundary multiplier: {config.boundary_multiplier}")
        print(f"Sigma (spatial smoothness): {config.sigma}")
        print(f"{'='*70}")
    
    # ========================================================================
    # STEP 1: Generate Intermediate Coordinates
    # ========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 1: Generate Intermediate Coordinates")
        print(f"{'='*70}")

    # Diagnostic: show shapes and max_pairs to explain number of generated spots
    try:
        n1, n2 = adata1.shape[0], adata2.shape[0]
        mp = config.max_transport_pairs if hasattr(config, 'max_transport_pairs') else None
        print(f"  > Diagnostics: adata1.n_spots={n1}, adata2.n_spots={n2}, config.max_transport_pairs={mp}")
    except Exception:
        pass
    
    intermediate_coords, valid_pairs, pair_weights = generate_intermediate_coordinates_weighted(
        adata1=adata1,
        adata2=adata2,
        alignment_matrix=alignment_matrix,
        t=t,
        max_pairs=config.max_transport_pairs
    )
    
    # Validate coordinates
    coord_validation = validate_coordinates(
        intermediate_coords,
        adata1.obsm['spatial'],
        adata2.obsm['spatial'],
        t
    )
    
    if not coord_validation['valid'] and verbose:
        print(f"  Warning: Coordinate validation issues: {coord_validation['issues']}")
    
    # ========================================================================
    # STEP 2: Create Ordered Query Slice (Distance-Weighted Averaging)
    # ========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 2: Create Ordered Query Slice")
        print(f"{'='*70}")
    
    ordered_query = create_ordered_query_slice(
        adata1=adata1,
        adata2=adata2,
        intermediate_coords=intermediate_coords,
        valid_pairs=valid_pairs,
        pair_weights=pair_weights,
        t=t,
        use_normalized=config.use_normalized,
        layer_key=config.expression_layer
    )
    
    # ========================================================================
    # STEP 3: Interpolate Parameter Clouds with Real OT
    # ========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 3: Parameter Cloud Interpolation")
        print(f"{'='*70}")
    
    interpolated_params = interpolate_and_assign_parameters(
        adata1=adata1,
        adata2=adata2,
        t=t,
        ot_method=config.ot_method,
        ot_regularization=config.ot_regularization,
        verbose=verbose
    )
    
    # ========================================================================
    # STEP 4: Generate Counts from Interpolated Parameters
    # This uses the COMPLETE single-slice simulation pipeline:
    # - Convert parameters to count models
    # - Generate counts
    # - Assign using rank-based spatial assignment (G-SRBA)
    # ========================================================================
    simulated_adata = generate_counts_from_interpolated_parameters(
        parameter_cloud=interpolated_params,
        ordered_query_slice=ordered_query,
        adata1=adata1,
        adata2=adata2,
        boundary_multiplier=config.boundary_multiplier,
        sigma=config.sigma,
        random_seed=config.random_seed,
        verbose=verbose
    )
    
    # Validate generated counts
    count_validation = validate_generated_counts(
        simulated_adata,
        interpolated_params,
        ordered_query
    )
    
    if not count_validation['valid'] and verbose:
        print(f"  Warning: Count validation issues: {count_validation['issues']}")
    
    # ========================================================================
    # Add Metadata to Final Result
    # ========================================================================
    simulated_adata.uns['interpolation'] = {
        't': t,
        'method': '3d_reconstruction_ot_gsrba',
        'ot_method': config.ot_method,
        'n_reference_pairs': len(valid_pairs),
        'sigma': config.sigma,
        'boundary_multiplier': config.boundary_multiplier,
        'expression_space': 'normalized' if config.use_normalized else 'raw',
        'coordinate_validation': coord_validation,
        'count_validation': count_validation
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Interpolation Complete!")
        print(f"{'='*70}")
        X_array = simulated_adata.X.toarray() if issparse(simulated_adata.X) else simulated_adata.X
        print(f"  Final shape: {simulated_adata.shape}")
        print(f"  Expression range: [{X_array.min():.0f}, {X_array.max():.0f}]")
        print(f"  Mean expression: {X_array.mean():.4f}")
        print(f"  Sparsity: {(X_array == 0).sum() / X_array.size:.1%}")
        print(f"{'='*70}\n")
    
    return simulated_adata


def interpolate_multiple_slices(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    alignment_matrix: Union[np.ndarray, csr_matrix],
    t_values: list = [0.25, 0.5, 0.75],
    config: Optional[InterpolationConfig] = None,
    **kwargs
) -> dict:
    """
    Interpolate multiple intermediate slices between two references.
    
    Args:
        adata1: First reference slice
        adata2: Second reference slice
        alignment_matrix: Alignment matrix
        t_values: List of interpolation parameters
        config: Base configuration (t will be overridden)
        **kwargs: Additional config parameters
        
    Returns:
        Dictionary mapping t -> interpolated AnnData
    """
    if config is None:
        config = InterpolationConfig(**kwargs)
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Interpolating {len(t_values)} intermediate slices")
    print(f"t values: {t_values}")
    print(f"{'='*70}\n")
    
    for i, t in enumerate(t_values):
        print(f"\n{'='*70}")
        print(f"Interpolation {i+1}/{len(t_values)}: t={t:.3f}")
        print(f"{'='*70}")
        
        config.t = t
        results[t] = interpolate_slices(adata1, adata2, alignment_matrix, config)
    
    print(f"\n✓ All {len(t_values)} interpolations complete!")
    
    return results
