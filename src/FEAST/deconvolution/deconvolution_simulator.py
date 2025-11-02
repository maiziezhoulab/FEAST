import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from scipy.sparse import issparse
from typing import Optional, Dict, Any, List

# Import core simulation components  
from ..FEAST_core.simulator import simulate_single_slice

global device   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeconvolutionSimulator:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.fitted = False
        
    def simulate_deconvolution_data(self,
                                   reference_adata: ad.AnnData,
                                   downsampling_factor: float = 0.25,
                                   grid_type: str = 'hexagonal',
                                   alpha: float = 0.01,
                                   cell_type_key: Optional[str] = None,
                                   # Single slice simulation parameters
                                   sigma: float = 1.0,
                                   visualize_fits: bool = False,
                                   use_heuristic_search: bool = False,
                                   alteration_config: Optional[Any] = None,
                                   boundary_multiplier: float = 1.1,
                                   **simulation_kwargs) -> ad.AnnData:
        """
        Generate deconvolution simulation data using the two-stage approach.
        
        Parameters:
        -----------
        reference_adata : AnnData
            Reference spatial transcriptomics data
        downsampling_factor : float, default=0.25
            Factor to reduce resolution (0.25 = 4x fewer spots)
        grid_type : str, default='hexagonal'
            Grid type for spatial aggregation ('hexagonal', 'square', 'kmeans')
        alpha : float, default=0.01
            Alpha parameter for tissue boundary detection
        cell_type_key : str, optional
            Key in reference_adata.obs for cell type annotations (for ground truth)
        
        Single Slice Simulation Parameters:
        ----------------------------------
        sigma : float, default=1.0
            Spatial smoothness parameter for G-SRBA algorithm
        visualize_fits : bool, default=False
            Whether to show parameter fitting visualizations
        use_heuristic_search : bool, default=False
            Whether to use heuristic parameter assignment
        alteration_config : AlterationConfig, optional
            Configuration for marginal distribution alterations
        boundary_multiplier : float, default=1.1
            Multiplier for maximum count boundary constraint
        **simulation_kwargs
            Additional parameters passed to simulate_single_slice
            
        Returns:
        --------
        AnnData
            Deconvolution simulation data with aggregated spots and ground truth
        """
            
        # Stage 1: Generate high-quality simulated slice
        if self.verbose:
            print(f"\n--- Stage 1: Single Slice Simulation ---")
            print(f"Reference data: {reference_adata.shape}")
            print(f"Sigma: {sigma}, Boundary multiplier: {boundary_multiplier}")
            
        simulated_slice = simulate_single_slice(
            adata=reference_adata,
            sigma=sigma,
            visualize_fits=visualize_fits,
            use_heuristic_search=use_heuristic_search,
            alteration_config=alteration_config,
            boundary_multiplier=boundary_multiplier,
            verbose=self.verbose,
            **simulation_kwargs
        )
        
        if self.verbose:
            print(f"✓ Simulated slice created: {simulated_slice.shape}")
            
        # Stage 2: Apply spatial downsampling/aggregation
        if self.verbose:
            print(f"\n--- Stage 2: Spatial Aggregation ---")
            print(f"Downsampling factor: {downsampling_factor}")
            print(f"Grid type: {grid_type}")
            print(f"Alpha: {alpha}")
            
        deconvolution_data = self._create_aggregated_spots(
            simulated_slice,
            downsampling_factor=downsampling_factor,
            grid_type=grid_type,
            alpha=alpha,
            cell_type_key=cell_type_key
        )
        
        if self.verbose:
            print(f"✓ Deconvolution data created: {deconvolution_data.shape}")
            print(f"✓ Resolution reduction: {reference_adata.shape[0]} → {deconvolution_data.shape[0]} spots")
            
        # Add simulation metadata (H5AD compatible - only simple types)
        deconvolution_data.uns['simulation_method'] = 'two_stage_simulation'
        deconvolution_data.uns['stage_1'] = 'single_slice_simulation'
        deconvolution_data.uns['stage_2'] = 'spatial_aggregation'
        deconvolution_data.uns['original_spots'] = int(reference_adata.shape[0])
        deconvolution_data.uns['simulated_spots'] = int(simulated_slice.shape[0])
        deconvolution_data.uns['final_spots'] = int(deconvolution_data.shape[0])
        deconvolution_data.uns['downsampling_factor'] = float(downsampling_factor)
        deconvolution_data.uns['grid_type'] = str(grid_type)
        deconvolution_data.uns['alpha'] = float(alpha)
        deconvolution_data.uns['sigma'] = float(sigma)
        deconvolution_data.uns['boundary_multiplier'] = float(boundary_multiplier)
        if alteration_config is not None:
            deconvolution_data.uns['alteration_config'] = str(alteration_config)
        
        if self.verbose:
            print(f"\n=== DECONVOLUTION SIMULATION COMPLETE ===")
            
        return deconvolution_data
        
    def _create_aggregated_spots(self,
                                simulated_adata: ad.AnnData,
                                downsampling_factor: float,
                                grid_type: str,
                                alpha: float,
                                cell_type_key: Optional[str]) -> ad.AnnData:

        from .generate_deconvolution import (
            create_low_resolution_grid,
            filter_grid_to_tissue_shape,
            assign_original_spots_to_grid,
            aggregate_gene_expression,
            calculate_cell_type_proportions_for_lowres
        )
        
        # Extract spatial coordinates and expression matrix
        original_coords = simulated_adata.obsm['spatial']
        
        # Convert expression matrix to dense if sparse
        if hasattr(simulated_adata.X, 'toarray'):
            expression_matrix = simulated_adata.X.toarray()
        else:
            expression_matrix = simulated_adata.X.copy()
            
        if self.verbose:
            print(f"  Creating low-resolution grid...")
            
        # Generate low-resolution grid
        lowres_grid = create_low_resolution_grid(
            original_coords, 
            downsampling_factor=downsampling_factor,
            grid_type=grid_type
        )
        
        if self.verbose:
            print(f"  Initial grid points: {len(lowres_grid)}")
            
        # Filter grid to tissue shape
        lowres_grid = filter_grid_to_tissue_shape(
            original_coords, 
            lowres_grid, 
            alpha=alpha
        )
        
        if self.verbose:
            print(f"  Filtered grid points: {len(lowres_grid)}")
            
        # Assign original spots to low-resolution grid points
        assignments = assign_original_spots_to_grid(original_coords, lowres_grid)
        
        if self.verbose:
            print(f"  Aggregating expression data...")
            
        # Aggregate gene expression
        aggregated_counts = aggregate_gene_expression(
            expression_matrix, 
            assignments, 
            len(lowres_grid)
        )
        
        # Create new AnnData object
        lowres_adata = ad.AnnData(
            X=aggregated_counts,
            var=simulated_adata.var.copy()
        )
        lowres_adata.obsm['spatial'] = lowres_grid
        
        # Calculate and store ground truth cell type proportions if available
        if cell_type_key is not None and cell_type_key in simulated_adata.obs:
            if self.verbose:
                print(f"  Calculating cell type proportions...")
                
            cell_types = simulated_adata.obs[cell_type_key].values
            proportions_df = calculate_cell_type_proportions_for_lowres(
                cell_types, 
                assignments, 
                len(lowres_grid)
            )
            
            # Store proportions in the AnnData object
            lowres_adata.obsm['cell_type_proportions'] = proportions_df.values
            lowres_adata.uns['cell_type_names'] = proportions_df.columns.values
            
            if self.verbose:
                print(f"  ✓ Ground truth proportions calculated for {len(proportions_df.columns)} cell types")
        
        # Store only H5AD-compatible metadata
        lowres_adata.uns['n_assignments'] = int(len(assignments))
        
        # Copy only simple metadata from original simulation
        for key, value in simulated_adata.uns.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                lowres_adata.uns[f'original_{key}'] = value
        
        # Add aggregation metadata (H5AD compatible)
        lowres_adata.uns['aggregation_original_spots'] = int(len(original_coords))
        lowres_adata.uns['aggregation_final_spots'] = int(len(lowres_grid))
        lowres_adata.uns['aggregation_downsampling_factor'] = float(downsampling_factor)
        lowres_adata.uns['aggregation_grid_type'] = str(grid_type)
        lowres_adata.uns['aggregation_alpha'] = float(alpha)
        lowres_adata.uns['has_ground_truth'] = bool(cell_type_key is not None and cell_type_key in simulated_adata.obs)
        
        return lowres_adata
        
    def create_deconvolution_benchmark_suite(self,
                                           reference_adata: ad.AnnData,
                                           downsampling_factors: List[float] = [0.1, 0.25, 0.5],
                                           grid_types: List[str] = ['hexagonal', 'square', 'kmeans'],
                                           cell_type_key: Optional[str] = None,
                                           sigma_values: List[float] = [0.5, 1.0, 1.5],
                                           alpha: float = 0.01,
                                           **simulation_kwargs) -> Dict[str, ad.AnnData]:
        """
        Create a comprehensive deconvolution benchmark suite with multiple parameter combinations.
        
        Parameters:
        -----------
        reference_adata : AnnData
            Reference spatial transcriptomics data
        downsampling_factors : List[float]
            Different resolution reduction factors to test
        grid_types : List[str]
            Different spatial grid types to test
        cell_type_key : str, optional
            Key for cell type annotations
        sigma_values : List[float]
            Different spatial smoothness values to test
        alpha : float
            Alpha parameter for tissue boundary detection
        **simulation_kwargs
            Additional parameters for single slice simulation
            
        Returns:
        --------
        Dict[str, AnnData]
            Dictionary of benchmark datasets with descriptive keys
        """
        if self.verbose:
            print("\n=== CREATING DECONVOLUTION BENCHMARK SUITE ===")
            print(f"Downsampling factors: {downsampling_factors}")
            print(f"Grid types: {grid_types}")
            print(f"Sigma values: {sigma_values}")
            
        benchmark_datasets = {}
        total_combinations = len(downsampling_factors) * len(grid_types) * len(sigma_values)
        current = 0
        
        for downsample in downsampling_factors:
            for grid_type in grid_types:
                for sigma in sigma_values:
                    current += 1
                    
                    if self.verbose:
                        print(f"\n--- Combination {current}/{total_combinations} ---")
                        print(f"Downsampling: {downsample}, Grid: {grid_type}, Sigma: {sigma}")
                        
                    key = f"downsample_{downsample}_grid_{grid_type}_sigma_{sigma}"
                    
                    try:
                        benchmark_data = self.simulate_deconvolution_data(
                            reference_adata=reference_adata,
                            downsampling_factor=downsample,
                            grid_type=grid_type,
                            alpha=alpha,
                            cell_type_key=cell_type_key,
                            sigma=sigma,
                            verbose=False,  # Reduce verbosity for batch processing
                            **simulation_kwargs
                        )
                        
                        benchmark_datasets[key] = benchmark_data
                        
                        if self.verbose:
                            print(f"✓ {key}: {benchmark_data.shape}")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"✗ Failed {key}: {e}")
                        continue
                        
        if self.verbose:
            print(f"\n✓ Benchmark suite complete! Created {len(benchmark_datasets)}/{total_combinations} datasets")
            
        return benchmark_datasets



def simulate_deconvolution_from_single_cells(reference_adata: ad.AnnData,
                                           cell_type_key: Optional[str] = None,
                                           downsampling_factor: float = 0.25,
                                           grid_type: str = 'hexagonal',
                                           sigma: float = 1.0,
                                           alpha: float = 0.01,
                                           verbose: bool = True,
                                           **kwargs) -> ad.AnnData:
    """
    Convenience function for deconvolution simulation using the two-stage approach.
    
    This function implements the two-stage approach:
    1. High-quality single slice simulation
    2. Spatial aggregation to create deconvolution data
    
    Parameters:
    -----------
    reference_adata : AnnData
        Reference spatial transcriptomics data
    cell_type_key : str, optional
        Key for cell type annotations (ground truth)
    downsampling_factor : float, default=0.25
        Factor to reduce spatial resolution
    grid_type : str, default='hexagonal'
        Type of spatial grid for aggregation
    sigma : float, default=1.0
        Spatial smoothness for single slice simulation
    alpha : float, default=0.01
        Alpha parameter for tissue boundary detection
    verbose : bool, default=True
        Whether to print progress messages
    **kwargs
        Additional parameters for single slice simulation
        
    Returns:
    --------
    AnnData
        Deconvolution simulation data
    """
    simulator = DeconvolutionSimulator(verbose=verbose)
    
    return simulator.simulate_deconvolution_data(
        reference_adata=reference_adata,
        downsampling_factor=downsampling_factor,
        grid_type=grid_type,
        alpha=alpha,
        cell_type_key=cell_type_key,
        sigma=sigma,
        **kwargs
    )


def create_deconvolution_benchmark_suite(reference_adata: ad.AnnData,
                                        cell_type_key: Optional[str] = None,
                                        downsampling_factors: List[float] = [0.1, 0.25, 0.5],
                                        sigma_values: List[float] = [0.5, 1.0, 1.5],
                                        verbose: bool = True) -> Dict[str, ad.AnnData]:
    """
    Create a comprehensive deconvolution benchmark suite.
    
    Parameters:
    -----------
    reference_adata : AnnData
        Reference spatial transcriptomics data
    cell_type_key : str, optional
        Key for cell type annotations
    downsampling_factors : List[float]
        Resolution reduction factors to test
    sigma_values : List[float]
        Spatial smoothness values to test
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    Dict[str, AnnData]
        Comprehensive benchmark suite
    """
    simulator = DeconvolutionSimulator(verbose=verbose)
    
    return simulator.create_deconvolution_benchmark_suite(
        reference_adata=reference_adata,
        cell_type_key=cell_type_key,
        downsampling_factors=downsampling_factors,
        sigma_values=sigma_values
    )