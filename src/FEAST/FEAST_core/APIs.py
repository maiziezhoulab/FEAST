import anndata as ad
import numpy as np
from typing import Union, List, Dict, Optional, Any
import warnings

# Import core simulation functions
from .simulator import simulate_single_slice, SpatialSimulator
# from .parameter_cloud_interpolation import (
#     batch_interpolate_parameter_clouds,
#     interpolate_slice_with_statistical_modeling,
#     interpolate_parameter_clouds_with_continuous_ot
# )

# from .continuous_ot_interpolation import (
#     interpolate_with_continuous_ot,
#     interpolate_with_trajectory_ot,
#     validate_interpolation_quality
# )

# Import specialized simulators
from ..alignment.alignment_simulator import (
    AlignmentSimulator,
    simulate_alignment_rotation,
    simulate_alignment_warp,
    generate_alignment_benchmark_suite
)

from ..deconvolution.deconvolution_simulator import (
    DeconvolutionSimulator,
    simulate_deconvolution_from_single_cells,
    create_deconvolution_benchmark_suite
)

from ..deconvolution.generate_deconvolution import (
    create_deconvolution_benchmark_data
)

from ..deconvolution.deconvolution_simulator import (
    DeconvolutionSimulator,
    simulate_deconvolution_from_single_cells,
    create_deconvolution_benchmark_suite
)


class FEAST:
    """
    Unified FEAST API for all spatial transcriptomics simulation tasks.
    
    This class provides a single entry point for:
    - Single slice simulation
    - Alignment simulation 
    - Deconvolution simulation
    - Multi-slice reconstruction
    
    The simulator automatically detects whether you're working with single or
    multiple slices and provides appropriate methods.
    """
    
    def __init__(self, adata: Union[ad.AnnData, List[ad.AnnData]], verbose: bool = True):
        """
        Initialize FEAST with spatial transcriptomics data.
        
        Parameters:
        -----------
        adata : AnnData or List[AnnData]
            Single AnnData object for single slice operations, or list of
            AnnData objects for multi-slice reconstruction
        verbose : bool
            Whether to print progress messages
        """
        self.verbose = verbose
        self.is_multi_slice = isinstance(adata, list)
        
        if self.is_multi_slice:
            self.adata_list = adata
            self.n_slices = len(adata)
            if self.verbose:
                print(f"✓ FEAST initialized with {self.n_slices} slices")
                for i, slice_data in enumerate(adata):
                    print(f"  Slice {i}: {slice_data.shape}")
        else:
            self.adata = adata
            if self.verbose:
                print(f"✓ FEAST initialized with single slice: {adata.shape}")
        
        # Initialize specialized simulators (lazy loading)
        self._core_simulator = None
        self._alignment_simulator = None  
        self._deconvolution_simulator = None
        self._deconvolution_simulator = None
    
    def _get_core_simulator(self):
        """Lazy initialization of core simulator."""
        if self._core_simulator is None:
            if self.is_multi_slice:
                warnings.warn("Core simulator requires single slice. Using first slice.")
                self._core_simulator = SpatialSimulator(self.adata_list[0])
            else:
                self._core_simulator = SpatialSimulator(self.adata)
        return self._core_simulator
    
    def _get_alignment_simulator(self):
        """Lazy initialization of alignment simulator.""" 
        if self._alignment_simulator is None:
            if self.is_multi_slice:
                warnings.warn("Alignment simulator requires single slice. Using first slice.")
                data_to_use = self.adata_list[0]
            else:
                data_to_use = self.adata
            self._alignment_simulator = AlignmentSimulator(data_to_use, verbose=self.verbose)
        return self._alignment_simulator
    
    def _get_deconvolution_simulator(self):
        """Lazy initialization of deconvolution simulator."""
        if self._deconvolution_simulator is None:
            self._deconvolution_simulator = DeconvolutionSimulator(verbose=self.verbose)
        return self._deconvolution_simulator
    
    def _get_deconvolution_simulator(self):
        """Lazy initialization of deconvolution simulator."""
        if self._deconvolution_simulator is None:
            self._deconvolution_simulator = DeconvolutionSimulator(verbose=self.verbose)
        return self._deconvolution_simulator
    
    # ===============================
    # SINGLE SLICE SIMULATION METHODS  
    # ===============================
    
    def simulate_single_slice(self, 
                            sigma: float = 1.0,
                            follower_sigma_factor: float = 0.1,
                            visualize_fits: bool = False,
                            num_simulation_cores: int = 12,
                            verbose: Optional[bool] = None,
                            clip_overshoot_factor: float = 0.1,
                            use_real_stats_directly: bool = False,
                            annotation_key: Optional[str] = None,
                            use_heuristic_search: bool = False,
                            min_accepted_error: float = 0.5,
                            assignment_weights: Optional[Dict] = None,
                            screening_pool_size: int = 100,
                            top_n_to_fully_evaluate: int = 10,
                            n_jobs: int = -1,
                            alteration_config: Optional[Any] = None,
                            boundary_multiplier: float = 1.1,
                            **kwargs) -> ad.AnnData:
        """
        Generate a single simulated spatial transcriptomics slice.
        
        Parameters:
        -----------
        sigma : float, default=1.0
            Spatial smoothness parameter for Gene-Spatial Relevance Based Assignment (G-SRBA).
            - sigma=0: Perfect pattern preservation (zero spatial change)
            - sigma=0.5-1.5: "Gentle" mode, introduces subtle, local variations  
            - sigma>2.0: "Exploratory" mode, introduces more significant changes
        follower_sigma_factor : float, default=0.1
            Factor for follower gene spatial smoothness (legacy parameter, now uses correlation-guided interpolation)
        visualize_fits : bool, default=False
            Whether to show fitting visualization plots
        num_simulation_cores : int, default=12
            Number of cores for parallel processing during simulation
        verbose : bool, optional
            Override default verbosity setting
        clip_overshoot_factor : float, default=0.1
            Factor for clipping expression overshoot during simulation
        use_real_stats_directly : bool, default=False
            Whether to use real statistics directly instead of parameter cloud fitting
        annotation_key : str, optional
            Key in adata.obs for annotation-based simulation
        use_heuristic_search : bool, default=False
            Whether to use heuristic search for parameter assignment
        min_accepted_error : float, default=0.5
            Minimum accepted error threshold for heuristic search
        assignment_weights : Dict, optional
            Weights for parameter assignment {'mean': 3.0, 'variance': 1.0, 'zero_prop': 1.0}
        screening_pool_size : int, default=100
            Size of screening pool for heuristic search
        top_n_to_fully_evaluate : int, default=10
            Number of top candidates to fully evaluate in heuristic search
        n_jobs : int, default=-1
            Number of parallel jobs (-1 uses all available cores)
        alteration_config : AlterationConfig, optional
            Configuration for marginal distribution alterations
        boundary_multiplier : float, default=1.1
            Multiplier for maximum count boundary constraint (1.1 = 110% of reference max)
        **kwargs
            Additional parameters passed to simulate_single_slice
            
        Returns:
        --------
        AnnData
            Simulated spatial transcriptomics data
        """
        if self.is_multi_slice:
            warnings.warn("Single slice simulation with multi-slice input. Using first slice.")
            adata_to_use = self.adata_list[0]
        else:
            adata_to_use = self.adata


        if verbose is None:
            verbose = self.verbose
            
        return simulate_single_slice(
            adata_to_use,
            sigma=sigma,
            follower_sigma_factor=follower_sigma_factor,
            visualize_fits=visualize_fits,
            num_simulation_cores=num_simulation_cores,
            verbose=verbose,
            clip_overshoot_factor=clip_overshoot_factor,
            use_real_stats_directly=use_real_stats_directly,
            annotation_key=annotation_key,
            use_heuristic_search=use_heuristic_search,
            min_accepted_error=min_accepted_error,
            assignment_weights=assignment_weights,
            screening_pool_size=screening_pool_size,
            top_n_to_fully_evaluate=top_n_to_fully_evaluate,
            n_jobs=n_jobs,
            alteration_config=alteration_config,
            boundary_multiplier=boundary_multiplier,
            **kwargs
        )
    
    # ===============================
    # ALIGNMENT SIMULATION METHODS
    # ===============================
    
    def simulate_alignment(self,
                         transformation_type: str = 'rotation',
                         rotation_angle: float = 0,
                         warp_strength: float = 0,
                         data_type: str = 'imaging',
                         filter_edge_spots: bool = True,
                         edge_margin_ratio: float = 0.03,
                         fit_params: Optional[Dict] = None,
                         expression_params: Optional[Dict] = None,
                         sigma: float = 0,
                         follower_sigma_factor: float = 0,
                         visualize_fits: bool = False,
                         num_simulation_cores: int = 12,
                         clip_overshoot_factor: float = 0.1,
                         use_real_stats_directly: bool = False,
                         annotation_key: Optional[str] = None,
                         use_heuristic_search: bool = False,
                         min_accepted_error: float = 0.5,
                         assignment_weights: Optional[Dict] = None,
                         screening_pool_size: int = 100,
                         top_n_to_fully_evaluate: int = 10,
                         n_jobs: int = -1,
                         alteration_config: Optional[Any] = None,
                         boundary_multiplier: float = 1.1,
                         verbose: Optional[bool] = None,
                         **kwargs) -> tuple:
        """
        Generate alignment simulation data with spatial transformations.
        
        Parameters:
        -----------
        transformation_type : str, default='rotation'
            Type of transformation ('rotation', 'warp', 'cut_move')
        rotation_angle : float, default=45.0
            Rotation angle in degrees (for rotation transformation)
        warp_strength : float, default=0.3
            Warping strength (for TPS warping)
        data_type : str, default='imaging'
            Data type for transformation ('imaging', 'sequencing')
        fit_params : Dict, optional
            Parameters for parameter fitting (merged into simulation parameters)
        expression_params : Dict, optional  
            Parameters for expression generation (merged into simulation parameters)
        
        # Single slice simulation parameters (same as simulate_single_slice):
        sigma : float, default=1.0
            Spatial smoothness parameter for G-SRBA
        follower_sigma_factor : float, default=0.1
            Factor for follower gene spatial smoothness
        visualize_fits : bool, default=False
            Whether to show fitting visualization plots
        num_simulation_cores : int, default=12
            Number of cores for parallel processing
        clip_overshoot_factor : float, default=0.1
            Factor for clipping expression overshoot
        use_real_stats_directly : bool, default=False
            Whether to use real statistics directly instead of parameter cloud fitting
        annotation_key : str, optional
            Key in adata.obs for annotation-based simulation
        use_heuristic_search : bool, default=False
            Whether to use heuristic search for parameter assignment
        min_accepted_error : float, default=0.5
            Minimum accepted error threshold for heuristic search
        assignment_weights : Dict, optional
            Weights for parameter assignment {'mean': 3.0, 'variance': 1.0, 'zero_prop': 1.0}
        screening_pool_size : int, default=100
            Size of screening pool for heuristic search
        top_n_to_fully_evaluate : int, default=10
            Number of top candidates to fully evaluate
        n_jobs : int, default=-1
            Number of parallel jobs (-1 uses all available cores)
        alteration_config : AlterationConfig, optional
            Configuration for marginal distribution alterations
        boundary_multiplier : float, default=1.1
            Multiplier for maximum count boundary constraint
        verbose : bool, optional
            Override default verbosity setting
        **kwargs
            Additional transformation parameters
            
        Returns:
        --------
        tuple
            (original_data, transformed_data) as AnnData objects
        """
        if self.is_multi_slice:
            warnings.warn("Alignment simulation with multi-slice input. Using first slice.")
            adata_to_use = self.adata_list[0]
        else:
            adata_to_use = self.adata
            
        if verbose is None:
            verbose = self.verbose
            
        # Merge all simulation parameters
        simulation_params = {
            'sigma': sigma,
            'follower_sigma_factor': follower_sigma_factor,
            'visualize_fits': visualize_fits,
            'num_simulation_cores': num_simulation_cores,
            'clip_overshoot_factor': clip_overshoot_factor,
            'use_real_stats_directly': use_real_stats_directly,
            'annotation_key': annotation_key,
            'use_heuristic_search': use_heuristic_search,
            'min_accepted_error': min_accepted_error,
            'assignment_weights': assignment_weights,
            'screening_pool_size': screening_pool_size,
            'top_n_to_fully_evaluate': top_n_to_fully_evaluate,
            'n_jobs': n_jobs,
            'alteration_config': alteration_config,
            'boundary_multiplier': boundary_multiplier,
            'verbose': verbose,
        }
        
        # Override with fit_params and expression_params if provided
        if fit_params:
            simulation_params.update(fit_params)
        if expression_params:
            simulation_params.update(expression_params)
        
        # Additional kwargs
        simulation_params.update(kwargs)
            
        if transformation_type == 'rotation':
            return simulate_alignment_rotation(
                adata_to_use,
                rotation_angle=rotation_angle,
                data_type=data_type,
                filter_edge_spots=filter_edge_spots,
                edge_margin_ratio=edge_margin_ratio,
                fit_params=simulation_params,
                expression_params={},  # Already merged above
                **kwargs
            )
        elif transformation_type == 'warp':
            return simulate_alignment_warp(
                adata_to_use,
                distort_level=warp_strength,
                filter_edge_spots=filter_edge_spots,
                edge_margin_ratio=edge_margin_ratio,
                fit_params=simulation_params,
                expression_params={},  # Already merged above
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported transformation type: {transformation_type}")
    
    def simulate_alignment_benchmark(self,
                                   transformations: Optional[List[str]] = None,
                                   parameters: Optional[Dict] = None,
                                   data_types: Optional[List[str]] = None) -> Dict[str, tuple]:
        """
        Generate comprehensive alignment benchmark suite.
        
        Parameters:
        -----------
        transformations : List[str], optional
            List of transformations to test
        parameters : Dict, optional
            Parameters for each transformation type  
        data_types : List[str], optional
            Data types to test
            
        Returns:
        --------
        Dict[str, tuple]
            Dictionary mapping scenario names to (original, transformed) data pairs
        """
        if self.is_multi_slice:
            warnings.warn("Alignment benchmark with multi-slice input. Using first slice.")
            adata_to_use = self.adata_list[0]
        else:
            adata_to_use = self.adata
            
        return generate_alignment_benchmark_suite(
            adata_to_use,
            transformations=transformations,
            parameters=parameters,
            data_types=data_types
        )
    
    
    def simulate_deconvolution(self,
                             cell_type_key: str,
                             downsampling_factor: float = 0.25,
                             cells_per_spot: int = 50,
                             aggregation_method: str = 'sum',
                             fractional_rounding: str = 'probabilistic',
                             use_ot_assignment: bool = True,
                             verbose: Optional[bool] = None) -> ad.AnnData:
        """
        Generate deconvolution simulation data using vine copula + spot marginals approach.
        
        This method implements the sophisticated simulation approach:
        1. Preserves vine copula relationships from real single cell data
        2. Derives marginal distributions from aggregated spots (cells combined into spots)
        3. Uses probabilistic fractional rounding (0.5 → 50% chance of 1, 50% chance of 0)
        4. Applies optimal transport for gene assignment following current simulator methodology
        5. Uses downsampling_factor to determine realistic number of spots to simulate
        
        Parameters:
        -----------
        cell_type_key : str
            Key in adata.obs containing cell type annotations (assumes single-cell input)
        downsampling_factor : float, default=0.25
            Factor to determine number of spots relative to single cells (0.25 = 4x fewer spots than cells)
        cells_per_spot : int, default=50
            Number of cells to aggregate per spot for marginal distribution fitting
        aggregation_method : str, default='sum'
            Method to aggregate cells into spots ('sum', 'mean')
        fractional_rounding : str, default='probabilistic'
            How to handle fractional counts:
            - 'probabilistic': 0.5 → 50% chance of 1, 50% chance of 0
            - 'round': standard rounding
            - 'floor': round down
        use_ot_assignment : bool, default=True
            Whether to use optimal transport for gene assignment (following current simulator)
        verbose : bool, optional
            Override default verbosity setting
            
        Returns:
        --------
        AnnData
            Simulated ST data with realistic spot-level expression and ground truth cell type proportions
        """
        if self.is_multi_slice:
            warnings.warn("Deconvolution simulation with multi-slice input. Using first slice.")
            adata_to_use = self.adata_list[0]
        else:
            adata_to_use = self.adata
            
        if verbose is None:
            verbose = self.verbose
            
        # Calculate number of spots based on downsampling factor
        n_cells = adata_to_use.shape[0]
        n_spots = int(n_cells * downsampling_factor)
        
        if verbose:
            print(f"Deconvolution simulation: {n_cells} cells → {n_spots} spots (factor: {downsampling_factor})")
            
        return simulate_deconvolution_from_single_cells(
            adata_to_use,
            cell_type_key=cell_type_key,
            n_spots=n_spots,
            cells_per_spot=cells_per_spot,
            aggregation_method=aggregation_method,
            fractional_rounding=fractional_rounding,
            verbose=verbose
        )
    
    def create_deconvolution_benchmark(self,
                                     cell_type_key: str,
                                     downsampling_factor: float = 0.25,
                                     grid_type: str = 'hexagonal',
                                     alpha: float = 0.01,
                                     verbose: Optional[bool] = None) -> ad.AnnData:
        """
        Create deconvolution benchmark data using fast spatial downsampling.
        
        This method creates lower-resolution spatial transcriptomics data with ground truth 
        cell type proportions by aggregating spots from existing high-resolution ST data.
        This is fast but just rearranges existing data rather than simulating new data.
        
        Parameters:
        -----------
        cell_type_key : str
            Key in adata.obs containing cell type annotations
        downsampling_factor : float, default=0.25
            Factor to reduce spatial resolution (0.25 = 4x fewer spots, 0.1 = 10x fewer spots)
        grid_type : str, default='hexagonal'
            Type of grid for downsampled spots ('hexagonal', 'square', 'kmeans')
        alpha : float, default=0.01
            Alpha shape parameter for tissue boundary detection
        verbose : bool, optional
            Override default verbosity setting
            
        Returns:
        --------
        AnnData
            Downsampled ST data with ground truth cell type proportions in .obsm['cell_type_proportions']
        """
        if self.is_multi_slice:
            warnings.warn("Deconvolution benchmark with multi-slice input. Using first slice.")
            adata_to_use = self.adata_list[0]
        else:
            adata_to_use = self.adata
            
        if verbose is None:
            verbose = self.verbose
            
        return create_deconvolution_benchmark_data(
            adata_to_use,
            downsampling_factor=downsampling_factor,
            grid_type=grid_type,
            cell_type_key=cell_type_key,
            alpha=alpha
        )
    
    def simulate_deconvolution_benchmark(self,
                                       cell_type_key: Optional[str] = None,
                                       downsampling_factors: List[float] = [0.1, 0.25, 0.5],
                                       grid_types: List[str] = ['hexagonal', 'square', 'kmeans'],
                                       alpha: float = 0.01,
                                       verbose: Optional[bool] = None) -> Dict[str, ad.AnnData]:
        """
        Generate comprehensive deconvolution benchmark suite.
        
        Creates multiple spatial resolution scenarios for testing deconvolution algorithms.
        
        Parameters:
        -----------
        cell_type_key : str, optional
            Key for cell type annotations (for ground truth)
        downsampling_factors : List[float]
            Different resolution reduction factors to test
        grid_types : List[str]
            Different spatial grid types to test  
        alpha : float
            Alpha shape parameter for tissue boundary detection
        verbose : bool, optional
            Override default verbosity setting
            
        Returns:
        --------
        Dict[str, AnnData]
            Dictionary mapping scenario names to benchmark datasets
        """
        if self.is_multi_slice:
            warnings.warn("Deconvolution benchmark with multi-slice input. Using first slice.")
            adata_to_use = self.adata_list[0]
        else:
            adata_to_use = self.adata
            
        if verbose is None:
            verbose = self.verbose
            
        return create_deconvolution_benchmark_suite(
            adata_to_use,
            cell_type_key=cell_type_key,
            downsampling_factors=downsampling_factors,
            grid_types=grid_types,
            alpha=alpha,
            verbose=verbose
        )
    
    # ===============================
    # DECONVOLUTION SIMULATION METHODS 
    # ===============================
    
    def simulate_deconvolution(self,
                                     downsampling_factor: float = 0.25,
                                     grid_type: str = 'hexagonal',
                                     cell_type_key: Optional[str] = None,
                                     alpha: float = 0.01,
                                     # Single slice simulation parameters
                                     sigma: float = 1.0,
                                     visualize_fits: bool = False,
                                     use_heuristic_search: bool = False,
                                     alteration_config: Optional[Any] = None,
                                     boundary_multiplier: float = 1.1,
                                     verbose: Optional[bool] = None,
                                     **simulation_kwargs) -> ad.AnnData:
        """
        Generate deconvolution simulation using two-stage approach:
        1. High-quality single slice simulation 
        2. Spatial aggregation to create deconvolution data
        
        This method combines the robust single slice simulation with spatial
        downsampling to create realistic deconvolution benchmark data.
        
        Parameters:
        -----------
        downsampling_factor : float, default=0.25
            Factor to reduce spatial resolution (0.25 = 4x fewer spots)
        grid_type : str, default='hexagonal'
            Type of spatial grid ('hexagonal', 'square', 'kmeans')
        cell_type_key : str, optional
            Key in adata.obs for cell type annotations (ground truth)
        alpha : float, default=0.01
            Alpha parameter for tissue boundary detection
            
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
        verbose : bool, optional
            Override default verbosity setting
        **simulation_kwargs
            Additional parameters passed to simulate_single_slice
            
        Returns:
        --------
        AnnData
            Deconvolution simulation data with spatial aggregation
        """
        if self.is_multi_slice:
            raise ValueError("Deconvolution simulation requires single slice input data")
            
        if verbose is None:
            verbose = self.verbose
            
        simulator = self._get_deconvolution_simulator()
        
        return simulator.simulate_deconvolution_data(
            reference_adata=self.reference_adata,
            downsampling_factor=downsampling_factor,
            grid_type=grid_type,
            alpha=alpha,
            cell_type_key=cell_type_key,
            sigma=sigma,
            visualize_fits=visualize_fits,
            use_heuristic_search=use_heuristic_search,
            alteration_config=alteration_config,
            boundary_multiplier=boundary_multiplier,
            verbose=verbose,
            **simulation_kwargs
        )
    
    def simulate_deconvolution_benchmark(self,
                                               cell_type_key: Optional[str] = None,
                                               downsampling_factors: List[float] = [0.1, 0.25, 0.5],
                                               grid_types: List[str] = ['hexagonal', 'square'],
                                               sigma_values: List[float] = [0.5, 1.0, 1.5],
                                               alpha: float = 0.01,
                                               verbose: Optional[bool] = None,
                                               **simulation_kwargs) -> Dict[str, ad.AnnData]:
        """
        Create comprehensive deconvolution benchmark suite.
        
        Generates multiple deconvolution scenarios by combining different
        spatial aggregation parameters with single slice simulation parameters.
        
        Parameters:
        -----------
        cell_type_key : str, optional
            Key for cell type annotations in reference data
        downsampling_factors : List[float]
            Different resolution reduction factors to test
        grid_types : List[str]  
            Different spatial grid types to test
        sigma_values : List[float]
            Different spatial smoothness values for single slice simulation
        alpha : float, default=0.01
            Alpha parameter for tissue boundary detection
        verbose : bool, optional
            Override default verbosity setting
        **simulation_kwargs
            Additional parameters for single slice simulation
            
        Returns:
        --------
        Dict[str, AnnData]
            Dictionary of benchmark datasets with descriptive keys
        """
        if self.is_multi_slice:
            raise ValueError("Deconvolution benchmark requires single slice input data")
            
        if verbose is None:
            verbose = self.verbose
            
        simulator = self._get_deconvolution_simulator()
        
        return simulator.create_deconvolution_benchmark_suite(
            reference_adata=self.reference_adata,
            downsampling_factors=downsampling_factors,
            grid_types=grid_types,
            cell_type_key=cell_type_key,
            sigma_values=sigma_values,
            alpha=alpha,
            **simulation_kwargs
        )
    
    