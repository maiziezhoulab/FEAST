
from .deconvolution_simulator import (
    DeconvolutionSimulator,
    simulate_deconvolution_from_single_cells,
    create_deconvolution_benchmark_suite
)

from .generate_deconvolution import (
    create_deconvolution_benchmark_data,
    create_low_resolution_grid,
    filter_grid_to_tissue_shape,
    assign_original_spots_to_grid,
    calculate_cell_type_proportions_for_lowres,
    aggregate_gene_expression
)

__all__ = [
    # Main simulator class
    'DeconvolutionSimulator',
    
    # High-level convenience functions
    'simulate_deconvolution_from_single_cells',
    'create_deconvolution_benchmark_suite',
    
    # Spatial benchmarking utilities
    'create_deconvolution_benchmark_data',
    'create_low_resolution_grid', 
    'filter_grid_to_tissue_shape',
    'assign_original_spots_to_grid',
    'calculate_cell_type_proportions_for_lowres',
    'aggregate_gene_expression'
]