import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union

# Import core simulation functions from the main simulator
from ..FEAST_core.simulator import (
    SpatialSimulator, 
    simulate_single_slice, 
    safe_calculate_qc_metrics
)

# Import alignment transformation classes
from .spatial_align_alter import SpatialTransformer, RotationTransformer, WarpTransformer


def _sanitize_params_for_hdf5(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert AlterationConfig objects to dictionaries for HDF5 serialization.
    
    AnnData's HDF5 writer cannot serialize custom Python objects like AlterationConfig.
    This function converts them to plain dictionaries that can be saved.
    
    Args:
        params: Dictionary that may contain AlterationConfig objects
        
    Returns:
        Sanitized dictionary with AlterationConfig converted to dict, or None if input is None
    """
    if params is None:
        return None
    
    # Use deep copy to handle nested dictionaries properly
    from copy import deepcopy
    sanitized = deepcopy(params)
    
    # Check if alteration_config exists and is an AlterationConfig object
    if 'alteration_config' in sanitized:
        try:
            from ..modeling.marginal_alteration import AlterationConfig
            if isinstance(sanitized['alteration_config'], AlterationConfig):
                print(f"[DEBUG] Converting AlterationConfig to dict: {type(sanitized['alteration_config'])} -> dict")
                sanitized['alteration_config'] = sanitized['alteration_config'].to_dict()
                print(f"[DEBUG] Conversion successful. New type: {type(sanitized['alteration_config'])}")
        except ImportError:
            # If we can't import AlterationConfig, leave it as is
            pass
        except Exception as e:
            print(f"[DEBUG] Error during conversion: {e}")
            import traceback
            traceback.print_exc()
    
    return sanitized


class AlignmentSimulator:
    """
    Comprehensive spatial alignment simulator that combines expression simulation 
    with spatial transformations for benchmarking alignment algorithms.
    """
    
    def __init__(self, reference_adata: ad.AnnData, model_params: dict = None):
        """
        Initialize alignment simulator with reference data.
        
        Args:
            reference_adata: Reference spatial transcriptomics data
            model_params: Pre-fitted model parameters (optional)
        """
        if 'spatial' not in reference_adata.obsm:
            raise ValueError("Reference AnnData must contain 'spatial' coordinates.")
        
        self.reference_adata = reference_adata.copy()
        self.reference_adata.var_names_make_unique()
        self.reference_adata.obs_names_make_unique()
        
        # Initialize the core spatial simulator
        self.core_simulator = SpatialSimulator(self.reference_adata, model_params)
        self._fitted = False
    
    def fit_model(self, **kwargs) -> 'AlignmentSimulator':
        """
        Fit the underlying expression simulation model.
        
        Args:
            **kwargs: Arguments passed to SpatialSimulator.fit_model()
        
        Returns:
            Self for method chaining
        """
        print("Fitting expression simulation model for alignment benchmarking...")
        self.core_simulator.fit_model(**kwargs)
        self._fitted = True
        return self
    
    def simulate_with_rotation(
        self, 
        rotation_angle: float = 30.0,
        data_type: str = 'imaging',
        center_correction: Union[float, np.ndarray] = 0,
        keep_bounds: bool = True,
        min_space: Optional[float] = None,
        max_grid_size: int = 10000,
        expression_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Generate a pair of aligned datasets: original and rotated with expression simulation.
        
        Args:
            rotation_angle: Rotation angle in degrees
            data_type: 'imaging' or 'sequencing' - determines transformation method
            center_correction: Center point for rotation
            keep_bounds: Whether to keep only spots within original bounds
            min_space: Minimal spacing for sequencing data (auto-calculated if None)
            max_grid_size: Maximum grid size for sequencing data
            expression_params: Parameters for expression simulation
        
        Returns:
            Tuple of (original_simulated, rotated_simulated) AnnData objects
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        # Set default expression parameters
        if expression_params is None:
            expression_params = {
                'sigma': 1.0,
                'verbose': True,
                'boundary_multiplier': 1.1
            }
        
        print(f"Generating alignment dataset with {rotation_angle}° rotation ({data_type} method)")
        
        # Step 1: Simulate expression for original data
        print("Simulating expression for original dataset...")
        original_simulated = self.core_simulator.simulate(**expression_params)
        
        # Step 2: Apply spatial rotation transformation
        print(f"Applying {data_type} rotation transformation...")
        rotation_transformer = RotationTransformer(original_simulated)
        
        if data_type.lower() == 'imaging':
            rotated_transformed = rotation_transformer.transform_imaging(
                rotation_angle=rotation_angle,
                center_correction=center_correction,
                keep_bounds=keep_bounds
            )
        elif data_type.lower() == 'sequencing':
            rotated_transformed = rotation_transformer.transform_sequencing(
                rotation_angle=rotation_angle,
                center_correction=center_correction,
                min_space=min_space,
                max_grid_size=max_grid_size
            )
        else:
            raise ValueError("data_type must be 'imaging' or 'sequencing'")
        
        # Step 3: Add metadata about the simulation
        original_simulated.uns['alignment_simulation'] = {
            'type': 'original',
            'rotation_angle': 0,
            'data_type': data_type,
            'expression_params': _sanitize_params_for_hdf5(expression_params)
        }
        
        rotated_transformed.uns['alignment_simulation'] = {
            'type': 'rotated',
            'rotation_angle': rotation_angle,
            'data_type': data_type,
            'expression_params': _sanitize_params_for_hdf5(expression_params)
        }
        
        # Calculate QC metrics for both datasets
        safe_calculate_qc_metrics(original_simulated, verbose=True)
        safe_calculate_qc_metrics(rotated_transformed, verbose=True)
        
        print(f"✓ Alignment dataset generation complete:")
        print(f"  Original: {original_simulated.shape} spots")
        print(f"  Rotated: {rotated_transformed.shape} spots")
        
        return original_simulated, rotated_transformed
    
    def simulate_with_warp(
        self,
        distort_level: float = 100,
        grid_size: int = 3,
        alpha: float = 1.0,
        apply_rotation: bool = True,
        expression_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Generate a pair of datasets: original and warped using thin plate splines.
        
        Args:
            distort_level: Level of spatial distortion
            grid_size: Grid size for control points
            alpha: TPS regularization parameter
            apply_rotation: Whether to apply additional rotation/translation
            expression_params: Parameters for expression simulation
        
        Returns:
            Tuple of (original_simulated, warped_simulated) AnnData objects
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        # Set default expression parameters
        if expression_params is None:
            expression_params = {
                'sigma': 1.0,
                'verbose': True,
                'boundary_multiplier': 1.1
            }
        
        print(f"Generating alignment dataset with TPS warping (distort_level={distort_level})")
        
        # Step 1: Simulate expression for original data
        print("Simulating expression for original dataset...")
        original_simulated = self.core_simulator.simulate(**expression_params)
        
        # Step 2: Apply TPS warping transformation
        print("Applying TPS warping transformation...")
        warp_transformer = WarpTransformer(
            original_simulated,
            distort_level=distort_level,
            grid_size=grid_size,
            alpha=alpha
        )
        warped_transformed = warp_transformer.transform(apply_rotation=apply_rotation)
        
        # Step 3: Add metadata
        original_simulated.uns['alignment_simulation'] = {
            'type': 'original',
            'transformation': 'none',
            'expression_params': _sanitize_params_for_hdf5(expression_params)
        }
        
        warped_transformed.uns['alignment_simulation'] = {
            'type': 'warped',
            'transformation': 'TPS',
            'distort_level': distort_level,
            'grid_size': grid_size,
            'alpha': alpha,
            'apply_rotation': apply_rotation,
            'expression_params': _sanitize_params_for_hdf5(expression_params)
        }
        
        # Calculate QC metrics
        safe_calculate_qc_metrics(original_simulated, verbose=True)
        safe_calculate_qc_metrics(warped_transformed, verbose=True)
        
        print(f"✓ TPS warping simulation complete:")
        print(f"  Original: {original_simulated.shape} spots")
        print(f"  Warped: {warped_transformed.shape} spots")
        
        return original_simulated, warped_transformed
    
    def simulate_with_cut_move(
        self,
        cut_bounds: Tuple[float, float, float, float],
        move_offset: Tuple[float, float],
        expression_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Generate datasets with cut-and-move transformation for alignment benchmarking.
        
        Args:
            cut_bounds: (x_min, x_max, y_min, y_max) for cutting region
            move_offset: (dx, dy) offset for moving the cut region
            expression_params: Parameters for expression simulation
        
        Returns:
            Tuple of (original_simulated, cut_moved_simulated) AnnData objects
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        # Set default expression parameters
        if expression_params is None:
            expression_params = {
                'sigma': 1.0,
                'verbose': True,
                'boundary_multiplier': 1.1
            }
        
        print(f"Generating cut-and-move alignment dataset")
        print(f"  Cut bounds: {cut_bounds}")
        print(f"  Move offset: {move_offset}")
        
        # Step 1: Simulate expression for original data
        print("Simulating expression for original dataset...")
        original_simulated = self.core_simulator.simulate(**expression_params)
        
        # Step 2: Apply cut-and-move transformation
        print("Applying cut-and-move transformation...")
        spatial_transformer = SpatialTransformer(original_simulated)
        
        # Cut the region
        x_min, x_max, y_min, y_max = cut_bounds
        cut_region = spatial_transformer.cut(x_min, x_max, y_min, y_max)
        
        # Move the cut region
        dx, dy = move_offset
        cut_moved = spatial_transformer.move(dx, dy)
        
        # Step 3: Add metadata
        original_simulated.uns['alignment_simulation'] = {
            'type': 'original',
            'transformation': 'none',
            'expression_params': _sanitize_params_for_hdf5(expression_params)
        }
        
        cut_moved.uns['alignment_simulation'] = {
            'type': 'cut_moved',
            'transformation': 'cut_move',
            'cut_bounds': cut_bounds,
            'move_offset': move_offset,
            'expression_params': _sanitize_params_for_hdf5(expression_params)
        }
        
        # Calculate QC metrics
        safe_calculate_qc_metrics(original_simulated, verbose=True)
        safe_calculate_qc_metrics(cut_moved, verbose=True)
        
        print(f"✓ Cut-and-move simulation complete:")
        print(f"  Original: {original_simulated.shape} spots")
        print(f"  Cut-moved: {cut_moved.shape} spots")
        
        return original_simulated, cut_moved
    
    def generate_benchmark_dataset(
        self,
        transformations: Dict[str, Dict[str, Any]],
        expression_params: Optional[Dict[str, Any]] = None,
        base_name: str = "alignment_benchmark"
    ) -> Dict[str, Tuple[ad.AnnData, ad.AnnData]]:
        """
        Generate a comprehensive benchmark dataset with multiple transformations.
        
        Args:
            transformations: Dictionary of transformation configurations
                Example:
                {
                    'rotation_30': {'type': 'rotation', 'angle': 30, 'data_type': 'imaging'},
                    'rotation_60': {'type': 'rotation', 'angle': 60, 'data_type': 'sequencing'},
                    'warp_light': {'type': 'warp', 'distort_level': 50, 'grid_size': 3},
                    'warp_heavy': {'type': 'warp', 'distort_level': 200, 'grid_size': 4},
                    'cut_move': {'type': 'cut_move', 'cut_bounds': (0, 100, 0, 100), 'move_offset': (20, 30)}
                }
            expression_params: Global expression parameters
            base_name: Base name for dataset identification
        
        Returns:
            Dictionary mapping transformation names to (original, transformed) data pairs
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        print(f"Generating comprehensive benchmark dataset with {len(transformations)} transformations")
        
        results = {}
        
        for name, config in transformations.items():
            print(f"\n--- Processing transformation: {name} ---")
            
            transform_type = config.get('type', '').lower()
            
            if transform_type == 'rotation':
                original, transformed = self.simulate_with_rotation(
                    rotation_angle=config.get('angle', 30),
                    data_type=config.get('data_type', 'imaging'),
                    center_correction=config.get('center_correction', 0),
                    keep_bounds=config.get('keep_bounds', True),
                    min_space=config.get('min_space'),
                    max_grid_size=config.get('max_grid_size', 10000),
                    expression_params=expression_params
                )
            
            elif transform_type == 'warp':
                original, transformed = self.simulate_with_warp(
                    distort_level=config.get('distort_level', 100),
                    grid_size=config.get('grid_size', 3),
                    alpha=config.get('alpha', 1.0),
                    apply_rotation=config.get('apply_rotation', True),
                    expression_params=expression_params
                )
            
            elif transform_type == 'cut_move':
                original, transformed = self.simulate_with_cut_move(
                    cut_bounds=config.get('cut_bounds', (0, 100, 0, 100)),
                    move_offset=config.get('move_offset', (20, 20)),
                    expression_params=expression_params
                )
            
            else:
                print(f"Warning: Unknown transformation type '{transform_type}' for {name}")
                continue
            
            # Add benchmark metadata
            original.uns['benchmark_info'] = {
                'dataset_name': base_name,
                'transformation_name': name,
                'is_reference': True
            }
            transformed.uns['benchmark_info'] = {
                'dataset_name': base_name,
                'transformation_name': name,
                'is_reference': False
            }
            
            results[name] = (original, transformed)
            print(f"✓ Completed {name}: original={original.shape}, transformed={transformed.shape}")
        
        print(f"\n✓ Benchmark dataset generation complete with {len(results)} transformation pairs")
        return results


# Convenience functions for quick alignment simulation using single slice simulator directly

def simulate_alignment_rotation(
    adata: ad.AnnData,
    rotation_angle: float = 30.0,
    data_type: str = 'imaging',
    fit_params: Optional[Dict[str, Any]] = None,
    expression_params: Optional[Dict[str, Any]] = None,
    filter_edge_spots: bool = True,
    edge_margin_ratio: float = 0.03,
    **kwargs
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Quick function to generate rotated alignment dataset using simulate_single_slice.
    
    Args:
        adata: Input spatial transcriptomics data
        rotation_angle: Rotation angle in degrees
        data_type: 'imaging' or 'sequencing'
        fit_params: Parameters for model fitting (passed to simulate_single_slice)
        expression_params: Parameters for expression simulation (passed to simulate_single_slice)
        **kwargs: Additional parameters for rotation
    
    Returns:
        Tuple of (original_simulated, rotated_simulated) datasets
    """
    print(f"Quick alignment simulation with {rotation_angle}° rotation using simulate_single_slice")
    
    # Merge fit_params and expression_params for simulate_single_slice
    simulation_params = {}
    if fit_params:
        simulation_params.update(fit_params)
    if expression_params:
        simulation_params.update(expression_params)
    
    # Set defaults if not provided
    if 'verbose' not in simulation_params:
        simulation_params['verbose'] = True
    if 'sigma' not in simulation_params:
        simulation_params['sigma'] = 1.0
    
    # Step 1: Generate original simulated data using single slice simulator
    print("Generating original dataset using simulate_single_slice...")
    original_simulated = simulate_single_slice(adata, **simulation_params)
    
    # Step 2: Apply spatial rotation transformation
    print(f"Applying {data_type} rotation transformation...")
    rotation_transformer = RotationTransformer(original_simulated)
    
    if data_type.lower() == 'imaging':
        rotated_transformed = rotation_transformer.transform_imaging(
            rotation_angle=rotation_angle,
            center_correction=kwargs.get('center_correction', 0),
            keep_bounds=kwargs.get('keep_bounds', True)
        )
    elif data_type.lower() == 'sequencing':
        rotated_transformed = rotation_transformer.transform_sequencing(
            rotation_angle=rotation_angle,
            center_correction=kwargs.get('center_correction', 0),
            min_space=kwargs.get('min_space'),
            max_grid_size=kwargs.get('max_grid_size', 10000)
        )
    else:
        raise ValueError("data_type must be 'imaging' or 'sequencing'")
    
    # Optional: filter edge spots from the transformed slice to remove cut edges
    if filter_edge_spots:
        rotated_before = rotated_transformed.shape[0]
        rotated_transformed = _filter_edge_spots(rotated_transformed, margin_ratio=edge_margin_ratio)
        removed_n = rotated_before - rotated_transformed.shape[0]
        if removed_n > 0:
            print(f"Filtered {removed_n} edge spots (margin_ratio={edge_margin_ratio:.3f}) from rotated slice")

    # Step 3: Add metadata
    # Convert AlterationConfig to dict if present (needed for HDF5 serialization)
    simulation_params_serializable = _sanitize_params_for_hdf5(simulation_params)
    
    original_simulated.uns['alignment_simulation'] = {
        'type': 'original',
        'rotation_angle': 0,
        'data_type': data_type,
        'simulation_method': 'simulate_single_slice',
        'simulation_params': simulation_params_serializable
    }
    
    rotated_transformed.uns['alignment_simulation'] = {
        'type': 'rotated',
        'rotation_angle': rotation_angle,
        'data_type': data_type,
        'simulation_method': 'simulate_single_slice',
        'simulation_params': simulation_params_serializable,
        'edge_filter': {'enabled': bool(filter_edge_spots), 'margin_ratio': float(edge_margin_ratio)}
    }
    
    print(f"✓ Alignment dataset generation complete using simulate_single_slice:")
    print(f"  Original: {original_simulated.shape} spots")
    print(f"  Rotated: {rotated_transformed.shape} spots")
    
    return original_simulated, rotated_transformed


def simulate_alignment_warp(
    adata: ad.AnnData,
    distort_level: float = 100,
    fit_params: Optional[Dict[str, Any]] = None,
    expression_params: Optional[Dict[str, Any]] = None,
    filter_edge_spots: bool = True,
    edge_margin_ratio: float = 0.03,
    **kwargs
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Quick function to generate TPS-warped alignment dataset using simulate_single_slice.
    
    Args:
        adata: Input spatial transcriptomics data
        distort_level: Level of spatial distortion
        fit_params: Parameters for model fitting (passed to simulate_single_slice)
        expression_params: Parameters for expression simulation (passed to simulate_single_slice)
        **kwargs: Additional parameters for warping
    
    Returns:
        Tuple of (original_simulated, warped_simulated) datasets
    """
    print(f"Quick alignment simulation with TPS warping (distort_level={distort_level}) using simulate_single_slice")
    
    # Merge fit_params and expression_params for simulate_single_slice
    simulation_params = {}
    if fit_params:
        simulation_params.update(fit_params)
    if expression_params:
        simulation_params.update(expression_params)
    
    # Set defaults if not provided
    if 'verbose' not in simulation_params:
        simulation_params['verbose'] = True
    if 'sigma' not in simulation_params:
        simulation_params['sigma'] = 1.0
    
    # Step 1: Generate original simulated data using single slice simulator
    print("Generating original dataset using simulate_single_slice...")
    original_simulated = simulate_single_slice(adata, **simulation_params)
    
    # Step 2: Apply TPS warping transformation
    print("Applying TPS warping transformation...")
    warp_transformer = WarpTransformer(
        original_simulated,
        distort_level=distort_level,
        grid_size=kwargs.get('grid_size', 3),
        alpha=kwargs.get('alpha', 1.0)
    )
    warped_transformed = warp_transformer.transform(
        apply_rotation=kwargs.get('apply_rotation', True)
    )
    
    # Optional: filter edge spots from the transformed slice to remove cut edges
    if filter_edge_spots:
        warped_before = warped_transformed.shape[0]
        warped_transformed = _filter_edge_spots(warped_transformed, margin_ratio=edge_margin_ratio)
        removed_n = warped_before - warped_transformed.shape[0]
        if removed_n > 0:
            print(f"Filtered {removed_n} edge spots (margin_ratio={edge_margin_ratio:.3f}) from warped slice")

    # Step 3: Add metadata
    # Convert AlterationConfig to dict if present (needed for HDF5 serialization)
    simulation_params_serializable = _sanitize_params_for_hdf5(simulation_params)
    
    original_simulated.uns['alignment_simulation'] = {
        'type': 'original',
        'transformation': 'none',
        'simulation_method': 'simulate_single_slice',
        'simulation_params': simulation_params_serializable
    }
    
    warped_transformed.uns['alignment_simulation'] = {
        'type': 'warped',
        'transformation': 'TPS',
        'distort_level': distort_level,
        'grid_size': kwargs.get('grid_size', 3),
        'alpha': kwargs.get('alpha', 1.0),
        'apply_rotation': kwargs.get('apply_rotation', True),
        'simulation_method': 'simulate_single_slice',
        'simulation_params': simulation_params_serializable,
        'edge_filter': {'enabled': bool(filter_edge_spots), 'margin_ratio': float(edge_margin_ratio)}
    }
    
    print(f"✓ TPS warping simulation complete using simulate_single_slice:")
    print(f"  Original: {original_simulated.shape} spots")
    print(f"  Warped: {warped_transformed.shape} spots")
    
    return original_simulated, warped_transformed


def generate_alignment_benchmark_suite(
    adata: ad.AnnData,
    transformations: Optional[Dict[str, Dict[str, Any]]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    expression_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[ad.AnnData, ad.AnnData]]:
    """
    Generate a comprehensive alignment benchmark suite using simulate_single_slice directly.
    
    Args:
        adata: Input spatial transcriptomics data
        transformations: Custom transformation configurations (uses defaults if None)
        fit_params: Parameters for model fitting (passed to simulate_single_slice)
        expression_params: Parameters for expression simulation (passed to simulate_single_slice)
    
    Returns:
        Dictionary of transformation name -> (original, transformed) pairs
    """
    print("Generating comprehensive alignment benchmark suite using simulate_single_slice")
    
    # Set default transformations if none provided
    if transformations is None:
        transformations = {
            # Rotation transformations
            'rotation_15_imaging': {
                'type': 'rotation', 'angle': 15, 'data_type': 'imaging'
            },
            'rotation_30_imaging': {
                'type': 'rotation', 'angle': 30, 'data_type': 'imaging'
            },
            'rotation_45_imaging': {
                'type': 'rotation', 'angle': 45, 'data_type': 'imaging'
            },
            'rotation_30_sequencing': {
                'type': 'rotation', 'angle': 30, 'data_type': 'sequencing'
            },
            
            # Warping transformations
            'warp_light': {
                'type': 'warp', 'distort_level': 50, 'grid_size': 3
            },
            'warp_medium': {
                'type': 'warp', 'distort_level': 100, 'grid_size': 3
            },
            'warp_heavy': {
                'type': 'warp', 'distort_level': 200, 'grid_size': 4
            },
        }
    
    # Prepare simulation parameters by merging fit_params and expression_params
    simulation_params = {}
    if fit_params:
        simulation_params.update(fit_params)
    if expression_params:
        simulation_params.update(expression_params)
    
    # Set defaults
    if 'verbose' not in simulation_params:
        simulation_params['verbose'] = True
    if 'sigma' not in simulation_params:
        simulation_params['sigma'] = 1.0
    
    results = {}
    
    for name, config in transformations.items():
        print(f"\n--- Processing transformation: {name} ---")
        
        transform_type = config.get('type', '').lower()
        
        if transform_type == 'rotation':
            original, transformed = simulate_alignment_rotation(
                adata,
                rotation_angle=config.get('angle', 30),
                data_type=config.get('data_type', 'imaging'),
                fit_params=fit_params,
                expression_params=expression_params,
                center_correction=config.get('center_correction', 0),
                keep_bounds=config.get('keep_bounds', True),
                min_space=config.get('min_space'),
                max_grid_size=config.get('max_grid_size', 10000)
            )
        
        elif transform_type == 'warp':
            original, transformed = simulate_alignment_warp(
                adata,
                distort_level=config.get('distort_level', 100),
                fit_params=fit_params,
                expression_params=expression_params,
                grid_size=config.get('grid_size', 3),
                alpha=config.get('alpha', 1.0),
                apply_rotation=config.get('apply_rotation', True)
            )
        
        elif transform_type == 'cut_move':
            # For cut_move, we need to use simulate_single_slice directly and then transform
            print("Generating original dataset using simulate_single_slice...")
            original = simulate_single_slice(adata, **simulation_params)
            
            # Apply cut-and-move transformation
            spatial_transformer = SpatialTransformer(original)
            cut_bounds = config.get('cut_bounds', (0, 100, 0, 100))
            move_offset = config.get('move_offset', (20, 20))
            
            x_min, x_max, y_min, y_max = cut_bounds
            dx, dy = move_offset
            
            cut_region = spatial_transformer.cut(x_min, x_max, y_min, y_max)
            transformed = spatial_transformer.move(dx, dy)
            
            # Add metadata
            # Convert AlterationConfig to dict if present (needed for HDF5 serialization)
            simulation_params_serializable = _sanitize_params_for_hdf5(simulation_params)
            
            original.uns['alignment_simulation'] = {
                'type': 'original',
                'transformation': 'none',
                'simulation_method': 'simulate_single_slice',
                'simulation_params': simulation_params_serializable
            }
            
            transformed.uns['alignment_simulation'] = {
                'type': 'cut_moved',
                'transformation': 'cut_move',
                'cut_bounds': cut_bounds,
                'move_offset': move_offset,
                'simulation_method': 'simulate_single_slice',
                'simulation_params': simulation_params_serializable
            }
        
        else:
            print(f"Warning: Unknown transformation type '{transform_type}' for {name}")
            continue
        
        # Add benchmark metadata
        original.uns['benchmark_info'] = {
            'dataset_name': 'alignment_benchmark_suite',
            'transformation_name': name,
            'is_reference': True,
            'simulation_method': 'simulate_single_slice'
        }
        transformed.uns['benchmark_info'] = {
            'dataset_name': 'alignment_benchmark_suite',
            'transformation_name': name,
            'is_reference': False,
            'simulation_method': 'simulate_single_slice'
        }
        
        results[name] = (original, transformed)
        print(f"✓ Completed {name}: original={original.shape}, transformed={transformed.shape}")
    
    print(f"\n✓ Benchmark dataset generation complete using simulate_single_slice with {len(results)} transformation pairs")
    return results


# -------------------------------
# Helper: filter edge spots
# -------------------------------
def _filter_edge_spots(adata: ad.AnnData, margin_ratio: float = 0.03) -> ad.AnnData:
    """
    Remove spots that lie within a margin band along the bounding-box edges.

    This helps discard edge-cut artifacts introduced by transformations
    (rotation/warping) when the slice gets cropped by bounds.

    Parameters:
    - adata: AnnData with obsm['spatial'] coordinates
    - margin_ratio: Fraction of width/height to treat as an edge band to drop

    Returns:
    - Filtered AnnData copy with edge-band spots removed
    """
    if 'spatial' not in adata.obsm:
        return adata
    coords = adata.obsm['spatial']
    if coords is None or coords.shape[1] < 2:
        return adata
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    # Compute margins along each axis
    mx = (x_max - x_min) * float(margin_ratio)
    my = (y_max - y_min) * float(margin_ratio)
    # Keep only interior spots
    mask = (x > x_min + mx) & (x < x_max - mx) & (y > y_min + my) & (y < y_max - my)
    if np.all(mask):
        return adata
    filtered = adata[mask].copy()
    # Record simple, H5AD-friendly metadata
    filtered.uns.setdefault('post_processing', {})
    filtered.uns['post_processing']['edge_filter'] = {
        'applied': True,
        'margin_ratio': float(margin_ratio),
        'removed_n': int((~mask).sum())
    }
    return filtered
