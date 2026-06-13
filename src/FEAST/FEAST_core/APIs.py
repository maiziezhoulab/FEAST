from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import warnings

import anndata as ad

from .simulator import simulate_single_slice
from ..alignment.alignment_simulator import (
    generate_alignment_benchmark_suite,
    simulate_alignment_rotation,
    simulate_alignment_warp,
)
from ..deconvolution.deconvolution_simulator import (
    create_deconvolution_benchmark_suite,
    simulate_deconvolution_from_single_cells,
)
from ..deconvolution.generate_deconvolution import create_deconvolution_benchmark_data


class FEAST:
    """Unified FEAST API for spatial transcriptomics simulation tasks."""

    def __init__(self, adata: Union[ad.AnnData, List[ad.AnnData]], verbose: bool = True):
        self.verbose = verbose
        self.is_multi_slice = isinstance(adata, list)

        if self.is_multi_slice:
            self.adata_list = adata
            self.n_slices = len(adata)
            if self.verbose:
                print(f"FEAST initialized with {self.n_slices} slices")
                for i, slice_data in enumerate(adata):
                    print(f"  Slice {i}: {slice_data.shape}")
        else:
            self.adata = adata
            if self.verbose:
                print(f"FEAST initialized with single slice: {adata.shape}")

    def _single_adata(self, context: str) -> ad.AnnData:
        if self.is_multi_slice:
            warnings.warn(f"{context} with multi-slice input. Using first slice.")
            return self.adata_list[0]
        return self.adata

    def simulate_single_slice(
        self,
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
        simulation_mode: str = "generative",
        quantile_calibration: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> ad.AnnData:
        if verbose is None:
            verbose = self.verbose
        return simulate_single_slice(
            self._single_adata("Single slice simulation"),
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
            simulation_mode=simulation_mode,
            quantile_calibration=quantile_calibration,
            random_seed=random_seed,
        )

    def simulate_alignment(
        self,
        transformation_type: str = "rotation",
        rotation_angle: float = 0,
        warp_strength: float = 0,
        data_type: str = "imaging",
        filter_edge_spots: bool = True,
        edge_margin_ratio: float = 0.03,
        center_correction: Any = 0,
        keep_bounds: bool = True,
        min_space: Optional[float] = None,
        max_grid_size: int = 10000,
        grid_size: int = 3,
        alpha: float = 1.0,
        apply_rotation: bool = True,
        fit_params: Optional[Dict] = None,
        expression_params: Optional[Dict] = None,
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
        simulation_mode: str = "generative",
        quantile_calibration: Optional[str] = None,
        random_seed: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> tuple:
        if verbose is None:
            verbose = self.verbose
        adata_to_use = self._single_adata("Alignment simulation")
        simulation_params = {
            "visualize_fits": visualize_fits,
            "num_simulation_cores": num_simulation_cores,
            "clip_overshoot_factor": clip_overshoot_factor,
            "use_real_stats_directly": use_real_stats_directly,
            "annotation_key": annotation_key,
            "use_heuristic_search": use_heuristic_search,
            "min_accepted_error": min_accepted_error,
            "assignment_weights": assignment_weights,
            "screening_pool_size": screening_pool_size,
            "top_n_to_fully_evaluate": top_n_to_fully_evaluate,
            "n_jobs": n_jobs,
            "alteration_config": alteration_config,
            "boundary_multiplier": boundary_multiplier,
            "simulation_mode": simulation_mode,
            "quantile_calibration": quantile_calibration,
            "random_seed": random_seed,
            "verbose": verbose,
        }
        if fit_params:
            simulation_params.update(fit_params)
        if expression_params:
            simulation_params.update(expression_params)

        if transformation_type == "rotation":
            return simulate_alignment_rotation(
                adata_to_use,
                rotation_angle=rotation_angle,
                data_type=data_type,
                filter_edge_spots=filter_edge_spots,
                edge_margin_ratio=edge_margin_ratio,
                fit_params=simulation_params,
                expression_params={},
                center_correction=center_correction,
                keep_bounds=keep_bounds,
                min_space=min_space,
                max_grid_size=max_grid_size,
            )
        if transformation_type == "warp":
            return simulate_alignment_warp(
                adata_to_use,
                distort_level=warp_strength,
                filter_edge_spots=filter_edge_spots,
                edge_margin_ratio=edge_margin_ratio,
                fit_params=simulation_params,
                expression_params={},
                grid_size=grid_size,
                alpha=alpha,
                apply_rotation=apply_rotation,
            )
        raise ValueError(f"Unsupported transformation type: {transformation_type}")

    def simulate_alignment_benchmark(
        self,
        transformations: Optional[List[str]] = None,
        parameters: Optional[Dict] = None,
        data_types: Optional[List[str]] = None,
    ) -> Dict[str, tuple]:
        return generate_alignment_benchmark_suite(
            self._single_adata("Alignment benchmark"),
            transformations=transformations,
            parameters=parameters,
            data_types=data_types,
        )

    def simulate_deconvolution(
        self,
        cell_type_key: Optional[str] = None,
        downsampling_factor: float = 0.25,
        cells_per_spot: int = 50,
        aggregation_method: str = "sum",
        fractional_rounding: str = "probabilistic",
        grid_type: str = "hexagonal",
        alpha: float = 0.01,
        verbose: Optional[bool] = None,
        **kwargs,
    ) -> ad.AnnData:
        adata_to_use = self._single_adata("Deconvolution simulation")
        if verbose is None:
            verbose = self.verbose

        if "spatial" in adata_to_use.obsm:
            return create_deconvolution_benchmark_data(
                adata_to_use,
                downsampling_factor=downsampling_factor,
                grid_type=grid_type,
                cell_type_key=cell_type_key,
                alpha=alpha,
            )

        if cell_type_key is None:
            raise ValueError("cell_type_key is required for single-cell deconvolution simulation.")
        n_cells = adata_to_use.shape[0]
        n_spots = int(n_cells * downsampling_factor)
        if verbose:
            print(f"Deconvolution simulation: {n_cells} cells -> {n_spots} spots")
        return simulate_deconvolution_from_single_cells(
            adata_to_use,
            cell_type_key=cell_type_key,
            n_spots=n_spots,
            cells_per_spot=cells_per_spot,
            aggregation_method=aggregation_method,
            fractional_rounding=fractional_rounding,
            verbose=verbose,
        )

    def create_deconvolution_benchmark(
        self,
        cell_type_key: Optional[str] = None,
        downsampling_factor: float = 0.25,
        grid_type: str = "hexagonal",
        alpha: float = 0.01,
        verbose: Optional[bool] = None,
    ) -> ad.AnnData:
        return create_deconvolution_benchmark_data(
            self._single_adata("Deconvolution benchmark"),
            downsampling_factor=downsampling_factor,
            grid_type=grid_type,
            cell_type_key=cell_type_key,
            alpha=alpha,
        )

    def simulate_deconvolution_benchmark(
        self,
        cell_type_key: Optional[str] = None,
        downsampling_factors: List[float] = [0.1, 0.25, 0.5],
        grid_types: List[str] = ["hexagonal", "square", "kmeans"],
        alpha: float = 0.01,
        verbose: Optional[bool] = None,
    ) -> Dict[str, ad.AnnData]:
        if verbose is None:
            verbose = self.verbose
        return create_deconvolution_benchmark_suite(
            self._single_adata("Deconvolution benchmark"),
            cell_type_key=cell_type_key,
            downsampling_factors=downsampling_factors,
            grid_types=grid_types,
            alpha=alpha,
            verbose=verbose,
        )
