"""Core simulation functionality."""

from .count_decoding import (
    decode_counts_by_spatial_intensity,
    generate_count_bag_from_model_params,
)
from .parameter_cloud import apply_batch_deformation, BatchDeformation
from .simulator import (
    PARAMETER_MODES,
    SPATIAL_MODES,
    SpatialSimulator,
    run_direct_fitting_from_real_stats,
    run_parameter_cloud_fitting,
    simulate_batch_effect,
    simulate_single_slice,
)
from .theta_transform import stats_to_theta, theta_to_stats

__all__ = [
    "PARAMETER_MODES",
    "SPATIAL_MODES",
    "SpatialSimulator",
    "apply_batch_deformation",
    "BatchDeformation",
    "decode_counts_by_spatial_intensity",
    "generate_count_bag_from_model_params",
    "run_direct_fitting_from_real_stats",
    "run_parameter_cloud_fitting",
    "simulate_batch_effect",
    "simulate_single_slice",
    "stats_to_theta",
    "theta_to_stats",
]
