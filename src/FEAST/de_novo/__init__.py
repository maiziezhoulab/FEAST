"""Public de novo simulation API."""

from __future__ import annotations

from .builder import (
    SimulationBlueprintBuilder,
    SimulationParameterBuilder,
    plot_blueprint,
    simulate_from_design,
)
from .conditional import (
    ReferenceFitConfig,
    SimulationConfig,
    SimulationReference,
    fit_reference,
    simulate_from_reference,
)
from .core import SliceBlueprint as SimulationBlueprint, load_blueprint
from .pattern import (
    SimulationPatternBuilder,
    compose_pattern,
    evaluate_motif,
    plot_pattern,
    plot_pattern_panel,
)
from .quantile_field import QuantileFieldConfig
from .local import simulate_local_references, select_z_references, calibrate_local_references
from .stack import simulate_stack
from .transport import TransportConfig
from .z_regularize import (
    calibrate_counts_to_regularized_means,
    class_anchor_weight,
    compute_z_coherence,
    regularize_mean_profiles,
    summarize_z_coherence_frame,
    z_penalty_matrix,
)
from .z_spot_smooth import (
    compute_spot_z_autocorrelation,
    smooth_cross_z_spots,
)

__all__ = [
    "SimulationBlueprint",
    "SimulationBlueprintBuilder",
    "SimulationParameterBuilder",
    "SimulationPatternBuilder",
    "ReferenceFitConfig",
    "SimulationReference",
    "SimulationConfig",
    "TransportConfig",
    "QuantileFieldConfig",
    "calibrate_counts_to_regularized_means",
    "class_anchor_weight",
    "compute_spot_z_autocorrelation",
    "compose_pattern",
    "compute_z_coherence",
    "evaluate_motif",
    "fit_reference",
    "load_blueprint",
    "plot_blueprint",
    "plot_pattern",
    "plot_pattern_panel",
    "regularize_mean_profiles",
    "simulate_from_design",
    "simulate_from_reference",
    "simulate_stack",
    "simulate_local_references",
    "select_z_references",
    "calibrate_local_references",
    "smooth_cross_z_spots",
    "summarize_z_coherence_frame",
    "z_penalty_matrix",
]


def __dir__():
    return sorted(__all__)
