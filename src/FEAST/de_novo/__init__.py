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
from .stack import simulate_stack

__all__ = [
    "SimulationBlueprint",
    "SimulationBlueprintBuilder",
    "SimulationParameterBuilder",
    "SimulationPatternBuilder",
    "ReferenceFitConfig",
    "SimulationReference",
    "SimulationConfig",
    "QuantileFieldConfig",
    "compose_pattern",
    "evaluate_motif",
    "fit_reference",
    "simulate_from_reference",
    "simulate_stack",
    "simulate_from_design",
    "load_blueprint",
    "plot_pattern",
    "plot_pattern_panel",
    "plot_blueprint",
]


def __dir__():
    return sorted(__all__)
