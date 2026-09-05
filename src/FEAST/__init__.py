"""FEAST | Feature-space-based modeling of Spatial Transcriptomics.

Primary API
-----------
:func:`simulate`      — produce synthetic ST data from a reference slice.
:func:`generate`      — create a virtual ST slice from a blueprint + parameter cloud.
:func:`generate_from` — deprecated compatibility alias for conditioned simulation.
:func:`fit`           — learn a parameter-cloud model from real data.

Alteration
----------
:class:`Alteration`   — expression alteration configuration for :func:`simulate`.

"""

from __future__ import annotations

from dataclasses import fields as _dataclass_fields
from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
import warnings as _warnings

__version__ = "1.0.5+local3d1"

# ---------------------------------------------------------------------------
# Primary verbs — thin wrappers that delegate to existing implementations
# ---------------------------------------------------------------------------

from .FEAST_core.simulator import simulate_single_slice as _simulate_single_slice
from .FEAST_core.simulator import simulate_batch_effect, characterize_batch
from .FEAST_core.parameter_cloud import GeneParameterSimulator, BatchDeformation
from .FEAST_core.theta_transform import stats_to_theta, theta_to_stats
from .modeling.marginal_alteration import AlterationConfig as _AlterationConfig
from .de_novo.builder import simulate_from_design as _simulate_from_design
from .de_novo.conditional import fit_reference as _fit_reference
from .de_novo.conditional import simulate_from_reference as _simulate_from_reference
from .de_novo.conditional import (
    ReferenceFitConfig,
    SimulationConfig,
    SimulationReference,
    estimate_assignment_randomness,
)
from .de_novo.local import simulate_local_references, calibrate_local_references
from .de_novo.stack import simulate_stack
from .de_novo.core import SliceBlueprint
from .de_novo.transport import TransportConfig

# ---------------------------------------------------------------------------
# Alteration — public name for the former AlterationConfig class
# ---------------------------------------------------------------------------

Alteration = _AlterationConfig

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def simulate(
    reference=None,
    target=None,
    *,
    condition_on: str | None = None,
    marginal_model: str = "parameter_cloud",
    parameter_cloud=None,
    transport: TransportConfig | None = None,
    fit_config: ReferenceFitConfig | None = None,
    alteration: _AlterationConfig | None = None,
    seed: int | None = None,
    parameter_mode: str = "hungarian",
    spatial_mode: str | None = None,
    verbose: bool = True,
    **kwargs,
):
    """Simulate an ST slice through one reference-to-target field backbone.

    Reference-based parameter-cloud simulation and label-conditioned transfer
    share the same public function and OT field implementation.  ``target=None``
    means the reference geometry is the target.  Set ``condition_on`` and
    ``marginal_model='empirical_reference'`` for conditional empirical-rank
    transfer.

    Parameters
    ----------
    reference:
        Reference AnnData, sequence of AnnData objects, or a fitted
        :class:`SimulationReference`.
    target:
        Optional target AnnData or :class:`SliceBlueprint`. Target expression
        is not read by the conditional path.
    condition_on:
        Reference and target ``.obs`` column used for label conditioning.
        Omit for global reference-based simulation.
    marginal_model:
        ``"parameter_cloud"`` or ``"empirical_reference"``.
    parameter_cloud:
        Optional global or label-specific parameter cloud for conditioned
        parameter-cloud decoding.
    transport:
        Shared :class:`TransportConfig`. A :class:`SimulationConfig` may be
        supplied when conditional quantile-field controls are also needed.
    fit_config:
        Reference fitting controls for conditioned generation.
    alteration:
        Optional :class:`Alteration` config to systematically modify
        expression statistics (mean, variance, or sparsity).
    seed:
        Run-wide seed for reproducible fitting, assignment, transport, and
        count generation. Seeded output is independent of ambient NumPy state.
    parameter_mode:
        ``"hungarian"`` (generative fitting) or ``"reference_stats"``
        (use reference stats directly).
    spatial_mode:
        ``"reference_rank"`` (rank-based spatial assignment) or
        ``"ot_spatial"`` (optimal-transport spatial assignment). When
        omitted, FEAST selects OT if ``target`` or ``transport`` is provided
        and reference-rank assignment otherwise.
    verbose:
        Print progress messages.
    kwargs:
        Advanced parameters passed through to the underlying simulator
        (e.g. ``assignment_solver``, ``ppf_method``, ``beta_n_jobs``).

    Returns
    -------
    :class:`~anndata.AnnData`
        Simulated slice with identical genes and spatial coordinates.
    """
    legacy_reference = kwargs.pop("adata", None)
    if reference is not None and legacy_reference is not None:
        raise ValueError("Provide reference or adata, not both.")
    if reference is None:
        reference = legacy_reference
        if reference is not None:
            _warnings.warn(
                "simulate(adata=...) is deprecated; use simulate(reference=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
    if reference is None:
        raise TypeError("simulate() is missing the required reference input.")

    if marginal_model not in {"parameter_cloud", "empirical_reference"}:
        raise ValueError(
            "marginal_model must be 'parameter_cloud' or 'empirical_reference'."
        )
    if transport is not None and not isinstance(transport, TransportConfig):
        raise TypeError("transport must be a TransportConfig or SimulationConfig.")

    conditional = (
        isinstance(reference, SimulationReference)
        or condition_on is not None
        or marginal_model == "empirical_reference"
    )
    if conditional:
        if condition_on is None and not isinstance(reference, SimulationReference):
            raise ValueError(
                "condition_on is required for empirical-reference conditional generation."
            )
        if alteration is not None:
            raise ValueError(
                "alteration is only supported with global parameter-cloud simulation."
            )
        model = (
            reference
            if isinstance(reference, SimulationReference)
            else _fit_reference(reference, label_key=condition_on, config=fit_config)
        )
        if condition_on is not None and condition_on != model.label_key:
            raise ValueError(
                "condition_on does not match the fitted SimulationReference label key "
                f"({condition_on!r} != {model.label_key!r})."
            )
        conditional_target = target
        if conditional_target is None:
            if isinstance(reference, SimulationReference):
                raise ValueError(
                    "target is required when reference is a fitted SimulationReference."
                )
            if not hasattr(reference, "obsm"):
                raise ValueError(
                    "target is required when conditional generation uses multiple references."
                )
            conditional_target = reference
        return _simulate_from_reference(
            model,
            conditional_target,
            parameter_cloud=parameter_cloud,
            config=_as_simulation_config(transport, verbose=verbose),
            random_seed=0 if seed is None else int(seed),
            marginal_model=marginal_model,
            **kwargs,
        )

    if parameter_cloud is not None:
        raise ValueError(
            "condition_on is required when supplying a parameter_cloud to simulate()."
        )
    target_adata = kwargs.pop("target_adata", None)
    if target is not None and target_adata is not None:
        raise ValueError("Provide target or target_adata, not both.")
    if target_adata is None:
        target_adata = target
    resolved_spatial_mode = (
        "ot_spatial"
        if target_adata is not None or transport is not None
        else "reference_rank"
    ) if spatial_mode is None else str(spatial_mode)
    if resolved_spatial_mode == "ot_spatial" and target_adata is None:
        target_adata = reference
    if resolved_spatial_mode == "reference_rank" and target_adata is not None:
        raise ValueError(
            "target is only used with spatial_mode='ot_spatial'; omit target or select OT."
        )
    return _simulate_single_slice(
        reference,
        alteration_config=alteration,
        random_seed=seed,
        parameter_mode=parameter_mode,
        spatial_mode=resolved_spatial_mode,
        verbose=verbose,
        target_adata=target_adata,
        transport_config=transport,
        **kwargs,
    )


def _as_simulation_config(
    transport: TransportConfig | None,
    *,
    verbose: bool,
) -> SimulationConfig:
    if transport is None:
        return SimulationConfig(verbose=bool(verbose))
    if isinstance(transport, SimulationConfig):
        return transport
    if not isinstance(transport, TransportConfig):
        raise TypeError("transport must be a TransportConfig or SimulationConfig.")
    values = {
        field.name: getattr(transport, field.name)
        for field in _dataclass_fields(TransportConfig)
    }
    return SimulationConfig(**values, verbose=bool(verbose))


def generate(
    blueprint,
    param_cloud,
    *,
    patterns=None,
    seed: int = 0,
    config: SimulationConfig | None = None,
    **kwargs,
):
    """Generate a virtual ST slice from a blueprint and parameter cloud.

    Build a synthetic slice from scratch — no reference data needed.
    Use :class:`SliceBlueprint` to define the spatial layout and
    ``param_cloud`` to specify per-gene expression distributions.

    Parameters
    ----------
    blueprint:
        :class:`SliceBlueprint` (or AnnData / dict / path to one) defining
        spot coordinates and domain labels.
    param_cloud:
        DataFrame or dict with columns ``mean``, ``variance``, ``zero_prop``
        for each gene.
    patterns:
        Optional spatial pattern specification (e.g. gradients, hotspots).
    seed:
        Random seed for reproducible output.
    config:
        :class:`SimulationConfig` for transport and quantile-field settings.
    kwargs:
        Passed through to :func:`~FEAST.de_novo.builder.simulate_from_design`.

    Returns
    -------
    :class:`~anndata.AnnData`
        Virtual slice.
    """
    return _simulate_from_design(
        blueprint,
        param_cloud,
        pattern_spec=patterns,
        random_seed=seed,
        config=config,
        **kwargs,
    )


def generate_from(
    reference,
    blueprint,
    *,
    seed: int | None = None,
    fit_config: ReferenceFitConfig | None = None,
    sim_config: SimulationConfig | None = None,
    label_key: str = "domain",
    **kwargs,
):
    """Generate a virtual ST slice conditioned on a real reference.

    Fits a reference model to the reference slice(s), then transports rank
    scores to the blueprint layout to produce a conditional virtual slice.

    Parameters
    ----------
    reference:
        Reference :class:`~anndata.AnnData` (or list of them).
    blueprint:
        :class:`SliceBlueprint` defining target coordinates and domains.
    seed:
        Random seed for reproducible output.
    fit_config:
        :class:`ReferenceFitConfig` controlling gene filtering and scaling.
    sim_config:
        :class:`SimulationConfig` controlling transport and rank-field settings.
    label_key:
        Column in ``reference.obs`` with domain labels.
    kwargs:
        Passed through to :func:`~FEAST.de_novo.conditional.simulate_from_reference`.

    Returns
    -------
    :class:`~anndata.AnnData`
        Conditional virtual slice.
    """
    _warnings.warn(
        "generate_from() is deprecated; use simulate(..., target=..., "
        "condition_on=..., marginal_model='empirical_reference') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return simulate(
        reference,
        target=blueprint,
        condition_on=label_key,
        marginal_model="empirical_reference",
        transport=sim_config,
        fit_config=fit_config,
        seed=seed,
        **kwargs,
    )


def fit(
    adata,
    *,
    max_zero_prop_components: int = 8,
    beta_early_stopping_patience: int = 2,
    beta_n_jobs: int = 1,
    ppf_method: str = "interp",
    visualize_fits: bool = False,
):
    """Fit a parameter-cloud model to real ST data.

    The returned :class:`GeneParameterSimulator` can be serialised, inspected,
    altered, and used for repeated simulation without re-fitting.

    Parameters
    ----------
    adata:
        Reference :class:`~anndata.AnnData`.
    max_zero_prop_components:
        Maximum number of components considered for the zero-proportion
        marginal model.
    beta_early_stopping_patience:
        Component-search patience for the beta mixture.
    beta_n_jobs:
        Parallel workers used by the beta-mixture component search.
    ppf_method:
        Student-t mixture inverse-CDF implementation, ``"interp"`` or
        ``"exact"``.
    visualize_fits:
        Show diagnostic marginal-fit plots.

    Returns
    -------
    :class:`GeneParameterSimulator`
        Fitted simulator ready for parameter inspection or count decoding.
    """
    sim = GeneParameterSimulator(
        max_zero_prop_components=max_zero_prop_components,
        beta_early_stopping_patience=beta_early_stopping_patience,
        beta_n_jobs=beta_n_jobs,
        ppf_method=ppf_method,
    )
    sim.fit(adata, visualize_fits=visualize_fits)
    return sim


# ---------------------------------------------------------------------------
# Lazy subpackage loading
# ---------------------------------------------------------------------------

def _module_exists(absolute_module_name: str) -> bool:
    return _find_spec(absolute_module_name) is not None


ALIGNMENT_AVAILABLE = _module_exists(__name__ + ".alignment")
DECONVOLUTION_AVAILABLE = _module_exists(__name__ + ".deconvolution")
DE_NOVO_AVAILABLE = _module_exists(__name__ + ".de_novo")


def __getattr__(name: str):
    # Lazy-load subpackages
    if name in ("alignment", "deconvolution", "de_novo"):
        return _import_module(__name__ + "." + name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)


__all__ = [
    # Primary verbs
    "simulate",
    "generate",
    "generate_from",
    "fit",
    # Batch effect
    "simulate_local_references",
    "simulate_stack",
    "calibrate_local_references",
    "simulate_batch_effect",
    "characterize_batch",
    "BatchDeformation",
    # Alteration
    "Alteration",
    # Classes
    "GeneParameterSimulator",
    "SliceBlueprint",
    "ReferenceFitConfig",
    "SimulationReference",
    "SimulationConfig",
    "TransportConfig",
    "estimate_assignment_randomness",
    # Functions
    "stats_to_theta",
    "theta_to_stats",
    # Subpackages
    "alignment",
    "deconvolution",
    "de_novo",
    # Availability flags
    "ALIGNMENT_AVAILABLE",
    "DECONVOLUTION_AVAILABLE",
    "DE_NOVO_AVAILABLE",
]
