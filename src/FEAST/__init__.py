"""FEAST | Feature-space-based modeling of Spatial Transcriptomics.

Primary API
-----------
:func:`simulate`      — produce synthetic ST data from a reference slice.
:func:`generate`      — create a virtual ST slice from a blueprint + parameter cloud.
:func:`generate_from` — create a virtual ST slice conditioned on a real reference.
:func:`fit`           — learn a parameter-cloud model from real data.

Alteration
----------
:class:`Alteration`   — expression alteration configuration for :func:`simulate`.

"""

from __future__ import annotations

from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec

__version__ = "1.0.2"

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
from .de_novo.conditional import ReferenceFitConfig, SimulationConfig, estimate_assignment_randomness
from .de_novo.core import SliceBlueprint

# ---------------------------------------------------------------------------
# Alteration — public name for the former AlterationConfig class
# ---------------------------------------------------------------------------

Alteration = _AlterationConfig

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def simulate(
    adata,
    *,
    alteration: _AlterationConfig | None = None,
    seed: int | None = None,
    parameter_mode: str = "hungarian",
    spatial_mode: str = "reference_rank",
    n_jobs: int = -1,
    verbose: bool = True,
    **kwargs,
):
    """Simulate a synthetic spatial transcriptomics slice from a reference.

    This is the primary entry point for single-slice simulation.  FEAST fits
    a parameter-cloud model (mean, variance, zero-proportion) to the reference
    data, then samples and decodes a synthetic slice that preserves the same
    statistical properties.

    Parameters
    ----------
    adata:
        Reference :class:`~anndata.AnnData` with ``.obsm['spatial']``.
    alteration:
        Optional :class:`Alteration` config to systematically modify
        expression statistics (mean, variance, or sparsity).
    seed:
        Random seed for reproducible output.
    parameter_mode:
        ``"hungarian"`` (generative fitting) or ``"reference_stats"``
        (use reference stats directly).
    spatial_mode:
        ``"reference_rank"`` (rank-based spatial assignment) or
        ``"ot_spatial"`` (optimal-transport spatial assignment).
    n_jobs:
        Number of parallel jobs (``-1`` = all CPUs).
    verbose:
        Print progress messages.
    **kwargs:
        Advanced parameters passed through to the underlying simulator
        (e.g. ``use_heuristic_search``, ``assignment_solver``,
        ``ppf_method``, ``beta_n_jobs``).

    Returns
    -------
    :class:`~anndata.AnnData`
        Simulated slice with identical genes and spatial coordinates.
    """
    return _simulate_single_slice(
        adata,
        alteration_config=alteration,
        random_seed=seed,
        parameter_mode=parameter_mode,
        spatial_mode=spatial_mode,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )


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
    **kwargs:
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
    **kwargs:
        Passed through to :func:`~FEAST.de_novo.conditional.simulate_from_reference`.

    Returns
    -------
    :class:`~anndata.AnnData`
        Conditional virtual slice.
    """
    model = _fit_reference(reference, label_key=label_key, config=fit_config)
    return _simulate_from_reference(
        model, blueprint, config=sim_config, random_seed=seed, **kwargs
    )


def fit(
    adata,
    *,
    seed: int | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
    **kwargs,
):
    """Fit a parameter-cloud model to real ST data.

    The returned :class:`GeneParameterSimulator` can be serialised, inspected,
    altered, and used for repeated simulation without re-fitting.

    Parameters
    ----------
    adata:
        Reference :class:`~anndata.AnnData`.
    seed:
        Random seed for reproducible fitting.
    n_jobs:
        Number of parallel jobs.
    verbose:
        Print progress messages.
    **kwargs:
        Passed through to :meth:`GeneParameterSimulator.fit`.

    Returns
    -------
    :class:`GeneParameterSimulator`
        Fitted simulator ready for parameter inspection or count decoding.
    """
    sim = GeneParameterSimulator(random_seed=seed, verbose=verbose, n_jobs=n_jobs)
    sim.fit(adata, **kwargs)
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
    "simulate_batch_effect",
    "characterize_batch",
    "BatchDeformation",
    # Alteration
    "Alteration",
    # Classes
    "GeneParameterSimulator",
    "SliceBlueprint",
    "ReferenceFitConfig",
    "SimulationConfig",
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
