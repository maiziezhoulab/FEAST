import inspect

import numpy as np
import pandas as pd
import anndata as ad
import pytest

import FEAST
from FEAST.FEAST_core.parameter_cloud import GeneParameterSimulator
from FEAST.FEAST_core.simulator import (
    SpatialSimulator,
    run_parameter_cloud_fitting,
    simulate_single_slice,
)
from FEAST.modeling.marginal_alteration import AlterationConfig
from FEAST.alignment import (
    AlignmentSimulator,
    RotationTransformer,
    simulate_alignment_rotation,
    simulate_alignment_warp,
)
from FEAST.deconvolution import (
    DeconvolutionSimulator,
    create_deconvolution_benchmark_suite,
    simulate_deconvolution_from_single_cells,
)


def _adata():
    X = np.array([[1, 0], [2, 1], [0, 3]], dtype=np.int32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["s1", "s2", "s3"]), var=pd.DataFrame(index=["g1", "g2"]))
    adata.obsm["spatial"] = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    return adata


def _model_params():
    return {
        "model_selected": ["Poisson", "NB"],
        "marginal_param1": [[0.0, 1.0, 2.0], [0.0, 2.0, 3.0]],
    }


def test_sigma_parameters_removed_from_public_core_api():
    public_callables = [
        FEAST.simulate,
        SpatialSimulator.simulate,
        simulate_single_slice,
        simulate_alignment_rotation,
        simulate_alignment_warp,
        DeconvolutionSimulator.simulate_deconvolution_data,
        DeconvolutionSimulator.create_deconvolution_benchmark_suite,
        simulate_deconvolution_from_single_cells,
        create_deconvolution_benchmark_suite,
    ]
    for fn in public_callables:
        assert "sigma" not in inspect.signature(fn).parameters
        assert "follower_sigma_factor" not in inspect.signature(fn).parameters

    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    with pytest.raises(TypeError):
        simulator.simulate(sigma=0.5, verbose=False)
    with pytest.raises(TypeError):
        simulate_single_slice(_adata(), sigma=1.0, verbose=False)


def test_deterministic_path_with_model_params():
    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    simulated = simulator.simulate(verbose=False)
    assert simulated.shape == (3, 2)
    assert "spatial" in simulated.obsm
    assert simulated.uns["simulation_method"] == "Quantile_Count_Decoding"
    assert simulated.uns["simulation_diagnostics"]["spatial_mode"] == "reference_rank"


def test_public_core_api_exposes_simulation_mode_and_random_seed():
    public_callables = [
        SpatialSimulator.fit_model,
        SpatialSimulator.simulate,
        simulate_single_slice,
    ]
    for fn in public_callables:
        params = inspect.signature(fn).parameters
        assert "random_seed" in params
    assert "seed" in inspect.signature(FEAST.simulate).parameters
    # SpatialSimulator.fit_model retains the internal "simulation_mode" parameter
    assert inspect.signature(SpatialSimulator.fit_model).parameters["simulation_mode"].default == "generative"
    # Public simulation functions use "parameter_mode" (translated internally to simulation_mode)
    for fn in [simulate_single_slice, FEAST.simulate]:
        assert inspect.signature(fn).parameters["parameter_mode"].default == "hungarian"


def test_obsolete_controls_are_removed_from_active_apis():
    removed = {
        "max_grid_size",
        "use_heuristic_search",
        "min_accepted_error",
        "screening_pool_size",
        "top_n_to_fully_evaluate",
        "n_jobs",
        "num_simulation_cores",
        "assignment_n_jobs",
    }
    callables = [
        FEAST.simulate,
        run_parameter_cloud_fitting,
        SpatialSimulator.fit_model,
        SpatialSimulator.simulate,
        simulate_single_slice,
        AlignmentSimulator.simulate_with_rotation,
        simulate_alignment_rotation,
        RotationTransformer.transform_sequencing,
        DeconvolutionSimulator.simulate_deconvolution_data,
        GeneParameterSimulator.assign_to_genes,
    ]
    for fn in callables:
        assert removed.isdisjoint(inspect.signature(fn).parameters)

    for fn in [run_parameter_cloud_fitting, SpatialSimulator.fit_model, simulate_single_slice]:
        parameters = inspect.signature(fn).parameters
        assert "beta_n_jobs" in parameters
        assert "convert_n_jobs" in parameters
    assert "spatial_mode" not in inspect.signature(SpatialSimulator.fit_model).parameters
    assert "spatial_mode" in inspect.signature(SpatialSimulator.simulate).parameters

    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    with pytest.raises(TypeError):
        simulator.simulate(num_simulation_cores=1, verbose=False)
    with pytest.raises(TypeError):
        FEAST.simulate(
            _adata(),
            parameter_mode="reference_stats",
            n_jobs=1,
            verbose=False,
        )
    with pytest.raises(TypeError):
        simulate_single_slice(_adata(), use_heuristic_search=False, verbose=False)
    with pytest.raises(TypeError):
        RotationTransformer(_adata()).transform_sequencing(max_grid_size=1)


def test_empirical_mode_with_model_params_uses_reference_rank_spatial_mode():
    model_params = _model_params()
    model_params["simulation_mode"] = "empirical"
    simulator = SpatialSimulator(_adata(), model_params=model_params)
    simulated = simulator.simulate(verbose=False, random_seed=11)
    diagnostics = simulated.uns["simulation_diagnostics"]
    assert diagnostics["simulation_mode"] == "empirical"
    assert diagnostics["assignment_method"] == "identity"
    assert diagnostics["spatial_mode"] == "reference_rank"


def test_public_empirical_single_slice_smoke():
    simulated = simulate_single_slice(
        _adata(),
        parameter_mode="reference_stats",
        alteration_config=AlterationConfig.mean_only(0.8),
        random_seed=3,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    diagnostics = simulated.uns["simulation_diagnostics"]
    assert simulated.shape == (3, 2)
    assert diagnostics["simulation_mode"] == "empirical"
    assert diagnostics["assignment_method"] == "identity"
    assert diagnostics["spatial_mode"] == "reference_rank"
    assert diagnostics["target_stage_achieved_change"]["mean"] == pytest.approx(0.8)
