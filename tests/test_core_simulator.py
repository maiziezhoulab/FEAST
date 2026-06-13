import inspect

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from FEAST.FEAST_core.simulator import SpatialSimulator, simulate_single_slice
from FEAST.FEAST_core.APIs import FEAST
from FEAST.modeling.marginal_alteration import AlterationConfig
from FEAST.alignment import simulate_alignment_rotation, simulate_alignment_warp
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
        SpatialSimulator.simulate,
        simulate_single_slice,
        FEAST.simulate_single_slice,
        FEAST.simulate_alignment,
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
    assert simulated.uns["simulation_diagnostics"]["count_decode_method"] == "quantile"


def test_public_core_api_exposes_simulation_mode_and_random_seed():
    public_callables = [
        SpatialSimulator.fit_model,
        SpatialSimulator.simulate,
        simulate_single_slice,
        FEAST.simulate_single_slice,
        FEAST.simulate_alignment,
    ]
    for fn in public_callables:
        params = inspect.signature(fn).parameters
        assert "random_seed" in params
    for fn in [SpatialSimulator.fit_model, simulate_single_slice, FEAST.simulate_single_slice, FEAST.simulate_alignment]:
        assert inspect.signature(fn).parameters["simulation_mode"].default == "generative"


def test_empirical_mode_with_model_params_uses_reference_rank_quantiles():
    model_params = _model_params()
    model_params["simulation_mode"] = "empirical"
    model_params["quantile_calibration"] = "reference_rank"
    simulator = SpatialSimulator(_adata(), model_params=model_params)
    simulated = simulator.simulate(verbose=False, random_seed=11)
    diagnostics = simulated.uns["simulation_diagnostics"]
    assert diagnostics["simulation_mode"] == "empirical"
    assert diagnostics["assignment_method"] == "identity"
    assert diagnostics["count_decode_method"] == "quantile"
    assert diagnostics["quantile_calibration"] == "reference_rank"


def test_public_empirical_single_slice_smoke():
    simulated = simulate_single_slice(
        _adata(),
        simulation_mode="empirical",
        alteration_config=AlterationConfig.mean_only(0.8),
        random_seed=3,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    diagnostics = simulated.uns["simulation_diagnostics"]
    assert simulated.shape == (3, 2)
    assert diagnostics["simulation_mode"] == "empirical"
    assert diagnostics["assignment_method"] == "identity"
    assert diagnostics["count_decode_method"] == "quantile"
    assert diagnostics["target_stage_achieved_change"]["mean"] == pytest.approx(0.8)
