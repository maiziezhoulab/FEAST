import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from FEAST.FEAST_core.simulator import SpatialSimulator, simulate_single_slice
from FEAST.modeling.marginal_alteration import AlterationConfig


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


def test_public_subsystem_imports():
    modules = [
        "FEAST.FEAST_core",
        "FEAST.FEAST_core.count_decoding",
        "FEAST.alignment",
        "FEAST.deconvolution",
        "FEAST.modeling",
        "FEAST.de_novo",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_deterministic_path_with_model_params():
    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    simulated = simulator.simulate(verbose=False)
    assert simulated.shape == (3, 2)
    assert "spatial" in simulated.obsm
    assert simulated.uns["simulation_method"] == "Quantile_Count_Decoding"
    assert simulated.uns["simulation_diagnostics"]["spatial_mode"] == "reference_rank"


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
