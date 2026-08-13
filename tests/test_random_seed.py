import anndata as ad
import numpy as np
import pandas as pd

import FEAST
from FEAST.FEAST_core.parameter_cloud import (
    GeneParameterSimulator,
    convert_params_for_new_simulator,
)
from FEAST.FEAST_core.simulator import run_parameter_cloud_fitting
from FEAST.modeling.Beta_mixture_model import BetaMixtureMarginalModeler


def _zip_zinb_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean": [0.5, 2.0],
            "variance": [0.6, 8.0],
            "zero_prop": [0.8, 0.65],
        },
        index=["zip_gene", "zinb_gene"],
    )


def _small_reference() -> ad.AnnData:
    reference = ad.AnnData(
        X=np.array([[1, 0], [2, 1], [0, 3]], dtype=np.int32),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    reference.obsm["spatial"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=float,
    )
    return reference


def _parameter_array(model_params: dict) -> np.ndarray:
    return np.asarray(model_params["marginal_param1"], dtype=np.float64)


def _beta_model() -> BetaMixtureMarginalModeler:
    model = BetaMixtureMarginalModeler(max_components=2)
    model.model_params = {
        "weights": np.array([0.35, 0.65]),
        "alphas": np.array([2.0, 7.0]),
        "betas": np.array([8.0, 3.0]),
        "n_params": 5,
    }
    model._is_fitted = True
    return model


def test_beta_ppf_cache_is_cleared_after_refit():
    model = BetaMixtureMarginalModeler(max_components=1)
    model.fit(np.linspace(0.04, 0.16, 100), visualize=False)
    low_median = model.ppf(0.5, n_samples=10_000, random_state=17)
    assert model._ppf_cache

    model.fit(np.linspace(0.84, 0.96, 100), visualize=False)
    assert not model._ppf_cache
    high_median = model.ppf(0.5, n_samples=10_000, random_state=17)

    assert high_median > low_median + 0.5


class _SeededCopula:
    def simulate(self, n, seeds):
        return np.random.default_rng(int(seeds[0])).random((n, 3))


class _LinearMarginal:
    def __init__(self, offset: float):
        self.offset = float(offset)

    def ppf(self, quantiles):
        return self.offset + np.asarray(quantiles, dtype=np.float64)


def _fitted_parameter_simulator() -> GeneParameterSimulator:
    simulator = GeneParameterSimulator()
    simulator.param_models = {
        "mean": _LinearMarginal(1.0),
        "variance": _LinearMarginal(2.0),
        "zero_prop": _beta_model(),
    }
    simulator.copula_model = _SeededCopula()
    simulator.original_stats = pd.DataFrame(
        {
            "mean": [0.5, 1.0, 1.5, 2.0],
            "variance": [1.0, 2.0, 3.0, 4.0],
            "zero_prop": [0.1, 0.2, 0.3, 0.4],
        },
        index=["g1", "g2", "g3", "g4"],
    )
    simulator.target_stats = simulator.original_stats.copy()
    simulator.fitted = True
    return simulator


def test_zip_zinb_conversion_seed_is_independent_of_global_numpy_state():
    np.random.seed(101)
    first = convert_params_for_new_simulator(
        _zip_zinb_stats(),
        n_spots=64,
        boundary_multiplier=np.inf,
        random_seed=7,
    )

    np.random.seed(909)
    second = convert_params_for_new_simulator(
        _zip_zinb_stats(),
        n_spots=64,
        boundary_multiplier=np.inf,
        random_seed=7,
    )

    assert first["model_selected"] == ["ZIP", "ZINB"]
    assert second["model_selected"] == first["model_selected"]
    np.testing.assert_array_equal(_parameter_array(first), _parameter_array(second))


def test_zip_zinb_conversion_seed_is_parallel_invariant():
    serial = convert_params_for_new_simulator(
        _zip_zinb_stats(),
        n_spots=64,
        boundary_multiplier=np.inf,
        n_jobs=1,
        random_seed=19,
    )
    parallel = convert_params_for_new_simulator(
        _zip_zinb_stats(),
        n_spots=64,
        boundary_multiplier=np.inf,
        n_jobs=2,
        random_seed=19,
    )

    assert parallel["model_selected"] == serial["model_selected"]
    np.testing.assert_array_equal(_parameter_array(parallel), _parameter_array(serial))


def test_global_numpy_seed_reproduces_unseeded_parallel_conversion():
    np.random.seed(41)
    first = convert_params_for_new_simulator(
        _zip_zinb_stats(),
        n_spots=64,
        boundary_multiplier=np.inf,
        n_jobs=2,
    )
    np.random.seed(41)
    second = convert_params_for_new_simulator(
        _zip_zinb_stats(),
        n_spots=64,
        boundary_multiplier=np.inf,
        n_jobs=2,
    )

    assert second["model_selected"] == first["model_selected"]
    np.testing.assert_array_equal(_parameter_array(second), _parameter_array(first))


def test_seeded_zip_zinb_conversion_does_not_advance_global_numpy_rng():
    np.random.seed(31415)
    expected = np.random.random(6)

    np.random.seed(31415)
    convert_params_for_new_simulator(
        _zip_zinb_stats(),
        n_spots=64,
        boundary_multiplier=np.inf,
        random_seed=23,
    )
    observed = np.random.random(6)

    np.testing.assert_array_equal(observed, expected)


def test_beta_ppf_seed_is_independent_of_global_numpy_state():
    quantiles = np.linspace(0.05, 0.95, 31)

    np.random.seed(12)
    first = _beta_model().ppf(quantiles, n_samples=10_000, random_state=5)

    np.random.seed(98)
    second = _beta_model().ppf(quantiles, n_samples=10_000, random_state=5)

    np.testing.assert_array_equal(first, second)


def test_parameter_cloud_seed_controls_beta_ppf_with_seed_zero():
    np.random.seed(12)
    first = _fitted_parameter_simulator().simulate(
        n_genes=4,
        verbose=False,
        random_seed=0,
    )
    np.random.seed(98)
    second = _fitted_parameter_simulator().simulate(
        n_genes=4,
        verbose=False,
        random_seed=0,
    )

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_parameter_fitting_propagates_run_seed_to_zip_correction():
    reference = _small_reference()

    np.random.seed(100)
    first = run_parameter_cloud_fitting(
        reference,
        simulation_mode="empirical",
        random_seed=7,
    )
    np.random.seed(200)
    second = run_parameter_cloud_fitting(
        reference,
        simulation_mode="empirical",
        random_seed=7,
    )

    assert first["model_selected"] == second["model_selected"]
    np.testing.assert_array_equal(_parameter_array(first), _parameter_array(second))


def test_public_simulate_seed_is_independent_of_global_numpy_state():
    reference = _small_reference()

    np.random.seed(123)
    first = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=0,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    np.random.seed(987)
    second = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=0,
        verbose=False,
        clip_overshoot_factor=0.0,
    )

    np.testing.assert_array_equal(first.X, second.X)


def test_public_simulate_respects_global_numpy_seed_when_seed_is_none():
    reference = _small_reference()

    np.random.seed(2468)
    first = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=None,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    np.random.seed(2468)
    second = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=None,
        verbose=False,
        clip_overshoot_factor=0.0,
    )

    np.testing.assert_array_equal(first.X, second.X)
