import numpy as np
import pandas as pd
import pytest

from FEAST.FEAST_core.count_decoding import decode_counts_by_spatial_intensity
from FEAST.modeling.StudentT_mixture_model import StudentTMixtureMarginalModeler
from FEAST.modeling.marginal_alteration import AlterationConfig, alter_marginal_model


def test_interpolated_studentt_ppf_refreshes_after_mean_alteration():
    data = np.geomspace(0.1, 100.0, 120)
    model = StudentTMixtureMarginalModeler(max_components=1, ppf_method="interp")
    model.fit(data, log_transform=True, visualize=False)

    q = np.array([0.25, 0.50, 0.75, 0.99])
    before = model.ppf(q)
    assert hasattr(model, "_ppf_interp")

    alter_marginal_model(
        model,
        mean_fold_change=2.0,
        variance_fold_change=1.0,
        dispersion_strength=0.0,
        preserve_original=False,
        verbose=False,
    )

    assert not hasattr(model, "_ppf_interp")
    after = model.ppf(q)

    np.testing.assert_allclose(after[1:3] / before[1:3], 2.0, rtol=0.15)
    assert after[-1] > model.data_range[1]


def _two_component_log_space_studentt(ppf_method):
    model = StudentTMixtureMarginalModeler(max_components=2, ppf_method=ppf_method)
    model._is_fitted = True
    model.log_transform = True
    model.data_range = (1e-3, 1e3)
    model.model_params = {
        "n_components": 2,
        "weights": np.array([0.5, 0.5]),
        "means": np.array([1.5, -1.5]),
        "scales": np.array([0.2, 0.2]),
        "dfs": np.array([5.0, 5.0]),
    }
    return model


@pytest.mark.parametrize("ppf_method", ["exact", "interp"])
def test_studentt_log_transform_ppf_is_monotone_in_original_space(ppf_method):
    model = _two_component_log_space_studentt(ppf_method)

    quantiles = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    values = model.ppf(quantiles)

    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) > 0)
    np.testing.assert_allclose(model.cdf(values), quantiles, atol=5e-3)


def test_spatial_intensity_decoder_preserves_mean_only_alteration_contrast():
    rng = np.random.default_rng(1)
    reference = np.r_[np.full(5, 2.0), rng.normal(130.0, 12.0, 95).clip(1.0)]
    mean = float(reference.mean())
    params = {
        "model_selected": ["NB"],
        "marginal_param1": [[0.0, 50.0, mean * 1.25]],
        "target_stats": pd.DataFrame(
            {"mean": [mean * 1.25], "variance": [mean * 1.25], "zero_prop": [0.25]}
        ),
        "parameter_diagnostics": {
            "requested_config": AlterationConfig.mean_only(1.25).to_dict(),
        },
    }

    spatial = decode_counts_by_spatial_intensity(
        reference[:, None],
        params,
        reference_X=reference[:, None],
        random_seed=7,
    )[:, 0]

    positive = spatial[spatial > 0]
    assert float(positive.max() / positive.min()) > 15.0
    assert np.mean(spatial == 0) < 0.05


def test_log_unbalanced_matches_pot_kl_objective_and_stopping_error():
    import ot
    from FEAST.de_novo._ot_transport import _sinkhorn_log_unbalanced

    cost = np.array([[0.01, 0.08, 0.2], [0.15, 0.04, 0.09]])
    a, b = np.array([0.3, 0.7]), np.array([0.2, 0.3, 0.5])
    expected, pot_log = ot.sinkhorn_unbalanced(
        a, b, cost, 0.05, 5.0, reg_type="kl", c=a[:, None] * b[None, :],
        numItermax=1000, stopThr=1e-5, log=True,
    )
    actual, solver_log = _sinkhorn_log_unbalanced(
        cost, a, b, reg=0.05, reg_m=5.0, numItermax=1000, stopThr=1e-5,
    )
    assert np.abs(actual - expected).sum() / np.abs(expected).sum() <= 1e-10
    assert np.max(np.abs(actual - expected)) <= 1e-12
    assert actual.sum() == pytest.approx(expected.sum(), rel=1e-10, abs=0.0)
    assert solver_log["niter"] == len(pot_log["err"])
    np.testing.assert_allclose(solver_log["err"], pot_log["err"], rtol=1e-9, atol=1e-14)
    assert solver_log["err"][-1] < 1e-5


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_log_scaling_change_is_stable_for_large_potentials(offset):
    from FEAST.de_novo._ot_transport import _stable_relative_change

    current, previous = np.array([0.2, 0.5]), np.array([0.1, 0.4])
    expected = np.max(np.abs(np.exp(current) - np.exp(previous))) / np.exp(0.5)
    assert _stable_relative_change(current + offset, previous + offset) == pytest.approx(expected)


def test_log_unbalanced_nonconvergence_raises():
    from FEAST.de_novo._ot_transport import OptimalTransportError, sinkhorn_transport

    with pytest.raises(OptimalTransportError, match="Unbalanced OT did not converge"):
        sinkhorn_transport(
            np.array([[0.0, 1.0], [2.0, 0.0]]), np.array([3.0, 7.0]), np.ones(2),
            unbalanced=True, method="sinkhorn_log", numItermax=1,
        )
