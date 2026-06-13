import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from FEAST.FEAST_core import count_decoding
from FEAST.FEAST_core.count_decoding import (
    decode_counts_from_quantiles,
    generate_count_bag_from_model_params,
    resolve_decode_method,
    resolve_quantile_calibration,
)


def _model_params():
    return {
        "model_selected": ["Poisson", "NB", "ZIP", "ZINB"],
        "marginal_param1": [
            [0.0, 1.0, 3.0],
            [0.0, 2.0, 5.0],
            [0.25, 1.0, 4.0],
            [0.2, 3.0, 6.0],
        ],
    }


def test_resolve_helpers():
    assert resolve_decode_method("AUTO", allow_auto=True) == "quantile"
    assert resolve_quantile_calibration("Raw") == "raw"
    with pytest.raises(ValueError):
        resolve_decode_method("auto")
    with pytest.raises(ValueError):
        resolve_quantile_calibration("bad")


def test_generate_count_bag_shape_and_boundary():
    reference = np.full((5, 4), 2, dtype=np.int32)
    counts = generate_count_bag_from_model_params(
        _model_params(),
        5,
        reference_X=reference,
        boundary_multiplier=1.0,
        random_seed=7,
    )
    assert counts.shape == (5, 4)
    assert np.issubdtype(counts.dtype, np.floating)
    assert np.all(counts >= 0)
    assert np.all(counts <= 2)


def test_quantile_decode_matches_scipy_ppf_raw():
    q = np.array([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.9, 0.9, 0.9, 0.9]])
    decoded = decode_counts_from_quantiles(
        q,
        _model_params(),
        method="quantile",
        quantile_calibration="raw",
    )
    expected_poisson = poisson.ppf(np.clip(q[:, 0], 1e-6, 1 - 1e-6), 3.0)
    p_nb = 2.0 / (2.0 + 5.0)
    expected_nb = nbinom.ppf(np.clip(q[:, 1], 1e-6, 1 - 1e-6), 2.0, p_nb)
    q_zip = np.clip((q[:, 2] - 0.25) / 0.75, 0.0, 1.0)
    expected_zip = poisson.ppf(q_zip, 4.0)
    expected_zip[q[:, 2] <= 0.25] = 0
    p_zinb = 3.0 / (3.0 + 6.0)
    q_zinb = np.clip((q[:, 3] - 0.2) / 0.8, 0.0, 1.0)
    expected_zinb = nbinom.ppf(q_zinb, 3.0, p_zinb)
    expected_zinb[q[:, 3] <= 0.2] = 0

    expected = np.rint(np.column_stack([expected_poisson, expected_nb, expected_zip, expected_zinb])).astype(np.int32)
    np.testing.assert_array_equal(decoded, expected)


def test_quantile_decode_does_not_sample(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("quantile mode should not sample")

    monkeypatch.setattr(count_decoding, "generate_count_bag_from_model_params", fail)
    decoded = decode_counts_from_quantiles(
        np.full((3, 4), 0.5),
        _model_params(),
        method="quantile",
        quantile_calibration="raw",
    )
    assert decoded.shape == (3, 4)


def test_rank_decode_and_spot_weight_validation():
    q = np.arange(12, dtype=float).reshape(3, 4)
    decoded = decode_counts_from_quantiles(q, _model_params(), method="rank", random_seed=1)
    assert decoded.shape == (3, 4)
    assert np.issubdtype(decoded.dtype, np.integer)
    assert np.all(decoded >= 0)
    with pytest.raises(ValueError):
        decode_counts_from_quantiles(q, _model_params(), spot_weights=np.ones(2))
