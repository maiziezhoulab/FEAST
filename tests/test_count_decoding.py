import numpy as np
import pytest

from FEAST.FEAST_core.count_decoding import (
    decode_counts_by_rank,
    decode_counts_by_spatial_intensity,
    generate_count_bag_from_model_params,
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


def test_decode_counts_by_rank_shape_and_dtype():
    q = np.arange(12, dtype=float).reshape(3, 4)
    decoded = decode_counts_by_rank(q, _model_params(), random_seed=1)
    assert decoded.shape == (3, 4)
    assert np.issubdtype(decoded.dtype, np.integer)
    assert np.all(decoded >= 0)


def test_decode_counts_by_rank_rank_ordering():
    """Higher quantile values should get higher (or equal) counts after sorting."""
    n_spots, n_genes = 20, 4
    q = np.random.default_rng(42).uniform(0, 1, (n_spots, n_genes))
    decoded = decode_counts_by_rank(q, _model_params(), random_seed=7)
    for g in range(n_genes):
        order = np.argsort(q[:, g])
        counts_ordered = decoded[order, g]
        assert np.all(np.diff(counts_ordered) >= 0), f"counts not monotonic for gene {g}"


def test_decode_counts_by_rank_with_weights():
    q = np.arange(12, dtype=float).reshape(3, 4)
    weights = np.array([1.0, 2.0, 3.0])
    decoded = decode_counts_by_rank(q, _model_params(), spot_weights=weights, random_seed=1)
    assert decoded.shape == (3, 4)
    assert np.all(decoded >= 0)


def test_decode_counts_by_rank_weight_validation():
    q = np.arange(12, dtype=float).reshape(3, 4)
    with pytest.raises(ValueError):
        decode_counts_by_rank(q, _model_params(), spot_weights=np.ones(2))


def test_decode_counts_by_rank_zero_spots():
    q = np.empty((0, 4))
    decoded = decode_counts_by_rank(q, _model_params())
    assert decoded.shape == (0, 4)


def test_decode_counts_by_rank_bad_quantiles():
    with pytest.raises(ValueError, match="2D array"):
        decode_counts_by_rank(np.array([1, 2, 3]), _model_params())


def test_spatial_intensity_decoder_preserves_high_contrast_range():
    """Spot-conditioned bags should not collapse a rare low-expression domain."""
    rng = np.random.default_rng(0)
    reference = np.r_[np.full(5, 2.0), rng.normal(130.0, 12.0, 95).clip(1.0)]
    mean = float(reference.mean())
    variance = float(reference.var())
    r = mean ** 2 / (variance - mean)
    params = {
        "model_selected": ["NB"],
        "marginal_param1": [[0.0, r, mean]],
        "target_stats": {"mean": [mean], "variance": [variance], "zero_prop": [0.0]},
    }

    iid = decode_counts_by_rank(reference[:, None], params, reference_X=reference[:, None], random_seed=7)[:, 0]
    spatial = decode_counts_by_spatial_intensity(
        reference[:, None], params, reference_X=reference[:, None], random_seed=7
    )[:, 0]

    def nonzero_range(values):
        positive = values[values > 0]
        return float(positive.max() / positive.min())

    assert nonzero_range(iid) < 10.0
    assert nonzero_range(spatial) > 15.0
    assert nonzero_range(spatial) > 3.0 * nonzero_range(iid)
