import numpy as np
import pandas as pd
import pytest

from FEAST.FEAST_core.count_decoding import decode_counts_by_rank, decode_counts_by_spatial_intensity
from FEAST.FEAST_core.parameter_cloud import GeneParameterSimulator
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


@pytest.mark.parametrize("assignment_blocks", [False, True])
def test_hybrid_assignment_handles_extreme_candidate_costs(assignment_blocks):
    simulator = GeneParameterSimulator()
    simulator.original_stats = pd.DataFrame(
        {
            "mean": [1.0, 2.0, 3.0, 4.0],
            "variance": [2.0, 3.0, 5.0, 9.0],
            "zero_prop": [0.1, 0.2, 0.3, 0.4],
        },
        index=["g1", "g2", "g3", "g4"],
    )
    synthetic = pd.DataFrame(
        {
            "mean": [1.0, 2.0, 3.0, 4.0, 1e39, 2e39, 3e39, 4e39],
            "variance": [2.0, 3.0, 5.0, 9.0, 1e39, 2e39, 3e39, 4e39],
            "zero_prop": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )

    assigned = simulator.assign_to_genes(
        synthetic,
        verbose=False,
        assignment_blocks=assignment_blocks,
        assignment_block_size=2,
        assignment_block_multiplier=2,
    )

    assert assigned["gene_id"].tolist() == ["g1", "g2", "g3", "g4"]
    assert np.all(np.isfinite(assigned[["mean", "variance", "zero_prop"]].to_numpy(dtype=float)))


def test_deprecated_sparsity_fold_change_maps_to_logit_shift():
    decreased = AlterationConfig.sparsity_only(fold_change=0.5)
    increased = AlterationConfig.sparsity_only(fold_change=2.0)

    assert decreased.apply_to_zero_prop
    assert increased.apply_to_zero_prop
    assert decreased.sparsity_logit_shift < 0
    assert increased.sparsity_logit_shift > 0
    np.testing.assert_allclose(decreased.sparsity_logit_shift, np.log(0.5))
    np.testing.assert_allclose(increased.sparsity_logit_shift, np.log(2.0))


def test_spatial_intensity_decoder_preserves_high_contrast_range():
    rng = np.random.default_rng(0)
    reference = np.r_[np.full(5, 2.0), rng.normal(130.0, 12.0, 95).clip(1.0)]
    mean = float(reference.mean())
    variance = float(reference.var())
    r = mean**2 / (variance - mean)
    params = {
        "model_selected": ["NB"],
        "marginal_param1": [[0.0, r, mean]],
    }

    iid = decode_counts_by_rank(reference[:, None], params, reference_X=reference[:, None], random_seed=7)[:, 0]
    spatial = decode_counts_by_spatial_intensity(
        reference[:, None],
        params,
        reference_X=reference[:, None],
        random_seed=7,
    )[:, 0]

    def nonzero_range(values):
        positive = values[values > 0]
        return float(positive.max() / positive.min())

    assert nonzero_range(iid) < 10.0
    assert nonzero_range(spatial) > 15.0
    assert nonzero_range(spatial) > 3.0 * nonzero_range(iid)


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
