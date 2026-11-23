import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import expit, logit

import FEAST
from FEAST.FEAST_core import count_decoding
from FEAST.FEAST_core.parameter_cloud import project_stats_to_feasible_domain, project_to_count_feasible
from FEAST.FEAST_core.theta_transform import affine_zero_probability
from FEAST.de_novo import builder, conditional, local
from FEAST.de_novo.quantile_field import weighted_stats_log_space


@pytest.mark.parametrize("conversion,normalize", [
    (builder._stats_frame_to_model_params, builder._coerce_stats_frame),
    (conditional._stats_frame_to_model_params, conditional._normalize_stats_frame),
])
def test_full_range_and_infeasible_totals(conversion, normalize):
    zeros = [0, 0.999, 1, 0.99, 0]
    stats = pd.DataFrame({"mean": [1, 0.001, 0.002, 0.001, 0], "variance": 0.0, "zero_prop": zeros},
                         index=list("abcde"))
    for project in (project_stats_to_feasible_domain, project_to_count_feasible):
        projected, _ = project(stats)
        np.testing.assert_array_equal(projected.zero_prop, zeros)
    params = conversion(normalize(stats, stats.index))
    params["target_stats"]["variance"] = 0.0
    diagnostics = {}
    counts = count_decoding.decode_counts_by_spatial_intensity(
        np.ones((1000, 5)), params, random_seed=7, diagnostics=diagnostics)
    np.testing.assert_array_equal(np.mean(counts == 0, axis=0), zeros)
    np.testing.assert_array_equal(counts.sum(axis=0), [1000, 1, 0, 10, 1000])
    np.testing.assert_array_equal(diagnostics["requested_total"], [1000, 1, 2, 1, 0])
    np.testing.assert_array_equal(diagnostics["total_adjustment"], [0, 0, -2, 9, 1000])
    assert np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0)
    np.testing.assert_array_equal(count_decoding._match_zero_count_and_mean(np.array([7, 8]), 0, 0), [1, 1])


def test_observed_statistics_and_logit_boundaries():
    counts = sparse.csr_matrix(([1, 0], ([0, 1], [0, 0])), shape=(1000, 1))
    for matrix in (counts, counts.toarray()):
        assert conditional._sparse_gene_stats(matrix, ["g"]).loc["g", "zero_prop"] == 0.999
    zeros = [0, 0.999, 1]
    stats = pd.DataFrame({"mean": 1, "variance": 1, "zero_prop": zeros}, index=list("abc"))
    np.testing.assert_array_equal(weighted_stats_log_space([stats], [1], stats.index).zero_prop, zeros)
    shifted = affine_zero_probability(zeros, 1, 1)
    np.testing.assert_allclose(shifted, [0, expit(logit(0.999) + 1), 1], rtol=1e-15)
    np.testing.assert_array_equal(affine_zero_probability(zeros, 0, 0), [0.5] * 3)


def test_local_observed_uses_eligible_merged_reference_groups(monkeypatch):
    genes = ["always", "never", "variable"]
    xy = np.column_stack((np.arange(5), [0, 1, 0, 1, 0]))
    refs = []
    for name, variable in [("left", [0, 0, 0, 2, 0]), ("right", [0, 2, 2, 2, 2]),
                           ("excluded", [0, 0, 0, 0, 0])]:
        counts = np.column_stack((np.ones(5), np.zeros(5), variable))
        ref = ad.AnnData(np.ones_like(counts), var=pd.DataFrame(index=genes),
                         layers={"counts": sparse.csr_matrix(counts)}, obsm={"spatial": xy})
        ref.obs["label"] = ["A", "A", "A", "A", "B"]
        ref.uns["reference_name"] = name
        refs.append(ref)
    target = refs[0].copy()

    def fitted_table(ref, members, label_key, genes, seed, cache_path):
        return pd.DataFrame({"mean": 1., "variance": 2., "zero_prop": 0.35}, index=genes), {}, False

    monkeypatch.setattr(local, "_generated_table", fitted_table)
    options = dict(label_key="label", n_references=2, reference_z={"left": 0, "right": 3, "excluded": 10},
                   target_z=1, min_positions=2, batch_deformation=([1, 1, 1], [0, 0, 1]), random_seed=3)
    modeled = FEAST.simulate_local_references(refs, target, **options)
    target.X[:] = 17
    observed = FEAST.simulate_local_references(refs, target, zero_target_source="observed", **options)
    record = json.loads(observed.uns["local_generation_json"])
    assert record["zero_target_source"] == "observed"
    assert set(record["merging"]["members"]["A"]) == {"A", "B"}
    weights = record["group_weights"]["A"]
    assert set(weights) == {"left", "right"}
    expected = expit(weights["left"] * logit(0.8) + weights["right"] * logit(0.2))
    np.testing.assert_allclose(observed.var["target_zero_prop"], [0, 1, expected])
    np.testing.assert_allclose(modeled.var["target_zero_prop"], expit(logit(0.35) + 1))
    np.testing.assert_array_equal(np.sum(observed.X == 0, axis=0), [0, 5, round(5 * expected)])
    np.testing.assert_array_equal(modeled.layers["feast_quantiles"], observed.layers["feast_quantiles"])
    with pytest.raises(ValueError, match="zero_target_source"):
        FEAST.simulate_local_references(refs, target, zero_target_source="invalid", **options)
