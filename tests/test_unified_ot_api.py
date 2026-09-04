import anndata as ad
import numpy as np
import pandas as pd
import pytest

import FEAST
from FEAST import ReferenceFitConfig, TransportConfig


def _reference() -> ad.AnnData:
    counts = np.array(
        [
            [8, 0, 1],
            [5, 1, 2],
            [2, 4, 0],
            [0, 7, 3],
            [1, 5, 6],
            [3, 2, 8],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {"domain": ["a", "a", "a", "b", "b", "b"]},
        index=[f"s{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    result = ad.AnnData(X=counts, obs=obs, var=var)
    result.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
        dtype=float,
    )
    return result


def _target(reference: ad.AnnData, fill_value: int) -> ad.AnnData:
    target = reference.copy()
    target.X = np.full(reference.shape, fill_value, dtype=np.int32)
    target.obsm["spatial"] = np.asarray(reference.obsm["spatial"])[::-1].copy()
    return target


def _fit_config() -> ReferenceFitConfig:
    return ReferenceFitConfig(
        min_gene_spots=1,
        min_gene_mean=0.0,
        max_gene_zero_prop=1.0,
    )


def _balanced_transport(**kwargs) -> TransportConfig:
    return TransportConfig(unbalanced_transport=False, **kwargs)


def test_parameter_cloud_and_empirical_marginals_share_conditioned_ot_field():
    reference = _reference()
    target = _target(reference, fill_value=999)
    cloud = pd.DataFrame(
        {
            "mean": [12.0, 10.0, 8.0],
            "variance": [18.0, 16.0, 14.0],
            "zero_prop": [0.05, 0.10, 0.15],
        },
        index=reference.var_names,
    )

    empirical = FEAST.simulate(
        reference,
        target,
        condition_on="domain",
        marginal_model="empirical_reference",
        transport=_balanced_transport(),
        fit_config=_fit_config(),
        seed=7,
        verbose=False,
    )
    parameterized = FEAST.simulate(
        reference,
        target,
        condition_on="domain",
        marginal_model="parameter_cloud",
        parameter_cloud=cloud,
        transport=_balanced_transport(),
        fit_config=_fit_config(),
        seed=7,
        verbose=False,
    )

    np.testing.assert_array_equal(
        empirical.layers["feast_quantiles"],
        parameterized.layers["feast_quantiles"],
    )
    assert empirical.uns["de_novo"]["marginal_model"] == "empirical_reference"
    assert parameterized.uns["de_novo"]["marginal_model"] == "parameter_cloud"
    assert empirical.uns["de_novo"]["decode_method"] == "empirical_rank"
    assert (
        parameterized.uns["de_novo"]["decode_method"]
        == "parameter_cloud_spatial_intensity"
    )
    assert (
        empirical.uns["de_novo"]["quantile_field"]["method_version"]
        == "unified_latent_ot_v1"
    )
    np.testing.assert_allclose(parameterized.var["target_mean"], cloud["mean"])


def test_label_specific_parameter_clouds_use_target_weighted_global_stats():
    reference = _reference()
    target = _target(reference, fill_value=0)
    target.obs["domain"] = ["a", "a", "a", "a", "b", "b"]
    cloud_a = pd.DataFrame(
        {
            "mean": [2.0, 4.0, 6.0],
            "variance": [3.0, 5.0, 7.0],
            "zero_prop": [0.1, 0.2, 0.3],
        },
        index=reference.var_names,
    )
    cloud_b = pd.DataFrame(
        {
            "mean": [8.0, 10.0, 12.0],
            "variance": [13.0, 15.0, 17.0],
            "zero_prop": [0.4, 0.5, 0.6],
        },
        index=reference.var_names,
    )

    result = FEAST.simulate(
        reference,
        target,
        condition_on="domain",
        marginal_model="parameter_cloud",
        parameter_cloud={"a": cloud_a, "b": cloud_b},
        transport=_balanced_transport(),
        fit_config=_fit_config(),
        seed=7,
        verbose=False,
    )

    np.testing.assert_allclose(result.var["target_mean"], [4.0, 6.0, 8.0])
    np.testing.assert_allclose(
        result.var["target_variance"],
        [43.0 / 3.0, 49.0 / 3.0, 55.0 / 3.0],
    )
    np.testing.assert_allclose(result.var["target_zero_prop"], [0.2, 0.3, 0.4])
    assert set(result.uns["de_novo"]["parameter_cloud_summary"]) == {"a", "b"}


def test_reference_based_ot_field_ignores_target_expression_and_uses_shared_method():
    reference = _reference()
    target_zero = _target(reference, fill_value=0)
    target_large = _target(reference, fill_value=999)

    common = dict(
        parameter_mode="reference_stats",
        transport=_balanced_transport(),
        seed=11,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    first = FEAST.simulate(reference, target_zero, **common)
    second = FEAST.simulate(reference, target_large, **common)

    np.testing.assert_array_equal(
        first.layers["feast_quantiles"],
        second.layers["feast_quantiles"],
    )
    assert first.uns["simulation_diagnostics"]["spatial_mode"] == "ot_spatial"
    assert first.uns["simulation_diagnostics"]["marginal_model"] == "parameter_cloud"
    assert (
        first.uns["simulation_diagnostics"]["transport"]["field_space"]
        == "normal_score"
    )
    assert (
        first.uns["simulation_diagnostics"]["method_version"]
        == "unified_latent_ot_v1"
    )


def test_assignment_randomness_is_validated_by_shared_transport_config():
    reference = _reference()
    target = _target(reference, fill_value=0)

    for invalid in (-0.01, 1.01):
        with pytest.raises(ValueError, match="assignment_randomness"):
            FEAST.simulate(
                reference,
                target,
                parameter_mode="reference_stats",
                transport=_balanced_transport(assignment_randomness=invalid),
                seed=3,
                verbose=False,
            )


@pytest.mark.parametrize("invalid", [0.0, -1e-6, 0.5, 0.6, np.nan, np.inf])
def test_global_ot_rejects_invalid_latent_clip_eps(invalid):
    reference = _reference()
    target = _target(reference, fill_value=0)

    with pytest.raises(ValueError, match="latent_clip_eps"):
        FEAST.simulate(
            reference,
            target,
            parameter_mode="reference_stats",
            transport=_balanced_transport(latent_clip_eps=invalid),
            seed=3,
            verbose=False,
        )


def test_balanced_sinkhorn_reports_positive_convergence_evidence():
    from FEAST.de_novo._ot_transport import sinkhorn_transport

    cost = np.array(
        [
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ]
    )
    mass = np.full(3, 1.0 / 3.0)

    _, diagnostics = sinkhorn_transport(
        cost,
        mass,
        mass,
        reg=0.5,
        numItermax=1000,
        stopThr=1e-9,
        return_diagnostics=True,
    )

    assert diagnostics["converged"] is True
    assert isinstance(diagnostics["iterations"], int)
    assert 0 <= diagnostics["iterations"] < diagnostics["max_iterations"]
    assert np.isfinite(diagnostics["final_error"])
    assert diagnostics["final_error"] < diagnostics["stop_threshold"]


@pytest.mark.parametrize("policy", ["raise", "warn"])
def test_balanced_sinkhorn_nonconvergence_obeys_policy(monkeypatch, policy):
    from FEAST.de_novo import _ot_transport

    def finite_nonconverged_plan(*args, **kwargs):
        assert kwargs["log"] is True
        return np.full((2, 2), 0.25), {"err": [0.2, 0.1], "niter": 2}

    monkeypatch.setattr(_ot_transport.ot, "sinkhorn", finite_nonconverged_plan)

    def call():
        return _ot_transport.sinkhorn_transport(
            np.zeros((2, 2)),
            np.ones(2),
            np.ones(2),
            numItermax=2,
            stopThr=1e-5,
            nonconvergence=policy,
            return_diagnostics=True,
        )

    if policy == "raise":
        with pytest.raises(_ot_transport.OptimalTransportError, match="Balanced OT"):
            call()
    else:
        with pytest.warns(
            _ot_transport.OptimalTransportConvergenceWarning,
            match="Balanced OT",
        ):
            _, diagnostics = call()
        assert diagnostics["converged"] is False
        assert diagnostics["iterations"] == 2
        assert diagnostics["final_error"] == pytest.approx(0.1)


def test_global_ot_diagnostics_roundtrip_through_h5ad(tmp_path):
    reference = _reference()
    target = _target(reference, fill_value=0)

    result = FEAST.simulate(
        reference,
        target,
        parameter_mode="reference_stats",
        transport=_balanced_transport(
            epsilon=0.5,
            sinkhorn_tol=1e-7,
            max_transport_pairs=12,
        ),
        seed=13,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    output = tmp_path / "global_ot.h5ad"
    result.write_h5ad(output)
    restored = ad.read_h5ad(output)

    np.testing.assert_array_equal(restored.X, result.X)
    np.testing.assert_array_equal(
        restored.layers["feast_quantiles"],
        result.layers["feast_quantiles"],
    )
    transport = restored.uns["simulation_diagnostics"]["transport"]
    blocks = transport["blocks"]
    assert transport["converged"]
    assert transport["n_blocks"] == 3
    assert blocks["format"] == "columnar_records_v1"
    assert blocks["n_records"] == transport["n_blocks"]
    assert set(blocks) == {
        "format",
        "n_records",
        "converged",
        "iterations",
        "final_error",
        "stop_threshold",
        "max_iterations",
        "unbalanced",
        "block_index",
        "source_spots",
        "target_spots",
        "transport_mass",
        "solver_method",
        "transport_backend",
        "transport_device",
        "transport_dtype",
    }
    assert list(blocks["converged"]) == ["True", "True", "True"]
    assert list(blocks["unbalanced"]) == ["False", "False", "False"]
    assert all(int(value) >= 0 for value in blocks["iterations"])
    assert all(
        float(value) < transport["stop_threshold"]
        for value in blocks["final_error"]
    )


def test_pot_torch_backend_preserves_device_dtype_and_solver_diagnostics():
    from FEAST.de_novo.transport import TransportConfig, transport_reference_field

    source_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target_coordinates = np.array([[0.1, 0.1], [0.8, 0.1]])
    source_quantiles = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    result = transport_reference_field(
        source_coordinates,
        target_coordinates,
        source_quantiles,
        config=TransportConfig(
            epsilon=0.5,
            unbalanced_transport=True,
            sinkhorn_method="sinkhorn_stabilized",
            transport_backend="torch",
            transport_device="cpu",
            transport_dtype="float64",
        ),
    )

    assert isinstance(result.latent_scores, np.ndarray)
    assert result.diagnostics["converged"] is True
    assert result.diagnostics["solver_method"] == "sinkhorn_stabilized"
    assert result.diagnostics["transport_backend"] == "torch"
    assert result.diagnostics["transport_device"] == "cpu"
    assert result.diagnostics["transport_dtype"] == "float64"


def test_pot_torch_and_numpy_stabilized_backends_agree_on_latent_field():
    from FEAST.de_novo.transport import TransportConfig, transport_reference_field

    source_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target_coordinates = np.array([[0.1, 0.1], [0.8, 0.1]])
    source_quantiles = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    common = dict(
        epsilon=0.5,
        unbalanced_transport=True,
        sinkhorn_method="sinkhorn_stabilized",
        transport_dtype="float64",
    )
    numpy_result = transport_reference_field(
        source_coordinates,
        target_coordinates,
        source_quantiles,
        config=TransportConfig(**common, transport_backend="numpy"),
    )
    torch_result = transport_reference_field(
        source_coordinates,
        target_coordinates,
        source_quantiles,
        config=TransportConfig(**common, transport_backend="torch", transport_device="cpu"),
    )

    np.testing.assert_allclose(torch_result.latent_scores, numpy_result.latent_scores, rtol=1e-6, atol=1e-6)


def test_transport_rejects_unavailable_cuda_without_cpu_fallback(monkeypatch):
    import torch
    from FEAST.de_novo.transport import TransportConfig, transport_reference_field

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requested POT transport device 'cuda:0' is unavailable"):
        transport_reference_field(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.5, 0.5]]),
            np.array([[0.2], [0.8]]),
            config=TransportConfig(
                transport_backend="torch",
                transport_device="cuda:0",
            ),
        )


def test_legacy_adata_keyword_is_a_deprecated_compatibility_alias():
    with pytest.warns(DeprecationWarning, match="reference"):
        result = FEAST.simulate(
            adata=_reference(),
            parameter_mode="reference_stats",
            seed=5,
            verbose=False,
        )

    assert result.shape == (6, 3)


def test_transport_config_without_target_uses_reference_as_ot_target():
    reference = _reference()

    result = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        transport=_balanced_transport(),
        seed=9,
        verbose=False,
    )

    np.testing.assert_array_equal(result.obsm["spatial"], reference.obsm["spatial"])
    assert result.uns["simulation_diagnostics"]["spatial_mode"] == "ot_spatial"
