import numpy as np
import pandas as pd
import pytest

from FEAST.FEAST_core.parameter_cloud import (
    GeneParameterSimulator,
    apply_alteration_to_stats,
    calculate_fold_change,
    convert_params_for_new_simulator,
    pseudo_observations,
    resolve_simulation_mode,
)
from FEAST.FEAST_core import simulator as simulator_module
from FEAST.FEAST_core.simulator import simulate_single_slice, SpatialSimulator, SPATIAL_MODES, PARAMETER_MODES
from FEAST.modeling.marginal_alteration import AlterationConfig


def _stats():
    return pd.DataFrame(
        {
            "mean": [1.0, 2.0, 4.0, 8.0],
            "variance": [2.0, 3.0, 5.0, 9.0],
            "zero_prop": [0.1, 0.2, 0.3, 0.4],
        },
        index=["g1", "g2", "g3", "g4"],
    )


def _adata():
    import anndata as ad
    X = np.array([[1, 0], [2, 1], [0, 3]], dtype=np.int32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["s1", "s2", "s3"]), var=pd.DataFrame(index=["g1", "g2"]))
    adata.obsm["spatial"] = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    return adata


def _adata_large():
    """Return an AnnData with enough genes (>=10) for mixture model fitting."""
    import anndata as ad
    rng = np.random.RandomState(42)
    n_obs, n_vars = 20, 12
    X = rng.poisson(lam=5, size=(n_obs, n_vars)).astype(np.int32)
    genes = [f"g{i}" for i in range(n_vars)]
    spots = [f"s{i}" for i in range(n_obs)]
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=spots), var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(n_obs, 2)).astype(float)
    return adata


def test_resolve_simulation_mode_validation_and_aliases():
    assert resolve_simulation_mode("GENERATIVE") == "generative"
    assert resolve_simulation_mode("real_stats") == "empirical"
    with pytest.raises(ValueError, match="simulation_mode"):
        resolve_simulation_mode("bad")


def test_spatial_mode_validation_via_public_api():
    """spatial_mode replaces quantile_calibration; validates through simulate_single_slice."""
    # reference_rank works without target_adata
    result = simulate_single_slice(_adata(), parameter_mode="reference_stats", spatial_mode="reference_rank",
                                   random_seed=1, verbose=False, clip_overshoot_factor=0.0)
    assert result.uns["simulation_diagnostics"]["spatial_mode"] == "reference_rank"

    # ot_spatial requires target_adata
    with pytest.raises(ValueError, match="target_adata is required"):
        simulate_single_slice(_adata(), spatial_mode="ot_spatial", verbose=False, clip_overshoot_factor=0.0)

    # Invalid spatial_mode
    with pytest.raises(ValueError, match="spatial_mode"):
        simulate_single_slice(_adata(), spatial_mode="iid", verbose=False, clip_overshoot_factor=0.0)


def _run_shared_transport_with_recorded_pairs(monkeypatch, max_transport_pairs=None):
    from FEAST.de_novo import transport as transport_module
    from FEAST.de_novo.transport import TransportConfig, transport_reference_field

    calls = []

    def fake_sinkhorn_transport(
        M,
        a,
        b,
        reg=0.05,
        numItermax=200,
        stopThr=1e-5,
        unbalanced=False,
        reg_m=5.0,
        nonconvergence="raise",
        return_diagnostics=False,
    ):
        calls.append(M.shape)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a / a.sum()
        b = b / b.sum()
        plan = np.outer(a, b).astype(np.float32)
        diagnostics = {
            "converged": True,
            "iterations": None,
            "final_error": None,
            "stop_threshold": stopThr,
            "max_iterations": numItermax,
            "unbalanced": unbalanced,
        }
        return (plan, diagnostics) if return_diagnostics else plan

    monkeypatch.setattr(transport_module, "sinkhorn_transport", fake_sinkhorn_transport)

    coords = np.column_stack([np.arange(20, dtype=float), np.zeros(20, dtype=float)])
    source_quantiles = np.column_stack(
        [
            (np.arange(20, dtype=float) + 0.5) / 20.0,
            (np.arange(20, 0, -1, dtype=float) - 0.5) / 20.0,
        ]
    )
    result = transport_reference_field(
        coords,
        coords,
        source_quantiles,
        config=TransportConfig(
            unbalanced_transport=False,
            max_transport_pairs=max_transport_pairs,
        ),
    )
    return calls, result


def test_shared_transport_uses_one_plan_without_pair_cap(monkeypatch):
    calls, result = _run_shared_transport_with_recorded_pairs(monkeypatch)

    assert calls == [(20, 20)]
    assert result.latent_scores.shape == (20, 2)
    assert result.diagnostics["blocked"] is False


def test_shared_transport_pair_cap_blocks_only_target_columns(monkeypatch):
    calls, result = _run_shared_transport_with_recorded_pairs(
        monkeypatch,
        max_transport_pairs=80,
    )

    assert len(calls) > 1
    assert max(n_src * n_tgt for n_src, n_tgt in calls) <= 80
    assert {n_src for n_src, _ in calls} == {20}
    assert result.latent_scores.shape == (20, 2)
    assert result.diagnostics["blocked"] is True


def test_shared_transport_memory_controls_validate():
    from FEAST.de_novo.transport import max_pairs_from_memory_budget

    with pytest.raises(ValueError, match="memory_budget_gb"):
        max_pairs_from_memory_budget(0)
    assert not hasattr(simulator_module, "_block_ot_transport")
    assert not hasattr(simulator_module, "_resolve_ot_block_max_pairs")


def test_parameter_mode_constants():
    """Verify PARAMETER_MODES and their translation to internal simulation_mode."""
    assert set(PARAMETER_MODES) == {"hungarian", "reference_stats"}
    # hungarian → generative (needs enough genes for model fitting)
    adata_large = _adata_large()
    result_gen = simulate_single_slice(adata_large, parameter_mode="hungarian", random_seed=1,
                                       verbose=False, clip_overshoot_factor=0.0)
    assert result_gen.uns["simulation_diagnostics"]["simulation_mode"] == "generative"
    # reference_stats → empirical
    result_emp = simulate_single_slice(_adata(), parameter_mode="reference_stats", random_seed=1,
                                       verbose=False, clip_overshoot_factor=0.0)
    assert result_emp.uns["simulation_diagnostics"]["simulation_mode"] == "empirical"


def test_apply_alteration_to_stats_changes_only_selected_columns():
    stats = _stats()
    altered = apply_alteration_to_stats(stats, AlterationConfig.mean_only(0.8))
    np.testing.assert_allclose(altered["mean"], stats["mean"] * 0.8)
    np.testing.assert_allclose(altered["variance"], stats["variance"])
    np.testing.assert_allclose(altered["zero_prop"], stats["zero_prop"])


def test_empirical_parameter_table_preserves_gene_ids_and_target_fold_change():
    simulator = GeneParameterSimulator()
    simulator.original_stats = _stats()
    table, diagnostics = simulator.build_gene_parameter_table(
        alteration_config=AlterationConfig.mean_only(0.8),
        simulation_mode="empirical",
        random_seed=123,
        verbose=False,
    )
    assert table["gene_id"].tolist() == ["g1", "g2", "g3", "g4"]
    assert diagnostics["assignment_method"] == "identity"
    assert diagnostics["gene_parameter_engine"] == "empirical"
    assert diagnostics["target_stage_achieved_change"]["mean"] == pytest.approx(0.8)
    np.testing.assert_allclose(table["variance"], _stats()["variance"])


def test_copula_rank_assignment_uses_pseudo_observation_space():
    simulator = GeneParameterSimulator()
    simulator.original_stats = _stats()
    synthetic = pd.DataFrame(
        {
            "mean": [8.0, 4.0, 2.0, 1.0],
            "variance": [9.0, 5.0, 3.0, 2.0],
            "zero_prop": [0.4, 0.3, 0.2, 0.1],
        }
    )
    synthetic_u = pseudo_observations(synthetic).to_numpy()
    assigned, diagnostics = simulator.assign_to_genes_copula_rank(
        synthetic,
        synthetic_u,
        random_seed=0,
        verbose=False,
    )
    assert diagnostics["assignment_method"] == "copula_rank"
    assert sorted(assigned["gene_id"].tolist()) == ["g1", "g2", "g3", "g4"]
    assert diagnostics["mean_cost"] >= 0.0


def test_copula_rank_assignment_can_select_from_overgenerated_pool():
    simulator = GeneParameterSimulator()
    simulator.original_stats = _stats()
    synthetic = pd.DataFrame(
        {
            "mean": [100.0, 1.0, 2.0, 4.0, 8.0],
            "variance": [100.0, 2.0, 3.0, 5.0, 9.0],
            "zero_prop": [0.9, 0.1, 0.2, 0.3, 0.4],
        }
    )
    synthetic_u = pseudo_observations(synthetic).to_numpy()
    assigned, diagnostics = simulator.assign_to_genes_copula_rank(
        synthetic,
        synthetic_u,
        random_seed=0,
        verbose=False,
    )
    assert diagnostics["n_profiles"] == 4
    assert diagnostics["n_candidates"] == 5
    assert sorted(assigned["gene_id"].tolist()) == ["g1", "g2", "g3", "g4"]


def test_convert_params_uses_gene_id_column_as_gene_names():
    stats = _stats().reset_index().rename(columns={"index": "gene_id"})
    converted = convert_params_for_new_simulator(stats)
    assert converted["genes"][0] == "g1"
    assert len(converted["model_selected"]) == 4


def test_calculate_fold_change_accepts_gene_id_column():
    stats = _stats()
    target = stats.copy()
    target["mean"] *= 0.9
    target = target.reset_index().rename(columns={"index": "gene_id"})
    changes = calculate_fold_change(stats, target)
    assert changes["mean"] == pytest.approx(0.9)
