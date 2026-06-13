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
from FEAST.FEAST_core.simulator import resolve_quantile_calibration_source
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


def test_resolve_simulation_mode_validation_and_aliases():
    assert resolve_simulation_mode("GENERATIVE") == "generative"
    assert resolve_simulation_mode("real_stats") == "empirical"
    with pytest.raises(ValueError, match="simulation_mode"):
        resolve_simulation_mode("bad")


def test_resolve_quantile_calibration_source_defaults_and_aliases():
    assert resolve_quantile_calibration_source(None, "empirical") == "reference_rank"
    assert resolve_quantile_calibration_source(None, "generative") == "raw"
    assert resolve_quantile_calibration_source("rank", "generative") == "reference_rank"
    assert resolve_quantile_calibration_source("iid", "empirical") == "raw"
    with pytest.raises(ValueError, match="quantile_calibration"):
        resolve_quantile_calibration_source("bad", "generative")


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
    assert diagnostics["assignment_method"] == "copula_rank_ot"
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
