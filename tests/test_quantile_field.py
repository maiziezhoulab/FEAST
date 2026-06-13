import numpy as np
import pandas as pd
import anndata as ad
import pytest

from FEAST.de_novo import (
    ReferenceFitConfig,
    SimulationBlueprintBuilder,
    SimulationConfig,
    SimulationParameterBuilder,
    SimulationPatternBuilder,
    fit_reference,
    simulate_from_design,
    simulate_from_reference,
    simulate_stack,
)
from FEAST.de_novo.quantile_field import (
    build_spatial_program_matrix,
    midpoint_rank_normalize,
    rank_normalize_by_scope,
)


def test_stable_ordinal_ties_produce_midpoint_quantiles():
    scores = np.ones((5, 1), dtype=np.float32)
    q = midpoint_rank_normalize(scores, tie_policy="stable_ordinal")
    np.testing.assert_allclose(q[:, 0], np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32))

    q_average = midpoint_rank_normalize(scores, tie_policy="average")
    np.testing.assert_allclose(q_average[:, 0], np.full(5, 0.5, dtype=np.float32))


def test_rank_scope_fallback_records_small_domains():
    scores = np.arange(6, dtype=np.float32).reshape(6, 1)
    labels = np.array(["a", "a", "b", "b", "c", "c"])
    coords = np.column_stack([np.arange(6, dtype=float), np.zeros(6), np.zeros(6)])

    q, meta = rank_normalize_by_scope(
        scores,
        labels=labels,
        coordinates=coords,
        rank_scope="domain_slice",
        tie_policy="stable_ordinal",
        clip_eps=1e-6,
        random_seed=0,
        tie_jitter_scale=1e-9,
        min_rank_scope_size=3,
    )

    assert q.shape == (6, 1)
    assert meta["fallbacks"]
    assert meta["fallbacks"][0]["from"] == "domain_slice"


def test_program_normalization_removes_weight_scale():
    blueprint = SimulationBlueprintBuilder.rectangular_grid(3, 3).build()
    program_a = [{"name": "x", "kind": "gradient", "axis": "x", "weight": 1.0}]
    program_b = [{"name": "x", "kind": "gradient", "axis": "x", "weight": 100.0}]

    B_a, _ = build_spatial_program_matrix(
        blueprint,
        program_a,
        label_key="domain",
        random_seed=0,
        boundary_softness=0.0,
        normalization="zscore",
    )
    B_b, _ = build_spatial_program_matrix(
        blueprint,
        program_b,
        label_key="domain",
        random_seed=0,
        boundary_softness=0.0,
        normalization="zscore",
    )

    np.testing.assert_allclose(B_a, B_b)


def test_design_pattern_uses_latent_program_metadata():
    genes = ["g1", "g2"]
    blueprint = SimulationBlueprintBuilder.rectangular_grid(3, 3).build()
    cloud = SimulationParameterBuilder.from_gene_names(genes).set_all(2.0, 3.0, 0.1).build()
    pattern_spec = SimulationPatternBuilder.from_gene_names(genes).gradient("g1", axis="x").build()

    result = simulate_from_design(blueprint, cloud, pattern_spec=pattern_spec, random_seed=4)

    qf = result.uns["de_novo"]["quantile_field"]
    assert qf["mode"] == "latent_program"
    assert qf["source"] == "program"
    assert "feast_quantiles" in result.layers
    assert "transported_quantiles" not in result.layers
    assert result.layers["feast_quantiles"].shape == (9, 2)


def test_design_quantile_mode_requires_matching_inputs():
    genes = ["g1", "g2"]
    blueprint = SimulationBlueprintBuilder.rectangular_grid(2, 2).build()
    cloud = SimulationParameterBuilder.from_gene_names(genes).set_all(2.0, 3.0, 0.1).build()
    quantiles = np.full((4, 2), 0.5, dtype=np.float32)

    with pytest.raises(ValueError, match="explicit_quantile mode requires quantiles"):
        simulate_from_design(
            blueprint,
            cloud,
            config=SimulationConfig(quantile_field_mode="explicit_quantile"),
        )

    with pytest.raises(ValueError, match="quantiles input requires"):
        simulate_from_design(
            blueprint,
            cloud,
            quantiles=quantiles,
            config=SimulationConfig(quantile_field_mode="latent_program"),
        )


def _reference(name: str, left_high: bool) -> ad.AnnData:
    if left_high:
        X = np.array([[9, 1], [8, 1], [1, 8], [1, 9]], dtype=np.int32)
    else:
        X = np.array([[1, 9], [1, 8], [8, 1], [9, 1]], dtype=np.int32)
    obs = pd.DataFrame({"domain": ["a", "a", "a", "a"]})
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    adata.uns["reference_name"] = name
    return adata


def test_reference_generation_defaults_to_latent_reference_mode():
    model = fit_reference(
        [_reference("r1", True), _reference("r2", False)],
        "domain",
        ReferenceFitConfig(min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0),
    )
    blueprint = SimulationBlueprintBuilder.rectangular_grid(2, 2).set_domains(["a"] * 4).build()

    result = simulate_from_reference(model, blueprint, random_seed=7)

    qf = result.uns["de_novo"]["quantile_field"]
    assert qf["mode"] == "latent_reference"
    assert qf["source"] == "reference_transport"
    assert "legacy_quantile_averaging_used" not in qf
    assert "feast_quantiles" in result.layers
    assert "transported_quantiles" not in result.layers


def test_reference_generation_rejects_legacy_quantile_mode():
    model = fit_reference(
        [_reference("r1", True), _reference("r2", False)],
        "domain",
        ReferenceFitConfig(min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0),
    )
    blueprint = SimulationBlueprintBuilder.rectangular_grid(2, 2).set_domains(["a"] * 4).build()
    config = SimulationConfig(quantile_field_mode="legacy_quantile_average")

    with pytest.raises(ValueError, match="quantile_field mode"):
        simulate_from_reference(model, blueprint, config=config, random_seed=7)


def test_reference_latent_transport_chunking_is_equivalent():
    model = fit_reference(
        [_reference("r1", True), _reference("r2", False)],
        "domain",
        ReferenceFitConfig(min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0),
    )
    blueprint = SimulationBlueprintBuilder.rectangular_grid(2, 2).set_domains(["a"] * 4).build()

    chunked = simulate_from_reference(
        model,
        blueprint,
        config=SimulationConfig(gene_chunk_size=1),
        random_seed=7,
    )
    unchunked = simulate_from_reference(
        model,
        blueprint,
        config=SimulationConfig(gene_chunk_size=32),
        random_seed=7,
    )

    np.testing.assert_allclose(chunked.layers["feast_quantiles"], unchunked.layers["feast_quantiles"])
    np.testing.assert_array_equal(chunked.layers["counts"], unchunked.layers["counts"])


def test_stack_records_latent_quantile_field_and_target_parameter_mode():
    refs = [_reference("r1", True), _reference("r2", False)]
    target_bp = SimulationBlueprintBuilder.rectangular_grid(2, 2).set_domains(["a"] * 4).build()

    result = simulate_stack(
        refs,
        reference_z_values=[0.0, 1.0],
        target_z_values=[0.5],
        target_blueprints={0.5: target_bp},
        random_seed=2,
    )[0.5]

    qf = result.uns["de_novo"]["quantile_field"]
    assert qf["mode"] == "latent_reference"
    assert qf["target_parameter_mode"] == "reference_weighted_log"
    assert "legacy_quantile_averaging_used" not in qf
    assert result.uns["de_novo"]["stack"]["target_z"] == 0.5
