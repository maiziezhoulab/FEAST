import anndata as ad
import numpy as np
import pandas as pd
import pytest

from FEAST.de_novo import (
    SimulationBlueprint,
    SimulationBlueprintBuilder,
    SimulationParameterBuilder,
    SimulationPatternBuilder,
    ReferenceFitConfig,
    fit_reference,
    load_blueprint,
    simulate_from_design,
    simulate_from_reference,
)


def test_blueprint_validation_and_roundtrip():
    bp = SimulationBlueprint(
        coordinates=np.array([[0, 0, 1], [1, 0, 1]], dtype=float),
        domain_map=np.array(["a", "b"]),
        obs=pd.DataFrame({"quality": [1, 2]}),
        metadata={"source": "test"},
    )
    assert bp.coordinates.shape == (2, 3)
    assert list(bp.obs["domain"]) == ["a", "b"]
    loaded = load_blueprint(bp.to_dict())
    assert loaded.n_spots == 2
    np.testing.assert_allclose(loaded.coordinates, bp.coordinates)


def test_blueprint_rejects_bad_shapes():
    with pytest.raises(ValueError):
        SimulationBlueprint(coordinates=np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        SimulationBlueprint(coordinates=np.zeros((2, 2)), mask=np.array([True]))
    with pytest.raises(ValueError):
        SimulationBlueprint(coordinates=np.zeros((2, 2)), obs=pd.DataFrame(index=[0]))


def test_simulate_from_design_smoke():
    genes = ["g1", "g2", "g3"]
    blueprint = SimulationBlueprintBuilder.rectangular_grid(2, 2).set_domains(["a", "a", "b", "b"]).build()
    parameter_cloud = SimulationParameterBuilder.from_gene_names(genes).set_all(2.0, 3.0, 0.1).build()
    pattern_spec = (
        SimulationPatternBuilder.from_gene_names(genes)
        .gradient("g1", axis="x")
        .hotspot("g2", center=[0.5, 0.5], radius=0.25)
        .build()
    )

    result = simulate_from_design(
        blueprint,
        parameter_cloud,
        pattern_spec=pattern_spec,
        random_seed=11,
    )

    assert result.shape == (4, 3)
    assert list(result.var_names) == genes
    assert "spatial" in result.obsm
    assert result.obsm["spatial"].shape == (4, 2)
    assert "counts" in result.layers
    assert "feast_quantiles" in result.layers
    assert "transported_quantiles" not in result.layers
    assert np.issubdtype(result.X.dtype, np.integer)
    assert np.all(result.X >= 0)
    assert result.uns["de_novo"]["designed_generation"] is True


def _reference(name: str) -> ad.AnnData:
    X = np.array(
        [
            [1, 0, 2],
            [2, 1, 0],
            [0, 3, 1],
            [4, 1, 2],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame({"domain": ["a", "a", "b", "b"]})
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.obsm["spatial"] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    adata.uns["reference_name"] = name
    return adata


def test_conditional_reference_generation_smoke():
    model = fit_reference(
        [_reference("r1"), _reference("r2")],
        "domain",
        ReferenceFitConfig(min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0),
    )
    blueprint = SimulationBlueprintBuilder.rectangular_grid(2, 2).set_domains(["a", "a", "b", "b"]).build()

    result = simulate_from_reference(model, blueprint, random_seed=5)

    assert result.shape == (4, 3)
    assert list(result.var_names) == ["g1", "g2", "g3"]
    assert "spatial" in result.obsm
    assert "counts" in result.layers
    assert "feast_quantiles" in result.layers
    assert "transported_quantiles" not in result.layers
    assert np.issubdtype(result.X.dtype, np.integer)
    assert np.all(result.X >= 0)
    assert result.uns["de_novo"]["conditional_generation"] is True
    assert set(result.uns["de_novo"]["transport_weights"]) == {"a", "b"}
