import numpy as np

from FEAST.de_novo import (
    SimulationBlueprintBuilder,
    SimulationParameterBuilder,
    SimulationPatternBuilder,
    simulate_from_design,
)


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


def test_simulate_from_design_quantile_input():
    genes = ["g1", "g2"]
    blueprint = SimulationBlueprintBuilder.rectangular_grid(2, 2).build()
    parameter_cloud = SimulationParameterBuilder.from_gene_names(genes).set_all(1.5, 2.0, 0.0).build()
    quantiles = np.array([[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.8, 0.9]], dtype=np.float32)
    result = simulate_from_design(
        blueprint,
        parameter_cloud,
        quantiles=quantiles,
        random_seed=3,
    )
    assert result.shape == (4, 2)
    np.testing.assert_allclose(result.layers["feast_quantiles"], quantiles)
    assert "transported_quantiles" not in result.layers
