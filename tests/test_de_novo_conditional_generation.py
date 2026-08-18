import numpy as np
import pandas as pd
import anndata as ad

from FEAST.de_novo import (
    SimulationBlueprintBuilder,
    ReferenceFitConfig,
    fit_reference,
    simulate_from_reference,
)


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
