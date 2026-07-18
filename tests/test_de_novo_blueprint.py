import numpy as np
import pandas as pd
import pytest

from FEAST.de_novo import SimulationBlueprintBuilder, SimulationParameterBuilder, SimulationBlueprint, load_blueprint


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


def test_blueprint_builder_rectangular_grid_and_domains():
    bp = (
        SimulationBlueprintBuilder.rectangular_grid(2, 3, spacing=(2.0, 1.0), origin=(10.0, 5.0))
        .set_domains(["a", "a", "b", "b", "c", "c"])
        .set_mask([True, True, True, False, True, True])
        .set_obs_column("batch", [1, 1, 1, 2, 2, 2])
        .set_metadata(version="test")
        .build()
    )
    assert bp.coordinates.shape == (6, 2)
    assert bp.grid_type == "rectangular"
    assert list(bp.obs["domain"]) == ["a", "a", "b", "b", "c", "c"]
    assert bp.metadata["version"] == "test"


def test_parameter_cloud_builder():
    cloud = (
        SimulationParameterBuilder.from_gene_names(["g1", "g2"])
        .set_all(2.0, 3.0, 0.1)
        .set_gene("g2", 5.0, 8.0, 0.2, label="domain_b")
        .build()
    )
    assert "__default__" in cloud
    assert "domain_b" in cloud
    assert list(cloud["__default__"].columns) == ["mean", "variance", "zero_prop"]
    assert cloud["domain_b"].loc["g2", "mean"] == 5.0
