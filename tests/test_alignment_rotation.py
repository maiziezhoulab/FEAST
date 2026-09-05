from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from FEAST.alignment import apply_spatial_transform, rotate_spatial


def _fixture() -> ad.AnnData:
    coords = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.5, 1.5], [3.0, 2.0], [-1.0, 0.5]]
    )
    obs = pd.DataFrame(index=[f"spot_{i}" for i in range(len(coords))])
    var = pd.DataFrame(index=["gene_a", "gene_b"])
    return ad.AnnData(
        X=np.arange(len(coords) * 2).reshape(len(coords), 2),
        obs=obs,
        var=var,
        obsm={"spatial": coords},
    )


@pytest.mark.parametrize("angle", [0.0, 1.0, 45.0, 90.0])
def test_centered_rotation_preserves_geometry_and_identity(angle: float) -> None:
    source = _fixture()
    source_coords = source.obsm["spatial"].copy()

    rotated = rotate_spatial(source, angle)

    assert rotated.shape == source.shape
    assert rotated.obs_names.equals(source.obs_names)
    np.testing.assert_allclose(rotated.obsm["spatial"].mean(axis=0), source_coords.mean(axis=0))
    np.testing.assert_allclose(pdist(rotated.obsm["spatial"]), pdist(source_coords))
    np.testing.assert_array_equal(rotated.obsm["spatial_original"], source_coords)
    np.testing.assert_array_equal(source.obsm["spatial"], source_coords)
    metadata = rotated.uns["feast_alignment_transform"]
    assert metadata["plate_bounds_applied"] is False
    assert "plate_bounds" not in metadata
    assert metadata["preserve_spot_count"] is True
    assert metadata["input_n_spots"] == source.n_obs
    assert metadata["retained_n_spots"] == source.n_obs
    assert metadata["dropped_n_spots"] == 0
    assert metadata["retained_fraction"] == 1.0


def test_recorded_inverse_recovers_original_coordinates() -> None:
    source = _fixture()
    rotated = rotate_spatial(source, 37.0)
    inverse = rotated.uns["feast_alignment_transform"]["inverse_matrix"]

    recovered = apply_spatial_transform(rotated.obsm["spatial"], inverse)

    np.testing.assert_allclose(recovered, source.obsm["spatial"], atol=1e-12)


def test_rotation_with_plate_bounds_crops_and_preserves_order() -> None:
    source = _fixture()
    source_coords = source.obsm["spatial"].copy()
    plate_bounds = np.array([[-2.1, -0.1], [0.1, 2.1]])

    rotated = rotate_spatial(
        source,
        90.0,
        center=np.array([0.0, 0.0]),
        plate_bounds=plate_bounds,
    )

    assert rotated.obs_names.tolist() == ["spot_0", "spot_1", "spot_2"]
    np.testing.assert_array_equal(rotated.X, source.X[:3])
    np.testing.assert_array_equal(rotated.obsm["spatial_original"], source_coords[:3])
    recovered = apply_spatial_transform(
        rotated.obsm["spatial"],
        rotated.uns["feast_alignment_transform"]["inverse_matrix"],
    )
    np.testing.assert_allclose(recovered, source_coords[:3], atol=1e-12)
    np.testing.assert_array_equal(source.obsm["spatial"], source_coords)

    metadata = rotated.uns["feast_alignment_transform"]
    np.testing.assert_array_equal(metadata["plate_bounds"], plate_bounds)
    assert metadata["plate_bounds_applied"] is True
    assert metadata["preserve_spot_identity"] is True
    assert metadata["preserve_spot_count"] is False
    assert metadata["input_n_spots"] == 5
    assert metadata["retained_n_spots"] == 3
    assert metadata["dropped_n_spots"] == 2
    assert metadata["retained_fraction"] == pytest.approx(0.6)
    assert metadata["n_spots"] == 5
