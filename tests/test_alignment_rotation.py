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


def test_plate_bounds_are_inclusive() -> None:
    source = _fixture()

    rotated = rotate_spatial(
        source,
        0.0,
        plate_bounds=np.array([[0.0, 0.0], [2.0, 1.5]]),
    )

    assert rotated.obs_names.tolist() == ["spot_0", "spot_1", "spot_2"]


def test_plate_crop_metadata_roundtrips_h5ad(tmp_path) -> None:
    source = _fixture()
    cropped = rotate_spatial(
        source,
        45.0,
        center=np.array([0.0, 0.0]),
        plate_bounds=np.array([[-0.25, -0.25], [1.5, 1.5]]),
    )
    output = tmp_path / "cropped.h5ad"

    cropped.write_h5ad(output)
    restored = ad.read_h5ad(output)

    assert restored.obs_names.equals(cropped.obs_names)
    metadata = restored.uns["feast_alignment_transform"]
    np.testing.assert_array_equal(
        metadata["plate_bounds"], np.array([[-0.25, -0.25], [1.5, 1.5]])
    )
    assert metadata["retained_n_spots"] == cropped.n_obs
    assert metadata["dropped_n_spots"] == source.n_obs - cropped.n_obs


def test_plate_bounds_can_drop_all_spots() -> None:
    cropped = rotate_spatial(
        _fixture(),
        0.0,
        plate_bounds=np.array([[100.0, 100.0], [101.0, 101.0]]),
    )

    assert cropped.n_obs == 0
    metadata = cropped.uns["feast_alignment_transform"]
    assert metadata["retained_n_spots"] == 0
    assert metadata["dropped_n_spots"] == 5
    assert metadata["retained_fraction"] == 0.0


@pytest.mark.parametrize(
    "plate_bounds",
    [
        np.array([0.0, 1.0]),
        np.array([[0.0, 0.0], [np.nan, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_rotation_rejects_invalid_plate_bounds(plate_bounds: np.ndarray) -> None:
    with pytest.raises(ValueError, match="plate_bounds"):
        rotate_spatial(_fixture(), 10.0, plate_bounds=plate_bounds)


def test_rotation_rejects_nonfinite_coordinates_and_duplicate_ids() -> None:
    nonfinite = _fixture()
    nonfinite.obsm["spatial"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rotate_spatial(nonfinite, 10.0)

    duplicate = _fixture()
    duplicate.obs_names = ["duplicate"] * duplicate.n_obs
    with pytest.raises(ValueError, match="identifiers"):
        rotate_spatial(duplicate, 10.0)
