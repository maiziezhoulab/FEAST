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


def test_recorded_inverse_recovers_original_coordinates() -> None:
    source = _fixture()
    rotated = rotate_spatial(source, 37.0)
    inverse = rotated.uns["feast_alignment_transform"]["inverse_matrix"]

    recovered = apply_spatial_transform(rotated.obsm["spatial"], inverse)

    np.testing.assert_allclose(recovered, source.obsm["spatial"], atol=1e-12)


def test_rotation_rejects_nonfinite_coordinates_and_duplicate_ids() -> None:
    nonfinite = _fixture()
    nonfinite.obsm["spatial"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rotate_spatial(nonfinite, 10.0)

    duplicate = _fixture()
    duplicate.obs_names = ["duplicate"] * duplicate.n_obs
    with pytest.raises(ValueError, match="identifiers"):
        rotate_spatial(duplicate, 10.0)
