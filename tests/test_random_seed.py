import anndata as ad
import numpy as np
import pandas as pd

import FEAST


def _small_reference() -> ad.AnnData:
    reference = ad.AnnData(
        X=np.array([[1, 0], [2, 1], [0, 3]], dtype=np.int32),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    reference.obsm["spatial"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=float,
    )
    return reference


def test_public_simulate_seed_is_independent_of_global_numpy_state():
    reference = _small_reference()

    np.random.seed(123)
    first = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=0,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    np.random.seed(987)
    second = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=0,
        verbose=False,
        clip_overshoot_factor=0.0,
    )

    np.testing.assert_array_equal(first.X, second.X)


def test_public_simulate_respects_global_numpy_seed_when_seed_is_none():
    reference = _small_reference()

    np.random.seed(2468)
    first = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=None,
        verbose=False,
        clip_overshoot_factor=0.0,
    )
    np.random.seed(2468)
    second = FEAST.simulate(
        reference,
        parameter_mode="reference_stats",
        seed=None,
        verbose=False,
        clip_overshoot_factor=0.0,
    )

    np.testing.assert_array_equal(first.X, second.X)
