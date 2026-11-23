import numpy as np
import pytest

from FEAST.FEAST_core.count_decoding import _match_zero_count_and_mean


@pytest.mark.parametrize(
    ("bag", "target_mean", "expected"),
    [
        pytest.param([0, 1, 2, 7], 2.5, [0, 1, 2, 7], id="already-feasible"),
        pytest.param([0, 1, 2, 7], 4.25, [0, 1, 3, 13], id="scale-excess"),
        pytest.param([0, 0, 1, 1], 2.0, [0, 2, 3, 3], id="all-positives-start-at-one"),
    ],
)
def test_match_zero_count_and_mean(bag, target_mean, expected):
    calibrated = _match_zero_count_and_mean(np.array(bag), target_zero=0.25, target_mean=target_mean)
    np.testing.assert_array_equal(calibrated, expected)
