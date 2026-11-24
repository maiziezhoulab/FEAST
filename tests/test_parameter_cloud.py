from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from FEAST.FEAST_core.parameter_cloud import GeneParameterSimulator, STAT_COLUMNS


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="NumPy longdouble has the float64 range on this platform",
)
@pytest.mark.parametrize("assignment_blocks", [False, True])
def test_student_t_tail_survives_until_assignment(monkeypatch, assignment_blocks):
    simulator = GeneParameterSimulator()
    simulator.fitted = True
    simulator.original_stats = pd.DataFrame(
        [[1.0, 2.0, 0.5]], columns=STAT_COLUMNS, index=pd.Index(["gene"], name="gene_id")
    )
    quantiles = np.array([[0.5, 0.5, 0.5], [0.5, 0.999999527747, 0.5]])
    simulator.copula_model = SimpleNamespace(simulate=lambda **kwargs: quantiles)
    monkeypatch.setattr(simulator.param_models["mean"], "ppf", lambda q: np.ones_like(q))
    monkeypatch.setattr(
        simulator.param_models["zero_prop"], "ppf", lambda q, random_state: np.full_like(q, 0.5)
    )
    variance = simulator.param_models["variance"]
    variance._is_fitted = True
    variance.log_transform = True
    variance.data_range = (1e-10, 328.25)
    variance._ppf_scale = 1.0
    variance._ppf_interp = lambda q: np.full_like(q, 2.0)
    monkeypatch.setattr(variance, "_ppf_model_space_scalar", lambda q: 451.294924557)

    candidates = variance.ppf(quantiles[:, 1])
    assert np.isfinite(candidates).all()
    assert candidates[1] > np.finfo(np.float64).max
    table, _ = simulator.build_gene_parameter_table(
        assignment_blocks=assignment_blocks, overgeneration_factor=2, random_seed=2026, verbose=False
    )
    pd.testing.assert_frame_equal(table, simulator.original_stats.reset_index(), check_exact=True)
