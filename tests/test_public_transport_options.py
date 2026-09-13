"""Public precision and local reference-selection behavior."""
import inspect

import FEAST
import numpy as np
import pytest
from FEAST.de_novo import transport
from test_local_generation import ref, fixed_table_spy, assert_shared_group_selection


def test_public_defaults():
    assert inspect.signature(FEAST.simulate_local_references).parameters['n_references'].default == 5
    for config_type in (FEAST.TransportConfig, FEAST.SimulationConfig):
        config = config_type()
        assert config.transport_dtype == 'float64'
        assert config.sinkhorn_iter == 1000
        assert config.sinkhorn_tol == 1e-5
        assert config.transport_nonconvergence == 'raise'
        assert config.max_transport_pairs == 25_000_000


@pytest.mark.parametrize('config_type', [FEAST.TransportConfig, FEAST.SimulationConfig])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_top_level_precision_reaches_solver(monkeypatch, config_type, backend, dtype):
    data = ref('reference', 0)
    observed = []
    solve = transport.sinkhorn_transport

    def capture(**kwargs):
        observed.append(str(kwargs['M'].dtype).removeprefix('torch.'))
        assert kwargs['a'].dtype == kwargs['b'].dtype == kwargs['M'].dtype
        assert kwargs['numItermax'] == 1000
        assert kwargs['stopThr'] == 1e-5
        assert kwargs['nonconvergence'] == 'raise'
        return solve(**kwargs)

    monkeypatch.setattr(transport, 'sinkhorn_transport', capture)
    config = config_type(transport_backend=backend, transport_device='cpu',
                         transport_dtype=dtype, sinkhorn_method='sinkhorn_log')
    result = FEAST.simulate(data, target=data, condition_on='class',
        marginal_model='empirical_reference', transport=config, seed=7, verbose=False)
    assert observed == [dtype]
    record = result.uns['de_novo']['transport_diagnostics']['a']
    assert str(record['transport_dtype'][0]) == dtype
    assert str(record['transport_backend'][0]) == backend
    assert str(record['transport_converged'][0]).lower() == 'true'
    assert float(record['transport_final_error'][0]) < 1e-5
    assert result.shape == data.shape
    np.testing.assert_array_equal(result.var_names, sorted(data.var_names))
    np.testing.assert_array_equal(result.obsm['spatial'], data.obsm['spatial'])


def geometry_references():
    target = ref('target', 0, ['a'] * 60 + ['b'] * 60)
    target.obs['spot_id'] = target.obs_names.to_numpy()
    target.obsm['spatial'][60:] += 20
    references = []
    for index, name in enumerate(['a_first', 'a_second', 'b_first', 'b_second', 'neutral']):
        source = ref(name, index, ['a'] * 60 + ['b'] * 60)
        # Two exact geometry matches per region; every other reference has
        # degenerate geometry there. All references still have full support.
        source.obsm['spatial'][:60] = target.obsm['spatial'][:60] if name.startswith('a_') else 0
        source.obsm['spatial'][60:] = target.obsm['spatial'][60:] if name.startswith('b_') else 20
        references.append(source)
    return references, target


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('count', [1, 2, 3, 5, None, 'default'])
def test_local_selection_preserves_support_and_precision(monkeypatch, count, dtype):
    references, target = geometry_references()
    calls = fixed_table_spy(monkeypatch)
    config = FEAST.SimulationConfig(transport_dtype=dtype, sinkhorn_method='sinkhorn_log',
                                   max_transport_pairs=1800)
    options = {} if count == 'default' else {'n_references': count}
    result = FEAST.simulate_local_references(references[::-1], target,
        label_key='class', config=config, random_seed=7, **options)
    metadata = assert_shared_group_selection(result, calls)
    expected_count = 5 if count in (None, 'default') else count
    assert metadata['n_references'] == (5 if count == 'default' else count)
    assert metadata['merging']['merges'] == []
    assert result.shape == target.shape
    np.testing.assert_array_equal(result.var_names, sorted(target.var_names))
    np.testing.assert_array_equal(result.obs['spot_id'], target.obs['spot_id'])
    np.testing.assert_array_equal(result.obs['class'], target.obs['class'])
    np.testing.assert_array_equal(result.obsm['spatial'], target.obsm['spatial'])
    for group, weights in metadata['group_weights'].items():
        assert len(weights) == expected_count
        assert sum(weights.values()) == pytest.approx(1.0)
        if expected_count <= 2:
            expected_names = [f'{group}_first', f'{group}_second'][:expected_count]
            assert set(weights) == set(expected_names)
        if expected_count == 2:
            assert list(weights.values()) == pytest.approx([.5, .5])
        if expected_count == 5:
            assert set(weights) == {r.uns['reference_name'] for r in references}
            assert weights[f'{group}_first'] > weights['neutral'] > 0
        table = result.uns['de_novo']['transport_diagnostics'][group]
        assert set(table['transport_dtype']) == {dtype}
        assert set(str(v).lower() for v in table['transport_converged']) == {'true'}
        assert np.asarray(table['transport_final_error'], dtype=float).max() < config.sinkhorn_tol
        np.testing.assert_array_equal(np.asarray(table['source_spots'], dtype=int), np.full(expected_count, 60))
        np.testing.assert_array_equal(np.asarray(table['target_spots'], dtype=int), np.full(expected_count, 60))
        np.testing.assert_array_equal(np.asarray(table['transport_blocks'], dtype=int), np.full(expected_count, 2))
    if expected_count == 2:
        assert len(metadata['participating_references']) == 4


def test_all_reference_mode_uses_each_regions_eligible_pool(monkeypatch):
    references = [ref('only_a', 0, ['a'] * 60), ref('only_b', 1, ['b'] * 60)]
    target = ref('target', 2, ['a'] * 60 + ['b'] * 60)
    calls = fixed_table_spy(monkeypatch)
    result = FEAST.simulate_local_references(references, target, label_key='class',
        n_references=None, config=FEAST.SimulationConfig(sinkhorn_method='sinkhorn_log'))
    metadata = assert_shared_group_selection(result, calls)
    assert metadata['group_weights'] == {'a': {'only_a': 1.0}, 'b': {'only_b': 1.0}}
    assert result.shape == target.shape
