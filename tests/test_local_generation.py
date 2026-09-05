import json
from dataclasses import replace
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from FEAST.de_novo import local
from FEAST.de_novo.conditional import fit_reference, simulate_from_reference, ReferenceFitConfig, SimulationConfig
from FEAST.de_novo.core import SliceBlueprint


def ref(name, z, labels=None):
    rng = np.random.default_rng(42 + z)
    labels = ['a'] * 60 if labels is None else labels
    counts = rng.poisson(np.arange(1, 9), (len(labels), 8))
    data = ad.AnnData(counts, obs=pd.DataFrame({'class': labels}, index=[f'{name}-{i}' for i in range(len(labels))]),
                     var=pd.DataFrame(index=[f'g{i}' for i in range(8)]))
    data.layers['counts'] = counts
    data.obsm['spatial'] = rng.normal(size=(len(labels), 2))
    data.uns['reference_name'] = name
    return data


def test_selection():
    z = {1: 0., 2: 1., 3: 2., 4: 4., 5: 6., 6: 8.}
    primary, weights, h = local.select_z_references(z, 3., n_references=5)
    assert primary == [3, 4, 2, 1, 5]
    assert h == 2 and sum(weights.values()) == pytest.approx(1)
    assert local.select_z_references(z, 3., n_references=2)[2] == h
    assert len(local.select_z_references(z, 3., n_references=None)[0]) == 6
    for count in [1, 7, 2.5]:
        with pytest.raises(ValueError):
            local.select_z_references(z, 3., n_references=count)


def test_merging_and_failure():
    data = ref('r', 0, ['a'] * 10 + ['b'] * 60)
    mapping, metadata = local.merge_local_labels([data], ['a', 'b'], label_key='class')
    assert mapping == {'a': 'b', 'b': 'b'}
    assert metadata['support']['b'] == [70]
    with pytest.raises(ValueError, match='no within-slice merge partner'):
        local.merge_local_labels([ref('tiny', 0, ['a'] * 4)], ['a'], label_key='class')


def test_core_path_donors_streams_and_roundtrip(monkeypatch, tmp_path):
    refs = [ref('lower', 0), ref('upper', 2), ref('donor', 8, ['b'] * 60)]
    target = ref('target', 1, ['a'] * 55 + ['b'] * 5)
    blueprint = SliceBlueprint(coordinates=target.obsm['spatial'], domain_map=target.obs['class'].to_numpy(), obs=target.obs.copy())
    calls = []
    real_convert = local.convert_params_for_new_simulator
    def convert(table, **kwargs):
        calls.append((table.copy(), kwargs['n_spots']))
        return real_convert(table, **kwargs)
    monkeypatch.setattr(local, 'convert_params_for_new_simulator', convert)
    def table(reference, members, label_key, genes, seed, cache_path):
        return pd.DataFrame({'mean': np.arange(1, 9), 'variance': np.arange(1, 9) * 2., 'zero_prop': .1}, index=genes), {}, False
    monkeypatch.setattr(local, '_generated_table', table)
    cfg = SimulationConfig(transport_backend='numpy', sinkhorn_iter=1000, sinkhorn_tol=1e-5, store_quantiles=True)
    kw = dict(label_key='class', n_references=2, reference_z={'lower': 0., 'upper': 2., 'donor': 8.}, target_z=1.,
              config=cfg, random_seed=9, batch_deformation=(np.ones(3), np.zeros(3)))
    output = local.simulate_local_references(refs, blueprint, **kw)
    assert [n for _, n in calls] == [55, 5]
    metadata = json.loads(output.uns['local_generation_json'])
    assert metadata['donors'] == {'b': 'donor'}
    assert set(metadata['group_weights']['a']) == {'lower', 'upper'}
    assert list(output.obs['class']) == list(target.obs['class'])
    assert np.array_equal(output.obsm['spatial'], blueprint.coordinates)
    assert output.uns['de_novo']['marginal_model'] == 'core'
    assert np.issubdtype(output.X.dtype, np.integer) and (output.X >= 0).all()
    other = local.simulate_local_references(refs, blueprint, count_seed=600, **kw)
    np.testing.assert_array_equal(other.layers['feast_quantiles'], output.layers['feast_quantiles'])
    assert not np.array_equal(other.X, output.X)
    again = local.simulate_local_references(list(reversed(refs)), blueprint, **kw)
    np.testing.assert_array_equal(again.X, output.X)
    path = tmp_path / 'counts.h5ad'
    output.write_h5ad(path)
    restored = ad.read_h5ad(path)
    np.testing.assert_array_equal(restored.X, output.X)
    assert restored.uns['local_generation_json'] == output.uns['local_generation_json']


def test_ot_unchanged_with_precomputed_params():
    data = ref('reference', 0)
    model = fit_reference(data, 'class', ReferenceFitConfig(min_gene_spots=0))
    cfg = SimulationConfig(store_quantiles=True, sinkhorn_iter=1000, sinkhorn_tol=1e-5)
    empirical = simulate_from_reference(model, data, config=cfg, random_seed=7)
    stats = model.references[0].labels['a'].stats
    params = local.convert_params_for_new_simulator(stats, n_spots=60, random_seed=1)
    params['target_stats'] = stats
    params['parameter_diagnostics'] = {'requested_config': {'apply_to_variance': True, 'apply_to_zero_prop': True}}
    core = simulate_from_reference(model, data, config=cfg, random_seed=7, count_model_params={'a': params})
    np.testing.assert_array_equal(empirical.layers['feast_quantiles'], core.layers['feast_quantiles'])
    assert empirical.uns['de_novo']['transport_diagnostics'] == core.uns['de_novo']['transport_diagnostics']


def test_fixed_reference_table_cache(monkeypatch, tmp_path):
    fits = []
    class Simulator:
        def __init__(self, **kwargs):
            assert kwargs['ppf_method'] == 'interp'
        def fit(self, data, **kwargs):
            self.genes = list(data.var_names)
            fits.append(list(data.obs_names))
        def build_gene_parameter_table(self, **kwargs):
            assert kwargs['simulation_mode'] == 'generative'
            assert kwargs['assignment_method'] == 'hybrid'
            assert kwargs['assignment_solver'] == 'scipy'
            assert kwargs['assignment_blocks'] is False
            assert kwargs['overgeneration_factor'] == 3.0
            return pd.DataFrame({'gene_id': self.genes, 'mean': 2., 'variance': 3., 'zero_prop': .2}), {}
    monkeypatch.setattr(local, 'GeneParameterSimulator', Simulator)
    data = ref('r', 0, ['a'] * 60 + ['b'] * 60)
    genes = list(data.var_names)
    cache = tmp_path / 'parameters.sqlite'
    first, _, hit = local._generated_table(data, ['a'], 'class', genes, 22, cache)
    assert not hit
    second, _, hit = local._generated_table(data, ['a'], 'class', genes, 22, cache)
    assert hit and len(fits) == 1
    pd.testing.assert_frame_equal(first, second, check_names=False)
    local._generated_table(data, ['a','b'], 'class', genes, 22, cache)
    assert len(fits) == 2 and len(fits[1]) == 120


def test_seed_streams_have_copula_compatible_range():
    for i in range(100):
        seeds = [local.local_seed(i, stream, 'reference') for stream in ['parameters', 'spatial', 'batch', 'counts']]
        assert all(0 <= seed < 2**31 for seed in seeds)
        assert len(set(seeds)) == 4


@pytest.mark.parametrize('gene_id_column', [False, True])
def test_precomputed_statistics_alignment_shared_by_decoder_and_metadata(monkeypatch, gene_id_column):
    from FEAST.de_novo import conditional
    data = ref('reference', 0)
    model = fit_reference(data, 'class', ReferenceFitConfig(min_gene_spots=0))
    stats = pd.DataFrame({'mean': np.arange(1., 9.), 'variance': np.arange(1., 9.) * 2,
                          'zero_prop': .1}, index=model.gene_names)
    params = local.convert_params_for_new_simulator(stats, n_spots=60, random_seed=5)
    params['parameter_diagnostics'] = {'requested_config': {'apply_to_variance': True, 'apply_to_zero_prop': True}}
    params['target_stats'] = stats
    cfg = SimulationConfig(sinkhorn_iter=1000, sinkhorn_tol=1e-5)
    ordered = simulate_from_reference(model, data, config=cfg, random_seed=7, count_model_params={'a': params})
    permuted = stats.iloc[::-1].copy()
    if gene_id_column:
        permuted = permuted.rename_axis('gene_id').reset_index()
    params['target_stats'] = permuted
    seen = {}
    decode = conditional.decode_counts_by_spatial_intensity
    aggregate = conditional._aggregate_label_parameter_clouds

    def capture_decode(intensity, model_params, **kwargs):
        seen['decoder'] = model_params['target_stats']
        return decode(intensity, model_params, **kwargs)

    def capture_metadata(label_clouds, *args):
        seen['metadata'] = label_clouds['a']
        return aggregate(label_clouds, *args)

    monkeypatch.setattr(conditional, 'decode_counts_by_spatial_intensity', capture_decode)
    monkeypatch.setattr(conditional, '_aggregate_label_parameter_clouds', capture_metadata)
    generated = simulate_from_reference(model, data, config=cfg, random_seed=7, count_model_params={'a': params})
    assert seen['decoder'] is seen['metadata']
    pd.testing.assert_frame_equal(seen['decoder'], stats, check_names=False)
    np.testing.assert_array_equal(generated.X, ordered.X)
    pd.testing.assert_frame_equal(generated.var, ordered.var)
    assert params['target_stats'] is permuted
    assert seen['decoder'] is not permuted


@pytest.mark.parametrize('problem', ['missing', 'duplicate'])
@pytest.mark.parametrize('gene_id_column', [False, True])
def test_invalid_precomputed_gene_ids_fail_before_ot(monkeypatch, problem, gene_id_column):
    from FEAST.de_novo import conditional
    data = ref('reference', 0)
    model = fit_reference(data, 'class', ReferenceFitConfig(min_gene_spots=0))
    stats = model.references[0].labels['a'].stats.copy()
    if problem == 'missing':
        stats = stats.iloc[1:]
    else:
        stats = pd.concat([stats, stats.iloc[:1]])
    if gene_id_column:
        stats = stats.rename_axis('gene_id').reset_index()
    params = {'genes': dict(enumerate(model.gene_names)), 'target_stats': stats}

    def unexpected_ot(**kwargs):
        pytest.fail('invalid gene identities must be rejected before OT')

    monkeypatch.setattr(conditional, 'transport_reference_field', unexpected_ot)
    with pytest.raises(ValueError, match=f'{problem} genes'):
        simulate_from_reference(model, data, count_model_params={'a': params})


def fixed_table_spy(monkeypatch):
    calls = []
    def table(reference, members, label_key, genes, seed, cache_path):
        calls.append((reference.uns['reference_name'], tuple(members)))
        return pd.DataFrame({'mean': np.arange(1., len(genes)+1),
                             'variance': np.arange(1., len(genes)+1)*2,
                             'zero_prop': .1}, index=genes), {}, False
    monkeypatch.setattr(local, '_generated_table', table)
    return calls


def assert_shared_group_selection(result, table_calls):
    metadata = json.loads(result.uns['local_generation_json'])
    expected_fits = []
    for group, weights in metadata['group_weights'].items():
        spatial_weights = result.uns['de_novo']['transport_weights'][group]
        assert set(spatial_weights) == set(weights)
        for name in weights:
            assert spatial_weights[name] == pytest.approx(weights[name], abs=1e-14)
            expected_fits.append((name, tuple(metadata['merging']['members'][group])))
        assert set(result.uns['de_novo']['transport_diagnostics'][group]['reference_name']) == set(weights)
    assert sorted(table_calls) == sorted(expected_fits)
    for original, modeled in zip(result.obs['class'], result.obs['_feast_model_group']):
        assert original in metadata['merging']['members'][modeled]
    return metadata


def test_unselected_reference_does_not_merge_labels(monkeypatch):
    good = ref('a_good', 0, ['a']*60 + ['b']*60)
    excluded = ref('z_excluded', 2, ['a'] + ['b']*100)
    calls = fixed_table_spy(monkeypatch)
    result = local.simulate_local_references([excluded, good], good, label_key='class',
        n_references=1, random_seed=5)
    metadata = assert_shared_group_selection(result, calls)
    assert metadata['primary_references'] == ['a_good']
    assert metadata['merging']['members'] == {'a': ['a'], 'b': ['b']}
    assert metadata['merging']['merges'] == []
    np.testing.assert_array_equal(result.obs['_feast_model_group'], good.obs['class'])


@pytest.mark.parametrize('n_references', [1, None])
def test_group_specific_selection_and_all_reference_behavior(monkeypatch, n_references):
    rng = np.random.default_rng(6)
    xy_a = rng.normal(size=(60, 2))
    xy_b = rng.normal(size=(60, 2)) + 20
    target = ref('target', 3, ['a']*60 + ['b']*60)
    target.obsm['spatial'] = np.vstack([xy_a, xy_b])
    left = ref('left', 1, ['a']*60 + ['b'])
    left.obsm['spatial'] = np.vstack([xy_a, [[100., 100.]]])
    right = ref('right', 2, ['a'] + ['b']*60)
    right.obsm['spatial'] = np.vstack([[[-100., -100.]], xy_b])
    calls = fixed_table_spy(monkeypatch)
    result = local.simulate_local_references([right, left], target, label_key='class',
        n_references=n_references, random_seed=6,
        config=SimulationConfig(sinkhorn_method='sinkhorn_log'))
    metadata = assert_shared_group_selection(result, calls)
    if n_references == 1:
        assert metadata['merging']['members'] == {'a': ['a'], 'b': ['b']}
        assert metadata['group_weights'] == {'a': {'left': 1.}, 'b': {'right': 1.}}
    else:
        assert metadata['merging']['members'] == {'b': ['a', 'b']}
        assert set(metadata['group_weights']['b']) == {'left', 'right'}


def test_selected_sparse_group_merges_and_geometry_ties_are_deterministic(monkeypatch):
    first = ref('a_first', 0, ['a']*5 + ['b']*60)
    tied = first.copy()
    tied.uns['reference_name'] = 'z_tied'
    calls = fixed_table_spy(monkeypatch)
    result = local.simulate_local_references([tied, first], first, label_key='class',
        n_references=1, random_seed=7)
    metadata = assert_shared_group_selection(result, calls)
    assert metadata['primary_references'] == ['a_first']
    assert metadata['merging']['members'] == {'b': ['a', 'b']}
    assert metadata['merging']['support']['b'] == [65]
    assert calls == [('a_first', ('a', 'b'))]


def test_geometry_reselection_rebuilds_merge_support_before_fitting(monkeypatch):
    rng = np.random.default_rng(47)
    xy_a = rng.normal(size=(5, 2))
    xy_b = rng.normal(size=(60, 2)) + 20
    target = ref('target', 3, ['a']*5 + ['b']*60)
    target.obsm['spatial'] = np.vstack([xy_a, xy_b])
    first = ref('first', 1, ['a']*5 + ['b']*60)
    first.obsm['spatial'] = np.vstack([xy_a, rng.normal(size=(60, 2))*5 + 20])
    second = ref('second', 2, ['a']*5 + ['b']*60)
    second.obsm['spatial'] = np.vstack([rng.normal(size=(5, 2)), xy_b])
    cfg = SimulationConfig(sinkhorn_method='sinkhorn_log')
    original_selection = local._select_geometry_groups([first, second], target.obsm['spatial'],
        target.obs['class'].to_numpy(), {'a': 'a', 'b': 'b'}, 'class', cfg, 1)
    assert original_selection == {'a': {'first': 1.}, 'b': {'second': 1.}}
    calls = fixed_table_spy(monkeypatch)
    result = local.simulate_local_references([first, second], target, label_key='class',
        n_references=1, random_seed=47, config=cfg)
    metadata = assert_shared_group_selection(result, calls)
    assert metadata['primary_references'] == ['second']
    assert metadata['merging']['members'] == {'b': ['a', 'b']}
    assert metadata['merging']['merges'][0]['support_before'] == [5]
    assert metadata['merging']['support']['b'] == [65]
    assert calls == [('second', ('a', 'b'))]
