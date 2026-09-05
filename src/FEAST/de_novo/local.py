"""Local multi-reference parameter fusion using the existing conditional OT field.

Targets contribute geometry and labels only. Persistent caches contain generated
reference statistics, never target parameters or target expression.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
import time

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..FEAST_core.parameter_cloud import (
    GeneParameterSimulator, apply_batch_deformation,
    convert_params_for_new_simulator,
)
from ..FEAST_core.theta_transform import stats_to_theta, theta_to_stats
from .conditional import (
    ReferenceFitConfig, SimulationConfig, SimulationReference,
    _blueprint_labels, _boundary_scores, _coordinates, _load_target_blueprint,
    _apply_coordinate_scale, _geometry_distance, _weights_from_geometry_distances,
    _reference_weights_for_label, fit_reference, simulate_from_reference,
)
from .quantile_field import weighted_stats_log_space
from .transport import normalize_coordinates


def local_seed(seed: int, stream: str, identity: str = "") -> int:
    """Stable SeedSequence entropy from literal identities, independent of order."""
    entropy = [int(seed), *str(stream + '\0' + identity).encode('utf-8')]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0] % (2**31))


def select_z_references(reference_z, target_z, *, n_references: int | None = 5):
    """Bracket in actual z; fill by distance and reference ID. Bandwidth uses pool."""
    names = list(reference_z)
    if n_references is not None and (isinstance(n_references, bool) or
            not isinstance(n_references, int) or not 2 <= n_references <= len(names)):
        raise ValueError("n_references must be None or between 2 and the primary pool size")
    ordered = sorted(names, key=lambda name: (reference_z[name], name))
    z = np.array([reference_z[name] for name in ordered], dtype=float)
    if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("reference actual-z values must be finite and unique")
    lower = [name for name in ordered if reference_z[name] < target_z]
    upper = [name for name in ordered if reference_z[name] > target_z]
    if not lower or not upper or target_z in z:
        raise ValueError("target must lie strictly between retained references")
    primary = [lower[-1], upper[0]]
    closest = sorted(names, key=lambda name: (abs(reference_z[name] - target_z), name))
    primary += [name for name in closest if name not in primary]
    primary = primary[:n_references]
    bandwidth = float(np.median(np.diff(z)))
    distances = np.array([abs(reference_z[name] - target_z) for name in primary])
    weights = np.exp(-(distances - distances.min()) / bandwidth)
    weights /= weights.sum()
    return primary, dict(zip(primary, weights.tolist())), bandwidth


def merge_local_labels(references, target_labels, *, label_key, min_positions=50,
                       reference_selection=None):
    """Merge the smallest positive slice/group support using within-slice 6-NN."""
    labels = sorted(set(map(str, target_labels)) | {
        str(v) for ref in references for v in ref.obs[label_key]})
    mapping = {label: label for label in labels}
    original = [ref.obs[label_key].astype(str).to_numpy() for ref in references]
    coords = [_coordinates(ref)[:, :2] for ref in references]
    graphs = []
    for xy in coords:
        k = min(7, len(xy))
        neighbors = NearestNeighbors(n_neighbors=k).fit(xy).kneighbors(xy, return_distance=False)
        source = np.repeat(np.arange(len(xy)), k)
        dest = neighbors.reshape(-1)
        keep = source != dest
        graphs.append((source[keep], dest[keep]))
    history = []
    while True:
        mapped = [np.array([mapping[v] for v in values]) for values in original]
        active = {}
        for group in sorted(set(mapping.values())):
            allowed = None if reference_selection is None else set().union(*(
                set(reference_selection.get(label, ()))
                for label, mapped_group in mapping.items() if mapped_group == group))
            active[group] = [i for i, ref in enumerate(references)
                             if allowed is None or ref.uns['reference_name'] in allowed]
        support = {group: [int(np.sum(values == group)) if i in active[group] else 0
                           for i, values in enumerate(mapped)] for group in active}
        weak = [(min(n for n in counts if n > 0), group)
                for group, counts in support.items() if any(0 < n < min_positions for n in counts)]
        if not weak:
            break
        _, group = min(weak)
        contacts = {}
        for i in active[group]:
            values, (source, dest) = mapped[i], graphs[i]
            others = values[dest[(values[source] == group) & (values[dest] != group)]]
            for other in others:
                contacts[other] = contacts.get(other, 0) + 1
        if contacts:
            into = min(contacts, key=lambda other: (-contacts[other], other))
        else:
            distances = {}
            for i in active[group]:
                values, xy = mapped[i], coords[i]
                if not np.any(values == group):
                    continue
                for other in sorted(set(values) - {group}):
                    distance = float(NearestNeighbors(n_neighbors=1).fit(xy[values == other])
                                     .kneighbors(xy[values == group])[0].min())
                    distances[other] = min(distances.get(other, float('inf')), distance)
            if not distances:
                raise ValueError(f"local group {group!r} has support {support[group]} and no within-slice merge partner")
            into = min(distances, key=lambda other: (distances[other], other))
        history.append({'group': group, 'into': into, 'support_before': support[group],
                        'cross_label_contacts': int(contacts.get(into, 0))})
        mapping = {label: into if value == group else value for label, value in mapping.items()}
    members = {group: sorted(label for label, value in mapping.items() if value == group)
               for group in sorted(set(mapping.values()))}
    return mapping, {'members': members, 'support': support, 'merges': history,
                     'reference_affected_positions': [int(sum(mapping[v] != v for v in values)) for values in original],
                     'target_affected_positions': int(sum(mapping[str(v)] != str(v) for v in target_labels))}


def _select_geometry_groups(references, target_coords, target_labels, mapping,
                            label_key, config, n_references):
    """Rank eligible references using metadata only and the existing OT geometry."""
    target_groups = np.array([mapping[label] for label in target_labels])
    target_xy = _apply_coordinate_scale(target_coords, config.coordinate_scale)
    target_boundary = _boundary_scores(target_xy, target_groups, 6)
    source_geometry = []
    for ref in references:
        labels = np.array([mapping[str(label)] for label in ref.obs[label_key]])
        xy = _apply_coordinate_scale(_coordinates(ref), config.coordinate_scale)
        source_geometry.append((ref.uns['reference_name'], labels, xy,
                                _boundary_scores(xy, labels, 6)))
    selection = {}
    for group in sorted(set(target_groups)):
        target_mask = target_groups == group
        normalized_target = normalize_coordinates(target_xy[target_mask])
        names, distances = [], []
        for name, labels, xy, boundary in source_geometry:
            mask = labels == group
            if not np.any(mask):
                continue
            names.append(name)
            distances.append(_geometry_distance(normalize_coordinates(xy[mask]),
                normalized_target, boundary[mask], target_boundary[target_mask]))
        if not names:
            raise ValueError(f'no permitted reference supports target group {group!r}')
        weights = _weights_from_geometry_distances(names, distances, config.reference_weight_eta)
        chosen = sorted(names, key=lambda name: (-weights[name], name))[:n_references]
        total = sum(weights[name] for name in chosen)
        selection[group] = {name: weights[name] / total for name in chosen}
    return selection


def _prepare_finite_geometry(references, blueprint, target_labels, label_key,
                             config, n_references, min_positions):
    """Resolve selection/merging together; only selected group support can merge.

    A merge changes group geometry and can change the highest-weight references.
    Recompute the mapping from original labels on each selected set, so dropped
    references leave no residual merges. If this procedure cycles without reaching
    a consistent mapping, it reports that failure rather than retaining an excluded donor's
    influence or silently changing the selection method.
    """
    labels = sorted(set(target_labels) | {
        str(label) for ref in references for label in ref.obs[label_key]})
    mapping = {label: label for label in labels}
    seen = set()
    while True:
        state = tuple(sorted(mapping.items()))
        if state in seen:
            raise ValueError('finite geometry reference selection and local merging cycle; '
                             'no consistent selected-reference mapping was reached')
        seen.add(state)
        selection = _select_geometry_groups(references, blueprint.coordinates,
            target_labels, mapping, label_key, config, n_references)
        primary = sorted({name for weights in selection.values() for name in weights})
        participating = [ref for ref in references if ref.uns['reference_name'] in primary]
        selection_by_label = {label: set(selection.get(mapping[label], {})) for label in labels}
        local_mapping, metadata = merge_local_labels(participating, target_labels,
            label_key=label_key, min_positions=min_positions, reference_selection=selection_by_label)
        updated = {label: local_mapping.get(label, label) for label in labels}
        if updated == mapping:
            metadata['support_scope'] = 'selected_reference_groups'
            return primary, selection, mapping, metadata
        mapping = updated


def _generated_table(reference, members, label_key, genes, seed, cache_path):
    name = str(reference.uns['reference_name'])
    key = json.dumps([name, members, list(genes), int(seed)], ensure_ascii=True)
    connection = None
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(cache_path, timeout=600)
        connection.execute('CREATE TABLE IF NOT EXISTS reference_stats (identity TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        record = connection.execute('SELECT payload FROM reference_stats WHERE identity=?', (key,)).fetchone()
        if record:
            connection.close()
            payload = json.loads(record[0])
            return pd.DataFrame(payload['stats'], index=genes), payload['diagnostics'], True
    mask = reference.obs[label_key].astype(str).isin(members).to_numpy()
    subset = reference[mask, genes].copy()
    if 'counts' in subset.layers:
        subset.X = subset.layers['counts'].copy()
    simulator = GeneParameterSimulator(ppf_method='interp', beta_n_jobs=1)
    simulator.hybrid_alpha = 0.2
    # Some fitting components consume NumPy's global stream. Isolate and restore it.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        simulator.fit(subset, visualize_fits=False)
        table, diagnostics = simulator.build_gene_parameter_table(
            simulation_mode='generative', assignment_method='hybrid',
            assignment_solver='scipy', assignment_blocks=False,
            overgeneration_factor=3.0, random_seed=seed, verbose=False)
    finally:
        np.random.set_state(state)
    table = table.set_index('gene_id').loc[genes, ['mean', 'variance', 'zero_prop']]
    if connection is not None:
        payload = json.dumps({'stats': table.to_dict(orient='list'), 'diagnostics': diagnostics})
        connection.execute('INSERT OR IGNORE INTO reference_stats VALUES (?, ?)', (key, payload))
        connection.commit()
        connection.close()
    return table, diagnostics, False


def simulate_local_references(
    reference_slices, target_blueprint, *, label_key: str,
    n_references: int | None = 5, reference_z=None, target_z=None,
    config=None, random_seed: int = 0, parameter_seed: int = 0,
    batch_seed: int | None = None, count_seed: int | None = None,
    cache_path=None, min_positions: int = 50, batch_sd: float = 0.005,
    batch_deformation=None, fit_cache=None,
):
    """Generate one target with local labels, fixed reference tables and core counts.

    With reference_z, selection is bracketed distance weighting (Study 06).
    Otherwise use the existing group geometry weights (Study 07). Donor slices
    remain distinct, and their OT/count contribution is limited to missing groups.
    """
    started = time.perf_counter()
    cfg = config or SimulationConfig()
    references = sorted(reference_slices, key=lambda ref: str(ref.uns['reference_name']))
    names = [str(ref.uns['reference_name']) for ref in references]
    if len(set(names)) != len(names):
        raise ValueError('reference names must be unique')
    by_name = dict(zip(names, references))
    blueprint = _load_target_blueprint(target_blueprint, label_key).active_subset()
    original_labels = _blueprint_labels(blueprint).astype(str)
    genes = sorted(map(str, references[0].var_names))
    if any(set(map(str, ref.var_names)) != set(genes) for ref in references):
        raise ValueError('all references must contain the declared gene panel')
    bandwidth = None
    finite_geometry = reference_z is None and n_references is not None
    if reference_z is not None:
        primary, base_weights, bandwidth = select_z_references(reference_z, target_z, n_references=n_references)
    else:
        if n_references is not None and (isinstance(n_references, bool) or
                not isinstance(n_references, int) or not 1 <= n_references <= len(references)):
            raise ValueError('n_references must be None or within the primary pool size')
        primary, base_weights = names, None
    if finite_geometry:
        primary, selected_group_weights, mapping, merge_metadata = _prepare_finite_geometry(
            references, blueprint, original_labels, label_key, cfg, n_references, min_positions)
        participating = list(primary)
        donor_candidates = {}
    else:
        # Select original-label donors before merging. Donors contribute only where
        # primary support is absent; merging can make such an exception unnecessary.
        participating = list(primary)
        donor_candidates = {}
        for label in sorted(set(original_labels)):
            if any(label in set(by_name[name].obs[label_key].astype(str)) for name in primary):
                continue
            supporting = [name for name in names if label in set(by_name[name].obs[label_key].astype(str))]
            if not supporting:
                raise ValueError(f'no permitted reference supports target label {label!r}')
            donor = min(supporting, key=lambda name: (abs(reference_z[name] - target_z), name))
            donor_candidates[label] = donor
            if donor not in participating:
                participating.append(donor)
        mapping, merge_metadata = merge_local_labels([by_name[name] for name in participating], original_labels,
                                                      label_key=label_key, min_positions=min_positions)
    modeling_labels = np.array([mapping[label] for label in original_labels])
    model_refs = []
    spatial_fit_seconds = 0.0
    for name in participating:
        source = by_name[name]
        mapping_key = tuple(sorted(mapping.items()))
        key = (mapping_key, tuple(genes))
        cached = None if fit_cache is None else fit_cache.get(name)
        fitted = cached[1] if cached is not None and cached[0] == key else None
        if fitted is None:
            before = time.perf_counter()
            ref = source[:, genes].copy()
            ref.obs['_feast_model_group'] = [mapping[str(label)] for label in ref.obs[label_key]]
            fitted = fit_reference(ref, '_feast_model_group', ReferenceFitConfig(
                min_gene_spots=0, min_gene_mean=0, max_gene_zero_prop=1,
                coordinate_scale=cfg.coordinate_scale))
            spatial_fit_seconds += time.perf_counter() - before
            if fit_cache is not None:
                fit_cache[name] = (key, fitted)
        model_refs.extend(fitted.references)
    model = SimulationReference(genes, '_feast_model_group', model_refs,
                                int(blueprint.coordinates.shape[1]), fitted.fit_config)
    local_blueprint = replace(blueprint, domain_map=modeling_labels, obs=blueprint.obs.copy())
    local_blueprint.obs['_feast_model_group'] = modeling_labels
    if finite_geometry:
        group_weights, donors = selected_group_weights, {}
    else:
        target_boundary = _boundary_scores(blueprint.coordinates, modeling_labels, 6)
        group_weights, donors = {}, {}
        for group in sorted(set(modeling_labels)):
            eligible = [ref for ref in model_refs if group in ref.labels and ref.reference_name in primary]
            if not eligible:
                candidates = [ref for ref in model_refs if group in ref.labels]
                donor = min(candidates, key=lambda ref: (abs(reference_z[ref.reference_name] - target_z), ref.reference_name))
                eligible = [donor]
                donors[group] = donor.reference_name
            if base_weights is None:
                mask = modeling_labels == group
                weights = _reference_weights_for_label(eligible, group,
                    normalize_coordinates(blueprint.coordinates[mask]), target_boundary[mask], cfg.reference_weight_eta)
                chosen = sorted(weights, key=lambda name: (-weights[name], name))[:n_references]
                weights = {name: weights[name] for name in chosen}
            else:
                weights = {ref.reference_name: base_weights.get(ref.reference_name, 1.0) for ref in eligible}
            total = sum(weights.values())
            group_weights[group] = {name: value / total for name, value in weights.items()}
    preprocessing_seconds = time.perf_counter() - started - spatial_fit_seconds
    rng = np.random.default_rng(local_seed(random_seed if batch_seed is None else batch_seed, 'batch'))
    D, b = ((1 + rng.normal(0, batch_sd, 3), rng.normal(0, batch_sd, 3))
            if batch_deformation is None else batch_deformation)
    params, diagnostics, tables = {}, {}, {}
    fit_seconds = conversion_seconds = 0.0
    for group, weights in group_weights.items():
        frames = []
        reference_diagnostics = {}
        for name in weights:
            before = time.perf_counter()
            members = merge_metadata['members'][group]
            seed = local_seed(parameter_seed, 'parameters', json.dumps([name, members]))
            table_key = ('reference_table', name, tuple(members), tuple(genes), seed)
            cached_table = None if fit_cache is None else fit_cache.get(table_key)
            if cached_table is None:
                table, diag, hit = _generated_table(by_name[name], members, label_key, genes, seed, cache_path)
                if fit_cache is not None:
                    fit_cache[table_key] = (table, diag)
            else:
                table, diag = cached_table
                hit = True
            fit_seconds += time.perf_counter() - before
            frames.append(table)
            reference_diagnostics[name] = {'cache_hit': hit, 'fit': diag}
        fused = weighted_stats_log_space(frames, list(weights.values()), genes)
        deformed = theta_to_stats(apply_batch_deformation(stats_to_theta(fused), D, b, alpha=1.0))
        deformed.index = genes
        before = time.perf_counter()
        params[group] = convert_params_for_new_simulator(deformed,
            n_spots=int(np.sum(modeling_labels == group)), boundary_multiplier=cfg.boundary_multiplier,
            random_seed=local_seed(random_seed, 'conversion', group), n_jobs=1)
        conversion_seconds += time.perf_counter() - before
        params[group]['target_stats'] = deformed
        params[group]['parameter_diagnostics'] = {'requested_config': {
            'apply_to_variance': True, 'apply_to_zero_prop': True},
            'preservation_policy': 'final_target_stats'}
        diagnostics[group] = reference_diagnostics
        tables[group] = {'fused': fused.to_dict(orient='list'), 'deformed': deformed.to_dict(orient='list')}
    generated = simulate_from_reference(model, local_blueprint, config=cfg,
        random_seed=local_seed(random_seed, 'spatial'),
        count_seed=local_seed(random_seed if count_seed is None else count_seed, 'counts'),
        count_model_params=params, group_reference_weights=group_weights)
    generated.obs_names = blueprint.obs.index.astype(str)
    generated.obs[label_key] = original_labels
    generated.obs['domain'] = original_labels
    generated.obs['_feast_model_group'] = modeling_labels
    generated.uns['local_generation_json'] = json.dumps({
        'n_references': n_references, 'primary_references': primary,
        'participating_references': participating, 'original_label_donor_candidates': donor_candidates,
        'weighting': 'actual_z_exponential' if reference_z is not None else 'region_geometry',
        'bandwidth': bandwidth, 'group_weights': group_weights, 'donors': donors,
        'merging': merge_metadata, 'min_positions': min_positions,
        'batch': {'D': np.asarray(D).tolist(), 'b': np.asarray(b).tolist(), 'alpha': 1.0, 'sd': batch_sd},
        'parameter_seed': parameter_seed, 'random_seed': random_seed,
        'batch_seed': batch_seed, 'count_seed': count_seed,
        'genes': genes, 'reference_fits': diagnostics, 'target_statistics': tables,
        'preservation_policy': 'final_target_stats',
        'timings': {'preprocessing': preprocessing_seconds, 'spatial_preparation': spatial_fit_seconds,
                    'parameter_fitting': fit_seconds, 'conversion': conversion_seconds,
                    'total': time.perf_counter() - started}}, sort_keys=True)
    return generated


def calibrate_local_references(reference_slices, *, label_key, n_references=5,
                               reference_z=None, config=None, random_seed=0,
                               cache_path=None, min_positions=50, batch_sd=0.005):
    """Three deterministic whole-reference holdouts; fit all genes, score 20."""
    from .conditional import _counts_matrix, _per_gene_moran
    references = sorted(reference_slices, key=lambda ref: (
        reference_z[str(ref.uns['reference_name'])] if reference_z is not None else str(ref.uns['reference_name'])))
    eligible = references[1:-1] if reference_z is not None else references
    if len(eligible) < 3:
        raise ValueError('three eligible reference holdouts are required')
    indices = np.rint(np.linspace(0, len(eligible) - 1, 3)).astype(int)
    folds = []
    for fold_index in indices:
        held = eligible[fold_index]
        name = str(held.uns['reference_name'])
        training = [ref for ref in references if str(ref.uns['reference_name']) != name]
        genes = list(map(str, training[0].var_names))
        sums = np.zeros(len(genes)); squares = sums.copy(); population = 0
        for ref in training:
            matrix = _counts_matrix(ref[:, genes])
            sums += matrix.sum(axis=0)
            squares += np.square(matrix.astype(float)).sum(axis=0)
            population += ref.n_obs
        variance = squares / population - (sums / population) ** 2
        selected = sorted(range(len(genes)), key=lambda i: (-variance[i], genes[i]))[:20]
        eval_genes = [genes[i] for i in selected]
        target = _load_target_blueprint(held, label_key)
        xy = target.coordinates[:, :2]
        neighbors = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        truth = _per_gene_moran(_counts_matrix(held[:, eval_genes]), neighbors)
        fold_z = None if reference_z is None else {key: value for key, value in reference_z.items() if key != name}
        scores = []
        fold_seed = local_seed(random_seed, 'calibration', name)
        fit_cache = {}
        for ar in np.round(np.arange(0, 0.50001, 0.05), 2):
            result = simulate_local_references(training, target, label_key=label_key,
                n_references=n_references, reference_z=fold_z,
                target_z=None if reference_z is None else reference_z[name],
                config=replace(config or SimulationConfig(), assignment_randomness=float(ar)),
                random_seed=fold_seed, parameter_seed=random_seed, batch_seed=fold_seed,
                cache_path=cache_path, min_positions=min_positions, batch_sd=batch_sd,
                fit_cache=fit_cache)
            observed = _per_gene_moran(_counts_matrix(result[:, eval_genes]), neighbors)
            valid = np.isfinite(truth) & np.isfinite(observed)
            if valid.sum() < 2 or np.std(truth[valid]) == 0 or np.std(observed[valid]) == 0:
                raise ValueError(f'{name}: undefined Moran correlation at AR={ar}')
            scores.append({'ar': float(ar), 'moran_correlation': float(np.corrcoef(truth[valid], observed[valid])[0, 1])})
            print(f'calibration {name} AR={ar:.2f} correlation={scores[-1]["moran_correlation"]:.6f}', flush=True)
        best = min(scores, key=lambda row: (-row['moran_correlation'], row['ar']))['ar']
        folds.append({'holdout': name, 'training_references': [str(ref.uns['reference_name']) for ref in training],
                      'evaluation_genes': eval_genes, 'scores': scores, 'assignment_randomness': best,
                      'generation': json.loads(result.uns['local_generation_json'])})
    return {'mode': 'reference_only_representative_median', 'reference_only': True,
            'n_references': n_references, 'min_positions': min_positions, 'batch_sd': batch_sd,
            'simulation_config': asdict(replace(config or SimulationConfig(), assignment_randomness=0.0)),
            'weighting': 'actual_z_exponential' if reference_z is not None else 'region_geometry',
            'count_generation': 'fixed_generative_reference_tables_full_core_conversion',
            'representative_estimates': folds,
            'observed_assignment_randomness': float(np.median([fold['assignment_randomness'] for fold in folds]))}
