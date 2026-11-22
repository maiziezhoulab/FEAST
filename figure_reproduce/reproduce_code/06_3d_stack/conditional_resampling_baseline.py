"""Same-class whole-spot resampling using Study 06's five reference weights."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse

@lru_cache(maxsize=12)
def reference_counts(path: Path, label_key: str, counts_layer: str):
    data = ad.read_h5ad(path)
    matrix = data.layers[counts_layer]
    return (matrix, data.obs[label_key].astype(str).to_numpy(), tuple(map(str, data.var_names)))

def prepare_sampling(row, density, data_dir, prefix, label_key, counts_layer, target_labels, genes):
    """Restrict donors by original class; renormalize the declared z weights."""
    permitted = set(map(int, density['reference_pool']))
    primary = list(map(int, row['primary_reference_slices']))
    donors = {str(item['label']): int(item['donor_slice']) for item in row['donor_support']}
    requested = set(primary) | set(donors.values())
    if not requested <= permitted or int(row['target_slice']) in requested:
        raise ValueError('resampling references must belong to the retained pool')
    references = {}
    for slice_id in sorted(requested):
        name = f'{prefix}.{slice_id:03d}'
        matrix, labels, observed_genes = reference_counts(Path(data_dir) / f'{name}.h5ad', label_key, counts_layer)
        if observed_genes != tuple(genes):
            raise ValueError(f'{name} gene order differs from the target')
        references[slice_id] = (matrix, labels)
    groups, support = ([], [])
    for label in sorted(set(target_labels)):
        target_indices = np.flatnonzero(target_labels == label)
        sources, probabilities, source_ids = ([], [], [])
        for slice_id in primary:
            matrix, labels = references[slice_id]
            indices = np.flatnonzero(labels == label)
            if indices.size:
                sources.append((matrix, indices))
                probabilities.append(float(row['reference_weights'][f'{prefix}.{slice_id:03d}']))
                source_ids.append(slice_id)
        if not sources:
            if label not in donors:
                raise ValueError(f"target {row['target_slice']} lacks a declared donor for {label}")
            slice_id = donors[label]
            matrix, labels = references[slice_id]
            indices = np.flatnonzero(labels == label)
            if not indices.size:
                raise ValueError(f'declared donor {slice_id} lacks class {label}')
            sources, probabilities, source_ids = ([(matrix, indices)], [1.0], [slice_id])
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities /= probabilities.sum()
        groups.append((target_indices, sources, probabilities))
        support.append({'class': label, 'n_target_spots': int(target_indices.size), 'reference_slices': source_ids, 'weights': probabilities.tolist(), 'uses_supporting_donor': not any((x in primary for x in source_ids))})
    return (groups, support)

def resample_spots(groups, shape, rng):
    """Copy complete donor vectors with replacement, preserving target row order."""
    result = np.empty(shape, dtype=np.float64)
    for target_indices, sources, probabilities in groups:
        choices = np.zeros(target_indices.size, dtype=int) if len(sources) == 1 else rng.choice(len(sources), size=target_indices.size, p=probabilities)
        for source_index, (matrix, donor_indices) in enumerate(sources):
            destination = target_indices[choices == source_index]
            if not destination.size:
                continue
            selected = rng.choice(donor_indices, size=destination.size, replace=True)
            values = matrix[selected]
            result[destination] = values.toarray() if sparse.issparse(values) else values
    return result
