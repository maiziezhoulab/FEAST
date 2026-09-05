from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd

from .conditional import (
    SimulationConfig,
)
from .core import SliceBlueprint, assign_generated_coordinates, load_blueprint


BlueprintInput = Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str, Path]


def simulate_stack(
    reference_slices: Sequence[ad.AnnData],
    reference_z_values: Sequence[float],
    target_z_values: Sequence[float],
    target_blueprints: Mapping[Any, BlueprintInput],
    *,
    label_key: str = "domain",
    n_references: int | None = 5,
    cache_path=None,
    config: Optional[SimulationConfig] = None,
    random_seed: int = 0,
) -> Dict[float, ad.AnnData]:
    gen_cfg = config or SimulationConfig()
    references = _coerce_reference_slices(reference_slices)
    reference_z = _coerce_numeric_sequence(reference_z_values, "reference_z_values")
    target_z = _coerce_numeric_sequence(target_z_values, "target_z_values")
    if len(references) != len(reference_z):
        raise ValueError("reference_slices and reference_z_values must have the same length.")
    if len(references) < 2:
        raise ValueError("simulate_stack requires at least two reference slices.")
    if not isinstance(target_blueprints, Mapping):
        raise TypeError("target_blueprints must be a mapping keyed by target z value.")
    if len(target_z) == 0:
        return {}

    order = np.argsort(reference_z)
    reference_z_sorted = reference_z[order]
    if len(np.unique(reference_z_sorted)) != len(reference_z_sorted):
        raise ValueError("reference_z_values must be unique.")
    references_sorted = [references[int(idx)] for idx in order]
    min_ref_z = float(reference_z_sorted[0])
    max_ref_z = float(reference_z_sorted[-1])

    from .local import simulate_local_references, local_seed
    references_sorted = [_reference_with_explicit_z(ref, z, idx)
                         for idx, (ref, z) in enumerate(zip(references_sorted, reference_z_sorted))]
    z_by_name = {ref.uns['reference_name']: float(z)
                 for ref, z in zip(references_sorted, reference_z_sorted)}
    output = {}
    fit_cache = {}
    for z in sorted(float(value) for value in target_z):
        blueprint = _target_blueprint_with_z(_lookup_target_blueprint(target_blueprints, z), z)
        result = simulate_local_references(references_sorted, blueprint, label_key=label_key,
            n_references=n_references, reference_z=z_by_name, target_z=z,
            config=gen_cfg, random_seed=local_seed(random_seed, 'target', str(z)),
            parameter_seed=random_seed, cache_path=cache_path, fit_cache=fit_cache)
        assign_generated_coordinates(result, blueprint.active_subset().coordinates, z_value=z)
        output[z] = result

    return output


def _coerce_reference_slices(reference_slices: Sequence[ad.AnnData]) -> list[ad.AnnData]:
    if isinstance(reference_slices, ad.AnnData):
        raise TypeError("reference_slices must be a sequence of AnnData objects, not a single AnnData.")
    references = list(reference_slices)
    if not references:
        raise ValueError("reference_slices must contain at least one AnnData object.")
    for idx, adata in enumerate(references):
        if not isinstance(adata, ad.AnnData):
            raise TypeError(f"reference_slices[{idx}] is not an AnnData object.")
    return references


def _coerce_numeric_sequence(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return arr


def _lookup_target_blueprint(target_blueprints: Mapping[Any, BlueprintInput], target_z: float) -> BlueprintInput:
    if target_z in target_blueprints:
        return target_blueprints[target_z]
    for key, value in target_blueprints.items():
        try:
            key_z = float(key)
        except (TypeError, ValueError):
            continue
        if np.isclose(key_z, float(target_z), rtol=0.0, atol=1e-12):
            return value
    raise KeyError(f"target_blueprints is missing a blueprint for target z value {target_z}.")


def _target_blueprint_with_z(source: BlueprintInput, target_z: float) -> SliceBlueprint:
    blueprint = load_blueprint(source)
    coords = np.asarray(blueprint.coordinates, dtype=float)
    if coords.shape[1] == 2:
        xyz = np.column_stack([coords, np.full(coords.shape[0], float(target_z), dtype=float)])
    else:
        if not np.allclose(coords[:, 2], float(target_z), rtol=1e-6, atol=1e-6):
            raise ValueError("3D target blueprint z coordinates must match the target z value.")
        xyz = coords.copy()
    return SliceBlueprint(
        coordinates=xyz,
        coordinate_mode=blueprint.coordinate_mode,
        grid_type=blueprint.grid_type,
        mask=None if blueprint.mask is None else blueprint.mask.copy(),
        domain_map=None if blueprint.domain_map is None else np.asarray(blueprint.domain_map).copy(),
        technology=blueprint.technology,
        obs=blueprint.obs.copy(),
        metadata=dict(blueprint.metadata),
    )


def _reference_with_explicit_z(adata: ad.AnnData, z_value: float, reference_index: int) -> ad.AnnData:
    if "spatial_3d" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial_3d"], dtype=float)
    elif "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
    else:
        raise ValueError("Each reference slice must contain obsm['spatial_3d'] or obsm['spatial'].")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Reference spatial coordinates must be a 2D array with at least two columns.")
    xy = coords[:, :2].copy()
    out = adata.copy()
    out.obsm["spatial"] = xy
    out.obsm["spatial_3d"] = np.column_stack([xy, np.full(xy.shape[0], float(z_value), dtype=float)])
    out.obs = out.obs.copy()
    out.obs["z"] = float(z_value)
    base_name = str(adata.uns.get("reference_name", f"reference_{reference_index:03d}"))
    out.uns["reference_name"] = f"{base_name}_z{float(z_value):g}"
    return out
