from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .core import SliceBlueprint, load_blueprint


ALLOWED_MOTIFS = {"layered", "gradient", "hotspot", "ring", "clustered", "diffuse"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _coerce_gene_names(gene_names: Sequence[str]) -> list[str]:
    names = [str(gene) for gene in gene_names]
    if len(names) == 0:
        raise ValueError("gene_names must contain at least one gene.")
    if len(set(names)) != len(names):
        raise ValueError("gene_names must be unique.")
    return names


def _ensure_pattern_blueprint(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    *,
    label_key: str,
) -> SliceBlueprint:
    loaded = load_blueprint(blueprint)
    if loaded.domain_map is None:
        loaded = SliceBlueprint(
            coordinates=loaded.coordinates,
            coordinate_mode=loaded.coordinate_mode,
            grid_type=loaded.grid_type,
            mask=loaded.mask,
            domain_map=np.full(loaded.n_spots, "global", dtype=object),
            technology=loaded.technology,
            obs=loaded.obs.copy(),
            metadata=dict(loaded.metadata),
        )
    if label_key not in loaded.obs:
        loaded.obs[label_key] = np.asarray(loaded.domain_map).astype(str)
    return loaded


def _normalized_coordinates(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    mins = coords.min(axis=0, keepdims=True)
    span = coords.max(axis=0, keepdims=True) - mins
    span[span <= 1e-8] = 1.0
    return (coords - mins) / span


def _scope_mask(blueprint: SliceBlueprint, motif: Mapping[str, Any], *, label_key: str) -> np.ndarray:
    labels = blueprint.obs[label_key].astype(str).to_numpy()
    scope = str(motif.get("scope", "global")).lower().strip()
    if scope == "global":
        return np.ones(blueprint.n_spots, dtype=bool)
    if scope == "domain":
        domain = str(motif.get("domain", ""))
        return labels == domain
    if scope == "domains":
        domains = [str(item) for item in motif.get("domains", [])]
        return np.isin(labels, domains)
    raise ValueError("motif scope must be one of {'global', 'domain', 'domains'}.")


def _scope_weights(
    blueprint: SliceBlueprint,
    motif: Mapping[str, Any],
    *,
    label_key: str,
    boundary_softness: float,
) -> np.ndarray:
    mask = _scope_mask(blueprint, motif, label_key=label_key)
    if np.all(mask):
        return np.ones(blueprint.n_spots, dtype=float)
    if not np.any(mask):
        return np.zeros(blueprint.n_spots, dtype=float)

    transition_width = float(motif.get("transition_width", boundary_softness))
    transition_width = float(np.clip(transition_width, 0.0, None))
    if transition_width <= 0.0:
        return mask.astype(float)

    coords = _normalized_coordinates(blueprint.coordinates)
    inside = coords[mask]
    outside_idx = ~mask
    if not np.any(outside_idx):
        return np.ones(blueprint.n_spots, dtype=float)

    nbrs = NearestNeighbors(n_neighbors=1).fit(inside)
    dists, _ = nbrs.kneighbors(coords[outside_idx])
    weights = np.ones(blueprint.n_spots, dtype=float)
    weights[outside_idx] = np.exp(-0.5 * (dists[:, 0] / max(transition_width, 1e-6)) ** 2)
    return weights


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = np.clip(arr, 0.0, None)
    vmax = float(np.max(arr)) if arr.size else 0.0
    if vmax <= 0:
        return np.zeros_like(arr)
    return arr / vmax


def diffuse_quantile_map(
    values: np.ndarray,
    coordinates: np.ndarray,
    diffusion_level: float,
    *,
    n_neighbors: int = 12,
    sigma: float = 0.15,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True
    if arr.ndim != 2:
        raise ValueError("values must be a 1D or 2D array.")

    n_spots, n_genes = arr.shape
    if n_spots < 2 or n_genes == 0:
        return arr[:, 0] if squeeze else arr

    level = float(np.clip(diffusion_level, 0.0, 1.0))
    if level <= 0:
        return arr[:, 0] if squeeze else arr

    coords_norm = _normalized_coordinates(np.asarray(coordinates, dtype=float))
    k = min(max(1, int(n_neighbors)), n_spots - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords_norm)
    dists, indices = nbrs.kneighbors(coords_norm)
    neighbor_idx = indices[:, 1:]
    neighbor_dist = dists[:, 1:]
    sigma_eff = max(float(sigma), 1e-6)
    weights = np.exp(-0.5 * (neighbor_dist / sigma_eff) ** 2).astype(np.float32)
    weight_sums = np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)

    neighbor_vals = arr[neighbor_idx, :]
    neighbor_mean = (weights[:, :, None] * neighbor_vals).sum(axis=1) / weight_sums
    smoothed = arr * (1.0 - level) + neighbor_mean * level
    smoothed = np.clip(smoothed, 0.0, 1.0)
    return smoothed[:, 0] if squeeze else smoothed


def _axis_index(axis_name: str, coordinate_dim: int) -> int:
    axis = _AXIS_INDEX.get(str(axis_name).lower().strip())
    if axis is None:
        raise ValueError("axis must be one of {'x', 'y', 'z'}.")
    if axis >= int(coordinate_dim):
        raise ValueError(f"axis '{axis_name}' requires {axis + 1}D coordinates.")
    return axis


def _default_center(coordinate_dim: int) -> list[float]:
    return [0.5] * int(coordinate_dim)


def _coerce_centers(centers: Any, coordinate_dim: int) -> np.ndarray:
    arr = np.asarray(centers, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != int(coordinate_dim):
            raise ValueError(f"center must have {int(coordinate_dim)} coordinates.")
        arr = arr.reshape(1, int(coordinate_dim))
    if arr.ndim != 2 or arr.shape[1] != int(coordinate_dim):
        raise ValueError(f"centers must be an array of shape (n, {int(coordinate_dim)}).")
    return arr


def _layered_pattern(coords: np.ndarray, motif: Mapping[str, Any]) -> np.ndarray:
    axis_name = str(motif.get("axis", "y")).lower().strip()
    axis = _axis_index(axis_name, coords.shape[1])
    centers = motif.get("centers", motif.get("center", 0.5))
    centers_arr = np.atleast_1d(np.asarray(centers, dtype=float)).reshape(-1)
    width = max(float(motif.get("width", 0.18)), 1e-6)
    values = np.zeros(coords.shape[0], dtype=float)
    for center in centers_arr:
        values = np.maximum(values, np.exp(-0.5 * ((coords[:, axis] - float(center)) / width) ** 2))
    return _normalize_map(values)


def _gradient_pattern(coords: np.ndarray, motif: Mapping[str, Any]) -> np.ndarray:
    axis_name = str(motif.get("axis", "x")).lower().strip()
    if axis_name in _AXIS_INDEX:
        axis = _axis_index(axis_name, coords.shape[1])
        values = coords[:, axis].copy()
    else:
        default_direction = np.zeros(coords.shape[1], dtype=float)
        default_direction[0] = 1.0
        direction = np.asarray(motif.get("direction", default_direction), dtype=float).reshape(-1)
        if direction.shape[0] != coords.shape[1]:
            raise ValueError(f"gradient direction must have {coords.shape[1]} components.")
        norm = np.linalg.norm(direction)
        if norm <= 1e-8:
            direction = default_direction
            norm = 1.0
        direction = direction / norm
        values = coords @ direction
    values = _normalize_map(values)
    if bool(motif.get("invert", False)):
        values = 1.0 - values
    return values


def _hotspot_pattern(coords: np.ndarray, motif: Mapping[str, Any]) -> np.ndarray:
    center = _coerce_centers(motif.get("center", _default_center(coords.shape[1])), coords.shape[1])[0]
    radius = max(float(motif.get("radius", 0.18)), 1e-6)
    dist = np.linalg.norm(coords - center[None, :], axis=1)
    values = np.exp(-0.5 * (dist / radius) ** 2)
    return _normalize_map(values)


def _ring_pattern(coords: np.ndarray, motif: Mapping[str, Any]) -> np.ndarray:
    center = _coerce_centers(motif.get("center", _default_center(coords.shape[1])), coords.shape[1])[0]
    radius = float(motif.get("radius", 0.3))
    width = max(float(motif.get("width", 0.08)), 1e-6)
    dist = np.linalg.norm(coords - center[None, :], axis=1)
    values = np.exp(-0.5 * ((dist - radius) / width) ** 2)
    return _normalize_map(values)


def _clustered_pattern(coords: np.ndarray, motif: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    centers = motif.get("centers")
    if centers is None:
        n_clusters = max(int(motif.get("n_clusters", 3)), 1)
        centers_arr = rng.random((n_clusters, coords.shape[1]), dtype=np.float64)
    else:
        centers_arr = _coerce_centers(centers, coords.shape[1])
    radius = max(float(motif.get("radius", 0.12)), 1e-6)
    values = np.zeros(coords.shape[0], dtype=float)
    for center in centers_arr:
        dist = np.linalg.norm(coords - center[None, :], axis=1)
        values = np.maximum(values, np.exp(-0.5 * (dist / radius) ** 2))
    return _normalize_map(values)


def _diffuse_pattern(coords: np.ndarray, motif: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    n_components = max(int(motif.get("n_components", 6)), 1)
    length_scale = max(float(motif.get("length_scale", 0.3)), 1e-6)
    centers = rng.random((n_components, coords.shape[1]), dtype=np.float64)
    weights = rng.uniform(0.5, 1.0, size=n_components)
    values = np.zeros(coords.shape[0], dtype=float)
    for weight, center in zip(weights, centers):
        dist = np.linalg.norm(coords - center[None, :], axis=1)
        values += float(weight) * np.exp(-0.5 * (dist / length_scale) ** 2)
    return _normalize_map(values)


def evaluate_motif(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    motif: Mapping[str, Any],
    *,
    label_key: str = "domain",
    random_seed: int = 0,
    boundary_softness: float = 0.0,
) -> np.ndarray:
    bp = _ensure_pattern_blueprint(blueprint, label_key=label_key)
    coords = _normalized_coordinates(bp.coordinates)
    kind = str(motif.get("kind", "")).lower().strip()
    if kind not in ALLOWED_MOTIFS:
        raise ValueError(f"Unsupported motif kind '{kind}'.")

    rng = np.random.default_rng(int(motif.get("seed", random_seed)))
    if kind == "layered":
        base_values = _layered_pattern(coords, motif)
    elif kind == "gradient":
        base_values = _gradient_pattern(coords, motif)
    elif kind == "hotspot":
        base_values = _hotspot_pattern(coords, motif)
    elif kind == "ring":
        base_values = _ring_pattern(coords, motif)
    elif kind == "clustered":
        base_values = _clustered_pattern(coords, motif, rng)
    elif kind == "diffuse":
        base_values = _diffuse_pattern(coords, motif, rng)
    else:
        raise AssertionError("unreachable motif kind")

    support = _scope_weights(
        bp,
        motif,
        label_key=label_key,
        boundary_softness=boundary_softness,
    )
    weight = float(motif.get("weight", 1.0))
    values = _normalize_map(base_values * support) * weight
    return values


def compose_pattern(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    motifs: Sequence[Mapping[str, Any]],
    *,
    label_key: str = "domain",
    random_seed: int = 0,
    boundary_softness: float = 0.0,
) -> np.ndarray:
    bp = _ensure_pattern_blueprint(blueprint, label_key=label_key)
    combined = np.zeros(bp.n_spots, dtype=float)
    for idx, motif in enumerate(motifs):
        combined += evaluate_motif(
            bp,
            motif,
            label_key=label_key,
            random_seed=int(random_seed) + idx,
            boundary_softness=boundary_softness,
        )
    return np.clip(combined, 0.0, 1.0)


class SimulationPatternBuilder:
    def __init__(self, gene_names: Sequence[str]) -> None:
        self._gene_names = _coerce_gene_names(gene_names)
        self._patterns: Dict[str, list[Dict[str, Any]]] = {gene: [] for gene in self._gene_names}

    @classmethod
    def from_gene_names(cls, gene_names: Sequence[str]) -> "SimulationPatternBuilder":
        return cls(gene_names)

    def add_motif(self, gene_name: str, kind: str, **kwargs: Any) -> "SimulationPatternBuilder":
        gene = str(gene_name)
        if gene not in self._patterns:
            raise KeyError(f"Unknown gene '{gene}'.")
        kind_name = str(kind).lower().strip()
        if kind_name not in ALLOWED_MOTIFS:
            raise ValueError(f"Unsupported motif kind '{kind_name}'.")
        motif = {"kind": kind_name, **kwargs}
        self._patterns[gene].append(motif)
        return self

    def layered(self, gene_name: str, **kwargs: Any) -> "SimulationPatternBuilder":
        return self.add_motif(gene_name, "layered", **kwargs)

    def gradient(self, gene_name: str, **kwargs: Any) -> "SimulationPatternBuilder":
        return self.add_motif(gene_name, "gradient", **kwargs)

    def hotspot(self, gene_name: str, **kwargs: Any) -> "SimulationPatternBuilder":
        return self.add_motif(gene_name, "hotspot", **kwargs)

    def ring(self, gene_name: str, **kwargs: Any) -> "SimulationPatternBuilder":
        return self.add_motif(gene_name, "ring", **kwargs)

    def clustered(self, gene_name: str, **kwargs: Any) -> "SimulationPatternBuilder":
        return self.add_motif(gene_name, "clustered", **kwargs)

    def diffuse(self, gene_name: str, **kwargs: Any) -> "SimulationPatternBuilder":
        return self.add_motif(gene_name, "diffuse", **kwargs)

    def build(self) -> Dict[str, list[Dict[str, Any]]]:
        return {gene: [dict(motif) for motif in motifs] for gene, motifs in self._patterns.items() if motifs}


def plot_pattern(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    pattern_spec: Mapping[str, Sequence[Mapping[str, Any]]],
    gene_name: str,
    *,
    label_key: str = "domain",
    random_seed: int = 0,
    boundary_softness: float = 0.0,
    ax=None,
    cmap: str = "magma",
    s: float = 16.0,
    title: Optional[str] = None,
):
    bp = _ensure_pattern_blueprint(blueprint, label_key=label_key)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
    pattern = compose_pattern(
        bp,
        pattern_spec.get(str(gene_name), []),
        label_key=label_key,
        random_seed=random_seed,
        boundary_softness=boundary_softness,
    )
    sca = ax.scatter(bp.coordinates[:, 0], bp.coordinates[:, 1], c=pattern, cmap=cmap, s=s)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or str(gene_name))
    return fig, ax, sca


def plot_pattern_panel(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    pattern_spec: Mapping[str, Sequence[Mapping[str, Any]]],
    genes: Sequence[str],
    *,
    label_key: str = "domain",
    random_seed: int = 0,
    boundary_softness: float = 0.0,
    cmap: str = "magma",
    s: float = 16.0,
):
    genes = [str(gene) for gene in genes]
    fig, axes = plt.subplots(1, len(genes), figsize=(4.0 * len(genes), 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    sca = None
    for idx, (ax, gene) in enumerate(zip(axes, genes)):
        _, _, sca = plot_pattern(
            blueprint,
            pattern_spec,
            gene,
            label_key=label_key,
            random_seed=int(random_seed) + idx,
            boundary_softness=boundary_softness,
            ax=ax,
            cmap=cmap,
            s=s,
            title=gene,
        )
    return fig, axes, sca
