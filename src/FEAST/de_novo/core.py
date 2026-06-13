from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class SliceBlueprint:
    coordinates: np.ndarray
    coordinate_mode: str = "generic"
    grid_type: str = "generic"
    mask: Optional[np.ndarray] = None
    domain_map: Optional[np.ndarray] = None
    technology: Optional[str] = None
    obs: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("SliceBlueprint.coordinates must be a 2D array with at least two columns.")
        if coords.shape[1] > 3:
            raise ValueError("SliceBlueprint.coordinates supports only 2D or 3D coordinates in v1.")
        self.coordinates = coords.copy()
        n_spots = self.coordinates.shape[0]

        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.ndim != 1 or mask.shape[0] != n_spots:
                raise ValueError("SliceBlueprint.mask must have one entry per spot.")
            self.mask = mask

        if self.domain_map is not None:
            domain_map = np.asarray(self.domain_map)
            if domain_map.ndim != 1 or domain_map.shape[0] != n_spots:
                raise ValueError("SliceBlueprint.domain_map must have one entry per spot.")
            self.domain_map = domain_map

        if self.obs.empty and len(self.obs.index) == 0:
            self.obs = pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)])
        else:
            self.obs = self.obs.reset_index(drop=True).copy()
        if len(self.obs) != n_spots:
            raise ValueError("SliceBlueprint.obs must contain one row per spot.")
        if self.domain_map is not None and "domain" not in self.obs:
            self.obs["domain"] = self.domain_map

    @property
    def n_spots(self) -> int:
        return self.coordinates.shape[0]

    @property
    def coordinate_dim(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def active_mask(self) -> np.ndarray:
        if self.mask is None:
            return np.ones(self.n_spots, dtype=bool)
        return np.asarray(self.mask, dtype=bool).copy()

    @property
    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask)

    @property
    def n_active_spots(self) -> int:
        return int(self.active_mask.sum())

    def active_subset(self) -> "SliceBlueprint":
        indices = self.active_indices
        if indices.size == 0:
            raise ValueError("SliceBlueprint mask drops all spots; at least one active spot is required.")
        domain_map = None if self.domain_map is None else np.asarray(self.domain_map)[indices]
        obs = self.obs.iloc[indices].reset_index(drop=True).copy()
        return SliceBlueprint(
            coordinates=self.coordinates[indices].copy(),
            coordinate_mode=self.coordinate_mode,
            grid_type=self.grid_type,
            mask=None,
            domain_map=domain_map,
            technology=self.technology,
            obs=obs,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": self.coordinates.tolist(),
            "coordinate_mode": self.coordinate_mode,
            "grid_type": self.grid_type,
            "mask": None if self.mask is None else self.mask.astype(bool).tolist(),
            "domain_map": None if self.domain_map is None else self.domain_map.tolist(),
            "technology": self.technology,
            "obs": self.obs.to_dict(orient="list"),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SliceBlueprint":
        obs_payload = payload.get("obs", {})
        obs = pd.DataFrame(obs_payload) if obs_payload else pd.DataFrame()
        return cls(
            coordinates=np.asarray(payload["coordinates"], dtype=float),
            coordinate_mode=payload.get("coordinate_mode", "generic"),
            grid_type=payload.get("grid_type", "generic"),
            mask=None if payload.get("mask") is None else np.asarray(payload["mask"], dtype=bool),
            domain_map=None if payload.get("domain_map") is None else np.asarray(payload["domain_map"]),
            technology=payload.get("technology"),
            obs=obs,
            metadata=dict(payload.get("metadata", {})),
        )


def load_blueprint(source: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str, Path]) -> SliceBlueprint:
    if isinstance(source, SliceBlueprint):
        return source
    if isinstance(source, ad.AnnData):
        if "spatial_3d" in source.obsm:
            coordinates = np.asarray(source.obsm["spatial_3d"], dtype=float)
        elif "spatial" in source.obsm:
            coordinates = np.asarray(source.obsm["spatial"], dtype=float)
        else:
            raise ValueError("Blueprint AnnData must contain obsm['spatial_3d'] or obsm['spatial'].")
        domain_map = None
        if "domain" in source.obs:
            domain_map = source.obs["domain"].to_numpy()
        elif "region" in source.obs:
            domain_map = source.obs["region"].to_numpy()
        return SliceBlueprint(
            coordinates=coordinates,
            grid_type=source.uns.get("grid_type", "generic"),
            domain_map=domain_map,
            technology=source.uns.get("technology"),
            obs=source.obs.copy(),
            metadata={"source": "anndata"},
        )
    if isinstance(source, (str, Path)):
        return load_blueprint(_load_mapping_file(source))
    if isinstance(source, Mapping):
        return SliceBlueprint.from_dict(source)
    raise TypeError("Unsupported blueprint input.")


def active_mask_metadata(blueprint: SliceBlueprint) -> Dict[str, Any]:
    return {
        "policy": "drop",
        "original_n_spots": int(blueprint.n_spots),
        "active_n_spots": int(blueprint.n_active_spots),
        "active_indices": [int(idx) for idx in blueprint.active_indices.tolist()],
    }


def assign_generated_coordinates(
    adata: ad.AnnData,
    coordinates: np.ndarray,
    *,
    z_value: Optional[float] = None,
    coordinate_system: str = "user_defined",
) -> None:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in {2, 3}:
        raise ValueError("Generated coordinates must have shape (n_spots, 2) or (n_spots, 3).")
    if coords.shape[0] != adata.n_obs:
        raise ValueError("Generated coordinate row count must match AnnData.n_obs.")

    if coords.shape[1] == 2:
        adata.obsm["spatial"] = coords.copy()
        if z_value is not None:
            xyz = np.column_stack([coords, np.full(coords.shape[0], float(z_value), dtype=float)])
            adata.obsm["spatial_3d"] = xyz
            adata.obs["z"] = float(z_value)
            _set_3d_coordinate_metadata(adata, coordinate_system=coordinate_system)
        return

    adata.obsm["spatial"] = coords[:, :2].copy()
    adata.obsm["spatial_3d"] = coords.copy()
    if z_value is not None:
        adata.obs["z"] = float(z_value)
    _set_3d_coordinate_metadata(adata, coordinate_system=coordinate_system)


def _set_3d_coordinate_metadata(adata: ad.AnnData, *, coordinate_system: str) -> None:
    de_novo = adata.uns.setdefault("de_novo", {})
    de_novo["coordinate_dim"] = 3
    de_novo["coordinate_keys"] = {"xy": "spatial", "xyz": "spatial_3d"}
    de_novo["coordinate_system"] = str(coordinate_system)


def _load_mapping_file(path: Union[str, Path]) -> Dict[str, Any]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() in {".yaml", ".yml"}:
        yaml = _import_yaml()
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping payload in {input_path}.")
    return dict(payload)


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("YAML support requires PyYAML to be installed.") from exc
    return yaml
