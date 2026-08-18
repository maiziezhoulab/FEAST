from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _stringify_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool, np.generic)):
        return str(_json_default(value))
    return json.dumps(value, default=_json_default, sort_keys=True)


def records_by_label_to_h5ad_uns(records_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Convert label -> list[record] diagnostics into AnnData/HDF5-safe metadata."""
    out: Dict[str, Dict[str, Any]] = {}
    for label, records in records_by_label.items():
        rows: List[Mapping[str, Any]] = [dict(row) for row in records]
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                key_s = str(key)
                if key_s not in seen:
                    seen.add(key_s)
                    keys.append(key_s)

        label_out: Dict[str, Any] = {
            "format": "columnar_records_v1",
            "n_records": int(len(rows)),
        }
        for key in keys:
            label_out[key] = [_stringify_metadata_value(row.get(key, "")) for row in rows]
        out[str(label)] = label_out
    return out


def records_to_h5ad_uns(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Convert a list of metadata records to an HDF5-safe columnar mapping."""
    return records_by_label_to_h5ad_uns({"records": records})["records"]


def encode_blueprint_h5ad_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Encode a blueprint while preserving absent optional-field semantics."""
    present_fields = {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }
    return encode_feast_h5ad_metadata(present_fields)


def encode_feast_h5ad_metadata(value: Any) -> Any:
    """Encode package-owned metadata for AnnData HDF5 I/O.

    AnnData cannot serialize a list of mappings or ``None`` nested inside
    ``.uns``. Record lists use the same columnar representation as transport
    diagnostics; simple mappings and arrays retain their structure. This
    intentionally changes the in-memory representation of FEAST-generated
    metadata and must not be applied to arbitrary user ``.uns`` content.
    """
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return encode_feast_h5ad_metadata(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): encode_feast_h5ad_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence):
        items = list(value)
        if items and all(isinstance(item, Mapping) for item in items):
            return records_to_h5ad_uns(items)
        array = np.asarray(items)
        if array.dtype != object:
            return array
        return [_stringify_metadata_value(item) for item in items]
    return str(value)
