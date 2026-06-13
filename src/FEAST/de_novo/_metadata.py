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
