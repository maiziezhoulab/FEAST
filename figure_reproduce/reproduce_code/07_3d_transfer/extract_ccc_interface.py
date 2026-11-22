#!/usr/bin/env python3
"""Shared native anatomical-face extraction for current CCC source preparation."""

from __future__ import annotations
import numpy as np
import pandas as pd
SPACING = .02


def shared_faces(geometry: pd.DataFrame, sender: str, receiver: str) -> pd.DataFrame:
    xyz = geometry[["x", "y", "z"]].to_numpy()
    origin = xyz.min(axis=0) - SPACING
    indices = np.rint((xyz - origin) / SPACING).astype(np.int32)
    np.testing.assert_allclose(origin + SPACING * indices, xyz, rtol=0, atol=1e-6)
    grid = np.zeros(tuple(indices.max(axis=0) + 2), dtype=np.uint8)
    grid[tuple(indices.T)] = np.where(geometry.region.eq(sender), 1, 2)
    records = []
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = lower.copy()
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        a, b = grid[tuple(lower)], grid[tuple(upper)]
        contact = ((a == 1) & (b == 2)) | ((a == 2) & (b == 1))
        ijk = np.argwhere(contact)
        normal_sign = np.where(a[tuple(ijk.T)] == 1, 1, -1)
        centers = ijk.astype(float)
        centers[:, axis] += .5
        centers = origin + centers * SPACING
        records.append(pd.DataFrame({"center_x": centers[:, 0], "center_y": centers[:, 1],
                                     "center_z": centers[:, 2], "normal_axis": axis,
                                     "normal_sign_sender_to_receiver": normal_sign}))
    faces = pd.concat(records, ignore_index=True)
    faces.index.name = "face_id"
    return faces.reset_index()
