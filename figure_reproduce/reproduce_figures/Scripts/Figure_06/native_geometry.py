"""Native-grid rendering for the atlas cutaway and communication interfaces."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import pyvista as pv
from skimage.measure import marching_cubes

AGES = ("E15.5", "E18.5")
SPACING = 0.02
VIEW = np.array([1.0, -1.0, 0.9])
VIEW /= np.linalg.norm(VIEW)
RIGHT = np.cross(-VIEW, [0.0, 0.0, 1.0])
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(RIGHT, -VIEW)
ASPECT = 2200 / 1700
WINDOW = (3300, 2550)
CMAPS = {
    "sender": LinearSegmentedColormap.from_list("sender_orange", ["#fff5eb", "#fdae6b", "#e6550d", "#8c2d04"]),
    "receiver": LinearSegmentedColormap.from_list("receiver_blue", ["#eff3ff", "#9ecae1", "#3182bd", "#08519c"]),
}


# Drawing functions from 07_3d_transfer/plot_ccc_spatial.py.


def project(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    delta = np.atleast_2d(points) - center
    return np.column_stack([0.5 + delta @ RIGHT / (2 * scale * ASPECT),
                            0.5 + delta @ UP / (2 * scale)])


def load_geometry(frame: pd.DataFrame) -> dict:
    xyz = frame[["x", "y", "z"]].to_numpy()
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    # Exact index permutation of the native grid: x=i, y=k, z=-j.
    indices = np.column_stack([frame.voxel_i - frame.voxel_i.min() + 1,
                               frame.voxel_k - frame.voxel_k.min() + 1,
                               frame.voxel_j.max() - frame.voxel_j + 1]).astype(np.int32)
    shape = tuple((indices.max(axis=0) + 2).tolist())
    origin = low - SPACING
    np.testing.assert_allclose(origin + indices * SPACING, xyz, rtol=0, atol=1e-6)
    codes = frame.region.cat.codes.to_numpy() + 1
    labels = frame.region.cat.categories.tolist()
    mask = np.zeros(shape, dtype=np.uint8)
    mask[tuple(indices.T)] = codes
    meshes, anchors = {}, {}
    for code, label in enumerate(labels, start=1):
        # Extract each region's surface on the native 20-um grid.
        vertices, faces, _, _ = marching_cubes((mask == code).astype(np.uint8), 0.5,
                                              spacing=(SPACING,) * 3, step_size=1)
        vertices += origin
        cells = np.column_stack([np.full(len(faces), 3), faces]).ravel()
        meshes[label] = pv.PolyData(vertices, cells)
        subset = xyz[codes == code]
        # Anchor on an actual voxel in the right half, never a centroid in a cavity.
        right_half = subset[subset[:, 0] >= (low[0] + high[0]) / 2]
        target = right_half.mean(axis=0)
        anchors[label] = right_half[np.argmin(np.sum((right_half - target) ** 2, axis=1))]
    delta = xyz - (low + high) / 2
    half_extents = [float(np.max(np.abs(delta @ RIGHT))), float(np.max(np.abs(delta @ UP)))]
    return {"indices": indices, "shape": shape, "origin": origin, "half_extents": half_extents,
            "center": (low + high) / 2, "bounds": [low.tolist(), high.tolist()],
            "meshes": meshes, "anchors": anchors}


def transparent_screenshot(plot, path: Path) -> None:
    """Export straight RGBA from VTK's white-composited RGB and native alpha."""
    rgba = plot.screenshot(transparent_background=True)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
    rgb = rgba[:, :, :3].astype(np.float32)
    rgb = np.divide(rgb - 255 * (1 - alpha), alpha,
                    out=np.zeros_like(rgb), where=alpha > 0)
    rgba[:, :, :3] = np.rint(rgb).clip(0, 255).astype(np.uint8)
    plt.imsave(path, rgba)


# Drawing functions from 07_3d_transfer/plot_ccc_annotated_anatomy.py.


def cutaway(frame: pd.DataFrame, geometry: dict) -> dict:
    """Expose a coronal face at 60% of the posterior-to-anterior extent."""
    indices = geometry["indices"]
    plane = int(round(indices[:, 1].min() + .6 * np.ptp(indices[:, 1])))
    mask = np.zeros(geometry["shape"], dtype=np.uint8)
    keep = indices[:, 1] >= plane
    codes = frame.region.cat.codes.to_numpy() + 1
    mask[tuple(indices[keep].T)] = codes[keep]
    xyz = frame[["x", "y", "z"]].to_numpy()
    meshes, anchors = {}, {}
    for code, label in enumerate(frame.region.cat.categories, 1):
        vertices, faces, _, _ = marching_cubes((mask == code).astype(np.uint8), .5, spacing=(SPACING,) * 3)
        vertices += geometry["origin"]
        meshes[label] = pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
        candidates = xyz[(codes == code) & (indices[:, 1] == plane) & (xyz[:, 0] >= geometry["center"][0])]
        assert len(candidates), f"{label} absent from the right side of the displayed cut face."
        target = candidates.mean(axis=0)
        anchors[label] = candidates[np.argmin(((candidates - target) ** 2).sum(axis=1))]
    return {**geometry, "meshes": meshes, "anchors": anchors,
            "cut_y_mm": float(geometry["origin"][1] + (plane - .5) * SPACING),
            "cut_voxel_center_y_mm": float(geometry["origin"][1] + plane * SPACING),
            "retained_positions": int(keep.sum())}


def render(geometry: dict, colors: dict, scale: float, path: Path) -> None:
    plot = pv.Plotter(off_screen=True, window_size=WINDOW)
    plot.set_background("white")
    plot.enable_parallel_projection()
    plot.ren_win.SetMultiSamples(8)
    for region, mesh in geometry["meshes"].items():
        plot.add_mesh(mesh, color=colors[region], opacity=1, smooth_shading=True,
                      ambient=.5, diffuse=.5, specular=.08, show_scalar_bar=False)
    center = geometry["center"]
    plot.camera_position = [center + 30 * VIEW, center, UP]
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()
    transparent_screenshot(plot, path)
    plot.close()
