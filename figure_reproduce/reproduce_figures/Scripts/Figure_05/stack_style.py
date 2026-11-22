"""Geometry and display-only textures for the archived two-bracket illustration."""
from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

VIRIDIS = mpl.colormaps["viridis"]
GENERATED = "#0072B2"


# Drawing functions from 08_workflow/build_anatomical_workflow.py.


def expression_grid(
    xy: np.ndarray,
    values: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    bins: int = 180,
) -> dict[str, Any]:
    x_edges = np.linspace(*limits[0], bins + 1)
    y_edges = np.linspace(*limits[1], bins + 1)
    weighted, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges), weights=values)
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges))
    smooth_weighted = gaussian_filter(weighted, sigma=1.0)
    smooth_count = gaussian_filter(count, sigma=1.0)
    texture = np.divide(smooth_weighted, smooth_count, out=np.zeros_like(smooth_weighted), where=smooth_count > 1e-3)
    alpha = np.clip(smooth_count / np.quantile(smooth_count[smooth_count > 0], 0.55), 0, 1)
    alpha[smooth_count < 0.08] = 0
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return {"x": x_centers, "y": y_centers, "texture": texture, "alpha": alpha}


def geometry_grid(
    xy: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    bins: int = 180,
) -> dict[str, Any]:
    return expression_grid(xy, np.ones(len(xy), dtype=float), limits, bins=bins)


def add_grid_plane(
    axis: plt.Axes,
    grid: dict[str, Any],
    display_z: float,
    norm: Normalize,
    *,
    color: str | None = None,
    alpha_scale: float = 1.0,
) -> None:
    x, y = np.meshgrid(grid["x"], grid["y"], indexing="ij")
    z = np.full_like(x, display_z, dtype=float)
    if color is None:
        rgba = VIRIDIS(norm(grid["texture"]))
    else:
        rgba = np.broadcast_to(mcolors.to_rgba(color), (*grid["texture"].shape, 4)).copy()
    rgba[..., 3] = grid["alpha"] * alpha_scale
    axis.plot_surface(x, y, z, facecolors=rgba, rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False, rasterized=True)


def add_grid_outline(axis: plt.Axes, grid: dict[str, Any], display_z: float, *, color: str, linewidth: float) -> None:
    axis.contour(
        grid["x"],
        grid["y"],
        grid["alpha"].T,
        levels=(0.15,),
        zdir="z",
        offset=display_z + 0.012,
        colors=(color,),
        linewidths=linewidth,
    )


def study06_display_positions(context_slices: list[int], target_slice: int) -> dict[int, float]:
    steps = np.full(max(len(context_slices) - 1, 0), 0.65, dtype=float)
    target_index = context_slices.index(target_slice)
    if 0 < target_index < len(context_slices) - 1:
        steps[target_index - 1:target_index + 1] = 1.25
    positions = np.concatenate([[0.0], np.cumsum(steps)])
    return dict(zip(context_slices, positions - positions[target_index]))


def render_study06_stack(axis: plt.Axes, data: dict[str, Any], *, include_blueprint: bool, include_generated: bool) -> None:
    context_slices = data["context_slices"]
    positions = study06_display_positions(context_slices, int(data["target"]["target_slice"]))
    lower_slice = int(data["target"]["lower_ref_slice"])
    upper_slice = int(data["target"]["upper_ref_slice"])
    target_slice = int(data["target"]["target_slice"])
    for slice_id, grid in data["context_grids"].items():
        if slice_id not in (lower_slice, upper_slice):
            add_grid_plane(axis, grid, positions[slice_id], data["norm"], alpha_scale=0.58)
            add_grid_outline(axis, grid, positions[slice_id], color="#AAB7C0", linewidth=0.65)
    add_grid_plane(axis, data["lower_grid"], positions[lower_slice], data["norm"], alpha_scale=0.96)
    if include_generated:
        add_grid_plane(axis, data["generated_grid"], positions[target_slice], data["norm"], alpha_scale=0.98)
        add_grid_outline(axis, data["generated_grid"], positions[target_slice], color=GENERATED, linewidth=1.25)
    elif include_blueprint:
        add_grid_outline(axis, data["blueprint_grid"], positions[target_slice], color="#AAB7C0", linewidth=1.15)
    add_grid_plane(axis, data["upper_grid"], positions[upper_slice], data["norm"], alpha_scale=0.96)
    xlim, ylim = data["limits"]
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_zlim(min(positions.values()) - 0.3, max(positions.values()) + 0.3)
    axis.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], 5.7))
    axis.view_init(elev=24, azim=-57)
    axis.set_proj_type("persp", focal_length=0.9)
    axis.set_axis_off()
