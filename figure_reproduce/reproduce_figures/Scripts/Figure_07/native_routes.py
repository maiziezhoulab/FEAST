"""Orthographic rendering and color calibration for the two plotted routes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

import native_geometry as common

WINDOW = (2640, 2040)
FOCUS = {"Wnt7a--Fzd2_Lrp5": ("IZ", "DorsalVZ"),
         "Vegfb--Fzd2": ("IZ", "DorsalVZ"),
         "Wnt7a--Fzd8_Lrp5": ("GE", "VentralVZ"),
         "Wnt7a--Fzd8_Lrp6": ("GE", "VentralVZ"),
         "Wnt7a--Fzd2_Lrp6": ("IZ", "DorsalVZ")}


# Drawing functions from 07_3d_transfer/plot_ccc_interface.py.


def face_mesh(faces: pd.DataFrame) -> pv.PolyData:
    half = common.SPACING / 2
    offsets = np.array([[[0, -half, -half], [0, half, -half], [0, half, half], [0, -half, half]],
                        [[-half, 0, -half], [half, 0, -half], [half, 0, half], [-half, 0, half]],
                        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]]])
    centers = faces[["center_x", "center_y", "center_z"]].to_numpy()
    vertices = centers[:, None, :] + offsets[faces.normal_axis.to_numpy()]
    cells = np.column_stack([np.full(len(faces), 4), np.arange(4 * len(faces)).reshape(-1, 4)])
    return pv.PolyData(vertices.reshape(-1, 3), cells.ravel())


# Drawing functions from 07_3d_transfer/plot_ccc_top5_3d.py.


def configure_plot():
    plot = pv.Plotter(off_screen=True, window_size=WINDOW)
    plot.ren_win.SetMultiSamples(0)
    plot.ren_win.SetAlphaBitPlanes(1)
    plot.set_background("white")
    plot.enable_parallel_projection()
    plot.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0)
    return plot


def position_stats(values, reference):
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    percentile = (left + right + 1) / (2 * len(reference))
    return percentile, {"n_positive_positions_drawn": len(values), "normalized_mass_pct_drawn": float(values.sum()),
                        "display_percentile_min": float(percentile.min()), "display_percentile_max": float(percentile.max())}


def add_positions(plot, xyz, values, role, reference):
    assert len(xyz) == len(values) and (values > 0).all()
    percentile, stats = position_stats(values, reference)
    rgba = common.CMAPS[role](percentile)
    rgba[:, 3] = .95 * percentile ** .7
    cloud = pv.PolyData(xyz)
    cloud["rgba"] = np.round(255 * rgba).astype(np.uint8)
    cube = pv.Cube(x_length=common.SPACING, y_length=common.SPACING, z_length=common.SPACING)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)
    plot.add_mesh(glyphs, scalars="rgba", rgba=True, lighting=False, show_scalar_bar=False)
    return stats


def finish_scene(plot, geometry, scale, path):
    center = geometry["center"]
    plot.camera_position = [center + 30 * common.VIEW, center, common.UP]
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()
    common.transparent_screenshot(plot, path)
    plot.close()


def interface_geometry(frames, geometries, edge_tables, input_root):
    result = {}
    for a, b in sorted(set(FOCUS.values())):
        name = f"{a}_{b}"
        result[name] = {}
        for age in common.AGES:
            frame = frames[age]
            xyz = frame[["x", "y", "z"]].to_numpy()
            mid = float(geometries[age]["center"][0])
            faces = pd.read_parquet(input_root / f"{age}_{name}_shared_faces.parquet")
            faces = faces.loc[faces.center_x.ge(mid)].copy()
            anchors = {}
            points = [faces[["center_x", "center_y", "center_z"]].to_numpy()]
            for (stage, key), table in edge_tables.items():
                if stage != age:
                    continue
                selected = table.sender_region.isin([a, b]) & table.receiver_region.isin([a, b])
                part = table.loc[selected]
                rows, cols = part.sender_row.to_numpy(), part.receiver_row.to_numpy()
                right = (xyz[rows, 0] >= mid) & (xyz[cols, 0] >= mid)
                points.extend([xyz[rows[right]], xyz[cols[right]]])
            all_points = np.vstack(points)
            low, high = all_points.min(axis=0), all_points.max(axis=0)
            center = (low + high) / 2
            delta = all_points - center
            extents = [float(np.abs(delta @ common.RIGHT).max()), float(np.abs(delta @ common.UP).max())]
            meshes = {}
            target = points[0].mean(axis=0)
            for region in (a, b):
                meshes[region] = geometries[age]["meshes"][region].clip(
                    normal=[1, 0, 0], origin=[mid, 0, 0], invert=False)
                assert meshes[region].points[:, 0].min() >= mid - 1e-5
                candidates = xyz[frame.region.eq(region).to_numpy() & (xyz[:, 0] >= mid)]
                anchors[region] = candidates[np.argmin(((candidates - target) ** 2).sum(axis=1))]
            result[name][age] = {"center": center, "half_extents": extents, "meshes": meshes,
                                  "anchors": anchors, "faces": faces, "contact": face_mesh(faces),
                                  "midline_x_mm": mid, "bounds": [low.tolist(), high.tolist()]}
        scale = 1.30 * max(max(g["half_extents"][1], g["half_extents"][0] / common.ASPECT) for g in result[name].values())
        result[name]["scale"] = scale
    return result


def local_scene(frame, geometry, edges, a, b, total, reference, scale, path, reuse):
    xyz = frame[["x", "y", "z"]].to_numpy()
    part = edges.loc[edges.sender_region.eq(a) & edges.receiver_region.eq(b)]
    rows, cols = part.sender_row.to_numpy(), part.receiver_row.to_numpy()
    keep = (xyz[rows, 0] >= geometry["midline_x_mm"]) & (xyz[cols, 0] >= geometry["midline_x_mm"])
    part = part.loc[keep]
    stats = {"n_edges_drawn": len(part), "right_hemisphere_W": float(part.score.sum())}
    plot = None
    if not (reuse and path.exists()):
        plot = configure_plot()
        for region, mesh in geometry["meshes"].items():
            plot.add_mesh(mesh, color="#a5aaae", opacity=.055, smooth_shading=True,
                          ambient=.7, diffuse=.3, show_scalar_bar=False)
        plot.add_mesh(geometry["contact"], color="#707a82", opacity=.08, lighting=False, show_scalar_bar=False)
    for role, column in [("sender", "sender_row"), ("receiver", "receiver_row")]:
        score = part.groupby(column, sort=True).score.sum()
        normalized = 100 * score.to_numpy() / total
        stats[role] = (position_stats(normalized, reference)[1] if plot is None else
                       add_positions(plot, xyz[score.index.to_numpy()], normalized, role, reference))
    if plot is not None:
        finish_scene(plot, geometry, scale, path)
    return stats
