"""Render S100 reference classes and full and enlarged mixed-spot compositions."""

import json

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import deconvolution_spatial as spatial
from layout import panel, panel_args, save_panel

SECTION = "100"
SOURCES = ("truth", "cell2location", "rctd")


def read_compositions(run, retained):
    tables = []
    for resolution in spatial.RESOLUTIONS:
        simulation = ad.read_h5ad(run / f"simulations/{SECTION}/resolution_{resolution}.h5ad", backed="r")
        try:
            spots = pd.Index([f"spot_{name}" for name in simulation.obs_names], name="spot_barcode")
            xy = simulation.obsm["spatial"]
            library = np.asarray(simulation.X[:].sum(axis=1)).ravel()
            metadata = pd.DataFrame({
                "x": xy[:, 0], "y": xy[:, 1], "library_size": library,
                "n_source_cells": simulation.obs["n_source_cells"].to_numpy(),
                "resolution": float(resolution),
            }, index=spots)
        finally:
            simulation.file.close()
        for source in SOURCES:
            suffix = f"{SECTION}/resolution_{resolution}_proportions.csv"
            path = run / (f"truth/{suffix}" if source == "truth" else f"methods/{source}/{suffix}")
            proportions = pd.read_csv(path, index_col=0)
            expected = spots[library > 0] if source == "rctd" else spots
            if not proportions.index.equals(expected) or not proportions.columns.is_unique:
                raise ValueError(f"Composition spot support differs for {path}")
            if source != "truth" and set(proportions.columns) != set(retained):
                raise ValueError(f"Fit class support differs for {path}")
            collapsed = proportions.loc[:, retained].copy()
            collapsed["Other"] = proportions.loc[:, proportions.columns.difference(retained)].sum(axis=1)
            tables.append(metadata.assign(source=source).join(collapsed.reindex(spots, fill_value=0)))
    return pd.concat(tables)


def draw_key(axis, palette):
    axis.axis("off")
    axis.text(0, 0.99, "Cell classes", va="top", fontsize=8.6, weight="bold")
    entries = list(palette.items())
    per_column = (len(entries) + 1) // 2
    for i, (name, color) in enumerate(entries):
        column, row = divmod(i, per_column)
        x = 0.535 * column
        y = 0.865 - 0.094 * row
        axis.add_patch(Rectangle((x, y - 0.022), 0.03, 0.038, color=color, lw=0))
        axis.text(x + 0.036, y, name, fontsize=5.9, va="center")


def main():
    args = panel_args("d", __doc__)
    study = args.results / "03_deconvolution"
    run = study / "cell_class"
    audit = pd.read_csv(run / "scores/support_audit.csv", dtype={"slice_id": str})
    audit = audit.loc[audit.slice_id.eq(SECTION)]
    retained_lists = [json.loads(value) for value in audit.retained_cell_class_order_json]
    if len(retained_lists) != 2 or retained_lists[0] != retained_lists[1]:
        raise ValueError("S100 fitted classes differ across resolutions")
    retained = retained_lists[0]
    palette = {label: spatial.FULL_CELL_TYPE_COLORS[label] for label in retained}
    palette.update([spatial.OTHER_CELL_CLASS])
    reference = ad.read_h5ad(args.inputs / f"03_deconvolution/Zhuang-ABCA-1.{SECTION}.h5ad", backed="r")
    try:
        xy = reference.obsm["spatial"]
        classes = reference.obs["cell_class"].astype(str)
        measured = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cell_class": classes})
        measured["display_class"] = classes.where(classes.isin(retained), "Other")
        measured.index.name = "cell_barcode"
    finally:
        reference.file.close()
    data = read_compositions(run, retained)
    tables = study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)
    measured.to_csv(tables / "d_reference_spots.csv.gz")
    data.to_csv(tables / "d_compositions.csv.gz")
    pd.Series(palette, name="color").rename_axis("cell_class").to_csv(tables / "d_palette.csv")

    xlim, ylim = spatial.common_limits(xy)
    roi = spatial.roi_bounds(xlim, ylim)
    distances = {
        resolution: spatial.nearest_neighbor_distance(data.loc[
            data.resolution.eq(float(resolution)) & data.source.eq("truth"), ["x", "y"]].to_numpy())
        for resolution in spatial.RESOLUTIONS
    }
    radii = {resolution: distance * spatial.FULL_RADIUS_SCALE[resolution] for resolution, distance in distances.items()}
    enlarged_radius = distances["0.1"] * spatial.ROI_RADIUS_SCALE
    (tables / "d_display.json").write_text(json.dumps({
        "section": SECTION, "x_limits": xlim, "y_limits": ylim, "roi_bounds": roi,
        "pie_radius": radii, "roi_pie_radius": enlarged_radius,
        "occupied_cells_only": True, "minimum_displayed_fraction": spatial.DISPLAY_THRESHOLD,
        "wedge_order": "clockwise from 90 degrees", "y_axis": "image coordinates, increasing down",
    }, indent=2) + "\n")

    fig = panel("d", label_x=1)
    grid = fig.add_gridspec(3, 4, left=0.029, right=0.994, top=0.904, bottom=0.007,
                          width_ratios=[1.13, 1.2, 1.2, 0.91], wspace=0.16, hspace=0.05)
    ref = fig.add_subplot(grid[0, 0])
    spatial.format_spatial_axis(ref, xlim, ylim)
    ref.scatter(measured.x, measured.y, c=measured.display_class.map(palette), s=0.43,
                linewidths=0, rasterized=True)
    ref.set_title("Single-cell\nMERFISH", fontsize=9, pad=3)
    key = fig.add_subplot(grid[1:, 0])
    key.set_position([0.025, 0.145, 0.27, 0.36])
    draw_key(key, palette)
    for row, source in enumerate(SOURCES):
        for column, resolution in enumerate(("0.25", "0.1", "roi"), start=1):
            axis = fig.add_subplot(grid[row, column])
            is_roi = resolution == "roi"
            factor = "0.1" if is_roi else resolution
            values = data.loc[data.resolution.eq(float(factor)) & data.source.eq(source)]
            occupied = values.loc[values.n_source_cells.gt(0)]
            coords = occupied[["x", "y"]].to_numpy()
            props = occupied[list(palette)].to_numpy()
            if is_roi:
                spatial.format_spatial_axis(axis, roi[:2], roi[2:])
                axis.add_collection(spatial.pie_collection(
                    coords, props, enlarged_radius, palette, crop=roi, linewidth=spatial.ROI_WEDGE_LINEWIDTH,
                ))
            else:
                spatial.format_spatial_axis(axis, xlim, ylim)
                axis.add_collection(spatial.pie_collection(coords, props, radii[factor], palette))
                if factor == "0.1":
                    spatial.add_roi_rectangle(axis, roi)
            if column == 1:
                axis.text(-0.035, 0.5, spatial.SOURCE_LABELS[source], rotation=90,
                          ha="right", va="center", fontsize=8.1, transform=axis.transAxes)
            if row == 0:
                axis.set_title({"0.25": "×0.25", "0.1": "×0.10", "roi": "Enlarged ROI"}[resolution],
                               fontsize=9, pad=3)
    fig.text(0.557, 0.972, "Resolution", ha="center", va="top", fontsize=9.1)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
