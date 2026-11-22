"""Render S100 reference classes and full and enlarged mixed-spot compositions."""

import json

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import deconvolution_spatial as spatial
from layout import manuscript_axes, manuscript_text, panel, panel_args, save_panel

SECTION = "100"
SOURCES = ("truth", "cell2location", "rctd")
COLUMN_CENTERS = (202.5, 327.6, 438.0, 548.2)
ROW_CENTERS = (386.3, 469.3, 551.6)


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
    axis.text(0, 0.99, "Cell classes", va="top", fontsize=8.5)
    entries = list(palette.items())
    per_column = (len(entries) + 1) // 2
    for i, (name, color) in enumerate(entries):
        column, row = divmod(i, per_column)
        x = 0.53 * column
        y = 0.835 - 0.113 * row
        axis.add_patch(Rectangle((x, y - 0.022), 0.03, 0.038, color=color, lw=0))
        axis.text(x + 0.036, y, name, fontsize=5.9, va="center")


def read_plot_data(path, palette):
    rows = pd.read_csv(path, dtype={"unit_id": str, "resolution": str}, float_precision="round_trip")
    measured = rows.loc[rows.panel.eq("single_cell_reference"), ["unit_id", "x", "y", "cell_class"]]
    measured = measured.set_index("unit_id", verify_integrity=True).rename(columns={"cell_class": "display_class"})
    grouped = rows.loc[~rows.panel.eq("single_cell_reference")].rename(columns={"panel": "source"})
    grouped["resolution"] = grouped.resolution.astype(float)
    keys = ["source", "resolution", "unit_id"]
    metadata = grouped.loc[:, keys + ["x", "y", "n_source_cells"]].drop_duplicates().set_index(keys, verify_integrity=True)
    proportions = grouped.pivot(index=keys, columns="cell_class", values="proportion")
    if (set(proportions.columns) != set(palette) or proportions.isna().any().any()
            or not set(measured.display_class).issubset(palette)):
        raise ValueError(f"Archive cell classes differ from scored support in {path}")
    for resolution in map(float, spatial.RESOLUTIONS):
        support = metadata.xs(("truth", resolution), level=("source", "resolution"))
        for source in SOURCES[1:]:
            pd.testing.assert_frame_equal(metadata.xs((source, resolution), level=("source", "resolution")), support)
    return measured, metadata.join(proportions).reset_index()


def main():
    args = panel_args("d", __doc__,
                      plot_data_help="Replay the archived deconvolution_spatial_plot_data.csv.gz.")
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
    tables = args.tables_dir or study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)
    if args.plot_data:
        measured, data = read_plot_data(args.plot_data, palette)
    else:
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
        measured.to_csv(tables / "d_reference_spots.csv.gz")
        data.to_csv(tables / "d_compositions.csv.gz")
    pd.Series(palette, name="color").rename_axis("cell_class").to_csv(tables / "d_palette.csv")

    xlim, ylim = spatial.common_limits(measured[["x", "y"]].to_numpy())
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
        "archived_plot_data": str(args.plot_data) if args.plot_data else None,
    }, indent=2) + "\n")

    fig = panel("d")
    ref = manuscript_axes(fig, "d", COLUMN_CENTERS[0] - 45, ROW_CENTERS[0] - 42, 90, 84)
    spatial.format_spatial_axis(ref, xlim, ylim)
    ref.scatter(measured.x, measured.y, c=measured.display_class.map(palette), s=0.43,
                linewidths=0, rasterized=True)
    manuscript_text(fig, "d", COLUMN_CENTERS[0], 330.5, "Single-cell\nMERFISH",
                    fontsize=9.7, ha="center", va="top", linespacing=0.95)
    key = manuscript_axes(fig, "d", 155, 451, 131, 89)
    draw_key(key, palette)
    for y, source in zip(ROW_CENTERS, SOURCES, strict=True):
        manuscript_text(fig, "d", 276.3, y, spatial.SOURCE_LABELS[source], rotation=90,
                        ha="center", va="center", fontsize=9.7)
        for x, resolution in zip(COLUMN_CENTERS[1:], ("0.25", "0.1", "roi"), strict=True):
            is_roi = resolution == "roi"
            width, height = (70, 76) if is_roi else (94, 82)
            axis = manuscript_axes(fig, "d", x - width / 2, y - height / 2, width, height)
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
    manuscript_text(fig, "d", 383, 334, "Resolution", ha="center", va="top", fontsize=9.7)
    for x, label in zip(COLUMN_CENTERS[1:3], ("×0.25", "×0.10"), strict=True):
        manuscript_text(fig, "d", x, 343.8, label, ha="center", va="baseline", fontsize=9.7)
    manuscript_text(fig, "d", COLUMN_CENTERS[3], 330.5, "Enlarged\nROI", ha="center",
                    va="top", fontsize=9.7, linespacing=0.95)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
