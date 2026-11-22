"""Render journal Figure 2b and save the full spot joins and display selections."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from layout import A_HEIGHT, B_HEIGHT, FIGURES, HEIGHT, INPUTS, PROJECT, RESULTS, WIDTH, configure_style, rect
from plot_slice_panel import PANEL_SPECS, SOURCES, SOURCE_LABELS, h5ad_path, read_spatial_marker
from plot_spatial_comparison import (
    DELTA_CMAP,
    DELTA_POINT_CAP,
    DELTA_SOURCES,
    EXPRESSION_POINT_CAP,
    _point_size,
    _sample_indices,
    paired_residual,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

STEM = "simulator_spatial_comparison"
COLUMN_CENTERS = (89.1, 179.7, 271.22, 361.05, 453.3, 544.4)
ROW_CENTERS = (332.2, 400.7, 469.2, 537.7, 614.8, 683.3, 751.8)
TITLES = (
    "Human DLPFC\n(10X Visium)",
    "Mouse Brain\n(MERFISH)",
    "Human metastatic\nlymph node\n(Open-ST)",
    "Human melanoma\nbrain metastasis\n(Slide-seq V2)",
    "Mouse Embryo\n(Stereo-seq)",
    "Human Lymph node\n(Xenium)",
)
EXPRESSION_NORM = Normalize(0, 1)
RESIDUAL_NORM = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


def text_at(figure, x, top, label, **kwargs):
    figure.text(x / WIDTH, (HEIGHT - top) / B_HEIGHT, label, **kwargs)


def figure_rect(x, top, width, height):
    return rect(x, top - (HEIGHT - B_HEIGHT), width, height, B_HEIGHT)


def display_spots(sample, gene, view, source, data, values, indices):
    xy = data.obsm["spatial"][indices]
    return pd.DataFrame({
        "sample": sample,
        "gene": gene,
        "view": view,
        "source": source,
        "draw_order": np.arange(len(xy)),
        "source_row": np.arange(data.n_obs)[indices],
        "spot_id": data.obs_names[indices],
        "x": xy[:, 0],
        "y": xy[:, 1],
        "log1p_value": values[indices],
    })


def draw_map(figure, column, row, points, scale, size, view, limits):
    ax = figure.add_axes(figure_rect(COLUMN_CENTERS[column] - 42, ROW_CENTERS[row] - 32.8, 84, 65.6))
    expression = view == "expression"
    ax.scatter(
        points["x"], points["y"], c=points["log1p_value"] / scale, s=size,
        cmap="viridis" if expression else DELTA_CMAP,
        norm=EXPRESSION_NORM if expression else RESIDUAL_NORM,
        marker="." if expression else "o", linewidths=0,
        alpha=1 if expression else 0.85, rasterized=True,
    )
    ax.set(xlim=limits[0], ylim=limits[1][::-1], aspect="equal")
    ax.axis("off")


def draw_keys(figure):
    for top, norm, cmap, label, ticks in (
        (237, EXPRESSION_NORM, "viridis", "Expression: log1p(value) / column q99.5 across all sources", np.linspace(0, 1, 6)),
        (803.6, RESIDUAL_NORM, DELTA_CMAP, "Error: [log1p(simulated) − log1p(reference)] / column q99(|Δ|)", np.linspace(-1, 1, 5)),
    ):
        center = 316.5
        axis = figure.add_axes(figure_rect(center - 150, top, 300, 4.5))
        bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axis, orientation="horizontal", ticks=ticks)
        bar.outline.set_linewidth(0.35)
        axis.tick_params(labelsize=5.6, length=1.5, width=0.35, pad=0.6)
        text_at(figure, center, top - 3.5, label, ha="center", va="bottom", fontsize=6.1)


def draw_header(figure, column, gene, residual_max):
    center = COLUMN_CENTERS[column]
    for line, baseline in zip(reversed(TITLES[column].splitlines()), (279.0, 269.8, 260.6)):
        text_at(figure, center, baseline, line, ha="center", va="baseline", fontsize=6.9)
    figure.add_artist(plt.Line2D(
        [(center - 31) / WIDTH, (center + 31) / WIDTH],
        [(HEIGHT - 283.9) / B_HEIGHT] * 2, transform=figure.transFigure,
        color="#444444", linewidth=0.35,
    ))
    text_at(figure, center, 291.4, f"{gene}  Δ q99 ±{residual_max:.3g}", ha="center",
            va="baseline", fontsize=6.42)


def labeled_figure():
    figure = plt.figure(figsize=(WIDTH / 72, B_HEIGHT / 72))
    text_at(figure, 6.9, 269.2, "b", va="baseline", fontsize=13.1, fontweight="bold")
    labels = [*SOURCE_LABELS, *(f"{source} Diff" for source in DELTA_SOURCES)]
    for top, label in zip(ROW_CENTERS, labels, strict=True):
        text_at(figure, 28.6, top, label, rotation=90, ha="center", va="center",
                fontsize=10, fontweight="bold", linespacing=1.0)
    draw_keys(figure)
    return figure


def render_archive(archived_pdf, archived_dir, output_dir):
    """Replay the independently verified Study 00 export with journal annotations."""
    import pymupdf

    configure_style()
    scale_path = archived_dir / f"{STEM}_color_limits.csv"
    scales = pd.read_csv(scale_path).set_index(["sample", "gene"], verify_integrity=True)
    if len(scales) != len(PANEL_SPECS) or not np.array_equal(scales.residual_vmin, -scales.residual_vmax):
        raise ValueError(f"Invalid archived color scales in {scale_path}")
    figure = labeled_figure()
    for column, spec in enumerate(PANEL_SPECS):
        draw_header(figure, column, spec["gene"], scales.loc[(spec["sample"], spec["gene"]), "residual_vmax"])

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as scratch:
        base = Path(scratch) / "base.pdf"
        result = Path(scratch) / "b.pdf"
        figure.savefig(base, dpi=600, metadata={"Title": "Figure 2b"})
        plt.close(figure)
        with pymupdf.open(archived_pdf) as source, pymupdf.open(base) as output:
            if len(source) != 1:
                raise ValueError(f"Expected one page in {archived_pdf}")
            tiles = source[0].get_image_info(xrefs=True)
            rows = len(ROW_CENTERS)
            if len(tiles) != rows * len(PANEL_SPECS) + 2:
                raise ValueError(f"Expected 42 map tiles and two color bars in {archived_pdf}")
            maps = tiles[:-2]
            first_top = min(tile["bbox"][1] for tile in maps)
            output_page = output[0]
            # The verified source has a column-major map sequence and a regular row pitch.
            factor = 0.520
            previous_center = float("-inf")
            for column, target_center in enumerate(COLUMN_CENTERS):
                column_tiles = maps[column * rows:(column + 1) * rows]
                centers = [(tile["bbox"][0] + tile["bbox"][2]) / 2 for tile in column_tiles]
                starts = [tile["bbox"][1] for tile in column_tiles]
                if (centers[0] <= previous_center or np.ptp(centers) >= 1
                        or not np.all(np.diff(starts) > 0)):
                    raise ValueError(f"Unrecognized map order in {archived_pdf}")
                previous_center = centers[0]
                for tile in column_tiles:
                    x0, y0, x1, y1 = tile["bbox"]
                    box = pymupdf.Rect(
                        target_center + (x0 - centers[0]) * factor,
                        300.8 - A_HEIGHT + (y0 - first_top) * factor,
                        target_center + (x1 - centers[0]) * factor,
                        300.8 - A_HEIGHT + (y1 - first_top) * factor,
                    )
                    image = source.extract_image(tile["xref"])
                    alpha = source.extract_image(image["smask"])["image"] if image["smask"] else None
                    inserted = output_page.insert_image(box, stream=image["image"], mask=alpha)
                    if alpha is not None:
                        _, mask_ref = output.xref_get_key(inserted, "SMask")
                        # PDF soft masks require DeviceGray, including PNGs with an ICC profile.
                        output.xref_set_key(int(mask_ref.split()[0]), "ColorSpace", "/DeviceGray")
            output.save(result, garbage=4, deflate=True)
        result.replace(output_dir / "b.pdf")


def project_path(path):
    path = path.resolve()
    return str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)


def render(archived_dir, source_dirs, results_dir, output_dir):
    configure_style()
    archived_residuals = pd.read_csv(
        archived_dir / f"{STEM}_residuals.csv.gz", dtype={"spot_id": str},
        float_precision="round_trip",
    ).set_index(["sample", "gene", "spot_id"], verify_integrity=True).sort_index()
    if len(archived_residuals) != 500_956:
        raise ValueError("Expected 500,956 reference-ordered archival spatial rows")
    archived_matches = pd.read_csv(
        archived_dir / f"{STEM}_spot_matching.csv", float_precision="round_trip",
    ).set_index(["sample", "gene", "source"], verify_integrity=True)

    figure = labeled_figure()
    residual_frames, displays, scales, matches = [], [], [], []
    for column, spec in enumerate(PANEL_SPECS):
        sample, gene, platform = spec["sample"], spec["gene"], spec["platform"]
        paths = {source: source_dirs[source] / h5ad_path(source, sample).name for source in SOURCES}
        markers = {source: read_spatial_marker(path, gene) for source, path in paths.items()}
        expressions = {source: np.log1p(marker.X[:, 0]) for source, marker in markers.items()}
        expression_max = float(np.percentile(np.concatenate(list(expressions.values())), 99.5))
        reference = markers["Reference"]
        xy = reference.obsm["spatial"]
        archived_section = archived_residuals.loc[(sample, gene)]
        if len(archived_section) != reference.n_obs or not reference.obs_names.isin(archived_section.index).all():
            raise ValueError(f"Archived spatial support differs for {sample} / {gene}")
        archived_section = archived_section.loc[reference.obs_names]
        np.testing.assert_array_equal(xy, archived_section[["x", "y"]].to_numpy())
        np.testing.assert_allclose(expressions["Reference"], archived_section["reference_log1p"], rtol=0, atol=1e-12)

        residuals = {}
        for source in DELTA_SOURCES:
            values, match = paired_residual(reference, markers[source], paths[source])
            if source != "FEAST":
                # Confirm the deposited values by spot ID, then retain their original precision.
                old_values = archived_section[f"{source}_log1p_residual"].to_numpy()
                np.testing.assert_allclose(values, old_values, rtol=0, atol=1e-12, err_msg=f"{sample} / {source}")
                old_match = archived_matches.loc[(sample, gene, source)]
                for key in ("spot_pairing", "paired_spots", "spots_reordered"):
                    if old_match[key] != match[key]:
                        raise ValueError(f"Retained spot diagnostic {key} differs for {sample} / {source}")
                np.testing.assert_allclose(
                    match["matched_coordinate_max_abs_difference"],
                    old_match["matched_coordinate_max_abs_difference"], rtol=0, atol=1e-12,
                )
                values = old_values
            residuals[source] = values
            matches.append({
                "sample": sample, "gene": gene, "source": source,
                "reference_path": project_path(paths["Reference"]),
                "simulation_path": project_path(paths[source]), **match,
            })

        residual_max = float(np.percentile(np.abs(np.concatenate(list(residuals.values()))), 99))
        if expression_max <= 0 or residual_max <= 0:
            raise ValueError(f"{sample} / {gene} has a degenerate color scale")
        scales.append({
            "sample": sample, "gene": gene, "expression_vmax": expression_max,
            "residual_vmin": -residual_max, "residual_vmax": residual_max,
        })
        residual_frames.append(pd.DataFrame({
            "sample": sample, "gene": gene, "spot_id": reference.obs_names, "x": xy[:, 0], "y": xy[:, 1],
            "reference_log1p": expressions["Reference"],
            **{f"{source}_log1p_residual": values for source, values in residuals.items()},
        }))

        padding = 0.025 * np.ptp(xy, axis=0)
        limits = list(zip(xy.min(axis=0) - padding, xy.max(axis=0) + padding, strict=True))
        for row, source in enumerate(SOURCES):
            marker = markers[source]
            indices = _sample_indices(marker.n_obs, EXPRESSION_POINT_CAP)
            points = display_spots(sample, gene, "expression", source, marker, expressions[source], indices)
            displays.append(points)
            size = _point_size(marker.n_obs, platform) * (WIDTH / 1158.89) ** 2
            draw_map(figure, column, row, points, expression_max, size, "expression", limits)

        visible = np.arange(reference.n_obs)[_sample_indices(reference.n_obs, DELTA_POINT_CAP)]
        for row, source in enumerate(DELTA_SOURCES, start=len(SOURCES)):
            values = residuals[source]
            indices = visible[np.argsort(np.abs(values[visible]))]
            points = display_spots(sample, gene, "residual", source, reference, values, indices)
            displays.append(points)
            size = _point_size(reference.n_obs, platform) * (WIDTH / 1158.89) ** 2
            draw_map(figure, column, row, points, residual_max, size, "residual", limits)

        draw_header(figure, column, gene, residual_max)
        print(f"Figure 2b: {sample} / {gene}, {reference.n_obs:,} joined spots", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_names = {
        "residuals": f"{STEM}_residuals.csv.gz",
        "color_limits": f"{STEM}_color_limits.csv",
        "spot_matching": f"{STEM}_spot_matching.csv",
        "displayed_spots": f"{STEM}_displayed_spots.csv.gz",
    }
    residual_table = pd.concat(residual_frames, ignore_index=True)
    display_table = pd.concat(displays, ignore_index=True)
    residual_table.to_csv(results_dir / table_names["residuals"], index=False)
    pd.DataFrame(scales).to_csv(results_dir / table_names["color_limits"], index=False)
    pd.DataFrame(matches).to_csv(results_dir / table_names["spot_matching"], index=False)
    display_table.to_csv(results_dir / table_names["displayed_spots"], index=False)
    figure.savefig(output_dir / "b.pdf", dpi=600, metadata={"Title": "Figure 2b"})
    plt.close(figure)

    provenance = {
        "figure": "Figure 2b",
        "paths_relative_to": "Final_reproduce",
        "input_directories": {source: project_path(path) for source, path in source_dirs.items()},
        "archived_spot_tables": project_path(archived_dir),
        "complete_pairing": "Exact unique observation IDs, or a complete unique coordinate bijection at eight decimals for disjoint IDs.",
        "retained_comparator_values": "Verified from staged markers against the archived residuals by sample, gene and spot ID.",
        "expression_vmax": "Pooled log1p expression 99.5th percentile over all four sources.",
        "residual_limits": "Pooled absolute log1p differences 99th percentile over all three simulated sources, applied symmetrically.",
        "display": {"seed": 2026, "expression_cap": EXPRESSION_POINT_CAP, "residual_cap": DELTA_POINT_CAP},
        "full_residual_rows": len(residual_table),
        "displayed_points": len(display_table),
        "match_records": len(matches),
        "tables": table_names,
    }
    (results_dir / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=INPUTS / "reference")
    parser.add_argument("--feast-dir", type=Path, default=RESULTS / "run/simulations")
    parser.add_argument("--srtsim-dir", type=Path, default=INPUTS / "external/SRTsim")
    parser.add_argument("--sccube-dir", type=Path, default=INPUTS / "external/scCube")
    parser.add_argument("--archived-dir", type=Path, default=RESULTS / "article/figure2")
    parser.add_argument("--archived-pdf", type=Path,
                        help="Replay map tiles from the verified simulator_spatial_comparison.pdf (requires PyMuPDF).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS / "figure2")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    if args.archived_pdf:
        render_archive(args.archived_pdf, args.archived_dir, args.output_dir)
    else:
        source_dirs = dict(zip(SOURCES, [args.reference_dir, args.feast_dir, args.srtsim_dir, args.sccube_dir], strict=True))
        render(args.archived_dir, source_dirs, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
