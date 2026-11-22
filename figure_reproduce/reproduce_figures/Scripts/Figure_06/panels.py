"""Atlas geometry, expression and ComFlow regional networks."""
import json

import matplotlib as mpl
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pandas as pd
from PIL import Image

from feast_plate import axes_box, limits, panel, panel_args, save, style_3d, text_at
from layout import FIGURE, FRAMES, TAG_LEFT
import network_style

ROOT = "results/07_3d_transfer"
AGES = ("E15.5", "E18.5")
GENES = ("Slc17a7", "Gad2", "Reln")
REGIONS = ("DorsalVZ", "IZ", "CP", "VentralVZ", "GE", "SeptalVZ", "Septum", "BT")
SUPPORT_STACK_ORDER = ("BT", "IZ", "GE", "CP", "DorsalVZ", "Septum", "VentralVZ", "SeptalVZ")
HEATMAP_ORDER = ("CP", "IZ", "DorsalVZ", "BT", "GE", "VentralVZ", "SeptalVZ", "Septum")
PANELS = ("A", "B", "C", "D")
NETWORK_TITLES = ("Measured E14", "E14 → DevCCF E15.5", "Measured E18", "E18 → DevCCF E18.5")
CCC_TITLE_TOP = 16
CCC_ROW_PITCH = 125


def region_colors(zenodo):
    tbl = pd.read_csv(zenodo / "inputs/07_3d_transfer/ccc/region_schema.tsv", sep="\t")
    return dict(zip(tbl.region_label, tbl.hex_color))


def centered_bounds(data, coordinates, common_bounds):
    """Center a view on its tissue while keeping the comparison's physical scale."""
    half_spans = np.ptp(common_bounds, axis=1) / 2
    centers = [(data[dim].min() + data[dim].max()) / 2 for dim in coordinates]
    return [(center - span, center + span) for center, span in zip(centers, half_spans)]


def image_crop(figure, path, box, anchors=None):
    with Image.open(path) as image:
        x0, y0, x1, y1 = image.getchannel("A").getbbox()
        width, height = image.size
        cropped = image.crop((x0, y0, x1, y1))
    x, top, w, h = box
    ax = axes_box(figure, x, top, w, h)
    ax.imshow(cropped, interpolation="antialiased")
    ax.set_axis_off()
    if anchors is not None:
        anchor_xy = {name: (x_norm * width - x0, (1-y_norm)*height - y0)
                     for name, (x_norm, y_norm) in anchors.items()}
        return ax, anchor_xy


def schematic(fig, zenodo):
    results = zenodo / ROOT
    art = results / "figure06_sources/artwork"
    record = json.loads((art / "provenance.json").read_text())
    colors = region_colors(zenodo)
    image_crop(fig, art / "coordinate_system.png", (7, 19, 62, 63))
    text_at(fig, 32.6, 85, "Coordinate\nSystem", ha="center", va="top", size=7.5)
    data = pd.read_parquet(results / "ccc_atlas/observed_st/GSM8323030_E18M_1_observed.parquet")
    axis = axes_box(fig, 78, 32, 53, 38)
    axis.scatter(data.x_um, data.y_um, color=data.region.astype(str).map(colors).fillna("#cccccc"),
                 s=.08, linewidths=0, rasterized=True)
    axis.set_aspect("equal")
    axis.set_axis_off()
    for i in (3, 2, 1):
        axis.add_patch(Rectangle((.04*i, .025*i), .87, .94, fill=False, lw=.35,
                                 edgecolor="#9ca9b4", transform=axis.transAxes, zorder=-1))
    text_at(fig, 102, 22, "External ST slices", ha="center", va="top", size=7.5)
    arrow = axes_box(fig, 75, 70, 60, 10)
    arrow.annotate("", (.98, .46), (.02, .46), arrowprops={"arrowstyle": "->", "lw": .65, "color": "#171717"})
    arrow.axis("off")
    text_at(fig, 102, 77, "FEAST", weight="bold", ha="center", va="top", size=8.5)
    text_at(fig, 102, 89, "Conditional Prediction", style="italic", ha="center", va="top", size=7)
    image_crop(fig, art / "translucent.png", (135, 20, 72, 57))
    cut, anchors = image_crop(fig, art / "cutaway.png", (230, 23, 56, 56), record["cutaway_callouts_axes_xy"])
    callouts = (
        (230, "right", (("DorsalVZ", 32), ("VentralVZ", 45), ("SeptalVZ", 59), ("Septum", 67))),
        (288, "left", (("CP", 37), ("IZ", 47), ("GE", 61), ("BT", 74))),
    )
    width, height = fig.get_size_inches() * 72
    for x, alignment, labels in callouts:
        for name, top in labels:
            cut.annotate(name, xy=anchors[name], xytext=(x / width, 1 - top / height),
                         textcoords=fig.transFigure, ha=alignment, va="center",
                         fontsize=4.5, color=colors[name], weight="bold",
                         arrowprops={"arrowstyle": "-", "lw": .27, "color": "#7b868a"}, annotation_clip=False)
    text_at(fig, 208, 6, "3D annotation", ha="center", va="top", size=8.5, weight="bold")
    text_at(fig, 224, 85, "3D Expression\nRebuild", ha="center", va="top", size=7.5)


def volume(fig, zenodo):
    data = zenodo / ROOT / "figure06_sources"
    info = json.loads((data / "provenance.json").read_text())
    samples = pd.read_parquet(data / "volume.parquet")
    bounds = limits(samples)
    # The journal plate uses one scale across all three genes and both ages.
    # Use the largest archived positive-percentile cap to retain the range of each gene.
    cap = max(info["display_caps"]["volume"][gene] for gene in GENES)
    norm = mpl.colors.Normalize(0, cap, clip=True)
    stages = [samples.loc[samples.age.eq(age)] for age in AGES]
    # Match the published isotropic crops for the two specimen rows.
    view_zooms = (1.77, 1.57)
    projected_centers = []
    for col, gene in enumerate(GENES):
        x = 15 + col*70
        text_at(fig, x+32, 3.3, gene, ha="center", va="top", weight="bold", size=7.5)
        for row, subset in enumerate(stages):
            axis = axes_box(fig, x, 6.8 + row*52, 64, 64, projection="3d")
            axis.scatter(subset.x, subset.y, subset.z, color="#c7ced4", s=.13, linewidths=0, alpha=.09,
                         rasterized=True, depthshade=False)
            positive = subset.loc[subset[gene].gt(0)].sort_values(gene, kind="stable")
            axis.scatter(positive.x, positive.y, positive.z, c=np.log1p(positive[gene]),
                         norm=norm, cmap="viridis", s=.12, alpha=.64, linewidths=0,
                         rasterized=True, depthshade=False)
            style_3d(axis, bounds, 23, -55)
            axis.set_box_aspect(np.ptp(bounds, axis=1), zoom=view_zooms[row])
            if col == 0:
                px, py, _ = proj3d.proj_transform(subset.x, subset.y, subset.z, axis.get_proj())
                projected_centers.append(((px.min() + px.max()) / 2, (py.min() + py.max()) / 2))
            # Pan the projected view without moving the anatomical coordinates or camera.
            px, py = projected_centers[row]
            half_width, half_height = axis.viewLim.width / 2, axis.viewLim.height / 2
            axis.viewLim.intervalx = (px - half_width, px + half_width)
            axis.viewLim.intervaly = (py - half_height, py + half_height)
    for index, age in enumerate(AGES):
        text_at(fig, 4.5, 49+52*index, age, rotation=90, ha="center", va="center", weight="bold", size=7)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
                       cax=axes_box(fig, 230.5, 11.8, 5.3, 104.8),
                       ticks=np.arange(0, cap + 1e-9, .5), format="%.1f")
    bar.outline.set_linewidth(.5)
    bar.ax.tick_params(labelsize=5.5, pad=1.5, length=1.6, width=.45)
    text_at(fig, 265, 66, "log1p counts", rotation=90, ha="center", va="center", size=6.25)


def expression_map(axis, data, coords, gene, cap, bounds):
    x, y = coords
    axis.scatter(data[x], data[y], s=.46, c="#dedede", marker="s", linewidths=0, rasterized=True)
    positive = data.loc[data[gene].gt(0)].sort_values(gene, kind="stable")
    axis.scatter(positive[x], positive[y], s=.60, c=np.log1p(positive[gene]), cmap="magma",
                 norm=mpl.colors.Normalize(0, cap, clip=True), marker="s", linewidths=0, rasterized=True)
    axis.set(xlim=bounds[0], ylim=bounds[1], aspect="equal")
    axis.set_axis_off()


def sections(fig, zenodo):
    root = zenodo / ROOT / "figure06_sources"
    colors = region_colors(zenodo)
    coverage = pd.read_csv(root / "region_coverage.csv")
    data = pd.read_parquet(root / "axial.parquet")
    levels = pd.read_csv(root / "levels.csv")
    caps = json.loads((root / "provenance.json").read_text())["display_caps"]["axial"]
    handles = [Patch(facecolor=colors[region], label=region) for region in REGIONS]
    fig.legend(handles=handles, ncols=8, loc="upper center", bbox_to_anchor=(.523, .975),
               columnspacing=1.06, handlelength=.95, handleheight=.7, handletextpad=.28, fontsize=6.25)
    xy = limits(data, .017)[:2]
    for i, age in enumerate(AGES):
        z = levels.loc[levels.age.eq(age) & levels.peak, "z_world"].item()
        table = coverage.loc[coverage.age.eq(age)].pivot(index="z_world", columns="region", values="n_spots").fillna(0)
        table = table.loc[:, list(REGIONS)].sort_index()
        top = 35 + 91*i
        ax = axes_box(fig, 33.5, top, 122.27, 65)
        values = table.sum(axis=1)
        order = list(SUPPORT_STACK_ORDER)
        ax.stackplot(table.index, *table[order].T.to_numpy(), colors=[colors[r] for r in order], linewidth=0)
        ax.plot(table.index, values, color="#333333", lw=.45)
        ax.axvline(z, color="#151515", lw=.55, linestyle=(0, (2, 2)))
        ax.scatter(z, values.loc[z], color="#151515", marker="D", s=7, linewidths=0, zorder=3)
        ax.annotate(f"displayed section\nz = {z:.2f}", xy=(z, values.loc[z]), xytext=(.99, .85),
                    textcoords="axes fraction", va="top", ha="right", fontsize=5.5, color="#555555",
                    arrowprops={"arrowstyle": "-", "lw": .45, "color": "#454545"})
        ax.set(xlim=(table.index.min(), table.index.max()), ylim=(0, 1.06*values.max()))
        ax.tick_params(axis="both", labelsize=5.5, length=1.3, pad=2.3)
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_major_locator(MultipleLocator(5000))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda n, _: "0" if n == 0 else f"{n/1e3:g}k"))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", lw=.3, zorder=-1)
        ax.spines[["top", "right"]].set_visible(False)
        text_at(fig, 33.5, top-15.3, f"{age} regional support", size=7, weight="bold", va="top")
        section = data.loc[data.age.eq(age)]
        section_bounds = centered_bounds(section, "xy", xy)
        for j, gene in enumerate(GENES):
            x = 159.1+j*57
            if i == 0:
                text_at(fig, x+27.5, 19.3, gene, ha="center", va="top", size=7.5, weight="bold")
            expression_map(axes_box(fig, x, top, 55, 65), section, ("x", "y"), gene, caps[gene], section_bounds)
    text_at(fig, 5.6, 112, "Blueprint spots per z level", size=6.25, rotation=90, ha="center", va="center")
    text_at(fig, 94.6, 207.3, "DevCCF z coordinate", size=6.25, ha="center", va="top")


def multiplane(fig, zenodo):
    root = zenodo / ROOT / "figure06_sources"
    record = json.loads((root / "provenance.json").read_text())
    sample = pd.read_parquet(root / "volume.parquet").query("age == 'E15.5'")
    planes = pd.read_parquet(root / "planes.parquet")
    # The common extent and color pools include the two undisplayed planes and both ages.
    extent = np.abs(planes[["u", "v"]].to_numpy()).max() * 1.04
    bounds = limits(sample, .035)
    caps = record["display_caps"]["planes"]
    early = record["ages"][0]
    for row, name in enumerate(("XZ", "YZ", "Oblique XZ 45°")):
        top = 33 + row*59
        selected = planes.loc[planes.age.eq("E15.5") & planes.plane.eq(name)]
        parameters = next(p for p in early["planes"] if p["plane"] == name)
        local = axes_box(fig, 1, top, 58, 58, projection="3d")
        local.scatter(sample.x, sample.y, sample.z, s=.095, c="#939ca5",
                      alpha=.07, linewidths=0, rasterized=True, depthshade=False)
        u, v = np.meshgrid([selected.u.min(), selected.u.max()], [selected.v.min(), selected.v.max()])
        surface = np.asarray(parameters["center"]) + u[..., None]*parameters["horizontal"] + v[..., None]*parameters["vertical"]
        local.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], color="#1882b3",
                           alpha=.35, linewidth=0, shade=False)
        style_3d(local, bounds, 25, -55, axes=True)
        local.set_box_aspect(np.ptp(bounds, axis=1), zoom=1.02)
        plane_bounds = centered_bounds(selected, "uv", [(-extent, extent)] * 2)
        for col, gene in enumerate(GENES):
            x = 65.6 + col*60
            if row == 0:
                text_at(fig, x+24, 19.3, gene, weight="bold", size=7.5, ha="center", va="top")
            expression_map(axes_box(fig, x, top, 48, 58), selected, ("u", "v"), gene,
                           caps[gene], plane_bounds)


def networks(fig, zenodo):
    root = zenodo / ROOT / "ccc_figure5"
    nodes = pd.read_csv(root / "regional_edges.csv")
    colors = region_colors(zenodo)
    pos = network_style.layout(REGIONS)
    labels = (
        ("DorsalVZ", 0, -51, "center"), ("IZ", 44, -35, "center"), ("CP", 49, 0, "left"),
        ("VentralVZ", 42, 43, "center"), ("GE", 0, 49, "center"),
        ("SeptalVZ", -42, 43, "center"), ("Septum", -46, 0, "right"), ("BT", -44, -35, "center"),
    )
    for i, letter in enumerate(PANELS):
        row, col = divmod(i, 2)
        x, y = 72.6 + 140*col, 87.4 + CCC_ROW_PITCH*row
        text_at(fig, x, CCC_TITLE_TOP + CCC_ROW_PITCH*row, NETWORK_TITLES[i],
                weight="bold", size=7, ha="center", va="top")
        ax = axes_box(fig, x-46, y-46, 92, 92)
        network_style.network(ax, nodes.loc[nodes.panel.eq(letter)], REGIONS, colors, pos, "balanced", labels=False)
        ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
        for region, dx, dy, align in labels:
            text_at(fig, x+dx, y+dy, region, size=6.5, color="#37444a", ha=align, va="center")
    key = axes_box(fig, 0, 273, 287, 24)
    for index, pct in enumerate((.1, 1, 10, 40)):
        xpos = 51.6 + 63*index
        key.plot([xpos-12, xpos+12], [9, 9], color="#65757b",
                 linewidth=network_style.edge_width(pct/100, "balanced"), solid_capstyle="butt")
        key.text(xpos, 0, f"{pct:g}%", size=6.5, ha="center", va="center")
    key.set(xlim=(0, 287), ylim=(-3, 21))
    key.axis("off")


def allocations(fig, zenodo):
    source = pd.read_csv(zenodo / ROOT / "ccc_figure5/regional_edges.csv")
    masking = np.eye(len(HEATMAP_ORDER), dtype=bool)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("regional_share", ["#ffffff", "#e9a6a1", "#b64342"])
    cmap.set_bad("#e5e8ea")
    norm = mpl.colors.Normalize(0, 2)
    for i, letter in enumerate(PANELS):
        row, col = divmod(i, 2)
        x, y, size = 41.6 + 140*col, 38.8 + CCC_ROW_PITCH*row, 98.28
        values = source.loc[source.panel.eq(letter)].pivot(
            index="sender_region", columns="receiver_region", values="share_percent").loc[HEATMAP_ORDER, HEATMAP_ORDER]
        axis = axes_box(fig, x, y, size, size)
        shown = np.ma.masked_array(values.to_numpy(), mask=masking)
        axis.pcolormesh(np.arange(9), np.arange(9), shown, cmap=cmap, norm=norm,
                        linewidth=.35, edgecolors="#eeeeee")
        axis.set(xlim=(0, 8), ylim=(8, 0), aspect="equal", yticks=np.arange(8)+.5)
        axis.set_yticklabels(HEATMAP_ORDER, fontsize=5.5)
        axis.set_xticks(np.arange(8)+.5 if row == 1 else [])
        if row == 1:
            axis.set_xticklabels(HEATMAP_ORDER, rotation=50, ha="right", fontsize=5.5, rotation_mode="anchor")
        axis.tick_params(length=0, pad=2)
        axis.spines[:].set_visible(False)
        text_at(fig, x+size/2, CCC_TITLE_TOP + CCC_ROW_PITCH*row, NETWORK_TITLES[i],
                weight="bold", size=7, ha="center", va="top")
    text_at(fig, 4.8, 150.7, "Sending region", rotation=90, ha="center", va="center", size=6.25)
    text_at(fig, 160.7, 301.1, "Receiving region", ha="center", va="bottom", size=6.25)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                           cax=axes_box(fig, 288.6, 59.8, 5, 180), ticks=np.arange(0, 2.01, .25), format="%.2f")
    colorbar.outline.set_linewidth(.4)
    colorbar.ax.tick_params(labelsize=5.5, pad=1.3, length=1.2, width=.4)
    text_at(fig, 293, 47.3, "Share (%)", size=6.25, ha="center", va="top")


PLOTS = {"a": schematic, "b": volume, "c": sections, "d": multiplane, "e": networks, "f": allocations}


def main(letter):
    plot = PLOTS[letter]
    args = panel_args(FIGURE, letter, plot.__name__).parse_args()
    fig = panel(letter, FRAMES)
    fig.texts[0].set_x(TAG_LEFT[letter] / FRAMES[letter][2])
    # The compact plate lets the row boxes meet where only annotations extend.
    with mpl.rc_context({"savefig.transparent": True, "text.color": "#111111"}):
        plot(fig, args.zenodo)
        save(fig, args.output)
