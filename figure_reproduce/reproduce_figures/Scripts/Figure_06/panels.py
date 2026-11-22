"""Atlas geometry, expression and ComFlow regional networks."""
import json

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image

from feast_plate import axes_box, limits, panel, panel_args, save, style_3d, text_at
from layout import FIGURE, FRAMES
import network_style

ROOT = "results/07_3d_transfer"
AGES = ("E15.5", "E18.5")
GENES = ("Slc17a7", "Gad2", "Reln")
REGIONS = ("DorsalVZ", "IZ", "CP", "VentralVZ", "GE", "SeptalVZ", "Septum", "BT")
HEATMAP_ORDER = ("CP", "IZ", "DorsalVZ", "BT", "GE", "VentralVZ", "SeptalVZ", "Septum")
PANELS = ("A", "B", "C", "D")
NETWORK_TITLES = ("Measured E14", "E14 → DevCCF E15.5", "Measured E18", "E18 → DevCCF E18.5")


def region_colors(zenodo):
    tbl = pd.read_csv(zenodo / "inputs/07_3d_transfer/ccc/region_schema.tsv", sep="\t")
    return dict(zip(tbl.region_label, tbl.hex_color))


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
    image_crop(fig, art / "coordinate_system.png", (7, 18, 65, 64))
    text_at(fig, 38, 86, "Coordinate\nSystem", ha="center", va="top", size=7.6)
    data = pd.read_parquet(results / "ccc_atlas/observed_st/GSM8323030_E18M_1_observed.parquet")
    axis = axes_box(fig, 75, 31, 55, 41)
    axis.scatter(data.x_um, data.y_um, color=data.region.astype(str).map(colors).fillna("#cccccc"),
                 s=.08, linewidths=0, rasterized=True)
    axis.set_aspect("equal")
    axis.set_axis_off()
    for i in (3, 2, 1):
        axis.add_patch(Rectangle((.04*i, .025*i), .87, .94, fill=False, lw=.35,
                                 edgecolor="#9ca9b4", transform=axis.transAxes, zorder=-1))
    text_at(fig, 105, 21, "External ST slices", ha="center", va="top", size=7.6)
    arrow = axes_box(fig, 73, 70, 61, 13)
    arrow.annotate("", (.98, .42), (.02, .42), arrowprops={"arrowstyle": "->", "lw": .65, "color": "#171717"})
    arrow.axis("off")
    text_at(fig, 103, 80, "FEAST", weight="bold", ha="center", va="top", size=8.4)
    text_at(fig, 103, 93, "Conditional Prediction", style="italic", ha="center", va="top", size=7.1)
    image_crop(fig, art / "translucent.png", (132, 20, 77, 58))
    cut, anchors = image_crop(fig, art / "cutaway.png", (226, 20, 60, 58), record["cutaway_callouts_axes_xy"])
    sides = (("DorsalVZ", "VentralVZ", "SeptalVZ", "Septum"), ("CP", "IZ", "GE", "BT"))
    for left, names in zip((True, False), sides):
        for i, name in enumerate(names):
            cut.annotate(name, xy=anchors[name], xytext=(-.065 if left else 1.065, .80 - .195*i),
                         textcoords="axes fraction", ha="right" if left else "left", va="center",
                         fontsize=4.9, color=colors[name], weight="bold",
                         arrowprops={"arrowstyle": "-", "lw": .27, "color": "#7b868a"}, annotation_clip=False)
    text_at(fig, 211, 8, "3D annotation", ha="center", va="top", size=8.7, weight="bold")
    text_at(fig, 229, 86, "3D Expression\nRebuild", ha="center", va="top", size=7.5)


def volume(fig, zenodo):
    data = zenodo / ROOT / "figure06_sources"
    info = json.loads((data / "provenance.json").read_text())
    samples = pd.read_parquet(data / "volume.parquet")
    bounds = limits(samples)
    for col, gene in enumerate(GENES):
        x = 15 + col*76.5
        text_at(fig, x+36, 3, gene, ha="center", va="top", weight="bold", size=7.7)
        cap = info["display_caps"]["volume"][gene]
        norm = mpl.colors.Normalize(0, cap, clip=True)
        for row, age in enumerate(AGES):
            axis = axes_box(fig, x, 12 + row*44, 74, 47, projection="3d")
            subset = samples.loc[samples.age.eq(age)]
            axis.scatter(subset.x, subset.y, subset.z, color="#c7ced4", s=.13, linewidths=0, alpha=.09,
                         rasterized=True, depthshade=False)
            positive = subset.loc[subset[gene].gt(0)].sort_values(gene, kind="stable")
            axis.scatter(positive.x, positive.y, positive.z, c=np.log1p(positive[gene]),
                         norm=norm, cmap="viridis", s=.12, alpha=.64, linewidths=0,
                         rasterized=True, depthshade=False)
            style_3d(axis, bounds, 23, -55)
            axis.set_box_aspect(np.ptp(bounds, axis=1), zoom=1.62)
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), orientation="horizontal",
                          cax=axes_box(fig, x + 10, 104, 51, 3.4), ticks=[0, cap])
        bar.set_ticklabels(["0", f"{cap:.2f}"])
        bar.ax.tick_params(labelsize=5.4, pad=.5, length=1)
    for index, age in enumerate(AGES):
        text_at(fig, 10, 38+44*index, age, rotation=90, ha="center", va="center", weight="bold", size=7)
    text_at(fig, 251, 68, "log1p counts", rotation=90, ha="center", va="center", size=6.5)


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
    fig.legend(handles=handles, ncols=8, loc="upper center", bbox_to_anchor=(.53, .998),
               columnspacing=.84, handlelength=.95, handleheight=.7, handletextpad=.28, fontsize=6.1)
    xy = limits(data)[:2]
    for i, age in enumerate(AGES):
        z = levels.loc[levels.age.eq(age) & levels.peak, "z_world"].item()
        table = coverage.loc[coverage.age.eq(age)].pivot(index="z_world", columns="region", values="n_spots").fillna(0)
        table = table.loc[:, list(REGIONS)].sort_index()
        top = 31 + 92*i
        ax = axes_box(fig, 34, top, 121, 65.5)
        values = table.sum(axis=1)
        order = list(reversed(REGIONS))
        ax.stackplot(table.index, *table[order].T.to_numpy(), colors=[colors[r] for r in order], linewidth=0)
        ax.plot(table.index, values, color="#333333", lw=.6)
        ax.axvline(z, color="#404040", lw=.65, linestyle=(0,(2,2)))
        x = table.index.max() - .07
        ax.annotate(f"displayed section\nz = {z:.2f} mm", xy=(z, values.loc[z]), xytext=(x, values.max()*.66),
                    ha="right", fontsize=5.8, color="#343b40",
                    arrowprops={"arrowstyle": "-", "lw": .45, "color": "#454545"})
        ax.set(xlim=(table.index.min(), table.index.max()), ylim=(0, 1.06*values.max()))
        ax.tick_params(axis="both", labelsize=5.6, length=1.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda n, _: "0" if n == 0 else f"{n/1e3:g}k"))
        ax.grid(axis="y", color="#dddddd", lw=.35, zorder=-1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{age} regional support", size=7.5, pad=2, weight="bold", loc="left")
        section = data.loc[data.age.eq(age)]
        for j, gene in enumerate(GENES):
            x = 162+j*54
            if i == 0:
                text_at(fig, x+26, 19, gene, ha="center", va="top", size=7.5, weight="bold")
            expression_map(axes_box(fig, x, top+1, 52, 62), section, ("x", "y"), gene, caps[gene], xy)
    text_at(fig, 7, 99, "Blueprint spots per z level", size=6.7, rotation=90, ha="center", va="center")
    text_at(fig, 95, 204, "DevCCF z (mm)", size=6.7, ha="center", va="bottom")


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
        top = 29 + row*57
        selected = planes.loc[planes.age.eq("E15.5") & planes.plane.eq(name)]
        parameters = next(p for p in early["planes"] if p["plane"] == name)
        local = axes_box(fig, 5, top, 55, 54, projection="3d")
        local.scatter(sample.x, sample.y, sample.z, s=.095, c="#939ca5",
                      alpha=.07, linewidths=0, rasterized=True, depthshade=False)
        u, v = np.meshgrid([selected.u.min(), selected.u.max()], [selected.v.min(), selected.v.max()])
        surface = np.asarray(parameters["center"]) + u[..., None]*parameters["horizontal"] + v[..., None]*parameters["vertical"]
        local.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], color="#1882b3",
                           alpha=.35, linewidth=0, shade=False)
        style_3d(local, bounds, 25, -55, axes=True)
        local.set_box_aspect(np.ptp(bounds, axis=1), zoom=1.12)
        for col, gene in enumerate(GENES):
            x = 68 + col*63
            if row == 0:
                text_at(fig, x+24, 17, gene, weight="bold", size=7.5, ha="center", va="top")
            expression_map(axes_box(fig, x, top, 51, 53), selected, ("u", "v"), gene,
                           caps[gene], [(-extent, extent)]*2)


def networks(fig, zenodo):
    root = zenodo / ROOT / "ccc_figure5"
    nodes = pd.read_csv(root / "regional_edges.csv")
    colors = region_colors(zenodo)
    pos = network_style.layout(REGIONS)
    for i, letter in enumerate(PANELS):
        row, col = divmod(i, 2)
        left, top = 15 + 139*col, 23 + 126*row
        text_at(fig, left+56, top-10, NETWORK_TITLES[i], weight="bold", size=7.6, ha="center", va="top")
        ax = axes_box(fig, left, top, 112, 109)
        network_style.network(ax, nodes.loc[nodes.panel.eq(letter)], REGIONS, colors, pos, "balanced")
        ax.set(xlim=(-1.8, 1.8), ylim=(-1.72,1.72))
    key = axes_box(fig, 33, 271, 224, 23)
    for index, pct in enumerate((.1, 1, 10, 40)):
        xpos = .10 + .265*index
        key.plot([xpos-.052, xpos+.052], [.59, .59], color="#65757b",
                 linewidth=network_style.edge_width(pct/100, "balanced"), solid_capstyle="butt")
        key.text(xpos, .18, f"{pct:g}%", size=7, ha="center", va="center")
    key.set(xlim=(0, 1), ylim=(0,1))
    key.axis("off")


def allocations(fig, zenodo):
    source = pd.read_csv(zenodo / ROOT / "ccc_figure5/regional_edges.csv")
    masking = np.eye(len(HEATMAP_ORDER), dtype=bool)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("regional_share", ["#ffffff", "#e9a6a1", "#b64342"])
    cmap.set_bad("#e1e5e7")
    norm = mpl.colors.Normalize(0, 2)
    for i, letter in enumerate(PANELS):
        row, col = divmod(i, 2)
        x, y, size = 37 + 139*col, 34 + 124*row, 98
        values = source.loc[source.panel.eq(letter)].pivot(
            index="sender_region", columns="receiver_region", values="share_percent").loc[HEATMAP_ORDER, HEATMAP_ORDER]
        axis = axes_box(fig, x, y, size, size)
        shown = np.ma.masked_array(values.to_numpy(), mask=masking)
        axis.pcolormesh(np.arange(9), np.arange(9), shown, cmap=cmap, norm=norm,
                        linewidth=.29, edgecolors="#e0e0e0")
        axis.set(xlim=(0, 8), ylim=(8, 0), aspect="equal", yticks=np.arange(8)+.5)
        axis.set_yticklabels(HEATMAP_ORDER, fontsize=6.1)
        axis.set_xticks(np.arange(8)+.5 if row == 1 else [])
        if row == 1:
            axis.set_xticklabels(HEATMAP_ORDER, rotation=48, ha="right", fontsize=6.1, rotation_mode="anchor")
        axis.tick_params(length=0, pad=1)
        axis.spines[:].set_visible(False)
        text_at(fig, x+size/2, y-22, NETWORK_TITLES[i], weight="bold", size=7.5, ha="center", va="top")
    text_at(fig, 9, 145, "Sending region", rotation=90, ha="center", va="center", size=6.9)
    text_at(fig, 160, 301, "Receiving region", ha="center", va="bottom", size=6.9)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=axes_box(fig, 286, 56, 5, 183),
                           ticks=np.arange(0, 2.01, .25))
    colorbar.ax.tick_params(labelsize=5.9, pad=1.3, length=1.2)
    colorbar.ax.set_title("Share (%)", size=6.4, pad=4)


PLOTS = {"a": schematic, "b": volume, "c": sections, "d": multiplane, "e": networks, "f": allocations}


def main(letter):
    plot = PLOTS[letter]
    args = panel_args(FIGURE, letter, plot.__name__).parse_args()
    fig = panel(letter, FRAMES)
    plot(fig, args.zenodo)
    save(fig, args.output)
