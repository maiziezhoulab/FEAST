"""Developmental communication figures from native archived ComFlow matrices."""
import json

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from feast_plate import INK, axes_box, panel, panel_args, save, text_at
from layout import FIGURE, FRAMES

ROOT = "results/07_3d_transfer/ccc_atlas"
AGES = ("E15.5", "E18.5")
CASES = (("Fzd2_Lrp5", "IZ", "DorsalVZ"), ("Fzd8_Lrp6", "GE", "VentralVZ"))
BLUE = "#0f4d92"
GRAY = "#879198"
ORANGE = "#d95f02"
ROLE_MAPS = {
    "Receiver": mpl.colors.LinearSegmentedColormap.from_list("receiver", ["#eff3ff", "#9ecae1", "#3182bd", "#08519c"]),
    "Sender": mpl.colors.LinearSegmentedColormap.from_list("sender", ["#fff5eb", "#fdae6b", "#e6550d", "#8c2d04"]),
}


def changes(fig, zenodo):
    data = zenodo / ROOT / "region_pairs_merged_contact"
    metrics = pd.read_csv(data / "LR_region_pair_metrics.csv").set_index("JSD_rank").sort_index().iloc[:10]
    contacts = pd.read_csv(data / "physical_contact_pairs.csv").sort_values("pair_order")
    rows = pd.read_csv(data / "LR_region_pair_changes.csv")
    matrix = rows.pivot(index="region_pair", columns="LR", values="delta_share_pp").loc[contacts.region_pair, metrics.LR]
    labels = [f"{row.region_pair}{'†' if row.stage_specific_contact else ''}" for row in contacts.itertuples(index=False)]
    ax = axes_box(fig, 68, 24, 249, 287)
    norm = mpl.colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
    image = ax.pcolormesh(np.arange(11), np.arange(len(labels)+1), matrix.to_numpy(),
                          cmap="RdBu_r", norm=norm, edgecolors="white", linewidth=.37)
    ax.set(xlim=(0,10), ylim=(len(labels),0), xticks=np.arange(10)+.5, yticks=np.arange(len(labels))+.5)
    xlabels = [f"{rank} {lr.replace('--','–')}" for rank, lr in metrics.LR.items()]
    ax.set_xticklabels(xlabels, fontsize=6, rotation=61, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, fontsize=6.1)
    ax.tick_params(length=0, pad=2)
    ax.spines[:].set_visible(False)
    for i, row in enumerate(matrix.to_numpy()):
        for j, value in enumerate(row):
            label = f"{value:+.2f}" if value else "0.00"
            ax.text(j+.5, i+.5, label, ha="center", va="center",
                    color="white" if abs(value) >= 12.5 else "#202326", fontsize=5.9)
    bar = fig.colorbar(image, cax=axes_box(fig, 324, 47, 5, 239), ticks=[-20,-10,0,10,20])
    bar.ax.tick_params(labelsize=5.9, length=1.5, pad=1)
    bar.ax.set_title("Δ share\n(pp)", size=6.2, pad=6)


def draw_line(fig, xy0, xy1, *, color=INK, lw=.65):
    w, h = fig.get_size_inches()*72
    fig.add_artist(Line2D([xy0[0]/w, xy1[0]/w], [1-xy0[1]/h, 1-xy1[1]/h],
                          transform=fig.transFigure, color=color, linewidth=lw, solid_capstyle="butt"))


def native_views(fig, zenodo):
    root = zenodo / ROOT / "native"
    meta = json.loads((root / "provenance.json").read_text())
    records = {(row["LR"], row["stage"], row["mode"]): row for row in meta["routes"]}
    centers = [76, 158]
    for col, (age, source) in enumerate(zip(AGES, ("E14", "E18"))):
        text_at(fig, centers[col], 8, f"{age} target", ha="center", va="top", weight="bold", size=7.45)
        text_at(fig, centers[col], 19, f"({source}-derived)", ha="center", va="top", color=GRAY, size=6.2)
    for i, (receptor, a, b) in enumerate(CASES):
        lr = f"Wnt7a--{receptor}"
        y_base = 39 + 157*i
        text_at(fig, 102, y_base-8, lr.replace("--", "–").replace("_", "/"), ha="right", va="top",
                weight="bold", size=7.55)
        text_at(fig, 108, y_base-7.5, f"{a}–{b}", ha="left", va="top", color="#68747c", size=6.65)
        for offset, (mode, label) in enumerate((("forward", f"{a}→{b}"), ("reverse", f"{b}→{a}"))):
            top = y_base + 71 * offset
            text_at(fig, 30, top+32, label, rotation=90, ha="center", va="center", size=6.55)
            for col, age in enumerate(AGES):
                record = records[lr, age, mode]
                with Image.open(root / record["source"]) as raw:
                    image = raw.crop(record["crop_pixels"])
                ratio = image.width / image.height
                width = record["display_width_pt"]
                height = width/ratio
                x = centers[col]
                axis = axes_box(fig, x - width/2, top + (62-height)/2, width, height)
                axis.imshow(image, interpolation="antialiased")
                axis.set_axis_off()
                length = record["scale_bar_pt"]
                draw_line(fig, (x-length/2, top+66), (x+length/2, top+66))
                text_at(fig, x+length/2+2, top+66, "1 mm", va="center", size=5.2)
    for i, role in enumerate(("Receiver", "Sender")):
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,100), cmap=ROLE_MAPS[role]),
                           cax=axes_box(fig, 207, 63+106*i, 5.5, 73), ticks=[0, 100])
        bar.ax.tick_params(labelsize=5.6, length=1., pad=1.1)
        text_at(fig, 210, 142+106*i, role, ha="center", va="top", fontsize=5.6)
    text_at(fig, 232, 151, "Endpoint mass percentile", rotation=90, ha="center", va="center",
            fontsize=6.1, color="#59646c")
    record = meta["routes"][0]
    view = np.asarray(record["view"])
    up = np.asarray(record["up"])
    right = np.cross(-view, [0., 0., 1.])
    right /= np.linalg.norm(right)
    xy = np.array([206., 335.])
    for vector, label in zip(np.eye(3), "RAS"):
        step = 9.8 * np.array([vector@right, -vector@up])
        draw_line(fig, xy, xy+step, color="#718089", lw=.62)
        text_at(fig, *(xy+1.35*step), label, va="center", ha="center", size=5.9, color="#5b6469")


def directional(fig, zenodo, case):
    receptor, a, b = CASES[case]
    data = pd.read_csv(zenodo / ROOT / "wnt_biology/directional_opportunity.csv")
    data = data.loc[data.receptor.eq(receptor) & data.interface.eq(f"{a}–{b}")]
    text_at(fig, 74, 11, f"{a}–{b}", va="top", ha="center", fontsize=8.8, weight="bold")
    axis = axes_box(fig, 32 if case==0 else 26, 30, 97, 104)
    ymax = float(data.mass_per_1000_pairs.max())
    for i, (sender, receiver, color, ls) in enumerate(((a,b,BLUE,"-"), (b,a,GRAY,"--"))):
        rows = data.loc[data.sender_region.eq(sender)].set_index("stage").loc[list(AGES)]
        field = rows.mass_per_1000_pairs.to_numpy()
        axis.plot([0,1], field, c=color, lw=1.15, linestyle=ls)
        axis.scatter([0], field[:1], s=14, facecolor="white", edgecolors=color, lw=.8, zorder=3)
        axis.scatter([1], field[1:], s=14, c=color, linewidths=0, zorder=3)
        y = field[1] + (.04 if i==0 else (-.02 if case==0 else .025)) * ymax
        axis.text(1.14, y, f"{sender}→{receiver}\n{rows.late_over_early.iloc[0]:.2f}×",
                  va="center", ha="left", size=5.75, color=color, linespacing=1)
    axis.set(xlim=(-.14,2.03), ylim=(0, ymax*1.25), xticks=[0,1])
    axis.set_xticklabels(AGES)
    axis.tick_params(labelsize=6.05)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.spines[["top", "right"]].set_visible(False)
    if case == 0:
        axis.set_ylabel("CCC mass / 10³ eligible pairs", size=6.2, labelpad=4)
    support = data.groupby("stage").eligible_pairs.first().loc[list(AGES)]
    text_at(fig, 81, 149, "Eligible pairs", ha="center", va="top", color="#68747c", size=6.15)
    text_at(fig, 81, 160, f"{support.iloc[0]:,} → {support.iloc[1]:,}", ha="center", va="top", size=8.1, weight="bold")


def molecular(fig, zenodo, case):
    receptor, a, b = CASES[case]
    rec_genes = receptor.split("_")
    lr = f"Wnt7a--{receptor}"
    root = zenodo / ROOT / "observed_st"
    counts = pd.read_csv(root / "measured_RNA_source_data.csv")
    support = pd.read_csv(root / "measured_receptor_codetection.csv")
    text_at(fig, 75, 12, lr.replace("--", "–").replace("_", "/"), va="top", ha="center", fontsize=8.8, weight="bold")
    table = counts.loc[counts.LR.eq(lr)].pivot(index="source_stage", columns="gene", values="mean_log1p_expression")
    table = table.loc[["E14", "E18"], ["Wnt7a", *rec_genes]]
    c = mpl.colors.LinearSegmentedColormap.from_list("measured_expression", ["#ffffff", "#9ecae1", BLUE])
    norm = mpl.colors.Normalize(0, counts.mean_log1p_expression.max())
    ax = axes_box(fig, 39, 49, 88, 72)
    ax.pcolormesh(np.arange(4), np.arange(3), table.to_numpy(), cmap=c, norm=norm,
                   edgecolors="white", linewidth=.45)
    for i, row in enumerate(table.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j+.5, i+.5, f"{value:.3f}", ha="center", va="center",
                    color="white" if norm(value)>.66 else INK, size=6.35)
    ax.set(xlim=(0,3), ylim=(2,0), xticks=np.arange(3)+.5, yticks=np.arange(2)+.5)
    ax.set_xticklabels(["Wnt7a", *rec_genes], fontsize=6.3)
    ax.set_yticklabels(["E14 input", "E18 input"], fontsize=6.3)
    ax.tick_params(length=0, pad=2)
    ax.spines[:].set_visible(False)
    text_at(fig, 53.7, 28, f"Sender\n{a}", fontweight="normal", size=6.05,
            color=ORANGE, ha="center", va="top", linespacing=1.13)
    text_at(fig, 97.4, 28, f"Receiver\n{b}", fontweight="normal", size=6.05,
            color=BLUE, ha="center", va="top", linespacing=1.13)
    pair = support.loc[support.LR.eq(lr)].set_index("source_stage").loc[["E14", "E18"]].joint_positive_pct
    text_at(fig, 84, 149, "Both receptor transcripts detected", ha="center", va="top", color="#68747c", size=5.9)
    text_at(fig, 84, 160, f"{pair.iloc[0]:.1f}% → {pair.iloc[1]:.1f}%", ha="center", va="top", size=8.15, weight="bold")


def main(letter):
    plots = {"a": changes, "b": native_views, "c": directional, "d": directional, "e": molecular, "f": molecular}
    args = panel_args(FIGURE, letter, plots[letter].__name__).parse_args()
    fig = panel(letter, FRAMES)
    case = {"c": 0, "d": 1, "e": 0, "f": 1}
    parameters = [case[letter]] if letter in case else []
    plots[letter](fig, args.zenodo, *parameters)
    save(fig, args.output)
