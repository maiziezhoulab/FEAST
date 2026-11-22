"""Developmental communication figures from native archived ComFlow matrices."""
import json

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image

from feast_plate import INK, axes_box, panel, panel_args, save, text_at
from layout import FIGURE, FRAMES, NATIVE_VIEW_HEIGHT

ROOT = "results/07_3d_transfer/ccc_atlas"
AGES = ("E15.5", "E18.5")
CASES = (("Fzd2_Lrp5", "IZ", "DorsalVZ"), ("Fzd8_Lrp6", "GE", "VentralVZ"))
BLUE = "#0f4d92"
GRAY = "#879198"
ORANGE = "#d95f02"
MUTED = "#68747c"
DELTA_MAP = mpl.colors.LinearSegmentedColormap.from_list(
    "contact_delta", ["#0f4d92", "#91bad8", "#ffffff", "#e6aba5", "#b64342"]
)
EXPRESSION_MAP = mpl.colors.LinearSegmentedColormap.from_list(
    "measured_expression", ["#ffffff", "#9ecae1", BLUE]
)
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
    ax = axes_box(fig, 68, 24, 250, 288)
    norm = mpl.colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
    image = ax.pcolormesh(np.arange(11), np.arange(len(labels) + 1), matrix.to_numpy(),
                          cmap=DELTA_MAP, norm=norm, edgecolors="white", linewidth=.37)
    ax.set(xlim=(0, 10), ylim=(len(labels), 0), xticks=np.arange(10) + .41,
           yticks=np.arange(len(labels)) + .5)
    xlabels = [f"{rank} {lr.replace('--', '–')}" for rank, lr in metrics.LR.items()]
    ax.set_xticklabels(xlabels, fontsize=6, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, fontsize=6.25)
    ax.tick_params(axis="x", length=0, pad=1.75)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines[:].set_visible(False)
    for i, row in enumerate(matrix.to_numpy()):
        for j, value in enumerate(row):
            label = f"{value:+.2f}" if round(value, 2) else "0.00"
            ax.text(j + .5, i + .5, label, ha="center", va="center",
                    color="white" if abs(value) > .78 * norm.vmax else "#17232b", fontsize=6)
    bar = fig.colorbar(image, cax=axes_box(fig, 325, 50, 4.5, 236), ticks=[-20, -10, 0, 10, 20])
    bar.ax.tick_params(labelsize=6, length=1.5, pad=1)
    bar.outline.set_linewidth(.4)
    text_at(fig, 327.25, 13, "Δ share", ha="center", size=6, color="black")
    text_at(fig, 327.25, 21, "(pp)", ha="center", size=6, color="black")


def draw_line(fig, xy0, xy1, *, color=INK, lw=.65):
    w, h = fig.get_size_inches()*72
    fig.add_artist(Line2D([xy0[0]/w, xy1[0]/w], [1-xy0[1]/h, 1-xy1[1]/h],
                          transform=fig.transFigure, color=color, linewidth=lw, solid_capstyle="butt"))


def native_views(fig, zenodo):
    root = zenodo / ROOT / "native"
    meta = json.loads((root / "provenance.json").read_text())
    records = {(row["LR"], row["stage"], row["mode"]): row for row in meta["routes"]}
    centers = [76.52, 164.52]
    for col, (age, source) in enumerate(zip(AGES, ("E14", "E18"))):
        text_at(fig, centers[col], 15.32, f"{age} target", ha="center", weight="bold", size=7.5)
        text_at(fig, centers[col], 24.32, f"({source}-derived)", ha="center", color=MUTED, size=6.25)
    for i, (receptor, a, b) in enumerate(CASES):
        lr = f"Wnt7a--{receptor}"
        y_base = 44.32 + 154 * i
        title_left = 68.44 - 2.61 * i
        title_top = 37.32 + 153 * i
        text_at(fig, title_left, title_top, lr.replace("--", "–").replace("_", "/"),
                weight="bold", size=7.5)
        text_at(fig, title_left + 69.09, title_top, f"{a}–{b}", color=MUTED, size=6.25)
        for offset, (mode, label) in enumerate((("forward", f"{a}→{b}"), ("reverse", f"{b}→{a}"))):
            top = y_base + 70 * offset
            height = NATIVE_VIEW_HEIGHT
            text_at(fig, 34.5, top + height / 2, label, rotation=90, ha="center", va="center", size=6.25)
            for col, age in enumerate(AGES):
                record = records[lr, age, mode]
                with Image.open(root / record["source"]) as raw:
                    image = raw.crop(record["crop_pixels"])
                width = height * image.width / image.height
                x = centers[col]
                axis = axes_box(fig, x - width / 2, top, width, height)
                axis.imshow(image, interpolation="antialiased")
                axis.set_axis_off()
                length = width / record["cropped_world_width_mm"]
                bar_y = top + height + 2.68
                draw_line(fig, (x - length / 2, bar_y), (x + length / 2, bar_y))
                text_at(fig, x + length / 2 + 2.15, bar_y + 1.9, "1 mm", size=5.5)
    for i, role in enumerate(("Receiver", "Sender")):
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap=ROLE_MAPS[role]),
                           cax=axes_box(fig, 210.03, 69.23 + 88 * i, 4.5, 76), ticks=[0, 100])
        bar.ax.tick_params(length=0, pad=2, labelsize=5.5, labelrotation=90)
        bar.outline.set_linewidth(.4)
    text_at(fig, 229.7, 151.23, "Endpoint mass percentile", rotation=90, ha="center", va="center",
            fontsize=6.1, color=MUTED)
    for x, role in zip(centers, ("Sender", "Receiver")):
        text_at(fig, x, 344.32, role, ha="center", size=6.5)
    record = meta["routes"][0]
    view = np.asarray(record["view"])
    up = np.asarray(record["up"])
    right = np.cross(-view, [0., 0., 1.])
    right /= np.linalg.norm(right)
    xy = np.array([202.52, 332.32])
    for vector, label in zip(np.eye(3), "RAS"):
        step = 7.2 * np.array([vector @ right, -vector @ up])
        draw_line(fig, xy, xy + step, color=MUTED, lw=.55)
        text_at(fig, *(xy + 1.78 * step), label, va="center", ha="center", size=5.5, color=MUTED)


def directional(fig, zenodo, case):
    receptor, a, b = CASES[case]
    data = pd.read_csv(zenodo / ROOT / "wnt_biology/directional_opportunity.csv")
    selection = data.receptor.eq(receptor) & data.interface.isin([f"{a}–{b}", f"{a}--{b}"])
    data = data.loc[selection].rename(columns={"n_eligible_pairs": "eligible_pairs"}).copy()
    data["mass_per_1000_pairs"] = 1000 * data.W / data.eligible_pairs
    data = data.set_index(["sender_region", "receiver_region", "stage"], verify_integrity=True).sort_index()
    text_at(fig, 87, 17.42, f"{a}–{b}", ha="center", fontsize=9, weight="bold")
    axis = axes_box(fig, 31, 30, 100, 104.839)
    ymax = float(data.mass_per_1000_pairs.max())
    directions = ((a, b, BLUE, "-"), (b, a, GRAY, "--"))
    label_offsets = ((7, 4.2), (7, -1.3)) if case == 0 else ((4, 3.6), (4, 3.6))
    for (sender, receiver, color, linestyle), offset in zip(directions, label_offsets):
        rows = data.xs((sender, receiver)).loc[list(AGES)]
        field = rows.mass_per_1000_pairs.to_numpy()
        axis.plot([0, 1], field, c=color, lw=1.15, linestyle=linestyle)
        axis.scatter([0], field[:1], s=14, facecolor="white", edgecolors=color, lw=.8, zorder=3)
        axis.scatter([1], field[1:], s=14, c=color, linewidths=0, zorder=3)
        fold = field[1] / field[0]
        axis.annotate(f"{sender}→{receiver}\n{fold:.2f}×", xy=(1, field[1]),
                      xytext=offset, textcoords="offset points",
                      va="center", ha="left", size=5.5, color=color, linespacing=1)
    axis.set(xlim=(-.15, 1.85), ylim=(0, ymax * 1.25), xticks=[0, 1])
    axis.set_xticklabels(AGES)
    axis.tick_params(labelsize=5.5, length=1.87, pad=2)
    axis.yaxis.set_major_locator(MultipleLocator(.25 if case == 0 else 2))
    axis.spines[["top", "right"]].set_visible(False)
    if case == 0:
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        text_at(fig, 6.5, 82.42, "CCC mass / 10³ eligible pairs", rotation=90, ha="center", va="center",
                size=6, color="black")
    support = data.eligible_pairs.unstack(level="stage").loc[:, list(AGES)]
    np.testing.assert_array_equal(support.loc[(a, b)], support.loc[(b, a)])
    early, late = support.loc[(a, b)].to_numpy()
    text_at(fig, 81, 153, "Eligible pairs", ha="center", color=MUTED, size=5.5)
    text_at(fig, 81, 163, f"{early:,} → {late:,}", ha="center", size=8, weight="bold")


def molecular(fig, zenodo, case):
    receptor, a, b = CASES[case]
    rec_genes = receptor.split("_")
    lr = f"Wnt7a--{receptor}"
    root = zenodo / ROOT / "observed_st"
    counts = pd.read_csv(root / "measured_RNA_source_data.csv")
    support = pd.read_csv(root / "measured_receptor_codetection.csv")
    text_at(fig, 72, 17.42, lr.replace("--", "–").replace("_", "/"), ha="center", fontsize=9, weight="bold")
    table = counts.loc[counts.LR.eq(lr)].pivot(index="source_stage", columns="gene", values="mean_log1p_expression")
    table = table.loc[["E14", "E18"], ["Wnt7a", *rec_genes]]
    norm = mpl.colors.Normalize(0, counts.mean_log1p_expression.max())
    ax = axes_box(fig, 39, 47.42, 84, 70)
    ax.pcolormesh(np.arange(4), np.arange(3), table.to_numpy(), cmap=EXPRESSION_MAP, norm=norm,
                   edgecolors="white", linewidth=.45)
    for i, row in enumerate(table.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j + .5, i + .5, f"{value:.3f}", ha="center", va="center",
                    color="white" if norm(value) > .66 else INK, size=6.25)
    ax.axvline(1, color="#8d969d", linewidth=.65)
    ax.set(xlim=(0, 3), ylim=(2, 0), xticks=np.arange(3) + .5, yticks=np.arange(2) + .5)
    ax.set_xticklabels(["Wnt7a", *rec_genes], fontsize=6.25)
    ax.set_yticklabels(["E14 input", "E18 input"], fontsize=5.5)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines[:].set_visible(False)
    for x, label, region, color in ((53, "Sender", a, ORANGE), (95, "Receiver", b, BLUE)):
        text_at(fig, x, 28, label, size=5.75, color=color, ha="center")
        text_at(fig, x, 36, region, size=5.75, color=color, ha="center")
    pair = support.loc[support.LR.eq(lr)].set_index("source_stage").loc[["E14", "E18"]].joint_positive_pct
    text_at(fig, 81, 153, "Both receptor transcripts detected", ha="center", color=MUTED, size=5.5)
    text_at(fig, 81, 163, f"{pair.iloc[0]:.1f}% → {pair.iloc[1]:.1f}%", ha="center", size=8, weight="bold")


def main(letter):
    plots = {"a": changes, "b": native_views, "c": directional, "d": directional, "e": molecular, "f": molecular}
    args = panel_args(FIGURE, letter, plots[letter].__name__).parse_args()
    fig = panel(letter, FRAMES)
    _, _, width, height = FRAMES[letter]
    fig.texts[0].set(position=(6 / width, 1 - 14 / height), va="baseline", fontsize=12.65)
    case = {"c": 0, "d": 1, "e": 0, "f": 1}
    parameters = [case[letter]] if letter in case else []
    plots[letter](fig, args.zenodo, *parameters)
    save(fig, args.output)
