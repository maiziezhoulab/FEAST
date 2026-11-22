"""Render the four displayed 04B conditions with method means, seeds and ranges."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from layout import METHODS, panel, panel_args, save_panel, style_axis

METRICS = (
    ("conditional_batch_asw_macro", "Batch ASW ↓"),
    ("conditional_batch_entropy_macro", "Within-layer batch entropy ↑"),
    ("layer_asw", "Layer separation ASW ↑"),
    ("layer_purity_k", "Layer purity (k=30) ↑"),
    ("conditional_mmd2_macro", "Distribution MMD² ↓"),
)
JITTER = {42: -0.024, 43: 0, 44: 0.024}


def draw_metric(axis, key, title, atomic, summary):
    for method, (color, marker) in METHODS.items():
        by_method = summary.loc[summary.metric.eq(key) & summary.method.eq(method)].sort_values("x_position")
        raw = by_method.loc[by_method.condition.eq("raw")]
        ladder = by_method.loc[by_method.condition.ne("raw")]
        linestyle = "--" if method == "Uncorrected" else "-"
        if method != "Uncorrected":
            axis.fill_between(ladder.x_position, ladder.seed_min, ladder.seed_max, color=color,
                              alpha=0.14, linewidth=0)
            raw_row = raw.iloc[0]
            axis.vlines(raw_row.x_position, raw_row.seed_min, raw_row.seed_max, color=color, lw=0.75, alpha=0.55)
            seeds = atomic.loc[atomic.metric.eq(key) & atomic.method.eq(method)]
            axis.scatter(seeds.x_position + seeds.method_seed.map(JITTER), seeds.value,
                         s=5.4, marker=marker, alpha=0.46, color=color, linewidths=0, zorder=3)
        axis.plot(ladder.x_position, ladder.seed_mean, color=color, ls=linestyle, marker=marker,
                  lw=1.05, ms=2.5, mew=0.3, mec="white", zorder=4)
        axis.scatter(raw.x_position, raw.seed_mean, s=7, color=color, marker=marker,
                     zorder=4, linewidths=0.3, edgecolors="white")
    style_axis(axis, max_ticks=4)
    axis.set_title(title, fontsize=7.5, weight="bold", pad=3.5)
    axis.axvline(-0.275, color="#c7c7c7", lw=0.55, ls=":", zorder=0)
    axis.set_xlim(-0.75, 1.145)
    axis.set_xticks([-0.55, 0.0, 0.5, 1.0], ["Raw", "0", "0.5", "1.0"])
    axis.margins(y=0.12)


def main():
    args = panel_args("b", __doc__)
    directory = args.results / "plot_data/real_slices"
    atomic = pd.read_csv(directory / "b_method_seed_rows.csv")
    summary = pd.read_csv(directory / "b_seed_summary.csv")
    expected_seeds = np.where(summary.method.eq("Uncorrected"), 1, 3)
    if len(atomic) != 200 or len(summary) != 80 or not np.array_equal(summary.seed_n, expected_seeds):
        raise ValueError("Expected the complete 04B displayed seed matrix")
    fig = panel("b", label_x=2)
    fig.text(0.55, 0.98, "Two-slice batch removal and layer preservation",
             ha="center", va="top", fontsize=8.15, weight="bold")
    axes = fig.subplots(3, 2)
    fig.subplots_adjust(left=0.09, right=0.991, top=0.873, bottom=0.095, wspace=0.38, hspace=0.6)
    for axis, (key, title) in zip(axes.flat, METRICS):
        draw_metric(axis, key, title, atomic, summary)
    axis = axes[-1, -1]
    axis.axis("off")
    handles = [
        Line2D([], [], color=color, marker=marker, label="Uncorrected input" if method == "Uncorrected" else method,
               ms=3.2, lw=1.05, ls="--" if method == "Uncorrected" else "-")
        for method, (color, marker) in METHODS.items()
    ]
    axis.legend(handles=handles, loc="upper left", borderaxespad=0, fontsize=7,
                handlelength=1.6, labelspacing=0.36)
    axis.text(0.0, 0.012, "Raw  Measured anchor\nα=0  Resampling control", va="bottom",
              ha="left", fontsize=6.4, color="#555555", transform=axis.transAxes)
    fig.text(0.55, 0.021, "Perturbation strength α", ha="center", fontsize=7.3)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
