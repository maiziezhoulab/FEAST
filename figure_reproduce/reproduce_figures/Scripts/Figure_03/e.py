"""Render rotation recovery with the imposed angle and geometric spot retention."""

from itertools import product

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from layout import HEADING_COLOR, manuscript_axes, manuscript_text, panel, panel_args, save_panel

CONDITIONS = {"baseline": "Baseline", "mean_0.5": "Mean ×0.5",
              "variance_2.0": "Variance ×2.0", "sparsity_0.5": "Zero odds ×0.5"}
METHODS = {"paste2": ("PASTE2", "#0F4D92", "o"), "spateo": ("Spateo", "#B64141", "s")}
ANGLES = (5, 15, 25, 35, 45, 60, 75)
FLOOR = 1e-4


def main():
    args = panel_args("e", __doc__)
    study = args.results / "02_alignment"
    score_paths = ("paste2/scores/alignment_metrics.csv", "spateo_scores/alignment_metrics.csv")
    rows = pd.concat([pd.read_csv(study / path) for path in score_paths], ignore_index=True)
    expected = set(product(METHODS, CONDITIONS, ANGLES))
    if len(rows) != len(expected) or set(zip(rows.method, rows.alteration, rows.angle_degrees)) != expected:
        raise ValueError("Expected 56 distinct alignment scores")
    columns = ["job_id", "method", "alteration", "angle_degrees", "retained_fraction", "rotation_recovery_error"]
    rows = rows[columns].copy().sort_values(["alteration", "method", "angle_degrees"])
    if not np.isfinite(rows.rotation_recovery_error).all() or rows.rotation_recovery_error.lt(0).any():
        raise ValueError("Invalid rotation error")
    fractions = rows.groupby("angle_degrees").retained_fraction
    if not np.allclose(fractions.min(), fractions.max()):
        raise ValueError("Retained fraction must agree across expression conditions and methods")
    rows["display_error"] = rows.rotation_recovery_error.clip(lower=FLOOR)
    rows["below_display_floor"] = rows.rotation_recovery_error.lt(FLOOR)
    tables = study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)
    rows.to_csv(tables / "e_rotation_recovery.csv", index=False)

    fig = panel("e")
    manuscript_text(fig, "e", 141.3, 595.3, "Alignment Benchmark", fontsize=11.8,
                    color=HEADING_COLOR, weight="bold", ha="left")
    axes = [manuscript_axes(fig, "e", x, 617.2, 90.0, 80.0)
            for x in (178.3, 282.0, 385.7, 489.4)]
    labels = [f"{angle:g}°\n{fraction:.0%}" for angle, fraction in fractions.first().items()]
    upper = max(1.25, 1.26 * rows.display_error.max())
    exponents = range(-4, int(np.floor(np.log10(upper))) + 1)
    for i, (axis, (alteration, title)) in enumerate(zip(axes, CONDITIONS.items(), strict=True)):
        axis.set_title(title, size=8, weight="bold", pad=6)
        for method, (_, color, marker) in METHODS.items():
            group = rows.loc[rows.method.eq(method) & rows.alteration.eq(alteration)]
            x = group.angle_degrees.to_numpy()
            axis.plot(x, group.display_error, color=color, lw=1.0)
            axis.plot(x, group.display_error, ls="", marker=marker, mfc="none",
                      mec=color, mew=0.65, ms=3.1, clip_on=False)
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(FixedLocator([10.0 ** exponent for exponent in exponents]))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: rf"$10^{{{round(np.log10(y))}}}$"))
        axis.set_ylim(FLOOR, upper)
        axis.set_xlim(2, 78)
        axis.set_xticks(ANGLES, labels, fontsize=4.6)
        axis.tick_params(width=0.55, length=2, pad=1.6, labelleft=i == 0)
        axis.tick_params(which="minor", width=0.4, length=1)
        axis.grid(axis="y", lw=0.4, color="#dedede")
    axes[0].set_ylabel("Rotation recovery error\n(degrees, log scale) ↓", fontsize=5.4)
    handles = [Line2D([], [], color=color, marker=marker, mfc="white", lw=1.1, ms=3, label=name)
               for name, color, marker in METHODS.values()]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.50, 1 - 27.5 / 155.833), ncols=2,
               fontsize=8, handlelength=1.4, columnspacing=0.6)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
