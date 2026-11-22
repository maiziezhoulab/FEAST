"""Render matched-support composition metrics by resolution and section."""

from itertools import product

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from layout import HEADING_COLOR, manuscript_axes, manuscript_text, panel, panel_args, save_panel, style_axis

METHODS = {"rctd": ("RCTD", "#365C8D", "o"), "cell2location": ("Cell2location", "#E07A3F", "s")}
SECTIONS = ("007", "050", "100")
FACTORS = (0.1, 0.25)
METRICS = (
    ("jsd_mean", "Composition divergence", (0.045, 0.060, 0.075, 0.090)),
    ("pearson_mean", "Composition correlation", (0.910, 0.920, 0.930, 0.940)),
    ("rmse", "Composition error", (0.048, 0.056, 0.064, 0.072)),
)
POSITIONS = ((0, 1), (1.9, 2.9), (3.8, 4.8))


def main():
    args = panel_args("c", __doc__)
    study = args.results / "03_deconvolution"
    source = study / "cell_class/scores"
    rows = pd.read_csv(source / "deconvolution_scores.csv", dtype={"slice_id": str})
    audit = pd.read_csv(source / "support_audit.csv", dtype={"slice_id": str})
    expected = set(product(SECTIONS, FACTORS, METHODS))
    if len(rows) != 12 or set(zip(rows.slice_id, rows.resolution, rows.method)) != expected or not rows.status.eq("ok").all():
        raise ValueError("Expected twelve successful composition scores on common support")
    counts = audit.groupby("slice_id").n_retained_cell_classes
    if len(audit) != 6 or not counts.nunique().eq(1).all():
        raise ValueError("Retained class count differs across displayed resolutions")
    tables = args.tables_dir or study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)
    rows.to_csv(tables / "c_scores.csv", index=False)
    audit.to_csv(tables / "c_support.csv", index=False)

    fig = panel("c")
    manuscript_text(fig, "c", 2.8, 327.1, "Deconvolution Benchmark", ha="left",
                    color=HEADING_COLOR, weight="bold", fontsize=11.8)
    tops = (345.3, 472.29, 599.5)
    axes = [manuscript_axes(fig, "c", 35.94, top, 97.54, 96.87) for top in tops]
    for axis, top, (metric, label, ticks) in zip(axes, tops, METRICS, strict=True):
        manuscript_text(fig, "c", 84.71, top - 8.7, label, ha="center", weight="bold", fontsize=8.1)
        for (left, right), section in zip(POSITIONS, SECTIONS, strict=True):
            for method, (_, color, marker) in METHODS.items():
                values = rows.loc[rows.slice_id.eq(section) & rows.method.eq(method)].sort_values("resolution")[metric]
                offset = -0.13 if method == "rctd" else 0.13
                x = [left + offset, right + offset]
                axis.plot(x, values, color=color, lw=0.8, alpha=0.58)
                axis.plot(x, values, ls="", marker=marker, color=color,
                          ms=3.9, mec="white", mew=0.4)
        for x in (1.45, 3.35):
            axis.axvline(x, color="#cccccc", ls=":", lw=0.55)
        style_axis(axis)
        low, high = rows[metric].min(), rows[metric].max()
        margin = max((high - low) * 0.08, 0.002 if metric == "pearson_mean" else 0.0)
        axis.set_ylim(min(ticks[0], low - margin), max(ticks[-1], high + margin))
        axis.set_yticks(ticks)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.set_xlim(-0.4, 5.25)
        axis.set_xticks([position for pair in POSITIONS for position in pair], ["0.10", "0.25"] * 3)
        axis.tick_params(labelsize=5.2)
    for (left, right), section in zip(POSITIONS, SECTIONS, strict=True):
        axes[-1].text((left + right) / 2, -0.128, f"S{section}\nn = {counts.first()[section]}",
                      ha="center", va="top", transform=axes[-1].get_xaxis_transform(),
                      fontsize=5.8, weight="bold", linespacing=1.0, color=HEADING_COLOR)
    handles = [Line2D([], [], color=color, marker=marker, lw=0.75, ms=3.7,
                      mec="white", mew=0.4, label=name) for name, color, marker in METHODS.values()]
    axes[1].legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 1.045),
                   ncols=2, fontsize=4.9, handlelength=1.5, columnspacing=1.0)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
