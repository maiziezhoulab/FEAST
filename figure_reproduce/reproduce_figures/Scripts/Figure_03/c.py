"""Render matched-support composition metrics by resolution and section."""

from itertools import product

import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from layout import panel, panel_args, save_panel, style_axis

METHODS = {"rctd": ("RCTD", "#4C78A8", "o"), "cell2location": ("Cell2location", "#D97731", "s")}
SECTIONS = ("007", "050", "100")
FACTORS = (0.1, 0.25)
METRICS = (
    ("jsd_mean", "Composition divergence ↓"),
    ("pearson_mean", "Composition correlation ↑"),
    ("rmse", "Composition error ↓"),
)


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
    tables = study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)
    rows.to_csv(tables / "c_scores.csv", index=False)
    audit.to_csv(tables / "c_support.csv", index=False)

    fig = panel("c")
    fig.text(0.51, 0.942, "Deconvolution benchmark", ha="center", weight="bold", fontsize=10.2)
    axes = fig.subplots(3, 1)
    fig.subplots_adjust(left=0.225, right=0.98, top=0.855, bottom=0.086, hspace=0.37)
    for axis, (metric, label) in zip(axes, METRICS):
        axis.set_title(label, weight="bold", fontsize=8.1, pad=4)
        for i, section in enumerate(SECTIONS):
            for method, (name, color, marker) in METHODS.items():
                values = rows.loc[rows.slice_id.eq(section) & rows.method.eq(method)].sort_values("resolution")[metric]
                offset = -0.07 if method == "rctd" else 0.07
                axis.plot([3 * i + offset, 3 * i + 1 + offset], values, color=color,
                          lw=1, marker=marker, ms=3.7, mec="white", mew=0.4,
                          label=name if i == 0 else None)
        for x in (2, 5):
            axis.axvline(x, color="#cccccc", ls=":", lw=0.55)
        style_axis(axis, max_ticks=5)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.set_xlim(-0.49, 7.49)
        axis.set_xticks([0, 1, 3, 4, 6, 7], ["0.10", "0.25"] * 3, fontsize=5.8)
    for i, section in enumerate(SECTIONS):
        axes[-1].text(3 * i + 0.5, -0.174, f"S{section}\nn = {counts.first()[section]}",
                      ha="center", va="top", transform=axes[-1].get_xaxis_transform(), fontsize=6.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.927),
               ncols=2, fontsize=6.6, handlelength=1.35, columnspacing=1.1)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
