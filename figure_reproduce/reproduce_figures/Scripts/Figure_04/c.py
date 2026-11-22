"""Render the interpolation-range primary metrics for both 04A modes."""

from itertools import product

import pandas as pd
from matplotlib.lines import Line2D

from layout import METHODS, MODES, panel, panel_args, save_panel, style_axis

METRICS = (
    ("batch_asw", "Batch separation ASW ↓"),
    ("batch_mixing_entropy_k30", "Batch mixing entropy (k=30) ↑"),
    ("mmd2_rbf_biased", "Distribution MMD² ↓"),
    ("paired_retrieval_top1", "Paired retrieval top-1 ↑"),
    ("domain_asw", "Domain separation ASW ↑"),
    ("domain_purity_k30", "Domain purity (k=30) ↑"),
)
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def main():
    args = panel_args("c", __doc__)
    data = pd.read_csv(args.results / "same_slice/metrics/atomic_metrics.csv")
    selected = data.loc[data.analysis_role.eq("primary") & data.alpha.isin(ALPHAS)].copy()
    expected = set(product(list(METHODS)[:3], MODES, ALPHAS))
    if len(selected) != 24 or set(zip(selected.method, selected["mode"], selected.alpha)) != expected:
        raise ValueError("Expected 24 primary 04A interpolation rows")
    table = args.results / "plot_data/same_slice/c_primary_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    selected[["method", "mode", "alpha", "representation", "support_variant",
              *(key for key, _ in METRICS)]].to_csv(table, index=False)
    fig = panel("c", label_x=1)
    fig.text(0.52, 0.982, "Same-slice batch removal and biological preservation",
             fontsize=9.05, weight="bold", ha="center", va="top")
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.806, bottom=0.16, wspace=0.37, hspace=0.54)
    for axis, (metric, label) in zip(axes.flat, METRICS):
        for (method, mode), subset in selected.groupby(["method", "mode"]):
            rows = subset.sort_values("alpha")
            color, marker = METHODS[method]
            axis.plot(rows.alpha, rows[metric], color=color, ls=MODES[mode][1], marker=marker,
                      lw=0.93, ms=2.05, mec="white", mew=0.23)
        style_axis(axis, max_ticks=4)
        axis.set_title(label, fontsize=7.1, weight="bold", pad=4)
        axis.set_xticks(ALPHAS, ["0.25", "0.50", "0.75", "1.00"])
        axis.set_xlim(0.21, 1.03)
    handles = [Line2D([], [], color=color, marker=marker, ms=3, lw=1.05, label=name)
               for name, (color, marker) in list(METHODS.items())[:3]]
    handles += [Line2D([], [], color="#555555", ls=style, lw=1.05, label=name)
                for name, style in MODES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.946),
               ncols=5, handlelength=1.3, columnspacing=0.9, fontsize=6.7)
    fig.text(0.515, 0.032, "Perturbation strength α", ha="center", fontsize=7.3)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
