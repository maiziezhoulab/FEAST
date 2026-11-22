"""Render the interpolation-range primary metrics for both 04A modes."""

from itertools import product

import pandas as pd
from matplotlib.ticker import MultipleLocator

from layout import METHODS, MODES, metric_axes, panel, panel_args, save_panel, style_axis, text_at

METRICS = (
    ("batch_asw", "Batch separation ASW ↓", 0.005),
    ("batch_mixing_entropy_k30", "Batch mixing entropy (k=30) ↑", 0.1),
    ("mmd2_rbf_biased", "Distribution MMD² ↓", 0.005),
    ("paired_retrieval_top1", "Paired retrieval top-1 ↑", 0.1),
    ("domain_asw", "Domain separation ASW ↑", 0.02),
    ("domain_purity_k30", "Domain purity (k=30) ↑", 0.02),
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
              *(key for key, _, _ in METRICS)]].to_csv(table, index=False)
    fig = panel("c", label_x=1, label_top=0)
    text_at(fig, 181, 5.5, "Same-slice batch removal and biological preservation",
            fontsize=8.6, weight="bold", ha="center", va="top")
    text_at(fig, 181, 98.0, "Biological signal preservation",
            fontsize=7.1, weight="bold", ha="center", va="top")
    axes = metric_axes(fig, (19.5, 134.4, 248.5), (28.5, 122.2))
    for axis, (metric, label, step) in zip(axes, METRICS):
        for (method, mode), subset in selected.groupby(["method", "mode"]):
            rows = subset.sort_values("alpha")
            color, marker = METHODS[method]
            axis.plot(rows.alpha, rows[metric], color=color, ls=MODES[mode][1], marker=marker,
                      lw=0.93, ms=2.05, mec="white", mew=0.23)
        style_axis(axis)
        axis.yaxis.set_major_locator(MultipleLocator(step))
        axis.set_title(label, fontsize=6.65, weight="bold", pad=4)
        axis.set_xticks(ALPHAS, ["0.25", "0.50", "0.75", "1.00"])
        axis.set_xlim(0.21, 1.03)
    text_at(fig, 181, 189, "Perturbation strength α", fontsize=6.7, ha="center", va="top")
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
