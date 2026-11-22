"""Render journal Figure 2a from the assembled 60-row Study 00 metric table."""

import argparse
import json
from pathlib import Path

from layout import A_HEIGHT, FIGURES, PROJECT, RESULTS, WIDTH, configure_style, rect
import plot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, MaxNLocator


def draw_axis(ax, long_table, spec):
    label = spec["label"]
    subset = long_table.loc[long_table["metric"] == label, ["method", "sample", "score"]]
    methods = plot.panel_method_order(long_table, spec)
    groups = [
        subset.loc[subset["method"] == method].sort_values("sample")["score"].to_numpy()
        for method in methods
    ]
    values = subset["score"].to_numpy()
    if not np.isfinite(values).all() or any(len(group) != 12 for group in groups):
        raise ValueError(f"{label} requires one finite score per section and simulator")

    zero_slot = None
    stats = boxplot_stats(groups, whis=1.5)
    if spec.get("yscale") == "log":
        if (values < 0).any() or not (values > 0).any():
            raise ValueError(f"{label} requires nonnegative scores with positive log support")
        positive_min = values[values > 0].min()
        largest = values.max()
        ax.set_yscale("log")
        ax.set_ylim(positive_min * 0.65, largest * 1.35)
        if (values == 0).any():
            # Put exact zeros in a labelled slot below the positive log range.
            zero_slot = positive_min / 4
            ax.set_ylim(zero_slot * 0.68, largest * 1.35)
            ticks = LogLocator(numticks=5).tick_values(positive_min * 0.65, largest * 1.35)
            ticks = ticks[(ticks >= positive_min * 0.65) & (ticks <= largest * 1.35)]
            formatter = LogFormatterMathtext()
            ax.yaxis.set_major_locator(FixedLocator([zero_slot, *ticks]))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: "0" if value == zero_slot else formatter(value))
            )
            ax.axhline(positive_min * 0.60, color="#777777", linewidth=0.45, linestyle=":")
            for summary in stats:
                for bound in ("q1", "med", "q3", "whislo", "whishi"):
                    if summary[bound] == 0:
                        summary[bound] = zero_slot
        else:
            ax.yaxis.set_major_locator(LogLocator(numticks=5))
        ax.tick_params(axis="y", which="minor", length=0)
    else:
        low, high = values.min(), values.max()
        margin = max(high - low, 0.1) * 0.10
        ax.set_ylim(low - margin, min(1.02, high + margin))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    positions = np.arange(1, len(methods) + 1)
    boxes = ax.bxp(
        stats, positions=positions, widths=0.52, patch_artist=True, showfliers=False,
        medianprops={"color": "#D97946", "linewidth": 0.7},
        boxprops={"edgecolor": "#444444", "linewidth": 0.45},
        whiskerprops={"color": "#444444", "linewidth": 0.45},
        capprops={"color": "#444444", "linewidth": 0.45},
    )
    for position, method, scores, box in zip(positions, methods, groups, boxes["boxes"], strict=True):
        color = plot.PALETTE[method]
        box.set_facecolor(color)
        box.set_alpha(0.72)
        offsets = position + np.linspace(-0.13, 0.13, len(scores))
        if zero_slot is not None:
            scores = np.where(scores == 0, zero_slot, scores)
        ax.scatter(offsets, scores, s=5, c=color, edgecolors="white", linewidths=0.18, alpha=0.9, zorder=3)

    ax.set_title(label, fontsize=7.3, fontweight="bold", pad=4)
    ax.set(xlim=(0.5, len(methods) + 0.5), xticks=positions)
    ax.set_xticklabels(methods, rotation=36, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", labelsize=5.4, pad=1.8, length=1.8, width=0.4)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.35, alpha=0.65)
    ax.set_axisbelow(True)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set(linewidth=0.45, color="#555555")


def render(metrics_path, results_dir, output_dir):
    configure_style()
    metrics = plot.load_metrics(metrics_path)
    long_table = plot.build_long_table(metrics)
    if len(long_table) != len(metrics) * len(plot.METRIC_SPECS):
        raise ValueError("Figure 2a requires all eight scores for each simulator/sample pair")
    medians = plot.build_median_table(long_table)

    figure = plt.figure(figsize=(WIDTH / 72, A_HEIGHT / 72))
    figure.text(7 / WIDTH, 1 - 4.8 / A_HEIGHT, "a", va="top", fontsize=13, fontweight="bold")
    figure.text(6.5 / WIDTH, 1 - 106 / A_HEIGHT, "Metric value", rotation=90,
                ha="center", va="center", fontsize=7.2)
    for index, spec in enumerate(plot.METRIC_SPECS):
        row, column = divmod(index, 4)
        ax = figure.add_axes(rect((29, 172, 316, 460)[column], (17, 131)[row], 120, 66, A_HEIGHT))
        draw_axis(ax, long_table, spec)

    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(results_dir / "plot_data_long.csv", index=False)
    medians.to_csv(results_dir / "metric_medians.csv", index=False)
    figure.savefig(output_dir / "a.pdf", metadata={"Title": "Figure 2a"})
    plt.close(figure)
    provenance = {
        "figure": "Figure 2a",
        "metrics_source": str(metrics_path.resolve().relative_to(PROJECT)) if metrics_path.resolve().is_relative_to(PROJECT) else str(metrics_path),
        "retained_rows": int(metrics["simulator"].ne("FEAST_reference_rank").sum()),
        "fresh_feast_rows": int(metrics["simulator"].eq("FEAST_reference_rank").sum()),
        "plotted_scores": len(long_table),
        "boxplots": "Cohort median order for each metric, 1.5 IQR whiskers, sample-sorted point offsets",
        "exact_zeros_by_log_metric": {
            spec["column"]: int(metrics[spec["column"]].eq(0).sum())
            for spec in plot.METRIC_SPECS if spec.get("yscale") == "log"
        },
        "exact_zero_display": "A labelled zero slot below the positive log range wherever needed.",
    }
    (results_dir / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=RESULTS / "metrics/simulator_quality_metrics.csv")
    parser.add_argument("--results-dir", type=Path, default=RESULTS / "figure2")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    render(args.metrics_csv, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
