"""Journal panels for the 2D and 3D conditional transfer benchmarks."""
import json

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

import benchmark_style
from feast_plate import BLUE, INK, axes_box, limits, panel, panel_args, save, style_3d, text_at
from layout import FIGURE, FRAMES
from stack_style import expression_grid, geometry_grid, render_study06_stack
from transfer_style import map_axis, plot_metric, task_order

S05 = "results/05_2d_conditional_transfer/figure05_sources"
S06 = "results/06_3d_stack/figure05_sources"
METRICS = (
    ("conditional_expression_wasserstein", "Conditional expression\nWasserstein ↓"),
    ("conditional_zero_fraction_wasserstein", "Conditional zero-fraction\nWasserstein ↓"),
    ("moran_profile_correlation", "Moran I profile\ncorrelation ↑"),
    ("residual_moran_profile_correlation", "Label-residual Moran I\ncorrelation ↑"),
)
GAPS = (3, 5, 10)
REFERENCE_COUNTS = (49, 30, 17)
PAIRED_TICK_STEPS = (.01, .01, .2, .2)
DISTRIBUTION_TICK_STEPS = (.025, .01, .2, .2)


def transfer(fig, zenodo):
    root = zenodo / S05
    points = pd.read_csv(root / "map_points.csv.gz")
    selected = pd.read_csv(root / "map_selection.csv")
    norm = mpl.colors.Normalize(0, np.log1p(points["count"].max()))
    context_gray = np.asarray(mpl.colors.to_rgb("#d8d8d8"))
    titles = ("Visium\nCross-section", "Visium\nWithheld half",
              "MERFISH\nCross-section", "MERFISH\nWithheld half")
    method_labels = ("Observed\nreference", "Held-out\ntruth", "FEAST", "Conditional\nresampling")
    text_centers = (47.8, 115.5, 198.3, 292.6)
    map_centers = (52, 117.5, 198.5, 292.5)
    widths = (52, 52, 99, 91)
    for col, case in enumerate(selected.sort_values("column").itertuples(index=False)):
        subset = points.loc[points.column.eq(case.column)]
        observed = subset.loc[subset.method.eq("source") & subset.retained]
        center, width = map_centers[col], widths[col]
        text_at(fig, text_centers[col], 2, titles[col], ha="center", va="top", size=8, weight="bold",
                color="black", linespacing=1)
        text_at(fig, text_centers[col], 30, case.gene, ha="center", va="top", size=8,
                style="italic", color="black")
        for row, method in enumerate(("source", "truth", "feast", "resampling")):
            r = subset.loc[subset.method.eq(method)]
            xy = r[["x", "y"]].to_numpy()
            populated = r.loc[r.retained]
            top = 44 + 55.85 * row
            axis = axes_box(fig, center - width / 2, top, width, 53.5)
            size = 1.5 if col < 2 else .30
            axis.scatter(*xy.T, c="#dddddd", s=size, linewidths=0, rasterized=True)
            if np.isfinite(case.split) and row:
                color = mpl.colormaps["viridis"](norm(np.log1p(observed["count"])))
                color[:, :3] = .35 * color[:, :3] + .65 * context_gray
                axis.scatter(observed.x, observed.y, c=color, s=size, linewidths=0, rasterized=True)
            axis.scatter(populated.x, populated.y, c=np.log1p(populated["count"]), cmap="viridis", norm=norm,
                         s=size, linewidths=0, rasterized=True)
            map_axis(axis, xy)
            if np.isfinite(case.split):
                axis.axvline(case.split, color="#555555", ls=(0, (3, 3)), lw=.34)
                if row == 0:
                    for a, b, label in [(r.x.min(), case.split, "Observed"), (case.split, r.x.max(), "Held out")]:
                        axis.text((a+b)/2, 1.007, label, transform=axis.get_xaxis_transform(), color="#444444",
                                  ha="center", va="bottom", size=5.3)
            if col == 0:
                text_at(fig, 17, top + 30, method_labels[row], rotation=90, size=8,
                        weight="bold", ha="center", va="center", color=BLUE if row == 2 else "#333333", linespacing=.95)
    cax = axes_box(fig, 88.5, 270, 157, 4.7)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax,
                       orientation="horizontal", ticks=np.arange(0, int(norm.vmax)+1))
    bar.outline.set_linewidth(.35)
    bar.ax.tick_params(labelsize=6.5, pad=1, length=1.4)
    bar.set_label("Expression · log1p(counts)", size=7, labelpad=0)


def paired(fig, zenodo):
    df = pd.read_csv(zenodo / S05 / "panel_b_pairs.csv", dtype={"source": str, "target": str})
    order = ["mode", "dataset", "direction", "source", "target"]
    df = df.merge(task_order(df), on=order, validate="many_to_one")
    legend = [Line2D([], [], marker="o", color=color, linestyle="None", markersize=3, label=label)
              for color, label in [(BLUE, "FEAST"), ("#999999", "Conditional resampling mean")]]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.98, 1.01), ncols=2,
               columnspacing=1.3, handletextpad=.1, handlelength=.7, fontsize=8)
    for i, (metric, title) in enumerate(METRICS):
        row, col = divmod(i, 2)
        ax = axes_box(fig, 25.5 + col * 119.5, 35.8 + row * 80.2, 100, 45)
        plot_metric(ax, df, metric, title)
        ax.yaxis.set_major_locator(MultipleLocator(PAIRED_TICK_STEPS[i]))
        ax.tick_params(axis="both", labelsize=6.5, pad=1.5, width=.3)
        ax.tick_params(axis="x", length=1.2)
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            for label in ax.get_xticklabels():
                label.set_fontsize(5.35)
                label.set_linespacing(.95)
        if col == 0 and row == 0:
            ax.set_ylabel("Metric value", labelpad=.5, fontsize=7)


def illustration(fig, zenodo):
    root = zenodo / S06
    parts = pd.read_parquet(root / "illustration_083.parquet")
    record = json.loads((root / "illustration_083.json").read_text())
    corners = parts[["x", "y"]].to_numpy()
    pad = np.ptp(corners, axis=0) * .025
    bounds = tuple(zip(corners.min(axis=0)-pad, corners.max(axis=0)+pad))
    data = {"context_slices": record["measured_slices"], "target": {"target_slice": 83,
            "lower_ref_slice": record["brackets"][0], "upper_ref_slice": record["brackets"][1]}, "limits": bounds}
    data["context_grids"] = {
        sid: expression_grid(d[["x", "y"]].to_numpy(), d.log1p_library.to_numpy(), bounds)
        for sid, d in parts.loc[parts.source.eq("measured")].groupby("slice_id")
    }
    data["lower_grid"], data["upper_grid"] = (data["context_grids"][sid] for sid in record["brackets"])
    generated = parts.loc[parts.source.eq("generated")]
    blueprint = parts.loc[parts.source.eq("blueprint")]
    data["blueprint_grid"] = geometry_grid(blueprint[["x", "y"]].to_numpy(), bounds)
    data["generated_grid"] = expression_grid(generated[["x", "y"]].to_numpy(), generated.log1p_library, bounds)
    data["norm"] = mpl.colors.Normalize(*parts.log1p_library.dropna().quantile(record["display_quantiles"]), clip=True)
    for x, show in [(17, False), (110, True)]:
        axis = axes_box(fig, x, 1, 64, 61, projection="3d")
        render_study06_stack(axis, data, include_blueprint=True, include_generated=show)
        axis.set_box_aspect(axis.get_box_aspect(), zoom=1.03)
    axis = axes_box(fig, 88, 19, 22, 30)
    axis.annotate("", (.92, .70), (.08, .70), arrowprops={"arrowstyle": "->", "lw": .68})
    axis.axis("off")
    text_at(fig, 99, 35, "FEAST", weight="bold", ha="center", va="top", size=8.3, color="black")
    width, height = fig.get_size_inches() * 72
    for top, marker, color, label in [(18, "|", INK, "Reference"), (34, ".", "#999999", "Generated")]:
        legend = Line2D([186 / width], [1 - top / height], transform=fig.transFigure,
                        marker=marker, linestyle="none", color=color, markersize=4)
        text_at(fig, 191, top, label, size=7, va="center")
        fig.add_artist(legend)
    table = pd.read_csv(root / "coverage.csv")
    depth_limits = (table.z.min() - .08, table.z.max() + .10)
    for j, (gap, color) in enumerate(zip(GAPS, ("#1683ad", "#dfae45", "#36a18b"))):
        arm = table.loc[table.gap.eq(gap)]
        ref = arm.source_kind.eq("real")
        top = 70 + j * 17.3
        ax = axes_box(fig, 23.4, top, 168.5, 7.7)
        ax.plot(arm.z[~ref], np.zeros((~ref).sum()), linestyle="none", marker=".", color="#969696", ms=1.43)
        ax.plot(arm.z[ref], np.zeros(ref.sum()), linestyle="none", marker="|", color=color, ms=4.5, mew=.8)
        ax.set(xlim=depth_limits, ylim=(-.7, .7), yticks=[])
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        text_at(fig, 23.4, top - .6, f"{ref.sum()} measured / {(~ref).sum()} generated", size=7, va="bottom", color="black")
        if j < 2:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xticks([0, 4, 8, 12])
            ax.tick_params(axis="x", labelsize=6.5, pad=1, length=1.5, width=.4)
            ax.spines["bottom"].set(linewidth=.4, color="#616b75")
            ax.set_xlabel("Slice z", size=7, labelpad=0)


def distributions(fig, zenodo):
    frame = pd.read_csv(zenodo / S06 / "panel_d_metrics.csv")
    support = pd.read_csv(zenodo / S06 / "panel_d_support.csv")
    names = ("FEAST", "Resampling", "SpatialZ")
    handles = [Patch(facecolor=benchmark_style.COLORS[method], label=name)
               for method, name in zip(benchmark_style.METHODS, names)]
    fig.legend(handles=handles, ncols=3, loc="upper left", bbox_to_anchor=(.017, .929), fontsize=8,
               columnspacing=2.8, handlelength=1.15)
    for index, (metric, title) in enumerate(METRICS):
        axis = axes_box(fig, 33.5 + index * 142.75, 55.9, 110, 74)
        benchmark_style.panel(axis, frame, metric, title)
        group = support.loc[support.metric.eq(metric)].groupby("gap").size()
        axis.set_xticks(range(3), [f"{count}\nn={group[gap]}" for gap, count in zip(GAPS, REFERENCE_COUNTS)])
        axis.tick_params(axis="both", labelsize=6.5, pad=1.4, width=.45)
        axis.yaxis.set_major_locator(MultipleLocator(DISTRIBUTION_TICK_STEPS[index]))
        ylabel = "Wasserstein distance" if index < 2 else ("Moran's I profile correlation" if index == 2 else
                                                           "Residual profile correlation")
        axis.set_ylabel(ylabel, fontsize=7, labelpad=2)
    text_at(fig, 302.5, 154.5, "Measured reference sections", size=7, ha="center", va="bottom", color="black")


def volumes(fig, zenodo):
    root = zenodo / S06
    all_points = pd.read_parquet(root / "native_volume.parquet")
    metadata = json.loads((root / "native_volume.json").read_text())
    bounds = limits(all_points)
    order = [("Observed", 0)] + [(method, gap) for gap in GAPS for method in ("FEAST", "SpatialZ")]
    left = 38.4
    width = 70
    for row, gene in enumerate(("Gad2", "Gfap")):
        top = 13 + row * 100.4
        norm = mpl.colors.Normalize(0, metadata["display_cap"][gene], clip=True)
        text_at(fig, 29, top + 45, gene, va="center", ha="center", size=14, rotation=90)
        for col, (method, gap) in enumerate(order):
            x = left + col * width
            if row == 0:
                text_at(fig, x + width / 2, 3, method, ha="center", va="top", size=10, weight="bold",
                        color=INK if gap == 0 else "black")
            ax = axes_box(fig, x + (width - 85) / 2, top + 2.5, 85, 91, projection="3d")
            ax.patch.set_alpha(0)
            points = all_points.loc[all_points.method.eq(method) & all_points.gap.eq(gap)]
            ax.scatter(points.x, points.y, points.z, color="#ccd3d9", s=.18, alpha=.08,
                       linewidths=0, depthshade=False, rasterized=True)
            for kind, marker, size in [("real", "^", .18), ("simulated", "o", .14)]:
                selected = points.loc[points.source_kind.eq(kind) & points[gene].gt(0)].sort_values(gene, kind="stable")
                ax.scatter(selected.x, selected.y, selected.z, c=np.log1p(selected[gene]), norm=norm, cmap="viridis",
                           s=size, marker=marker, alpha=.85, linewidths=0, depthshade=False, rasterized=True)
            style_3d(ax, bounds, 25, -55)
            ax.set_box_aspect(np.ptp(bounds, axis=1), zoom=1.27)
        cax = axes_box(fig, 543, top + 4, 3.6, 87.5)
        tick_step = .5 if row == 0 else 1
        ticks = np.arange(0, norm.vmax, tick_step)
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, ticks=ticks)
        bar.outline.set_linewidth(.4)
        bar.ax.tick_params(labelsize=6.5, length=1.7, pad=1.7)
        bar.set_label("log1p counts", size=7, labelpad=2)
    for index, count in enumerate(REFERENCE_COUNTS):
        center = left + (2 + 2*index)*width
        ax = axes_box(fig, center - 35, 208, 70, 2)
        ax.plot([0, 1], [.5, .5], color="black", linewidth=.75)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        text_at(fig, center, 212, f"{count} measured sections", ha="center", va="top", size=10, color="black")


PLOTS = {"a": transfer, "b": paired, "c": illustration, "d": distributions, "e": volumes}


def main(letter):
    plot = PLOTS[letter]
    args = panel_args(FIGURE, letter, plot.__name__).parse_args()
    fig = panel(letter, FRAMES)
    plot(fig, args.zenodo)
    save(fig, args.output)
