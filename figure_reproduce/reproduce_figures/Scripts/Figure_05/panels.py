"""Journal panels for the 2D and 3D conditional transfer benchmarks."""
import json

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
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


def transfer(fig, zenodo):
    root = zenodo / S05
    points = pd.read_csv(root / "map_points.csv.gz")
    selected = pd.read_csv(root / "map_selection.csv")
    norm = mpl.colors.Normalize(0, np.log1p(points["count"].max()))
    context_gray = np.asarray(mpl.colors.to_rgb("#dddddd"))
    titles = ("Visium\nCross-section", "Visium\nWithheld half",
              "MERFISH\nCross-section", "MERFISH\nWithheld half")
    method_labels = ("Observed\nreference", "Held-out\ntruth", "FEAST", "Conditional\nresampling")
    left = (26, 92, 147, 244)
    widths = (55, 53, 93, 87)
    for col, case in enumerate(selected.itertuples(index=False)):
        subset = points.loc[points.column.eq(case.column)]
        observed = subset.loc[subset.method.eq("source") & subset.retained]
        center = left[col] + widths[col] / 2
        text_at(fig, center, 2, titles[col], ha="center", va="top", size=8, weight="bold", color="black", linespacing=1)
        text_at(fig, center, 29, case.gene, ha="center", va="top", size=8, style="italic")
        for row, method in enumerate(("source", "truth", "feast", "resampling")):
            r = subset.loc[subset.method.eq(method)]
            xy = r[["x", "y"]].to_numpy()
            populated = r.loc[r.retained]
            axis = axes_box(fig, left[col], 43 + 55 * row, widths[col], 53)
            size = 2.15 if col < 2 else .62
            axis.scatter(*xy.T, c="#e1e1e1", s=size, linewidths=0, rasterized=True)
            if np.isfinite(case.split) and row:
                color = mpl.colormaps["viridis"](norm(np.log1p(observed["count"])))
                color[:, :3] = .35 * color[:, :3] + .65 * context_gray
                axis.scatter(observed.x, observed.y, c=color, s=size, linewidths=0, rasterized=True)
            axis.scatter(populated.x, populated.y, c=np.log1p(populated["count"]), cmap="viridis", norm=norm,
                         s=size, linewidths=0, rasterized=True)
            map_axis(axis, xy)
            if np.isfinite(case.split):
                axis.axvline(case.split, color="#696969", ls=(0, (2, 2)), lw=.45)
                if row == 0:
                    for a, b, label in [(r.x.min(), case.split, "Observed"), (case.split, r.x.max(), "Held out")]:
                        axis.text((a+b)/2, 1.007, label, transform=axis.get_xaxis_transform(), color="#444444",
                                  ha="center", va="bottom", size=5.3)
            if col == 0:
                text_at(fig, 17, 69 + row*55, method_labels[row], rotation=90, size=7.8,
                        weight="bold", ha="center", va="center", color=BLUE if row == 2 else INK, linespacing=.95)
    cax = axes_box(fig, 89, 270, 157, 5)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax,
                       orientation="horizontal", ticks=np.arange(0, int(norm.vmax)+1))
    bar.ax.tick_params(labelsize=6, pad=1, length=1.4)
    bar.set_label("Expression · log1p(counts)", size=6.8, labelpad=.5)


def paired(fig, zenodo):
    df = pd.read_csv(zenodo / S05 / "panel_b_pairs.csv", dtype={"source": str, "target": str})
    order = ["mode", "dataset", "direction", "source", "target"]
    df = df.merge(task_order(df), on=order, validate="many_to_one")
    legend = [Line2D([], [], marker="o", color=color, linestyle="None", markersize=3, label=label)
              for color, label in [(BLUE, "FEAST"), ("#999999", "Conditional resampling mean")]]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.993, 1.012), ncols=2,
               columnspacing=1.3, handletextpad=.1, handlelength=.7, fontsize=6.8)
    for i, (metric, title) in enumerate(METRICS):
        row, col = divmod(i, 2)
        ax = axes_box(fig, 25 + col*120.5, 33 + row*76, 99, 45)
        plot_metric(ax, df, metric, title)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=6, pad=1.5)
        ax.tick_params(axis="x", length=1)
        if row == 0:
            ax.set_xticklabels([])
        else:
            for label in ax.get_xticklabels():
                label.set_fontsize(5.15)
                label.set_linespacing(.95)
        if col == 0 and row == 0:
            ax.set_ylabel("Metric value", labelpad=3, fontsize=6.6)


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
    for x, show in [(16, False), (109, True)]:
        axis = axes_box(fig, x, -3, 64, 62, projection="3d")
        render_study06_stack(axis, data, include_blueprint=True, include_generated=show)
        axis.set_box_aspect(axis.get_box_aspect(), zoom=1.03)
    axis = axes_box(fig, 84, 20, 24, 30)
    axis.annotate("", (.92, .70), (.08, .70), arrowprops={"arrowstyle": "->", "lw": .68})
    axis.axis("off")
    text_at(fig, 95, 36, "FEAST", weight="bold", ha="center", va="top", size=7.7)
    text_at(fig, 183, 14, "│  Reference", size=6.6, va="top")
    text_at(fig, 183, 31, "•  Generated", size=6.6, va="top", color="#555555")
    table = pd.read_csv(root / "coverage.csv")
    for j, (gap, color) in enumerate(zip(GAPS, ("#1683ad", "#dfae45", "#36a18b"))):
        arm = table.loc[table.gap.eq(gap)]
        ref = arm.source_kind.eq("real")
        top = 63 + j * 16.3
        ax = axes_box(fig, 23, top, 175, 7.7)
        ax.plot(arm.z[~ref], np.zeros((~ref).sum()), linestyle="none", marker=".", color="#969696", ms=1.43)
        ax.plot(arm.z[ref], np.zeros(ref.sum()), linestyle="none", marker="|", color=color, ms=4.5, mew=.8)
        ax.set(xlim=(-.08, arm.z.max() + .10), ylim=(-.7, .7), yticks=[])
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        text_at(fig, 23, top-1, f"{ref.sum()} measured / {(~ref).sum()} generated", size=6.9, va="bottom")
        if j < 2:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xticks([0, 4, 8, 12])
            ax.tick_params(axis="x", labelsize=5.7, pad=1)
            ax.set_xlabel("Slice z (mm)", size=6.6, labelpad=0)


def distributions(fig, zenodo):
    frame = pd.read_csv(zenodo / S06 / "panel_d_metrics.csv")
    support = pd.read_csv(zenodo / S06 / "panel_d_support.csv")
    names = ("FEAST", "Resampling", "SpatialZ")
    handles = [Patch(facecolor=benchmark_style.COLORS[method], label=name)
               for method, name in zip(benchmark_style.METHODS, names)]
    fig.legend(handles=handles, ncols=3, loc="upper left", bbox_to_anchor=(.026, .935), fontsize=8.4,
               columnspacing=2.5, handlelength=1.05)
    for index, (metric, title) in enumerate(METRICS):
        axis = axes_box(fig, 36 + index*143.5, 52, 111, 75)
        benchmark_style.panel(axis, frame, metric, title)
        group = support.loc[support.metric.eq(metric)].groupby("gap").size()
        axis.set_xticks(range(3), [f"{count}\nn={group[gap]}" for gap, count in zip(GAPS, REFERENCE_COUNTS)])
        axis.set_xlabel("")
        axis.tick_params(axis="both", labelsize=6.9, pad=1.3)
        axis.yaxis.set_major_locator(MaxNLocator(5))
        ylabel = "Wasserstein distance" if index < 2 else ("Moran I profile correlation" if index == 2 else
                                                           "Residual profile correlation")
        axis.set_ylabel(ylabel, fontsize=6.9, labelpad=2)
    text_at(fig, 291, 157, "Measured reference sections", size=7.45, ha="center", va="bottom")


def volumes(fig, zenodo):
    root = zenodo / S06
    all_points = pd.read_parquet(root / "native_volume.parquet")
    metadata = json.loads((root / "native_volume.json").read_text())
    bounds = limits(all_points)
    order = [("Observed", 0)] + [(method, gap) for gap in GAPS for method in ("FEAST", "SpatialZ")]
    left = 42
    width = 70.6
    for row, gene in enumerate(("Gad2", "Gfap")):
        top = 13 + row * 98
        norm = mpl.colors.Normalize(0, metadata["display_cap"][gene], clip=True)
        text_at(fig, 28, top + 49, gene, va="center", ha="center", size=11.7, rotation=90)
        for col, (method, gap) in enumerate(order):
            x = left + col * width
            if row == 0:
                text_at(fig, x + width/2, 3, method, ha="center", va="top", size=9.4, weight="bold")
            ax = axes_box(fig, x + (width-85)/2, top, 85, 91, projection="3d")
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
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=axes_box(fig, 546, top + 5, 4.2, 86))
        bar.ax.tick_params(labelsize=6.5, length=1.7, pad=1.7)
        bar.set_label("log1p counts", size=7.2, labelpad=2)
    for index, count in enumerate(REFERENCE_COUNTS):
        center = left + (2 + 2*index)*width
        ax = axes_box(fig, center - 33, 210, 66, 2)
        ax.plot([0, 1], [.5, .5], color=INK, linewidth=.55)
        ax.axis("off")
        text_at(fig, center, 216, f"{count} measured sections", ha="center", va="top", size=8.7)


PLOTS = {"a": transfer, "b": paired, "c": illustration, "d": distributions, "e": volumes}


def main(letter):
    plot = PLOTS[letter]
    args = panel_args(FIGURE, letter, plot.__name__).parse_args()
    fig = panel(letter, FRAMES)
    plot(fig, args.zenodo)
    save(fig, args.output)
