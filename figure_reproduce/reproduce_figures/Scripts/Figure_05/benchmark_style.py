"""Three-method section distributions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("feast", "resampling", "spatialz")
COLORS = {"feast": "#0072B2", "resampling": "#777777", "spatialz": "#B64342"}


# Drawing functions from 06_3d_stack/plot_resampling_comparison.py.


def panel(axis: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    gaps = [3, 5, 10]
    positions = np.arange(len(gaps), dtype=float)
    if metric.startswith('conditional_'):
        supported = frame.pivot(index=['gap','target_slice'], columns='method', values=metric).dropna().index
        frame = frame.set_index(['gap','target_slice']).loc[lambda x: x.index.isin(supported)].reset_index()
    method_frames = {method: frame[frame.method == method] for method in METHODS}
    for method, offset in zip(METHODS, (-.24, 0, .24)):
        data = [
            method_frames[method][method_frames[method]["gap"] == gap][metric].to_numpy(
                dtype=float
            )
            for gap in gaps
        ]
        box = axis.boxplot(
            data,
            positions=positions + offset,
            widths=0.19,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#24313F", "linewidth": 1.5},
            whiskerprops={"color": "#444444", "linewidth": 0.8},
            capprops={"color": "#444444", "linewidth": 0.8},
            boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.20)
            patch.set_edgecolor(COLORS[method])
        for gap_index, values in enumerate(data):
            jitter = (
                np.zeros(1)
                if len(values) == 1
                else np.linspace(-0.07, 0.07, len(values))
            )
            axis.scatter(
                np.full(len(values), positions[gap_index] + offset) + jitter,
                values,
                s=5.3,
                c=COLORS[method],
                edgecolors="white",
                linewidths=0.2,
                alpha=0.55,
                zorder=3,
            )
    axis.set_title(title, fontsize=8, fontweight="bold", pad=7, loc="left")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_linewidth(.45)
        axis.spines[spine_name].set_color("#555555")
