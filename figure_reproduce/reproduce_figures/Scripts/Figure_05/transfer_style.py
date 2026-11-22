"""Drawing functions for the published 2D markers and paired task metrics."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = (
    ("dlpfc", "cross_slice", "DLPFC\ncross-slice"),
    ("dlpfc", "mask_half", "DLPFC\nhalf-slice"),
    ("merfish", "cross_slice", "MERFISH\ncross-slice"),
    ("merfish", "mask_half", "MERFISH\nhalf-slice"),
)


# Drawing functions from 05_2d_conditional_transfer/plot_expanded_report.py.


def map_axis(ax, xy):
    span = np.ptp(xy, axis=0)
    ax.set_xlim(xy[:, 0].min() - span[0] * 0.03, xy[:, 0].max() + span[0] * 0.03)
    ax.set_ylim(xy[:, 1].max() + span[1] * 0.03, xy[:, 1].min() - span[1] * 0.03)
    ax.set_aspect("equal")
    ax.set_axis_off()


# Drawing functions from 05_2d_conditional_transfer/plot_resampling_comparison.py.


def task_label(row: pd.Series) -> str:
    if row["dataset"] == "dlpfc":
        if row["mode"] == "mask_half":
            return f"D {str(row['source'])[-3:]} half"
        return f"D {str(row['source'])[-3:]}→{str(row['target'])[-3:]}"
    if row["mode"] == "mask_half":
        return f"M {str(row['source'])[-3:]} half"
    return f"M {str(row['source'])[-3:]}→{str(row['target'])[-3:]}"


def task_order(frame: pd.DataFrame) -> pd.DataFrame:
    tasks = frame[
        list(("mode", "dataset", "direction", "source", "target"))
    ].drop_duplicates()
    tasks["dataset_order"] = tasks["dataset"].map({"dlpfc": 0, "merfish": 1})
    tasks["mode_order"] = tasks["mode"].map({"cross_slice": 0, "mask_half": 1})
    tasks = tasks.sort_values(
        ["dataset_order", "mode_order", "source", "target", "direction"]
    ).reset_index(drop=True)
    tasks["task_index"] = np.arange(len(tasks))
    tasks["task_label"] = tasks.apply(task_label, axis=1)
    return tasks


def plot_metric(axis: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    metric_rows = frame[frame["metric"] == metric].sort_values("task_index")
    for group_index, (dataset, mode, _label) in enumerate(GROUPS):
        rows = metric_rows[
            (metric_rows["dataset"] == dataset) & (metric_rows["mode"] == mode)
        ]
        jitter = (
            np.zeros(1) if len(rows) == 1 else np.linspace(-0.055, 0.055, len(rows))
        )
        baseline_x = np.full(len(rows), group_index - 0.14) + jitter
        feast_x = np.full(len(rows), group_index + 0.14) + jitter
        for left_x, right_x, left_y, right_y in zip(
            baseline_x,
            feast_x,
            rows["baseline_mean"],
            rows["feast"],
            strict=True,
        ):
            axis.plot(
                [left_x, right_x],
                [left_y, right_y],
                color="#B9B9B9",
                linewidth=0.55,
                alpha=0.75,
                zorder=1,
            )
        axis.scatter(
            baseline_x,
            rows["baseline_mean"],
            s=5.4,
            facecolor="#999999",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        axis.scatter(
            feast_x,
            rows["feast"],
            s=6.5,
            facecolor="#0072B2",
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axis.hlines(
            np.median(rows["baseline_mean"]),
            group_index - 0.24,
            group_index - 0.04,
            color="#D97946",
            linewidth=0.80,
            zorder=4,
        )
        axis.hlines(
            np.median(rows["feast"]),
            group_index + 0.04,
            group_index + 0.24,
            color="#D97946",
            linewidth=0.80,
            zorder=4,
        )
    axis.set_title(title, fontsize=8, fontweight="bold", pad=7, color="black")
    axis.set_xticks(
        np.arange(len(GROUPS)), [label for _dataset, _mode, label in GROUPS]
    )
    axis.set_xlim(-.39, len(GROUPS) - .61)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_linewidth(.3)
        axis.spines[spine_name].set_color("#555555")
