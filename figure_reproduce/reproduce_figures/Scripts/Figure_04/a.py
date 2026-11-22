"""Render the two integration designs and the roles of the plotted conditions."""

import pandas as pd
from matplotlib.patches import Rectangle

from layout import panel, panel_args, save_panel


def main():
    args = panel_args("a", __doc__)
    table = pd.read_csv(args.results / "plot_data/design_table.csv")
    fig = panel("a", label_x=1)
    axis = fig.add_axes((0.024, 0, 0.968, 1), zorder=-1)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    split, head_bottom, head_top, row_bottom = 0.58, 0.793, 0.967, 0.245
    columns = (0.014, 0.203, 0.60)
    for low, high, color in [(0, split, "#eef4f9"), (split, 1, "#f9f1e9")]:
        axis.add_patch(Rectangle((low, head_bottom), high - low, head_top - head_bottom,
                                 facecolor=color, edgecolor="none"))
    for x, title, color in zip(columns, table.columns, ["#24313f", "#35679c", "#a46737"]):
        axis.text(x, 0.878, title, va="center", size=8.7, color=color, weight="bold")
    row_height = (head_bottom - row_bottom) / len(table)
    for i, row in enumerate(table.itertuples(index=False, name=None)):
        y = head_bottom - (i + 0.5) * row_height
        for j, (x, value) in enumerate(zip(columns, row)):
            axis.text(x, y, value, va="center", fontsize=6.8,
                      color="#25313f", weight="bold" if j == 0 else "normal")
        axis.hlines(y - row_height / 2, 0, 1, color="#d9dfe5", lw=0.52)
    axis.hlines(head_bottom, 0, 1, color="#acb7c2", lw=0.7)
    axis.vlines(split, row_bottom, head_top, color="#d0d8df", lw=0.55)
    axis.text(columns[0], 0.145, "α=0  Identical same-slice inputs · two-slice resampling control",
              size=6.6, va="center", color="#4e5964")
    axis.text(columns[0], 0.055, "α=1  Primary endpoint     α>1  Extrapolation stress tests",
              size=6.6, va="center", color="#4e5964")
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
