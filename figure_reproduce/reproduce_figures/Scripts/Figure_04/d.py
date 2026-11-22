"""Render the diagonal-affine uncorrected ladder and stored gene parameters."""

import json

import pandas as pd
from matplotlib.patches import Rectangle

from embedding_layout import draw_umap_pair
from layout import BATCH_COLORS, panel, panel_args, save_panel

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
STEM = "uncorrected_embedding_alpha_ladder"


def main():
    args = panel_args("d", __doc__)
    source = args.results / "plot_data/same_slice"
    embedding = pd.read_csv(source / f"{STEM}_plot_data.csv.gz")
    cloud = pd.read_csv(source / f"{STEM}_parameter_cloud_plot_data.csv")
    settings = json.loads((source / f"{STEM}_provenance.json").read_text())
    limits = settings["parameter_display_limits"]
    fig = panel("d", label_x=10)
    grid = fig.add_gridspec(3, 7, left=0.044, right=0.982, top=0.902, bottom=0.021,
                          height_ratios=[1, 1, 0.92], wspace=0.1, hspace=0.13)
    for column, alpha in enumerate(ALPHAS):
        views = [fig.add_subplot(grid[row, column]) for row in (0, 1)]
        points = embedding.loc[embedding.alpha.eq(alpha)]
        if points.groupby("batch").size().to_dict() != {"ref": 3611, "query": 3611}:
            raise ValueError(f"Incorrect UMAP spot support for alpha={alpha:g}")
        draw_umap_pair(views, points, point_size=0.63)
        top = views[0].get_position()
        fig.text(top.x0 + top.width / 2, 0.967, f"α = {alpha:g}", weight="bold",
                 fontsize=8.4, ha="center", va="top")
        if alpha == 0:
            fig.text(top.x0 + top.width / 2, 0.926, "Identical inputs", fontsize=6.8,
                     ha="center", va="top", weight="bold")
        if alpha == 1:
            lower = views[1].get_position()
            fig.add_artist(Rectangle((top.x0, lower.y0), top.width, top.y1 - lower.y0,
                                     transform=fig.transFigure, fill=False, ec="#949494", lw=0.55))
        if alpha == 1.25:
            stress_center = (grid[0, 5].get_position(fig).x0 + grid[0, 6].get_position(fig).x1) / 2
            fig.text(stress_center, 0.926, "Stress tests", fontsize=6.8, weight="bold", ha="center", va="top")
        parameter = cloud.loc[cloud.alpha.eq(alpha)]
        if len(parameter) != settings["selected_gene_count"]:
            raise ValueError(f"Parameter subset mismatch for alpha={alpha:g}")
        axis = fig.add_subplot(grid[2, column], projection="3d")
        axis.scatter(parameter.log_mean, parameter.log_variance, parameter.zero_proportion,
                     c=BATCH_COLORS["query"], s=0.95, alpha=0.28, depthshade=False,
                     linewidths=0, rasterized=True)
        axis.set(xlim=limits["log_mean"], ylim=limits["log_variance"], zlim=limits["zero_proportion"],
                 xticks=[-4, -3, -2], yticks=[-3.5, -2.5, -1.5], zticks=[0.8, 0.9, 1.0])
        axis.view_init(elev=settings["camera"]["elevation"], azim=settings["camera"]["azimuth"])
        axis.set_box_aspect((1, 1, 0.85), zoom=0.99)
        for direction in ("x", "y", "z"):
            axis.tick_params(axis=direction, labelsize=5.5, pad=-2)
        if column == 0:
            for row, text in enumerate(("Batch", "Cortical layer", "Gene parameters")):
                position = grid[row, 0].get_position(fig)
                fig.text(0.019, (position.y0 + position.y1) / 2, text, fontsize=7.4,
                         weight="bold", rotation=90, ha="center", va="center")
    fig.text(0.51, 0.27, "x  Log mean    |    y  Log variance    |    z  Zero proportion",
             fontsize=7.3, ha="center")
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
