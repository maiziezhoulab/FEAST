"""Render the diagonal-affine uncorrected ladder and stored gene parameters."""

import json

import pandas as pd

from embedding_layout import draw_umap_pair
from layout import BATCH_COLORS, axes_box, panel, panel_args, save_panel, text_at

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
    centers = (57.5, 137.5, 217.5, 297.5, 377.5, 457.5, 537.5)
    for y, label in ((54.5, "Batch"), (131.5, "Cortical layer"), (207.5, "Gene parameters")):
        text_at(fig, 11.5, y, label, fontsize=7.2, weight="bold",
                rotation=90, ha="center", va="center")
    text_at(fig, (centers[-2] + centers[-1]) / 2, 12.4, "Stress tests",
            fontsize=6.8, weight="bold", ha="center", va="top")
    for column, alpha in enumerate(ALPHAS):
        center = centers[column]
        views = [axes_box(fig, center - 36, top, 72, 72) for top in (18.5, 95.5)]
        points = embedding.loc[embedding.alpha.eq(alpha)]
        if points.groupby("batch").size().to_dict() != {"ref": 3611, "query": 3611}:
            raise ValueError(f"Incorrect UMAP spot support for alpha={alpha:g}")
        draw_umap_pair(views, points, point_size=0.45)
        text_at(fig, center, 2, f"α = {alpha:g}", weight="bold", fontsize=8.1, ha="center", va="top")
        if alpha == 0:
            text_at(fig, center, 12.4, "Identical inputs", fontsize=6.8,
                    ha="center", va="top", weight="bold")
        parameter = cloud.loc[cloud.alpha.eq(alpha)]
        if len(parameter) != settings["selected_gene_count"]:
            raise ValueError(f"Parameter subset mismatch for alpha={alpha:g}")
        axis = axes_box(fig, center - 38, 172, 72, 72, projection="3d")
        axis.scatter(parameter.log_mean, parameter.log_variance, parameter.zero_proportion,
                     c=BATCH_COLORS["query"], s=0.95, alpha=0.28, depthshade=False,
                     linewidths=0, rasterized=True)
        axis.set(xlim=limits["log_mean"], ylim=limits["log_variance"], zlim=limits["zero_proportion"],
                 xticks=[-4, -3, -2], yticks=[-3.5, -2.5, -1.5], zticks=[0.8, 0.9, 1.0])
        axis.view_init(elev=settings["camera"]["elevation"], azim=settings["camera"]["azimuth"])
        axis.set_box_aspect((1, 1, 0.85), zoom=1.0)
        axis.tick_params(labelsize=5.3, pad=-3.5)
        axis.tick_params(axis="y", labelsize=4.9)
    text_at(fig, 297.5, 169.5, "x: log mean   |   y: log variance   |   z: zero proportion",
            fontsize=7.3, ha="center", va="top")
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
