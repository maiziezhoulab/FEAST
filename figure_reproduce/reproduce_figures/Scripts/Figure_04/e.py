"""Render diagonal-affine alpha-one embeddings and reference-vote layer maps."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from embedding_layout import draw_umap_pair, handles
from layout import BATCH_COLORS, LAYER_COLORS, panel, panel_args, save_panel

METHODS = ("Uncorrected", "GraphST", "STAMP", "scVI")


def draw_spatial(axis, data, label, limits):
    axis.scatter(data.x, data.y, c=data[label].map(LAYER_COLORS), s=1.27,
                 linewidths=0, rasterized=True)
    axis.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    axis.axis("off")


def main():
    args = panel_args("e", __doc__)
    source = args.results / "plot_data/same_slice"
    table = pd.read_csv(source / "representative_embedding_context_data.csv")
    pred = pd.read_csv(source / "same_slice_layer_transfer_spatial_predictions.csv")
    scores = pd.read_csv(source / "same_slice_layer_transfer_spatial_scores.csv")
    canonical = pred.loc[pred.method.eq("Uncorrected")].set_index("source_spot_id")
    if len(canonical) != 3611 or not canonical.index.is_unique:
        raise ValueError("Expected 3,611 unique paired DLPFC spot IDs")
    xy = canonical[["x", "y"]].to_numpy()
    padding = np.ptp(xy, axis=0) * 0.04
    low, high = xy.min(axis=0) - padding, xy.max(axis=0) + padding
    limits = ((low[0], high[0]), (high[1], low[1]))

    fig = panel("e", label_x=10)
    axes = fig.subplots(3, 5, gridspec_kw={"height_ratios": [1, 1, 1.16]})
    fig.subplots_adjust(left=0.027, right=0.984, top=0.913, bottom=0.013, wspace=0.15, hspace=0.18)
    legend_layout = [
        (axes[0, 0], "Batch mixing", BATCH_COLORS, 1),
        (axes[1, 0], "Layer structure", LAYER_COLORS, 2),
    ]
    for axis, title, palette, ncols in legend_layout:
        axis.axis("off")
        axis.text(0.03, 0.945, title, weight="bold", va="top", ha="left", fontsize=9, transform=axis.transAxes)
        axis.legend(handles=handles(palette), ncols=ncols, loc="upper left", bbox_to_anchor=(0, 0.76),
                    fontsize=7.5, handletextpad=0.36, handlelength=1.2, columnspacing=0.65)
    axes[0, 0].text(0.64, 1.17, "α = 1", weight="bold", fontsize=8.4,
                    transform=axes[0, 0].transAxes, va="top")
    draw_spatial(axes[2, 0], canonical, "ground_truth", limits)
    axes[2, 0].set_title("Ground truth · DLPFC 151673", size=7.55, weight="bold", pad=4)
    fig.text(0.603, 0.347, "Predicted layers · Macro-F1", fontsize=8.6,
             weight="bold", va="bottom", ha="center")
    for column, method in enumerate(METHODS, start=1):
        name = "Uncorrected input" if method == "Uncorrected" else method
        embedding = table.loc[table.representation.eq(name)]
        prediction = pred.loc[pred.method.eq(method)].set_index("source_spot_id")
        if len(embedding) != 7222 or len(prediction) != 3611 or not prediction.index.is_unique:
            raise ValueError(f"Unexpected display support for {method}")
        prediction = prediction.loc[canonical.index]
        pd.testing.assert_frame_equal(prediction[["x", "y", "ground_truth"]], canonical[["x", "y", "ground_truth"]])
        for batch in BATCH_COLORS:
            labels = embedding.loc[embedding.batch.eq(batch)].set_index("source_spot_id").ground_truth
            if not labels.index.is_unique or not labels.loc[canonical.index].equals(canonical.ground_truth):
                raise ValueError(f"Unexpected display label support for {method} {batch}")
        draw_umap_pair(axes[:2, column], embedding, point_size=0.72)
        axes[0, column].set_title(name, fontsize=9.1, weight="bold", pad=4)
        draw_spatial(axes[2, column], prediction, "predicted_layer", limits)
        stored = scores.loc[scores.method.eq(method), "macro_f1"].item()
        observed = f1_score(prediction.ground_truth, prediction.predicted_layer, labels=list(LAYER_COLORS),
                            average="macro", zero_division=0)
        if not np.isclose(observed, stored, atol=1e-12, rtol=0):
            raise ValueError(f"Printed macro-F1 differs from predictions for {method}")
        axes[2, column].set_title(f"{observed:.3f}", fontsize=8.2, weight="bold", pad=4)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
