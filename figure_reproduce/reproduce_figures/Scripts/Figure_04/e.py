"""Render diagonal-affine alpha-one embeddings and reference-vote layer maps."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from embedding_layout import draw_umap_pair, handles
from layout import BATCH_COLORS, LAYER_COLORS, axes_box, panel, panel_args, save_panel, text_at

METHODS = ("Uncorrected", "GraphST", "STAMP", "scVI")


def draw_spatial(axis, data, label, limits):
    axis.scatter(data.x, data.y, c=data[label].map(LAYER_COLORS), s=1.05,
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
    legend_layout = [
        (22, "Batch mixing", BATCH_COLORS, 1),
        (76.5, "Layer structure", LAYER_COLORS, 2),
    ]
    for top, title, palette, ncols in legend_layout:
        text_at(fig, 20, top, title, weight="bold", va="top", ha="left", fontsize=10)
        axis = axes_box(fig, 19, top + 15, 100, 57)
        axis.axis("off")
        axis.legend(handles=handles(palette), ncols=ncols, loc="upper left", fontsize=8.1,
                    handletextpad=0.45, handlelength=1.25, columnspacing=0.65)
    text_at(fig, 68.5, 10.5, "α = 1", weight="bold", fontsize=8.3, va="top")
    text_at(fig, 28, 172, "Ground Truth: DLPFC\n151673", fontsize=8.15, weight="bold", va="top")
    ground_truth = axes_box(fig, 29.5, 192.5, 78, 78)
    draw_spatial(ground_truth, canonical, "ground_truth", limits)
    text_at(fig, 364, 172, "Predicted layers · Macro-F1", fontsize=8.15,
            weight="bold", va="top", ha="center")
    for center, method in zip((190, 306, 422, 538), METHODS):
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
        views = [axes_box(fig, center - 39, top, 78, 78) for top in (28, 99)]
        draw_umap_pair(views, embedding, point_size=0.51)
        text_at(fig, center, 15.5, name, fontsize=9.3, weight="bold", ha="center", va="top")
        spatial = axes_box(fig, center - 39, 192.5, 78, 78)
        draw_spatial(spatial, prediction, "predicted_layer", limits)
        stored = scores.loc[scores.method.eq(method), "macro_f1"].item()
        observed = f1_score(prediction.ground_truth, prediction.predicted_layer, labels=list(LAYER_COLORS),
                            average="macro", zero_division=0)
        if not np.isclose(observed, stored, atol=1e-12, rtol=0):
            raise ValueError(f"Printed macro-F1 differs from predictions for {method}")
        text_at(fig, center, 182, f"{observed:.3f}", fontsize=8.2, weight="bold", ha="center", va="top")
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
