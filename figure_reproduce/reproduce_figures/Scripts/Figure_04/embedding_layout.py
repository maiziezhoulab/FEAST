"""Common drawing primitives for Figure 4's paired embedding views."""

import numpy as np
from matplotlib.lines import Line2D

from layout import BATCH_COLORS, LAYER_COLORS


def draw_umap_pair(axes, data, *, point_size):
    coordinates = data[["umap_1", "umap_2"]].to_numpy()
    if not np.isfinite(coordinates).all() or not len(coordinates):
        raise ValueError("Invalid saved embedding coordinates")
    order = np.random.default_rng(42).permutation(len(data))
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    center = (lower + upper) / 2
    radius = (upper - lower).max() * 0.54
    xy = coordinates[order]
    for axis, key, palette in zip(axes, ("batch", "ground_truth"), (BATCH_COLORS, LAYER_COLORS)):
        axis.scatter(xy[:, 0], xy[:, 1], c=data[key].map(palette).to_numpy()[order],
                     s=point_size, linewidths=0, alpha=0.88, rasterized=True)
        axis.set(xlim=(center[0] - radius, center[0] + radius),
                 ylim=(center[1] - radius, center[1] + radius), aspect="equal")
        axis.axis("off")


def handles(palette):
    labels = {"ref": "Reference", "query": "Query"}
    return [Line2D([], [], marker="o", linestyle="none", color=color, markersize=4.5,
                   label=labels.get(key, key.replace("_", " ")))
            for key, color in palette.items()]
