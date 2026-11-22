"""Render the six clustering maps per method with ARIs from the drawn partitions."""

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import adjusted_rand_score

from layout import CLUSTERING, panel, panel_args, save_panel

SECTION = "151676"
LABEL_KEYS = {"GraphST": "domain", "STAGATE_mclust": "mclust", "Leiden_unsupervised": "predicted_cluster"}
COLUMNS = (
    ("ground_truth", "Ground truth"),
    ("sparsity_neg_1", "Shift −1.00"),
    ("sparsity_neg_0_25", "Shift −0.25"),
    ("baseline", "FEAST baseline\nShift 0"),
    ("sparsity_pos_0_25", "Shift +0.25"),
    ("sparsity_pos_1", "Shift +1.00"),
)
PALETTE = ("#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#D4A6C8",
           "#D1BC32", "#426c6d")


def read_spots(path, key):
    section = ad.read_h5ad(path, backed="r")
    try:
        xy = np.asarray(section.obsm["spatial"])
        if not section.obs_names.is_unique or section.obs[key].isna().any():
            raise ValueError(f"Invalid spot IDs or labels in {path}")
        return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "label": section.obs[key].astype(str)},
                            index=section.obs_names.rename("spot_barcode"))
    finally:
        section.file.close()


def draw_map(axis, spots, limits):
    labels = sorted(spots.label.unique())
    mapping = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}
    colors = spots.label.map(mapping)
    axis.scatter(spots.x, spots.y, c=colors, s=0.75, alpha=0.84, linewidths=0, rasterized=True)
    axis.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    axis.set_axis_off()
    return colors


def main():
    args = panel_args("b", __doc__)
    study = args.results / "01_clustering"
    measured = read_spots(args.inputs / f"01_clustering/{SECTION}.h5ad", "ground_truth")
    limits = ((measured.x.min() - 80, measured.x.max() + 80),
              (measured.y.max() + 80, measured.y.min() - 80))
    tables = study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)

    fig = panel("b", label_x=1)
    fig.text(0.52, 0.983, "Adjusted Rand index (ARI)", ha="center", va="top", fontsize=8.6)
    axes = fig.subplots(3, 6)
    fig.subplots_adjust(left=0.033, right=0.996, top=0.875, bottom=0.047, wspace=0.055, hspace=0.31)
    for axis, (simulation, title) in zip(axes[0], COLUMNS):
        axis.set_title(title, fontsize=8.2, pad=3, weight="bold" if simulation in {"ground_truth", "baseline"} else "normal")

    maps = []
    aris = []
    for row, (method, (name, _, _)) in enumerate(CLUSTERING.items()):
        axes[row, 0].text(-0.13, 0.5, name, ha="right", va="center", rotation=90,
                          transform=axes[row, 0].transAxes, fontsize=8)
        for column, (simulation, _) in enumerate(COLUMNS):
            if column == 0:
                spots = measured.copy()
            else:
                path = study / "methods" / method / SECTION / simulation / "result.h5ad"
                spots = read_spots(path, LABEL_KEYS[method])
                if set(spots.index) != set(measured.index):
                    raise ValueError(f"Barcode support changed in {path}")
                spots = spots.loc[measured.index].copy()
                if not np.array_equal(spots[["x", "y"]], measured[["x", "y"]]):
                    raise ValueError(f"Spot coordinates changed in {path}")
                ari = adjusted_rand_score(measured.label, spots.label)
                axes[row, column].text(0.5, -0.045, f"{ari:.3f}", ha="center", va="top",
                                       transform=axes[row, column].transAxes, fontsize=9.1, style="italic")
                aris.append({"slice_id": SECTION, "method": method, "simulation_id": simulation,
                             "ARI": ari, "printed_ARI": f"{ari:.3f}", "n_spots": len(spots)})
            spots["color"] = draw_map(axes[row, column], spots, limits)
            spots["ground_truth"] = measured.label
            maps.append(spots.assign(slice_id=SECTION, method=method, simulation_id=simulation))
    top = axes[0, 0].get_position()
    bottom = axes[-1, 0].get_position()
    fig.add_artist(FancyBboxPatch(
        (top.x0 - 0.001, bottom.y0 - 0.023), top.width + 0.002, top.y1 - bottom.y0 + 0.028,
        boxstyle="round,pad=0.001,rounding_size=0.037",
        transform=fig.transFigure, fill=False, ec="#6849cc", lw=0.65,
    ))
    pd.DataFrame(aris).to_csv(tables / "b_ari.csv", index=False)
    pd.concat(maps).to_csv(tables / "b_spots.csv.gz")
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
