"""Render the six clustering maps per method with ARIs from the drawn partitions."""

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import adjusted_rand_score

from layout import CLUSTERING, manuscript_axes, manuscript_text, panel, panel_args, save_panel

SECTION = "151676"
LABEL_KEYS = {"GraphST": "domain", "STAGATE_mclust": "mclust", "Leiden_unsupervised": "predicted_cluster"}
COLUMNS = (
    ("ground_truth", "Ground Truth"),
    ("sparsity_neg_1", "Shift −1.00"),
    ("sparsity_neg_0_25", "Shift −0.25"),
    ("baseline", "FEAST baseline"),
    ("sparsity_pos_0_25", "Shift +0.25"),
    ("sparsity_pos_1", "Shift +1.00"),
)
PALETTE = ("#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#D4A6C8",
           "#D1BC32", "#426c6d")
COLUMN_CENTERS = (189.8, 262.0, 334.2, 406.2, 478.3, 550.3)
ROW_CENTERS = (77.6, 174.3, 271.4)


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

    fig = panel("b")
    manuscript_text(fig, "b", 370.5, 10.0, "Adjusted Rand index (ARI)", ha="center", fontsize=8.2)
    for x, (simulation, title) in zip(COLUMN_CENTERS, COLUMNS, strict=True):
        highlighted = simulation in {"ground_truth", "baseline"}
        manuscript_text(fig, "b", x, 26.3 if simulation == "baseline" else 33.3, title,
                        ha="center", fontsize=8.8, weight="bold" if highlighted else "normal")
        if simulation == "baseline":
            manuscript_text(fig, "b", x, 38.4, "Shift 0", ha="center", fontsize=8.8)

    maps = []
    aris = []
    for y, (method, (name, _, _)) in zip(ROW_CENTERS, CLUSTERING.items(), strict=True):
        manuscript_text(fig, "b", 145.4, y, name, ha="center", va="center", rotation=90, fontsize=8)
        for x, (simulation, _) in zip(COLUMN_CENTERS, COLUMNS, strict=True):
            axis = manuscript_axes(fig, "b", x - 34.8, y - 33.4, 69.6, 66.8)
            if simulation == "ground_truth":
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
                manuscript_text(fig, "b", x, y + 44.1, f"{ari:.3f}", ha="center",
                                fontsize=9.7, style="italic", color="black")
                aris.append({"slice_id": SECTION, "method": method, "simulation_id": simulation,
                             "ARI": ari, "printed_ARI": f"{ari:.3f}", "n_spots": len(spots)})
            spots["color"] = draw_map(axis, spots, limits)
            spots["ground_truth"] = measured.label
            maps.append(spots.assign(slice_id=SECTION, method=method, simulation_id=simulation))
    width, height = fig.get_size_inches() * 72
    fig.add_artist(FancyBboxPatch(
        (12.5 / width, 1 - 310.0 / height), 73.1 / width, (310.0 - 41.1) / height,
        boxstyle="round,pad=0.001,rounding_size=0.037",
        transform=fig.transFigure, fill=False, ec="#6849cc", lw=0.65,
    ))
    pd.DataFrame(aris).to_csv(tables / "b_ari.csv", index=False)
    pd.concat(maps).to_csv(tables / "b_spots.csv.gz")
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
