"""Visualize the dataset-specific fidelity and reference-ordered gene clouds."""
import json

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from feast_plate import axes_box, text_at

DATA = "results/00_simulator_benchmark/s1"
METHODS = ["FEAST", "SRTsim", "Splatter", "Splatter Simple", "scCube"]
LABELS = ["DLPFC 151670", "DLPFC 151675", "DLPFC 151676", "MERFISH 006", "MERFISH 007",
          "MERFISH 120", "Open-ST 005", "Open-ST 006", "Slide-seq 001", "Stereo-seq E10.5",
          "Stereo-seq E12.5", "Xenium lymph node"]
METRICS = (
    ("input_mean_corr", "Mean correlation ↑"),
    ("input_variance_corr", "Variance correlation ↑"),
    ("relative_error_mean", "Mean relative error ↓"),
    ("moran_i_correlation", "Moran I correlation ↑"),
    ("gene_zero_fraction_wasserstein", "Zero-fraction\ndistance ↓"),
    ("zero_mask_jaccard", "Zero-mask Jaccard ↑"),
    ("cosine_divergence", "Cosine divergence ↓"),
    ("library_size_wasserstein", "Library-size\ndistance ↓"),
)


def number(value):
    if value == 0 or .01 <= abs(value) < 1000:
        return f"{value:g}"
    return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e−").replace("e-", "e−").replace("e+", "e")


def heatmaps(fig, zenodo):
    """Show eight metric-specific heatmaps over the full benchmark."""
    root = zenodo / DATA
    table = pd.read_csv(root / "Fig_S1_metrics.csv")
    meta = json.loads((root / "provenance.json").read_text())
    text_at(fig, 282, 4, "Dataset-specific simulation fidelity", ha="center", va="top", size=11.3, weight="bold")
    for i, (metric, title) in enumerate(METRICS):
        row, col = divmod(i, 4)
        x, top = 109.5 + 115.8 * col, 36.5 + 191.5 * row
        ax = axes_box(fig, x, top, 94.5, 115)
        values = table.loc[table.metric.eq(metric)].pivot(index="sample", columns="method", values="value")
        scale = meta["metric_scales"][metric]
        lo, hi = scale["limits"]
        norm = mpl.colors.LogNorm(lo, hi) if scale["kind"] == "log" else mpl.colors.Normalize(lo, hi)
        cmap = "RdBu" if scale["kind"] == "correlation" else "Blues_r" if scale["direction"] == "lower" else "Blues"
        image = ax.imshow(values.loc[meta["sample_order"], METHODS].to_numpy(), cmap=cmap, norm=norm,
                          aspect="auto", interpolation="nearest")
        ax.set_xticks(range(5), ["FEAST", "SRTsim", "Splatter", "Spl. S.", "scCube"],
                      rotation=46, ha="right", rotation_mode="anchor", fontsize=8.1)
        ax.set_yticks(range(12), LABELS if col == 0 else [""]*12, fontsize=8.15)
        ax.tick_params(length=0, pad=1.5)
        ax.spines[:].set_visible(False)
        ax.set_title(title, fontsize=8.8, weight="bold", pad=4, linespacing=1)
        for separator in (2.5, 5.5, 7.5, 8.5, 10.5):
            ax.axhline(separator, lw=.75, color="white")
        midpoint = 10**round((np.log10(lo)+np.log10(hi))/2) if scale["kind"] == "log" else (lo+hi)/2
        ticks = [lo, midpoint, hi]
        bar = fig.colorbar(image, cax=axes_box(fig, x, top+151, 94.5, 4.5), orientation="horizontal", ticks=ticks)
        bar.set_ticklabels([number(v).replace("-", "−") for v in ticks])
        bar.outline.set_visible(False)
        bar.ax.minorticks_off()
        bar.ax.tick_params(length=1.5, labelsize=7.9, pad=1)


def cloud(fig, zenodo, sample, heading, colorbar):
    root = zenodo / DATA
    table = pd.read_csv(root / "Fig_S1_clouds.csv.gz", dtype={"gene_id":str})
    table = table.loc[table["sample"].eq(sample)]
    meta = json.loads((root / "provenance.json").read_text())["cloud_limits"][sample]
    n_genes = table.gene_id.nunique()
    n_spots = table.n_spots.iloc[0]
    text_at(fig, 282, 9, f"{heading} · {n_genes:,} genes · {n_spots:,} spots per source",
            ha="center", va="top", size=10.8, weight="bold")
    for i, source in enumerate(["Reference", *METHODS]):
        x = 41 + 87.9*i
        ax = axes_box(fig, x, 48, 76, 100)
        selected = table.loc[table.source.eq(source)]
        points = ax.scatter(selected.log1p_mean, selected.log1p_variance, c=selected.zero_fraction,
                            cmap="viridis", vmin=0, vmax=1, s=1.1 if n_genes > 2000 else 2.4,
                            alpha=.70, linewidths=0, rasterized=True)
        ax.set(xlim=meta["log1p_mean"], ylim=meta["log1p_variance"])
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.spines[["right","top"]].set_visible(False)
        ax.tick_params(labelsize=8, labelleft=i==0, length=2)
        ax.set_title(source.replace("Splatter Simple","Splatter\nSimple"), fontsize=8.8,
                     weight="bold", pad=4, linespacing=1)
        if i == 0:
            ax.set_ylabel("log(1 + variance)", fontsize=8.35, labelpad=0)
    text_at(fig, 282, 177, "log(1 + mean)", ha="center", va="center", size=8.35)
    if colorbar:
        bar = fig.colorbar(points, cax=axes_box(fig, 459, 177, 91, 4.4), orientation="horizontal", ticks=[0,.5,1])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=1.5, labelsize=7.8, pad=1.3)
        bar.ax.set_title("Zero proportion", size=8.15, pad=4)


def dlpfc(fig, zenodo):
    """Show gene moments for DLPFC section 151670."""
    cloud(fig, zenodo, "DLPFC_151670", "DLPFC 151670", False)


def merfish(fig, zenodo):
    """Show gene moments for MERFISH section 120."""
    cloud(fig, zenodo, "MERFISH_120", "MERFISH 120", True)


PLOTS = {"a": heatmaps, "b": dlpfc, "c": merfish}
