"""Display verified sensitivity scores and row-normalized cortical-layer transfer."""
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from feast_plate import BLUE, axes_box, text_at

DATA = "results/04_batch_effect_removal/figure_s3"
METRICS = (
    ("batch_mixing_entropy_k30", "Batch mixing entropy"),
    ("domain_purity_k30", "Domain purity"),
    ("paired_retrieval_top1", "Paired retrieval top-1"),
)
MODES = (("shift_only","Shift only"), ("diagonal_affine","Diagonal affine"))
METHODS = ("Uncorrected","GraphST","STAMP","scVI")
LAYERS = [f"Layer_{i}" for i in range(1,7)] + ["WM"]
SHORT = [f"L{i}" for i in range(1,7)] + ["WM"]
SHADE = "#f2eee7"


def sensitivity(fig, zenodo):
    """Show GraphST PCA-10 minus PCA-20 across modes and supports."""
    table = pd.read_csv(zenodo / DATA / "sensitivity.csv")
    text_at(fig, 267, 4, "GraphST representation sensitivity", ha="center", va="top", size=11, weight="bold")
    for col, (metric, title) in enumerate(METRICS):
        ax = axes_box(fig, 123+132.8*col, 42, 111, 208)
        labels, positions = [], []
        group = table.loc[table.metric.eq(metric)]
        cursor = 0
        for mode, label in MODES:
            values = group.loc[group["mode"].eq(mode)].sort_values(["alpha","excluded_pair_count"])
            center = cursor + (len(values)-1)/2
            for row in values.itertuples(index=False):
                position = cursor
                if row.alpha > 1:
                    ax.axhspan(position-.48, position+.48, facecolor=SHADE, edgecolor="none", zorder=0)
                ax.plot([0,row.difference], [position,position], color="#9babb5", lw=.68, zorder=1)
                if row.excluded_pair_count:
                    ax.scatter(row.difference, position, marker="D", s=15, facecolor="white",
                               edgecolor=BLUE, lw=.8, zorder=3)
                else:
                    ax.scatter(row.difference, position, s=17, c=BLUE, linewidths=0, zorder=3)
                legend = f"{row.alpha:.2f}"
                if row.excluded_pair_count:
                    legend += f"  (−{row.excluded_pair_count})"
                labels.append(legend)
                positions.append(position)
                cursor += 1
            if col == 0:
                ax.text(-.78, center, label, transform=ax.get_yaxis_transform(), rotation=90,
                        ha="center",va="center",fontsize=9)
            cursor += .78
        low,high = min(0,group.difference.min()), max(0,group.difference.max())
        margin = .10*(high-low)
        ax.set(xlim=(low-margin,high+margin), ylim=(max(positions)+.65,-.65), yticks=positions)
        ax.set_yticklabels(labels if col==0 else [""]*len(labels), fontsize=8)
        ax.axvline(0,c="#808080",lw=.65,zorder=1)
        ax.spines[["left","right","top"]].set_visible(False)
        ax.tick_params(axis="y",length=0,pad=3)
        ax.tick_params(axis="x",labelsize=8.2)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(title, size=9.3, weight="bold", pad=5)
    text_at(fig,105,36,"α", ha="center", va="center", size=9)
    text_at(fig,321,273,"Difference: PCA-10 − PCA-20", ha="center", va="center", size=9)
    handles = [Line2D([],[],marker="o",color=BLUE,lw=0,ms=4,label="All spots"),
               Line2D([],[],marker="D",markerfacecolor="white",markeredgecolor=BLUE,lw=0,ms=4,
                      label="Excluded pairs (count)"),
               Patch(facecolor=SHADE,edgecolor="none",label="Extrapolation: α > 1")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54,.025), ncols=3,
               fontsize=8.35, handletextpad=.55, columnspacing=1.2)


def heatmaps(fig, zenodo, mode):
    table = pd.read_csv(zenodo / DATA / "confusion.csv")
    scores = pd.read_csv(zenodo / DATA / "macro_f1.csv")
    text_at(fig,267,7,f"{dict(MODES)[mode]} · α = 1", ha="center",va="top",size=11,weight="bold")
    for index,method in enumerate(METHODS):
        x = 60 + 110.5*index
        ax = axes_box(fig,x,66,96.5,96.5)
        part = table.loc[table["mode"].eq(mode) & table.method.eq(method)]
        matrix = part.pivot(index="true_layer",columns="predicted_layer",values="row_fraction").loc[LAYERS,LAYERS]
        sizes = part.groupby("true_layer").n_true.first().loc[LAYERS]
        image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(7), SHORT, fontsize=8)
        ax.set_yticks(range(7),[f"{label} ({n})" for label,n in zip(SHORT,sizes)] if index==0 else [""]*7,fontsize=8)
        ax.tick_params(length=0, pad=2.4)
        ax.spines[:].set_visible(False)
        score = scores.loc[scores.method.eq(method) & scores["mode"].eq(mode), "macro_f1"].item()
        ax.set_title(f"{method}\nMacro-F1 = {score:.3f}", size=9.2, linespacing=1.15, pad=5)
    bar = fig.colorbar(image,cax=axes_box(fig,507,66,6,96.5),ticks=[0,.5,1])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=1.6,labelsize=7.8,pad=1.4)
    text_at(fig,8,114,"True layer (n)",rotation=90,ha="center",va="center",size=8.6)
    text_at(fig,281,192,"Predicted layer",ha="center",va="center",size=8.8)
    text_at(fig,494,192,"Row fraction",ha="center",va="center",size=8.3)


def shift_only(fig, zenodo):
    """Show the four cortical-layer confusion matrices for the shift-only endpoint."""
    heatmaps(fig,zenodo,"shift_only")


def diagonal_affine(fig, zenodo):
    """Show the four cortical-layer confusion matrices for the diagonal-affine endpoint."""
    heatmaps(fig,zenodo,"diagonal_affine")


PLOTS = {"a":sensitivity,"b":shift_only,"c":diagonal_affine}
