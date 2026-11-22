"""Figure 3 page geometry and print settings."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/feast-final-mpl")

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.ticker import MaxNLocator

PROJECT = Path(__file__).resolve().parents[3]
INPUTS = PROJECT / "Zenodo/inputs"
RESULTS = PROJECT / "Zenodo/results"
OUTPUT = PROJECT / "reproduce_figures/Outputs/Figure_03"
PLATE = "Figure3_clust_decon.pdf"
PAGE = (584.904, 722.033)

# Rectangles in the journal page's top-left coordinate system, in PDF points.
PANELS = {
    "a": (0, 0, 141, 308),
    "b": (141, 0, 443.904, 311),
    "c": (0, 308, 141, 414.033),
    "d": (141, 311, 443.904, 272),
    "e": (141, 583, 443.904, 139.033),
}

CLUSTERING = {
    "GraphST": ("GraphST", "#0072B2", "o"),
    "STAGATE_mclust": ("STAGATE", "#E69F00", "s"),
    "Leiden_unsupervised": ("Leiden", "#009E73", "^"),
}


def panel_args(letter, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT / f"{letter}.pdf")
    return parser.parse_args()


def configure_style():
    default = PROJECT.parent / "miniconda3/fonts"
    path = Path(os.environ.get("FEAST_FONT_DIR", default))
    for filename in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        font_manager.fontManager.addfont(path / filename)
    rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.sf": "Arial",
        "mathtext.tt": "Arial",
        "mathtext.cal": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.fallback": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.frameon": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
    })


def panel(letter, *, label_x=7):
    configure_style()
    _, _, width, height = PANELS[letter]
    figure = plt.figure(figsize=(width / 72, height / 72))
    figure.text(label_x / width, 1 - 3 / height, letter, weight="bold",
                fontsize=13, ha="left", va="top")
    return figure


def style_axis(axis, *, max_ticks=5):
    axis.grid(axis="y", lw=0.4, color="#dedede")
    axis.tick_params(width=0.55, length=2, pad=2)
    axis.yaxis.set_major_locator(MaxNLocator(max_ticks))


def save_panel(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=600, metadata={"Creator": "FEAST reproduction", "CreationDate": None})
    plt.close(figure)
