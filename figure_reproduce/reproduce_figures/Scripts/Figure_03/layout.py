"""Figure 3 page geometry and print settings."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/feast-final-mpl")

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feast_plate import axes_box, register_arial_fonts, text_at

PROJECT = Path(__file__).resolve().parents[3]
INPUTS = PROJECT / "Zenodo/inputs"
RESULTS = PROJECT / "Zenodo/results"
OUTPUT = PROJECT / "reproduce_figures/Outputs/Figure_03"
PLATE = "Figure3_clust_decon.pdf"
PAGE = (584.904, 722.033)

# Rectangles in the journal page's top-left coordinate system, in PDF points.
PANELS = {
    "a": (0, 0, 141, 312),
    "b": (141, 0, 443.904, 320),
    "c": (0, 298.1, 155, 423.933),
    "d": (141, 298.1, 443.904, 292.5),
    "e": (141, 566.2, 443.904, 155.833),
}
HEADING_COLOR = "#333333"

CLUSTERING = {
    "GraphST": ("GraphST", "#0072B2", "o"),
    "STAGATE_mclust": ("STAGATE", "#E69F00", "s"),
    "Leiden_unsupervised": ("Leiden", "#009E73", "^"),
}


def panel_args(letter, description, *, plot_data_help=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT / f"{letter}.pdf")
    if plot_data_help:
        parser.add_argument("--plot-data", type=Path, help=plot_data_help)
    return parser.parse_args()


def configure_style():
    register_arial_fonts()
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


def panel(letter):
    configure_style()
    left, top, width, height = PANELS[letter]
    figure = plt.figure(figsize=(width / 72, height / 72))
    text_at(figure, 8.6 if left == 0 else 0, 12.1, letter,
            weight="bold", fontsize=12.5, ha="left", va="baseline")
    return figure


def manuscript_axes(figure, letter, x, top, width, height, **kwargs):
    left, panel_top, _, _ = PANELS[letter]
    return axes_box(figure, x - left, top - panel_top, width, height, **kwargs)


def manuscript_text(figure, letter, x, top, value, **kwargs):
    left, panel_top, _, _ = PANELS[letter]
    return text_at(figure, x - left, top - panel_top, value, **kwargs)


def style_axis(axis, *, max_ticks=5):
    axis.grid(axis="y", lw=0.4, color="#dedede")
    axis.tick_params(width=0.55, length=2, pad=2)
    axis.yaxis.set_major_locator(MaxNLocator(max_ticks))


def save_panel(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The journal's tags and neighbouring labels occupy overlapping panel bounds.
    figure.savefig(path, dpi=600, transparent=True,
                   metadata={"Creator": "FEAST reproduction", "CreationDate": None})
    plt.close(figure)
