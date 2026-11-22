"""Figure 4 page geometry and print settings."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/feast-final-mpl")

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feast_plate import axes_box, set_style, text_at

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "Zenodo/results/04_batch_effect_removal"
OUTPUT = PROJECT / "reproduce_figures/Outputs/Figure_04"
PLATE = "Figure4_batch_removal.pdf"
PAGE = (583.0011, 802.6094)

# Rectangles in the journal page's top-left coordinate system, in PDF points.
PANELS = {
    "a": (0, 0, 355, 87),
    "b": (355, 0, 228.0011, 283),
    "c": (0, 87, 355, 196),
    "d": (0, 283, 583.0011, 248),
    "e": (0, 531, 583.0011, 271.6094),
}

METRIC_FRAME = (94.4, 57.4)

METHODS = {
    "GraphST": ("#0072B2", "o"),
    "STAMP": ("#D55E00", "s"),
    "scVI": ("#009E73", "D"),
    "Uncorrected": ("#777777", "X"),
}
MODES = {"shift_only": ("Shift only", "-"), "diagonal_affine": ("Diagonal affine", "--")}
BATCH_COLORS = {"ref": "#0072B2", "query": "#D55E00"}
LAYER_COLORS = {
    "Layer_1": "#E69F00",
    "Layer_2": "#56B4E9",
    "Layer_3": "#009E73",
    "Layer_4": "#F0E442",
    "Layer_5": "#0072B2",
    "Layer_6": "#D55E00",
    "WM": "#CC79A7",
}


def panel_args(letter, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT / f"{letter}.pdf")
    return parser.parse_args()


def configure_style():
    set_style()
    rcParams.update({
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.frameon": False,
        "grid.linewidth": 0.45,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
    })


def panel(letter, *, label_x=7, label_top=3):
    configure_style()
    _, _, width, height = PANELS[letter]
    figure = plt.figure(figsize=(width / 72, height / 72))
    text_at(figure, label_x, label_top, letter, weight="bold", fontsize=13, ha="left", va="top")
    return figure


def metric_axes(figure, columns, rows):
    """Give both metric panels the same inner plotting dimensions."""
    return [axes_box(figure, x, top, *METRIC_FRAME) for top in rows for x in columns]


def style_axis(axis, *, max_ticks=5):
    axis.grid(axis="y", lw=0.4, color="#dedede")
    axis.tick_params(width=0.48, length=1.7, pad=1.5, labelsize=5.6)
    for spine in ("left", "bottom"):
        axis.spines[spine].set(color="#707070", linewidth=0.45)
    axis.yaxis.set_major_locator(MaxNLocator(max_ticks))


def save_panel(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=600, metadata={"Creator": "FEAST reproduction", "CreationDate": None})
    plt.close(figure)
