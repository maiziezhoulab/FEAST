"""Paths and journal page geometry shared by the Figure 2 entry points."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
STUDY = "00_simulator_benchmark"
INPUTS = PROJECT / "Zenodo/inputs" / STUDY
RESULTS = PROJECT / "Zenodo/results" / STUDY
FIGURES = PROJECT / "reproduce_figures/Outputs/Figure_02"

# Original journal PDF MediaBox, in PDF points.
WIDTH = 583.3209
HEIGHT = 818.2211
A_HEIGHT = 227
B_HEIGHT = HEIGHT - A_HEIGHT


def rect(x, top, width, height, panel_height):
    """Convert a top-left rectangle in PDF points to Matplotlib figure coordinates."""
    return [x / WIDTH, 1 - (top + height) / panel_height, width / WIDTH, height / panel_height]


def configure_style():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    lab_fonts = PROJECT.parent / "miniconda3/fonts"
    if lab_fonts.is_dir():
        os.environ.setdefault("FEAST_FONT_DIR", str(lab_fonts))
    from plot import configure_matplotlib
    from matplotlib import rcParams

    configure_matplotlib()
    rcParams["mathtext.cal"] = "Arial"
