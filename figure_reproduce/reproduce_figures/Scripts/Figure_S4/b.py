"""Render Supplementary Figure S4b."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layout import FIGURE, FRAMES
from panels import PLOTS
from supplement import render_panel

if __name__ == "__main__":
    render_panel(FIGURE, FRAMES, PLOTS, "b")
