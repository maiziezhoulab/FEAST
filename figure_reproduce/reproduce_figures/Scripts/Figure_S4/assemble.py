"""Assemble Supplementary Figure S4."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layout import FIGURE, FRAMES, PAGE, PLATE
from supplement import assemble

if __name__ == "__main__":
    assemble(FIGURE, PLATE, PAGE, FRAMES)
