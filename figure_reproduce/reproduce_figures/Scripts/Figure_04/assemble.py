"""Assemble the five panel PDFs on the journal canvas without resampling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feast_plate import assemble_main

from layout import PAGE, PANELS, PLATE


if __name__ == "__main__":
    assemble_main("Figure_04", PLATE, PAGE, PANELS)
