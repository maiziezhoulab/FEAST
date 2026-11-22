"""Assemble the five Figure 3 panel PDFs without resampling."""

import argparse
from pathlib import Path

from layout import OUTPUT, PAGE, PANELS, PLATE
from feast_plate import assemble_pdf


def assemble(panel_dir, output):
    assemble_pdf(panel_dir, output, PAGE, PANELS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", "--panel-dir", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT / PLATE)
    args = parser.parse_args()
    assemble(args.panels, args.output)


if __name__ == "__main__":
    main()
