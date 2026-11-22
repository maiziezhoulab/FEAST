"""Place the two Figure 2 panel PDFs on the journal canvas."""

import argparse
from pathlib import Path

from layout import FIGURES, HEIGHT, PANELS, WIDTH
from feast_plate import assemble_pdf


def assemble(panel_dir, output_path):
    assemble_pdf(panel_dir, output_path, (WIDTH, HEIGHT), PANELS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", "--panels", type=Path, default=FIGURES)
    parser.add_argument("--output", type=Path, default=FIGURES / "Figure2_simulator_bench.pdf")
    args = parser.parse_args()
    assemble(args.panel_dir, args.output)


if __name__ == "__main__":
    main()
