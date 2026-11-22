"""Export supplementary panels and assemble them at the journal page size."""
import argparse
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt

from feast_plate import assemble_pdf, output_dir, panel, panel_args

DPI = 600
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"


def render_panel(figure_id, frames, plots, letter):
    draw = plots[letter]
    parser = panel_args(figure_id, letter, draw.__doc__)
    parser.add_argument("--pdf-only", action="store_true", help="Write just the component PDF")
    args = parser.parse_args()
    fig = panel(letter, frames)
    mpl.rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.fallback": None,
    })
    draw(fig, args.zenodo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, metadata={"CreationDate": None, "ModDate": None})
    if not args.pdf_only:
        fig.savefig(args.output.with_suffix(".svg"), dpi=DPI, metadata={"Date": None})
        render_png(args.output)
    plt.close(fig)


def render_png(pdf):
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", str(pdf),
                    str(pdf.with_suffix(""))], check=True)


def assemble_svg(panel_dir, output, page, frames):
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", XLINK)
    width, height = page
    root = ET.Element(f"{{{SVG}}}svg", width=f"{width}pt", height=f"{height}pt", viewBox=f"0 0 {width} {height}")
    for letter, (x, y, w, h) in frames.items():
        child = ET.parse(panel_dir / f"{letter}.svg").getroot()
        for element in child.iter():
            for key, value in element.attrib.items():
                if key == "id":
                    element.set(key, f"{letter}-{value}")
                elif key == f"{{{XLINK}}}href" and value.startswith("#"):
                    element.set(key, f"#{letter}-{value[1:]}")
                elif "url(#" in value:
                    element.set(key, re.sub(r"url\(#([^)]*)\)", rf"url(#{letter}-\1)", value))
        child.attrib.update(x=str(x), y=str(y), width=str(w), height=str(h))
        root.append(child)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def assemble(figure_id, plate, page, frames):
    parser = argparse.ArgumentParser(description=f"Assemble {plate} from the individual supplement panels.")
    parser.add_argument("--panels", "--panel-dir", type=Path, default=output_dir(figure_id))
    parser.add_argument("--output", type=Path, default=output_dir(figure_id) / plate)
    parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()
    assemble_pdf(args.panels, args.output, page, frames)
    if not args.pdf_only:
        assemble_svg(args.panels, args.output.with_suffix(".svg"), page, frames)
        render_png(args.output)
