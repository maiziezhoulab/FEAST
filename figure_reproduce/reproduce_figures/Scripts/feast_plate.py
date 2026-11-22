"""Shared typography, panel CLI and PDF placement for Figures 5–7."""
import argparse
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FINAL = Path(__file__).resolve().parents[2]
INK = "#1d2930"
BLUE = "#0072b2"
DPI = 500


def set_style():
    font_dir = Path(os.environ.get("FEAST_FONT_DIR", FINAL.parent / "miniconda3/fonts"))
    for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        font_manager.fontManager.addfont(font_dir / name)
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .55, "axes.labelsize": 7, "axes.titlesize": 8, "axes.labelcolor": INK,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.unicode_minus": True,
        "legend.frameon": False, "text.color": INK, "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })


def panel_args(figure_id, letter, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--zenodo", type=Path, default=FINAL / "Zenodo")
    parser.add_argument("--output", type=Path, default=output_dir(figure_id) / f"{letter}.pdf")
    return parser


def output_dir(figure_id):
    return FINAL / "reproduce_figures/Outputs" / figure_id


def panel(letter, frames):
    _, _, width, height = frames[letter]
    set_style()
    fig = plt.figure(figsize=(width / 72, height / 72))
    fig.text(4 / width, 1 - 3 / height, letter, ha="left", va="top", fontsize=12, weight="bold", color="black")
    return fig


def save(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def axes_box(fig, x, top, width, height, **kwargs):
    w, h = fig.get_size_inches() * 72
    return fig.add_axes((x / w, 1 - (top + height) / h, width / w, height / h), **kwargs)


def text_at(fig, x, top, text, **kwargs):
    w, h = fig.get_size_inches() * 72
    return fig.text(x / w, 1 - top / h, text, **kwargs)


def limits(points, margin=.03):
    bounds = []
    for dim in "xyz":
        low, high = float(points[dim].min()), float(points[dim].max())
        pad = max(margin * (high - low), .01)
        bounds.append((low - pad, high + pad))
    return bounds


def style_3d(axis, bounds, elev, azim, axes=False):
    axis.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
    axis.set_box_aspect(np.ptp(bounds, axis=1))
    axis.view_init(elev=elev, azim=azim)
    axis.grid(False)
    if axes:
        axis.set(xticks=[], yticks=[], zticks=[])
        for a in (axis.xaxis, axis.yaxis, axis.zaxis):
            a.pane.fill = True
            a.pane.set_facecolor("#f2f3f5")
            a.pane.set_edgecolor("#b6bfc5")
    else:
        axis.set_axis_off()


def assemble_main(figure_id, plate, page, frames):
    parser = argparse.ArgumentParser(description=f"Assemble {plate} from the individual panel PDFs.")
    parser.add_argument("--panels", "--panel-dir", type=Path, default=output_dir(figure_id))
    parser.add_argument("--output", type=Path, default=output_dir(figure_id) / plate)
    args = parser.parse_args()
    assemble_pdf(args.panels, args.output, page, frames)


def assemble_pdf(panel_dir, output, page, frames):
    width, height = page
    files = {letter: (panel_dir / f"{letter}.pdf").resolve(strict=True) for letter in frames}
    commands = [f"<< /PageSize [{width} {height}] >> setpagedevice"]
    for letter, (x, top, _, panel_height) in frames.items():
        filename = os.fsencode(files[letter]).hex()
        commands += [
            f"gsave {x:.5f} {height - top - panel_height:.5f} translate",
            f"<{filename}> (r) file runpdfbegin PDFFile 0 .PDFDrawPage runpdfend",
            "grestore",
        ]
    commands += [f"[ /MediaBox [0 0 {width} {height}] /PAGE pdfmark", "showpage"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        pending = Path(scratch) / output.name
        permitted = os.pathsep.join(map(str, files.values()))
        subprocess.run([
            "gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dNEWPDF",
            "-dPDFSTOPONERROR", "-sDEVICE=pdfwrite", "-dAutoRotatePages=/None",
            f"--permit-file-read={permitted}", f"-sOutputFile={pending}",
            "-c", "\n".join(commands),
        ], check=True)
        pending.replace(output)
