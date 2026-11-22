"""Shared typography, panel CLI and vector PDF placement."""
import argparse
import os
from pathlib import Path
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


def register_arial_fonts():
    """Use the configured font bundle or the Arial installed on this system."""
    font_names = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")
    mac_names = ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf")
    custom_dir = os.environ.get("FEAST_FONT_DIR")
    locations = []
    if custom_dir:
        locations.append((Path(custom_dir), font_names))
        locations.append((Path(custom_dir), mac_names))
    locations.extend((
        (FINAL.parent / "miniconda3/fonts", font_names),
        (Path("/System/Library/Fonts/Supplemental"), mac_names),
    ))
    for directory, names in locations:
        paths = [directory / name for name in names]
        if all(path.is_file() for path in paths):
            for path in paths:
                font_manager.fontManager.addfont(path)
            return


def set_style():
    register_arial_fonts()
    try:
        font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
        font_family = "Arial"
    except ValueError:
        font_family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": font_family, "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
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
    from pypdf import PdfReader, PdfWriter

    width, height = page
    files = {letter: (panel_dir / f"{letter}.pdf").resolve(strict=True) for letter in frames}
    writer = PdfWriter()
    plate = writer.add_blank_page(width=width, height=height)
    for letter, (x, top, _, panel_height) in frames.items():
        reader = PdfReader(files[letter])
        if len(reader.pages) != 1:
            raise ValueError(f"Expected a single panel page in {files[letter]}")
        writer.pdf_header = max(writer.pdf_header, reader.pdf_header)
        source = reader.pages[0]
        if source.rotation:
            source.transfer_rotation_to_content()
        plate.merge_translated_page(
            source,
            tx=x - float(source.cropbox.left),
            ty=height - top - panel_height - float(source.cropbox.bottom),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        pending = Path(scratch) / output.name
        writer.write(pending)
        pending.replace(output)
