"""Place the two rendered Figure 2 PDFs on the original journal canvas."""

import argparse
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from layout import B_HEIGHT, FIGURES, HEIGHT, WIDTH


def assemble(panel_dir, output_path):
    panel_a = (panel_dir / "a.pdf").resolve(strict=True)
    panel_b = (panel_dir / "b.pdf").resolve(strict=True)

    def draw(path):
        # A hexadecimal PostScript string keeps filenames literal.
        return f"<{os.fsencode(path).hex()}> (r) file runpdfbegin PDFFile 0 .PDFDrawPage runpdfend"

    postscript = f"""
        << /PageSize [{WIDTH} {HEIGHT}] >> setpagedevice
        {draw(panel_b)}
        gsave 0 {B_HEIGHT:.4f} translate
        {draw(panel_a)}
        grestore
        [ /MediaBox [0 0 {WIDTH} {HEIGHT}] /PAGE pdfmark
        showpage
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as scratch:
        pending = Path(scratch) / output_path.name
        subprocess.run([
            "gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dNEWPDF",
            "-dPDFSTOPONERROR", "-sDEVICE=pdfwrite", "-dAutoRotatePages=/None",
            f"--permit-file-read={panel_a}{os.pathsep}{panel_b}", f"-sOutputFile={pending}",
            "-c", postscript,
        ], check=True)
        pending.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=FIGURES)
    parser.add_argument("--output", type=Path, default=FIGURES / "Figure2_simulator_bench.pdf")
    args = parser.parse_args()
    assemble(args.panel_dir, args.output)


if __name__ == "__main__":
    main()
