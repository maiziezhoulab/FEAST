"""Assemble the five panel PDFs on the journal canvas without resampling."""

import argparse
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from layout import OUTPUT, PAGE, PANELS, PLATE


def assemble(panel_dir, output):
    width, height = PAGE
    files = {letter: (panel_dir / f"{letter}.pdf").resolve(strict=True) for letter in PANELS}
    commands = [f"<< /PageSize [{width} {height}] >> setpagedevice"]
    for letter, (x, top, _, panel_height) in PANELS.items():
        filename = os.fsencode(files[letter]).hex()
        commands += [
            f"gsave {x:.4f} {height - top - panel_height:.4f} translate",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT / PLATE)
    args = parser.parse_args()
    assemble(args.panels, args.output)


if __name__ == "__main__":
    main()
