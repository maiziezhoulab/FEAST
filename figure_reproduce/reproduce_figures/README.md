# Manuscript figure reproduction

The scripts in `Scripts/` reproduce Figures 2–7 and Supplementary Figures S1–S5. Figure 1 is the manuscript workflow schematic and has no analysis-derived panels. The manuscript PDFs set the page dimensions, panel order, and placement for each assembled figure.

## Render and assemble

Use the FEAST Python 3.11 environment with `pypdf` for PDF assembly and `pyarrow` for the archived Parquet inputs. Install them with `python -m pip install pypdf pyarrow` if needed. Arial is used when installed. Set `FEAST_FONT_DIR` to a directory containing the Arial font files if they are outside a standard system location.

By default, the scripts read the external study inputs and results under `figure_reproduce/Zenodo/`. Use `--help` on any panel script to see its input and output options. Each lettered script writes its panel PDF to `figure_reproduce/reproduce_figures/Outputs/<figure>/`. Run `assemble.py` in the same script directory once all required panels are available.

For example, run Figure 7 from the FEAST repository root with the unpacked study archive at `/path/to/Zenodo`.

```sh
scripts=figure_reproduce/reproduce_figures/Scripts/Figure_07
for panel in a b c d e f; do
    python "$scripts/$panel.py" --zenodo /path/to/Zenodo || exit
done
python "$scripts/assemble.py"
```

Supplementary scripts also export PNG and SVG files by default. They use Poppler's `pdftoppm` for PNG export. Pass `--pdf-only` to both the panel scripts and the assembler for PDF-only output.

When the Study 00 map tiles have already been exported, Figure 2b can replay `simulator_spatial_comparison.pdf` with the current journal labels and layout. Pass the export with `--archived-pdf` and its matching color-scale table directory with `--archived-dir`. This mode requires PyMuPDF.

Figure 3a accepts `alteration_plot_data.csv` with `--plot-data` to recover the archived signed-shift levels. Figure 3d accepts `deconvolution_spatial_plot_data.csv.gz` with the same flag to replay the saved spatial compositions. Both scripts check the exports against the corresponding study results.
