# Reproduce the manuscript figures

From the FEAST repository root, use a Python 3.11 environment and run:

```bash
python -m pip install -r figure_reproduce/reproduce_figures/requirements.txt
bash figure_reproduce/reproduce_figures/render_pdfs.sh
```

The command renders **46 panel PDFs** and assembles Figures **2–7** and **S1–S5** from the included [plotting inputs](plot_data/README.md). It also copies the supplied Figure 1 workflow schematic. The PDFs appear in `figure_reproduce/reproduce_figures/Outputs/Figure_*/`. Use Arial for manuscript typography. Set `FEAST_FONT_DIR` to your Arial font directory if needed.

To rebuild individual figures, pass their identifiers:

```bash
bash figure_reproduce/reproduce_figures/render_pdfs.sh 03 05 06
```

## What each render uses

| Figure | Included visualization inputs |
| --- | --- |
| 1 | The manuscript workflow schematic in PDF format. |
| 2 | Scores for five simulators across 12 sections. Panel b reconstructs the journal layout from 42 verified map tiles in the Study 00 spatial-comparison export and the matching color-scale table. |
| 3 | Clustering scores and spot labels, deconvolution scores and spatial compositions, and alignment errors. Panel b recalculates 15 adjusted Rand indices from one 56 KiB [spot table](plot_data/Figure_03/b_partitions.csv.gz). |
| 4 | Study design, method and seed summaries, embeddings, gene-parameter values and layer predictions. |
| 5 | Transfer map coordinates, counts and retention masks, paired scores, slice-083 expression and geometry, section coverage and volume marker tables. |
| 6 | Atlas geometry artwork, measured-section coordinates and regions, marker and plane projections, and regional communication tables. |
| 7 | Region-pair metrics, native route images and their framing, directional opportunities and measured expression summaries. |
| S1–S5 | Gene statistics, perturbation and alignment summaries, integration scores, transfer diagnostics, matched-section markers and spatial cluster projections. |

## Using the full study archive

The rendering command uses only the Git checkout and software listed above. Re-running simulation, fitting or scoring from raw data is a separate workflow in [reproduce_code](../reproduce_code/). The individual [panel scripts](Scripts/) retain their original options for full study inputs. For example, `Figure_03/b.py` can regenerate the compact clustering table from the section and 15 method H5ADs. Use `--help` on a panel script to see its options.

To export supplementary PNG and SVG files as well, invoke the corresponding panel scripts with `--zenodo figure_reproduce/reproduce_figures/plot_data` from the repository root, then run their assembler without `--pdf-only`. PNG export requires Poppler’s `pdftoppm`.
