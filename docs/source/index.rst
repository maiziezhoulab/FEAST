FEAST | Spatial Transcriptomics Simulator
=========================================

**FEAST** (Feature-space-based modeling of Spatial Transcriptomics) generates
high-fidelity synthetic spatial transcriptomics data by modeling gene expression
through a parameter cloud capturing mean, variance, and sparsity.

Key capabilities:

- **Single-slice simulation** — reproduce realistic ST data from a reference
- **Controllable alterations** — systematically modify expression statistics for benchmarking
- **Alignment benchmarks** — paired datasets with controlled geometric transformations
- **Deconvolution ground truth** — multi-resolution data with known cell-type compositions
- **De novo generation** — build virtual slices from blueprints, motifs, and parameter clouds

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   tutorials
   api

Repository scope
----------------

The FEAST repository contains the installable Python tool, documentation,
tutorial notebooks, and article reproduction scripts under ``figure_reproduce/``.

Run ``bash figure_reproduce/reproduce_figures/render_pdfs.sh`` from the checkout
to render the manuscript PDFs with the included plotting inputs. Repeating the
underlying analyses uses the full study archive. See
:doc:`tutorials` for the inputs used by the six package notebooks.
