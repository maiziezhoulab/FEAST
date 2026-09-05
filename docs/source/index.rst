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

The FEAST repository contains the installable Python tool, a compact test
suite, documentation source, and a tutorial notebook. Article reproduction
code is maintained separately in ``FEAST_reproduce``.

Local ``validation/`` runs, ``_archive/`` material, datasets, and generated
outputs are excluded through ``.gitignore``. They are not included in a fresh
Git clone. The tracked ``tests/`` directory contains the tool's automated
checks; it is separate from local research validation.
