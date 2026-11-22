# FEAST | Parameter-cloud modeling of spatial transcriptomics for simulation and de novo virtual slices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**FEAST** simulates spatial transcriptomics counts by modeling each gene's mean, variance and sparsity. Use a measured slice to generate new counts on the same tissue layout, or design a virtual slice from a spatial blueprint. The package also supports alignment, deconvolution and batch-effect benchmarks.

## Documentation

[FEAST documentation](https://maiziezhoulab.github.io/FEAST/) · [Tutorials](https://maiziezhoulab.github.io/FEAST/tutorials.html) · [API reference](https://maiziezhoulab.github.io/FEAST/api.html)

## Installation

FEAST-py 1.1.0 requires Python 3.11. In a Python 3.11 environment, install the published package and check which copy Python imports.

```bash
python -m pip install "FEAST-py==1.1.0"
python -c "from importlib.metadata import version; import FEAST; print(version('FEAST-py'), FEAST.__file__)"
```

For an editable Conda installation, follow the [source installation guide](https://maiziezhoulab.github.io/FEAST/installation.html).

## Simulate a slice

```python
import scanpy as sc
from FEAST import Alteration, simulate

# Supply raw counts in X and XY coordinates in obsm["spatial"].
adata = sc.read_h5ad("your_spatial_data.h5ad")
simulated_adata = simulate(adata, seed=7)

# Request a twofold change in the target mean expression.
altered_adata = simulate(adata, alteration=Alteration.mean_only(fold_change=2.0), seed=7)
```

The returned AnnData contains simulated counts in `.X` and retains the reference spot coordinates.

## Tutorials

The six notebooks cover single-slice simulation and GraphST clustering, alignment, deconvolution, scVI batch correction, 2D conditional transfer and 3D atlas transfer. The [tutorial guide](https://github.com/maiziezhoulab/FEAST/blob/v1.1.0/tutorial/README.md) lists the input files and environment for each example.

## Article Reproduction

The article's analysis and plotting scripts are in [figure_reproduce](https://github.com/maiziezhoulab/FEAST/tree/v1.1.0/figure_reproduce).
The `reproduce_code/` subdirectory contains the study and preprocessing scripts, and
`reproduce_figures/Scripts/` contains the plotting scripts. To replay an article
workflow, use the corresponding study configuration, inputs and results in the
external reproduction workspace.
