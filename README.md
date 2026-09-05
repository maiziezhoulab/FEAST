# FEAST | Parameter-cloud modeling of spatial transcriptomics for simulation and de novo virtual slices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**FEAST** (FEAture-space based modeling for Spatial Transcriptomics) is a computational framework for simulating spatial transcriptomics (ST) data. By modeling gene expression through a parameter cloud capturing mean, variance, and sparsity, FEAST generates high-fidelity synthetic ST slices with controllable biological and technical variations.

## Key Features

- **High-Fidelity Simulation**: Generate realistic ST data that preserves gene-level statistics, spatial patterns, and biological heterogeneity
- **Controllable Alterations**: Systematically modify gene expression (mean, variance, sparsity) for robust benchmarking
- **Multiple ST Technologies**: Support for Visium, MERFISH, Stereo-seq, Slide-seq, Xenium, and OpenST
- **Alignment Benchmarks**: Create paired datasets with controlled geometric transformations (rotation, warping) for testing alignment algorithms
- **Deconvolution Ground Truth**: Generate multi-resolution data with known cell-type compositions
- **De Novo Virtual Slices**: Generate slices from blueprints, spatial motifs, parameter clouds, and conditional references


## Installation

### Conda Environment (Recommended)
```bash
git clone https://github.com/maiziezhoulab/FEAST
cd FEAST
conda env create -f environment.yml
conda activate feast-py311-conda
pip install --no-deps -r requirements.txt
pip install --no-deps -e .
```

### Existing Source Checkout
```bash
cd FEAST
conda env create -f environment.yml
conda activate feast-py311-conda
pip install --no-deps -r requirements.txt
pip install --no-deps -e .
```

### Dependencies
- Python 3.11
- scanpy
- anndata
- numpy
- scipy
- pandas
- scikit-learn
- pyvinecopulib
- POT (Python Optimal Transport)
- tps (Thin Plate Spline)

## Tutorial notebooks

Start with the [six real-data tutorials](tutorial/README.md): single-slice simulation with GraphST, alignment, deconvolution, same-slice batch correction, 2D conditional transfer, and 3D atlas transfer. Each notebook explains the main FEAST calls; available result previews are clearly labeled with their reproduction sources. Notebooks 00, 02 and 04 share the same 151675 input. See [data and environment setup](tutorial/README.md#setup) for required inputs and pending download links.

## Quick Start

### Single Slice Simulation

```python
from FEAST import simulator
import scanpy as sc

# Load your reference data
adata = sc.read_h5ad("your_spatial_data.h5ad")

# Simple simulation with default parameters
simulated_adata = simulator.simulate_single_slice(
    adata=adata,
    verbose=True
)

# Simulation with expression alteration
from FEAST.modeling.marginal_alteration import AlterationConfig

alteration_config = AlterationConfig.mean_only(fold_change=2.0)
altered_adata = simulator.simulate_single_slice(
    adata=adata,
    alteration_config=alteration_config
)
```

### Alignment Simulation

```python
from FEAST import alignment

# Generate paired datasets with rotation for alignment benchmarking
original, rotated = alignment.simulate_alignment_rotation(
    adata=adata,
    rotation_angle=30.0,  # degrees
    data_type='imaging'  # or 'sequencing'
)
```

### Deconvolution Simulation

```python
from FEAST import deconvolution

# Generate multi-resolution data with known cell-type compositions
deconv_adata = deconvolution.create_deconvolution_benchmark_data(
    adata=single_cell_adata,
    downsampling_factor=0.25,
    grid_type='hexagonal',
    cell_type_key='cell_type'
)
```

### De Novo Virtual Slice Generation

```python
from FEAST import de_novo

genes = ["GeneA", "GeneB", "GeneC"]
blueprint = (
    de_novo.SimulationBlueprintBuilder.rectangular_grid(4, 4)
    .set_domains(["cortex"] * 8 + ["medulla"] * 8)
    .build()
)
parameter_cloud = (
    de_novo.SimulationParameterBuilder.from_gene_names(genes)
    .set_all(mean=3.0, variance=5.0, zero_prop=0.2)
    .build()
)
patterns = (
    de_novo.SimulationPatternBuilder.from_gene_names(genes)
    .gradient("GeneA", axis="x")
    .hotspot("GeneB", center=[0.5, 0.5], radius=0.25)
    .build()
)

virtual_slice = de_novo.simulate_from_design(
    blueprint,
    parameter_cloud,
    pattern_spec=patterns,
    random_seed=7,
)

# Final rank-normalized quantiles are available when storage is enabled.
quantiles = virtual_slice.layers["feast_quantiles"]
```

De novo generation builds a latent rank-score field from shared spatial motifs,
rank-normalizes that field into `feast_quantiles`, and decodes counts with the
target parameter cloud. Reference-conditioned virtual slices use the same
latent H-to-Q path after transporting reference rank evidence.

## Agent Skill for FEAST Users

The repository includes a [FEAST agent skill](.agents/skills/feast/SKILL.md)
to help an AI assistant guide you through preparing inputs, choosing a simulation
workflow, running FEAST, troubleshooting, and interpreting results.

Ask your assistant to read `.agents/skills/feast/SKILL.md` from this repository
and describe your data and goal. For example:

> Read `.agents/skills/feast/SKILL.md` and help me simulate a spatial
> transcriptomics dataset from my reference `.h5ad`, preserving its spatial
> layout and reducing mean expression by 20%.

You can also explore the guides directly:

- [Installation skill](.agents/skills/feast-install/SKILL.md): agent instructions
  for choosing an environment, installing FEAST, resolving setup problems, and
  checking the installation with a small simulation. Ask your assistant to read
  this skill when setting up FEAST.
- [Worked examples](.agents/skills/feast/references/examples.md): six examples
  using synthetic data, covering reference simulation, expression alterations,
  designed and conditional slices, alignment, and deconvolution.
- [Further applications](.agents/skills/feast/references/applications.md): ways
  to use existing FEAST APIs for robustness studies, spatial pattern detection,
  batch-effect experiments, and virtual slice generation.

## Article Reproduction

This repository contains the FEAST computational tool. Code for reproducing
the article's analyses belongs in the separate `FEAST_reproduce` repository.
Historical local `reproduction/` and `benchmark_scripts/` folders are archived
under `_archive/2026-09-04/` and are not included in a Git clone or the package.

Interpolation APIs and external reconstruction wrappers are intentionally
excluded from this version.

**Note**: FEAST is actively maintained. If you have any question, please let me know!
