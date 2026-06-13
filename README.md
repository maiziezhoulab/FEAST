# FEAST | Parameter-cloud modeling of spatial transcriptomics for simulation and de novo virtual slices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**FEAST** (Feature-space-based modeling of Spatial Transcriptomics) is a computational framework for simulating spatial transcriptomics (ST) data. By modeling gene expression through a parameter cloud capturing mean, variance, and sparsity, FEAST generates high-fidelity synthetic ST slices with controllable biological and technical variations.

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

##  Tutorials

Try FEAST with notebook! Comprehensive Jupyter notebooks are provided in the repository:

- **[example_single_sim.ipynb](example_single_sim.ipynb)**: Basic single-slice simulation for both sequencing-based and imaging-based ST data





## Architecture

```
FEAST/
├── FEAST_core/          # Core simulation engine
│   ├── simulator.py     # Main deterministic simulation logic
│   ├── count_decoding.py # Shared rank and quantile count decoding
│   ├── parameter_cloud.py  # Parameter cloud modeling
│   └── APIs.py          # Unified FEAST API
├── alignment/           # Alignment simulation
│   ├── alignment_simulator.py
│   └── spatial_align_alter.py  # Rotation & warping transformations
├── deconvolution/       # Deconvolution simulation
│   ├── deconvolution_simulator.py
│   └── generate_deconvolution.py
├── de_novo/             # Blueprint and conditional virtual-slice generation
│   ├── builder.py
│   ├── conditional.py
│   ├── quantile_field.py
│   └── pattern.py
└── modeling/            # Statistical models
    ├── StudentT_mixture_model.py
    ├── Beta_mixture_model.py
    └── marginal_alteration.py
```

## Reproduction Scripts

The `reproduction/` folder contains scripts to reproduce all benchmarking results from the paper. Each subdirectory corresponds to a specific analysis:

```
reproduction/
├── 1_Simulator_benchmark/     # Figure 2: Simulation fidelity evaluation
├── 2_Clustering_simulation/   # Figure 3: Clustering robustness testing
├── 3_Alignment_simulation/    # Figure 4: Alignment algorithm benchmarking
└── 4_Deconvolution_simulation/# Supp Fig: Deconvolution ground truth generation
```

Interpolation APIs and external reconstruction wrappers are intentionally excluded from this version.

### Dataset Organization

All scripts expect datasets in a `data/` directory with the following naming convention:

```
data/
├── DLPFC_{sample_id}.h5ad          # Human DLPFC sections
├── MERFISH_{slice_id}.h5ad         # Mouse brain MERFISH slices
├── OpenST_{slice_id}.h5ad          # Lymph node OpenST slices
├── Stereoseq_{sample_id}.h5ad      # Mouse embryo Stereo-seq slices
├── Slideseq_{sample_id}.h5ad       # Slide-seqV2 slices
└── Xenium_{sample_id}.h5ad         # Xenium tissue slices
```

### Required Datasets

| Dataset | Technology | Source | Usage | Files |
|---------|-----------|---------|--------|-------|
| **DLPFC** | 10X Visium | [spatialLIBD](http://research.libd.org/spatialLIBD/) | Simulation, Clustering, Alignment | `DLPFC_151670.h5ad`<br>`DLPFC_151676.h5ad`<br>`DLPFC_151675.h5ad` |
| **MERFISH** | MERFISH | [Allen Brain Atlas](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html) | Simulation, Deconvolution | `MERFISH_006.h5ad`<br>`MERFISH_007.h5ad` |
| **OpenST** | OpenST | [GEO: GSE251926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926) | Simulation | `OpenST_005.h5ad`<br>`OpenST_006.h5ad` |
| **Stereo-seq** | Stereo-seq | [MOSTA](https://www.sciencedirect.com/science/article/pii/S0092867422003993) | Simulation | `Stereoseq_E14_5_E2S2.h5ad` |
| **Slide-seq** | Slide-seqV2 | [SODB](https://gene.ai.tencent.com/SpatialOmics/dataset?datasetID=119) | Simulation | `Slideseq_001.h5ad` |
| **Xenium** | Xenium | [10X Genomics](https://www.10xgenomics.com/datasets/human-lymph-node-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard) | Simulation | `Xenium_LymphNode.h5ad` |


**Note**: FEAST is actively maintained. If you have any question, please let me know!
