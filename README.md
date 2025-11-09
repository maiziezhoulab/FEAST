# FEAST-sim: Feature-Space Spatial Transcriptomics Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**FEAST-sim** (FEAture-space of Spatial Transcriptomics derived SIMulator) is a comprehensive computational framework for simulating and interpolating spatial transcriptomics (ST) data. By modeling gene expression through a "parameter cloud" - a latent manifold capturing mean, variance, and sparsity - FEAST-sim generates high-fidelity synthetic ST slices with controllable biological and technical variations.

## Key Features

- **High-Fidelity Simulation**: Generate realistic ST data that preserves gene-level statistics, spatial patterns, and biological heterogeneity
- **Controllable Alterations**: Systematically modify gene expression (mean, variance, sparsity) for robust benchmarking
- **Multiple ST Technologies**: Support for Visium, MERFISH, Stereo-seq, Slide-seq, Xenium, and OpenST
- **Alignment Benchmarks**: Create paired datasets with controlled geometric transformations (rotation, warping) for testing alignment algorithms
- **Deconvolution Ground Truth**: Generate multi-resolution data with known cell-type compositions
- **3D Interpolation**: Reconstruct missing tissue slices using optimal transport in parameter space


## Installation

### From PyPI (Recommended)
```bash
pip install FEAST-sim
```

### Directly From Source
```bash
git clone https://github.com/maiziezhoulab/FEAST-sim.git
cd FEAST-sim
pip install -e .
```

### Dependencies
- Python >= 3.8
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
    sigma=1.0,  # Spatial smoothness parameter
    verbose=True
)

# Simulation with expression alteration
from FEAST.modeling.marginal_alteration import AlterationConfig

alteration_config = AlterationConfig.mean_only(fold_change=2.0)
altered_adata = simulator.simulate_single_slice(
    adata=adata,
    alteration_config=alteration_config,
    sigma=1.0
)
```

### Alignment Simulation

```python
from FEAST.alignment.alignment_simulator import simulate_alignment_rotation

# Generate paired datasets with rotation for alignment benchmarking
original, rotated = simulate_alignment_rotation(
    adata=adata,
    rotation_angle=30.0,  # degrees
    data_type='imaging',  # or 'sequencing'
    sigma=0  # Perfect pattern preservation
)
```

### Deconvolution Simulation

```python
from FEAST.deconvolution.generate_deconvolution import create_deconvolution_benchmark_data

# Generate multi-resolution data with known cell-type compositions
deconv_adata = create_deconvolution_benchmark_data(
    adata=single_cell_adata,
    downsampling_factor=0.25,
    grid_type='hexagonal',
    cell_type_key='cell_type'
)
```

### 3D Slice Interpolation

```python
from FEAST.interpolation.interpolation_pipeline import interpolate_slices

# Interpolate missing slices between consecutive sections
interpolated_slices = interpolate_slices(
    adata_list=[slice_k, slice_k_plus_1],
    n_interpolate=3,  # Number of intermediate slices
    alpha=0.01,
    verbose=True
)
```

##  Tutorials

Try FEAST-sim with notebook! Comprehensive Jupyter notebooks are provided in the repository:

- **[example_single_sim.ipynb](example_single_sim.ipynb)**: Basic single-slice simulation for both sequencing-based and imaging-based ST data
  - Visualization of parameter clouds
  - Quality evaluation metrics
  - Expression alteration examples

To get the datasets for tutorial, you can download via https://drive.google.com/drive/folders/1lOQasZ9nxIDIZwlqEQDCBY0kJaA7GwZD?usp=drive_link


## Architecture

```
FEAST-sim/
├── FEAST_core/          # Core simulation engine
│   ├── simulator.py     # Main simulation logic (G-SRBA algorithm)
│   ├── parameter_cloud.py  # Parameter cloud modeling
│   └── APIs.py          # Unified FEAST API
├── alignment/           # Alignment simulation
│   ├── alignment_simulator.py
│   └── spatial_align_alter.py  # Rotation & warping transformations
├── deconvolution/       # Deconvolution simulation
│   ├── deconvolution_simulator.py
│   └── generate_deconvolution.py
├── interpolation/       # 3D interpolation
│   ├── interpolation_pipeline.py
│   ├── parameter_interpolation.py
│   └── coordinate_generation.py
└── modeling/            # Statistical models
    ├── StudentT_mixture_model.py
    ├── Beta_mixture_model.py
    └── marginal_alteration.py
```

## 📊 Reproduction Scripts

The `reproduction/` folder contains scripts to reproduce all benchmarking results from the paper. Each subdirectory corresponds to a specific analysis:

```
reproduction/
├── 1_Simulator_benchmark/     # Figure 2: Simulation fidelity evaluation
├── 2_Clustering_simulation/   # Figure 3: Clustering robustness testing
├── 3_Alignment_simulation/    # Figure 4: Alignment algorithm benchmarking
├── 4_Deconvolution_simulation/# Supp Fig: Deconvolution ground truth generation
└── 5_Interpolation_simulation/# Figure 5: 3D slice interpolation evaluation
```

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
| **MERFISH** | MERFISH | [Allen Brain Atlas](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html) | Simulation, Deconvolution, Interpolation | `MERFISH_006.h5ad`<br>`MERFISH_007.h5ad`<br>`MERFISH_005-009.h5ad` (5 files) |
| **OpenST** | OpenST | [GEO: GSE251926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926) | Simulation | `OpenST_005.h5ad`<br>`OpenST_006.h5ad` |
| **Stereo-seq** | Stereo-seq | [MOSTA](https://www.sciencedirect.com/science/article/pii/S0092867422003993) | Simulation | `Stereoseq_E14_5_E2S2.h5ad` |
| **Slide-seq** | Slide-seqV2 | [STOmics](https://gene.ai.tencent.com/SpatialOmics/dataset?datasetID=120) | Simulation | `Slideseq_001.h5ad` |
| **Xenium** | Xenium | [10X Genomics](https://www.10xgenomics.com/datasets/human-lymph-node-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard) | Simulation | `Xenium_LymphNode.h5ad` |

### Quick Setup

1. **Download datasets** from the sources above
2. **Rename files** following the naming convention (e.g., `151670.h5ad` → `DLPFC_151670.h5ad`)
3. **Place in `data/` directory** at repository root
4. **Install benchmark tools**:
   ```bash
   # For clustering benchmarks
   pip install GraphST STAGATE mclust
   
   # For alignment benchmarks
   pip install spateo-release SpaCEL
   
   # For deconvolution benchmarks
   pip install cell2location
   ```

### Running Benchmarks

Each directory contains ready-to-run scripts:

```bash
# Simulator benchmark (Figure 2)
cd reproduction/1_Simulator_benchmark
python ParameterCloud_sim.py

# Clustering benchmark (Figure 3)
cd reproduction/2_Clustering_simulation
bash run_clust_pipeline.bash

# Alignment benchmark (Figure 4)
cd reproduction/3_Alignment_simulation
bash run_alignment.bash

# Deconvolution simulation
cd reproduction/4_Deconvolution_simulation
python run_deconvolution.py

# Interpolation benchmark (Figure 5)
cd reproduction/5_Interpolation_simulation
python alignment_with_gap.py
python interpolation_sim_pipeline.py
```

### Output Structure

Results are saved in each subdirectory:
- `results/`: Main benchmark outputs (CSVs, plots)
- `simulated_data/`: Generated synthetic datasets
- `reconstructions/`: Interpolated slices (for 3D interpolation)

### Notes

- Scripts automatically create output directories
- Most benchmarks can run in parallel (adjust `n_jobs` parameter)
- GPU recommended for alignment and interpolation (uses Spateo)
- Expected runtime: 1-8 hours per benchmark depending on dataset size

## 📄 Citation

If you use FEAST-sim in your research, please cite:

```bibtex
@article{chen2025feast,
  title={From features to slice: parameter-cloud simulation and 3D interpolation of spatial transcriptomics},
  author={Chen, Yiru and Xie, Manfei and Hu, Yunfei and Yuan, Weiman and Li, Bingshan and Zhang, Lu and Zhou, Xin Maizie},
  journal={bioRxiv},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📧 Contact

- **Yiru Chen**: yiru.22@intl.zju.edu.cn
- **Xin Maizie Zhou**: maizie.zhou@vanderbilt.edu
- **Lu Zhang**: ericluzhang@hkbu.edu.hk

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


**Note**: FEAST-sim is actively maintained. If you have any question, please let me know!