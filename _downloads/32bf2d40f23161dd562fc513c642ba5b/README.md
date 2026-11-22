# FEAST tutorials

Six concise, real-data notebooks explain the main simulation calls and how to interpret their outputs. Start with [setup below](#setup).

| Notebook | Main task | Key result |
|---|---|---|
| [00 · Single slice + GraphST](00_single_slice_graphst.ipynb) | Simulate DLPFC 151675 and cluster its expression | Gene-statistic checks, layer map and clustering ARI |
| [02 · Alignment](02_alignment.ipynb) | Rotate a simulated 151675 slice within a fixed plate | Partial overlap and known spot correspondence |
| [03 · Deconvolution](03_deconvolution.ipynb) | Simulate MERFISH cells and aggregate them into spots | Known cell-class fractions |
| [04 · Batch effect removal](04_batch_effect_removal.ipynb) | Generate two batches from 151675 and correct with scVI | Batch mixing and retained layer structure |
| [05 · 2D conditional transfer](05_2d_conditional_transfer.ipynb) | Transfer 151675 expression onto known 151676 anatomy | Observed and generated expression on matching spots |
| [07 · 3D atlas transfer](07_3d_transfer.ipynb) | Transfer expression from five male E14 references to E15.5 DevCCF sections | Atlas regions and expression at three z levels |

The saved outputs were verified with FEAST-py 1.0.6, which uses the same simulation code as 1.1.0. Run each notebook from top to bottom. Generated data and method logs are written under `tutorial/outputs/`.

## Setup

Use the full `tutorial/` directory from a repository checkout so the notebooks can find their shared helpers. Install FEAST in Python 3.11, then add Jupyter, scikit-misc for variable-gene selection, and PyYAML for the atlas helper. [Installation instructions](../README.md#installation)

```bash
python -m pip install "FEAST-py==1.1.0" jupyterlab ipykernel scikit-misc pyyaml
python -m ipykernel install --user --name feast-tutorial --display-name "Python 3 (FEAST)"
cd /path/to/FEAST
jupyter lab tutorial/
```

Select **Python 3 (FEAST)** in Jupyter. The tutorial loader copies `layers['counts']` into `X` when provided. Otherwise it uses `X` directly. The shared loader checks count values and identifiers; it does not filter, normalize or relabel inputs.

## Prepared inputs for notebooks 00–05

These examples use the prepared H5ADs from the article’s reproduction data bundle. The original measurements come from the [spatialLIBD HumanPilot release](https://research.libd.org/spatialLIBD/) and the [Allen Zhuang-ABCA-1 release](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html).

| File under `tutorial/data/` | Used by | Required annotation |
|---|---|---|
| `dlpfc/151675.h5ad` | 00, 02, 04; source in 05 | `obs['ground_truth']` |
| `dlpfc/151676.h5ad` | 05, target | `obs['ground_truth']` |
| `dlpfc/151670.h5ad` | 05, common gene panel only | Gene index; expression is not used |
| `merfish/Zhuang-ABCA-1.007.h5ad` | 03 | `obs['cell_class']` |

Run these commands from the FEAST repository root. Set `INPUTS` to the absolute path of the extracted bundle's `Zenodo/inputs` directory. To use another tutorial data directory, set `FEAST_TUTORIAL_DATA` to its absolute path before starting Jupyter and keep the same subdirectories.

```bash
INPUTS=/absolute/path/to/Zenodo/inputs
DATA="${FEAST_TUTORIAL_DATA:-tutorial/data}"
mkdir -p "$DATA/dlpfc" "$DATA/merfish"
ln -s "$INPUTS/01_clustering/151670.h5ad" "$DATA/dlpfc/151670.h5ad"
ln -s "$INPUTS/02_alignment/151675.h5ad" "$DATA/dlpfc/151675.h5ad"
ln -s "$INPUTS/01_clustering/151676.h5ad" "$DATA/dlpfc/151676.h5ad"
ln -s "$INPUTS/03_deconvolution/Zhuang-ABCA-1.007.h5ad" "$DATA/merfish/Zhuang-ABCA-1.007.h5ad"
```

## GraphST and scVI

Notebook 00 needs GraphST, R, mclust and rpy2. Notebook 04 needs scvi-tools. Each worker runs in its own CUDA-enabled environment. Set the Python executable for the worker you plan to use before launching Jupyter. [GraphST installation](https://github.com/JinmiaoChenLab/GraphST) · [scvi-tools installation](https://docs.scvi-tools.org/en/stable/installation.html)

```bash
export GRAPHST_PYTHON=/path/to/graphst-environment/bin/python
export SCVI_PYTHON=/path/to/scvi-environment/bin/python
```

Training logs are under `tutorial/outputs/`. [GraphST worker](graphst_step.py) · [scVI worker](scvi_step.py)

## Case 07: public sources and prepared inputs

Use the five E14 male MERFISH sections (GSM8323035–GSM8323039) and the merged E15.5 DevCCF atlas supplied in `Zenodo/inputs/07_3d_transfer/`. With `INPUTS` and `DATA` set as above, make the study directory available to the notebook:

```bash
ln -s "$INPUTS/07_3d_transfer" "$DATA/07"
```

The notebook reads the annotated H5ADs, merged NIfTI and region schema. Run it from a full FEAST checkout so it can import the article’s atlas coordinate helpers from `figure_reproduce/`. It generates sections 47, 78 and 109 on CUDA using two eligible references per region and the article’s E15.5 assignment-randomness setting of 0.30. Loading the five 550-gene references requires substantial host memory. [MERFISH source (GEO)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE269617) · [DevCCF source (Figshare)](https://figshare.com/articles/dataset/DevCCFv1/26377171)
