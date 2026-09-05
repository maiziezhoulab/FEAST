# FEAST tutorials

Six concise, real-data notebooks explain the main simulation calls and how to interpret their outputs. Start with [setup below](#setup).

| Notebook | Main task | Key result |
|---|---|---|
| [00 · Single slice + GraphST](00_single_slice_graphst.ipynb) | Simulate DLPFC 151675 and cluster its expression | Gene-statistic checks, layer map and clustering ARI |
| [02 · Alignment](02_alignment.ipynb) | Rotate a simulated 151675 slice within a fixed plate | Partial overlap and known spot correspondence |
| [03 · Deconvolution](03_deconvolution.ipynb) | Simulate MERFISH cells and aggregate them into spots | Known cell-class fractions |
| [04 · Batch effect removal](04_batch_effect_removal.ipynb) | Generate two batches from 151675 and correct with scVI | Batch mixing and retained layer structure |
| [05 · 2D conditional transfer](05_2d_conditional_transfer.ipynb) | Transfer 151675 expression onto known 151676 anatomy | Observed and generated expression on matching spots |
| [07 · 3D atlas transfer](07_3d_transfer.ipynb) | Transfer E14 reference expression onto E15.5 DevCCF sections | Atlas regions and expression at three z levels |

**Notebooks 00, 02 and 04 all use the same `data/dlpfc/151675.h5ad` file.** Notebook 04 uses fixed, documented deformation coefficients, so no second reference file is needed.

Numbers follow the article studies. Notebook 00 combines the single-slice and clustering examples; Study 06's separate 3D-stack workflow is outside this collection.

Open the notebooks from `FEAST/` or `FEAST/tutorial/` with a FEAST Python 3.11 kernel. Each can be run independently after its inputs are supplied. GraphST and scVI run through two short scripts in their own environments; their core calls are visible in the notebooks and linked scripts. `_utils.py` contains only shared input and plotting helpers.

Notebooks 00 and 04 await fresh results for their shared 151675 input. The other saved images are previews drawn from existing reproduction results, with source notes in each notebook. They are not outputs of a fresh execution of these notebooks. Code cells start unexecuted. Full simulation, GPU training and atlas generation can be substantial; this collection selects one condition per task, without replacing the article's full benchmark workflows.

Notebook code, explanations and preview images belong in Git. `data/`, `outputs/` and Jupyter checkpoints stay local. Public download links for the author's prepared inputs remain pending; case 07 additionally needs the region-annotation and atlas-label preparation described in [case 07 setup](#case-07-public-sources-and-prepared-inputs).

## Setup

These notebooks use real datasets and the prepared annotations used by the article workflows. They do not download data automatically or import code from `FEAST_reproduce`.

## FEAST environment

Install FEAST from the repository's [installation instructions](../README.md#installation). In that environment, install the notebook tools and the Seurat-v3 dependency used by notebooks 00 and 04:

```bash
python -m pip install jupyterlab ipykernel scikit-misc
python -m ipykernel install --user --name feast-tutorial --display-name "Python 3 (FEAST)"
cd /path/to/FEAST
jupyter lab tutorial/
```

Select **Python 3 (FEAST)** in Jupyter. FEAST generation reads raw counts from `layers['counts']` when provided, otherwise from `X`. The shared loader checks count values and identifiers; it does not filter, normalize or relabel inputs.

## Prepared inputs for notebooks 00–05

The author will supply the prepared H5AD files. **Their public download URLs have not yet been supplied.** A generic raw dataset download does not necessarily contain these exact gene identifiers, labels or coordinate conventions.

| File under `tutorial/data/` | Used by | Required annotation |
|---|---|---|
| `dlpfc/151675.h5ad` | 00, 02, 04; source in 05 | `obs['ground_truth']` |
| `dlpfc/151676.h5ad` | 05, target | `obs['ground_truth']` |
| `dlpfc/151670.h5ad` | 05, common gene panel only | Gene index; expression is not used |
| `merfish/Zhuang-ABCA-1.007.h5ad` | 03 | `obs['cell_class']` |

**00, 02 and 04 require only `dlpfc/151675.h5ad`.** The additional DLPFC slices are needed only by the conditional-transfer example.

All files require unique gene and observation identifiers and `obsm['spatial']` with XY coordinates. Use raw, nonnegative counts. Do not silently substitute fine `cell_type` labels for the Study 03 `cell_class` annotation.

Place the supplied files at these paths, or set `FEAST_TUTORIAL_DATA` to another directory with the same layout **before starting Jupyter**. For a local reproduction checkout, the existing inputs can be linked without copying or changing them:

```bash
# Run from FEAST/. Replace this with your local reproduction checkout.
FEAST_REPRODUCE=/path/to/FEAST_reproduce
mkdir -p tutorial/data/dlpfc tutorial/data/merfish
ln -s "$FEAST_REPRODUCE/05_2d_conditional_transfer/data/local/dlpfc/151670.h5ad" tutorial/data/dlpfc/151670.h5ad
ln -s "$FEAST_REPRODUCE/05_2d_conditional_transfer/data/local/dlpfc/151675.h5ad" tutorial/data/dlpfc/151675.h5ad
ln -s "$FEAST_REPRODUCE/05_2d_conditional_transfer/data/local/dlpfc/151676.h5ad" tutorial/data/dlpfc/151676.h5ad
ln -s "$FEAST_REPRODUCE/03_deconvolution/data/local/Zhuang-ABCA-1.007.h5ad" tutorial/data/merfish/Zhuang-ABCA-1.007.h5ad
```

Create each link only when its destination is absent. Downloaded data and links are ignored by Git.

## GraphST and scVI

GraphST's documented Python/dependency environment differs from FEAST's Python 3.11 environment. Follow the [GraphST installation and tutorial](https://github.com/JinmiaoChenLab/GraphST), including R, `mclust` and `rpy2`. Use a separate environment; do not downgrade FEAST to match GraphST. The notebook exchanges a small H5AD containing expression, labels and coordinates with [graphst_step.py](graphst_step.py).

For notebook 04, use an existing environment installed according to the [scvi-tools installation guide](https://docs.scvi-tools.org/en/stable/installation.html). [scvi_step.py](scvi_step.py) contains the complete correction step, using the [SCVI API](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html).

Before launching Jupyter, point to the **Python executable**, not the environment directory:

```bash
export GRAPHST_PYTHON=/path/to/graphst-environment/bin/python
export SCVI_PYTHON=/path/to/scvi-environment/bin/python
```

The supplied method commands request CUDA, matching their reproduction examples. A missing CUDA device is an error. Training logs are written below `tutorial/outputs/`. No package installation or environment modification occurs from the notebooks.

## Case 07: public sources and prepared inputs

The expression source is [GEO GSE269617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE269617); the anatomical source is [DevCCFv1 on Figshare](https://figshare.com/articles/dataset/DevCCFv1/26377171). Use the GEO supplementary download for `GSE269617_RAW.tar`, and the Figshare file browser for the E15.5 atlas annotations and ontology. The [DevCCF publication](https://www.nature.com/articles/s41467-024-53254-w) describes the atlas.

**Those are upstream raw resources, not ready-to-run FEAST inputs.** The current Study 07 uses derived broad-region reference annotations and a merged atlas label map. Replacing these with an arbitrary annotation or a differently oriented atlas changes the experiment.

The preparation sequence used by the existing study is:

1. Convert the GEO count and cell-metadata tables into per-sample H5ADs, preserving cell IDs, XY coordinates and count values.
2. Recover the author's annotations, then apply the study's existing broad-region annotation procedure to the reference cells.
3. Convert the DevCCF annotations to the study coordinate convention and map atlas structure IDs to the same broad-region vocabulary.
4. Extract the expression-free blueprint with Study 07's `prepare.py`: each section records `z_index`, `z_world`, voxel indices, XY coordinates and region labels.

The existing local preparation implementations are `preprocess_gse269617.py`, `recover_gse269617_annotations.py`, `annotate_telencephalon_regions.py` and `build_devccfv1_coordinate_resource.py` under the research project's `Reproduce/Preprocess_helper/`. **These preprocessing scripts are not distributed in this tool repository.** An independent public run therefore still needs their published preparation workflow or a downloadable prepared input bundle. The upstream downloads alone are not enough to run the notebook.

The exact inputs consumed by notebook 07 are:

```text
tutorial/data/07/
├── references/
│   ├── GSM8323035_E14M_1.h5ad
│   ├── ... GSM8323036–GSM8323039, E14M_2–E14M_5 ...
│   ├── GSM8323040_E14F_1.h5ad
│   └── ... GSM8323041–GSM8323044, E14F_2–E14F_5 ...
└── E15.5.blueprint.json.gz
```

All ten references must contain the common 550-gene panel, raw counts, XY coordinates and `obs['region']`. Retained atlas regions are `BT`, `CP`, `DorsalVZ`, `GE`, `IZ`, `SeptalVZ`, `Septum` and `VentralVZ`. The prepared blueprint has 158 E15.5 levels; the notebook selects original indices 47, 78 and 109 before generation. It uses all ten reference slices.

For the existing local study, use its prepared files:

```bash
# Run from FEAST/. These paths are local input locations, not downloads.
FEAST_PROCESSED=/path/to/Datasets/Processed
FEAST_REPRODUCE=/path/to/FEAST_reproduce
mkdir -p tutorial/data/07
ln -s "$FEAST_PROCESSED/GSE269617/h5ad_region_annotated" tutorial/data/07/references
ln -s "$FEAST_REPRODUCE/07_3d_transfer/data/local/blueprints_ot110/E15.5.blueprint.json.gz" tutorial/data/07/E15.5.blueprint.json.gz
```

Use the FEAST environment with working Torch/CUDA for generation. Fitting ten real reference files can require substantial RAM even though the notebook displays only three sections. E18.5 has a different cohort and reference-only randomness calibration; its full workflow remains in Study 07.
