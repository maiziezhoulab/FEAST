# Worked FEAST examples

Use a Python 3.11 environment with FEAST and its dependencies installed. Run
these examples in your analysis notebook or Python session, in the order shown.
They need no downloaded data and write no files. If you need to install FEAST,
follow the repository's `README.md` or `docs/source/installation.rst` first.

## 1. Construct a small count reference

This artificial 8-by-8 grid is for learning the API, not a realistic tissue model.
It provides counts, spatial coordinates, domains, and synthetic cell-type labels
used by later examples.

```python
import anndata as ad
import numpy as np
import pandas as pd
from FEAST import Alteration, ReferenceFitConfig, generate, simulate
from FEAST import de_novo

rng = np.random.default_rng(7)
xx, yy = np.meshgrid(np.arange(8), np.arange(8))
coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
genes = ["GeneA", "GeneB", "GeneC"]
means = np.column_stack([
    1.0 + coords[:, 0],
    1.0 + coords[:, 1],
    np.full(64, 3.0),
])
reference = ad.AnnData(
    X=rng.poisson(means).astype(np.int32),
    obs=pd.DataFrame({
        "domain": np.where(coords[:, 1] < 4, "a", "b"),
        "cell_type": np.where(coords[:, 0] < 4, "type_1", "type_2"),
    }, index=[f"cell_{i}" for i in range(64)]),
    var=pd.DataFrame(index=genes),
)
reference.obsm["spatial"] = coords
reference.layers["counts"] = reference.X.copy()
```

For measured input, load an `.h5ad` and confirm the count matrix, coordinate
units, and relevant label columns. A dissociated single-cell dataset without
spatial locations cannot directly supply the spatial aggregation example.

## 2. Simulate a reference slice and alter its expression

For this three-gene example, use reference statistics directly. The default
`hungarian` mode instead fits gene-parameter distributions and should be explored
with a suitable reference containing enough genes for that modeling task.

```python
simulated = simulate(
    reference, parameter_mode="reference_stats", seed=7, verbose=False,
)
altered = simulate(
    reference,
    parameter_mode="reference_stats",
    alteration=Alteration.mean_only(fold_change=0.8),
    seed=7,
    verbose=False,
)
assert simulated.shape == reference.shape
np.testing.assert_allclose(simulated.obsm["spatial"], coords)
print(simulated.uns["simulation_diagnostics"])
```

Example application: compare a downstream method across an authorized range of
expression mean changes. Record realized means, variances, and zero proportions;
an alteration factor does not imply every realized count scales by that factor.

## 3. Generate a designed slice with spatial motifs

```python
blueprint = (
    de_novo.SimulationBlueprintBuilder.rectangular_grid(8, 8)
    .set_domains(reference.obs["domain"].tolist())
    .build()
)
cloud = (
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
designed = generate(blueprint, cloud, patterns=patterns, seed=7)
assert designed.shape == (64, 3)
assert list(designed.var_names) == genes
assert np.issubdtype(designed.X.dtype, np.integer)
assert np.all(designed.X >= 0)
quantiles = designed.layers["feast_quantiles"]
```

Motifs are evaluated on normalized coordinates; the hotspot center above is
in that normalized space. FEAST builds latent spatial scores, converts them to
ranks/quantiles, and decodes counts using the parameter cloud. The hotspot radius
therefore is not a radius in micrometers, and motif amplitude is not a count value.
Example application: assess spatial pattern detection using known design motifs.

## 4. Generate on a target geometry conditioned on domain labels

```python
conditioned = simulate(
    reference,
    target=blueprint,
    condition_on="domain",
    marginal_model="empirical_reference",
    fit_config=ReferenceFitConfig(
        min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0,
    ),
    seed=7,
    verbose=False,
)
assert conditioned.n_obs == blueprint.n_spots
assert list(conditioned.var_names) == genes
assert conditioned.uns["de_novo"]["conditional_generation"]
```

The permissive gene filters are for this toy example. Choose real-data filtering
to match the analysis. Reference and target labels must correspond. Target
expression does not condition this path; geometry and labels do. A changed
target can have different spot counts, and filtering can change the gene set.

For repeated targets, use `de_novo.fit_reference(reference, label_key="domain",
config=...)` once and pass its `SimulationReference` to `simulate(model,
target=..., marginal_model="empirical_reference", seed=...)`. `FEAST.fit`
returns a different model and is not interchangeable with this fitted reference.

## 5. Make a known rotation for alignment evaluation

```python
from FEAST.alignment import rotate_spatial

rotated = rotate_spatial(reference, angle_degrees=30.0)
assert rotated.obs_names.equals(reference.obs_names)
np.testing.assert_array_equal(rotated.X, reference.X)
assert not np.allclose(rotated.obsm["spatial"], coords)
```

This tests a pure geometric perturbation. To include simulated expression, use
`simulate_alignment_rotation(reference, rotation_angle=30.0,
data_type="imaging", expression_params={"random_seed": 7})`; check its current
bounds and edge-filtering options because the returned pair can have unequal
spot sets. Match observations by identifiers when evaluating correspondence.

## 6. Aggregate cells into spots with known cell-type proportions

```python
from FEAST.deconvolution import create_deconvolution_benchmark_data

benchmark = create_deconvolution_benchmark_data(
    reference,
    downsampling_factor=0.25,
    grid_type="square",
    cell_type_key="cell_type",
)
proportions = benchmark.obsm["cell_type_proportions"]
cell_types = benchmark.uns["cell_type_names"]
assignments = benchmark.uns["spot_assignments"]
assert proportions.shape == (benchmark.n_obs, len(cell_types))
occupied = np.bincount(assignments, minlength=benchmark.n_obs) > 0
np.testing.assert_allclose(proportions[occupied].sum(axis=1), 1.0)
np.testing.assert_allclose(benchmark.X.sum(axis=0), reference.X.sum(axis=0))
```

Example application: compare inferred cell-type proportions to the stored ground
truth in `cell_type_names` order. The downsampling factor guides grid construction;
tissue filtering affects the final number of spots. Inspect coordinate scale and
the tissue-shape `alpha` setting if a real-data grid is empty. Omitting or misspelling
the cell-type key produces counts without ground-truth proportions.

## Adapt and inspect

Check shape, gene ordering, coordinate alignment, nonnegative counts, and the
diagnostics relevant to the workflow. For a reproducibility check, repeat a call
with the same explicit seed. Demonstration checks establish API behavior; claims
about biological fidelity need suitable reference data and scientific evaluation.
Write requested results to the user's chosen analysis/output directory. Keep
the simulation settings and seed alongside the output so the run can be repeated.
