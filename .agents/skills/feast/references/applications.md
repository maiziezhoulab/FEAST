# Further applications with FEAST

Use this guide when a user asks what else they can do with FEAST or how to extend
their analysis. These directions combine existing public APIs in user scripts or
notebooks. They are possible study designs, not claims of validated performance.
Start with the relevant [worked example](examples.md).

## Choose an application

| Research question | How to use FEAST | What to inspect |
| --- | --- | --- |
| Is a downstream method robust to expression changes? | Simulate an agreed range of mean, variance, or sparsity alterations with `simulate` and `Alteration`. | Realized gene statistics and downstream performance across settings and seeds. |
| Can a method detect known spatial patterns? | Use `generate` with gradients, hotspots, rings, layered, clustered, or diffuse motifs. | The intended motif, the realized count pattern, and detection performance. A design motif is not an exact count map. |
| How does target layout affect an analysis? | Construct blueprints from user-provided coordinates or grid builders; use `generate` or label-conditioned `simulate`. | Coordinates, domain assignments, retained genes, and whether results depend on layout or expression changes. |
| How well does an alignment method recover correspondence? | Use `rotate_spatial` for a pure rotation, or alignment simulation functions for expression plus geometric perturbation. | Known transforms, retained spot identifiers, and correspondence error. |
| How does spatial aggregation affect deconvolution? | Aggregate spatial single-cell counts at several requested downsampling settings with `create_deconvolution_benchmark_data`. | Actual spot counts and estimated versus stored cell-type proportions in matching column order. |
| How variable is a conclusion across simulated datasets? | Repeat the chosen workflow with explicit seeds and the same scientific settings. | Variation in the user's selected summaries or downstream metrics; simulation repeats are not independent biological samples. |
| Can several target tissues share a reference model? | Fit once with `de_novo.fit_reference`, then call `simulate` with that model and each labeled target. | Target label coverage, retained genes, and differences in the resulting spatial patterns. |

## Batch-effect experiments

FEAST exposes `simulate_batch_effect` and `characterize_batch` for experiments
with changes in gene-parameter statistics. `simulate_batch_effect` accepts a
reference, three-element scaling and shift vectors `D` and `b`, an effect strength
`alpha`, and `random_seed`. Those vectors act in transformed parameter space;
they are not direct count fold changes. Consult the public function documentation
when choosing settings and inspect realized means, variances, and sparsity.

Use this route when the scientific question concerns that model of batch effects.
For a straightforward mean or sparsity perturbation, start with `Alteration`.
Do not present simulated batch effects as a complete model of platform differences.

## Virtual slices between measured sections

For users with multiple labeled reference sections and known z positions,
`de_novo.simulate_stack` can generate virtual slices at specified interior z
positions. Supply reference slices, their z values, target z values, and target
blueprints keyed by z. It requires at least two references, unique reference z
values, and targets strictly inside the reference range that do not equal a
reference z value.

Before running, establish common coordinate units, corresponding domain labels,
and the desired target layouts. Inspect both individual slices and changes across
z. The resulting slices are simulations under the chosen assumptions; this API
does not establish that an unmeasured tissue section has been reconstructed
accurately, and it does not support extrapolating outside the reference range.

## Turn an idea into a practical analysis

Use the user's research question to choose which setting varies, which inputs
remain fixed, and which output will answer the question. Reuse an existing
example and start with a small run. Keep settings, seeds, and meaningful labels
with results in the user's analysis directory. Compare realized expression and
geometry before attributing downstream changes to the intended perturbation.

For an unsupported motif, modality, or biological mechanism, explain what current
FEAST controls can express and where the requested behavior exceeds them. Offer
an approximation only if its assumptions are acceptable for the user's question.
