---
name: feast
description: Help users apply FEAST to their data. Use for setup, input preparation, reference-based simulation, designed virtual slices, conditional generation, expression alterations, alignment and deconvolution benchmarks, interpreting outputs, and exploring further applications with existing APIs. Not for developing FEAST itself, the unrelated Feast feature store, or microbial source-tracking tools.
---

# FEAST

Help researchers use FEAST through its public Python APIs: choose a workflow,
prepare their data, adapt an example, run the simulation, and interpret the output.
Assume the user wants to carry out an analysis. Write user scripts or notebooks
as needed, and keep package development outside this skill's scope.

## Start from the user's task

Identify the desired output and available inputs: a reference count matrix,
spatial coordinates, domain or cell-type labels, or a design without reference
data. Use information already supplied. Ask only for missing scientific choices
or data details that affect the workflow, and offer the synthetic example when
the user wants to learn without providing a dataset.

- [Installation skill](../feast-install/SKILL.md): environment selection,
  installation commands, import troubleshooting, and a small verification run.
- [Worked examples](references/examples.md): small synthetic inputs, simulation,
  alterations, designed and conditioned slices, alignment, and deconvolution.
- [Further applications](references/applications.md): ways to combine existing
  FEAST capabilities for the user's research and benchmarking questions.

Read only the reference needed for the request. Use an existing FEAST environment
when available; this checkout supports Python 3.11. If installation is needed,
follow the linked installation skill and the current repository environment
files. An installed package can be used from the user's analysis
directory; access to package source is not required for ordinary use. Consult
public API documentation or installed function signatures when a version differs
from an example.

## Choose the workflow

| User objective | Entry point | Key choice |
| --- | --- | --- |
| Simulate from measured counts | `FEAST.simulate(reference, seed=...)` | Default `parameter_mode="hungarian"` fits a parameter cloud; `"reference_stats"` uses reference gene statistics directly. |
| Change expression statistics | `FEAST.simulate(..., alteration=FEAST.Alteration...)` | Supported on the global parameter-cloud path; conditional simulation rejects `alteration`. |
| Design a slice without reference data | `FEAST.generate(blueprint, param_cloud, patterns=..., seed=...)` | Specify geometry, per-gene statistics, and optional spatial motifs. |
| Transfer expression to labeled target geometry | `FEAST.simulate(reference, target=..., condition_on=..., marginal_model="empirical_reference", seed=...)` | Supply corresponding reference and target labels; target expression is not used. |
| Fit gene-parameter distributions | `FEAST.fit(adata)` | Returns `GeneParameterSimulator`; this differs from the `SimulationReference` returned by `de_novo.fit_reference`. |
| Rotate existing coordinates | `FEAST.alignment.rotate_spatial` | Preserves expression and spot identities by default. |
| Simulate an alignment pair | `FEAST.alignment.simulate_alignment_rotation` | Convenience defaults can filter edge spots; check retained IDs before matching. |
| Aggregate spatial cells with known proportions | `FEAST.deconvolution.create_deconvolution_benchmark_data` | Requires spatial single-cell counts and a cell-type column for ground truth. |

Use `simulate` for new conditional examples; `generate_from` is deprecated.
The top-level wrappers use `seed`; many lower-level functions use `random_seed`.
Global simulation selects `reference_rank` by default and OT when a target or
transport configuration is supplied. Do not silently change those modes to
resolve a failure: they change the simulation being requested.

## Inputs and interpretation

Use nonnegative count data in AnnData `.X`, unique observation and gene names,
and coordinates in `.obsm["spatial"]`. Confirm where raw counts are stored
before adapting a normalized dataset; do not overwrite the user's original object.
Some paths also inspect `layers["counts"]`, so keep it consistent with `.X`
when preparing count inputs. Conditional generation additionally needs labels
and may filter genes through `ReferenceFitConfig` (default minimum 20 expressing
spots). Relaxed filters in toy examples are demonstration choices.

Generated counts are stochastic realizations. Requested mean, variance, zero
proportion, or motifs are design targets, not exact guarantees for a small slice.
Compare realized summaries and spatial patterns against the user's objective.
For de novo workflows, `layers["feast_quantiles"]`, when stored, holds the final
rank-normalized quantiles used for count decoding; it is not expression data.
Quantile storage depends on `SimulationConfig` and output size.

Start with a small example when learning or troubleshooting. Report the API,
seed, scientific settings, output shape, and relevant diagnostics. Estimate
memory before scaling dense expression or OT work. Change scientific settings
only when requested or when the user has authorized the corresponding choice.

## Troubleshoot and deliver results

| Symptom | What to check |
| --- | --- |
| Import fails | The active Python environment, FEAST installation, and the named missing dependency. |
| Missing spatial coordinates | Whether `.obsm["spatial"]` exists and has one coordinate row per observation. |
| Conditional generation retains no genes | Count matrix location and the fit filters relative to dataset size; explain any proposed filter change. |
| Target labels fail to match | The selected `.obs` column and corresponding labels in the target blueprint. |
| No deconvolution ground truth | The exact cell-type column name and whether it was passed as `cell_type_key`. |
| Empty aggregation grid | Coordinate units, tissue geometry, and the `alpha` setting. |
| Requested statistics differ from realized counts | Sample size, stochastic variation, model feasibility, and simulation diagnostics. |

Run an adapted example on a small input before a large analysis when useful.
Inspect the output shape, gene names, spatial coordinates, count values, and
workflow-specific ground truth. Preserve the user's original data and write
requested results to their chosen analysis/output directory, outside the skill.
Provide runnable code, explain the parameters that matter for the question, and
report what was actually run. Distinguish a successful simulation from evidence
that it represents the biology of interest. If a requested capability is not
available, explain the limitation and any suitable existing workflow without
silently modifying FEAST.
