---
name: feast-install
description: Install FEAST spatial transcriptomics software from this repository, select or reuse a Python environment, diagnose installation and import failures, and verify a usable installation for analysis. Use when a user asks an agent to set up FEAST or its notebook environment. Does not cover package development, publishing releases, the Feast feature store, or microbial source-tracking software.
---

# Install FEAST for analysis

Help the user obtain a working FEAST installation and identify the interpreter
to use for their scripts or notebooks. Follow the current checkout's dependency
files; do not change package source or dependency constraints to force installation.

## Select the checkout and environment

This skill is stored at `.agents/skills/feast-install/` inside FEAST. Resolve the
repository root from that location or the user's supplied checkout. Read
`docs/source/installation.rst`, `environment.yml`, `requirements.txt`, and
`pyproject.toml` before installing; they may change after this guide was written.

The current package requires Python 3.11. Its distribution name is `FEAST-py`,
and the Python import is `FEAST`. Do not install the unrelated `feast` package.

Inspect the selected interpreter with `python --version`, `python -m pip --version`,
and, when available, `conda env list`. Reuse a compatible FEAST environment that
the user has selected. Otherwise create a dedicated environment using the
documented recipe below. Avoid installing into Conda base or changing a shared
analysis environment without a reason grounded in the user's request.

Use the existing checkout when present. For a user who has no checkout, obtain
it with `git clone https://github.com/maiziezhoulab/FEAST` in their chosen parent
directory, then work from `FEAST/`. Do not clone over an existing directory.

## Documented Conda installation

Run from the FEAST repository root. The commands below use the name currently
specified by `environment.yml`. If that environment already exists, inspect and
reuse it when suitable; do not rerun creation or delete it to resolve a name clash.
Use a different dedicated name if another environment is needed.

```bash
conda env create -f environment.yml
conda run -n feast-py311-conda python -m pip install --no-deps -r requirements.txt
conda run -n feast-py311-conda python -m pip install --no-deps -e .
```

`conda run` makes each agent command select the same interpreter without relying
on shell activation carrying over between tool calls. For an environment created
at a custom prefix, use `conda run -p /path/to/environment` instead. In an
interactive terminal, the user can activate the environment and run the same
`python -m pip` commands directly.

The Conda file supplies the main dependency stack. `requirements.txt` supplies
the additional thin-plate-spline dependency, imported as `tps`. `--no-deps` is
appropriate for this staged recipe, not as a way to ignore missing dependencies
in an arbitrary environment. Installing only `requirements.txt` does not install
the full FEAST dependency stack. The editable install uses this checkout, so keep
it available at its installed location.

The supplied Conda file selects a CUDA-enabled PyTorch build. Core synthetic
examples and the default NumPy transport backend do not require a GPU. If the
file cannot solve on the user's platform, inspect the actual solver error and
platform before proposing an alternative environment. For platform-specific
PyTorch installation commands, consult its current official installation guidance;
do not guess accelerator package versions or rewrite the tracked environment
file as a local workaround. A requested GPU workflow needs a separate check
that Torch can access the intended device.

## Verify the selected installation

Run these with the environment's Python, using `conda run` or its absolute
interpreter path for agent commands. Do not add `PYTHONPATH=src`: verification
should establish that the package is installed and resolves to the intended copy.

```bash
python -m pip check
python -c "import sys, FEAST; from importlib.metadata import version; print(sys.executable); print(version('FEAST-py')); print(FEAST.__file__)"
```

Then run this small in-memory example with that same interpreter. It needs no
datasets, GPU, or output files.

```python
import numpy as np
from FEAST import generate, de_novo

blueprint = de_novo.SimulationBlueprintBuilder.rectangular_grid(2, 2).build()
cloud = (
    de_novo.SimulationParameterBuilder.from_gene_names(["gene_a", "gene_b"])
    .set_all(mean=2.0, variance=3.0, zero_prop=0.1)
    .build()
)
result = generate(blueprint, cloud, seed=7)
assert result.shape == (4, 2)
assert np.issubdtype(result.X.dtype, np.integer)
assert np.all(result.X >= 0)
assert result.obsm["spatial"].shape == (4, 2)
print("FEAST installation example passed:", result.shape)
```

Report any dependency conflicts from `pip check` separately from example success.
For ordinary installation, this check and example are enough; real-data tutorials
and the repository test suite are not prerequisites for using the package.

## Notebooks and common installation problems

| Situation | Action |
| --- | --- |
| Notebook cannot import FEAST but terminal can | Inspect `sys.executable` in both; select a kernel backed by the FEAST environment. If needed for the requested notebook setup, install `ipykernel` there and register a user kernel with that interpreter. |
| Agent shell does not recognize `conda activate` | Use `conda run -n NAME` or the environment's absolute Python path. |
| Python version is rejected | Select/create Python 3.11; preserve the package's declared constraint. |
| Import resolves to another checkout | Check `FEAST.__file__`, the selected interpreter, and any `PYTHONPATH`; install the requested checkout into the intended environment. |
| `tps` is missing | Complete the `requirements.txt` step with the same interpreter. The dependency is named `thin-plate-spline`. |
| Dependency import or binary compatibility fails | Read the specific error and compare installed versions with the checkout's dependency files; repair the affected dedicated environment rather than repeatedly upgrading everything. |
| Downloads or solving fail | Distinguish network/access failure from dependency incompatibility. Report the failed step and cause; preserve certificate verification and configured package-source security. |
| Matplotlib or Numba cannot write a cache | Check the environment's `MPLCONFIGDIR` and `NUMBA_CACHE_DIR` and choose user-writable locations if needed. |

For real-data tutorials, read `tutorial/README.md` for the selected notebook's
data and environment requirements. GraphST and scVI are run in separate
environments by those tutorials; do not add them to the core FEAST environment
merely to verify FEAST. Missing prepared input files are a data-setup issue,
not proof that installation failed.

Finish by reporting the checkout, environment name or path, interpreter, package
version, checks actually run, and any unresolved installation issue. For analysis
help after installation, use the [FEAST user skill](../feast/SKILL.md).
