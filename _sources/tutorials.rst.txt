Tutorials
=========

Start with single-slice simulation or choose the notebook for your task.
Each walkthrough includes the code and its results. To run one yourself,
use a Python 3.11 kernel and the full FEAST checkout. Alignment, deconvolution
and 2D transfer run on CPU. GraphST, scVI and atlas generation use CUDA.
The :download:`setup guide <../../tutorial/README.md>` lists the input paths,
kernel setup and worker environments for these notebooks.

.. toctree::
   :maxdepth: 1

   notebooks/00_single_slice_graphst
   notebooks/02_alignment
   notebooks/03_deconvolution
   notebooks/04_batch_effect_removal
   notebooks/05_2d_conditional_transfer
   notebooks/07_3d_transfer

De novo virtual slice generation
--------------------------------

For a small example that runs without a data download, build a 16-spot slice
with two domains, three genes and two spatial patterns.

.. code-block:: python

    from FEAST import generate
    from FEAST import de_novo

    genes = ["GeneA", "GeneB", "GeneC"]

    blueprint = (
        de_novo.SimulationBlueprintBuilder.rectangular_grid(4, 4)
        .set_domains(["cortex"] * 8 + ["medulla"] * 8)
        .build()
    )

    param_cloud = (
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

    virtual_slice = generate(blueprint, param_cloud, patterns=patterns, seed=7)
