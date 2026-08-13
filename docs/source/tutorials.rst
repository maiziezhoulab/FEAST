Tutorials
=========

The repository includes a Jupyter notebook demonstrating core workflows.

- **Single-slice simulation** (:file:`example_single_sim.ipynb`): Simulate
  sequencing-based (Visium) and imaging-based (MERFISH) ST data, with and
  without controlled expression alterations.

Single-slice simulation
-----------------------

Basic simulation from a reference AnnData:

.. code-block:: python

    from FEAST import simulate
    import scanpy as sc

    adata = sc.read_h5ad("your_spatial_data.h5ad")
    simulated = simulate(adata, verbose=True)

Simulation with expression alteration:

.. code-block:: python

    from FEAST import simulate, Alteration

    config = Alteration.mean_only(fold_change=0.95)
    altered = simulate(adata, alteration=config)

Alignment simulation
--------------------

Generate paired datasets with known geometric transformations for benchmarking
alignment algorithms.  Use :func:`~FEAST.spatial_transform.rotate` to
transform coordinates, then :func:`simulate` with the transformed spatial
positions:

.. code-block:: python

    from FEAST import simulate
    from FEAST.spatial_transform import rotate
    import numpy as np

    coords = adata.obsm["spatial"]
    rotated_coords = rotate(coords, angle=30.0)
    adata.obsm["spatial"] = rotated_coords
    simulated = simulate(adata)

For a one-step convenience function with built-in edge handling and benchmark
metadata, use the alignment subpackage:

.. code-block:: python

    from FEAST.alignment import simulate_alignment_rotation

    original, rotated = simulate_alignment_rotation(
        adata, rotation_angle=30.0, data_type="imaging",
    )

Deconvolution simulation
------------------------

Create ground-truth data with known cell-type mixtures:

.. code-block:: python

    from FEAST.deconvolution import create_deconvolution_benchmark_data

    benchmark = create_deconvolution_benchmark_data(
        adata=single_cell_adata,
        downsampling_factor=0.25,
        grid_type="hexagonal",
        cell_type_key="cell_type",
    )

De novo virtual slice generation
--------------------------------

Build virtual slices from blueprints, parameter clouds, and spatial patterns:

.. code-block:: python

    from FEAST import generate, SliceBlueprint, Alteration
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

Conditional generation from a reference:

.. code-block:: python

    from FEAST import generate_from, SliceBlueprint

    virtual = generate_from(
        reference_adata, blueprint,
        label_key="domain", seed=42,
    )
