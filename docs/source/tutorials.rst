Tutorials
=========

The ``tutorial/`` directory contains six concise notebooks using real data:

- :download:`00: single-slice simulation and GraphST clustering <../../tutorial/00_single_slice_graphst.ipynb>`
- :download:`02: alignment simulation with known correspondence <../../tutorial/02_alignment.ipynb>`
- :download:`03: deconvolution mixtures and composition truth <../../tutorial/03_deconvolution.ipynb>`
- :download:`04: same-slice batch simulation and scVI correction <../../tutorial/04_batch_effect_removal.ipynb>`
- :download:`05: 2D conditional expression transfer <../../tutorial/05_2d_conditional_transfer.ipynb>`
- :download:`07: conditional expression on 3D atlas sections <../../tutorial/07_3d_transfer.ipynb>`

Use the complete ``tutorial/`` folder from a repository checkout: the notebooks
share input/plotting helpers, two external-method scripts and small preview
images. Read ``tutorial/README.md`` for file names, environment setup and pending
prepared-data download links. Case 07 needs prepared region annotations and
atlas labels in addition to the public GEO and DevCCF downloads.

The saved previews come from existing reproduction artifacts and are explicitly
distinguished from fresh notebook execution. Tutorial data and generated outputs
are ignored by Git; notebook source and preview images are included.

Single-slice simulation
-----------------------

Use a reference AnnData with nonnegative expression counts in ``.X``, unique
spot identifiers in ``.obs_names``, gene identifiers in ``.var_names``, and
spatial coordinates in ``.obsm["spatial"]``. The rotation examples require
two-dimensional coordinates.

.. code-block:: python

    from FEAST import simulate
    import scanpy as sc

    adata = sc.read_h5ad("your_spatial_data.h5ad")
    simulated = simulate(adata, seed=7, verbose=True)

The default ``parameter_mode="hungarian"`` fits a parameter cloud. To use
the reference gene statistics directly, select
``parameter_mode="reference_stats"``. With no target or transport settings,
the spatial assignment defaults to ``"reference_rank"``.

Simulation with expression alteration:

.. code-block:: python

    from FEAST import simulate, Alteration

    config = Alteration.mean_only(fold_change=0.95)
    altered = simulate(adata, alteration=config, seed=7)

Alignment simulation
--------------------

Use :func:`~FEAST.alignment.rotate_spatial` to return a rotated copy of a
slice. By default, rotation preserves expression, spot identifiers, and spot
count, and rotates around the coordinate centroid:

.. code-block:: python

    from FEAST.alignment import rotate_spatial

    rotated = rotate_spatial(adata, angle_degrees=30.0)

To simulate expression and then generate an alignment pair, use
:func:`~FEAST.alignment.simulate_alignment_rotation`. This convenience
function uses its own bounds and edge filtering defaults, so the rotated
slice can retain fewer spots:

.. code-block:: python

    from FEAST.alignment import simulate_alignment_rotation

    original, rotated = simulate_alignment_rotation(
        adata, rotation_angle=30.0, data_type="imaging",
        expression_params={"random_seed": 7},
    )

Deconvolution simulation
------------------------

Use spatially resolved single-cell counts with coordinates in
``.obsm["spatial"]`` and cell-type labels in ``.obs["cell_type"]``:

.. code-block:: python

    from FEAST.deconvolution import create_deconvolution_benchmark_data

    single_cell_adata = sc.read_h5ad("your_spatial_single_cell_data.h5ad")

    benchmark = create_deconvolution_benchmark_data(
        adata=single_cell_adata,
        downsampling_factor=0.25,
        grid_type="hexagonal",
        cell_type_key="cell_type",
    )

The result stores proportions in ``.obsm["cell_type_proportions"]`` and
their column labels in ``.uns["cell_type_names"]``. Without the requested
cell-type column, this function produces counts without ground-truth
proportions.

De novo virtual slice generation
--------------------------------

Build virtual slices from blueprints, parameter clouds, and spatial patterns:

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

Conditional generation from a reference
---------------------------------------

Use the same :func:`FEAST.simulate` entry point with ``condition_on`` and
``marginal_model="empirical_reference"``. The reference below must contain
``.obs["domain"]`` labels matching the ``"cortex"`` and ``"medulla"``
domains of the blueprint above. Reference fitting applies the filters in
:class:`FEAST.ReferenceFitConfig`, including a default minimum of 20
expressing spots per retained gene. Check these settings for small datasets.

.. code-block:: python

    from FEAST import simulate

    reference_adata = sc.read_h5ad("your_labeled_reference.h5ad")

    virtual = simulate(
        reference_adata,
        target=blueprint,
        condition_on="domain",
        marginal_model="empirical_reference",
        seed=42,
    )

``generate_from`` remains available as a deprecated compatibility alias.
