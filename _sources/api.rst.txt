API Reference
=============

This page documents the main public entry points and the builders used in
the tutorials. Start with :func:`FEAST.simulate` for reference-based data
and :func:`FEAST.generate` for a slice designed from a blueprint.

Simulation and fitting
----------------------

.. autofunction:: FEAST.simulate

.. autofunction:: FEAST.generate

.. autofunction:: FEAST.fit

.. autofunction:: FEAST.de_novo.fit_reference

.. autofunction:: FEAST.estimate_assignment_randomness

.. autoclass:: FEAST.GeneParameterSimulator

Configuration
-------------

.. autoclass:: FEAST.Alteration
   :members: mean_only, variance_only, sparsity_only

.. autoclass:: FEAST.TransportConfig

Transport uses NumPy on CPU with float64 solver arithmetic by default.
For GPU transport, use ``transport_backend="torch"`` and
``transport_device="cuda:0"`` in a compatible CUDA environment.

.. autoclass:: FEAST.ReferenceFitConfig

.. autoclass:: FEAST.SimulationConfig

.. autoclass:: FEAST.SimulationReference

De novo builders
----------------

.. autoclass:: FEAST.SliceBlueprint

.. autoclass:: FEAST.de_novo.SimulationBlueprintBuilder
   :members: rectangular_grid, set_domains, build
   :undoc-members:

.. autoclass:: FEAST.de_novo.SimulationParameterBuilder
   :members: from_gene_names, set_all, set_gene, build
   :undoc-members:

.. autoclass:: FEAST.de_novo.SimulationPatternBuilder
   :members: from_gene_names, gradient, hotspot, build
   :undoc-members:

Alignment and deconvolution
---------------------------

.. autofunction:: FEAST.alignment.rotate_spatial

.. autofunction:: FEAST.alignment.apply_spatial_transform

.. autofunction:: FEAST.alignment.simulate_alignment_rotation

.. autofunction:: FEAST.alignment.simulate_alignment_warp

.. autofunction:: FEAST.deconvolution.create_deconvolution_benchmark_data

Batch simulation
----------------

Notebook 04 uses a ``BatchDeformation`` to store three scale coefficients
(``D``) and offsets (``b``). Supply them to ``simulate_batch_effect`` with
``alpha=0`` for a simulated baseline or ``alpha=1`` for the full deformation.
Use ``characterize_batch`` to estimate coefficients from two measured slices.

.. autoclass:: FEAST.BatchDeformation

.. autofunction:: FEAST.characterize_batch

.. autofunction:: FEAST.simulate_batch_effect

Parameter utilities
-------------------

.. autofunction:: FEAST.stats_to_theta

.. autofunction:: FEAST.theta_to_stats

Compatibility
-------------

``generate_from`` is a deprecated compatibility alias. Use
``simulate(reference, target=blueprint, condition_on="domain",
marginal_model="empirical_reference", seed=42)`` for conditional generation.

Local multi-reference 3D generation
-----------------------------------

.. autofunction:: FEAST.simulate_local_references

.. autofunction:: FEAST.simulate_stack

.. autofunction:: FEAST.calibrate_local_references

Give each reference a unique ``uns["reference_name"]``, the same gene panel
and the label column named by ``label_key``. For atlas transfer,
``n_references`` defaults to five references per modeling region.
Use ``None`` to include all eligible references. Stack reconstruction
uses references on both sides of the target z position.
