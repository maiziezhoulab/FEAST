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

.. autoclass:: FEAST.GeneParameterSimulator

Configuration
-------------

.. autoclass:: FEAST.Alteration
   :members: mean_only, variance_only, sparsity_only

.. autoclass:: FEAST.TransportConfig

The transport defaults use the NumPy backend on CPU, unbalanced Sinkhorn
transport, ``epsilon=0.05``, ``sinkhorn_iter=1000``, and
``sinkhorn_tol=1e-5``. Nonconvergence raises an error by default.
``sinkhorn_method`` also accepts ``"sinkhorn_log"``,
``"sinkhorn_stabilized"``, and ``"sinkhorn_translation_invariant"``.
To select PyTorch, set ``transport_backend="torch"`` and an explicit
``transport_device`` such as ``"cpu"`` or ``"cuda:0"``; CUDA requires
an available compatible GPU.

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

Parameter utilities
-------------------

.. autofunction:: FEAST.stats_to_theta

.. autofunction:: FEAST.theta_to_stats

Compatibility
-------------

``generate_from`` is a deprecated compatibility alias. Use
``simulate(reference, target=blueprint, condition_on="domain",
marginal_model="empirical_reference", seed=42)`` for conditional generation.
