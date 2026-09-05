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

Local multi-reference 3D generation
-----------------------------------

.. autofunction:: FEAST.simulate_local_references

.. autofunction:: FEAST.simulate_stack

.. autofunction:: FEAST.calibrate_local_references

The conditional decoder also accepts keyword-only ``count_model_params``, a
mapping from modeling group to full core converter output. Each entry must
include gene-indexed ``target_stats`` and explicit preservation-policy metadata.
``target_stats`` may identify genes through its index or a ``gene_id`` column.
FEAST rejects missing or duplicate genes and aligns a copy to the model's gene
order. The same aligned table supplies metadata and count decoding; the caller's
table is not modified.
``group_reference_weights`` selects the same applicable references for OT and
count decoding; ``count_seed`` separates count randomness from the spatial seed.

Each reference passed to ``simulate_local_references`` must define a unique
``uns["reference_name"]``. Keep cache paths scoped to the same unchanged source
references and generation configuration.

``simulate_local_references`` fits and caches generated reference parameter
tables, merges locally sparse labels (positive reference populations below 50),
fuses gene statistics in log/logit space, applies a per-target batch draw, and
converts at the target group population. It keeps original labels in output and
records local memberships, donors, weights, requested statistics and timings in
``uns['local_generation_json']``. ``uns['de_novo']['count_diagnostics']`` records
realized statistics, intensity attenuation and clipping.

The keyword-only ``n_references`` defaults to 5. In stack reconstruction,
references always bracket the target in actual z; the remaining references are
nearest by distance with reference-ID tie breaks. ``None`` uses the full pool.
The bandwidth remains the median adjacent actual-z spacing of that pool.
External transfer uses existing per-group geometry weights and permits ``None``
to retain the full cohort. For finite counts, geometry selection and local
merging are resolved together before fitting: only selected reference/group
support can trigger a merge. If a merge changes which references rank highest,
the mapping is rebuilt from original labels on the new selection. OT and count
parameter fitting consume that same final mapping and reference weights. If
selection and merging cycle without a consistent mapping, generation reports
the failure instead of retaining an excluded reference's influence. No whole-stack statistical model or z smoothing is
used. Existing two-dimensional empirical conditional callers remain supported.
