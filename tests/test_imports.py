import importlib


def test_package_import_surface():
    import FEAST

    assert FEAST.__all__ == [
        "FEAST",
        "simulator",
        "alignment",
        "deconvolution",
        "de_novo",
        "ALIGNMENT_AVAILABLE",
        "DECONVOLUTION_AVAILABLE",
        "DE_NOVO_AVAILABLE",
    ]
    assert hasattr(FEAST, "FEAST")
    assert hasattr(FEAST, "de_novo")
    assert hasattr(FEAST, "alignment")
    assert hasattr(FEAST, "deconvolution")
    assert not hasattr(FEAST, "interpolate_slices")
    assert not hasattr(FEAST, "InterpolationConfig")
    assert not hasattr(FEAST, "INTERPOLATION_AVAILABLE")
    assert not hasattr(FEAST, "alignment_simulator")
    assert not hasattr(FEAST, "deconvolution_simulator")
    assert FEAST.DE_NOVO_AVAILABLE is True


def test_public_subsystem_imports():
    modules = [
        "FEAST.FEAST_core",
        "FEAST.FEAST_core.count_decoding",
        "FEAST.alignment",
        "FEAST.deconvolution",
        "FEAST.modeling",
        "FEAST.de_novo",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_de_novo_public_api_imports():
    import FEAST.de_novo as de_novo

    assert de_novo.__all__ == [
        "SimulationBlueprint",
        "SimulationBlueprintBuilder",
        "SimulationParameterBuilder",
        "SimulationPatternBuilder",
        "ReferenceFitConfig",
        "SimulationReference",
        "SimulationConfig",
        "QuantileFieldConfig",
        "compose_pattern",
        "evaluate_motif",
        "fit_reference",
        "simulate_from_reference",
        "simulate_stack",
        "simulate_from_design",
        "load_blueprint",
        "plot_pattern",
        "plot_pattern_panel",
        "plot_blueprint",
    ]

    from FEAST.de_novo import (
        SimulationBlueprintBuilder,
        ReferenceFitConfig,
        SimulationReference,
        SimulationParameterBuilder,
        SimulationBlueprint,
        SimulationPatternBuilder,
        SimulationConfig,
        QuantileFieldConfig,
        compose_pattern,
        evaluate_motif,
        fit_reference,
        simulate_from_reference,
        simulate_stack,
        simulate_from_design,
        load_blueprint,
        plot_blueprint,
        plot_pattern,
        plot_pattern_panel,
    )

    assert SimulationBlueprint is not None
    assert SimulationBlueprintBuilder is not None
    assert SimulationParameterBuilder is not None
    assert SimulationPatternBuilder is not None
    assert ReferenceFitConfig is not None
    assert SimulationReference is not None
    assert SimulationConfig is not None
    assert QuantileFieldConfig is not None
    assert compose_pattern is not None
    assert evaluate_motif is not None
    assert fit_reference is not None
    assert simulate_from_reference is not None
    assert simulate_stack is not None
    assert simulate_from_design is not None
    assert load_blueprint is not None
    assert plot_pattern is not None
    assert plot_pattern_panel is not None
    assert plot_blueprint is not None


def test_de_novo_old_api_names_not_public():
    import FEAST.de_novo as de_novo

    old_names = [
        "SliceBlueprint",
        "BlueprintBuilder",
        "ParameterCloudBuilder",
        "SpatialPatternBuilder",
        "ConditionalReferenceConfig",
        "ConditionalReferenceModel",
        "VirtualSliceGenerationConfig",
        "compose_gene_pattern",
        "evaluate_spatial_motif",
        "fit_virtual_slice_reference",
        "generate_virtual_slice",
        "generate_virtual_slice_from_design",
        "plot_gene_pattern",
    ]
    for name in old_names:
        assert not hasattr(de_novo, name)


def test_de_novo_old_implementation_names_removed():
    builder = importlib.import_module("FEAST.de_novo.builder")
    conditional = importlib.import_module("FEAST.de_novo.conditional")
    pattern = importlib.import_module("FEAST.de_novo.pattern")

    assert hasattr(builder, "SimulationBlueprintBuilder")
    assert hasattr(builder, "SimulationParameterBuilder")
    assert hasattr(builder, "simulate_from_design")
    for name in ["BlueprintBuilder", "ParameterCloudBuilder", "generate_virtual_slice_from_design"]:
        assert not hasattr(builder, name)

    assert hasattr(conditional, "ReferenceFitConfig")
    assert hasattr(conditional, "SimulationReference")
    assert hasattr(conditional, "SimulationConfig")
    assert hasattr(conditional, "fit_reference")
    assert hasattr(conditional, "simulate_from_reference")
    for name in [
        "ConditionalReferenceConfig",
        "ConditionalReferenceModel",
        "VirtualSliceGenerationConfig",
        "fit_virtual_slice_reference",
        "generate_virtual_slice",
    ]:
        assert not hasattr(conditional, name)

    assert hasattr(pattern, "SimulationPatternBuilder")
    assert hasattr(pattern, "compose_pattern")
    assert hasattr(pattern, "evaluate_motif")
    assert hasattr(pattern, "plot_pattern")
    for name in ["SpatialPatternBuilder", "compose_gene_pattern", "evaluate_spatial_motif", "plot_gene_pattern"]:
        assert not hasattr(pattern, name)


def test_removed_private_api_helpers_are_absent():
    from FEAST.FEAST_core.APIs import FEAST

    for name in ["_get_core_simulator", "_get_alignment_simulator", "_get_deconvolution_simulator"]:
        assert not hasattr(FEAST, name)
