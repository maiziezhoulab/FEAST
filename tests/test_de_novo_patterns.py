import numpy as np

from FEAST.de_novo import SimulationBlueprintBuilder, SimulationPatternBuilder, compose_pattern, evaluate_motif


def _blueprint():
    return SimulationBlueprintBuilder.rectangular_grid(5, 5).set_domains(["a"] * 10 + ["b"] * 15).build()


def test_each_motif_kind_returns_unit_interval_values():
    bp = _blueprint()
    motifs = [
        {"kind": "layered", "axis": "y", "center": 0.5},
        {"kind": "gradient", "axis": "x"},
        {"kind": "hotspot", "center": [0.5, 0.5]},
        {"kind": "ring", "center": [0.5, 0.5], "radius": 0.3},
        {"kind": "clustered", "n_clusters": 2},
        {"kind": "diffuse", "n_components": 3},
    ]
    for idx, motif in enumerate(motifs):
        values = evaluate_motif(bp, motif, random_seed=idx)
        assert values.shape == (bp.n_spots,)
        assert np.all(values >= 0)
        assert np.all(values <= 1)


def test_domain_scoped_composed_pattern():
    bp = _blueprint()
    pattern = compose_pattern(
        bp,
        [
            {"kind": "gradient", "scope": "domain", "domain": "a", "axis": "x"},
            {"kind": "hotspot", "scope": "domain", "domain": "b", "center": [0.5, 0.5]},
        ],
    )
    assert pattern.shape == (bp.n_spots,)
    assert np.all(pattern >= 0)
    assert np.all(pattern <= 1)


def test_spatial_pattern_builder():
    spec = (
        SimulationPatternBuilder.from_gene_names(["g1", "g2"])
        .gradient("g1", axis="x")
        .hotspot("g2", center=[0.5, 0.5])
        .build()
    )
    assert set(spec) == {"g1", "g2"}
    assert spec["g1"][0]["kind"] == "gradient"
