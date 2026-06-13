import importlib
from pathlib import Path

import pytest


def test_interpolation_package_absent():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("FEAST.interpolation")


def test_de_novo_does_not_import_interpolation():
    root = Path(__file__).resolve().parents[1] / "src" / "FEAST" / "de_novo"
    for path in root.glob("*.py"):
        assert "FEAST.interpolation" not in path.read_text()
        assert "..interpolation" not in path.read_text()


def test_removed_simulator_noise_symbols_absent():
    simulator_path = Path(__file__).resolve().parents[1] / "src" / "FEAST" / "FEAST_core" / "simulator.py"
    source = simulator_path.read_text()
    for token in [
        "base_noise",
        "smooth_noise",
        "noise_scale",
        "_apply_local_neighbor_swapping",
        "Gentle",
        "Exploratory",
    ]:
        assert token not in source
