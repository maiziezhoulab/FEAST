"""Public FEAST package surface."""

from __future__ import annotations

from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec

__version__ = "1.0.1"

from .FEAST_core.APIs import FEAST
from .FEAST_core import simulator


def _module_exists(absolute_module_name: str) -> bool:
    return _find_spec(absolute_module_name) is not None


ALIGNMENT_AVAILABLE = _module_exists(__name__ + ".alignment")
DECONVOLUTION_AVAILABLE = _module_exists(__name__ + ".deconvolution")
DE_NOVO_AVAILABLE = _module_exists(__name__ + ".de_novo")


def __getattr__(name: str):
    if name == "alignment":
        return _import_module(__name__ + ".alignment")
    if name == "deconvolution":
        return _import_module(__name__ + ".deconvolution")
    if name == "de_novo":
        return _import_module(__name__ + ".de_novo")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)


__all__ = [
    "FEAST",
    "simulator",
    "alignment",
    "deconvolution",
    "de_novo",
    "ALIGNMENT_AVAILABLE",
    "DECONVOLUTION_AVAILABLE",
    "DE_NOVO_AVAILABLE",
]
