__version__ = "0.1.6"

# Import the unified API as the main FEAST class
from .FEAST_core.APIs import (
    FEAST
)

# Import key components to make them available at the top level
from .FEAST_core import simulator


from .alignment import alignment_simulator
ALIGNMENT_AVAILABLE = True


from .deconvolution import deconvolution_simulator
DECONVOLUTION_AVAILABLE = True


from .interpolation import (
    interpolate_slices,
    InterpolationConfig
)
INTERPOLATION_AVAILABLE = True

__all__ = [
    # Main unified API
    "FEAST",
    # 3D Interpolation module (new!)
    "interpolate_slices",
    "InterpolationConfig",
    
    # Legacy components
    "simulator", 
    "alignment_simulator",
    "deconvolution_simulator",
    
    # Feature flags
    "ALIGNMENT_AVAILABLE",
    "DECONVOLUTION_AVAILABLE",
    "INTERPOLATION_AVAILABLE"
]

