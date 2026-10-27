from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vorflow")
except PackageNotFoundError:
    __version__ = "0+unknown"

from ._log import set_verbosity

# Default: show progress messages, like the historical print() output.
# Call vorflow.set_verbosity(0) to silence everything except warnings/errors.
set_verbosity(1)

from ._barrier import BarrierMeshError
from .blueprint import ConceptualMesh
from .engine import MeshGenerator
from .tessellator import VoronoiTessellator
from .fields import (
    ExponentialField,
    GeometricGrowthField,
    MeshField,
    ThresholdField,
)

__all__ = ["ConceptualMesh", "MeshGenerator", "VoronoiTessellator",
           "MeshField", "ThresholdField", "ExponentialField",
           "GeometricGrowthField",
           "BarrierMeshError", "set_verbosity"]
