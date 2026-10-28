"""vorflow with 20 Lloyd passes.

Same spec translation as ``vorflow_adapter``; only the tessellator settings
differ. Written centres are still the (moved) generators. Reported as its own
tool so it can be compared with plain vorflow at matched cell counts.
"""

from pathlib import Path

from ..case import Case
from ..grid import Grid
from . import vorflow_adapter

TOOL = "vorflow_lloyd20"
TESSELLATOR_KWARGS = {"lloyd_iterations": 20}


def build(case: Case, scale: float, ws: Path) -> Grid:
    """Generate the grid for one case at one spec scale."""
    grid = vorflow_adapter.build(case, scale, ws, tessellator_kwargs=TESSELLATOR_KWARGS, tool=TOOL)
    grid.info.update(TESSELLATOR_KWARGS)
    return grid
