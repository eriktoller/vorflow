"""Lloyd polish of the free mesh nodes around the barrier sites; the sites and every fixed node stay put.

The barrier sites are closed-form and exact, so only the free background
generators near them move (``frames`` picks those within POLISH_BAND x
their size of a circle and passes the nodes within POLISH_HALO more as
fixed), with vorflow's density-weighted Lloyd (``_lloyd.relax``;
rho = h^-4, h the Gmsh node sizes smoothed as for ``lloyd_iterations``).
One iteration:

1. Voronoi of sites + free + fixed generators; weighted centroids.
2. Each free generator moves DAMPING x (centroid - x) where
   ``_lloyd.accept_moves`` allows (it stays in its zone and crosses no
   embedded or barrier line).
3. A move that lands inside a barrier circle is pushed radially out of it
   (``_lloyd.keep_out_of_circles``); one that cannot be pushed clear, or
   whose push crosses a line or leaves the zone, stays.
4. Stop when the largest residual / h falls below TOLERANCE, or after the
   given number of iterations.

DAMPING 1.3 over-relaxes mildly: the barrier rows are fixed, so plain
Lloyd (1.0) converges slowly next to them, while 1.6 with 10 iterations
left the barrier cells' outer neighbours half-way and lowered their
compactness.

Then vorflow's short-face cleanup opens faces shorter than SHORT_FACE x
their connector, with moves into a circle refused. The circles stay empty,
so the barrier faces stay exact.
"""
from __future__ import annotations

import numpy as np

from .. import _lloyd

ITERATIONS = 20
DAMPING = 1.3
MAX_DAMPING = 2.0   # over-relaxation beyond this diverges
TOLERANCE = 2e-3
SHORT_FACE = 0.1


def lloyd(sites: np.ndarray, free_xy: np.ndarray, owners: np.ndarray, fixed_xy: np.ndarray, circles: tuple,
          size_fn, domain, lines, iterations: int = ITERATIONS, damping: float = DAMPING,
          tolerance: float = TOLERANCE) -> tuple:
    """(free xy, report) after the polish iterations (module docstring); everything in global coordinates."""
    nodes = np.vstack([sites, free_xy, fixed_xy])
    free = np.zeros(len(nodes), dtype=bool)
    free[len(sites):len(sites) + len(free_xy)] = True
    all_owners = np.full(len(nodes), None, dtype=object)
    all_owners[free] = owners
    moved, report = _lloyd.relax(nodes, free, size_fn, domain, all_owners, lines, iterations, damping, tolerance,
                                 circles=circles, max_damping=MAX_DAMPING)
    return moved[free], {"iterations": report["iterations"], "residual": report["max_rel_shift"],
                         "rejected": report["rejected"]}


def open_short_faces(sites: np.ndarray, free_xy: np.ndarray, owners: np.ndarray, fixed_xy: np.ndarray,
                     circles: tuple, lines, fraction: float = SHORT_FACE) -> tuple:
    """(free xy, report): vorflow's short-face cleanup on the free generators, kept out of the circles."""
    gens = np.vstack([sites, free_xy, fixed_xy])
    free = np.zeros(len(gens), dtype=bool)
    free[len(sites):len(sites) + len(free_xy)] = True
    all_owners = np.full(len(gens), None, dtype=object)
    all_owners[free] = owners
    moved, report = _lloyd.remove_short_faces(gens, free, all_owners, lines, fraction, circles=circles)
    return moved[free], report
