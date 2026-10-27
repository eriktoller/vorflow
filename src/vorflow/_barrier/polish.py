"""Lloyd polish of the free mesh nodes around the barrier sites; the sites and every fixed node stay put.

The barrier sites are closed-form and exact, so only the free background
generators move, with vorflow's density-weighted Lloyd (``_lloyd``; rho =
h^-4, h the Gmsh node sizes smoothed as for ``lloyd_iterations``). One
iteration:

1. Voronoi of sites + free + fixed generators; weighted centroids.
2. Each free generator moves DAMPING x (centroid - x) where
   ``_lloyd.accept_moves`` allows (it stays in its zone and crosses no
   embedded or barrier line).
3. A move that lands inside a barrier circle is pushed radially out of it
   (``_lloyd.keep_out_of_circles``); one that cannot be pushed clear stays.
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
TOLERANCE = 2e-3
SHORT_FACE = 0.1


def lloyd(sites: np.ndarray, free_xy: np.ndarray, owners: np.ndarray, fixed_xy: np.ndarray, circles: tuple,
          size_fn, domain, lines, iterations: int = ITERATIONS, damping: float = DAMPING,
          tolerance: float = TOLERANCE) -> tuple:
    """(free xy, report) after the polish iterations (module docstring); everything in global coordinates."""
    report = {"iterations": 0, "residual": 0.0, "rejected": 0}
    if len(free_xy) == 0:
        return free_xy, report
    n_sites = len(sites)
    for it in range(iterations):
        gens = np.vstack([sites, free_xy, fixed_xy])
        idx = np.arange(n_sites, n_sites + len(free_xy))
        centroids = _lloyd.weighted_centroids(gens, idx, size_fn, domain)
        target = free_xy + damping * (centroids - free_xy)
        ok = _lloyd.accept_moves(free_xy, target, owners, lines)
        target, ok = _lloyd.keep_out_of_circles(free_xy, target, ok, owners, circles)
        residual = np.where(ok, np.hypot(*(centroids - free_xy).T), 0.0) / size_fn(free_xy)
        free_xy = np.where(ok[:, None], target, free_xy)
        report = {"iterations": it + 1, "residual": float(residual.max()),
                  "rejected": report["rejected"] + int((~ok).sum())}
        if report["residual"] < tolerance:
            break
    return free_xy, report


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
