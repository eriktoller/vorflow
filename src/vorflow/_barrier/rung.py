"""Ladders in narrow wedges: one row of cells across the sector between two straight legs.

Sites sit on the wedge's bisector at distances sigma_k from the apex.
Consecutive sites are separated by a *chord*: the perpendicular bisector
of sigma_k-1 and sigma_k, at y_k = (sigma_k-1 + sigma_k) / 2 along the
bisector, which meets both legs at the same distance t_k = y_k / cos(theta/2).
Each chord end is a 1D node whose circle passes through both sites, and
the mirror image of each site across the leg is the leg's outer site. So
every leg segment between two chords is an exact Voronoi face, whatever the
sigmas are, as long as they increase (Berge et al. 2019's ladder; band's
rung zone).

- Plain wedge: sigma_0 sits at 2/3 of the first chord (the apex triangle is
  the cell of sigma_0); the apex node's radius is sigma_0.
- Truncated spike: the first chord is at y_0 (the two new ends); sigma_0 is
  the mirror of sigma_1 across it and lies on the apex side, so its cell
  holds the tip.

Strip heights grow by at most LADDER_GROWTH, stay below the local spacing
and below LADDER_ASPECT x the chord width. The ladder ends at the first
chord wide enough for a pair row on each leg (LADDER_WIDE x the last strip
height), or where the legs end (LADDER_LEG_FRACTION of the shorter piece).
"""
from __future__ import annotations

import math

import numpy as np

LADDER_GROWTH = 1.3
LADDER_ASPECT = 1.2
LADDER_WIDE = 1.7       # 2 x half-offset (0.5) + 0.7 gap, in units of the strip height
LADDER_LEG_FRACTION = 0.45
LADDER_MAX = 60


def ladder(theta: float, t0: float, sigma0: float | None, spacing, t_limit: float) -> tuple:
    """(chord positions y along the bisector, site positions sigma) of a ladder.

    ``spacing(t)`` is the local target spacing at leg distance t from the
    apex; ``t_limit`` the farthest chord position along the legs. With
    ``t0 > 0`` the ladder starts at a truncation chord (sigma0 is ignored),
    else at the apex with ``sigma0`` (or 2/3 of the first chord).
    """
    c, tn = math.cos(theta / 2), math.tan(theta / 2)
    y_limit = t_limit * c
    if t0 > 0:
        y = [t0 * c]
        height = min(LADDER_ASPECT * 2 * y[0] * tn, spacing(t0), 0.5 * (y_limit - y[0]))
        sigma = [y[0] - height / 2, y[0] + height / 2]
    else:
        y1 = min(spacing(0.0) * c, 0.5 * y_limit)
        s0 = sigma0 if sigma0 is not None else 2.0 * y1 / 3.0
        y1 = max(y1, 1.25 * s0)
        sigma, y, height = [s0, 2 * y1 - s0], [y1], y1
    while len(y) < LADDER_MAX:
        offset = sigma[-1] - y[-1]
        target = min(LADDER_GROWTH * height, spacing(y[-1] / c), LADDER_ASPECT * 2 * y[-1] * tn)
        height = max(target, 1.5 * offset)
        y_next = y[-1] + height
        if y_next > y_limit:
            if y_limit > sigma[-1] + 0.2 * offset:
                y.append(y_limit)
            else:
                sigma.pop()
            break
        y.append(y_next)
        if 2 * y_next * tn >= LADDER_WIDE * height:
            break
        sigma.append(2 * y_next - sigma[-1])
    return np.array(y), np.array(sigma[:len(y)])


def ladder_points(apex: np.ndarray, u_a: np.ndarray, u_b: np.ndarray, theta: float, y: np.ndarray,
                  sigma: np.ndarray) -> tuple:
    """(chord ends on leg a, on leg b, sites) as coordinates."""
    c = math.cos(theta / 2)
    bis = (u_a + u_b) / np.hypot(*(u_a + u_b))
    t = y / c
    return apex + np.outer(t, u_a), apex + np.outer(t, u_b), apex + np.outer(sigma, bis)
