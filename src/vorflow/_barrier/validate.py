"""The validity gate of a barrier layout: the empty-circle and arc conditions.

``defects`` lists every place where the closed-form sites would not make
each barrier segment exactly one Voronoi face:

- a pair whose half-offset is below H_MIN x d, or whose sites are not on
  the circles of its segment's two end nodes (``pair``);
- a node where the arcs spanned by two segments' pairs overlap, or another
  site on the node's circle lies inside a pair's arc (``arc``). Around a
  node, the two sites of a segment span an arc through the segment's
  direction; the segment is a face starting at the node only if no other
  site lies inside that arc;
- a site or feature node strictly inside any circle (``site in circle``,
  ``feature in circle``);
- a site outside the domain (``site outside``).

When none is left, every segment is exactly a Voronoi face (Berge et al.
2019, the fracture condition), whatever other generators lie outside the
circles.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from scipy.spatial import cKDTree

from .sites import Layout

INSIDE_EPS = 1e-9


# ---------------------------------------------------------------------------
def _angle(v: np.ndarray) -> np.ndarray:
    """Angles of the vectors ``v`` (n, 2)."""
    return np.arctan2(v[:, 1], v[:, 0])


def arc_defects(layout: Layout) -> list:
    """Nodes where the arcs of two segments' pairs overlap, or another site lies inside a pair's arc.

    Around a node, the two sites of a segment span an arc through the
    segment's direction; the segment is a face starting at the node only
    if no other site on the circle lies inside that arc.
    """
    seg, xy = layout.segments, layout.xy
    rows = []  # (node, direction, half-width)
    for end in (0, 1):
        node, other = seg[:, end], seg[:, 1 - end]
        direction = _angle(xy[other] - xy[node])
        site = _angle(layout.pairs[:, 0] - xy[node])
        half = np.abs((site - direction + np.pi) % (2 * np.pi) - np.pi)
        rows.append(np.column_stack([node, direction, half]))
    rows = np.vstack(rows)
    on_circle = {}
    for p, v in zip(layout.extra, layout.extra_node):
        if v >= 0:
            on_circle.setdefault(int(v), []).append(math.atan2(*(p - xy[v])[::-1]))
    out = []
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    starts = np.flatnonzero(np.r_[True, rows[1:, 0] != rows[:-1, 0]])
    for a, b in zip(starts, np.r_[starts[1:], len(rows)]):
        node = int(rows[a, 0])
        dirs, half = rows[a:b, 1], rows[a:b, 2]
        tol = 1e-7
        for k in range(len(dirs)):
            gap = np.abs((dirs - dirs[k] + np.pi) % (2 * np.pi) - np.pi)
            clash = (gap < half + half[k] - tol)
            clash[k] = False
            extra = np.array(on_circle.get(node, []))
            inside = len(extra) and (np.abs((extra - dirs[k] + np.pi) % (2 * np.pi) - np.pi) < half[k] - tol).any()
            if clash.any() or inside:
                out.append(("arc", xy[node].copy()))
                break
    return out


def defects(layout: Layout, others: np.ndarray, domain) -> list:
    """[(kind, location)] of every defect: bad pairs, overlapping arcs, points inside circles, sites outside."""
    out = []
    mid = 0.5 * (layout.xy[layout.segments[:, 0]] + layout.xy[layout.segments[:, 1]])
    out += [("pair", m) for m in mid[~layout.pair_ok]]
    out += arc_defects(layout)
    pts = np.vstack([layout.sites, others]) if len(others) else layout.sites
    hits = cKDTree(pts).query_ball_point(layout.xy, layout.radius * (1 - INSIDE_EPS))
    for c, idx in enumerate(hits):
        for k in idx:
            kind = "site in circle" if k < len(layout.sites) else "feature in circle"
            out.append((kind, 0.5 * (layout.xy[c] + pts[k])))
    scale = 1e-9 * max(1.0, float(np.abs(layout.xy).max()))
    outside = ~shapely.covers(domain.buffer(scale), shapely.points(layout.sites))
    out += [("site outside", p) for p in layout.sites[outside]]
    return out


def all_finite(layout: Layout) -> bool:
    """Guard against runaway constructions."""
    return bool(np.isfinite(layout.radius).all() and np.isfinite(layout.sites).all())
