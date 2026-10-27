"""Close parallel barriers: two barriers closer than MERGE_GAP x resolution over a stretch share one line of faces.

Two barriers that run side by side leave a gap that needs a row of cells
of its own: a corridor (``corridor``) whose cells are at most 1.5 x the gap
long, or two rows of faces a fraction of the gap long. Below MERGE_GAP x lc
(lc the finer resolution of the two) that is many cells for a gap the
model does not resolve anyway. Instead both stretches are moved onto their
midline, which then carries the faces of both barriers (``barrier_lines``
lists it under both lines; an HFB on those faces stands for both
barriers, so the user gives it their combined properties).

A *stretch* is a run of samples, every SAMPLE_STEP x lc along one line,
closer than MERGE_GAP x lc to the other line and at least MIN_STRETCH x lc
long. The midline joins the midpoints between each sample and the nearest
point of the other line. Where the stretch ends and a line goes on, the
line leaves the midline at LEAVE_FACTOR x ``sharp_angle`` back to its own
course, so the two lines open at twice that, a workable junction. Where
both lines end in the stretch the midline ends half-way between their
ends, on the domain boundary when both ended on it, so a barrier that
ended on the boundary still does. No point moves more than about half the
local gap, at most MERGE_GAP / 2 x lc; a stretch whose move would exceed
MAX_MOVE x lc is left as it is.

A stretch is also left as it is when the two lines touch or cross in it (a
junction or a crossing at a small angle: ``wedge``), when it or its
transitions hold a vertex that either line shares with a third line, a
standard-line crossing or a point on the boundary (other than both lines'
ends there), or when its new course would meet another line or leave the
domain. Every merge is recorded with its place, gap, length, largest move
and lines.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring

MERGE_GAP = 0.25     # x lc: barriers closer than this over a stretch are merged onto their midline
MIN_STRETCH = 1.0    # x lc: a shorter close stretch is left (the caps or a corridor handle it)
SAMPLE_STEP = 0.125  # x lc: spacing of the samples along a line
LEAVE_FACTOR = 0.75  # x sharp_angle: the angle at which a line leaves the midline (each side)
MAX_MOVE = 0.15      # x lc: a merge that would move a line further is not made
MIDLINE_TOL = 0.01   # x lc: the midline is simplified within this
MAX_MERGES = 1000


def _positions(xy: np.ndarray) -> np.ndarray:
    """Arc position of every vertex of ``xy``."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def _samples(xy: np.ndarray, step: float) -> tuple:
    """(arc positions, points) every ``step`` along ``xy``, and at its vertices."""
    pos = _positions(xy)
    s = np.union1d(pos, np.arange(0.0, pos[-1], step))
    return s, shapely.get_coordinates(shapely.line_interpolate_point(LineString(xy), s))


def _runs(mask: np.ndarray) -> list:
    """(first, last) index of every run of True in ``mask``."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def _at(xy: np.ndarray, s: float) -> np.ndarray:
    """The point at arc position ``s`` along ``xy``."""
    return np.array(LineString(xy).interpolate(float(s)).coords[0])


def _dedupe(xy: np.ndarray) -> np.ndarray:
    """``xy`` without consecutive repeated points."""
    keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 0])
    return xy[keep]


def _end_kind(xy: np.ndarray, i: int, counts: dict, on_boundary) -> str | None:
    """End i of a line: "free", "boundary", or None when another line shares it."""
    if counts[tuple(xy[i])] > 1:
        return None
    return "boundary" if on_boundary(xy[i][None])[0] else "free"


def _on_ring(domain, p: np.ndarray) -> np.ndarray:
    """The point of the domain boundary nearest to ``p``."""
    ring = domain.boundary
    return np.array(ring.interpolate(ring.project(Point(p))).coords[0])


def _side(xy: np.ndarray, lo: float, hi: float, first: np.ndarray, last: np.ndarray, tan: float) -> dict:
    """How one line joins the midline from ``first`` to ``last``, the midline lying along its run [lo, hi]."""
    length = _positions(xy)[-1]
    at_start, at_stop = lo <= 0.0, hi >= length
    s0 = 0.0 if at_start else max(lo - np.hypot(*(first - _at(xy, lo))) / tan, 0.0)
    s1 = length if at_stop else min(hi + np.hypot(*(last - _at(xy, hi))) / tan, length)
    return {"s0": s0, "s1": s1, "at_start": at_start, "at_stop": at_stop}


def _blocked(keep: np.ndarray, xy: np.ndarray, side: dict) -> bool:
    """True when a must-keep vertex of the line lies strictly inside the part it gives up."""
    pos = _positions(xy)
    return bool((keep & (pos > side["s0"]) & (pos < side["s1"])).any())


def _rebuilt(xy: np.ndarray, side: dict, midline: np.ndarray) -> np.ndarray:
    """``xy`` with the part ``side`` gives up replaced by its transitions and the midline."""
    pos = _positions(xy)
    middle = [midline]
    if not side["at_start"]:
        middle.insert(0, _at(xy, side["s0"])[None])
    if not side["at_stop"]:
        middle.append(_at(xy, side["s1"])[None])
    before = xy[pos < side["s0"]] if not side["at_start"] else xy[:0]
    after = xy[pos > side["s1"]] if not side["at_stop"] else xy[:0]
    return _dedupe(np.vstack([before, *middle, after]))


def _ends_meet(kinds: list, flags: list) -> str | None:
    """How the two lines' ends on one side of a stretch meet: "boundary", "free", "one" (one line goes on),
    "none" (both go on), or None when they cannot be merged there."""
    if not any(flags):
        return "none"
    ends = [k for k, f in zip(kinds, flags) if f]
    if None in ends:
        return None  # an end shared with a third line stays where it is
    if all(flags):
        return ends[0] if ends[0] == ends[1] else None
    return "one" if ends[0] == "free" else None  # a boundary end must keep the boundary


def _plan(a, b, s: np.ndarray, pts: np.ndarray, lc: float, sharp: float, counts: dict, on_boundary,
          domain) -> tuple | None:
    """(new xy of a, new xy of b, midline) merging the close run ``pts`` (at ``s``) of line a with line b."""
    bxy = b.xy
    t = LineString(bxy).project(shapely.points(pts))
    reverse = t[-1] < t[0]
    if reverse:
        bxy = bxy[::-1]
        t = LineString(bxy).project(shapely.points(pts))
    t_max = LineString(bxy).length
    if np.any(np.diff(t) < 0):
        return None  # the other line folds back along the stretch
    # Samples past the other line's ends all face its end point: keep only the first of them.
    lo = int(np.flatnonzero(t <= 0.0)[-1]) if (t <= 0.0).any() else 0
    hi = int(np.flatnonzero(t >= t_max)[0]) if (t >= t_max).any() else len(t) - 1
    if s[hi] - s[lo] < MIN_STRETCH * lc:
        return None
    s, pts, t = s[lo:hi + 1], pts[lo:hi + 1], t[lo:hi + 1]
    mid = 0.5 * (pts + shapely.get_coordinates(shapely.line_interpolate_point(LineString(bxy), t)))
    tan = math.tan(LEAVE_FACTOR * sharp)
    sa = _side(a.xy, s[0], s[-1], mid[0], mid[-1], tan)
    sb = _side(bxy, t[0], t[-1], mid[0], mid[-1], tan)
    kinds_b = [_end_kind(bxy, i, counts, on_boundary) for i in (0, -1)]
    for e, (flag_a, flag_b) in enumerate(((sa["at_start"], sb["at_start"]), (sa["at_stop"], sb["at_stop"]))):
        meet = _ends_meet([_end_kind(a.xy, -e, counts, on_boundary), kinds_b[e]], [flag_a, flag_b])
        if meet is None:
            return None
        if meet == "boundary":
            mid[-e] = _on_ring(domain, mid[-e])
    keep_b = b.keep[::-1] if reverse else b.keep
    if _blocked(a.keep, a.xy, sa) or _blocked(keep_b, bxy, sb):
        return None
    midline = shapely.get_coordinates(LineString(mid).simplify(MIDLINE_TOL * lc, preserve_topology=False))
    new_b = _rebuilt(bxy, sb, midline)
    return _rebuilt(a.xy, sa, midline), (new_b[::-1].copy() if reverse else new_b), midline


def _moved(new: np.ndarray, old: np.ndarray, step: float) -> float:
    """Hausdorff distance between a line's new and old course."""
    a, b = (shapely.segmentize(LineString(xy), step) for xy in (new, old))
    return float(shapely.hausdorff_distance(a, b))


def _changed(new: np.ndarray, old: np.ndarray):
    """The parts of ``new`` that are not segments of ``old``."""
    old_segments = {(tuple(p), tuple(q)) for p, q in zip(old[:-1], old[1:])}
    old_segments |= {(q, p) for p, q in old_segments}
    parts = [[p, q] for p, q in zip(new[:-1], new[1:]) if (tuple(p), tuple(q)) not in old_segments]
    return MultiLineString(parts) if parts else None


def _keep(new: np.ndarray, old_xy: np.ndarray, old_keep: np.ndarray, midline: np.ndarray) -> np.ndarray:
    """Must-keep flags of ``new``: those of the vertices it kept, and the midline's ends."""
    flags = {tuple(p): bool(k) for p, k in zip(old_xy, old_keep)}
    flags[tuple(midline[0])] = flags[tuple(midline[-1])] = True
    return np.array([flags.get(tuple(p), False) for p in new])


def _merge_pair(lines: list, k: int, j: int, domain, tol: float, sharp: float, counts: dict,
                on_boundary) -> dict | None:
    """Merge the first mergeable stretch of lines k and j in place; the record, or None."""
    a, b = lines[k], lines[j]
    lc = min(a.lc, b.lc)
    if not math.isfinite(lc):
        return None
    gap = MERGE_GAP * lc
    other = LineString(b.xy)
    s, pts = _samples(a.xy, SAMPLE_STEP * lc)
    d = shapely.distance(other, shapely.points(pts))
    region = domain.buffer(tol)
    rest = [LineString(c.xy) for i, c in enumerate(lines) if i not in (k, j) and len(c.xy) > 1]
    rest = shapely.union_all(rest) if rest else None
    for i0, i1 in _runs(d < gap):
        if s[i1] - s[i0] < MIN_STRETCH * lc or d[i0:i1 + 1].min() <= tol:
            continue
        if substring(LineString(a.xy), s[i0], s[i1]).distance(other) <= tol:
            continue  # the lines touch or cross here: a junction or a crossing (``wedge``)
        plan = _plan(a, b, s[i0:i1 + 1], pts[i0:i1 + 1], lc, sharp, counts, on_boundary, domain)
        if plan is None:
            continue
        new_a, new_b, midline = plan
        moved = max(_moved(new_a, a.xy, SAMPLE_STEP * lc), _moved(new_b, b.xy, SAMPLE_STEP * lc))
        changed = [g for g in (_changed(new_a, a.xy), _changed(new_b, b.xy)) if g is not None]
        if moved > MAX_MOVE * lc or not all(region.covers(g) for g in changed):
            continue
        if rest is not None and any(g.distance(rest) <= tol for g in changed):
            continue
        lines[k] = dataclasses.replace(a, xy=new_a, keep=_keep(new_a, a.xy, a.keep, midline))
        lines[j] = dataclasses.replace(b, xy=new_b, keep=_keep(new_b, b.xy, b.keep, midline))
        centre = LineString(midline).interpolate(0.5, normalized=True)
        return {"location": np.array(centre.coords[0]), "gap": float(d[i0:i1 + 1].max()),
                "length": float(LineString(midline).length), "deviation": moved, "lines": [a.line_id, b.line_id]}
    return None


def merge_parallel(lines: list, domain, tol: float, sharp: float, on_boundary) -> tuple:
    """(lines, records): stretches of two barriers closer than MERGE_GAP x lc merged onto their midline
    (module docstring). ``lines`` are copied first; ``on_boundary(xy)`` masks points on the boundary."""
    lines = [dataclasses.replace(b, xy=b.xy.copy(), keep=b.keep.copy()) for b in lines]
    records, failed = [], set()
    for _ in range(MAX_MERGES):
        record = _next_merge(lines, domain, tol, sharp, on_boundary, failed)
        if record is None:
            break
        records.append(record)
    return lines, records


def _next_merge(lines: list, domain, tol: float, sharp: float, on_boundary, failed: set) -> dict | None:
    """Make the first merge that fits, or None when there is none left."""
    if len(lines) < 2:
        return None
    counts = {}
    for b in lines:
        for p in map(tuple, b.xy):
            counts[p] = counts.get(p, 0) + 1
    geoms = [LineString(b.xy) for b in lines]
    tree = shapely.STRtree(geoms)
    for k, b in enumerate(lines):
        reach = MERGE_GAP * b.lc if math.isfinite(b.lc) else 0.0
        for j in sorted(int(i) for i in tree.query(geoms[k], predicate="dwithin", distance=reach)):
            if j <= k or (k, j) in failed:
                continue
            record = _merge_pair(lines, k, j, domain, tol, sharp, counts, on_boundary)
            if record is not None:
                return record
            failed.add((k, j))
    return None
