"""Barrier line cleaning before meshing: end snapping, noding, simplification, noise.

``ConceptualMesh.generate()`` runs ``clean_barrier_lines`` on the clipped
barrier lines, in a local frame (the lower-left corner of the domain is the
origin, so UTM-sized coordinates behave like small ones). Each step works on
the output of the previous one:

1. **Snap ends.** An end within ``snap_tol = max(connectivity_tolerance,
   simplify_tolerance)`` of another barrier moves onto it (onto its vertex
   when one is within the tolerance); an end that overshoots another barrier
   by at most ``snap_tol`` is trimmed back to the crossing; an end within
   ``snap_tol`` of the domain boundary moves onto it. Every snap is recorded.
   A barrier that collapses to a point (a branch no longer than ``snap_tol``
   whose free end snaps back onto the barrier it starts on) is dropped as a
   stub.
2. **Drop boundary parts.** Segments lying on the domain boundary (every
   point within ``snap_tol``) are removed: a barrier there separates
   nothing. A barrier with such a segment inside is split in two, one made
   only of them is dropped. Recorded, with the length removed.
3. **Node.** Crossings and junctions with other barriers and standard lines
   (and the projections of their ends within the tolerance) are inserted as
   vertices, and every vertex shared by several barriers gets exactly one
   coordinate. These vertices, the ends and points on the domain boundary
   must be kept by the next two steps.
4. **Simplify** (only lines with ``simplify_tolerance > 0``): the fewest
   chords between the line's own vertices that keep every dropped vertex
   within SIMPLIFY_MARGIN x the tolerance. A chord of length L costs
   ``1 + SHORT_PENALTY max(0, 1 - L / lc)^2``, so short chords are avoided.
   Unlike a per-line Douglas-Peucker pass this runs after the junctions are
   joined, so it cannot pull a splay off the fault it starts on.
5. **Clean digitising noise** (always): a vertex closer than NOISE_FRACTION x
   lc to a neighbour is dropped (the one whose removal moves the line least),
   unless it must be kept. The largest move is recorded and warned about.

Barriers are not densified: the barrier construction places its own nodes.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import shapely
import shapely.ops
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

NOISE_FRACTION = 0.1      # x lc: vertices closer than this to a neighbour are digitising noise
SIMPLIFY_MARGIN = 0.98    # simplification uses this x simplify_tolerance
SHORT_PENALTY = 4.0       # simplification cost of a chord of length L: 1 + SHORT_PENALTY max(0, 1 - L/lc)^2
MERGE_FRACTION = 1e-9     # x extent: vertices closer than this are one vertex


@dataclasses.dataclass
class BarrierLine:
    """One barrier polyline in local coordinates, with what cleaning did to it."""
    line_id: object
    xy: np.ndarray
    lc: float
    tol: float
    keep: np.ndarray | None = None  # must-keep vertices (ends, junctions, crossings, boundary points)
    events: list = dataclasses.field(default_factory=list)
    source: int = -1                # the caller's index of the line this one came from


def _dedupe(xy: np.ndarray) -> np.ndarray:
    """``xy`` without consecutive repeated points."""
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.hypot(*np.diff(xy, axis=0).T) > 0
    return xy[keep]


def _event(kind: str, xy: np.ndarray, distance: float, note: str) -> dict:
    """A cleaning record (local coordinates; the caller shifts ``location`` to global)."""
    return {"kind": kind, "location": (float(xy[0]), float(xy[1])), "distance": float(distance), "note": note}


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------
def _trim_overshoot(barrier: BarrierLine, end: int, others, tol: float, start_tol: float) -> float | None:
    """Trim ``end`` (0 or -1) back to its last crossing with ``others`` if it overshoots by at most ``tol``.

    A meeting at the other end (within ``start_tol`` of it: the junction a
    branch starts on) is not a crossing. Returns the trimmed length, or None
    when nothing was trimmed.
    """
    line = LineString(barrier.xy if end == -1 else barrier.xy[::-1])
    hits = line.intersection(others)
    pts = [g for g in getattr(hits, "geoms", [hits]) if not g.is_empty and g.geom_type == "Point"]
    along = [d for d in (line.project(p) for p in pts) if d > start_tol]
    if not along:
        return None
    along = max(along)
    overshoot = line.length - along
    if overshoot <= 0 or overshoot > tol:
        return None
    kept = shapely.ops.substring(line, 0.0, along)
    xy = np.asarray(kept.coords)[:, :2]
    barrier.xy = xy if end == -1 else xy[::-1]
    return overshoot


def _snap_target(point: np.ndarray, others: list, tol: float) -> np.ndarray | None:
    """Where an end at ``point`` moves: a vertex of ``others`` within tol, else their nearest point within tol."""
    best, best_d = None, np.inf
    for xy in others:
        d = np.hypot(*(xy - point).T)
        k = int(np.argmin(d))
        if 0 < d[k] <= tol and d[k] < best_d:
            best, best_d = xy[k], d[k]
    if best is not None:
        return best.copy()
    for xy in others:
        if len(xy) < 2:
            continue
        line = LineString(xy)
        d = line.distance(Point(point))
        if 0 < d <= tol and d < best_d:
            best, best_d = np.array(line.interpolate(line.project(Point(point))).coords[0]), d
    return best


def snap_ends(barriers: list, domain, connectivity_tolerance: float, merge_tol: float) -> None:
    """Join barrier ends that nearly meet another barrier or the domain boundary (step 1)."""
    boundary = domain.boundary
    for k, barrier in enumerate(barriers):
        tol = max(connectivity_tolerance, barrier.tol)
        for end in (0, -1):
            others = [b.xy for j, b in enumerate(barriers) if j != k]
            if len(barrier.xy) > 3:  # a nearly closed loop snaps onto its own other end
                others.append(barrier.xy[[-1 if end == 0 else 0]])
            point = barrier.xy[end].copy()
            if others:
                union = shapely.union_all([LineString(o) if len(o) > 1 else Point(o[0]) for o in others])
                if union.distance(Point(point)) == 0:
                    continue
                trimmed = _trim_overshoot(barrier, end, union, tol, merge_tol)
                if trimmed is not None:
                    barrier.events.append(_event("trim", point, trimmed,
                                                 f"end trimmed back to the barrier it overshoots (tolerance {tol:.3g})"))
                    continue
                target = _snap_target(point, others, tol)
                if target is not None:
                    barrier.xy[end] = target
                    barrier.events.append(_event("snap", point, np.hypot(*(target - point)),
                                                 f"end moved onto another barrier (tolerance {tol:.3g})"))
                    continue
            d = boundary.distance(Point(point))
            if 0 < d <= tol:
                barrier.xy[end] = np.array(boundary.interpolate(boundary.project(Point(point))).coords[0])
                barrier.events.append(_event("snap", point, d,
                                             f"end moved onto the domain boundary (tolerance {tol:.3g})"))
        barrier.xy = _dedupe(barrier.xy)


def path_length(xy: np.ndarray) -> float:
    """Length of the polyline ``xy``."""
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def drop_collapsed(barriers: list, lengths: list, merge_tol: float) -> tuple:
    """(kept, dropped): barriers that snapping left shorter than ``merge_tol`` are dropped as stubs (step 1).

    ``lengths`` holds each barrier's length before snapping.
    """
    kept, dropped = [], []
    for b, length in zip(barriers, lengths):
        if len(b.xy) >= 2 and path_length(b.xy) > merge_tol:
            kept.append(b)
            continue
        b.events.append(_event("stub", b.xy[0], length, f"dropped: the whole {length:.3g} long line snapped onto "
                                                         "one point (shorter than its snap tolerance)"))
        dropped.append(b)
    return kept, dropped


def _boundary_segments(xy: np.ndarray, boundary, tol: float, fractions=(0.0, 0.25, 0.5, 0.75, 1.0)) -> np.ndarray:
    """Mask of the segments of ``xy`` lying on ``boundary``: every point at ``fractions`` of it within ``tol``."""
    t = np.asarray(fractions)
    a, b = xy[:-1], xy[1:]
    pts = a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]
    d = shapely.distance(boundary, shapely.points(pts.reshape(-1, 2))).reshape(len(a), len(t))
    return (d <= tol).all(axis=1)


def _runs(mask: np.ndarray) -> list:
    """(start, stop) of every run of True in ``mask``, stop exclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def drop_boundary_parts(barriers: list, domain, connectivity_tolerance: float) -> tuple:
    """(kept, dropped): barriers without their segments on the domain boundary (step 2).

    A barrier keeps one part per run of segments off the boundary (its
    events go with the first); one with no such segment is dropped.
    """
    kept, dropped = [], []
    for b in barriers:
        tol = max(connectivity_tolerance, b.tol)
        on = _boundary_segments(b.xy, domain.boundary, tol)
        if not on.any():
            kept.append(b)
            continue
        seg = np.hypot(*np.diff(b.xy, axis=0).T)
        k = int(np.argmax(np.where(on, seg, -1.0)))
        b.events.append(_event("boundary", 0.5 * (b.xy[k] + b.xy[k + 1]), float(seg[on].sum()),
                               f"dropped {float(seg[on].sum()):.3g} of it lying on the domain boundary "
                               f"(tolerance {tol:.3g}): a barrier there separates nothing"))
        parts = _runs(~on)
        if not parts:
            dropped.append(b)
        for n, (start, stop) in enumerate(parts):
            part = dataclasses.replace(b, xy=b.xy[start:stop + 1].copy(), events=b.events if n == 0 else [])
            _end_on_boundary(part, start > 0, stop < len(on), domain.boundary, tol)
            kept.append(part)
    return kept, dropped


def _end_on_boundary(part: BarrierLine, at_start: bool, at_stop: bool, boundary, tol: float) -> None:
    """Move the ends of ``part`` where a boundary run was cut off onto the boundary, so it stays sealed there.

    Those ends are within ``tol`` of the boundary (they ended a segment
    lying on it); left where they are, they would open a gap that short.
    """
    for end, cut in ((0, at_start), (-1, at_stop)):
        point = Point(part.xy[end])
        d = boundary.distance(point)
        if not cut or d == 0:
            continue
        part.xy[end] = np.array(boundary.interpolate(boundary.project(point)).coords[0])
        part.events.append(_event("snap", np.asarray(point.coords[0]), d,
                                  f"end moved onto the domain boundary where its part along it was dropped "
                                  f"(tolerance {tol:.3g})"))
    part.xy = _dedupe(part.xy)


# ---------------------------------------------------------------------------
# Noding
# ---------------------------------------------------------------------------
def _meeting_points(line: LineString, others: list, tol: float) -> list:
    """Points where ``line`` meets ``others``: crossings, overlap ends, and their ends within ``tol`` of it."""
    pts = []
    for other in others:
        hits = line.intersection(other)
        for g in getattr(hits, "geoms", [hits]):
            if g.is_empty:
                continue
            if g.geom_type == "Point":
                pts.append(np.array(g.coords[0]))
            elif g.geom_type == "LineString":
                pts += [np.array(g.coords[0]), np.array(g.coords[-1])]
        for end in (Point(other.coords[0]), Point(other.coords[-1])):
            if line.distance(end) <= tol:
                pts.append(np.array(line.interpolate(line.project(end)).coords[0]))
    return pts


def _insert(xy: np.ndarray, points: list, tol: float) -> tuple:
    """(vertices, inserted mask): ``points`` (on the line) inserted in order along it."""
    line = LineString(xy)
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    extra = np.array([line.project(Point(p)) for p in points]) if points else np.empty(0)
    all_d = np.concatenate([along, extra])
    all_xy = np.vstack([xy] + [np.asarray(points).reshape(-1, 2)])
    inserted = np.concatenate([np.zeros(len(xy), dtype=bool), np.ones(len(extra), dtype=bool)])
    order = np.lexsort((~inserted, all_d))
    out_xy, out_ins = [all_xy[order[0]]], [inserted[order[0]]]
    for i in order[1:]:
        if np.hypot(*(all_xy[i] - out_xy[-1])) <= tol:
            if inserted[i]:  # an inserted junction point replaces a vertex a roundoff away
                out_xy[-1], out_ins[-1] = all_xy[i], True
            continue
        out_xy.append(all_xy[i])
        out_ins.append(inserted[i])
    return np.array(out_xy), np.array(out_ins)


def _unify_vertices(barriers: list, tol: float) -> None:
    """Vertices of different barriers within ``tol`` take the coordinate of the first one."""
    xy = np.vstack([b.xy for b in barriers])
    rep = np.arange(len(xy))
    for i, j in sorted(cKDTree(xy).query_pairs(tol)):
        rep[j] = min(rep[j], rep[i])
    rep = rep[rep]
    start = 0
    for b in barriers:
        b.xy = xy[rep[start:start + len(b.xy)]].copy()
        start += len(b.xy)


def node_lines(barriers: list, standard: list, domain, tol: float) -> None:
    """Insert crossings and junctions as vertices, give shared vertices one coordinate, set ``keep`` (step 2)."""
    lines = [LineString(b.xy) for b in barriers]
    inserted = []
    for k, barrier in enumerate(barriers):
        others = [ln for j, ln in enumerate(lines) if j != k] + standard
        if len(barrier.xy) > 3 and np.hypot(*(barrier.xy[0] - barrier.xy[-1])) <= tol:
            barrier.xy[-1] = barrier.xy[0]
        xy, ins = _insert(barrier.xy, _meeting_points(lines[k], others, tol), tol)
        barrier.xy = xy
        inserted.append(ins)
    _unify_vertices(barriers, tol)
    counts = {}
    for b in barriers:
        for p in map(tuple, b.xy):
            counts[p] = counts.get(p, 0) + 1
    for barrier, ins in zip(barriers, inserted):
        barrier.keep = must_keep(barrier.xy, counts, domain, tol) | ins


def must_keep(xy: np.ndarray, counts: dict, domain, tol: float) -> np.ndarray:
    """Vertices no later step may drop: ends, vertices shared with other barriers, points on the boundary."""
    keep = np.zeros(len(xy), dtype=bool)
    keep[[0, -1]] = True
    keep |= np.array([counts[tuple(p)] > 1 for p in xy])
    keep |= shapely.distance(domain.boundary, shapely.points(xy)) <= tol
    if np.array_equal(xy[0], xy[-1]):  # closed ring: its closing vertex is a plain vertex
        keep[[0, -1]] = counts[tuple(xy[0])] > 2
    return keep


# ---------------------------------------------------------------------------
# Simplification and noise
# ---------------------------------------------------------------------------
def _chord_ok(xy: np.ndarray, i: int, j: int, tol: float) -> bool:
    """Every vertex strictly between i and j lies within ``tol`` of the chord (i, j)."""
    if j == i + 1:
        return True
    a, ab = xy[i], xy[j] - xy[i]
    pts = xy[i + 1:j] - a
    t = np.clip(pts @ ab / (ab @ ab), 0.0, 1.0)
    return bool(np.hypot(*(pts - np.outer(t, ab)).T).max() <= tol)


def fewest_chords(xy: np.ndarray, keep: np.ndarray, tol: float, lc: float,
                  short_penalty: float = SHORT_PENALTY) -> np.ndarray:
    """Indices of the cheapest chain of chords within ``tol`` (shortest path; must-keep vertices kept)."""
    n = len(xy)
    best = np.full(n, np.inf)
    prev = np.full(n, -1)
    best[0] = 0.0
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            if not _chord_ok(xy, i, j, tol):
                break
            length = np.hypot(*(xy[j] - xy[i]))
            cost = best[i] + 1.0 + short_penalty * max(0.0, 1.0 - length / lc) ** 2
            if cost < best[j]:
                best[j], prev[j] = cost, i
            if keep[i]:
                break
    chain = [n - 1]
    while chain[-1] != 0:
        chain.append(prev[chain[-1]])
    return np.array(chain[::-1])


def simplify_lines(barriers: list, margin: float = SIMPLIFY_MARGIN) -> None:
    """Simplify every barrier with simplify_tolerance > 0 by ``fewest_chords`` (step 3)."""
    for b in barriers:
        if b.tol <= 0 or len(b.xy) < 3:
            continue
        idx = fewest_chords(b.xy, b.keep, margin * b.tol, b.lc)
        if len(idx) < len(b.xy):
            b.events.append(_event("simplify", b.xy[0], 0.0,
                                   f"simplified from {len(b.xy)} to {len(idx)} vertices within {b.tol:g}"))
        b.xy, b.keep = b.xy[idx], b.keep[idx]


def _offset(p: np.ndarray, a: np.ndarray, c: np.ndarray) -> float:
    """Distance from p to the segment a-c."""
    ac = c - a
    t = float(np.clip((p - a) @ ac / max(ac @ ac, 1e-300), 0.0, 1.0))
    return float(np.hypot(*(p - a - t * ac)))


def clean_noise(barriers: list, fraction: float = NOISE_FRACTION) -> None:
    """Drop vertices closer than ``fraction`` x lc to a neighbour, least move first (step 4)."""
    for b in barriers:
        dropped, largest, where = 0, 0.0, None
        while len(b.xy) > 2 and np.isfinite(b.lc):
            seg = np.hypot(*np.diff(b.xy, axis=0).T)
            options = []
            for s in np.flatnonzero(seg < fraction * b.lc):
                for v in (s, s + 1):
                    if 0 < v < len(b.xy) - 1 and not b.keep[v]:
                        options.append((_offset(b.xy[v], b.xy[v - 1], b.xy[v + 1]), v))
            if not options:
                break
            move, v = min(options)
            if move >= largest:
                largest, where = move, b.xy[v].copy()
            b.xy, b.keep = np.delete(b.xy, v, axis=0), np.delete(b.keep, v)
            dropped += 1
        if dropped:
            b.events.append(_event("noise", where, largest,
                                   f"dropped {dropped} vertices closer than {fraction:g} x lc to a neighbour "
                                   "(digitising noise)"))


def clean_barrier_lines(barriers: list, standard: list, domain, connectivity_tolerance: float) -> tuple:
    """(kept, dropped): steps 1-5 of the module docstring on ``barriers`` (local coordinates).

    ``kept`` are the cleaned lines (a line split in step 2 gives several,
    with the same ``source``); ``dropped`` the lines removed whole, with the
    events saying why.
    """
    if not barriers:
        return [], []
    extent = float(max(np.ptp(np.asarray(domain.bounds).reshape(2, 2), axis=0)))
    merge_tol = max(MERGE_FRACTION * extent, 1e-12)
    for b in barriers:
        b.xy = _dedupe(np.asarray(b.xy, dtype=float)[:, :2])
    lengths = [path_length(b.xy) for b in barriers]
    snap_ends(barriers, domain, connectivity_tolerance, merge_tol)
    barriers, collapsed = drop_collapsed(barriers, lengths, merge_tol)
    barriers, on_boundary = drop_boundary_parts(barriers, domain, connectivity_tolerance)
    if barriers:
        node_lines(barriers, standard, domain, merge_tol)
        simplify_lines(barriers)
        clean_noise(barriers)
    return barriers, collapsed + on_boundary
