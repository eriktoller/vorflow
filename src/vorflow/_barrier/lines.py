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
   An end does not snap onto a barrier it runs alongside (its end segment
   within ``snap_tol`` of it and within SNAP_PARALLEL_ANGLE of parallel), and
   when both ends of a barrier would snap onto the same other barrier that it
   lies within ``snap_tol`` of, neither does: a close parallel barrier is not
   folded onto its neighbour. A barrier that collapses to a point (a branch
   no longer than ``snap_tol`` whose free end snaps back onto the barrier it
   starts on) is dropped as a stub.
2. **Drop boundary parts.** Segments lying on the domain boundary (every
   point within ``snap_tol``) are removed: a barrier there separates
   nothing. A barrier with such a segment inside is split in two, one made
   only of them is dropped. Recorded, with the length removed. Where a part
   along the boundary was cut off, the new end is moved onto the boundary,
   so a barrier that ended on it still does and its compartment stays
   sealed.
3. **Node.** Crossings and junctions with other barriers, the line itself
   and standard lines (and the projections of their ends within the
   tolerance) are inserted as vertices, and every vertex shared by several
   barriers (or twice by one) gets exactly one coordinate. These vertices,
   the ends and points on the domain boundary must be kept by the next two
   steps.
4. **Simplify** (only lines with ``simplify_tolerance > 0``): the fewest
   chords between the line's own vertices that keep every dropped vertex
   within SIMPLIFY_MARGIN x the tolerance. A chord of length L costs
   ``1 + SHORT_PENALTY max(0, 1 - L / lc)^2``, so short chords are avoided.
   Unlike a per-line Douglas-Peucker pass this runs after the junctions are
   joined, so it cannot pull a splay off the fault it starts on. A chord
   spans at most MAX_SPAN input vertices, and the chords reachable from a
   vertex are found block by block (the valid directions narrow as the
   chord grows), so a trace of n vertices costs about n x MAX_SPAN, not n^2.
5. **Clean digitising noise** (always): a vertex closer than NOISE_FRACTION x
   lc to a neighbour is dropped (the one whose removal moves the line least,
   kept in a heap), unless it must be kept or dropping it would fold the
   line back sharper than NOISE_FOLD where it did not fold before (the two
   corners of a U-turn narrower than the noise length: dropping one would
   leave a spike). The largest move is recorded and warned about.

A barrier without its own ``resolution`` uses the finest resolution of the
zones it crosses for lc here (with none, step 5 skips it). Barriers are not
densified: the barrier construction places its own nodes.
"""
from __future__ import annotations

import dataclasses
import heapq
import math

import numpy as np
import shapely
import shapely.ops
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

NOISE_FRACTION = 0.1      # x lc: vertices closer than this to a neighbour are digitising noise
NOISE_FOLD = math.radians(18.0)  # ... unless dropping one folds the line back sharper than this (a narrow U-turn)
SIMPLIFY_MARGIN = 0.98    # simplification uses this x simplify_tolerance
SHORT_PENALTY = 4.0       # simplification cost of a chord of length L: 1 + SHORT_PENALTY max(0, 1 - L/lc)^2
MAX_SPAN = 512            # a simplified chord spans at most this many input vertices
REACH_BLOCK = 32          # chords judged per block when looking back from a vertex (the blocks double)
REACH_EPS = 1e-9          # relative margin below which a chord is judged by the exact distance check
MERGE_FRACTION = 1e-9     # x extent: vertices closer than this are one vertex
SNAP_PARALLEL_ANGLE = math.radians(12.0)  # an end segment this close to parallel with its snap target runs along it


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


def _snap_target(point: np.ndarray, others: list, tol: float) -> tuple:
    """(where an end at ``point`` moves, index into ``others`` of the line it moves onto), or (None, -1).

    A vertex of ``others`` within tol wins, else their nearest point within tol.
    """
    best, best_k, best_d = None, -1, np.inf
    for k, xy in enumerate(others):
        d = np.hypot(*(xy - point).T)
        i = int(np.argmin(d))
        if 0 < d[i] <= tol and d[i] < best_d:
            best, best_k, best_d = xy[i].copy(), k, d[i]
    if best is not None:
        return best, best_k
    for k, xy in enumerate(others):
        if len(xy) < 2:
            continue
        line = LineString(xy)
        d = line.distance(Point(point))
        if 0 < d <= tol and d < best_d:
            best, best_k, best_d = np.array(line.interpolate(line.project(Point(point))).coords[0]), k, d
    return best, best_k


def runs_alongside(xy: np.ndarray, end: int, target: np.ndarray, tol: float,
                   angle: float = SNAP_PARALLEL_ANGLE) -> bool:
    """True when the end segment of ``xy`` at ``end`` runs alongside the line ``target``.

    That is: both its vertices are within ``tol`` of ``target`` and it is
    within ``angle`` of the target segment nearest to its end. Snapping
    such an end would pull a parallel neighbour onto the target instead of
    closing a near miss; the gap between them is a corridor (``corridor``).
    """
    if len(target) < 2 or len(xy) < 2:
        return False
    a, b = (xy[0], xy[1]) if end == 0 else (xy[-1], xy[-2])
    line = LineString(target)
    if line.distance(Point(b)) > tol:
        return False
    segments = shapely.linestrings(np.stack([target[:-1], target[1:]], axis=1))
    k = int(np.argmin(shapely.distance(segments, Point(a))))
    u, w = b - a, target[k + 1] - target[k]
    cos = abs(float(u @ w)) / max(float(np.hypot(*u) * np.hypot(*w)), 1e-300)
    return bool(cos >= np.cos(angle))


def snap_ends(barriers: list, domain, connectivity_tolerance: float, merge_tol: float) -> None:
    """Join barrier ends that nearly meet another barrier or the domain boundary (step 1).

    An end whose last segment runs alongside the barrier it would snap onto
    is left alone (``runs_alongside``), and so is a barrier whose two ends
    would both snap onto the same other barrier while lying within the
    tolerance of it all along: snapping would merge two parallel barriers.
    """
    boundary = domain.boundary
    for k, barrier in enumerate(barriers):
        tol = max(connectivity_tolerance, barrier.tol)
        before_xy, before_events = barrier.xy.copy(), len(barrier.events)
        targets = []
        for end in (0, -1):
            owners = [j for j in range(len(barriers)) if j != k]
            others = [barriers[j].xy for j in owners]
            if len(barrier.xy) > 3:  # a nearly closed loop snaps onto its own other end
                owners.append(k)
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
                target, j = _snap_target(point, others, tol)
                if target is not None and owners[j] != k and runs_alongside(barrier.xy, end, others[j], tol):
                    continue
                if target is not None:
                    barrier.xy[end] = target
                    targets.append(owners[j])
                    barrier.events.append(_event("snap", point, np.hypot(*(target - point)),
                                                 f"end moved onto another barrier (tolerance {tol:.3g})"))
                    continue
            d = boundary.distance(Point(point))
            if 0 < d <= tol:
                barrier.xy[end] = np.array(boundary.interpolate(boundary.project(Point(point))).coords[0])
                barrier.events.append(_event("snap", point, d,
                                             f"end moved onto the domain boundary (tolerance {tol:.3g})"))
        if len(targets) == 2 and targets[0] == targets[1] != k and _within(barrier.xy, barriers[targets[0]].xy, tol):
            barrier.xy, barrier.events = before_xy, barrier.events[:before_events]
        barrier.xy = _dedupe(barrier.xy)


def _within(xy: np.ndarray, target: np.ndarray, tol: float) -> bool:
    """True when every vertex of ``xy`` lies within ``tol`` of the line ``target``."""
    return len(target) > 1 and bool((shapely.distance(LineString(target), shapely.points(xy)) <= tol).all())


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


def self_crossings(xy: np.ndarray) -> dict:
    """{segment index: points on it} where the polyline ``xy`` crosses or touches itself (not at its own
    vertices shared by neighbouring segments)."""
    segments = shapely.linestrings(np.stack([xy[:-1], xy[1:]], axis=1))
    closed = len(xy) > 3 and np.array_equal(xy[0], xy[-1])
    out = {}
    tree = shapely.STRtree(segments)
    left, right = tree.query(segments, predicate="intersects")
    for i, j in zip(left, right):
        if j <= i + 1 or (closed and i == 0 and j == len(segments) - 1):
            continue
        hit = shapely.intersection(segments[i], segments[j])
        pts = shapely.get_coordinates(hit)
        for k in (i, j):
            out.setdefault(int(k), []).extend(pts)
    return out


def node_self(xy: np.ndarray, tol: float) -> np.ndarray:
    """``xy`` with the points where it crosses or touches itself inserted on both passes."""
    if len(xy) < 4 or LineString(xy).is_simple:
        return xy
    extra = self_crossings(xy)
    out = [xy[0]]
    for k in range(len(xy) - 1):
        a, b = xy[k], xy[k + 1]
        ab = b - a
        pts = sorted(extra.get(k, []), key=lambda p: float((p - a) @ ab))
        for p in pts + [b]:
            if np.hypot(*(p - out[-1])) > tol:
                out.append(np.asarray(p, dtype=float))
            elif p is not b:
                out[-1] = np.asarray(p, dtype=float)  # a crossing a roundoff from a vertex takes its place
    return np.array(out)


def node_lines(barriers: list, standard: list, domain, tol: float) -> None:
    """Insert crossings and junctions (also of a line with itself) as vertices, give shared vertices one
    coordinate, set ``keep`` (step 3)."""
    for b in barriers:
        b.xy = node_self(b.xy, tol)
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


def _reach_block(xy: np.ndarray, j: int, idx: np.ndarray, state: dict, tol: float) -> tuple:
    """(fast verdict, needs the exact check) of the chords (i, j) for ``idx`` (descending, below the last block).

    The chord is within ``tol`` of every vertex k between i and j exactly
    when it is within ``tol`` of the ray from j through each of them (the
    direction to i lies in every wedge of half-width asin(tol / r_k) about
    the direction to k, r_k = |k - j|) and no k lies beyond i (r_k > r_i).
    The wedges are intersected incrementally (``state``: the bounds, the
    reference direction, the largest r and whether a borderline vertex was
    passed), so a chord is judged in O(1); borderline cases (within
    REACH_EPS) are left to ``_chord_ok``.
    """
    v = xy[idx] - xy[j]
    r = np.hypot(*v.T)
    wide = r > tol * (1 + REACH_EPS)
    if state["ref"] is None and wide.any():
        state["ref"] = v[int(np.argmax(wide))] / r[int(np.argmax(wide))]
    ref = state["ref"]
    if ref is None:
        theta = np.zeros(len(idx))
    else:
        theta = np.arctan2(ref[0] * v[:, 1] - ref[1] * v[:, 0], v @ ref)
    half = np.arcsin(np.minimum(tol / np.where(wide, r, np.inf), 1.0))
    lo_k = np.where(wide, theta - half, -np.inf)
    hi_k = np.where(wide, theta + half, np.inf)
    border = np.abs(r - tol) <= tol * REACH_EPS
    # exclusive scans: chord (idx[p], j) sees the vertices of earlier blocks and idx[:p]
    lo = np.maximum.accumulate(np.concatenate([[state["lo"]], lo_k]))
    hi = np.minimum.accumulate(np.concatenate([[state["hi"]], hi_k]))
    rmax = np.maximum.accumulate(np.concatenate([[state["rmax"]], r]))
    risky = np.logical_or.accumulate(np.concatenate([[state["border"]], border]))
    state.update(lo=lo[-1], hi=hi[-1], rmax=rmax[-1], border=bool(risky[-1]))
    slack = np.minimum(hi[:-1] - theta, theta - lo[:-1])
    exact = risky[:-1] | (rmax[:-1] > r * (1 - REACH_EPS)) | (np.abs(slack) <= REACH_EPS)
    return slack > 0, exact


def chord_reach(xy: np.ndarray, j: int, floor: int, tol: float) -> int:
    """Smallest i >= ``floor`` with every chord (i', j), i <= i' < j, within ``tol``; scanning down from j - 1
    and stopping at the first chord that is not (``_chord_ok``), in blocks of growing size."""
    state = {"ref": None, "lo": -np.inf, "hi": np.inf, "rmax": 0.0, "border": False}
    top, size = j - 1, REACH_BLOCK
    while top >= floor:
        idx = np.arange(top, max(floor, top - size + 1) - 1, -1)
        fast, exact = _reach_block(xy, j, idx, state, tol)
        for p in np.flatnonzero(exact | ~fast):
            if not (_chord_ok(xy, int(idx[p]), j, tol) if exact[p] else fast[p]):
                return int(idx[p]) + 1
        top, size = int(idx[-1]) - 1, 2 * size
    return floor


def fewest_chords(xy: np.ndarray, keep: np.ndarray, tol: float, lc: float,
                  short_penalty: float = SHORT_PENALTY, max_span: int = MAX_SPAN) -> np.ndarray:
    """Indices of the cheapest chain of chords within ``tol`` (shortest path; must-keep vertices kept).

    Vertex j is reached from every i back to the first chord that is not
    within ``tol``, the last must-keep vertex, or ``max_span`` vertices
    (``chord_reach``), the cheapest i first among equals counting down from
    j - 1. The span bound keeps the cost linear in the vertex count; it only
    matters for a line that stays within the tolerance of a chord over more
    than ``max_span`` vertices, which then keeps one vertex per span.
    """
    n = len(xy)
    best = np.full(n, np.inf)
    prev = np.full(n, -1)
    best[0] = 0.0
    last_keep = np.maximum.accumulate(np.where(keep, np.arange(n), 0))
    for j in range(1, n):
        start = chord_reach(xy, j, max(int(last_keep[j - 1]), j - max_span), tol)
        i = np.arange(j - 1, start - 1, -1)
        length = np.hypot(*(xy[j] - xy[i]).T)
        short = np.maximum(0.0, 1.0 - length / lc)
        cost = best[i] + 1.0 + short_penalty * (short * short)
        k = int(np.argmin(cost))
        best[j], prev[j] = cost[k], i[k]
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


def _noise_candidate(xy: np.ndarray, keep: np.ndarray, prev: np.ndarray, nxt: np.ndarray, v: int,
                     short: float) -> float | None:
    """The move of dropping vertex ``v`` if it may be dropped (inner, not must-keep, next to a segment shorter
    than ``short``), else None."""
    p, n = prev[v], nxt[v]
    if p < 0 or n < 0 or keep[v]:
        return None
    if not (np.hypot(*(xy[v] - xy[p])) < short or np.hypot(*(xy[n] - xy[v])) < short):
        return None
    if folds(xy, prev[p], p, v, n, nxt[n]):
        return None
    return _offset(xy[v], xy[p], xy[n])


def _interior(xy: np.ndarray, u: int, a: int, b: int) -> float:
    """Angle at vertex u between its legs to a and b."""
    da, db = xy[a] - xy[u], xy[b] - xy[u]
    return math.acos(float(np.clip(da @ db / (np.hypot(*da) * np.hypot(*db)), -1.0, 1.0)))


def folds(xy: np.ndarray, before: int, p: int, v: int, n: int, after: int) -> bool:
    """True when dropping vertex v (between p and n; ``before`` and ``after`` their other neighbours, -1 for
    none), which is not a fold itself, would fold the line back at p or n sharper than NOISE_FOLD: a U-turn
    narrower than the noise length would become a spike."""
    if _interior(xy, v, p, n) < NOISE_FOLD:
        return False
    for u, other, beyond in ((p, n, before), (n, p, after)):
        if beyond >= 0 and _interior(xy, u, beyond, other) < NOISE_FOLD <= _interior(xy, u, beyond, v):
            return True
    return False


def drop_noise(xy: np.ndarray, keep: np.ndarray, short: float) -> tuple:
    """(kept mask, vertices dropped, largest move, where): vertices next to a segment shorter than ``short``
    dropped one at a time, least move first (ties: first along the line), until none is left or two remain.

    A heap of the candidates with lazy invalidation: dropping a vertex only
    changes the two vertices on either side of it (``folds`` looks two ahead).
    """
    n = len(xy)
    prev, nxt = np.arange(-1, n - 1), np.arange(1, n + 1)
    nxt[-1] = -1
    version = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    heap = []
    for v in range(n):
        move = _noise_candidate(xy, keep, prev, nxt, v, short)
        if move is not None:
            heap.append((move, v, 0))
    heapq.heapify(heap)
    count, dropped, largest, where = n, 0, 0.0, None
    while count > 2 and heap:
        move, v, ver = heapq.heappop(heap)
        if not alive[v] or ver != version[v]:
            continue
        if move >= largest:
            largest, where = move, xy[v].copy()
        alive[v] = False
        p, q = prev[v], nxt[v]
        nxt[p], prev[q] = q, p
        count -= 1
        dropped += 1
        for u in (prev[p] if p >= 0 else -1, p, q, nxt[q] if q >= 0 else -1):
            if u < 0:
                continue
            version[u] += 1
            move = _noise_candidate(xy, keep, prev, nxt, u, short)
            if move is not None:
                heapq.heappush(heap, (move, u, int(version[u])))
    return alive, dropped, largest, where


def clean_noise(barriers: list, fraction: float = NOISE_FRACTION) -> None:
    """Drop vertices closer than ``fraction`` x lc to a neighbour, least move first (step 5)."""
    for b in barriers:
        if not np.isfinite(b.lc):
            continue
        alive, dropped, largest, where = drop_noise(b.xy, b.keep, fraction * b.lc)
        b.xy, b.keep = b.xy[alive], b.keep[alive]
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
