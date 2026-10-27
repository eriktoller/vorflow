"""Acute wedges: barrier legs meeting at less than ``sharp_angle`` are merged near where they meet.

Two barrier legs that leave a junction (a splay, a T-branch, a crossing, a
fan) or the domain boundary at a few degrees bound a sliver that the
empty-circle construction can only fill with ever smaller faces. Instead,
the legs are merged over a short reach, so the sliver disappears and the
legs leave the merged part at a workable angle. The move of every barrier
is bounded: no point of a re-drawn leg lies more than DEVIATION x lc from
the leg as drawn (lc the finest resolution of the lines involved; checked
with the Hausdorff distance; the stem shortens until it holds).

*Junctions.* At a hard node J with three or more barrier legs, the
narrowest sector below ``sharp_angle`` is merged first. Its two legs (or
two groups of legs that already share a piece) get a stem from J to
M = J + m b, b the sector's bisector and m = DEVIATION x lc / sin(Theta / 2)
for an opening Theta, so M is DEVIATION x lc from both legs. From M each leg
runs straight to its point P at arc length L along its path, with the
largest L for which the two still leave M at least OPEN_FACTOR x
``sharp_angle`` apart (and never past the next vertex the leg shares with
another line, a standard-line crossing or its end). The stem carries both
legs' lines, so their faces there are one row (``barrier_lines`` lists the
stem under each line). Then the node is looked at again: a fan or a star
of narrow legs becomes a short cascade of stems, and a crossing at a small
angle gets a stem on both sides, so the two lines share a stretch through
the crossing. A sector narrower than FAN_ANGLE next to another one (a leg
with narrow sectors on both sides, as in a fan or a 16-leg star), or at a
node merged already, is merged the same way, with FAN_ANGLE in place of
``sharp_angle``: the sector plan of ``network`` cannot fit sites into two
adjacent sectors that narrow, or into a crowded node, while one alone is
fine. The move is checked against the lines as they came in, so
it stays within DEVIATION x lc however many merges touch a line.

*Boundary exits.* A barrier end on the domain boundary (or a hole edge)
whose leg makes an angle theta below ``sharp_angle`` with the boundary
moves along the boundary to Q, q = DEVIATION x lc / sin(theta) away, and
runs straight from Q to its point at arc length L, again the largest L
with an angle of at least OPEN_FACTOR x ``sharp_angle`` to the boundary at
Q. The stretch from J to Q lies on the boundary, where a barrier separates
nothing; the barrier still ends on the boundary, so it stays sealed.

A spike (one line folding back on itself, or two lines ending together:
a node with two legs) keeps the uncovered-tip rule of ``network``; a pair
whose legs are the two sides of a fold at a junction is left as it is.
Every merge is recorded with its place, angle, reach, largest move and
lines.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point

from .lines import BarrierLine

DEVIATION = 0.1        # x lc: the largest move of a barrier by a merge
OPEN_FACTOR = 1.5      # x sharp_angle: the angle at which two merged legs leave the stem (or the boundary)
FAN_ANGLE = math.radians(29.0)  # two adjacent sectors narrower than this are merged too (fans, stars)
MIN_LEG = 0.2          # x lc: a re-drawn leg shorter than this is not made; the reach shrinks instead
SHRINK = 0.7           # the stem is shortened by this factor when a merge does not fit
TRIES = 8              # ... at most this many times
REACH_SAMPLES = 64
BISECT_STEPS = 40
MAX_MERGES = 10000


@dataclasses.dataclass
class Leg:
    """A barrier leg leaving a node: line ``k`` from its vertex ``i`` in direction ``step`` (+1 or -1)."""
    k: int
    i: int
    step: int
    angle: float


def _occurrences(lines: list) -> dict:
    """{vertex coordinate: [(line, vertex index), ...]} over all lines."""
    out = {}
    for k, b in enumerate(lines):
        for i, p in enumerate(map(tuple, b.xy)):
            out.setdefault(p, []).append((k, i))
    return out


def _legs(lines: list, places: list) -> list:
    """The legs leaving a node from its ``places`` (line, vertex index), sorted by angle; legs of several lines
    along one shared piece (the same next vertex) count once, as a list of their members."""
    groups = {}
    for k, i in places:
        xy = lines[k].xy
        for step in (1, -1):
            if 0 <= i + step < len(xy):
                d = xy[i + step] - xy[i]
                groups.setdefault(tuple(xy[i + step]), []).append(
                    Leg(k, i, step, math.atan2(d[1], d[0]) % (2 * math.pi)))
    return sorted(groups.values(), key=lambda group: group[0].angle)


def _pairs(groups: list, sharp: float, crowded: bool, fan: float = FAN_ANGLE) -> list:
    """(pair of consecutive leg groups, threshold) for every sector to merge, narrowest first: those narrower
    than ``sharp``, and those narrower than ``fan`` next to another one or at a ``crowded`` node (one merged
    already; module docstring)."""
    n = len(groups)
    if n < 2:
        return []
    gaps = [(groups[(t + 1) % n][0].angle - groups[t][0].angle) % (2 * math.pi) for t in range(n)]
    out = []
    for t, g in enumerate(gaps):
        if g < sharp:
            out.append((g, t, sharp))
        elif g < fan and n > 2 and (crowded or min(gaps[t - 1], gaps[(t + 1) % n]) < fan):
            out.append((g, t, fan))
    return [((groups[t], groups[(t + 1) % n]), limit) for _, t, limit in sorted(out)]


def _path(xy: np.ndarray, i: int, step: int, stop: np.ndarray) -> np.ndarray:
    """Vertices of ``xy`` from i in direction ``step`` up to (and including) the first ``stop`` vertex."""
    idx = [i]
    j = i + step
    while 0 <= j < len(xy):
        idx.append(j)
        if stop[j]:
            break
        j += step
    return xy[idx]


def _along(path: np.ndarray, length: float) -> tuple:
    """(point at arc length ``length`` along ``path``, number of path vertices strictly before it)."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if length >= cum[-1]:
        return path[-1].copy(), len(path) - 1
    j = int(np.searchsorted(cum, length, side="right")) - 1
    f = (length - cum[j]) / seg[j]
    return path[j] + f * (path[j + 1] - path[j]), j + 1


def _length(path: np.ndarray) -> float:
    """Arc length of ``path``."""
    return float(np.hypot(*np.diff(path, axis=0).T).sum())


def _min_opening(apex: np.ndarray, points: list, outer: tuple | None = None) -> float:
    """Smallest angle at ``apex`` between consecutive directions to ``points`` (in the given order), and to the
    ``outer`` reference directions when given (boundary exits)."""
    dirs = [math.atan2(*(p - apex)[::-1]) for p in points]
    if outer is not None:
        dirs = [outer[0]] + dirs
    out = math.inf
    for a, b in zip(dirs[:-1], dirs[1:]):
        out = min(out, abs((b - a + math.pi) % (2 * math.pi) - math.pi))
    return out


def _bisect_reach(opening, lo: float, hi: float, target: float) -> float | None:
    """The largest L in (lo, hi] with opening(L) >= target, or None.

    The opening is not monotonic in L for a fan (legs on one side of the
    stem start out parallel), so L is scanned on REACH_SAMPLES log-spaced
    values first, then refined by bisection past the last one that opens.
    """
    grid = lo * (hi / lo) ** np.linspace(0.0, 1.0, REACH_SAMPLES + 1)[1:]
    ok = [opening(r) >= target for r in grid]
    if not any(ok):
        return None
    last = max(t for t, good in enumerate(ok) if good)
    if last == len(grid) - 1:
        return float(hi)
    a, b = grid[last], grid[last + 1]
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (a + b)
        a, b = (mid, b) if opening(mid) >= target else (a, mid)
    return float(a)


def _stops(lines: list, counts: dict) -> list:
    """Per line, the vertices a re-drawn leg must not pass: shared ones, must-keep ones and the ends."""
    out = []
    for b in lines:
        stop = np.array([counts[tuple(p)] > 1 for p in b.xy]) | b.keep
        stop[[0, -1]] = True
        out.append(stop)
    return out


def _replace(b: BarrierLine, i: int, step: int, count: int, new: list) -> None:
    """Replace the ``count`` vertices after vertex i (towards ``step``) of ``b`` by ``new`` (in that direction).

    The first new vertex is must-keep (a stem end or a new end); a last new
    vertex that coincides with the next kept vertex merges into it.
    """
    xy, keep = (b.xy, b.keep) if step == 1 else (b.xy[::-1], b.keep[::-1])
    i = i if step == 1 else len(xy) - 1 - i
    new_xy = np.asarray(new, dtype=float).reshape(-1, 2)
    new_keep = np.zeros(len(new_xy), dtype=bool)
    new_keep[0] = True
    tail = i + 1 + count
    if tail < len(xy) and np.array_equal(new_xy[-1], xy[tail]):
        new_xy, new_keep = new_xy[:-1], new_keep[:-1]
    out_xy = np.vstack([xy[:i + 1], new_xy, xy[tail:]])
    out_keep = np.concatenate([keep[:i + 1], new_keep, keep[tail:]])
    b.xy, b.keep = (out_xy, out_keep) if step == 1 else (out_xy[::-1].copy(), out_keep[::-1].copy())


def _span(leg: Leg, count: int) -> set:
    """Vertex indices of line ``leg.k`` that a re-drawn leg replaces (with its node)."""
    return {leg.i + leg.step * t for t in range(count + 1)}


def _clear_of_others(lines: list, involved: set, new_geom, ends: list, tol: float) -> bool:
    """True when ``new_geom`` meets the lines not involved only at ``ends``."""
    others = [LineString(b.xy) for k, b in enumerate(lines) if k not in involved and len(b.xy) > 1]
    if not others:
        return True
    hits = shapely.get_coordinates(new_geom.intersection(shapely.union_all(others)))
    if len(hits) == 0:
        return True
    ends = np.asarray(ends, dtype=float).reshape(-1, 2)
    return bool((np.hypot(*(hits[:, None, :] - ends[None]).transpose(2, 0, 1)).min(axis=1) <= tol).all())


def _deviation(new_xy: np.ndarray, old_xy: np.ndarray, centre: np.ndarray, radius: float, step: float) -> float:
    """Hausdorff distance between a re-drawn line and its original, both changed only within ``radius`` of
    ``centre``: the largest distance from a point of either (within the radius, every ``step``) to the other."""
    window = Point(centre).buffer(radius)
    new, old = LineString(new_xy), LineString(old_xy)
    out = 0.0
    for part, other in ((new, old), (old, new)):
        pts = shapely.get_coordinates(shapely.segmentize(part.intersection(window), step))
        if len(pts):
            out = max(out, float(shapely.distance(other, shapely.points(pts)).max()))
    return out


def _tentative(lines: list, changes: list) -> dict:
    """{line: new BarrierLine} after applying ``changes`` [(line, i, step, count, new vertices)] to copies, in an
    order that keeps the later indices valid (from the line's far end back)."""
    out = {}
    for k, i, step, count, new in sorted(changes, key=lambda c: (c[0], -c[1])):
        b = out.get(k) or dataclasses.replace(lines[k], xy=lines[k].xy.copy(), keep=lines[k].keep.copy())
        _replace(b, i, step, count, new)
        out[k] = b
    return out


def _within_bound(trial: dict, originals: list, centre: np.ndarray, radius: float, bound: float) -> float | None:
    """The largest move of the lines in ``trial`` from their originals, or None when one exceeds ``bound``."""
    worst = 0.0
    for k, b in trial.items():
        moved = _deviation(b.xy, originals[k].xy, centre, radius, bound / 4)
        if moved > bound * (1 + 1e-6):
            return None
        worst = max(worst, moved)
    return worst


def _merge_pair(lines: list, originals: list, apex: np.ndarray, pair: tuple, stops: list, limit: float,
                tol: float) -> dict | None:
    """Merge one narrow pair of leg groups in place (module docstring); the record, or None when it does not fit."""
    members = [leg for group in pair for leg in group]
    lc = min(lines[leg.k].lc for leg in members)
    if not math.isfinite(lc):
        return None
    first, second = pair[0][0], pair[1][0]
    span = (second.angle - first.angle) % (2 * math.pi)
    mid = first.angle + span / 2
    bis = np.array([math.cos(mid), math.sin(mid)])
    paths = [_path(lines[g[0].k].xy, g[0].i, g[0].step, stops[g[0].k]) for g in pair]
    caps = [_length(p) for p in paths]
    taken = [(leg.k, i) for g, p in zip(pair, paths) for leg in g for i in _span(leg, len(p) - 1)]
    if len(taken) != len(set(taken)):
        return None  # two legs of one line over the same vertices: a fold or a loop, not a wedge
    bound = DEVIATION * lc
    m = bound / math.sin(span / 2)
    m = min(m, min(caps) * math.cos(span / 2) / (1 + MIN_LEG))
    for _ in range(TRIES):
        stem = apex + m * bis

        def ends_at(reach, paths=paths, caps=caps):
            return [_along(p, min(reach, c))[0] for p, c in zip(paths, caps)]

        reach = _bisect_reach(lambda r: _min_opening(stem, ends_at(r)), m, max(caps), OPEN_FACTOR * limit)
        if reach is not None:
            cuts = [_along(p, min(reach, c)) for p, c in zip(paths, caps)]
            pts = [q for q, _ in cuts]
            radius = max(np.hypot(*(q - apex)) for q in pts) + 2 * bound
            trial = _tentative(lines, [(leg.k, leg.i, leg.step, n - 1, [stem, q])
                                       for g, (q, n) in zip(pair, cuts) for leg in g])
            new_geom = MultiLineString([[apex, stem]] + [[stem, q] for q in pts])
            moved = _within_bound(trial, originals, apex, radius, bound)
            if moved is not None and min(np.hypot(*(q - stem)) for q in pts) >= MIN_LEG * lc \
                    and _clear_of_others(lines, set(trial), new_geom, [apex] + pts, tol):
                for k, b in trial.items():
                    lines[k] = b
                return {"kind": "junction", "location": apex.copy(), "angle": math.degrees(span),
                        "reach": float(radius - 2 * bound), "deviation": moved,
                        "lines": sorted({lines[leg.k].line_id for leg in members}, key=str)}
        m *= SHRINK
    return None


def _boundary_ray(domain, xy: np.ndarray, leg_angle: float, tol: float) -> tuple | None:
    """(ring, along-ring position of xy, direction sign, angle to the leg) of the boundary ray at ``xy`` that
    makes the narrowest angle with the leg, inside the domain."""
    rings = [LineString(r.coords) for p in getattr(domain, "geoms", [domain]) for r in [p.exterior, *p.interiors]]
    ring = min(rings, key=lambda r: r.distance(Point(xy)))
    d = ring.project(Point(xy))
    step = max(tol * 1e3, 1e-6 * ring.length)
    best = None
    for sign in (-1, 1):
        p = np.array(ring.interpolate((d + sign * step) % ring.length).coords[0])
        a = math.atan2(p[1] - xy[1], p[0] - xy[0])
        theta = abs((leg_angle - a + math.pi) % (2 * math.pi) - math.pi)
        if best is None or theta < best[3]:
            best = (ring, d, sign, theta, a)
    return best


def _along_ring(ring: LineString, start: float, sign: int, length: float, count: int = 16) -> np.ndarray:
    """``count`` + 1 points along ``ring`` from position ``start`` over ``length`` in direction ``sign``."""
    pos = (start + sign * np.linspace(0.0, length, count + 1)) % ring.length
    return shapely.get_coordinates(shapely.line_interpolate_point(ring, pos))


def _merge_boundary_exit(lines: list, originals: list, k: int, end: int, stop: np.ndarray, domain, sharp: float,
                         tol: float) -> dict | None:
    """Collapse the first reach of barrier ``k`` leaving the boundary at a narrow angle onto it (module docstring)."""
    b = lines[k]
    i, step = (0, 1) if end == 0 else (len(b.xy) - 1, -1)
    apex = b.xy[i].copy()
    d = b.xy[i + step] - apex
    ring, along, sign, theta, _ = _boundary_ray(domain, apex, math.atan2(d[1], d[0]), tol)
    if not 0 < theta < sharp or not math.isfinite(b.lc):
        return None
    path = _path(b.xy, i, step, stop)
    cap = _length(path)
    bound = DEVIATION * b.lc
    q = min(bound / math.sin(theta), 0.5 * cap)
    region = domain.buffer(tol)
    probe = max(tol * 1e3, 1e-6 * ring.length)
    for _ in range(TRIES):
        foot, ahead = _along_ring(ring, along + sign * q, sign, probe, 1)
        outer = (math.atan2(*(ahead - foot)[::-1]),)
        reach = _bisect_reach(lambda r: _min_opening(foot, [_along(path, min(r, cap))[0]], outer), q, cap,
                              OPEN_FACTOR * sharp)
        if reach is not None:
            end_pt, n = _along(path, min(reach, cap))
            chord = LineString([foot, end_pt])
            trial = _tentative(lines, [(k, i, step, n - 1, [end_pt])])
            trial[k].xy[0 if step == 1 else -1] = foot
            # The original's first reach is matched by the boundary stretch it now ends on.
            new_with_boundary = dataclasses.replace(
                trial[k], xy=np.vstack([_along_ring(ring, along, sign, q)[:-1], trial[k].xy]) if step == 1
                else np.vstack([trial[k].xy, _along_ring(ring, along, sign, q)[::-1][1:]]))
            moved = _within_bound({k: new_with_boundary}, originals, apex, np.hypot(*(end_pt - apex)) + 2 * bound,
                                  bound)
            if moved is not None and np.hypot(*(end_pt - foot)) >= MIN_LEG * b.lc and region.covers(chord) \
                    and _clear_of_others(lines, {k}, chord, [end_pt], tol):
                lines[k] = trial[k]
                return {"kind": "boundary", "location": apex, "angle": math.degrees(theta),
                        "reach": float(np.hypot(*(end_pt - apex))), "deviation": moved, "lines": [b.line_id]}
        q *= SHRINK
    return None


def collapse_wedges(lines: list, domain, tol: float, sharp: float, on_boundary) -> tuple:
    """(lines, records): narrow junction sectors and boundary exits merged (module docstring).

    ``lines`` are copied first. ``on_boundary(xy)`` masks points on the
    domain boundary.
    """
    originals = lines
    lines = [dataclasses.replace(b, xy=b.xy.copy(), keep=b.keep.copy()) for b in lines]
    records, failed = [], set()
    for _ in range(MAX_MERGES):
        crowded = {tuple(r["location"]) for r in records if r["kind"] == "junction"}
        record = _next_merge(lines, originals, domain, tol, sharp, on_boundary, failed, crowded)
        if record is None:
            break
        records.append(record)
    return lines, records


def _next_merge(lines: list, originals: list, domain, tol: float, sharp: float, on_boundary, failed: set,
                crowded: set) -> dict | None:
    """Make the first merge that fits, or None when there is none left."""
    occurrences = _occurrences(lines)
    counts = {p: len(v) for p, v in occurrences.items()}
    stops = _stops(lines, counts)
    nodes = list(occurrences)
    boundary = on_boundary(np.array(nodes)) if nodes else np.zeros(0, dtype=bool)
    for p, at_boundary in zip(nodes, boundary):
        places = occurrences[p]
        if p in failed or (len(places) < 2 and not at_boundary):
            continue  # one interior vertex of one line has two legs: a bend, not a junction
        legs = _legs(lines, places)
        apex = np.array(p)
        if at_boundary:
            if len(legs) == 1 and len(legs[0]) == 1:
                leg = legs[0][0]
                record = _merge_boundary_exit(lines, originals, leg.k, 0 if leg.i == 0 else -1, stops[leg.k], domain,
                                              sharp, tol)
                if record is not None:
                    return record
                failed.add(p)
            continue
        if sum(len(group) for group in legs) < 3:
            continue  # a spike (one line, or two lines ending together): see ``network``
        for pair, limit in _pairs(legs, sharp, p in crowded):
            record = _merge_pair(lines, originals, apex, pair, stops, limit, tol)
            if record is not None:
                return record
        failed.add(p)
    return None
