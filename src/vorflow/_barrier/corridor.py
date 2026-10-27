"""Corridors: one row of cells shared across a narrow gap.

Two close, nearly parallel barrier pieces (closer than CORRIDOR_ANGLE, gap
below CORRIDOR_WIDTH x the target spacing h0), or a barrier running along a
straight edge of the domain boundary (closer than HUG_WIDTH x h0), would
otherwise need faces short enough for a row of pair sites on each side of
the gap (the barrier and boundary caps of ``sizing``), so a 0.5 lc gap
gets two rows of cells a quarter lc long. Instead the gap gets one row of
sites shared by both sides.

*Geometry.* M is a straight line through the gap: the bisector of the two
pieces, or the boundary edge itself. Sites sit on M at x = sigma_k.
Consecutive sites are separated by a *chord*, the perpendicular to M at
c_k = (sigma_k-1 + sigma_k) / 2. Where a chord meets a side there is a 1D
node whose circle passes through both sites (they are mirror images
across the chord), and each side segment between two chords has the site
between them and that site's mirror image across the side as its pair. So
every side segment is an exact Voronoi face: the ladder of ``rung``
without an apex. Across a boundary edge the sites' cells are cut by the
edge, which leaves one row of cells between the barrier and the boundary
(the sites are side-shared onto the boundary). A vertex of a side inside
the corridor is a forced chord.

*Strips* (``strips``). The chord spacing follows L(x) = min(h,
CORRIDOR_ASPECT x gap); the gap of a boundary run is twice the distance to
the edge (the run is a corridor between the barrier and its mirror image). Between forced chords the sites are spaced at equal
steps of the integral of 1/L; the offset of the first site after a forced
chord equals that of the last site before it. Every site must lie in the
middle 1 - 2 CORRIDOR_FOOT of its strip.

*Ends.*

- tip: both sides end at free ends ALIGN_FRACTION x L apart along M or
  closer (a boundary run: its side ends at a free end). The last site lies
  past both ends, each end node's radius is derived from it, and each end
  keeps its own tip site straight on (``network``). A single tip site
  mirrored across the end chord would sit point-symmetric to the outer
  site about the end node, so the line between those two cells would run
  exactly through the end of the barrier.
- boundary: both sides end on a straight piece of the domain boundary,
  which is the end chord; no tip site.
- stagger: one side ends at a free end, the other goes on. The end chord
  runs from the free end to the other side, tilted STAGGER_TILT away from
  the corridor (for the reason above), and the mirror image of the last
  site across it is the explicit pair site of the next segment of the
  other side; its cell wraps around the free end.
- mouth: a side ends at a kink (a hard node with two pieces). The pieces go
  on with regular segments from the chord's nodes, whose radii are derived
  from the last site.

Anything else (a junction, a well or another line in the gap, a boundary
corner) gets no corridor. A corridor that cannot be laid out, or whose
region keeps a defect, is dropped and its barriers get the regular
construction (``pipeline``).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

CORRIDOR_ANGLE = math.radians(12.0)
CORRIDOR_WIDTH = 1.2     # x h0: two barrier pieces closer than this share a corridor
HUG_WIDTH = 0.6          # x h0: a barrier closer than this to a boundary edge runs along it
CORRIDOR_ASPECT = 1.5    # strip length <= this x the gap
CORRIDOR_FOOT = 0.2      # a site keeps this fraction of its strip from both chords
MIN_OVERLAP = 2.0        # x h0: shorter facing stretches get no corridor
ALIGN_FRACTION = 0.25    # x L: two free ends this close along M end the corridor together
STAGGER_TILT = math.radians(10.0)
EDGE_MARGIN = 1.0        # x h0: a boundary run keeps this far from the ends of its edge
BISECT_STEPS = 60
GRID_POINTS = 48         # per interval between forced chords, for the integral of 1/L


@dataclasses.dataclass
class SidePiece:
    """A piece of a corridor side, with its hard nodes in the order along M."""
    piece: int
    start: int
    end: int


@dataclasses.dataclass
class Corridor:
    """A gap between one or two sides that gets a single shared row of sites.

    ``sides`` holds one list of SidePieces (a run along the boundary edge M)
    or two (two barriers); ``ends`` the end kinds at x0 and x1; ``end_nodes``
    the hard nodes on each end chord.
    """
    origin: np.ndarray
    m: np.ndarray
    sides: list
    x0: float
    x1: float
    ends: tuple
    end_nodes: tuple
    region: Polygon
    key: tuple


@dataclasses.dataclass
class CorridorLayout:
    """Runs of fixed 1D nodes with their explicit sites."""
    runs: list      # (piece, t from the piece's first node (ascending), sites of the segments between)
    extra: list     # (extra site, hard node whose circle it is on); none so far


# ---------------------------------------------------------------------------
# Small geometry
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    """``v`` / |v|."""
    return v / np.hypot(*v)


def _rot90(v: np.ndarray) -> np.ndarray:
    """``v`` turned 90 degrees CCW."""
    return np.array([-v[1], v[0]])


def proj(c: Corridor, xy: np.ndarray) -> np.ndarray:
    """Position along M of the points ``xy``."""
    return (np.asarray(xy, dtype=float) - c.origin) @ c.m


def _reflect(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mirror image of ``p`` across the line through a and b."""
    t = _unit(b - a)
    q = p - a
    return a + 2 * (q @ t) * t - q


def _piece_xy(net, sp: SidePiece) -> tuple:
    """(first, last) coordinates of a side piece along M."""
    return net.hard_xy[sp.start], net.hard_xy[sp.end]


def side_point(net, c: Corridor, side: list, x: float) -> tuple:
    """(side piece index, t along it from its first node along M, xy) where the chord at ``x`` meets ``side``."""
    for k, sp in enumerate(side):
        p0, p1 = _piece_xy(net, sp)
        x0, x1 = float(proj(c, p0)), float(proj(c, p1))
        if min(x0, x1) - 1e-9 <= x <= max(x0, x1) + 1e-9 or k == len(side) - 1:
            f = (x - x0) / (x1 - x0)
            return k, f * float(np.hypot(*(p1 - p0))), p0 + f * (p1 - p0)
    raise AssertionError("a corridor side has no pieces")


def width(net, c: Corridor, x: float) -> float:
    """The gap at ``x``: between the two sides, or between the side and its mirror image across the edge."""
    a = side_point(net, c, c.sides[0], x)[2]
    if len(c.sides) == 2:
        return float(np.hypot(*(a - side_point(net, c, c.sides[1], x)[2])))
    return 2.0 * float(np.hypot(*(a - c.origin - x * c.m)))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _degree(net) -> np.ndarray:
    """Number of pieces at each hard node."""
    return np.bincount(net.pieces.ravel(), minlength=len(net.hard_xy))


def _node_kind(net, v: int, degree: np.ndarray, rung_nodes: set) -> str:
    """"free", "kink", "boundary" or "other" (junction, ladder node) for a corridor end."""
    if v in rung_nodes:
        return "other"
    if net.plans[v].on_boundary:
        return "boundary"
    return {1: "free", 2: "kink"}.get(int(degree[v]), "other")


def _bisector(a0: np.ndarray, ua: np.ndarray, b0: np.ndarray, ub: np.ndarray) -> tuple:
    """(origin, m): the line midway between the lines (a0, ua) and (b0, ub), m oriented like ua."""
    na = _rot90(ua)
    na = na if na @ (b0 - a0) > 0 else -na
    nb = _rot90(ub)
    nb = nb if nb @ (a0 - b0) > 0 else -nb
    k = na - nb
    m = _unit(_rot90(k))
    m = m if m @ ua > 0 else -m
    origin = k * (na @ a0 - nb @ b0) / (k @ k)
    return origin + ((a0 - origin) @ m) * m, m


def _oriented(net, p: int, m: np.ndarray) -> SidePiece:
    """Piece ``p`` with its nodes in the order along ``m``."""
    a, b = (int(v) for v in net.pieces[p])
    return SidePiece(p, a, b) if (net.hard_xy[b] - net.hard_xy[a]) @ m >= 0 else SidePiece(p, b, a)


@dataclasses.dataclass
class Obstacles:
    """Everything a corridor's gap must be free of: barrier pieces (first), standard lines and feature nodes."""
    geoms: list
    tree: shapely.STRtree


def obstacles_of(net, standard: list, features: np.ndarray) -> Obstacles:
    """The barrier pieces, standard lines and feature nodes, indexed (pieces keep their numbers)."""
    geoms = [LineString(net.hard_xy[pc]) for pc in net.pieces] + list(standard)
    geoms += [Point(f) for f in np.asarray(features, dtype=float).reshape(-1, 2)]
    return Obstacles(geoms, shapely.STRtree(geoms))


def _clear(region: Polygon, obstacles: Obstacles, own: set, domain) -> bool:
    """True when ``region`` lies in the domain and no obstacle except the pieces in ``own`` reaches into it."""
    if region.is_empty or region.area <= 0 or not domain.buffer(1e-9 * region.length).covers(region):
        return False
    inner = region.buffer(-1e-6 * math.sqrt(region.area))
    if inner.is_empty:
        return True
    hits = obstacles.tree.query(inner, predicate="intersects")
    return all(int(k) in own for k in hits)


def _end_kind(kinds: list, aligned: bool, boundary_chord: bool) -> str | None:
    """Kind of a pair corridor's end from the kinds of the ending nodes (None: no corridor)."""
    if aligned:
        if kinds == ["free", "free"]:
            return "tip"
        if kinds == ["boundary", "boundary"] and boundary_chord:
            return "boundary"
        return None
    return {"free": "stagger", "kink": "mouth"}.get(kinds[0])


def _pair_corridor(net, pa: int, pb: int, size_fn, domain, obstacles: Obstacles, degree, rung_nodes) -> Corridor | None:
    """The corridor between pieces ``pa`` and ``pb``, or None when they do not make one."""
    xy = net.hard_xy
    A0, A1 = xy[net.pieces[pa]]
    B0, B1 = xy[net.pieces[pb]]
    ua, ub = _unit(A1 - A0), _unit(B1 - B0)
    if ua @ ub < 0:
        ub = -ub
    if math.acos(float(np.clip(ua @ ub, -1.0, 1.0))) >= CORRIDOR_ANGLE:
        return None
    if LineString([A0, A1]).intersects(LineString([B0, B1])):
        return None
    origin, m = _bisector(A0, ua, B0, ub)
    sa, sb = _oriented(net, pa, m), _oriented(net, pb, m)
    probe = Corridor(origin, m, [[sa], [sb]], 0.0, 0.0, (), (), Polygon(), (pa, pb))
    xa = proj(probe, xy[[sa.start, sa.end]])
    xb = proj(probe, xy[[sb.start, sb.end]])
    x0, x1 = max(xa[0], xb[0]), min(xa[1], xb[1])
    if x1 <= x0:
        return None
    pts = [side_point(net, probe, s, x)[2] for s, x in (([sa], x0), ([sa], x1), ([sb], x1), ([sb], x0))]
    region = Polygon(pts)
    h0 = min(float(net.piece_lc[pa]), float(net.piece_lc[pb]),
             float(size_fn(np.array(region.centroid.coords))[0]))
    gaps = [width(net, probe, x0), width(net, probe, x1)]
    if x1 - x0 < MIN_OVERLAP * h0 or max(gaps) > CORRIDOR_WIDTH * h0:
        return None
    if not _clear(region, obstacles, {pa, pb}, domain):
        return None
    ends, end_nodes, xs = [], [], []
    for x, nodes_x, nodes in ((x0, (xa[0], xb[0]), (sa.start, sb.start)), (x1, (xa[1], xb[1]), (sa.end, sb.end))):
        align = ALIGN_FRACTION * min(h0, CORRIDOR_ASPECT * min(gaps))
        aligned = abs(nodes_x[0] - nodes_x[1]) <= align
        ending = list(nodes) if aligned else [nodes[int(np.argmin(np.abs(np.array(nodes_x) - x)))]]
        kinds = [_node_kind(net, v, degree, rung_nodes) for v in ending]
        chord = LineString(xy[list(nodes)])
        kind = _end_kind(kinds, aligned, aligned and domain.boundary.buffer(1e-6 * chord.length).covers(chord))
        if kind is None:
            return None
        ends.append(kind)
        end_nodes.append(ending)
        along = proj(probe, xy[ending])
        xs.append(float(along.max() if x == x0 else along.min()) if kind == "tip" else float(along.mean()))
    probe.x0, probe.x1 = xs
    probe.ends, probe.end_nodes, probe.region = tuple(ends), tuple(end_nodes), region
    return probe


def pair_corridors(net, size_fn, domain, standard: list, features: np.ndarray, skip: set) -> list:
    """Corridors between two close, nearly parallel barrier pieces (each piece in one corridor at most)."""
    degree = _degree(net)
    rung_nodes = {int(v) for r in net.rungs for v in r.nodes}
    rung_pieces = {int(p) for r in net.rungs for p in r.pieces}
    lines = [LineString(net.hard_xy[pc]) for pc in net.pieces]
    tree = shapely.STRtree(lines)
    obstacles = obstacles_of(net, standard, features)
    candidates = []
    for pa, line in enumerate(lines):
        if pa in rung_pieces:
            continue
        reach = CORRIDOR_WIDTH * float(net.piece_lc[pa])
        for pb in tree.query(line, predicate="dwithin", distance=reach):
            pb = int(pb)
            if pb <= pa or pb in rung_pieces or set(net.pieces[pa]) & set(net.pieces[pb]):
                continue
            if (pa, pb) in skip:
                continue
            candidates.append((line.distance(lines[pb]), pa, pb))
    used, out = set(), []
    for _, pa, pb in sorted(candidates):
        if pa in used or pb in used:
            continue
        c = _pair_corridor(net, pa, pb, size_fn, domain, obstacles, degree, rung_nodes)
        if c is not None:
            out.append(c)
            used.update((pa, pb))
    return out


def _edges(domain) -> list:
    """Straight edges (start, end) of every ring of the domain."""
    out = []
    for poly in getattr(domain, "geoms", [domain]):
        for ring in [poly.exterior, *poly.interiors]:
            coords = np.asarray(ring.coords, dtype=float)[:, :2]
            out += [(coords[k], coords[k + 1]) for k in range(len(coords) - 1)]
    return out


def _hugged_edge(net, p: int, edges: list, size_fn, domain, obstacles: Obstacles) -> tuple | None:
    """(edge index, h0) of the boundary edge piece ``p`` runs along, or None."""
    a, b = net.hard_xy[net.pieces[p]]
    u = _unit(b - a)
    best = None
    for k, (e0, e1) in enumerate(edges):
        w = _unit(e1 - e0)
        if math.acos(min(1.0, abs(float(u @ w)))) >= CORRIDOR_ANGLE:
            continue
        n = _rot90(w)
        d = np.array([(a - e0) @ n, (b - e0) @ n])
        if (d[0] > 0) != (d[1] > 0) or np.abs(d).min() <= 0:
            continue
        t = np.array([(a - e0) @ w, (b - e0) @ w])
        h0 = min(float(net.piece_lc[p]), float(size_fn(np.array([0.5 * (a + b)]))[0]))
        length = float(np.hypot(*(e1 - e0)))
        if t.min() < EDGE_MARGIN * h0 or t.max() > length - EDGE_MARGIN * h0:
            continue
        if np.abs(d).max() > HUG_WIDTH * h0:
            continue
        feet = [e0 + t[0] * w, e0 + t[1] * w]
        region = Polygon([a, b, feet[1], feet[0]])
        if not _clear(region, obstacles, {p}, domain):
            continue
        rest = domain.boundary.difference(LineString([e0, e1]).buffer(1e-9 * length))
        if not rest.is_empty and rest.distance(LineString([a, b])) < h0:
            continue
        if best is None or np.abs(d).max() < best[2]:
            best = (k, h0, float(np.abs(d).max()))
    return None if best is None else best[:2]


def boundary_runs(net, size_fn, domain, standard: list, features: np.ndarray, skip: set, used: set) -> list:
    """Corridors of barrier chains running along a straight boundary edge."""
    degree = _degree(net)
    rung_nodes = {int(v) for r in net.rungs for v in r.nodes}
    rung_pieces = {int(p) for r in net.rungs for p in r.pieces}
    edges = _edges(domain)
    obstacles = obstacles_of(net, standard, features)
    hugging = {}
    for p in range(len(net.pieces)):
        if p in rung_pieces or p in used:
            continue
        found = _hugged_edge(net, p, edges, size_fn, domain, obstacles)
        if found is not None:
            hugging[p] = found
    out = []
    for k in sorted({e for e, _ in hugging.values()}):
        e0, e1 = edges[k]
        pieces = [p for p, (e, _) in hugging.items() if e == k]
        for chain in _chains(net, pieces, degree, rung_nodes):
            c = _run(net, chain, e0, e1, degree, rung_nodes, min(hugging[p][1] for p in chain))
            if c is not None and c.key not in skip:
                out.append(c)
    return out


def _chains(net, pieces: list, degree: np.ndarray, rung_nodes: set) -> list:
    """``pieces`` grouped into chains joined at kinks (hard nodes with two pieces, both in ``pieces``)."""
    members = set(pieces)
    left = set(pieces)
    out = []
    while left:
        chain = [left.pop()]
        grown = True
        while grown:
            grown = False
            for end in (0, -1):
                p = chain[end]
                for v in net.pieces[p]:
                    if degree[v] != 2 or v in rung_nodes or net.plans[v].on_boundary:
                        continue
                    q = next(int(q) for q in np.flatnonzero((net.pieces == v).any(axis=1)) if q != p)
                    if q in members and q in left:
                        left.discard(q)
                        chain.insert(0, q) if end == 0 else chain.append(q)
                        grown = True
        out.append(chain)
    return out


def _run(net, chain: list, e0: np.ndarray, e1: np.ndarray, degree, rung_nodes, h0: float) -> Corridor | None:
    """The boundary corridor of a chain along the edge e0-e1, or None."""
    m = _unit(e1 - e0)
    sides = [_oriented(net, p, m) for p in chain]
    sides.sort(key=lambda sp: float((net.hard_xy[sp.start] - e0) @ m))
    for s0, s1 in zip(sides[:-1], sides[1:]):
        if s0.end != s1.start:
            return None
    xy = net.hard_xy
    c = Corridor(e0.copy(), m, [sides], float((xy[sides[0].start] - e0) @ m), float((xy[sides[-1].end] - e0) @ m),
                 (), (), Polygon(), ("boundary",) + tuple(sorted(chain)))
    ends = []
    for v in (sides[0].start, sides[-1].end):
        kind = _node_kind(net, v, degree, rung_nodes)
        if kind not in ("free", "kink"):
            return None
        ends.append("tip" if kind == "free" else "mouth")
    c.ends, c.end_nodes = tuple(ends), ([sides[0].start], [sides[-1].end])
    pts = [xy[sp.start] for sp in sides] + [xy[sides[-1].end]]
    feet = [e0 + float((p - e0) @ m) * m for p in pts]
    c.region = Polygon(pts + feet[::-1])
    return c


def find_corridors(net, size_fn, domain, standard: list, features: np.ndarray, skip: set) -> list:
    """Pair corridors first, then boundary runs on the pieces left."""
    pairs = pair_corridors(net, size_fn, domain, standard, features, skip)
    used = {sp.piece for c in pairs for side in c.sides for sp in side}
    return pairs + boundary_runs(net, size_fn, domain, standard, features, skip, used)


def chord_nodes(c: Corridor) -> set:
    """Hard nodes whose radius ``c`` sets: its end nodes (except free tips) and the vertices of its sides."""
    out = {int(v) for kind, nodes in zip(c.ends, c.end_nodes) if kind != "tip" for v in nodes}
    for side in c.sides:
        out.update(int(sp.end) for sp in side[:-1])
    return out


def contains(c: Corridor, piece: int, xy: np.ndarray) -> np.ndarray:
    """Mask of the samples ``xy`` of ``piece`` that lie in the corridor's stretch."""
    if all(sp.piece != piece for side in c.sides for sp in side):
        return np.zeros(len(xy), dtype=bool)
    x = proj(c, xy)
    tol = 1e-9 * max(1.0, abs(c.x1 - c.x0))
    return (x >= c.x0 - tol) & (x <= c.x1 + tol)


# ---------------------------------------------------------------------------
# Strips
# ---------------------------------------------------------------------------
def _margin(a: float, b: float, sites: np.ndarray) -> float:
    """Smallest fraction of its strip any site keeps from a chord (strips between a, the site midpoints, b)."""
    chords = np.concatenate([[a], 0.5 * (sites[1:] + sites[:-1]), [b]])
    ratio = (sites - chords[:-1]) / (chords[1:] - chords[:-1])
    return float(min(ratio.min(), (1 - ratio).min()))


def _interval_sites(a: float, b: float, eps: float, m: int, grid: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    """``m`` sites in (a, b): the first ``eps`` past a, then equal steps of q (the integral of 1/L)
    such that the last one sits half its spacing before b."""
    first = a + eps
    if first >= b:
        return None
    if m == 1:
        return np.array([first])
    q1 = float(np.interp(first, grid, q))

    def place(step: float) -> np.ndarray:
        return np.interp(q1 + step * np.arange(m), q, grid)

    lo, hi = 0.0, (float(np.interp(b, grid, q)) - q1) / (m - 1)
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        s = place(mid)
        if s[-1] + 0.5 * (s[-1] - s[-2]) > b:
            hi = mid
        else:
            lo = mid
    s = place(lo)
    return s if np.all(np.diff(s) > 0) and s[-1] < b else None


def strips(forced: np.ndarray, spacing, eps0: float, foot: float = CORRIDOR_FOOT) -> np.ndarray | None:
    """Site positions along M between the forced chords (module docstring), or None when a strip fails."""
    out = []
    eps = eps0
    for a, b in zip(forced[:-1], forced[1:]):
        grid = np.linspace(a, b, GRID_POINTS)
        inv = 1.0 / spacing(grid)
        q = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(grid))])
        target = max(1, int(round(q[-1])))
        best = None
        for m in sorted({target, target - 1, target + 1} - {0}, key=lambda k: abs(k - q[-1])):
            s = _interval_sites(a, b, eps, m, grid, q)
            if s is None:
                continue
            margin = _margin(a, b, s)
            if margin >= foot and (best is None or margin > best[0] + 0.1):
                best = (margin, s)
        if best is None:
            return None
        out.append(best[1])
        eps = b - best[1][-1]
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _length(net, sp: SidePiece) -> float:
    """Length of a side piece."""
    return float(np.hypot(*np.subtract(*_piece_xy(net, sp))))


def _spacing_fn(c: Corridor, net, field, aspect: float):
    """L(x) = min(h on the sides, aspect x gap)."""
    tables = []
    for side in c.sides:
        x = np.concatenate([proj(c, field.xy[field.piece == sp.piece]) for sp in side])
        h = np.concatenate([field.h[field.piece == sp.piece] for sp in side])
        order = np.argsort(x)
        tables.append((x[order], h[order]))
    probe = np.unique(np.concatenate([np.linspace(c.x0, c.x1, 64), _forced(c, net)]))
    gaps = np.array([width(net, c, float(x)) for x in probe])

    def spacing(x: np.ndarray) -> np.ndarray:
        h = np.minimum.reduce([np.interp(x, tx, th) for tx, th in tables])
        return np.minimum(h, aspect * np.interp(x, probe, gaps))

    return spacing


def _forced(c: Corridor, net) -> np.ndarray:
    """Chord positions fixed by the ends and the side vertices."""
    inner = [float(proj(c, net.hard_xy[sp.end])) for side in c.sides for sp in side[:-1]]
    inner = [x for x in inner if c.x0 < x < c.x1]
    return np.unique(np.array([c.x0, *inner, c.x1]))


def _end_chord(c: Corridor, net, end: int, x: float) -> tuple:
    """(point, direction) of the end chord at ``end``: tilted from the free end at a stagger, else perpendicular to M."""
    n = _rot90(c.m)
    if c.ends[end] != "stagger":
        return c.origin + x * c.m, n
    a = net.hard_xy[c.end_nodes[end][0]]
    ending_side = next(k for k, side in enumerate(c.sides) if c.end_nodes[end][0] in (side[0].start, side[-1].end))
    other = side_point(net, c, c.sides[1 - ending_side], x)[2]
    n = _unit(other - a)
    return a, _unit(n + math.tan(STAGGER_TILT) * (c.m if end == 1 else -c.m))


def _on_chord(net, side: list, point: np.ndarray, direction: np.ndarray) -> tuple:
    """(side piece index, t) where the line through ``point`` along ``direction`` crosses ``side``."""
    for i, sp in enumerate(side):
        p0, p1 = _piece_xy(net, sp)
        u = p1 - p0
        den = u[0] * direction[1] - u[1] * direction[0]
        if abs(den) < 1e-15:
            continue
        w = point - p0
        f = (w[0] * direction[1] - w[1] * direction[0]) / den
        if -1e-9 <= f <= 1 + 1e-9 or i == len(side) - 1:
            return i, f * _length(net, sp)
    raise AssertionError("an end chord misses a corridor side")


def _chord_nodes(net, c: Corridor, side: list, chords: np.ndarray) -> list:
    """(side piece index, t along M order) where each chord meets ``side``; end chords take its ending nodes."""
    out = []
    for k, x in enumerate(chords):
        if k in (0, len(chords) - 1):
            end = 0 if k == 0 else 1
            idx = 0 if end == 0 else len(side) - 1
            node = side[0].start if end == 0 else side[-1].end
            if node in c.end_nodes[end]:
                out.append((idx, 0.0 if end == 0 else _length(net, side[idx])))
                continue
            if c.ends[end] == "stagger":
                out.append(_on_chord(net, side, *_end_chord(c, net, end, float(x))))
                continue
        out.append(side_point(net, c, side, float(x))[:2])
    return out


def _runs_of_side(net, side: list, nodes: list, seg_sites: list) -> list:
    """(piece, t from the piece's first node, sites) per side piece, from the chord nodes in order along M.

    A chord node at a side vertex ends one piece's run and starts the next one's.
    """
    per = {}
    for j, (i, t) in enumerate(nodes):
        per.setdefault(i, []).append((t, j))
        length = _length(net, side[i])
        if i + 1 < len(side) and abs(t - length) <= 1e-9 * max(1.0, length):
            per.setdefault(i + 1, []).append((0.0, j))
    out = []
    for i, items in sorted(per.items()):
        if len(items) < 2:
            continue
        sp = side[i]
        t = np.array([a for a, _ in items])
        sites = [seg_sites[j] for _, j in items[:-1]]
        if net.pieces[sp.piece][0] == sp.start:
            out.append((sp.piece, t, sites))
        else:
            out.append((sp.piece, _length(net, sp) - t[::-1], sites[::-1]))
    return out


def _stagger_node(net, side: list, node: tuple, tip: np.ndarray, at_start: bool) -> tuple | None:
    """(side piece index, t) of the far node of the continuing side's segment whose site is the tip site.

    The segment is twice the distance from the chord to the tip site's
    foot, or reaches the piece's free end if that is close.
    """
    i, t = node
    sp = side[i]
    p0, p1 = _piece_xy(net, sp)
    length = _length(net, sp)
    foot = float((tip - p0) @ _unit(p1 - p0))
    half = (t - foot) if at_start else (foot - t)
    if half <= 0:
        return None
    new = t - 2 * half if at_start else t + 2 * half
    if 0.3 * half <= new <= length - 0.3 * half:
        return i, new
    end_node = sp.start if at_start else sp.end
    room = t if at_start else length - t
    free = int((net.pieces == end_node).any(axis=1).sum()) == 1 and not net.plans[end_node].on_boundary
    if free and room * (1 - CORRIDOR_FOOT) >= half:
        return i, 0.0 if at_start else length
    return None


def lay_out(c: Corridor, net, field, aspect: float = CORRIDOR_ASPECT) -> CorridorLayout | None:
    """Fixed 1D nodes and explicit sites of a corridor from the current spacing, or None."""
    spacing = _spacing_fn(c, net, field, aspect)
    forced = _forced(c, net)
    sites_x = strips(forced, spacing, 0.5 * float(spacing(np.array([c.x0]))[0]))
    if sites_x is None:
        return None
    chords = np.concatenate([[forced[0]], 0.5 * (sites_x[1:] + sites_x[:-1]), [forced[-1]]])
    sites = c.origin + np.outer(sites_x, c.m)
    runs = []
    for side in c.sides:
        nodes = _chord_nodes(net, c, side, chords)
        seg_sites = list(sites)
        for end, kind in enumerate(c.ends):
            ending = {side[0].start, side[-1].end} & set(c.end_nodes[end])
            if kind != "stagger" or ending:
                continue
            point, direction = _end_chord(c, net, end, float(chords[0] if end == 0 else chords[-1]))
            tip = _reflect(sites[0] if end == 0 else sites[-1], point, point + direction)
            new = _stagger_node(net, side, nodes[0] if end == 0 else nodes[-1], tip, end == 0)
            if new is None:
                return None
            if end == 0:
                nodes.insert(0, new)
                seg_sites.insert(0, tip)
            else:
                nodes.append(new)
                seg_sites.append(tip)
        runs += _runs_of_side(net, side, nodes, seg_sites)
    return CorridorLayout(runs, [])
