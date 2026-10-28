"""Corridors: one row of cells shared across a narrow gap.

Two close, nearly parallel barrier strands (closer than CORRIDOR_ANGLE, gap
below CORRIDOR_WIDTH x the target spacing h0), or a barrier running along a
straight edge of the domain boundary (closer than HUG_WIDTH x h0), would
otherwise need faces short enough for a row of pair sites on each side of
the gap (the barrier and boundary caps of ``sizing``), so a 0.5 lc gap
gets two rows of cells a quarter lc long. Instead the gap gets one row of
sites shared by both sides. The row keeps the gap open along its length,
so flow along the gap is modelled, however narrow it is.

A *strand* is a run of pieces joined at kinks (hard nodes with two pieces,
off the boundary and off every ladder), each within CORRIDOR_ANGLE of M
and going the same way along it: a digitised or wavy line, or the
straight middle of a lens. The two strands may be two lines or the two
legs of one line (a U-turn, a thin loop): see the ends below.

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

*Marching* (``march``). From the start chord, each next site goes on M (the
bisector of the current side pieces; the edge for a boundary run) at the
local spacing L = min(h, CORRIDOR_ASPECT x gap), where the gap of a
boundary run is the distance to the edge (the edge cuts its cells to that
width). So the cells are at most
CORRIDOR_ASPECT times longer than wide, and their number grows smoothly as
the gap narrows (1 / gap below h / CORRIDOR_ASPECT). The steps are evened
out so that the sites straddle the next event (a side vertex or the end)
symmetrically, steps grow or shrink by at most MAX_GROWTH (a target that
changes faster is followed more slowly) and consecutive steps differ by at
most MAX_STEP_RATIO. A
side vertex is a forced chord: the next site is where the circle about the
vertex through the last site meets the next M, so the perpendicular
bisector of the two sites passes through the vertex and M bends there.

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
- bend: both sides end at kinks joined by a short path of kinks (at most
  BEND_REACH x the gap long): the turn of a U, or the end of a thin loop
  or of a closed lens. The path may pass through the apex of a ladder
  between two of its pieces (a lens's tapered end); the bend replaces the
  ladder.
  Every circle along the path passes through the last site, so each piece
  of the path has the last site and its mirror image across the piece as
  its pair: the last cell fills the turn.
- fork: both sides end at kinks at the same place along M and go on apart
  (the end of a lens's straight middle). The mirror image of the last site
  across the line through the two kinks lies on both kinks' circles; it is
  the site of the first segment of both pieces beyond, each twice as long
  as the distance to the site's foot.

Anything else (a junction, a well or another line in the gap, a boundary
corner) gets no corridor; the barriers there get the regular construction,
with faces capped by the gap (``sizing``), which also keeps the gap open. A
corridor that cannot be laid out, or whose region keeps a defect, is
dropped the same way (``pipeline``), and its pieces get no other pair
corridor. With ``MeshGenerator(gap_field=...)``, a gap where GapField asks
for more than one cell across gets no corridor either (``gap_field_wins``):
the barriers get their own faces, sized by GapField, with its rows of cells
between them.
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
CORRIDOR_ASPECT = 3.0    # strip length <= this x the gap (cells at most 3:1; see design note 08)
CORRIDOR_FOOT = 0.2      # a site keeps this fraction of its strip from both chords
MIN_OVERLAP = 2.0        # x h0: shorter facing stretches get no corridor
ALIGN_FRACTION = 0.25    # x L: two free ends this close along M end the corridor together
STAGGER_TILT = math.radians(10.0)
EDGE_MARGIN = 1.0        # x h0: a boundary run keeps this far from the ends of its edge
COLLINEAR_FRACTION = 1e-9  # x piece length: pieces this close to one line are collinear, not a corridor
BEND_REACH = 3.0         # x gap: the longest path of kinks that joins two sides' ends as a bend (a U-turn)
MAX_STRAND = 10000       # pieces in a strand or a bend at most


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
    the hard nodes on each end chord. x0 and x1 bound the corridor along
    (origin, m), for the size caps.
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
    gap: float = 0.0          # the narrowest gap (a boundary run: twice its distance to the edge)
    paths: tuple = ((), ())   # per end: the pieces of a bend from one side's end to the other's
    forks: tuple = ((), ())   # per end: the piece each side goes on into past a fork
    span: tuple | None = None  # (lo, hi) along M of the sides' stretch laid out by the corridor (default x0, x1)


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


def _nearest_on(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The point of segment a-b nearest to ``p``."""
    d = b - a
    return a + float(np.clip((p - a) @ d / (d @ d), 0.0, 1.0)) * d


def _bisector(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> tuple:
    """(origin, m): the line midway between the lines through pieces a0-a1 and b0-b1, m oriented like a1 - a0.

    Each normal is oriented from the piece's midpoint towards the nearest
    point of the other piece, so a short piece of a wavy strand facing a
    long one, and pieces that meet, work too. Collinear pieces have no
    bisector; their common line is returned (callers reject such pairs).
    """
    ua, ub = _unit(a1 - a0), _unit(b1 - b0)
    mid_a, mid_b = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
    na = _rot90(ua)
    na = na if na @ (_nearest_on(b0, b1, mid_a) - mid_a) > 0 else -na
    nb = _rot90(ub)
    nb = nb if nb @ (_nearest_on(a0, a1, mid_b) - mid_b) > 0 else -nb
    k = na - nb
    if k @ k <= COLLINEAR_FRACTION ** 2:
        return a0.copy(), ua
    m = _unit(_rot90(k))
    m = m if m @ ua > 0 else -m
    origin = k * (na @ a0 - nb @ b0) / (k @ k)
    return origin + ((a0 - origin) @ m) * m, m


def collinear(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
    """True when piece b lies on the line through piece a (to COLLINEAR_FRACTION of the longer piece)."""
    scale = max(float(np.hypot(*(a1 - a0))), float(np.hypot(*(b1 - b0))))
    offsets = (np.array([b0, b1]) - a0) @ _rot90(_unit(a1 - a0))
    return bool(np.abs(offsets).max() <= COLLINEAR_FRACTION * scale)


def _oriented(net, p: int, m: np.ndarray) -> SidePiece:
    """Piece ``p`` with its nodes in the order along ``m``."""
    a, b = (int(v) for v in net.pieces[p])
    return SidePiece(p, a, b) if (net.hard_xy[b] - net.hard_xy[a]) @ m >= 0 else SidePiece(p, b, a)


@dataclasses.dataclass
class Obstacles:
    """Everything a corridor's gap must be free of: barrier pieces (first), standard and fixed lines (zone
    edges, quad-buffer strips) and feature nodes."""
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
        if kinds == ["kink", "kink"]:
            return "fork"
        return None
    return {"free": "stagger", "kink": "mouth"}.get(kinds[0])


def _other_piece(net, v: int, p: int) -> int:
    """The piece at kink ``v`` other than ``p``."""
    return next(int(q) for q in np.flatnonzero((net.pieces == v).any(axis=1)) if q != p)


def _far_node(net, p: int, v: int) -> int:
    """The node of piece ``p`` other than ``v``."""
    a, b = (int(x) for x in net.pieces[p])
    return b if a == v else a


def _joinable(net, v: int, degree: np.ndarray, rung_nodes: set) -> bool:
    """True when a strand may go on through hard node ``v``: a kink off the boundary and off every ladder."""
    return int(degree[v]) == 2 and v not in rung_nodes and not net.plans[v].on_boundary


def _strand(net, p: int, m: np.ndarray, degree: np.ndarray, rung_nodes: set, rung_pieces: set,
            blocked: set) -> list:
    """Piece ``p`` and the pieces it runs on into through kinks, each within CORRIDOR_ANGLE of ``m`` and going
    the same way along it, as SidePieces in order along ``m`` (none with a node in ``blocked``)."""
    cos_max = math.cos(CORRIDOR_ANGLE)
    out = [_oriented(net, p, m)]
    seen = {p}
    for forward in (True, False):
        while len(out) <= MAX_STRAND:
            v = out[-1].end if forward else out[0].start
            if not _joinable(net, v, degree, rung_nodes):
                break
            q = _other_piece(net, v, out[-1].piece if forward else out[0].piece)
            w = _far_node(net, q, v)
            step = net.hard_xy[w] - net.hard_xy[v]
            along = float(step @ m) / float(np.hypot(*step))
            if q in seen or q in rung_pieces or w in blocked or (along if forward else -along) < cos_max:
                break
            seen.add(q)
            sp = SidePiece(q, v, w) if forward else SidePiece(q, w, v)
            out = out + [sp] if forward else [sp] + out
    return out


def _trim(net, side: list, probe, x0: float, x1: float) -> list:
    """The pieces of ``side`` whose stretch along M overlaps (x0, x1)."""
    out = []
    for sp in side:
        a, b = (float(x) for x in proj(probe, np.array(_piece_xy(net, sp))))
        if b > x0 and a < x1:
            out.append(sp)
    return out


def _side_line(net, probe, side: list, x0: float, x1: float) -> list:
    """Points of ``side`` from x0 to x1 along M: where the chords there meet it, and its vertices between."""
    pts = [side_point(net, probe, side, x0)[2]]
    for sp in side[:-1]:
        x = float(proj(probe, net.hard_xy[sp.end]))
        if x0 < x < x1:
            pts.append(net.hard_xy[sp.end])
    return pts + [side_point(net, probe, side, x1)[2]]


def _bend_path(net, a: int, pa: int, b: int, reach: float, degree: np.ndarray, rung_nodes: set,
               apexes: set = frozenset()) -> list | None:
    """Pieces of the short path of kinks from ``a`` (leaving piece ``pa``) to ``b``, at most ``reach`` long.

    The path may pass through the apex of a ladder between two of its
    pieces (``apexes``: the tapered end of a lens); the bend replaces it.
    """
    path, v, p, length = [], a, pa, 0.0
    while len(path) < MAX_STRAND:
        q = _other_piece(net, v, p)
        w = _far_node(net, q, v)
        length += float(np.hypot(*(net.hard_xy[w] - net.hard_xy[v])))
        if q in path or length > reach:
            return None
        path.append(q)
        if w == b:
            return path
        apex = w in apexes and int(degree[w]) == 2 and not net.plans[w].on_boundary
        if not (_joinable(net, w, degree, rung_nodes) or apex):
            return None
        v, p = w, q
    return None


def _pair_end(net, probe, sides: list, end: int, x: float, gap: float, h0: float, domain, degree: np.ndarray,
              rung_nodes: set, rung_pieces: set = frozenset(), apexes: set = frozenset()) -> tuple | None:
    """(kind, ending nodes, x, bend path, fork pieces) of a pair corridor's end ``end`` at ``x``, or None."""
    xy = net.hard_xy
    nodes = [side[0].start if end == 0 else side[-1].end for side in sides]
    pieces = [side[0].piece if end == 0 else side[-1].piece for side in sides]
    kinds = [_node_kind(net, v, degree, rung_nodes) for v in nodes]
    if kinds == ["kink", "kink"]:
        path = _bend_path(net, nodes[0], pieces[0], nodes[1], BEND_REACH * gap, degree, rung_nodes, apexes)
        if path is not None:
            return "bend", nodes, float(proj(probe, xy[nodes]).mean()), path, ()
    nodes_x = proj(probe, xy[nodes])
    aligned = abs(float(nodes_x[0] - nodes_x[1])) <= ALIGN_FRACTION * min(h0, CORRIDOR_ASPECT * gap)
    ending = nodes if aligned else [nodes[int(np.argmin(np.abs(nodes_x - x)))]]
    kinds = [_node_kind(net, v, degree, rung_nodes) for v in ending]
    chord = LineString(xy[nodes])
    kind = _end_kind(kinds, aligned, aligned and domain.boundary.buffer(1e-6 * chord.length).covers(chord))
    if kind is None:
        return None
    along = proj(probe, xy[ending])
    if kind == "tip":
        return kind, ending, float(along.max() if end == 0 else along.min()), (), ()
    forks = tuple(_other_piece(net, v, p) for v, p in zip(nodes, pieces)) if kind == "fork" else ()
    if set(forks) & rung_pieces:
        return None
    return kind, ending, float(along.mean()), (), forks


def _pair_corridor(net, pa: int, pb: int, size_fn, domain, obstacles: Obstacles, degree, rung_nodes,
                   rung_pieces: set, used: set) -> Corridor | None:
    """The corridor between the strands through pieces ``pa`` and ``pb``, or None when they do not make one."""
    xy = net.hard_xy
    A0, A1 = xy[net.pieces[pa]]
    B0, B1 = xy[net.pieces[pb]]
    ua, ub = _unit(A1 - A0), _unit(B1 - B0)
    if ua @ ub < 0:
        ub, B0, B1 = -ub, B1, B0
    if math.acos(float(np.clip(ua @ ub, -1.0, 1.0))) >= CORRIDOR_ANGLE:
        return None
    if LineString([A0, A1]).intersects(LineString([B0, B1])) or collinear(A0, A1, B0, B1):
        return None
    origin, m = _bisector(A0, A1, B0, B1)
    side_a = _strand(net, pa, m, degree, rung_nodes, rung_pieces | used, set())
    nodes_a = {v for sp in side_a for v in (sp.start, sp.end)}
    side_b = _strand(net, pb, m, degree, rung_nodes, rung_pieces | used, nodes_a)
    probe = Corridor(origin, m, [side_a, side_b], 0.0, 0.0, (), (), Polygon(), (pa, pb))
    xa = proj(probe, xy[[side_a[0].start, side_a[-1].end]])
    xb = proj(probe, xy[[side_b[0].start, side_b[-1].end]])
    x0, x1 = max(xa[0], xb[0]), min(xa[1], xb[1])
    if x1 <= x0:
        return None
    sides = [_trim(net, side_a, probe, x0, x1), _trim(net, side_b, probe, x0, x1)]
    probe.sides = sides
    lines = [_side_line(net, probe, side, x0, x1) for side in sides]
    if LineString(lines[0]).intersects(LineString(lines[1])):
        return None
    region = Polygon(lines[0] + lines[1][::-1])
    if not region.is_valid:
        return None
    pieces = [sp.piece for side in sides for sp in side]
    h0 = min(float(net.piece_lc[pieces].min()), float(size_fn(np.array(region.centroid.coords))[0]))
    xs = [x0, x1] + [float(x) for side in sides for sp in side for x in proj(probe, xy[[sp.start, sp.end]])
                     if x0 < x < x1]
    gaps = [width(net, probe, x) for x in xs]
    if x1 - x0 < MIN_OVERLAP * h0 or max(gaps) > CORRIDOR_WIDTH * h0:
        return None
    if not _clear(region, obstacles, set(pieces), domain):
        return None
    apexes = {int(r.nodes[0]) for r in net.rungs if r.t0 == 0}
    ends = [_pair_end(net, probe, sides, end, x, min(gaps), h0, domain, degree, rung_nodes, rung_pieces, apexes)
            for end, x in ((0, x0), (1, x1))]
    if None in ends:
        return None
    if ends[0][0] == ends[1][0] == "bend" and set(ends[0][3]) & set(ends[1][3]):
        return None
    probe.ends = tuple(e[0] for e in ends)
    probe.end_nodes = tuple(e[1] for e in ends)
    probe.x0, probe.x1 = ends[0][2], ends[1][2]
    probe.region, probe.gap = region, float(min(gaps))
    whole = ("tip", "boundary", "bend", "fork")  # both sides run to their end nodes
    lo = [float(proj(probe, xy[sides[k][0].start])) for k in (0, 1)] if probe.ends[0] in whole else [x0]
    hi = [float(proj(probe, xy[sides[k][-1].end])) for k in (0, 1)] if probe.ends[1] in whole else [x1]
    probe.span = (min(lo + [x0]), max(hi + [x1]))
    probe.paths = tuple(e[3] for e in ends)
    probe.forks = tuple(e[4] for e in ends)
    probe.key = tuple(sorted(members(probe)))
    return probe


def members(c: Corridor) -> set:
    """Every piece a corridor lays out: its sides, the pieces of its bends and the first pieces past its forks."""
    return {sp.piece for side in c.sides for sp in side} | {p for path in c.paths for p in path} | \
        {p for fork in c.forks for p in fork}


def _banned(skip: set) -> set:
    """Pieces of the pair corridors in ``skip`` (dropped corridors): no other pair corridor may use them."""
    return {int(p) for key in skip if key[:1] != ("boundary",) for p in key}


def pair_corridors(net, size_fn, domain, standard: list, features: np.ndarray, skip: set) -> list:
    """Corridors between two close, nearly parallel barrier strands (each piece in one corridor at most)."""
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
            candidates.append((line.distance(lines[pb]), pa, pb))
    used, out = set(_banned(skip)), []
    for _, pa, pb in sorted(candidates):
        if pa in used or pb in used:
            continue
        c = _pair_corridor(net, pa, pb, size_fn, domain, obstacles, degree, rung_nodes, rung_pieces, used)
        if c is None or members(c) & used:
            continue
        out.append(c)
        used |= members(c)
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
    c.gap = min(float(np.hypot(*(p - f))) for p, f in zip(pts, feet))
    return c


def gap_field_wins(c: Corridor, size_fn) -> bool:
    """True when GapField (``size_fn.gap``) asks for more than one cell across the gap of ``c``.

    Its cells then go across the gap instead of the corridor's one row: the
    barriers there get their own faces, sized by GapField.
    """
    gap = getattr(size_fn, "gap", None)
    return gap is not None and gap.cells_across(c.region) > 1.0


def find_corridors(net, size_fn, domain, standard: list, features: np.ndarray, skip: set) -> list:
    """Pair corridors first, then boundary runs on the pieces left; none where GapField wins."""
    pairs = [c for c in pair_corridors(net, size_fn, domain, standard, features, skip)
             if not gap_field_wins(c, size_fn)]
    used = {sp.piece for c in pairs for side in c.sides for sp in side}
    runs = boundary_runs(net, size_fn, domain, standard, features, skip, used)
    return pairs + [c for c in runs if not gap_field_wins(c, size_fn)]


def bent_rungs(net, corridors: list) -> list:
    """The ladders of ``net`` whose two legs lie on the bend of a corridor (the bend replaces them)."""
    bent = {p for c in corridors for path in c.paths for p in path}
    return [r for r in net.rungs if r.t0 == 0 and set(int(p) for p in r.pieces) <= bent]


def chord_nodes(net, c: Corridor) -> set:
    """Hard nodes on an end chord of ``c`` (except free tips), whose radius the corridor sets.

    The side vertices the corridor passes are handled in ``sites`` (a hard
    node whose segments all have explicit sites gets no sites of its own);
    the kinks along a bend are chord nodes too.
    """
    out = {int(v) for kind, nodes in zip(c.ends, c.end_nodes) if kind != "tip" for v in nodes}
    return out | {int(v) for path in c.paths for p in path for v in net.pieces[p]}


def contains(c: Corridor, piece: int, xy: np.ndarray) -> np.ndarray:
    """Mask of the samples ``xy`` of ``piece`` that lie in the corridor's stretch (all of a bend's)."""
    if any(piece in path for path in c.paths):
        return np.ones(len(xy), dtype=bool)
    if all(sp.piece != piece for side in c.sides for sp in side):
        return np.zeros(len(xy), dtype=bool)
    x = proj(c, xy)
    lo, hi = c.span if c.span is not None else (c.x0, c.x1)
    tol = 1e-9 * max(1.0, abs(hi - lo))
    return (x >= lo - tol) & (x <= hi + tol)


# ---------------------------------------------------------------------------
# Marching layout
# ---------------------------------------------------------------------------
def _length(net, sp: SidePiece) -> float:
    """Length of a side piece."""
    return float(np.hypot(*np.subtract(*_piece_xy(net, sp))))


def local_m(c: Corridor, net, idx: list) -> tuple:
    """(origin, m) of M for the current side pieces ``idx``: their bisector, or the boundary edge."""
    if len(c.sides) == 1:
        return c.origin, c.m
    a0, a1 = _piece_xy(net, c.sides[0][idx[0]])
    b0, b1 = _piece_xy(net, c.sides[1][idx[1]])
    return _bisector(a0, a1, b0, b1)


def _hit(net, sp: SidePiece, point: np.ndarray, direction: np.ndarray) -> float:
    """t along ``sp`` (from its first node along M) where the line through ``point`` along ``direction`` meets it."""
    p0, p1 = _piece_xy(net, sp)
    u = p1 - p0
    den = u[0] * direction[1] - u[1] * direction[0]
    if abs(den) < 1e-15 * np.hypot(*u):
        return math.nan
    w = point - p0
    return float((w[0] * direction[1] - w[1] * direction[0]) / den * np.hypot(*u))


def _h_at(net, field, sp: SidePiece, t: float) -> float:
    """Target spacing h on side piece ``sp`` at ``t`` from its first node along M."""
    sel = field.piece == sp.piece
    s = t if net.pieces[sp.piece][0] == sp.start else _length(net, sp) - t
    return float(np.interp(s, field.s[sel], field.h[sel]))


def _gap_and_h(c: Corridor, net, field, idx: list, point: np.ndarray, m: np.ndarray) -> tuple:
    """(gap across M at ``point``, min h on the sides there); a boundary run's gap is twice its distance."""
    n = _rot90(m)
    feet, hs = [], []
    for side, i in zip(c.sides, idx):
        sp = side[i]
        t = float(np.clip(_hit(net, sp, point, n), 0.0, _length(net, sp)))
        p0, p1 = _piece_xy(net, sp)
        feet.append(p0 + t / _length(net, sp) * (p1 - p0))
        hs.append(_h_at(net, field, sp, t))
    gap = np.hypot(*(feet[0] - feet[1])) if len(feet) == 2 else np.hypot(*(feet[0] - point))
    return float(gap), min(hs)


def _end_targets(c: Corridor, net, end: int) -> np.ndarray:
    """Coordinates of the hard nodes the corridor ends on at ``end``."""
    return net.hard_xy[list(c.end_nodes[end])]


def _end_position(c: Corridor, net, end: int, o: np.ndarray, m: np.ndarray) -> float:
    """Where along (o, m) the corridor's sites must stop (x1 end) or start (x0 end)."""
    x = (_end_targets(c, net, end) - o) @ m
    return float(x.min() if end == 1 else x.max())


def _end_chord(c: Corridor, net, end: int, idx: list) -> tuple:
    """(point, direction) of the end chord at ``end``: tilted from the free end at a stagger, else across M."""
    o, m = local_m(c, net, idx)
    n = _rot90(m)
    x = _end_position(c, net, end, o, m)
    if c.ends[end] != "stagger":
        return o + x * m, n
    a = net.hard_xy[c.end_nodes[end][0]]
    k = next(k for k, side in enumerate(c.sides) if c.end_nodes[end][0] in (side[0].start, side[-1].end))
    sp = c.sides[1 - k][idx[1 - k]]
    t = _hit(net, sp, a, n)
    p0, p1 = _piece_xy(net, sp)
    other = p0 + t / _length(net, sp) * (p1 - p0)
    return a, _unit(_unit(other - a) + math.tan(STAGGER_TILT) * (m if end == 1 else -m))


def _circle_on_line(v: np.ndarray, r: float, o: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """The forward point of the circle (v, r) on the line (o, m), or None when they miss."""
    b = float(m @ (o - v))
    disc = b * b - float((o - v) @ (o - v)) + r * r
    if disc < 0:
        return None
    return o + (-b + math.sqrt(disc)) * m


def _chord_node(net, side: list, i: int, t_prev: float, point: np.ndarray, direction: np.ndarray) -> float | None:
    """t on side piece ``i`` where a chord crosses it, or None when it falls outside the piece or goes back."""
    sp = side[i]
    t = _hit(net, sp, point, direction)
    length = _length(net, sp)
    tol = 1e-9 * max(1.0, length)
    if not math.isfinite(t) or t <= t_prev + tol or t >= length - tol:
        return None
    return t


class _March:
    """The sites and chord nodes of a corridor, laid out from start to end (see ``lay_out``)."""

    def __init__(self, c: Corridor, net, field, aspect: float):
        self.c, self.net, self.field, self.aspect = c, net, field, aspect
        self.idx = [0] * len(c.sides)          # current piece of every side
        self.last_t = [0.0] * len(c.sides)     # t of the last chord node on it
        self.sites = []
        self.nodes = [[] for _ in c.sides]     # per side: (side piece index, t) per chord

    def spacing(self, point: np.ndarray, m: np.ndarray) -> float:
        """L at ``point``: min(h on the sides, aspect x gap)."""
        gap, h = _gap_and_h(self.c, self.net, self.field, self.idx, point, m)
        return min(h, self.aspect * gap)

    def vertices(self) -> list:
        """(side, coordinates) of the next vertex of every side that has one."""
        return [(k, self.net.hard_xy[side[self.idx[k]].end]) for k, side in enumerate(self.c.sides)
                if self.idx[k] < len(side) - 1]

    def add_chord(self, a: np.ndarray, b: np.ndarray, forced: dict) -> bool:
        """Record the chord between sites a and b (its nodes on every side); False when it does not fit."""
        mid, direction = 0.5 * (a + b), _rot90(_unit(b - a))
        found = {}
        for k, side in enumerate(self.c.sides):
            i = self.idx[k]
            found[k] = _length(self.net, side[i]) if k in forced else \
                _chord_node(self.net, side, i, self.last_t[k], mid, direction)
            if found[k] is None:
                return False
        for k, t in found.items():
            self.nodes[k].append((self.idx[k], t))
            self.last_t[k] = t
            if k in forced:
                self.idx[k] += 1
                self.last_t[k] = 0.0
        return True

    def start(self) -> bool:
        """The start chord's nodes and the first site."""
        c, net = self.c, self.net
        o, m = local_m(c, net, self.idx)
        x0 = _end_position(c, net, 0, o, m)
        point, direction = _end_chord(c, net, 0, self.idx)
        for k, side in enumerate(c.sides):
            if side[0].start in c.end_nodes[0]:
                self.nodes[k].append((0, 0.0))
                continue
            t = _hit(net, side[0], point, direction)
            if not 0.0 < t < _length(net, side[0]):
                return False
            self.nodes[k].append((0, t))
            self.last_t[k] = t
        target = min([float((v - o) @ m) for _, v in self.vertices()] + [_end_position(c, net, 1, o, m)])
        if target <= x0:
            return False
        spacing = self.spacing(o + x0 * m, m)
        count = max(1, math.ceil((target - x0) / spacing - 1e-9))
        self.sites.append(o + (x0 + 0.5 * (target - x0) / count) * m)
        return True

    def step(self) -> str:
        """Place the next site: "more", "done" or "fail"."""
        c, net = self.c, self.net
        sigma = self.sites[-1]
        o, m = local_m(c, net, self.idx)
        s = float((sigma - o) @ m)
        spacing = self.spacing(sigma, m)
        if not spacing > 0:
            return "fail"
        end = _end_position(c, net, 1, o, m)
        events = [(float((v - o) @ m), k, v) for k, v in self.vertices()]
        events.append((end, None, None))
        p, k, v = min(events, key=lambda e: e[0])
        count = max(1, math.ceil((p - s) / spacing + 0.5 - 1e-9))  # steps of at most the spacing
        previous = self.last_step()
        if previous is not None:  # grow, or shrink, by at most MAX_GROWTH per step
            while (p - s) / max(count - 0.5, 0.5) > MAX_GROWTH * previous and count < MAX_STRIPS:
                count += 1
            while count > 1 and (p - s) / (count - 0.5) < previous / MAX_GROWTH:
                count -= 1
        if k is None and count <= 1:
            return "done"
        if k is not None and count <= 1:
            return self.forced(k, v)
        nxt = o + (s + (p - s) / (count - 0.5)) * m
        for k2, v2 in self.vertices():
            if np.hypot(*(v2 - sigma)) <= np.hypot(*(v2 - nxt)):
                return self.forced(k2, v2)
        return self.place(nxt, {})

    def last_step(self) -> float | None:
        """Distance between the last two sites, or None before the second site."""
        if len(self.sites) >= 2:
            return float(np.hypot(*(self.sites[-1] - self.sites[-2])))
        return None

    def forced(self, k: int, v: np.ndarray) -> str:
        """The next site such that the chord passes through vertex ``v`` of side ``k``."""
        sigma = self.sites[-1]
        after = list(self.idx)
        after[k] += 1
        o, m = local_m(self.c, self.net, after)
        nxt = _circle_on_line(v, float(np.hypot(*(v - sigma))), o, m)
        if nxt is None:
            return "fail"
        forced = {k: v}
        for k2, v2 in self.vertices():
            if k2 != k and np.hypot(*(v2 - sigma)) <= np.hypot(*(v2 - nxt)):
                nxt = _reflect(sigma, v, v2)
                forced[k2] = v2
        return self.place(nxt, forced)

    def place(self, nxt: np.ndarray, forced: dict) -> str:
        """Add site ``nxt`` after the last one, if the step is forward and in proportion."""
        sigma = self.sites[-1]
        o, m = local_m(self.c, self.net, self.idx)
        step = float(np.hypot(*(nxt - sigma)))
        if (nxt - sigma) @ m <= 0 or len(self.sites) > MAX_STRIPS:
            return "fail"
        previous = self.last_step()
        if previous is not None and not 1.0 / MAX_STEP_RATIO <= step / previous <= MAX_STEP_RATIO:
            return "fail"
        if not self.add_chord(sigma, nxt, forced):
            return "fail"
        self.sites.append(nxt)
        return "more"

    def finish(self) -> bool:
        """The end chord's nodes."""
        c, net = self.c, self.net
        point, direction = _end_chord(c, net, 1, self.idx)
        for k, side in enumerate(c.sides):
            i = self.idx[k]
            if side[-1].end in c.end_nodes[1]:
                if i != len(side) - 1:
                    return False
                self.nodes[k].append((i, _length(net, side[i])))
                continue
            t = _chord_node(net, side, i, self.last_t[k], point, direction)
            if t is None:
                return False
            self.nodes[k].append((i, t))
        return True


MAX_STEP_RATIO = 2.5    # consecutive site spacings differ by at most this factor
MAX_GROWTH = 1.5        # the march lengthens or shortens its steps by at most this factor
MAX_STRIPS = 5000


def march(c: Corridor, net, field, aspect: float = CORRIDOR_ASPECT) -> _March | None:
    """Sites and chord nodes of ``c`` from its start to its end, or None when they do not fit.

    From the start chord, each next site goes on M (the bisector of the
    current side pieces) at the local spacing, evened out so that the sites
    straddle the next event (a side vertex or the end) symmetrically. A side
    vertex is a forced chord: the next site is where the circle about the
    vertex through the last site meets the next M, so the perpendicular
    bisector of the two sites passes through the vertex.
    """
    run = _March(c, net, field, aspect)
    if not run.start():
        return None
    while True:
        state = run.step()
        if state == "fail":
            return None
        if state == "done":
            return run if run.finish() else None


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
    run = march(c, net, field, aspect)
    if run is None:
        return None
    sites = run.sites
    runs = []
    for k, side in enumerate(c.sides):
        nodes = list(run.nodes[k])
        seg_sites = list(sites)
        for end, kind in enumerate(c.ends):
            ending = {side[0].start, side[-1].end} & set(c.end_nodes[end])
            if kind != "stagger" or ending:
                continue
            point, direction = _end_chord(c, net, end, [0] * len(c.sides) if end == 0 else run.idx)
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
    for end in (0, 1):
        sigma = sites[0] if end == 0 else sites[-1]
        if c.ends[end] == "bend":
            runs += [(p, np.array([0.0, float(np.hypot(*np.subtract(*net.hard_xy[net.pieces[p]])))]), [sigma])
                     for p in c.paths[end]]
        if c.ends[end] == "fork":
            fork = _fork_runs(c, net, end, sigma)
            if fork is None:
                return None
            runs += fork
    return CorridorLayout(runs, [])


def _fork_runs(c: Corridor, net, end: int, sigma: np.ndarray) -> list | None:
    """Runs of the first segment of each piece past a fork: its site is the mirror of the end site ``sigma``
    across the line through the two ending kinks, and the segment is twice the distance to that site's foot."""
    a, b = (net.hard_xy[v] for v in c.end_nodes[end])
    beyond = _reflect(sigma, a, b)
    out = []
    for v, p in zip(c.end_nodes[end], c.forks[end]):
        far = _far_node(net, p, v)
        length = float(np.hypot(*(net.hard_xy[far] - net.hard_xy[v])))
        foot = float((beyond - net.hard_xy[v]) @ _unit(net.hard_xy[far] - net.hard_xy[v]))
        if foot <= 0 or 2 * foot > length - CORRIDOR_FOOT * foot:
            return None
        t = np.array([0.0, 2 * foot]) if v == net.pieces[p][0] else np.array([length - 2 * foot, length])
        out.append((p, t, [beyond]))
    return out
