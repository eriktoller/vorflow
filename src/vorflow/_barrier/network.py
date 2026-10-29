"""Barrier network: hard nodes, pieces, stubs, sharp spikes, and the sector plan at every hard node.

*Hard nodes* are the vertices of the cleaned barrier lines (ends, junctions,
crossings, turning vertices). *Pieces* are the straight runs between them.
At a hard node the *legs* are its pieces, the two domain-boundary rays when
the node is on the boundary, the rays of a standard line through it, and a
tangent ray at a free tip. Consecutive legs bound *sectors*.

Before the plan, two things are removed from the lines the barrier faces
must follow (both are reported as uncovered length):

- *stubs*: a dangling piece shorter than STUB_FRACTION x lc past a junction
  (an overshoot left by digitising);
- *sharp spike tips*: where a line folds back at less than SHARP_ANGLE, the
  last TIP_FRACTION x lc of both legs. The two new ends are joined by a
  ladder (see ``sites.rung_geometry``) so the cells stay regular; the tip
  itself lies inside cells that straddle it.

Between the two, barrier legs meeting a junction or the boundary at less
than SHARP_ANGLE are merged over a short reach (``wedge``), which moves the
lines a bounded distance and is reported.

Sector plan (per node). A barrier leg's first pair of sites sits at +-phi
from the leg on the node's circle.

- *rung*: a sector between two barrier legs narrower than RUNG_ANGLE gets a
  ladder of sites on its bisector (one row of cells across the wedge);
  phi = theta / 2 on both legs. Narrowest first; a leg takes part in one
  rung only.
- *shared*: the two sites of a sector become one, phi_a + phi_b = theta. A
  boundary or standard-line ray counts with phi = 0, so its shared site
  lies on the boundary or on the standard line (band's side-share).
- *split*: the two sites keep an empty arc of at least GAP_SPLIT
  (GAP_SPLIT_RAY next to a ray); a wide arc gets fill sites.

The shared set is chosen by trying every subset of the candidate sectors
(at most MAX_SUBSET of them; greedy beyond) and solving the phis closest to
PHI0 under the plan's constraints (shared equalities, split gaps, phi
bounds; Dykstra's alternating projections): the plan with the least
violation wins, then the smallest sum (phi - PHI0)^2 plus (GAP_GOOD -
gap)^2 for every split gap between two barrier legs narrower than GAP_GOOD
(two sites that close make two thin cells). A node *prescribes* its legs' first sites when
it has a shared or rung sector, three or more barrier legs, or lies on the
boundary.
"""
from __future__ import annotations

import dataclasses
import functools
import itertools
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from .lines import BarrierLine
from .wedge import collapse_wedges

COLLINEAR_SIN = 1e-9
PHI0 = math.radians(45.0)
PHI_MIN = math.radians(8.0)
PHI_MAX = math.radians(70.0)
GAP_SPLIT = math.radians(30.0)
GAP_SPLIT_RAY = math.radians(15.0)
GAP_GOOD = math.radians(45.0)        # a narrower split gap between two barrier legs costs (GAP_GOOD - gap)^2
SHARE_MAX = math.radians(150.0)
SIDE_SHARE_MAX = math.radians(70.0)  # a leg-ray sector narrower than this may put the leg's site on the ray
RAY_MIN = math.radians(25.0)         # a standard-line ray closer than this to a barrier leg is ignored
FILL_ANGLE = math.radians(100.0)
RUNG_ANGLE = math.radians(60.0)
SHARP_ANGLE = math.radians(18.0)
TIP_FRACTION = 0.45   # x lc: length of each leg of a sharp spike left without barrier faces
TIP_LEG_FRACTION = 0.3  # ... at most this x the leg
STUB_FRACTION = 0.1   # x lc: shorter dangling pieces past a junction are dropped
MAX_SUBSET = 10
DYKSTRA_TOL = 1e-10  # radians: a sweep that moves the phis less than this ends the projections
TWO_PI = 2 * math.pi


@dataclasses.dataclass
class Leg:
    """A leg at a hard node: kind "piece", "boundary", "ray" or "tip"."""
    angle: float
    kind: str
    piece: int = -1
    phi: float = PHI0
    fixed: bool = False


@dataclasses.dataclass
class Rung:
    """A ladder between two straight legs from ``apex`` (directions u_a, u_b, opening ``theta``).

    ``nodes``: the hard node where each leg's piece starts (the apex node, or
    for a truncated spike the two new ends ``t0`` from the apex);
    ``pieces``: the two pieces.
    """
    apex: np.ndarray
    u_a: np.ndarray
    u_b: np.ndarray
    theta: float
    t0: float
    nodes: tuple
    pieces: tuple


@dataclasses.dataclass
class NodePlan:
    """Legs (sorted by angle), sectors, the shared set and derived flags of one hard node."""
    legs: list
    sectors: list
    shared: list
    prescribes: bool
    on_boundary: bool
    tip_direction: np.ndarray | None
    fill_directions: list
    rung_end: bool = False
    violation: float = 0.0


@dataclasses.dataclass
class Network:
    """Hard nodes, pieces, plans and rungs of all barriers."""
    hard_xy: np.ndarray
    pieces: np.ndarray
    piece_lc: np.ndarray
    plans: list
    rungs: list
    lines: list          # the barrier polylines the faces follow (after stubs and spike tips)
    uncovered: list      # (kind, length, xy, line_id) of barrier parts left without faces
    corridors: list = dataclasses.field(default_factory=list)  # see ``corridor``
    collapsed: list = dataclasses.field(default_factory=list)  # acute wedges merged (``wedge``)

    @functools.cached_property
    def node_pieces(self) -> list:
        """The pieces at each hard node, in piece order (``pieces`` never changes after construction)."""
        out = [[] for _ in range(len(self.hard_xy))]
        for p, (a, b) in enumerate(self.pieces):
            out[int(a)].append(p)
            if b != a:
                out[int(b)].append(p)
        return out


# ---------------------------------------------------------------------------
# Lines -> nodes and pieces
# ---------------------------------------------------------------------------
def drop_collinear(xy: np.ndarray, keep: np.ndarray) -> tuple:
    """(xy, keep) without vertices where the line goes straight on (unless must-keep)."""
    out = [0]
    for i in range(1, len(xy) - 1):
        a, b = xy[i] - xy[out[-1]], xy[i + 1] - xy[i]
        sin = (a[0] * b[1] - a[1] * b[0]) / (np.hypot(*a) * np.hypot(*b))
        if keep[i] or abs(sin) > COLLINEAR_SIN or np.dot(a, b) < 0:
            out.append(i)
    out.append(len(xy) - 1)
    return xy[out], keep[out]


def nodes_and_pieces(lines: list) -> tuple:
    """(hard_xy, pieces, piece_lc) from barrier lines whose shared vertices have identical coordinates."""
    index, hard, pieces = {}, [], {}
    for b in lines:
        ids = []
        for p in map(tuple, b.xy):
            if p not in index:
                index[p] = len(hard)
                hard.append(p)
            ids.append(index[p])
        for i, j in zip(ids[:-1], ids[1:]):
            key = (min(i, j), max(i, j))
            if i != j:
                pieces[key] = ((i, j), min(pieces[key][1], b.lc) if key in pieces else b.lc)
    items = list(pieces.values())
    return (np.array(hard, dtype=float).reshape(-1, 2), np.array([p for p, _ in items], dtype=int).reshape(-1, 2),
            np.array([lc for _, lc in items], dtype=float))


def _degrees(pieces: np.ndarray, n: int) -> np.ndarray:
    """Number of pieces at each node."""
    return np.bincount(pieces.ravel(), minlength=n)


def _on_boundary(domain, xy: np.ndarray, tol: float) -> np.ndarray:
    """Mask of points within ``tol`` of the domain boundary."""
    return shapely.distance(domain.boundary, shapely.points(xy)) <= tol


def drop_stubs(lines: list, domain, tol: float, uncovered: list) -> list:
    """Remove dangling end pieces shorter than STUB_FRACTION x lc whose other end is a junction."""
    while True:
        hard, pieces, _ = nodes_and_pieces(lines)
        degree = _degrees(pieces, len(hard))
        boundary = _on_boundary(domain, hard, tol)
        changed = False
        for b in lines:
            for end, nxt in ((0, 1), (-1, -2)):
                if len(b.xy) < 2:
                    continue
                e, w = tuple(b.xy[end]), tuple(b.xy[nxt])
                ie, iw = _node_id(hard, e), _node_id(hard, w)
                length = float(np.hypot(*np.subtract(e, w)))
                if degree[ie] == 1 and not boundary[ie] and degree[iw] >= 3 and length < STUB_FRACTION * b.lc:
                    uncovered.append(("stub", length, np.array(e), b.line_id))
                    b.xy = b.xy[1:] if end == 0 else b.xy[:-1]
                    b.keep = b.keep[1:] if end == 0 else b.keep[:-1]
                    changed = True
                    break
            if changed:
                break
        lines = [b for b in lines if len(b.xy) >= 2]
        if not changed:
            return lines


def _node_id(hard: np.ndarray, p: tuple) -> int:
    """Index of the hard node at exactly ``p``."""
    return int(np.flatnonzero((hard[:, 0] == p[0]) & (hard[:, 1] == p[1]))[0])


def _sector_between(u: np.ndarray, w: np.ndarray) -> float:
    """Angle between two unit vectors."""
    return math.acos(float(np.clip(u @ w, -1.0, 1.0)))


def truncate_spikes(lines: list, standard: list, domain, tol: float, uncovered: list,
                    sharp_angle: float = SHARP_ANGLE) -> tuple:
    """(lines, spikes): lines with the tips of folds sharper than SHARP_ANGLE removed.

    ``spikes`` lists (apex, u_a, u_b, theta, t0, a_xy, b_xy) for the ladders
    that join the two new ends.
    """
    hard, pieces, lc = nodes_and_pieces(lines)
    degree = _degrees(pieces, len(hard))
    boundary = _on_boundary(domain, hard, tol)
    std = shapely.union_all(standard) if standard else None
    spikes, cut = [], {}
    for v in np.flatnonzero(degree == 2):
        if boundary[v] or (std is not None and std.distance(Point(hard[v])) <= tol):
            continue
        ps = np.flatnonzero((pieces == v).any(axis=1))
        others = [pieces[p][1] if pieces[p][0] == v else pieces[p][0] for p in ps]
        vec = hard[others] - hard[v]
        length = np.hypot(*vec.T)
        u = vec / length[:, None]
        theta = _sector_between(u[0], u[1])
        if theta >= sharp_angle:
            continue
        t0 = min(TIP_FRACTION * float(lc[ps].min()), TIP_LEG_FRACTION * float(length.min()))
        a, b = hard[v] + t0 * u[0], hard[v] + t0 * u[1]
        cut[tuple(hard[v])] = {tuple(hard[others[0]]): a, tuple(hard[others[1]]): b}
        spikes.append((hard[v].copy(), u[0], u[1], theta, t0, a, b))
        apex_line = next(b.line_id for b in lines if (b.xy == hard[v]).all(axis=1).any())
        uncovered.append(("spike tip", 2 * t0, hard[v].copy(), apex_line))
    if not cut:
        return lines, spikes
    out = []
    for line in lines:
        out += _cut_line(line, cut)
    return out, spikes


def _cut_line(line: BarrierLine, cut: dict) -> list:
    """``line`` split at spike apexes, each new end moved to its truncation point."""
    parts, start = [], 0
    original, xy, keep = line.xy, line.xy.copy(), line.keep
    for i in range(len(xy)):
        apex = tuple(original[i])
        if apex not in cut:
            continue
        ends = cut[apex]
        if i > start:  # part ending at the apex: its last vertex becomes the truncation point
            seg = xy[start:i + 1].copy()
            seg[-1] = ends[tuple(original[i - 1])]
            parts.append((seg, keep[start:i + 1].copy()))
        start = i
        if i < len(xy) - 1:  # the next part starts at the truncation point on the following leg
            xy[i] = ends[tuple(original[i + 1])]
    if len(xy) - start >= 2:
        parts.append((xy[start:], keep[start:].copy()))
    out = []
    for k, (pxy, pkeep) in enumerate(parts):
        pkeep[[0, -1]] = True
        out.append(BarrierLine(line.line_id, pxy, line.lc, line.tol, pkeep))
    return out


# ---------------------------------------------------------------------------
# Legs and plans
# ---------------------------------------------------------------------------
def _boundary_rays(domain, xy: np.ndarray, tol: float) -> list:
    """Angles of the domain boundary leaving the boundary point ``xy``.

    They are measured from the foot of ``xy`` on the boundary, not from
    ``xy``: a node up to ``tol`` off the boundary would tilt rays over the
    short probe by up to tol / step, and a shared site on such a ray falls
    outside the domain.
    """
    rings = [LineString(r.coords) for p in getattr(domain, "geoms", [domain]) for r in [p.exterior, *p.interiors]]
    ring = min(rings, key=lambda r: r.distance(Point(xy)))
    d = ring.project(Point(xy))
    foot = np.array(ring.interpolate(d).coords[0])
    step = max(tol * 1e3, 1e-6 * ring.length)
    out = []
    for dd in ((d - step) % ring.length, (d + step) % ring.length):
        p = np.array(ring.interpolate(dd).coords[0])
        out.append(math.atan2(p[1] - foot[1], p[0] - foot[0]))
    return out


def _standard_rays(standard: list, xy: np.ndarray, tol: float) -> list:
    """Angles of the standard lines leaving ``xy`` (two where one passes through, one where one ends)."""
    out = []
    for line in standard:
        if line.distance(Point(xy)) > tol:
            continue
        d = line.project(Point(xy))
        step = max(1e3 * tol, 1e-6 * line.length)
        for dd in (d - step, d + step):
            if 0.0 <= dd <= line.length:
                p = np.array(line.interpolate(dd).coords[0])
                out.append(math.atan2(p[1] - xy[1], p[0] - xy[0]))
    return out


def _angle_gap(a: float, b: float) -> float:
    """Smallest absolute difference between two angles."""
    return abs((a - b + math.pi) % TWO_PI - math.pi)


def node_legs(node: int, hard: np.ndarray, pieces: np.ndarray, domain, standard: list, tol: float,
              rung_end: bool) -> tuple:
    """(legs sorted by angle, on_boundary, tip direction or None)."""
    xy = hard[node]
    legs = []
    for p in np.flatnonzero((pieces == node).any(axis=1)):
        other = hard[pieces[p][1] if pieces[p][0] == node else pieces[p][0]]
        legs.append(Leg(math.atan2(other[1] - xy[1], other[0] - xy[0]), "piece", int(p)))
    on_boundary = bool(domain.boundary.distance(Point(xy)) <= tol)
    piece_angles = [leg.angle for leg in legs]
    if on_boundary:
        legs += [Leg(a, "boundary", phi=0.0, fixed=True) for a in _boundary_rays(domain, xy, tol)]
    else:
        for a in _standard_rays(standard, xy, tol):
            if min(_angle_gap(a, b) for b in piece_angles) >= RAY_MIN:
                legs.append(Leg(a, "ray", phi=0.0, fixed=True))
    tip = None
    if len(legs) == 1 and not rung_end:
        angle = legs[0].angle + math.pi
        tip = np.array([math.cos(angle), math.sin(angle)])
        legs.append(Leg(angle, "tip", phi=0.0, fixed=True))
    legs.sort(key=lambda leg: leg.angle % TWO_PI)
    return legs, on_boundary, tip


def sectors_of(legs: list, xy: np.ndarray, domain) -> list:
    """(leg i, leg i+1, CCW angle) for each sector inside the domain."""
    out = []
    n = len(legs)
    for k in range(n):
        a, b = legs[k], legs[(k + 1) % n]
        theta = TWO_PI if n == 1 else (b.angle - a.angle) % TWO_PI
        if a.kind == "boundary" and b.kind == "boundary":
            mid = a.angle + theta / 2
            probe = xy + 1e-6 * max(1.0, float(np.abs(xy).max())) * np.array([math.cos(mid), math.sin(mid)])
            if not domain.covers(Point(probe)):
                continue
        out.append((k, (k + 1) % n, theta))
    return out


def _constraints(legs: list, sectors: list, shared: list, skip: list, free: list) -> list:
    """(coefficients over the free legs, right-hand side, is_equality) of a plan's linear constraints."""
    col = {k: j for j, k in enumerate(free)}
    rows = []
    for a, b, theta in sectors:
        if (a, b, theta) in skip or "tip" in (legs[a].kind, legs[b].kind):
            continue
        if legs[a].kind == legs[b].kind == "boundary":
            continue
        is_shared = (a, b, theta) in shared
        gap = 0.0 if is_shared else (GAP_SPLIT_RAY if {"boundary", "ray"} & {legs[a].kind, legs[b].kind}
                                     else GAP_SPLIT)
        row, rhs = np.zeros(len(free)), theta - gap
        for k in (a, b):
            if k in col:
                row[col[k]] += 1.0
            else:
                rhs -= legs[k].phi
        if row.any():
            rows.append((row, rhs, is_shared))
    return rows


def _phi_solution(legs: list, shared: list, sectors: list = (), skip: list = (), sweeps: int = 300) -> dict:
    """{free piece leg: phi}: the phis closest to PHI0 under the plan's constraints (Dykstra's projections).

    Shared sectors: phi_a + phi_b = theta; split sectors: phi_a + phi_b <=
    theta - gap; PHI_MIN <= phi <= PHI_MAX. When they cannot all hold the
    result is approximate and ``_violation`` reports by how much.
    """
    free = [k for k, leg in enumerate(legs) if leg.kind == "piece" and not leg.fixed]
    rows = _constraints(legs, sectors, shared, skip, free)
    x = np.full(len(free), PHI0)
    if not free:
        return {}
    incr = [np.zeros(len(free)) for _ in range(len(rows) + 1)]
    for _ in range(sweeps):
        x_old = x.copy()
        for r, (row, rhs, eq) in enumerate(rows):
            y = x + incr[r]
            excess = row @ y - rhs
            if eq or excess > 0:
                y = y - row * excess / (row @ row)
            incr[r] = x + incr[r] - y
            x = y
        y = np.clip(x + incr[-1], PHI_MIN, PHI_MAX)
        incr[-1] = x + incr[-1] - y
        x = y
        if np.abs(x - x_old).max() < DYKSTRA_TOL:
            break
    return {k: float(v) for k, v in zip(free, x)}


def _leg_phi(legs: list, phis: dict, k: int) -> float:
    """phi of leg k under a candidate solution."""
    return phis.get(k, legs[k].phi)


def _violation(legs: list, sectors: list, shared: list, phis: dict, skip: list = ()) -> float:
    """Radians by which a plan breaks the phi bounds, the shared equalities or the split gaps."""
    total = sum(max(0.0, PHI_MIN - p) + max(0.0, p - PHI_MAX) for p in phis.values())
    for a, b, theta in sectors:
        if (a, b, theta) in skip:
            continue
        pa, pb = _leg_phi(legs, phis, a), _leg_phi(legs, phis, b)
        if (a, b, theta) in shared:
            total += abs(pa + pb - theta)
            continue
        if "tip" in (legs[a].kind, legs[b].kind):
            continue
        gap = GAP_SPLIT_RAY if {"boundary", "ray"} & {legs[a].kind, legs[b].kind} else GAP_SPLIT
        total += max(0.0, pa + pb + gap - theta)
    return total


def _candidates(legs: list, sectors: list, rung_sectors: list) -> list:
    """Sectors that may be shared."""
    out = []
    for a, b, theta in sectors:
        kinds = {legs[a].kind, legs[b].kind}
        if (a, b, theta) in rung_sectors or "tip" in kinds or theta >= SHARE_MAX:
            continue
        if legs[a].fixed and legs[b].fixed:
            continue
        if kinds & {"boundary", "ray"} and theta >= SIDE_SHARE_MAX:
            continue
        out.append((a, b, theta))
    return out


def _gap_cost(legs: list, sectors: list, shared: list, phis: dict, skip: list) -> float:
    """Sum of (GAP_GOOD - gap)^2 over the split sectors between two barrier legs with a gap below GAP_GOOD."""
    total = 0.0
    for a, b, theta in sectors:
        if (a, b, theta) in shared or (a, b, theta) in skip or legs[a].kind != "piece" or legs[b].kind != "piece":
            continue
        gap = theta - _leg_phi(legs, phis, a) - _leg_phi(legs, phis, b)
        total += max(0.0, GAP_GOOD - gap) ** 2
    return total


def choose_shared(legs: list, sectors: list, candidates: list, skip: list) -> tuple:
    """(shared sectors, violation) of the best plan (see module docstring)."""
    subsets = []
    if len(candidates) <= MAX_SUBSET:
        for r in range(len(candidates) + 1):
            subsets += list(itertools.combinations(candidates, r))
    else:  # greedy: narrowest first
        ordered = sorted(candidates, key=lambda s: s[2])
        subsets = [tuple(ordered[:r]) for r in range(len(ordered) + 1)]
    best, best_key = [], None
    for shared in subsets:
        phis = _phi_solution(legs, list(shared), sectors, skip)
        violation = _violation(legs, sectors, list(shared), phis, skip)
        cost = sum((p - PHI0) ** 2 for p in phis.values()) + _gap_cost(legs, sectors, list(shared), phis, skip)
        key = (violation > 1e-6, round(violation, 9), round(cost, 9))
        if best_key is None or key < best_key:
            best, best_key = list(shared), key
    return best, best_key[1]


def _fill_directions(legs: list, split: list) -> list:
    """Unit directions of fill sites spread over the empty arc of each wide split sector."""
    out = []
    for a, b, theta in split:
        if "tip" in (legs[a].kind, legs[b].kind) or (legs[a].kind == "boundary" and legs[b].kind == "boundary"):
            continue
        gap = theta - legs[a].phi - legs[b].phi
        count = int(math.floor(gap / FILL_ANGLE + 0.5)) - 1
        for j in range(1, count + 1):
            angle = legs[a].angle + legs[a].phi + j * gap / (count + 1)
            out.append(np.array([math.cos(angle), math.sin(angle)]))
    return out


def node_plan(node: int, hard: np.ndarray, pieces: np.ndarray, domain, standard: list, tol: float,
              rung_end: bool, used_legs: set, rungs: list) -> NodePlan:
    """Legs, rungs, shared sectors and phis at one hard node (``rungs`` gets the node's new rungs)."""
    legs, on_boundary, tip = node_legs(node, hard, pieces, domain, standard, tol, rung_end)
    sectors = sectors_of(legs, hard[node], domain)
    rung_sectors = []
    if not rung_end:
        narrow = [s for s in sectors if legs[s[0]].kind == legs[s[1]].kind == "piece" and s[2] < RUNG_ANGLE]
        for a, b, theta in sorted(narrow, key=lambda s: s[2]):
            if legs[a].piece in used_legs or legs[b].piece in used_legs:
                continue
            rung_sectors.append((a, b, theta))
            used_legs.update((legs[a].piece, legs[b].piece))
            for k in (a, b):
                legs[k].phi, legs[k].fixed = theta / 2, True
            ua = np.array([math.cos(legs[a].angle), math.sin(legs[a].angle)])
            ub = np.array([math.cos(legs[b].angle), math.sin(legs[b].angle)])
            rungs.append(Rung(hard[node].copy(), ua, ub, theta, 0.0, (node, node), (legs[a].piece, legs[b].piece)))
    n_piece = sum(leg.kind == "piece" for leg in legs)
    must = on_boundary or n_piece >= 3 or bool(rung_sectors)
    shared, violation = choose_shared(legs, sectors, _candidates(legs, sectors, rung_sectors), rung_sectors)
    prescribes = (must or bool(shared) or rung_end) and n_piece > 0
    fills = []
    if prescribes and not rung_end:
        for k, value in _phi_solution(legs, shared, sectors, rung_sectors).items():
            legs[k].phi = value
        all_shared = shared + rung_sectors
        fills = _fill_directions(legs, [s for s in sectors if s not in all_shared])
    return NodePlan(legs, sectors, shared + rung_sectors, prescribes, on_boundary, tip, fills, rung_end, violation)


def replan_chord_nodes(net: Network, nodes: set, domain, standard: list, tol: float) -> None:
    """Re-plan ``nodes`` as chord nodes of a corridor: no own sites, radius derived from the corridor's sites."""
    for v in sorted(nodes):
        net.plans[v] = node_plan(v, net.hard_xy, net.pieces, domain, standard, tol, True, set(), [])


def build_network(lines: list, standard: list, domain, tol: float, sharp_angle: float = SHARP_ANGLE) -> Network:
    """Network of the cleaned barrier ``lines`` (stubs dropped, sharp spikes truncated) with node plans.

    ``sharp_angle`` is in radians.
    """
    uncovered = []
    lines = drop_stubs([dataclasses.replace(b, xy=b.xy.copy(), keep=b.keep.copy()) for b in lines],
                       domain, tol, uncovered)
    lines = [dataclasses.replace(b, **dict(zip(("xy", "keep"), drop_collinear(b.xy, b.keep)))) for b in lines]
    on_boundary = functools.partial(_on_boundary, domain, tol=tol)
    lines, collapsed = collapse_wedges(lines, domain, tol, sharp_angle, on_boundary)
    lines, spikes = truncate_spikes(lines, standard, domain, tol, uncovered, sharp_angle)
    hard, pieces, lc = nodes_and_pieces(lines)
    rungs, rung_ends = [], set()
    for apex, ua, ub, theta, t0, a, b in spikes:
        na, nb = _node_id(hard, tuple(a)), _node_id(hard, tuple(b))
        pa = int(np.flatnonzero((pieces == na).any(axis=1))[0])
        pb = int(np.flatnonzero((pieces == nb).any(axis=1))[0])
        rungs.append(Rung(apex, ua, ub, theta, t0, (na, nb), (pa, pb)))
        rung_ends.update((na, nb))
    used = {p for r in rungs for p in r.pieces}
    plans = [node_plan(v, hard, pieces, domain, standard, tol, v in rung_ends, used, rungs)
             for v in range(len(hard))]
    return Network(hard, pieces, lc, plans, rungs, lines, uncovered, collapsed=collapsed)
