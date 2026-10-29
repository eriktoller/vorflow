"""Barrier sites in closed form from the 1D layout, and the validity check.

Layout. Every piece is cut into segments by 1D nodes: *fixed runs* (ladder
chords at a rung end, corridor chords anywhere along the piece), and equal
steps of the integral of 1/h in the stretches between them. Every 1D node
has a circle; every segment gets a *pair* of sites, mirror images across
it, on both end circles:

- regular segment: the two intersections of its end circles;
- first segment at a prescribing node v: ``v + R_v rot(leg, +-phi)``, and
  the far node's radius is derived through it (UPR's shrink);
- single-segment piece between two prescribing nodes: where the two
  prescribed rays meet; both radii derived;
- ladder or corridor segment: the explicit site and its mirror; the
  chord nodes' radii are derived.

Free radii start at F x the mean adjacent segment length and are made
consistent on Q = R^2 (ccvt): each regular pair's foot lies in the middle
(1 - 2 FOOT) of its segment and its half-offset is at least OFFSET x d.
Tips (free ends) and fills (wide empty arcs at prescribing nodes) get a
site on the node circle; a truncated spike's tip site is the mirror of its
first ladder site across the truncation chord, a corridor's that of its
last site across the end chord.

The validity check of the result is in ``validate``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial import cKDTree

from .network import Network
from .rung import ladder_points
from .sizing import SizeField, integral, junction_radius, piece_lengths, raw_count

F = 1.0 / math.sqrt(2.0)
FOOT = 0.3
OFFSET = 0.3
H_MIN = 0.05
ON_CIRCLE_EPS = 1e-7
UNIQUE_FRACTION = 1e-9


@dataclasses.dataclass
class Layout:
    """1D nodes (hard nodes first), segments and the closed-form sites."""
    xy: np.ndarray
    radius: np.ndarray
    segments: np.ndarray
    seg_piece: np.ndarray
    pairs: np.ndarray        # (m, 2, 2): the two sites of each segment
    pair_ok: np.ndarray      # (m,) pair shape within limits
    extra: np.ndarray        # tip, fill and truncated-tip sites
    extra_node: np.ndarray   # node whose circle each extra site is on (-1 for truncated tips)
    sites: np.ndarray        # unique sites
    seg_size: np.ndarray     # (m,) segment lengths


def _rot(u: np.ndarray, angle: float) -> np.ndarray:
    """``u`` rotated CCW by ``angle``."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def _positions(field: SizeField, piece: int, lo: float, hi: float, count: int) -> np.ndarray:
    """``count - 1`` interior positions in [lo, hi] at equal steps of the integral of 1/h."""
    sel = field.piece == piece
    s, h = field.s[sel], field.h[sel]
    grid = np.linspace(lo, hi, 64)
    inv = 1.0 / np.interp(grid, s, h)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(grid))])
    return np.interp(cum[-1] * np.arange(1, count) / count, cum, grid)


def _count(field: SizeField, piece: int, lo: float, hi: float) -> int:
    """round of the integral of 1/h over [lo, hi], at least 1."""
    sel = field.piece == piece
    s, h = field.s[sel], field.h[sel]
    grid = np.linspace(lo, hi, 64)
    return max(1, int(round(integral(1.0 / np.interp(grid, s, h), grid))))


def _ladder_prefix(net: Network, field: SizeField) -> dict:
    """{(piece, node): (chord distances from node, ladder sites, (y, sigma, rung))} for every ladder leg."""
    out = {}
    for rung, (y, sigma) in zip(net.rungs, field.ladders):
        ca, cb, pts = ladder_points(rung.apex, rung.u_a, rung.u_b, rung.theta, y, sigma)
        t = y / math.cos(rung.theta / 2) - rung.t0
        if rung.t0 == 0:
            t = np.concatenate([[0.0], t])
        for (p, n) in zip(rung.pieces, rung.nodes):
            out[(p, n)] = (t, pts, rung)
    return out


def fixed_runs(net: Network, field: SizeField) -> tuple:
    """({piece: [(t ascending from the piece's first node, explicit sites of the segments between)]}, extra sites).

    Ladders give a run at one end of each leg (and a truncated spike's tip
    site); corridors give runs anywhere along a piece (and tip sites).
    Extra sites are (point, node whose circle it is on, -1 for none).
    """
    lengths = piece_lengths(net)
    runs, extra = {}, []
    for (p, node), (t, pts, rung) in _ladder_prefix(net, field).items():
        k0 = 0 if rung.t0 == 0 else 1
        sites = [pts[j + k0] for j in range(len(t) - 1)]
        if node == net.pieces[p][0]:
            runs.setdefault(p, []).append((np.asarray(t, dtype=float), sites))
        else:
            runs.setdefault(p, []).append((lengths[p] - np.asarray(t, dtype=float)[::-1], sites[::-1]))
        if rung.t0 > 0 and node == rung.nodes[0]:
            extra.append((pts[0], -1))
    for lay in field.corridors:
        for p, t, sites in lay.runs:
            runs.setdefault(p, []).append((np.asarray(t, dtype=float), list(sites)))
        extra += lay.extra
    for p, items in runs.items():
        items.sort(key=lambda r: r[0][0])
        for (ta, _), (tb, _) in zip(items[:-1], items[1:]):
            assert ta[-1] <= tb[0] + 1e-9 * lengths[p], f"two fixed runs overlap on piece {p}"
    return runs, extra


def _end_kind(net: Network, piece: int, node: int) -> str:
    """"rung", "single" or "free": what fixes the first segment of ``piece`` at hard ``node`` (no run there)."""
    plan = net.plans[node]
    if plan.rung_end:
        return "rung"
    return "single" if plan.prescribes else "free"


def _middle_count(raw: int, kind_a: str, kind_b: str) -> int:
    """Segments between the fixed parts of a piece so that no node gets two derived radii."""
    if kind_a == kind_b == "single":
        return 1 if raw == 1 else max(raw, 3)
    if {kind_a, kind_b} == {"single", "rung"}:
        return max(raw, 2)
    return raw


def _piece_arc(net: Network, field: SizeField, p: int, runs: list, single_ends: list) -> tuple:
    """(positions along piece p, explicit site or None per segment); records the single ends (segment offsets)."""
    a, b = net.pieces[p]
    length = piece_lengths(net)[p]
    tol = 1e-9 * max(1.0, length)
    arc, sites = [0.0], []
    cursor, kind = 0.0, _end_kind(net, p, a)

    def stretch(lo: float, hi: float, kind_lo: str, kind_hi: str) -> None:
        whole = kind_lo != "rung" and kind_hi != "rung" and lo <= tol and hi >= length - tol
        raw = raw_count(field, p) if whole else _count(field, p, lo, hi)
        count = _middle_count(raw, kind_lo, kind_hi)
        first = len(sites)
        arc.extend(list(_positions(field, p, lo, hi, count)) + [hi])
        sites.extend([None] * count)
        if kind_lo == "single" and lo <= tol:
            single_ends.append((first, a, p, 1))
        if kind_hi == "single" and hi >= length - tol:
            single_ends.append((len(sites) - 1, b, p, -1))

    for t, run_sites in runs:
        if t[0] > cursor + tol:
            stretch(cursor, float(t[0]), kind, "rung")
        arc.extend(float(x) for x in t[1:])
        sites.extend(run_sites)
        cursor, kind = float(t[-1]), "rung"
    if cursor < length - tol:
        stretch(cursor, length, kind, _end_kind(net, p, b))
    return arc, sites


def build_layout(net: Network, field: SizeField) -> tuple:
    """(node xy, segments, seg_piece, explicit sites {segment: site}, single ends, extra sites)."""
    lengths = piece_lengths(net)
    runs, extra = fixed_runs(net, field)
    xy = [p for p in net.hard_xy]
    segments, seg_piece, explicit = [], [], {}
    single_ends = []
    for p, (a, b) in enumerate(net.pieces):
        local = []
        arc, sites = _piece_arc(net, field, p, runs.get(p, []), local)
        start, u = net.hard_xy[a], (net.hard_xy[b] - net.hard_xy[a]) / lengths[p]
        ids = [a]
        for t in arc[1:-1]:
            ids.append(len(xy))
            xy.append(start + t * u)
        ids.append(b)
        first = len(segments)
        segments += list(zip(ids[:-1], ids[1:]))
        seg_piece += [p] * (len(ids) - 1)
        for k, site in enumerate(sites):
            if site is not None:
                explicit[first + k] = np.asarray(site, dtype=float)
        single_ends += [(first + k, node, piece, step) for k, node, piece, step in local]
    return np.array(xy), np.array(segments, dtype=int), np.array(seg_piece, dtype=int), explicit, single_ends, extra


def covered_nodes(net: Network, segments: np.ndarray, explicit: dict) -> set:
    """Hard nodes with two or more segments, all with explicit sites (inside a ladder or corridor run).

    Their radius comes from those sites; they get no tip, fill or
    prescribed sites of their own.
    """
    n = len(net.hard_xy)
    total = np.bincount(segments[segments < n].ravel(), minlength=n) if len(segments) else np.zeros(n, int)
    ends = segments[list(explicit)].ravel() if explicit else np.zeros(0, dtype=int)
    fixed = np.bincount(ends[ends < n], minlength=n)
    return {int(v) for v in np.flatnonzero((total >= 2) & (fixed == total))}


def _node_radius(net: Network, field: SizeField, xy: np.ndarray, segments: np.ndarray,
                 single_ends: list, covered: set) -> dict:
    """R_v of every prescribing hard node (ladder sigma_0, forced, or the smallest first segment allows)."""
    lengths = piece_lengths(net)
    out = {}
    seg_len = np.hypot(*(xy[segments[:, 1]] - xy[segments[:, 0]]).T)
    by_node = {}
    for seg, node, p, _ in single_ends:
        by_node.setdefault(node, []).append((seg, p))
    for v, plan in enumerate(net.plans):
        if not plan.prescribes or plan.rung_end or v in covered:
            continue
        r_v, forced = junction_radius(net, field, v, lengths)
        if v in field.radius or forced:
            out[v] = r_v
            continue
        limits = [seg_len[seg] / (2 * math.cos(_phi(net, v, p))) for seg, p in by_node.get(v, [])]
        if limits:
            out[v] = min(limits)
    return out


def _phi(net: Network, node: int, piece: int) -> float:
    """phi of ``piece`` at ``node``."""
    return next(leg.phi for leg in net.plans[node].legs if leg.kind == "piece" and leg.piece == piece)


def _prescribed_sites(net: Network, xy: np.ndarray, segments: np.ndarray, seg_piece: np.ndarray,
                      single_ends: list, radius_v: dict, explicit: dict) -> None:
    """Add the explicit site of every single and double segment."""
    by_seg = {}
    for seg, node, p, _ in single_ends:
        if seg not in explicit:
            by_seg.setdefault(seg, []).append((node, p))
    for seg, ends in by_seg.items():
        i, j = segments[seg]
        if len(ends) == 2:  # double: where the two prescribed rays meet
            (na, p), (nb, _) = ends
            pa, pb = _phi(net, na, p), _phi(net, nb, p)
            d = np.hypot(*(xy[nb] - xy[na]))
            u = (xy[nb] - xy[na]) / d
            explicit[seg] = xy[na] + d * math.sin(pb) / math.sin(pa + pb) * _rot(u, pa)
            continue
        node, p = ends[0]
        if node not in radius_v:
            continue
        other = j if i == node else i
        u = (xy[other] - xy[node]) / np.hypot(*(xy[other] - xy[node]))
        explicit[seg] = xy[node] + radius_v[node] * _rot(u, _phi(net, node, p))


def _consistent_q(q: np.ndarray, segments: np.ndarray, d: np.ndarray, free: np.ndarray, sweeps: int = 200) -> np.ndarray:
    """Alternating projections of Q = R^2 onto the per-segment foot and offset constraints (free nodes move)."""
    q = q.copy()
    i, j = segments[:, 0], segments[:, 1]
    gap = (1 - 2 * FOOT) * d ** 2
    floor = np.zeros(len(q))
    np.maximum.at(floor, i, d ** 2 * (0.25 + OFFSET ** 2))
    np.maximum.at(floor, j, d ** 2 * (0.25 + OFFSET ** 2))
    floor = np.where(free, floor, 0.0)
    for _ in range(sweeps):
        q = np.maximum(q, floor)
        excess = np.abs(q[i] - q[j]) - gap
        bad = np.flatnonzero(excess > 1e-12 * d ** 2)
        if len(bad) == 0:
            break
        for k in bad:
            a, b = (i[k], j[k]) if q[i[k]] > q[j[k]] else (j[k], i[k])
            e = q[a] - q[b] - gap[k]
            if e <= 0:
                continue
            if free[a] and free[b]:
                q[a] -= 0.5 * e
                q[b] += 0.5 * e
            elif free[a]:
                q[a] -= e
            elif free[b]:
                q[b] += e
    return q


def construct(net: Network, field: SizeField) -> Layout:
    """Nodes, radii and sites of the whole barrier network."""
    xy, segments, seg_piece, explicit, single_ends, extra = build_layout(net, field)
    seg_len = np.hypot(*(xy[segments[:, 1]] - xy[segments[:, 0]]).T)
    covered = covered_nodes(net, segments, explicit)
    radius_v = _node_radius(net, field, xy, segments, single_ends, covered)
    _prescribed_sites(net, xy, segments, seg_piece, single_ends, radius_v, explicit)
    n = len(xy)
    radius = np.full(n, np.nan)
    for v, r in radius_v.items():
        radius[v] = r
    for seg, site in explicit.items():
        for node in segments[seg]:
            if np.isnan(radius[node]):
                radius[node] = np.hypot(*(xy[node] - site))
    total = np.bincount(segments.ravel(), weights=np.repeat(seg_len, 2), minlength=n)
    count = np.bincount(segments.ravel(), minlength=n)
    free = np.isnan(radius)
    initial = np.where(free, F * total / np.maximum(count, 1), radius)
    regular = np.array([seg not in explicit for seg in range(len(segments))], dtype=bool)
    radius = np.sqrt(_consistent_q(initial ** 2, segments[regular], seg_len[regular], free))
    pairs, ok = _pairs(xy, radius, segments, seg_len, explicit)
    extra_pts, extra_node = _tip_sites(net, xy, radius, field, extra, covered)
    sites = np.vstack([pairs.reshape(-1, 2), extra_pts])
    extent = float(max(np.ptp(xy, axis=0).max(), 1.0))
    unique = _unique_points(sites, UNIQUE_FRACTION * extent)
    return Layout(xy, radius, segments, seg_piece, pairs, ok, extra_pts, extra_node, unique, seg_len)


def _pairs(xy: np.ndarray, radius: np.ndarray, segments: np.ndarray, d: np.ndarray, explicit: dict) -> tuple:
    """((m, 2, 2) pairs, (m,) shape ok): circle intersections, or each explicit site and its mirror."""
    i, j = segments[:, 0], segments[:, 1]
    u = (xy[j] - xy[i]) / d[:, None]
    nrm = np.column_stack([-u[:, 1], u[:, 0]])
    a = (d ** 2 + radius[i] ** 2 - radius[j] ** 2) / (2 * d)
    h2 = radius[i] ** 2 - a ** 2
    h = np.sqrt(np.maximum(h2, 0.0))
    ok = h2 > 0
    for seg, site in explicit.items():
        a[seg] = (site - xy[i[seg]]) @ u[seg]
        h[seg] = abs((site - xy[i[seg]]) @ nrm[seg])
        on_i = abs(np.hypot(*(site - xy[i[seg]])) - radius[i[seg]]) <= ON_CIRCLE_EPS * radius[i[seg]]
        on_j = abs(np.hypot(*(site - xy[j[seg]])) - radius[j[seg]]) <= ON_CIRCLE_EPS * radius[j[seg]]
        ok[seg] = bool(on_i and on_j)
    left = xy[i] + a[:, None] * u + h[:, None] * nrm
    right = xy[i] + a[:, None] * u - h[:, None] * nrm
    ok &= h >= H_MIN * d
    return np.stack([left, right], axis=1), ok


def _tip_sites(net: Network, xy: np.ndarray, radius: np.ndarray, field: SizeField, extra: list,
               covered: set) -> tuple:
    """(points, node) of tip and fill sites on hard-node circles, plus the ladder and corridor extra sites."""
    pts, nodes = [], []
    for v, plan in enumerate(net.plans):
        if v in covered:
            continue
        dirs = ([plan.tip_direction] if plan.tip_direction is not None else []) + list(plan.fill_directions)
        for d in dirs:
            pts.append(xy[v] + radius[v] * d)
            nodes.append(v)
    for p, v in extra:
        pts.append(p)
        nodes.append(v)
    return np.array(pts).reshape(-1, 2), np.array(nodes, dtype=int)


def _unique_points(points: np.ndarray, tol: float) -> np.ndarray:
    """Points with those within ``tol`` of an earlier one removed (shared sites)."""
    rep = np.arange(len(points))
    for a, b in sorted(cKDTree(points).query_pairs(tol)):
        rep[b] = min(rep[b], rep[a])
    return points[np.unique(rep[rep])]
