"""Target spacing h along the barrier network (the length of the barrier faces).

1. ``size_function``: vorflow's own size field evaluated in Python (no
   probe Gmsh run): the background lc, every finer polygon (its lc inside,
   growing by growth_factor - 1 per unit distance outside), finer standard
   lines and points (lc + gradient x distance), and, with
   ``MeshGenerator(gap_field=...)``, the size GapField asks for in narrow
   gaps (``GapSizes``). Barriers are not part of it.
2. Samples every lc / SAMPLES_PER_LC along each piece get
   ``h0 = min(barrier lc, size field)``.
3. Caps (the local feature size the empty-circle condition needs; no
   prototype had them):
   - other barrier parts: ``h <= BARRIER_CAP x e``, e the distance to a
     network point that is far along the network (geodesic distance at
     least GEODESIC_RATIO x e). With BARRIER_CAP = 0.49 sqrt 2 the circles of
     two such parts (radius h / sqrt 2) cannot overlap. Pairs that both lie
     within the protection radius of a common hard node (or of a ladder's
     apex) are skipped: the node's own plan handles them;
   - domain boundary: ``h <= BOUNDARY_CAP x e`` (pair sites, h/2 off the
     line, stay inside), except within the protection radius of a node on
     the boundary;
   - feature nodes (wells, hex-ring seeds, embedded points): ``h <=
     FEATURE_CAP x e`` (they stay outside every circle, sites keep 0.4 e
     away);
   - fixed lines (``Obstacle``: embedded standard lines, internal zone
     edges, quad-buffer strips): ``h <= OBSTACLE_CAP x e``, e the distance
     to the line (beyond a strip's half-width), so their Gmsh nodes stay
     outside the circles too; except on the line itself, within
     CROSSING_REACH x h of where it crosses a barrier (a line crossing at
     10 degrees is that far within h of it; the crossing's own plan handles
     it) and within PROTECT_FRACTION x h of a barrier end next to it.
4. A piece shorter than h is a source of ``L^(1 - SHORT_BLEND) h^SHORT_BLEND``
   (between its own length and h; upr's size_blend), so one short
   digitised piece does not shrink everything around it.
5. Lipschitz bound (GRADATION) over the network by a multi-source Dijkstra.
   Prescribing nodes are not linked across their legs.
6. Ladders are laid out from this spacing; along their legs h is held to
   the last strip height plus GRADATION x distance, and the first segment
   of every other leg of a prescribing node is set to 2 R_v cos(phi)
   (ccvt's junction sizes), R_v being the smallest radius any leg allows
   (or the ladder's sigma_0, or the value a single-segment piece forces).
7. Corridors (``corridor``) are laid out last. Inside a corridor the two
   sides do not cap each other (and a boundary run is not capped by its
   edge); h along them is the strip length, so the gradation carries it to
   the pieces beyond.
"""
from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import corridor as corridors
from .network import Network
from .rung import LADDER_LEG_FRACTION, ladder

SAMPLES_PER_LC = 16
GRADATION = 0.3
BARRIER_CAP = 0.49 * math.sqrt(2.0)
BOUNDARY_CAP = 1.6
FEATURE_CAP = 0.85
OBSTACLE_CAP = FEATURE_CAP
CROSSING_REACH = 7.0     # x h: no obstacle cap this close to where the obstacle crosses a barrier
GEODESIC_RATIO = 2.0
PROTECT_FRACTION = 1.5   # x h at the node: protection radius of a hard node
CAP_REACH = 1.5          # x h: caps are only looked for this far
PAIR_CHUNK = 20000       # samples per neighbour query in ``near_pairs``
SOURCE_CHUNK = 256       # hard nodes per Dijkstra call in ``node_distances``
SHORT_BLEND = 0.5
JUNCTION_PASSES = 3
CAUSES = ("size field", "barrier", "boundary", "feature", "fixed line", "short piece", "repair", "gap field")


# ---------------------------------------------------------------------------
# vorflow's size field
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class GapSizes:
    """The sizes a ``fields.GapField`` asks for (its ``_gap.GapEstimate``), at local points.

    ``origin`` is the local frame's origin. Only the estimate's active medial
    samples set a size, as in the Gmsh field.
    """
    estimate: object
    origin: np.ndarray

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        """GapField's size at local points xy (the background size where it asks for nothing)."""
        return self.estimate.sizes_at(np.asarray(xy, dtype=float).reshape(-1, 2) + self.origin)

    def cells_across(self, region) -> float:
        """Most cells GapField asks for across a gap inside ``region`` (local), 0 where it asks for none."""
        est = self.estimate
        act = est.active
        if not act.any() or region is None or region.is_empty:
            return 0.0
        inside = shapely.contains_xy(region, *(est.xy[act] - self.origin).T)
        return float(np.max(est.gap[act][inside] / est.size[act][inside], initial=0.0))


@dataclasses.dataclass
class SizeFunction:
    """``size(xy)``: vorflow's target mesh size at local points, evaluated in Python.

    ``sources`` holds ``(geometry, is_area, ramp)`` per sizing feature, in
    local coordinates; ``ramp(distance)`` is the size the feature's fields
    ask for at that distance (see ``fields.MeshField.python_size``). An area
    holds its size at distance 0 inside. The background lc caps everything,
    and ``gap`` (``GapSizes``), when set, caps it in narrow gaps.
    """
    sources: list
    background_lc: float
    gap: GapSizes | None = None

    def __post_init__(self):
        for geom, _, _ in self.sources:
            shapely.prepare(geom)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        pts = shapely.points(np.asarray(xy, dtype=float).reshape(-1, 2))
        out = np.full(len(pts), float(self.background_lc))
        for geom, is_area, ramp in self.sources:
            d = shapely.distance(geom, pts)
            if is_area:
                d = np.where(shapely.covers(geom, pts), 0.0, d)
            out = np.minimum(out, ramp(d))
        if self.gap is not None:
            out = np.minimum(out, self.gap(shapely.get_coordinates(pts)))
        return out


def size_function(sources: list, background_lc: float, gap: GapSizes | None = None) -> SizeFunction:
    """``SizeFunction`` of the sizing features ``sources`` and, optionally, GapField's sizes."""
    return SizeFunction(sources, float(background_lc), gap)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Obstacle:
    """A fixed line whose Gmsh nodes the barrier circles must not reach (local coordinates)."""
    label: str
    geometry: object         # the line (or a zone's inner edges)
    half_width: float        # nodes lie up to this far from it (quad-buffer strips), else 0
    crossings: np.ndarray    # (k, 2) points where it meets the barrier lines
    ends: np.ndarray         # (k, 2) points of it next to a barrier end


@dataclasses.dataclass
class SizeField:
    """Target spacing h at samples along every piece, plus the ladders laid out from it."""
    piece: np.ndarray
    s: np.ndarray
    h: np.ndarray
    xy: np.ndarray
    hard_of: np.ndarray      # hard node of a sample at a piece end, else -1
    decoupled: set
    ladders: list = dataclasses.field(default_factory=list)  # (y, sigma) per network rung
    radius: dict = dataclasses.field(default_factory=dict)   # node -> R forced by a ladder
    corridors: list = dataclasses.field(default_factory=list)  # CorridorLayout per network corridor
    failed: list = dataclasses.field(default_factory=list)     # keys of corridors that could not be laid out
    capped: np.ndarray | None = None  # h after the caps, before grading
    cause: np.ndarray | None = None   # index into CAUSES of what set ``capped``


def piece_lengths(net: Network) -> np.ndarray:
    """Length of every piece."""
    return np.hypot(*(net.hard_xy[net.pieces[:, 1]] - net.hard_xy[net.pieces[:, 0]]).T)


def samples(net: Network) -> SizeField:
    """Evenly spaced samples along every piece (h not set)."""
    lengths = piece_lengths(net)
    piece, s, xy, hard_of = [], [], [], []
    for p, (a, b) in enumerate(net.pieces):
        m = max(2, int(math.ceil(lengths[p] / (net.piece_lc[p] / SAMPLES_PER_LC))) + 1)
        sp = np.linspace(0.0, lengths[p], m)
        piece.append(np.full(m, p))
        s.append(sp)
        xy.append(net.hard_xy[a] + np.outer(sp / lengths[p], net.hard_xy[b] - net.hard_xy[a]))
        tag = np.full(m, -1)
        tag[0], tag[-1] = a, b
        hard_of.append(tag)
    decoupled = {v for v, plan in enumerate(net.plans) if plan.prescribes}
    return SizeField(np.concatenate(piece), np.concatenate(s), np.zeros(sum(map(len, s))), np.vstack(xy),
                     np.concatenate(hard_of), decoupled)


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------
def protection_radii(net: Network, field: SizeField, h: np.ndarray) -> np.ndarray:
    """Protection radius per hard node: PROTECT_FRACTION x the spacing there."""
    rho = np.zeros(len(net.hard_xy))
    ends = field.hard_of >= 0
    np.maximum.at(rho, field.hard_of[ends], PROTECT_FRACTION * h[ends])
    return rho


def rung_extent(theta: float, spacing: float) -> float:
    """Distance from the apex where a ladder of opening ``theta`` usually ends."""
    return 0.85 * spacing / max(math.sin(theta / 2), 1e-3) + spacing


def node_graph(net: Network):
    """The network as a sparse graph of the hard nodes, weighted by piece length."""
    n = len(net.hard_xy)
    return coo_matrix((piece_lengths(net), (net.pieces[:, 0], net.pieces[:, 1])), shape=(n, n)).tocsr()


def node_distances(graph, limit: float, src: np.ndarray, dst: np.ndarray, chunk: int = SOURCE_CHUNK) -> np.ndarray:
    """Shortest distance in ``graph`` (``node_graph``) from node ``src[k]`` to ``dst[k]`` (inf beyond ``limit``).

    Only the sources asked for are searched, ``chunk`` at a time, so the
    memory is chunk x hard nodes rather than hard nodes squared.
    """
    out = np.full(len(src), np.inf)
    sources, which = np.unique(np.asarray(src, dtype=np.int64), return_inverse=True)
    order = np.argsort(which, kind="stable")
    bounds = np.searchsorted(which[order], np.arange(0, len(sources) + chunk, chunk))
    for c, start in enumerate(range(0, len(sources), chunk)):
        block = dijkstra(graph, directed=False, indices=sources[start:start + chunk], limit=limit)
        sel = order[bounds[c]:bounds[c + 1]]
        out[sel] = block[which[sel] - start, dst[sel]]
    return out


def near_pair_chunks(field: SizeField, h0: np.ndarray, chunk: int = PAIR_CHUNK):
    """Per ``chunk`` samples i: (i, j) of the samples j on another piece within CAP_REACH x h0[i] of i.

    Each sample looks only as far as its own reach, so the memory grows
    with the number of close pairs in a chunk, not with the coarsest h0
    over the finest sample spacing (a single global radius did). A pair
    within reach of both samples comes once from each.
    """
    tree = cKDTree(field.xy)
    radius = CAP_REACH * h0
    for start in range(0, len(h0), chunk):
        stop = min(start + chunk, len(h0))
        hits = tree.query_ball_point(field.xy[start:stop], radius[start:stop], return_sorted=False)
        counts = np.fromiter(map(len, hits), dtype=np.int64, count=len(hits))
        j = np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64, count=int(counts.sum()))
        i = np.repeat(np.arange(start, stop), counts)
        keep = field.piece[i] != field.piece[j]
        yield i[keep], j[keep]


def near_pairs(field: SizeField, h0: np.ndarray, chunk: int = PAIR_CHUNK) -> tuple:
    """(i, j), i < j, of samples on different pieces closer than CAP_REACH x the larger of their h0."""
    rows, cols = [], []
    for i, j in near_pair_chunks(field, h0, chunk):
        rows.append(np.minimum(i, j))
        cols.append(np.maximum(i, j))
    key = np.unique(np.concatenate(rows) * len(h0) + np.concatenate(cols)) if rows else np.empty(0, dtype=np.int64)
    return key // len(h0), key % len(h0)


def _geodesic(net: Network, field: SizeField, graph, limit: float, i: np.ndarray, j: np.ndarray,
              rho: np.ndarray, extents: list, inside: np.ndarray) -> np.ndarray:
    """Network distance between samples i and j (0 where a node's plan, a ladder or a corridor handles them)."""
    lengths = piece_lengths(net)
    pi, pj = field.piece[i], field.piece[j]
    geo = np.full(len(i), np.inf)
    ends = [(ei, ej) for ei in (0, 1) for ej in (0, 1)]
    n_pieces = len(net.pieces)
    piece_pairs, which = np.unique(pi * n_pieces + pj, return_inverse=True)
    qi, qj = piece_pairs // n_pieces, piece_pairs % n_pieces
    between = node_distances(graph, limit, np.concatenate([net.pieces[qi, ei] for ei, _ in ends]),
                             np.concatenate([net.pieces[qj, ej] for _, ej in ends])).reshape(len(ends), -1)
    for (ei, ej), piece_d in zip(ends, between):
        ui, uj = net.pieces[pi, ei], net.pieces[pj, ej]
        graph_d = piece_d[which]
        di = field.s[i] if ei == 0 else lengths[pi] - field.s[i]
        dj = field.s[j] if ej == 0 else lengths[pj] - field.s[j]
        geo = np.minimum(geo, di + graph_d + dj)
        shared = ui == uj  # both within the protection radius of a common node: the plan's business
        near = shared & (np.hypot(*(field.xy[i] - net.hard_xy[ui]).T) < rho[ui]) & \
            (np.hypot(*(field.xy[j] - net.hard_xy[ui]).T) < rho[ui])
        geo[near] = 0.0
    for rung, extent in zip(net.rungs, extents):
        on_legs = np.isin(pi, rung.pieces) & np.isin(pj, rung.pieces)
        near = on_legs & (np.hypot(*(field.xy[i] - rung.apex).T) < extent) & \
            (np.hypot(*(field.xy[j] - rung.apex).T) < extent)
        geo[near] = 0.0
    for mask in inside:
        geo[mask[i] & mask[j]] = 0.0
    return geo


def _corridor_masks(net: Network, field: SizeField) -> list:
    """Per corridor, the mask of the samples in its stretch."""
    out = []
    for c in net.corridors:
        inside = np.zeros(len(field.s), dtype=bool)
        for p in [sp.piece for side in c.sides for sp in side] + [p for path in c.paths for p in path]:
            sel = np.flatnonzero(field.piece == p)
            inside[sel] = corridors.contains(c, p, field.xy[sel])
        out.append(inside)
    return out


def barrier_cap(net: Network, field: SizeField, h0: np.ndarray, rho: np.ndarray, extents: list) -> np.ndarray:
    """Per sample: BARRIER_CAP x distance to the nearest network point that is far along the network.

    The close sample pairs are taken a chunk at a time (``near_pair_chunks``);
    a pair seen twice caps both samples by the same value.
    """
    reach = CAP_REACH * float(h0.max())
    limit = GEODESIC_RATIO * reach + 2 * float(piece_lengths(net).max())
    graph = node_graph(net)
    inside = _corridor_masks(net, field)
    cap = np.full(len(h0), np.inf)
    for i, j in near_pair_chunks(field, h0):
        e = np.hypot(*(field.xy[i] - field.xy[j]).T)
        keep = e < CAP_REACH * np.maximum(h0[i], h0[j])
        i, j, e = i[keep], j[keep], e[keep]
        if len(i) == 0:
            continue
        geo = _geodesic(net, field, graph, limit, i, j, rho, extents, inside)
        far = (geo > GEODESIC_RATIO * e) & (e > 0)
        np.minimum.at(cap, i[far], BARRIER_CAP * e[far])
        np.minimum.at(cap, j[far], BARRIER_CAP * e[far])
    return cap


def boundary_cap(net: Network, field: SizeField, domain, rho: np.ndarray) -> np.ndarray:
    """Per sample: BOUNDARY_CAP x distance to the domain boundary, away from the piece's boundary nodes."""
    e = shapely.distance(domain.boundary, shapely.points(field.xy))
    cap = BOUNDARY_CAP * e
    for end in (0, 1):
        node = net.pieces[field.piece, end]
        on = np.array([net.plans[v].on_boundary for v in node])
        near = on & (np.hypot(*(field.xy - net.hard_xy[node]).T) < rho[node])
        cap[near] = np.inf
    for c in net.corridors:
        if len(c.sides) == 1:
            for sp in c.sides[0]:
                sel = np.flatnonzero(field.piece == sp.piece)
                cap[sel[corridors.contains(c, sp.piece, field.xy[sel])]] = np.inf
    return cap


def feature_cap(field: SizeField, features: np.ndarray) -> np.ndarray:
    """Per sample: FEATURE_CAP x distance to the nearest feature node."""
    if len(features) == 0:
        return np.full(len(field.s), np.inf)
    d, _ = cKDTree(features).query(field.xy)
    return FEATURE_CAP * d


def _within(points: np.ndarray, xy: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """Mask of ``xy`` closer than ``reach`` to one of ``points``."""
    if len(points) == 0 or len(xy) == 0:
        return np.zeros(len(xy), dtype=bool)
    return cKDTree(points).query(xy)[0] < reach


def near_crossing(obstacle: Obstacle, xy: np.ndarray, h: np.ndarray, extra: float = 0.0) -> np.ndarray:
    """Mask of the points ``xy`` where ``obstacle`` meets a barrier: within (CROSSING_REACH + extra) x ``h`` of
    a crossing, or (PROTECT_FRACTION + extra) x ``h`` of a barrier end next to it."""
    return _within(obstacle.crossings, xy, (CROSSING_REACH + extra) * h) | \
        _within(obstacle.ends, xy, (PROTECT_FRACTION + extra) * h)


def obstacle_cap(field: SizeField, obstacles: list, h0: np.ndarray, tol: float) -> np.ndarray:
    """Per sample: OBSTACLE_CAP x distance to the nearest obstacle, away from its crossings (module docstring)."""
    cap = np.full(len(field.s), np.inf)
    pts = shapely.points(field.xy)
    for ob in obstacles:
        e = np.maximum(shapely.distance(ob.geometry, pts) - ob.half_width, 0.0)
        exempt = near_crossing(ob, field.xy, h0) | (e <= tol)
        cap = np.minimum(cap, np.where(exempt, np.inf, OBSTACLE_CAP * e))
    return cap


# ---------------------------------------------------------------------------
# Lipschitz and junction sizes
# ---------------------------------------------------------------------------
def _sample_graph(field: SizeField):
    """Graph of the samples: neighbours along a piece, and samples on the same (non-prescribing) hard node."""
    n = len(field.s)
    same = field.piece[1:] == field.piece[:-1]
    k = np.flatnonzero(same)
    rows, cols, w = list(k), list(k + 1), list(np.abs(field.s[k + 1] - field.s[k]))
    hard = np.flatnonzero(field.hard_of >= 0)
    for node in np.unique(field.hard_of[hard]):
        if node in field.decoupled:
            continue
        members = hard[field.hard_of[hard] == node]
        rows += [members[0]] * (len(members) - 1)
        cols += list(members[1:])
        w += [1e-12] * (len(members) - 1)
    return coo_matrix((w, (rows, cols)), shape=(n, n))


def lipschitz(field: SizeField, h: np.ndarray, gradation: float = GRADATION) -> np.ndarray:
    """min over samples y of h(y) + gradation x network distance (Dijkstra from a virtual source)."""
    n = len(h)
    graph = _sample_graph(field)
    rows = np.concatenate([graph.row, np.full(n, n)])
    cols = np.concatenate([graph.col, np.arange(n)])
    w = np.concatenate([graph.data, h / gradation])
    full = coo_matrix((w, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    return np.minimum(h, gradation * dijkstra(full, directed=False, indices=n)[:n])


def _leg_phi(net: Network, node: int, piece: int) -> float:
    """phi of ``piece`` at hard ``node``."""
    return next(leg.phi for leg in net.plans[node].legs if leg.kind == "piece" and leg.piece == piece)


def integral(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of y over x (np.trapezoid needs NumPy 2)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def raw_count(field: SizeField, piece: int) -> int:
    """Segments of ``piece`` from h alone: round of the integral of 1/h, at least 1."""
    sel = field.piece == piece
    return max(1, int(round(integral(1.0 / field.h[sel], field.s[sel]))))


def _other(net: Network, piece: int, node: int) -> int:
    """The far end of ``piece`` from ``node``."""
    a, b = net.pieces[piece]
    return b if a == node else a


def junction_radius(net: Network, field: SizeField, v: int, lengths: np.ndarray) -> tuple:
    """(R_v, {piece: forced}) at a prescribing node: the ladder's sigma_0, a forced value, or the smallest allowed."""
    plan = net.plans[v]
    rung_pieces = {p for r in net.rungs for p in r.pieces if v in r.nodes}
    forced, limits = {}, []
    for leg in plan.legs:
        if leg.kind != "piece" or leg.piece in rung_pieces:
            continue
        p = leg.piece
        other = _other(net, p, v)
        if net.plans[other].prescribes and not net.plans[other].rung_end and raw_count(field, p) <= 1 \
                and p not in {q for r in net.rungs for q in r.pieces}:
            phi_o = _leg_phi(net, other, p)
            forced[p] = lengths[p] * math.sin(phi_o) / math.sin(leg.phi + phi_o)
        else:
            h0 = field.h[(field.piece == p) & (field.hard_of == v)][0]
            limits.append(h0 / (2 * math.cos(leg.phi)))
    if v in field.radius:
        return field.radius[v], forced
    if forced:
        return min(forced.values()), forced
    return (min(limits) if limits else None), forced


def junction_sizes(net: Network, field: SizeField) -> None:
    """Lower h at the start of every leg of a prescribing node so its first segment is 2 R_v cos(phi)."""
    lengths = piece_lengths(net)
    for v, plan in enumerate(net.plans):
        if not plan.prescribes or plan.rung_end:
            continue
        r_v, forced = junction_radius(net, field, v, lengths)
        if r_v is None:
            continue
        rung_pieces = {p for r in net.rungs for p in r.pieces if v in r.nodes}
        for leg in plan.legs:
            if leg.kind != "piece" or leg.piece in forced or leg.piece in rung_pieces:
                continue
            p = leg.piece
            d = 2 * r_v * math.cos(leg.phi)
            sel = field.piece == p
            along = field.s[sel] if net.pieces[p][0] == v else lengths[p] - field.s[sel]
            idx = np.flatnonzero(sel)[along <= d]
            field.h[idx] = np.minimum(field.h[idx], d)


# ---------------------------------------------------------------------------
# Ladders
# ---------------------------------------------------------------------------
def _leg_along(net: Network, field: SizeField, piece: int, node: int) -> np.ndarray:
    """Distance of each sample of ``piece`` from ``node`` (one of its ends)."""
    sel = field.piece == piece
    return field.s[sel] if net.pieces[piece][0] == node else piece_lengths(net)[piece] - field.s[sel]


def lay_ladders(net: Network, field: SizeField, gradation: float = GRADATION) -> None:
    """Ladder positions of every rung from the current h; h along its legs follows the last strip."""
    lengths = piece_lengths(net)
    field.ladders, field.radius = [], {}
    sigma0 = {}
    for rung in net.rungs:
        legs = [(p, n) for p, n in zip(rung.pieces, rung.nodes)]
        along = [_leg_along(net, field, p, n) + rung.t0 for p, n in legs]
        hs = [field.h[field.piece == p] for p, _ in legs]

        def spacing(t, along=along, hs=hs):
            return float(min(np.interp(t, a[np.argsort(a)], h[np.argsort(a)]) for a, h in zip(along, hs)))

        t_limit = rung.t0 + LADDER_LEG_FRACTION * min(lengths[p] for p, _ in legs)
        apex = rung.nodes[0] if rung.t0 == 0 else None
        y, sigma = ladder(rung.theta, rung.t0, sigma0.get(apex), spacing, t_limit)
        if apex is not None:
            sigma0.setdefault(apex, float(sigma[0]))
            field.radius[apex] = sigma0[apex]
        field.ladders.append((y, sigma))
        t_end = y[-1] / math.cos(rung.theta / 2)
        last = float(np.diff(y)[-1]) if len(y) > 1 else float(y[-1])
        for (p, n), a in zip(legs, along):
            idx = np.flatnonzero(field.piece == p)
            beyond = a >= t_end
            field.h[idx[beyond]] = np.minimum(field.h[idx[beyond]], last + gradation * (a[beyond] - t_end))


def lay_corridors(net: Network, field: SizeField) -> None:
    """Corridor layouts from the current h; h along their runs becomes the segment length."""
    field.corridors, field.failed = [], []
    for c in net.corridors:
        lay = corridors.lay_out(c, net, field)
        if lay is None:
            field.failed.append(c.key)
            continue
        field.corridors.append(lay)
        for p, t, _ in lay.runs:
            sel = np.flatnonzero(field.piece == p)
            k = np.searchsorted(t, field.s[sel], side="right") - 1
            inside = (k >= 0) & (k < len(t) - 1)
            field.h[sel[inside]] = np.diff(t)[k[inside]]


def capped_field(net: Network, size_fn, domain, features: np.ndarray, obstacles: list = (),
                 refine: np.ndarray | None = None, tol: float = 0.0) -> SizeField:
    """Samples with h from the size field and the caps (steps 1-4 of the module docstring), not yet graded.

    ``refine`` holds per-sample upper bounds of h (repair rounds).
    """
    field = samples(net)
    lc = net.piece_lc[field.piece]
    h0 = np.minimum(lc, size_fn(field.xy))
    rho = protection_radii(net, field, h0)
    extents = [rung.t0 + rung_extent(rung.theta, float(h0[np.isin(field.hard_of, rung.nodes)].max()))
               for rung in net.rungs]
    for rung, extent in zip(net.rungs, extents):
        for n in rung.nodes:
            rho[n] = max(rho[n], extent)
    caps = np.vstack([h0, barrier_cap(net, field, h0, rho, extents), boundary_cap(net, field, domain, rho),
                      feature_cap(field, features), obstacle_cap(field, obstacles, h0, tol)])
    h, cause = caps.min(axis=0), caps.argmin(axis=0)
    gap = getattr(size_fn, "gap", None)
    if gap is not None:
        set_by_gap = (cause == CAUSES.index("size field")) & (gap(field.xy) <= caps[0] * (1 + 1e-9))
        cause = np.where(set_by_gap, CAUSES.index("gap field"), cause)
    lengths = piece_lengths(net)[field.piece]
    short = lengths < h
    h_short = np.where(short, lengths ** (1 - SHORT_BLEND) * h ** SHORT_BLEND, h)
    cause = np.where(h_short < h, CAUSES.index("short piece"), cause)
    h = h_short
    if refine is not None:
        cause = np.where(refine < h, CAUSES.index("repair"), cause)
        h = np.minimum(h, refine)
    field.h, field.capped, field.cause = h, h.copy(), cause
    return field


def grade_field(net: Network, field: SizeField, gradation: float = GRADATION) -> SizeField:
    """``field`` graded (steps 5-7 of the module docstring), in place; h must be positive."""
    assert (field.h > 0).all(), "barrier face lengths must be positive before grading"
    field.h = lipschitz(field, field.h, gradation)
    lay_ladders(net, field, gradation)
    lay_corridors(net, field)
    field.h = lipschitz(field, field.h, gradation)
    for _ in range(JUNCTION_PASSES):
        junction_sizes(net, field)
        field.h = lipschitz(field, field.h, gradation)
    lay_ladders(net, field, gradation)
    lay_corridors(net, field)
    return field

