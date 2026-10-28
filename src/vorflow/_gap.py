"""Local gap width between embedded features, and the mesh size it asks for.

Pure shapely + SciPy; ``fields.GapField`` turns the result into a Gmsh field.

The gap is read off an approximate medial axis of the feature linework:

1. The linework of every embedded polygon ring and line is noded with
   ``union_all`` and merged into *pieces* between junctions. Embedded points
   are kept as separate walls.
2. Each piece is simplified (removing vertex jitter below the tolerance) and
   sampled at a fixed spacing, keeping its vertices.
3. The Delaunay triangulation of the samples gives the Voronoi vertices (the
   triangle circumcentres). A circumcentre inside the domain is a *medial
   sample* when two of its three samples
   * subtend at least ``MIN_ANGLE`` at it, which drops convex corners, wide
     junctions and neighbouring samples on one piece; and
   * lie more than ``ARC_FACTOR`` x their chord apart *along the noded
     linework* (infinite between unconnected walls). A polygon's own narrow
     neck therefore counts, while a junction or corner wider than about
     30 degrees does not.
   Pairs touching linework of an exempt feature never count.
4. The circumradius r is half the local gap. Spikes are removed by replacing
   r with the median r of the medial samples inside its own ball.

The size a medial sample asks for is ``s = clip(2 r / k, min_size, background_lc)``.
It holds inside the sample's inscribed ball, so it reaches the walls on both
sides, and grows at gradient G outside it:

    h(x) = min(background_lc, min_m s_m + G * max(0, |x - m| - r_m))

A medial sample is *active* only when s is below both ``background_lc`` and
the size the model's own feature fields already give there (estimated from
the features' default growth fields). With no active sample there is no field.

Limits: vertex jitter larger than the simplification tolerance creates small
notches that read as narrow gaps; wedges narrower than about 30 degrees (a
line meeting the boundary at a shallow angle) are refined towards their tip,
down to ``min_size``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import Delaunay, QhullError, cKDTree

MIN_ANGLE = 150.0  # degrees between the two walls of a gap, seen from the medial sample
ARC_FACTOR = 3.0  # walls count as distinct when this many chords apart along the linework
SPACING_FRACTION = 0.5  # sample spacing as a fraction of min_size
SIMPLIFY_FRACTION = 0.25  # simplification tolerance as a fraction of min_size
EXEMPT_FRACTION = 1e-3  # a piece within this fraction of min_size of exempt linework is exempt
VIEW_MAX_LEVELS = 12  # cap on the halvings of the view grid spacing


@dataclasses.dataclass
class GapWalls:
    """Embedded geometry the gaps are measured between.

    ``polygons`` also define the domain. ``exempt`` holds linework (exempt
    polygon boundaries and exempt lines) that never forms a side of a gap.
    Exempt points are left out of ``points`` altogether.
    """
    polygons: list
    lines: list
    points: list
    exempt: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SizeSource:
    """A feature's own size field: ``size + gradient * distance`` from ``geometry``."""
    geometry: object
    size: float
    gradient: float


@dataclasses.dataclass
class _Samples:
    """Wall samples and the noded network they lie on."""
    xy: np.ndarray  # (N, 2)
    piece: np.ndarray  # piece id per sample (points after the pieces)
    arc: np.ndarray  # arc position along the piece
    exempt: np.ndarray  # per sample: lies on exempt linework
    piece_len: np.ndarray  # per piece/point
    closed: np.ndarray  # per piece/point: an isolated closed ring
    ends: np.ndarray  # per piece/point: (start node, end node); -1 for rings and points
    graph: object  # csr adjacency of the junction nodes, weighted by piece length


@dataclasses.dataclass
class GapEstimate:
    """Medial samples of the narrow gaps and the size each asks for."""
    xy: np.ndarray  # (M, 2) medial samples with gap / k below background_lc
    r: np.ndarray  # smoothed half gap
    size: np.ndarray  # clip(2 r / k, min_size, background_lc)
    spec: np.ndarray  # size the model's own fields give at xy (estimate)
    cells_across: float
    min_size: float
    background_lc: float
    gradient: float
    n_wall_samples: int = 0

    @property
    def gap(self) -> np.ndarray:
        """Gap width at each medial sample."""
        return 2.0 * self.r

    @property
    def active(self) -> np.ndarray:
        """Medial samples whose size is below the background and the model's own size."""
        return self.size < np.minimum(self.spec, self.background_lc)

    @property
    def floor_bound(self) -> np.ndarray:
        """Active medial samples where ``min_size`` is above gap / k."""
        return self.active & (self.gap / self.cells_across < self.min_size)

    def sizes_at(self, xy, chunk: int = 20000) -> np.ndarray:
        """h(x) at points xy (N, 2), capped at background_lc."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        h = np.full(len(xy), self.background_lc)
        act = self.active
        if not act.any():
            return h
        m, r, s = self.xy[act], self.r[act], self.size[act]
        reach = float(np.max(r + (self.background_lc - s) / self.gradient))
        tree_m = cKDTree(m)
        for i0 in range(0, len(xy), chunk):
            h[i0:i0 + chunk] = _chunk_sizes(xy[i0:i0 + chunk], tree_m, m, r, s, reach,
                                            self.gradient, self.background_lc)
        return h


def _chunk_sizes(q, tree_m, m, r, s, reach, gradient, background_lc):
    """h at the query points q from the medial samples within ``reach``."""
    h = np.full(len(q), background_lc)
    pairs = cKDTree(q).sparse_distance_matrix(tree_m, reach, output_type="coo_matrix")
    rows, cols, d = pairs.row, pairs.col, pairs.data
    # sparse_distance_matrix drops exact zero distances; add them back.
    zero_d, zero_q = cKDTree(q).query(m, k=1, distance_upper_bound=1e-12)
    hit = np.isfinite(zero_d)
    rows = np.concatenate([rows, zero_q[hit]])
    cols = np.concatenate([cols, np.flatnonzero(hit)])
    d = np.concatenate([d, np.zeros(int(hit.sum()))])
    values = s[cols] + gradient * np.maximum(0.0, d - r[cols])
    np.minimum.at(h, rows, values)
    return h


def _parts(geom) -> list:
    """Non-empty single-part geometries of ``geom``."""
    if geom is None or geom.is_empty:
        return []
    return [g for g in shapely.get_parts(geom) if not g.is_empty]


def wall_pieces(walls: GapWalls) -> tuple[list, list]:
    """Noded linework pieces between junctions, and the wall points."""
    linework = []
    for poly in walls.polygons:
        for part in _parts(poly):
            linework.extend(shapely.LineString(ring.coords) for ring in (part.exterior, *part.interiors))
    for line in walls.lines:
        linework.extend(_parts(line))
    pieces = []
    if linework:
        merged = shapely.line_merge(shapely.union_all(linework))
        pieces = _parts(merged)
    points = [p for point in walls.points for p in _parts(point)]
    return pieces, points


def _piece_exempt(pieces: list, exempt: list, tol: float) -> np.ndarray:
    """True for pieces lying on exempt linework (tested at their midpoint)."""
    flags = np.zeros(len(pieces), dtype=bool)
    exempt = [g for g in exempt if g is not None and not g.is_empty]
    if not pieces or not exempt:
        return flags
    exempt_union = shapely.union_all(exempt)
    shapely.prepare(exempt_union)
    mids = shapely.line_interpolate_point(np.asarray(pieces, dtype=object), 0.5, normalized=True)
    return shapely.dwithin(exempt_union, mids, tol)


def _sample_piece(piece, spacing: float, simplify: float):
    """Coordinates every ``spacing`` along a piece (vertices kept) and their arc positions."""
    if simplify > 0:
        piece = shapely.simplify(piece, simplify, preserve_topology=False)
    xy = shapely.get_coordinates(shapely.segmentize(piece, spacing))
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    return xy, arc


def _junction_graph(end_xy: np.ndarray, closed: np.ndarray, piece_len: np.ndarray, scale: float):
    """Piece end nodes (merged by coordinate), updated ring flags and the node graph."""
    n = len(end_xy)
    keys = np.round(end_xy.reshape(-1, 2) / scale)
    _, node = np.unique(keys, axis=0, return_inverse=True)
    node = node.ravel().reshape(n, 2)
    uses = np.bincount(node.ravel())
    # A closed piece whose node no other piece uses is an isolated ring.
    isolated = closed & (uses[node[:, 0]] == 2)
    ends = np.where(isolated[:, None], -1, node)
    n_nodes = int(node.max()) + 1
    return ends, isolated, _shortest_edges(ends[~isolated], piece_len[~isolated], n_nodes)


def _shortest_edges(edges: np.ndarray, weights: np.ndarray, n_nodes: int):
    """Sparse node graph keeping the shortest of parallel edges and no self-loops."""
    edges = np.sort(edges, axis=1)
    loop = edges[:, 0] == edges[:, 1]
    edges, weights = edges[~loop], weights[~loop]
    order = np.lexsort((weights, edges[:, 1], edges[:, 0]))
    edges, weights = edges[order], weights[order]
    first = np.ones(len(edges), dtype=bool)
    first[1:] = np.any(edges[1:] != edges[:-1], axis=1)
    edges, weights = edges[first], weights[first]
    return coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes)).tocsr()


def sample_walls(pieces: list, points: list, spacing: float, simplify: float,
                 exempt: list = (), tol: float = 0.0) -> _Samples:
    """Samples along every piece, plus the wall points, with the junction network."""
    xy, piece_id, arc = [], [], []
    piece_len, closed, end_xy = [], [], []
    for i, piece in enumerate(pieces):
        c, s = _sample_piece(piece, spacing, simplify)
        ring = bool(np.allclose(c[0], c[-1]))
        piece_len.append(s[-1])
        closed.append(ring)
        end_xy.append([c[0], c[-1]])
        if ring:
            c, s = c[:-1], s[:-1]
        xy.append(c)
        piece_id.append(np.full(len(c), i))
        arc.append(s)
    n_pieces = len(pieces)
    for j, point in enumerate(points):
        xy.append(np.array([[point.x, point.y]]))
        piece_id.append(np.array([n_pieces + j]))
        arc.append(np.zeros(1))
    if not xy:
        empty = np.zeros(0)
        return _Samples(np.zeros((0, 2)), empty.astype(int), empty, empty.astype(bool), empty,
                        empty.astype(bool), np.zeros((0, 2), dtype=int), coo_matrix((0, 0)).tocsr())
    xy, piece_id, arc = (np.concatenate(a) for a in (xy, piece_id, arc))
    scale = 1e-9 * max(1.0, float(np.abs(xy).max()))
    piece_len = np.concatenate([np.asarray(piece_len, dtype=float), np.zeros(len(points))])
    closed = np.concatenate([np.asarray(closed, dtype=bool), np.zeros(len(points), dtype=bool)])
    ends = -np.ones((len(piece_len), 2), dtype=int)
    graph = coo_matrix((0, 0)).tocsr()
    if n_pieces:
        ends[:n_pieces], closed[:n_pieces], graph = _junction_graph(
            np.asarray(end_xy), closed[:n_pieces], piece_len[:n_pieces], scale
        )
    exempt_piece = np.concatenate([_piece_exempt(pieces, list(exempt), tol), np.zeros(len(points), dtype=bool)])
    # Coincident samples (shared junction vertices) are kept once.
    _, first = np.unique(np.round(xy / scale), axis=0, return_index=True)
    first = np.sort(first)
    return _Samples(xy[first], piece_id[first], arc[first], exempt_piece[piece_id[first]],
                    piece_len, closed, ends, graph)


def _node_distances(samples: _Samples, pieces_a: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Shortest paths (up to ``limit``) from the end nodes of ``pieces_a``; row index per node."""
    sources = np.unique(samples.ends[pieces_a].ravel())
    sources = sources[sources >= 0]
    row = -np.ones(samples.graph.shape[0], dtype=int)
    if len(sources) == 0:
        return np.zeros((0, samples.graph.shape[0])), row
    row[sources] = np.arange(len(sources))
    return dijkstra(samples.graph, directed=False, indices=sources, limit=limit), row


def along_linework(samples: _Samples, a: np.ndarray, b: np.ndarray, limit: float = np.inf) -> np.ndarray:
    """Distance along the noded linework between samples a and b (inf if unconnected or beyond ``limit``)."""
    i, j = samples.piece[a], samples.piece[b]
    sa, sb = samples.arc[a], samples.arc[b]
    len_i, len_j = samples.piece_len[i], samples.piece_len[j]
    da = np.abs(sa - sb)
    same_piece = np.where(samples.closed[i], np.minimum(da, len_i - da), da)
    direct = np.where(i == j, same_piece, np.inf)
    via = np.full(len(a), np.inf)
    linked = (samples.ends[i, 0] >= 0) & (samples.ends[j, 0] >= 0)
    if linked.any() and samples.graph.shape[0]:
        ii, jj = i[linked], j[linked]
        node_dist, row = _node_distances(samples, ii, limit)
        to_a = (sa[linked], len_i[linked] - sa[linked])
        to_b = (sb[linked], len_j[linked] - sb[linked])
        best = np.full(int(linked.sum()), np.inf)
        for x in (0, 1):
            for y in (0, 1):
                d_nodes = node_dist[row[samples.ends[ii, x]], samples.ends[jj, y]]
                best = np.minimum(best, to_a[x] + d_nodes + to_b[y])
        via[linked] = best
    return np.minimum(direct, via)


def _circumcentres(xy: np.ndarray, simplices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Circumcentres and circumradii of the triangles, computed relative to their first vertex."""
    a = xy[simplices[:, 0]]
    b = xy[simplices[:, 1]] - a
    c = xy[simplices[:, 2]] - a
    d = 2.0 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    b2, c2 = np.sum(b * b, axis=1), np.sum(c * c, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = (c[:, 1] * b2 - b[:, 1] * c2) / d
        uy = (b[:, 0] * c2 - c[:, 0] * b2) / d
    centres = a + np.column_stack([ux, uy])
    return centres, np.hypot(ux, uy)


def _pair_is_gap(samples: _Samples, pa, pb, centres, cos_min: float, limit: float) -> np.ndarray:
    """Whether the sample pairs (pa, pb) are the two walls of a gap at ``centres``."""
    va = samples.xy[pa] - centres
    vb = samples.xy[pb] - centres
    norms = np.hypot(*va.T) * np.hypot(*vb.T)
    cos_angle = np.einsum("ij,ij->i", va, vb) / np.maximum(norms, 1e-300)
    candidate = (cos_angle <= cos_min) & ~samples.exempt[pa] & ~samples.exempt[pb]
    out = np.zeros(len(pa), dtype=bool)
    idx = np.flatnonzero(candidate)
    if len(idx):
        chord = np.hypot(*(samples.xy[pa[idx]] - samples.xy[pb[idx]]).T)
        out[idx] = along_linework(samples, pa[idx], pb[idx], limit) > ARC_FACTOR * chord
    return out


def medial_samples(domain, samples: _Samples, max_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Medial samples inside ``domain`` with radius below ``max_radius``, and their radii."""
    if len(samples.xy) < 3:
        return np.zeros((0, 2)), np.zeros(0)
    try:
        simplices = Delaunay(samples.xy).simplices
    except QhullError:
        # All samples collinear: there is no gap to measure.
        return np.zeros((0, 2)), np.zeros(0)
    centres, r = _circumcentres(samples.xy, simplices)
    keep = np.isfinite(r) & (r < max_radius)
    simplices, centres, r = simplices[keep], centres[keep], r[keep]
    inside = shapely.contains_xy(domain, centres[:, 0], centres[:, 1])
    simplices, centres, r = simplices[inside], centres[inside], r[inside]
    cos_min = np.cos(np.radians(MIN_ANGLE))
    # Along-linework distances only matter up to ARC_FACTOR x the widest chord.
    limit = ARC_FACTOR * 2.0 * max_radius
    is_gap = np.zeros(len(r), dtype=bool)
    for ia, ib in ((0, 1), (1, 2), (0, 2)):
        is_gap |= _pair_is_gap(samples, simplices[:, ia], simplices[:, ib], centres, cos_min, limit)
    return centres[is_gap], r[is_gap]


def smooth_radius(xy: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Median r over the medial samples inside each sample's ball (spike removal)."""
    if len(r) == 0:
        return r
    neighbours = cKDTree(xy).query_ball_point(xy, r)
    return np.array([np.median(r[nbrs]) if nbrs else r[i] for i, nbrs in enumerate(neighbours)])


def spec_sizes(xy: np.ndarray, sources: list, background_lc: float) -> np.ndarray:
    """Size the model's own feature fields give at xy (min over sources, capped at background_lc)."""
    h = np.full(len(xy), float(background_lc))
    if len(xy) == 0:
        return h
    pts = shapely.points(xy)
    for source in sources:
        d = shapely.distance(pts, source.geometry)
        h = np.minimum(h, source.size + source.gradient * d)
    return h


def estimate_gaps(walls: GapWalls, background_lc: float, cells_across: float, gradient: float,
                  min_size: float, size_sources: list = ()) -> GapEstimate:
    """Gap estimate for the walls: medial samples narrower than cells_across x background_lc."""
    domain = shapely.union_all([g for g in walls.polygons if g is not None and not g.is_empty])
    pieces, points = wall_pieces(walls)
    shapely.prepare(domain)
    samples = sample_walls(pieces, points, SPACING_FRACTION * min_size, SIMPLIFY_FRACTION * min_size,
                           exempt=walls.exempt, tol=EXEMPT_FRACTION * min_size)
    # Only gaps that could ask for less than the background matter.
    max_radius = 0.5 * cells_across * background_lc
    if domain.is_empty:
        xy, r_raw = np.zeros((0, 2)), np.zeros(0)
    else:
        xy, r_raw = medial_samples(domain, samples, max_radius)
    r = smooth_radius(xy, r_raw)
    size = np.clip(2.0 * r / cells_across, min_size, background_lc)
    return GapEstimate(xy=xy, r=r, size=size, spec=spec_sizes(xy, list(size_sources), background_lc),
                       cells_across=float(cells_across), min_size=float(min_size),
                       background_lc=float(background_lc), gradient=float(gradient),
                       n_wall_samples=len(samples.xy))


def floor_locations(est: GapEstimate, separation: float, max_locations: int = 5) -> tuple[int, list]:
    """Places where min_size binds: their count, and the narrowest few as (x, y, gap).

    Floor-bound medial samples closer than ``separation`` belong to one place;
    each place is reported at its narrowest sample.
    """
    idx = np.flatnonzero(est.floor_bound)
    if len(idx) == 0:
        return 0, []
    pairs = cKDTree(est.xy[idx]).query_pairs(separation, output_type="ndarray")
    adjacency = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
    n_places, place = connected_components(adjacency, directed=False)
    order = np.lexsort((est.gap[idx], place))
    first = np.ones(len(order), dtype=bool)
    first[1:] = place[order][1:] != place[order][:-1]
    narrowest = idx[order[first]]
    narrowest = narrowest[np.argsort(est.gap[narrowest], kind="stable")][:max_locations]
    return int(n_places), [(float(est.xy[k, 0]), float(est.xy[k, 1]), float(est.gap[k])) for k in narrowest]


def view_points(est: GapEstimate, bounds, margin: float) -> np.ndarray:
    """Adaptive points to sample h on: a background grid plus finer grids near active gaps.

    Level l has spacing a = background_lc / 2**l and is kept where h may lie
    in [a, 2a] (within r + (2a - s) / G + a of a medial sample with s <= 2a),
    down to a <= half the smallest active size. The medial samples are added too.
    """
    xmin, ymin, xmax, ymax = bounds
    xmin, ymin, xmax, ymax = xmin - margin, ymin - margin, xmax + margin, ymax + margin
    bg = est.background_lc
    gx, gy = np.arange(xmin, xmax + bg, bg), np.arange(ymin, ymax + bg, bg)
    pts = [np.column_stack([g.ravel() for g in np.meshgrid(gx, gy)])]
    act = est.active
    if act.any():
        m, r, s = est.xy[act], est.r[act], est.size[act]
        pts.append(m)
        for level in range(1, VIEW_MAX_LEVELS + 1):
            a = bg / 2 ** level
            near = s <= 2.0 * a
            if near.any():
                reach = r[near] + np.maximum(0.0, 2.0 * a - s[near]) / est.gradient + a
                pts.append(_grid_near(m[near], reach, a, xmin, ymin))
            if a <= 0.5 * s.min():
                break
    xy = np.concatenate(pts)
    inside = (xy[:, 0] >= xmin - 1e-9) & (xy[:, 0] <= xmax + bg) & (xy[:, 1] >= ymin - 1e-9) & (xy[:, 1] <= ymax + bg)
    return np.unique(np.round(xy[inside], 9), axis=0)


def _grid_near(centres: np.ndarray, reach: np.ndarray, a: float, xmin: float, ymin: float) -> np.ndarray:
    """Grid points of spacing a (anchored at xmin, ymin) within the box of each centre's reach."""
    # Thin the centres on a coarse grid first to limit duplicate boxes.
    key = np.floor(centres / max(a, 0.25 * reach.min())).astype(np.int64)
    _, uniq = np.unique(key, axis=0, return_index=True)
    cells = set()
    for (cx, cy), rad in zip(centres[uniq], reach[uniq]):
        i0, i1 = int(np.floor((cx - rad - xmin) / a)), int(np.ceil((cx + rad - xmin) / a))
        j0, j1 = int(np.floor((cy - rad - ymin) / a)), int(np.ceil((cy + rad - ymin) / a))
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        cells.update(zip(ii.ravel().tolist(), jj.ravel().tolist()))
    ij = np.array(sorted(cells), dtype=float)
    return np.column_stack([xmin + ij[:, 0] * a, ymin + ij[:, 1] * a])


def view_triangles(est: GapEstimate, bounds, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, triangles, h values) of the triangulated gap field."""
    xy = view_points(est, bounds, margin)
    triangles = Delaunay(xy).simplices
    return xy, triangles, est.sizes_at(xy)
