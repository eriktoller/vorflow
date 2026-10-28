"""Pre-meshing input checks behind ``ConceptualMesh.check()``.

The checks look for input geometry that later produces bad Voronoi cells:
line ends and points that just miss another feature, vertices or segments far
closer together than the local mesh size, thin polygon strips and lines that
run close to another feature. Each finding is measured against the local mesh
size ``h`` at the defect, as ``ratio = distance / h``, and is an error or a
warning by the thresholds in ``THRESHOLDS``.

``h`` approximates the size field Gmsh will aim for: the background size,
capped by every feature's size ramp (``size0`` up to ``plateau`` from the
feature, growing at ``gradient`` beyond it). Inside a polygon the distance is
zero, so ``h`` there is at most the polygon's resolution.

Only embedded features are checked; field-only features feed ``h`` alone.
Barrier, straddle and quad-buffer features are checked as the plain lines and
polygons they are drawn from.

Everything here is pure shapely/NumPy and works on clean GeoDataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point

from ._features import is_embedded, line_parts, polygon_parts

# (error below, warning below) on ratio = distance / h. None: never an error.
THRESHOLDS = {
    "short_segment": (0.1, 0.25),
    "close_vertices": (0.1, 0.5),
    "line_end_near_miss": (0.5, 1.0),
    "point_near_miss": (0.5, 1.0),
    "sliver_polygon": (0.5, 1.0),
    "narrow_gap": (None, 1.0),
}
# Kinds whose warning band includes ratio == warning threshold.
INCLUSIVE_WARNING = {"line_end_near_miss", "point_near_miss", "narrow_gap"}
SEVERITY_RANK = {"error": 0, "warning": 1}

# Points closer than this fraction of the model extent are one Gmsh point.
MERGE_FRACTION = 1e-9
# Gap checks sample every line and ring at this fraction of the local h.
GAP_SAMPLE_FRACTION = 0.25
# Gap samples closer than this many h to where the two features touch are a
# junction, not a gap.
GAP_CONTACT_EXCLUSION = 2.0
# Largest warning ratio of the gap checks; bounds the neighbour search.
GAP_REACH = 1.0
# Findings closer than this many local sizes h are one defect site.
DEDUPE_REACH = 2.0

COLUMNS = [
    "feature_type", "feature_id", "other_type", "other_id", "kind", "site_kinds", "severity",
    "distance", "h", "ratio", "x", "y", "geometry",
]
ID_COLUMNS = {"polygon": "zone_id", "line": "line_id", "point": "point_id"}


@dataclass(frozen=True)
class SizeField:
    """Target mesh size h(x): the background size capped by each feature's size ramp."""

    background_lc: float
    sources: tuple  # (geometry, size0, plateau, gradient)

    def at(self, xy) -> np.ndarray:
        """Local h at each (x, y) row of ``xy``."""
        pts = shapely.points(np.asarray(xy, dtype=float).reshape(-1, 2))
        h = np.full(len(pts), self.background_lc)
        for geometry, size0, plateau, gradient in self.sources:
            distance = shapely.distance(geometry, pts)
            h = np.minimum(h, size0 + gradient * np.maximum(distance - plateau, 0.0))
        return h


@dataclass(frozen=True)
class Chain:
    """A vertex chain Gmsh receives: a polygon ring (closed) or a line part (open)."""

    feature: tuple  # (feature_type, feature_id)
    geometry: object  # LineString or LinearRing
    xy: np.ndarray  # vertices, without the closing repeat of a ring
    closed: bool
    polygon: object = None  # the polygon part a ring bounds


@dataclass(frozen=True)
class Samples:
    """Points along a chain every GAP_SAMPLE_FRACTION x local h."""

    s: np.ndarray  # distance along the chain
    points: np.ndarray  # shapely points
    xy: np.ndarray
    h: np.ndarray


def size_field(sources, background_lc: float, default_gradient: float) -> SizeField:
    """SizeField from ``(geometry, (size0, plateau, gradient))`` ramps finer than the background.

    A ramp with an unknown gradient (None) grows at ``default_gradient``.
    """
    kept = []
    for geometry, (size0, plateau, gradient) in sources:
        if size0 is None or size0 >= background_lc:
            continue
        gradient = default_gradient if gradient is None else gradient
        kept.append((geometry, float(size0), float(plateau), float(gradient)))
    return SizeField(float(background_lc), tuple(kept))


def severity(kind: str, ratio: float) -> str | None:
    """'error', 'warning' or None (not reported) for a ratio of a check kind."""
    error_below, warning_below = THRESHOLDS[kind]
    if error_below is not None and ratio < error_below:
        return "error"
    if ratio < warning_below or (kind in INCLUSIVE_WARNING and ratio == warning_below):
        return "warning"
    return None


def _reported(kind: str, distance, h) -> np.ndarray:
    """Mask of distances that fall in the warning band of ``kind``."""
    warning_below = THRESHOLDS[kind][1]
    if kind in INCLUSIVE_WARNING:
        return distance <= warning_below * h
    return distance < warning_below * h


def _finding(kind, feature, other, distance, h, xy, geometry) -> dict | None:
    """One finding row, or None when the ratio is outside the reported bands."""
    ratio = float(distance) / float(h)
    level = severity(kind, ratio)
    if level is None:
        return None
    return {
        "feature_type": feature[0], "feature_id": feature[1],
        "other_type": None if other is None else other[0],
        "other_id": None if other is None else other[1],
        "kind": kind, "severity": level, "distance": float(distance), "h": float(h),
        "ratio": ratio, "x": float(xy[0]), "y": float(xy[1]), "geometry": geometry,
    }


def _embedded_rows(gdf) -> list:
    """(index, row) pairs of the embedded features of a clean GeoDataFrame."""
    if gdf is None or gdf.empty:
        return []
    return [(idx, row) for idx, row in gdf.iterrows() if is_embedded(row)]


def _feature(feature_type: str, row) -> tuple:
    """(feature_type, user id) of a clean row."""
    return feature_type, row.get(ID_COLUMNS[feature_type])


def build_chains(polys, lines) -> list[Chain]:
    """Rings of every embedded polygon part and every embedded line part."""
    chains = []
    for _, row in _embedded_rows(polys):
        for part in polygon_parts(row.geometry):
            for ring in [part.exterior, *part.interiors]:
                xy = np.asarray(ring.coords)[:-1, :2]
                chains.append(Chain(_feature("polygon", row), ring, xy, True, part))
    for _, row in _embedded_rows(lines):
        for part in line_parts(row.geometry):
            chains.append(Chain(_feature("line", row), part, np.asarray(part.coords)[:, :2], False))
    return chains


def embedded_points(points) -> list[tuple]:
    """(feature, Point) of every embedded point feature."""
    return [(_feature("point", row), row.geometry) for _, row in _embedded_rows(points)
            if not row.geometry.is_empty]


# ---------------------------------------------------------------- checks

def check_short_segments(chains, size: SizeField, tol: float) -> list[dict]:
    """Consecutive vertices of one ring or line closer than the short-segment band."""
    findings = []
    for chain in chains:
        start = chain.xy
        end = np.roll(chain.xy, -1, axis=0) if chain.closed else chain.xy[1:]
        start = start if chain.closed else start[:-1]
        length = np.hypot(*(end - start).T)
        mid = 0.5 * (start + end)
        h = size.at(mid)
        for i in np.flatnonzero((length > tol) & _reported("short_segment", length, h)):
            findings.append(_finding("short_segment", chain.feature, None, length[i], h[i], mid[i],
                                     LineString([start[i], end[i]])))
    return findings


def _crossings(chains) -> list[tuple]:
    """(line feature, other feature, xy) of points where a line crosses another feature's linework."""
    line_idx = [i for i, c in enumerate(chains) if not c.closed]
    if not line_idx:
        return []
    tree = shapely.STRtree([c.geometry for c in chains])
    crossings = []
    for i in line_idx:
        a = chains[i]
        for j in tree.query(a.geometry, predicate="intersects"):
            b = chains[j]
            if b.feature == a.feature or (not b.closed and j <= i):
                continue
            hit = shapely.intersection(a.geometry, b.geometry)
            pts = [g for g in getattr(hit, "geoms", [hit]) if g.geom_type == "Point"]
            if pts:
                crossings.append((a.feature, b.feature, shapely.get_coordinates(pts)))
    return crossings


def check_close_vertices(chains, points, size: SizeField, tol: float) -> list[dict]:
    """Distinct Gmsh points closer than the close-vertices band.

    Gmsh points are chain vertices, point features and line crossings.
    Coincident points (within ``tol``) are one point after fragmentation.
    Two vertices of one chain count only when they are far apart along it (a
    fold or neck); neighbours along a chain are left to the short-segment check.
    """
    # owners[k] = (feature, partner feature, chain index or None)
    xy, owner, pos, owners = [], [], [], []
    for c, chain in enumerate(chains):
        xy.append(chain.xy)
        owner += [len(owners)] * len(chain.xy)
        pos += list(range(len(chain.xy)))
        owners.append((chain.feature, None, c))
    for feature, point in points:
        xy.append(np.array([[point.x, point.y]]))
        owner.append(len(owners))
        pos.append(0)
        owners.append((feature, None, None))
    for feature, partner, coords in _crossings(chains):
        xy.append(coords[:, :2])
        owner += [len(owners)] * len(coords)
        pos += [0] * len(coords)
        owners.append((feature, partner, None))
    if not xy:
        return []
    xy = np.vstack(xy)
    owner, pos = np.asarray(owner), np.asarray(pos)

    keys = np.round(xy / tol).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    pts = shapely.points(xy[first])
    reach = THRESHOLDS["close_vertices"][1] * size.background_lc
    left, right = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=reach)
    keep = left < right
    left, right = first[left[keep]], first[right[keep]]
    if len(left) == 0:
        return []
    distance = np.hypot(*(xy[left] - xy[right]).T)
    mid = 0.5 * (xy[left] + xy[right])
    h = size.at(mid)
    candidate = (distance > tol) & _reported("close_vertices", distance, h)

    arcs = {c: _arc_lengths(chain) for c, chain in enumerate(chains)}
    findings = []
    for a, b, d, m, hh in zip(left[candidate], right[candidate], distance[candidate],
                              mid[candidate], h[candidate]):
        feature_a, partner_a, chain_a = owners[owner[a]]
        feature_b, partner_b, chain_b = owners[owner[b]]
        if chain_a is not None and chain_a == chain_b:
            arc = abs(arcs[chain_a][pos[a]] - arcs[chain_a][pos[b]])
            if chains[chain_a].closed:
                arc = min(arc, arcs[chain_a][-1] - arc)
            if arc < max(3.0 * d, 2.0 * hh):
                continue
        other = feature_b
        if other == feature_a:
            other = partner_b or partner_a or feature_b
        findings.append(_finding("close_vertices", feature_a, other, d, hh, m,
                                 LineString([xy[a], xy[b]])))
    return findings


def _arc_lengths(chain: Chain) -> np.ndarray:
    """Distance along a chain to each vertex; a ring's last entry is its length."""
    xy = np.vstack([chain.xy, chain.xy[:1]]) if chain.closed else chain.xy
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def check_near_misses(chains, points, size: SizeField, tol: float) -> list[dict]:
    """Line ends and point features that touch nothing but lie within the near-miss band of a feature."""
    targets = [(c.feature, c.geometry) for c in chains] + list(points)
    if not targets:
        return []
    tree = shapely.STRtree([g for _, g in targets])
    reach = THRESHOLDS["line_end_near_miss"][1] * size.background_lc
    probes = []
    for chain in chains:
        if not chain.closed:
            for xy in (chain.xy[0], chain.xy[-1]):
                probes.append(("line_end_near_miss", chain.feature, Point(xy)))
    probes += [("point_near_miss", feature, point) for feature, point in points]

    findings = []
    for kind, feature, probe in probes:
        near = [targets[j] for j in tree.query(probe, predicate="dwithin", distance=reach)
                if targets[j][0] != feature]
        if not near:
            continue
        distances = [shapely.distance(probe, g) for _, g in near]
        if min(distances) <= tol:  # snapped onto another feature
            continue
        k = int(np.argmin(distances))
        h = size.at([[probe.x, probe.y]])[0]
        if _reported(kind, distances[k], h):
            findings.append(_finding(kind, feature, near[k][0], distances[k], h, (probe.x, probe.y),
                                     shapely.shortest_line(probe, near[k][1])))
    return findings


def check_thin_parts(polys, size: SizeField, tol: float) -> list[dict]:
    """Whole embedded polygon parts narrower than the sliver band (maximum inscribed circle)."""
    findings = []
    for _, row in _embedded_rows(polys):
        for part in polygon_parts(row.geometry):
            circle = shapely.maximum_inscribed_circle(part, tolerance=1e-3 * np.sqrt(part.area))
            centre = circle.coords[0]
            width = 2.0 * circle.length
            h = size.at([centre])[0]
            if width > tol and _reported("sliver_polygon", width, h):
                findings.append(_finding("sliver_polygon", _feature("polygon", row), None, width, h,
                                         centre, part))
    return findings


def _samples(chain: Chain, size: SizeField) -> Samples:
    """Samples along a chain every GAP_SAMPLE_FRACTION x local h."""
    geometry = chain.geometry
    n_coarse = max(int(np.ceil(geometry.length / (GAP_SAMPLE_FRACTION * size.background_lc))), 1)
    coarse = np.linspace(0.0, geometry.length, n_coarse + 1)
    h_coarse = size.at(shapely.get_coordinates(shapely.line_interpolate_point(geometry, coarse)))
    s = [coarse[:1]]
    for k in range(n_coarse):
        step = GAP_SAMPLE_FRACTION * min(h_coarse[k], h_coarse[k + 1])
        n = max(int(np.ceil((coarse[k + 1] - coarse[k]) / step)), 1)
        s.append(np.linspace(coarse[k], coarse[k + 1], n + 1)[1:])
    s = np.concatenate(s)
    pts = shapely.line_interpolate_point(geometry, s)
    xy = shapely.get_coordinates(pts)
    return Samples(s, pts, xy, size.at(xy))


def _away_from_ends(chain: Chain, samples: Samples) -> np.ndarray:
    """Mask of samples more than 1 h from a line's ends (near misses cover the ends)."""
    if chain.closed:
        return np.ones(len(samples.s), dtype=bool)
    ends = shapely.multipoints([chain.xy[0], chain.xy[-1]])
    return shapely.distance(ends, samples.points) > samples.h


def _runs(idx: np.ndarray, s: np.ndarray, max_step: float) -> list[np.ndarray]:
    """Split sorted sample indices into runs of neighbouring samples."""
    if len(idx) == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(s[idx]) > max_step) + 1)


def _gap_finding(kind, feature, other, samples: Samples, run, k, distance, transect) -> dict:
    """A gap finding at sample k; its geometry spans the flagged run and the transect."""
    stretch = LineString(samples.xy[run]) if len(run) > 1 else Point(samples.xy[run[0]])
    return _finding(kind, feature, other, distance, samples.h[k], samples.xy[k],
                    shapely.union(stretch, transect))


def check_pair_gaps(chains, samples, size: SizeField, tol: float) -> list[dict]:
    """Stretches of a chain within the gap band of another chain, away from where they touch.

    A gap with a line on either side is a ``narrow_gap``; one between two
    polygon rings is a thin strip of polygon, a ``sliver_polygon``, and
    counts only when the gap lies inside one of the two polygons.
    """
    findings = []
    tree = shapely.STRtree([c.geometry for c in chains])
    reach = GAP_REACH * size.background_lc
    for a, chain_a in enumerate(chains):
        sa = samples[a]
        max_step = 2.0 * GAP_SAMPLE_FRACTION * sa.h.max()
        for b in tree.query(chain_a.geometry, predicate="dwithin", distance=reach):
            chain_b = chains[b]
            if b == a or (chain_a.closed == chain_b.closed and b < a):
                continue
            if chain_a.closed and not chain_b.closed:  # the line takes side a, so its ends are excluded
                continue
            kind = "sliver_polygon" if chain_a.closed and chain_b.closed else "narrow_gap"
            distance = shapely.distance(chain_b.geometry, sa.points)
            flag = (distance > tol) & _reported(kind, distance, sa.h) & _away_from_ends(chain_a, sa)
            if flag.any() and kind == "sliver_polygon":
                nearest = shapely.get_coordinates(shapely.shortest_line(sa.points, chain_b.geometry))
                mid = shapely.points(0.5 * (nearest[0::2] + nearest[1::2]))
                flag &= shapely.contains(chain_a.polygon, mid) | shapely.contains(chain_b.polygon, mid)
            if not flag.any():
                continue
            contact = shapely.intersection(chain_a.geometry, chain_b.geometry)
            if not contact.is_empty:
                flag &= shapely.distance(contact, sa.points) > GAP_CONTACT_EXCLUSION * sa.h
            for run in _runs(np.flatnonzero(flag), sa.s, max_step):
                k = run[np.argmin(distance[run])]
                transect = shapely.shortest_line(sa.points[k], chain_b.geometry)
                findings.append(_gap_finding(kind, chain_a.feature, chain_b.feature, sa, run, k,
                                             distance[k], transect))
    return findings


def check_self_gaps(chains, samples, tol: float) -> list[dict]:
    """Folds and necks: two stretches of one chain within the gap band, far apart along it.

    On a ring the gap must lie inside the polygon (a thin strip of it, not a
    narrow notch outside it).
    """
    findings = []
    for chain, sa in zip(chains, samples):
        kind = "sliver_polygon" if chain.closed else "narrow_gap"
        warning_below = THRESHOLDS[kind][1]
        i, j = shapely.STRtree(sa.points).query(sa.points, predicate="dwithin",
                                                distance=warning_below * sa.h.max())
        keep = i < j
        i, j = i[keep], j[keep]
        distance = np.hypot(*(sa.xy[i] - sa.xy[j]).T)
        arc = np.abs(sa.s[i] - sa.s[j])
        if chain.closed:
            arc = np.minimum(arc, chain.geometry.length - arc)
        h = np.minimum(sa.h[i], sa.h[j])
        away = _away_from_ends(chain, sa)
        ok = ((distance > tol) & _reported(kind, distance, h) & (arc > np.maximum(3.0 * distance, 2.0 * h))
              & away[i] & away[j])
        if chain.closed and ok.any():
            mid = shapely.points(0.5 * (sa.xy[i] + sa.xy[j]))
            ok &= shapely.contains(chain.polygon, mid)
        if not ok.any():
            continue
        best = {}
        for ii, jj, dd in zip(i[ok], j[ok], distance[ok]):
            if ii not in best or dd < best[ii][1]:
                best[ii] = (jj, dd)
        idx = np.array(sorted(best))
        max_step = 2.0 * GAP_SAMPLE_FRACTION * sa.h.max()
        for run in _runs(idx, sa.s, max_step):
            k = min(run, key=lambda q: best[q][1])
            jj, dd = best[k]
            transect = LineString([sa.xy[k], sa.xy[jj]])
            findings.append(_gap_finding(kind, chain.feature, chain.feature, sa, run, k, dd, transect))
    return findings


# ---------------------------------------------------------------- results

def deduplicate(findings: list[dict]) -> list[dict]:
    """One finding per defect site: the most severe of each cluster.

    Findings whose geometries lie within DEDUPE_REACH x the smaller of their
    two local sizes h of each other belong to one site (clusters chain
    through shared members). The kept finding is the most severe, then the
    lowest ratio; its ``site_kinds`` lists every kind found at the site.
    """
    if len(findings) < 2:
        return [{**f, "site_kinds": f["kind"]} for f in findings]
    geoms = np.array([f["geometry"] for f in findings], dtype=object)
    h = np.array([f["h"] for f in findings])
    left, right = shapely.STRtree(geoms).query(geoms, predicate="dwithin",
                                               distance=DEDUPE_REACH * h.max())
    near = shapely.distance(geoms[left], geoms[right]) <= DEDUPE_REACH * np.minimum(h[left], h[right])
    n = len(findings)
    graph = coo_matrix((np.ones(near.sum()), (left[near], right[near])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    best, kinds = {}, {}
    for label, finding in zip(labels, findings):
        kinds.setdefault(label, set()).add(finding["kind"])
        rank = (SEVERITY_RANK[finding["severity"]], finding["ratio"])
        if label not in best or rank < best[label][0]:
            best[label] = (rank, finding)
    return [{**finding, "site_kinds": ", ".join(sorted(kinds[label]))}
            for label, (_, finding) in best.items()]


def to_frame(findings: list[dict], crs) -> gpd.GeoDataFrame:
    """Findings as a GeoDataFrame, errors first, then by ratio."""
    if not findings:
        return gpd.GeoDataFrame(columns=COLUMNS, geometry="geometry", crs=crs)
    frame = gpd.GeoDataFrame(findings, columns=COLUMNS, geometry="geometry", crs=crs)
    frame["_rank"] = frame["severity"].map(SEVERITY_RANK)
    frame = frame.sort_values(["_rank", "ratio"], kind="mergesort").drop(columns="_rank")
    return frame.reset_index(drop=True)


def merge_tolerance(chains, points) -> float:
    """Distance below which two points are one Gmsh point: MERGE_FRACTION x model extent."""
    geoms = [c.geometry for c in chains] + [p for _, p in points]
    if not geoms:
        return MERGE_FRACTION
    xmin, ymin, xmax, ymax = shapely.total_bounds(geoms)
    return MERGE_FRACTION * max(xmax - xmin, ymax - ymin, 1.0)


def find_defects(polys, lines, points, size: SizeField, connectivity_tolerance: float, crs=None):
    """Run every check on clean frames; return the deduplicated findings as a GeoDataFrame."""
    chains = build_chains(polys, lines)
    point_features = embedded_points(points)
    tol = merge_tolerance(chains, point_features)
    findings = check_near_misses(chains, point_features, size, connectivity_tolerance)
    findings += check_close_vertices(chains, point_features, size, tol)
    findings += check_short_segments(chains, size, tol)
    findings += check_thin_parts(polys, size, tol)
    samples = [_samples(chain, size) for chain in chains]
    findings += check_pair_gaps(chains, samples, size, tol)
    findings += check_self_gaps(chains, samples, tol)
    findings = [f for f in findings if f is not None]
    return to_frame(deduplicate(findings), crs)
