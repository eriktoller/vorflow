"""Concave-cell checks and the optional split of concave cells into convex pieces.

Clipping the Voronoi diagram to the domain makes a cell concave wherever the
domain boundary has a reflex vertex (a hole corner, or an outer-ring corner
that bends into the domain): Gmsh puts a node on that vertex, so its cell is
cut into an L or V shape with the generator on the reflex vertex.

MODFLOW 6 PRT tracks particles through a DISV cell on a fan of triangles that
all share the cell centroid, one per face. It states no convexity condition,
but if a face is not visible from the centroid one of those triangles has a
negative area, and PRT can loop forever in that cell without an error.
``centroid_fan_fraction`` measures this: the smallest fan triangle relative to
the cell area, negative when a face is hidden from the centroid.

``split_concave_cells`` cuts each concave cell along diagonals from its reflex
vertices until every piece is convex. A diagonal joins two existing vertices,
so the neighbouring cells do not change. When no diagonal resolves a reflex
vertex (common where the cell has few vertices opposite the corner), the cut
follows the angle bisector to the boundary and the new vertex is inserted into
the neighbour that shares that face: the neighbour keeps its shape but gains a
vertex. A barrier face never gains one, as each barrier segment must stay one
face: where the bisector ends inside a barrier face, two diagonals to the ends
of that face cut it off instead (both lie either side of the bisector, so they
resolve the corner). The cut always starts at the reflex vertex, where the generator sits,
so the generator would lie at the end of the new face; every piece therefore
takes its centroid as x/y. A zero-width spike (a vertex where the ring turns
straight back, as a barrier end can leave) is dropped first; the base of the
spike is inserted into the neighbour in the same way. A cell that cannot be
split (one with holes, or one where no cut makes progress) is left as it is
with a warning.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import split
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

# A cell is concave when its area is below this fraction short of its convex
# hull's area (convexity < 1 - CONCAVE_TOLERANCE). Clipping leaves vertices
# within roundoff of collinear, so exact convexity is not a usable test.
CONCAVE_TOLERANCE = 1e-6

# A vertex is reflex when the cross product of its edges, relative to their
# lengths (the sine of the turn), is below minus this.
REFLEX_TOLERANCE = 1e-9

# Distance, relative to the cell perimeter, used to test that a diagonal lies
# inside its polygon, to snap cut ends and to check that a split preserved
# the area.
SPLIT_TOLERANCE = 1e-9

# Added distance relative to the coordinate magnitude: roundoff grows with the
# coordinates (e.g. UTM northings), whatever the cell size.
COORDINATE_TOLERANCE = 1e-12


class SplitError(ValueError):
    """A concave cell that cannot be cut into convex pieces."""


def centroid_fan_fraction(geoms: np.ndarray) -> np.ndarray:
    """Smallest centroid-fan triangle of each polygon over its area.

    NaN for anything other than a non-empty single polygon without holes with
    a positive area: a hole hides faces, and such cells are not DISV cells.
    Zero-length edges (repeated vertices) have no fan triangle and are skipped.
    """
    geoms = np.asarray(geoms, dtype=object)
    out = np.full(len(geoms), np.nan)
    simple = (
        (shapely.get_type_id(geoms) == shapely.GeometryType.POLYGON)
        & ~shapely.is_empty(geoms)
        & (shapely.get_num_interior_rings(geoms) == 0)
        & (shapely.area(geoms) > 0.0)
    )
    if not simple.any():
        return out
    polys = geoms[simple]
    centroids = shapely.get_coordinates(shapely.centroid(polys))
    coords, ring_of = shapely.get_coordinates(shapely.get_exterior_ring(polys), return_index=True)
    # Consecutive coordinates of one closed ring are its edges.
    edge = (ring_of[:-1] == ring_of[1:]) & np.any(coords[:-1] != coords[1:], axis=1)
    owner = ring_of[:-1][edge]
    a = coords[:-1][edge] - centroids[owner]
    b = coords[1:][edge] - centroids[owner]
    twice_fan = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    twice_area = np.bincount(owner, weights=twice_fan, minlength=len(polys))
    # Orient every ring counter-clockwise: positive fan triangles face the centroid.
    sign = np.sign(twice_area)
    fraction = twice_fan * sign[owner] / np.abs(twice_area[owner])
    smallest = np.full(len(polys), np.inf)
    np.minimum.at(smallest, owner, fraction)
    out[simple] = smallest
    return out


def _ring(poly: Polygon) -> np.ndarray:
    """Counter-clockwise exterior vertices without the closing repeat or repeated vertices."""
    xy = np.asarray(orient(poly, 1.0).exterior.coords)[:-1]
    return xy[np.any(xy != np.roll(xy, 1, axis=0), axis=1)]


def _tolerance(poly: Polygon) -> float:
    """Distance below which two points of ``poly`` count as the same."""
    return SPLIT_TOLERANCE * poly.length + COORDINATE_TOLERANCE * float(np.abs(poly.bounds).max())


def _turns(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sine and cosine of the turn at each vertex of a ring."""
    e1 = xy - np.roll(xy, 1, axis=0)
    e2 = np.roll(xy, -1, axis=0) - xy
    scale = np.hypot(*e1.T) * np.hypot(*e2.T)
    sine = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / scale
    cosine = (e1 * e2).sum(axis=1) / scale
    return sine, cosine


def reflex_vertices(poly: Polygon) -> list[int]:
    """Positions in ``_ring(poly)`` of the vertices with an interior angle above 180 degrees."""
    sine, _ = _turns(_ring(poly))
    return np.flatnonzero(sine < -REFLEX_TOLERANCE).tolist()


def _drop_spikes(poly: Polygon) -> tuple[Polygon, list[np.ndarray]]:
    """``poly`` without zero-width spikes, and the vertices either side of each removed tip.

    A spike tip is a vertex where the ring turns straight back on itself.
    """
    xy = _ring(poly)
    bases = []
    while len(xy) > 3:
        sine, cosine = _turns(xy)
        tips = np.flatnonzero((np.abs(sine) <= REFLEX_TOLERANCE) & (cosine < 0.0))
        if len(tips) == 0:
            break
        tip = tips[0]
        bases.extend([xy[tip - 1], xy[(tip + 1) % len(xy)]])
        xy = np.delete(xy, tip, axis=0)
        # Removing a tip can leave its two bases on one point.
        xy = xy[np.any(xy != np.roll(xy, 1, axis=0), axis=1)]
    if not bases:
        return poly, bases
    if len(xy) < 3:
        raise SplitError("the cell collapses once its zero-width spikes are removed")
    return Polygon(xy), bases


def split_pinches(geoms: np.ndarray, tolerance: float) -> np.ndarray:
    """``geoms`` with each polygon whose exterior passes twice through one point split there into its lobes.

    Clipping a cell around a sharp notch tip can compute the tip twice, a
    roundoff apart, so the ring touches itself through a neck of near-zero
    width that GEOS still counts as valid. Two exterior vertices of one
    polygon closer than ``tolerance`` that are not neighbours on the ring
    are such a pinch. The pinched polygon becomes a MultiPolygon of its
    lobes, which the caller explodes into cells; a polygon whose fix would
    change its area or leave a lobe with a hole is kept as it is.
    """
    geoms = np.asarray(geoms, dtype=object).copy()
    polygons = np.flatnonzero(shapely.get_type_id(geoms) == shapely.GeometryType.POLYGON)
    coords, owner = shapely.get_coordinates(shapely.get_exterior_ring(geoms[polygons]), return_index=True)
    # Without the closing repeat, a ring's first and last vertices are neighbours too.
    closing = np.r_[owner[1:] != owner[:-1], True]
    coords, owner = coords[~closing], owner[~closing]
    pairs = cKDTree(coords).query_pairs(tolerance, output_type="ndarray")
    if len(pairs) == 0:
        return geoms
    ring_size = np.bincount(owner)[owner[pairs[:, 0]]]
    apart = np.ptp(pairs, axis=1)
    pinch = (owner[pairs[:, 0]] == owner[pairs[:, 1]]) & (apart > 1) & (apart < ring_size - 1)
    for k in polygons[np.unique(owner[pairs[pinch, 0]])]:
        geoms[k] = _split_pinch(geoms[k], tolerance)
    return geoms


def _split_pinch(poly: Polygon, tolerance: float):
    """``poly`` split into its lobes where non-neighbouring exterior vertices lie within ``tolerance``."""
    xy = np.asarray(poly.exterior.coords)[:-1]
    n = len(xy)
    for i, j in cKDTree(xy).query_pairs(tolerance, output_type="ndarray"):
        if 1 < abs(int(j) - int(i)) < n - 1:
            xy[max(i, j)] = xy[min(i, j)]
    parts = [part for part in shapely.get_parts(make_valid(Polygon(xy, poly.interiors)))
             if part.geom_type == "Polygon" and part.area > 0.0]
    if len(parts) < 2 or any(part.interiors for part in parts):
        return poly
    if abs(sum(part.area for part in parts) - poly.area) > SPLIT_TOLERANCE * poly.area:
        return poly
    return MultiPolygon(parts)


def _compactness(poly: Polygon) -> float:
    return 4 * np.pi * poly.area / poly.length**2


def _split_ring(xy: np.ndarray, i: int, j: int) -> tuple[Polygon, Polygon]:
    """The two polygons on either side of the diagonal between ring vertices i and j."""
    n = len(xy)
    first = [xy[k % n] for k in range(i, j + 1 if j > i else j + n + 1)]
    second = [xy[k % n] for k in range(j, i + 1 if i > j else i + n + 1)]
    return Polygon(first), Polygon(second)


def _resolves(pieces: tuple[Polygon, Polygon], vertex: np.ndarray) -> bool:
    """True when ``vertex`` is reflex in neither piece."""
    # The pieces reuse the ring coordinates, so the vertex is matched exactly.
    return not any(np.array_equal(_ring(p)[k], vertex) for p in pieces for k in reflex_vertices(p))


def _best_diagonal(poly: Polygon, r: int):
    """((piece, piece), score) for the best diagonal from reflex vertex r that resolves it, or None.

    Fewer reflex vertices left is better, then a more compact smaller piece.
    """
    xy = _ring(poly)
    n = len(xy)
    tolerance = _tolerance(poly)
    shrunk = poly.buffer(-tolerance)
    best = None
    for j in range(n):
        if j in (r, (r - 1) % n, (r + 1) % n):
            continue
        diagonal = LineString([xy[r], xy[j]])
        if not (shrunk.intersects(diagonal) and poly.covers(diagonal)):
            continue
        pieces = _split_ring(xy, r, j)
        if not all(p.is_valid for p in pieces):
            continue
        if abs(sum(p.area for p in pieces) - poly.area) > tolerance * poly.length:
            continue
        if not _resolves(pieces, xy[r]):
            continue
        score = (len(reflex_vertices(pieces[0])) + len(reflex_vertices(pieces[1])),
                 -min(_compactness(p) for p in pieces))
        if best is None or score < best[1]:
            best = (pieces, score)
    return best


def _bisector_cut(poly: Polygon, r: int, barriers=None) -> tuple[list[Polygon], np.ndarray | None]:
    """Pieces from cutting along the bisector of reflex vertex r, and the new vertex where it ends.

    Where the bisector ends on a ``barriers`` line away from a vertex (inside
    a barrier face, which the new vertex would cut in two), two diagonals to
    the ends of that face cut it off instead, and no vertex is added (the
    returned end is None).
    """
    xy = _ring(poly)
    n = len(xy)
    vertex = xy[r]
    to_prev = xy[(r - 1) % n] - vertex
    to_next = xy[(r + 1) % n] - vertex
    bisector = -(to_prev / np.hypot(*to_prev) + to_next / np.hypot(*to_next))
    bisector /= np.hypot(*bisector)
    tolerance = _tolerance(poly)
    ray = LineString([vertex, vertex + bisector * 2 * poly.length])
    hits = shapely.get_parts(ray.intersection(poly.boundary))
    ends = [np.asarray(h.coords[0]) for h in hits
            if h.geom_type == "Point" and h.distance(Point(vertex)) > tolerance]
    if not ends:
        raise SplitError("the bisector of a reflex vertex meets no other boundary point")
    end = min(ends, key=lambda p: np.hypot(*(p - vertex)))
    if _inside_barrier_face(end, xy, barriers, tolerance):
        return _barrier_face_fan(poly, r, end), None
    snapped = shapely.snap(poly, Point(end), tolerance)
    pieces = list(shapely.get_parts(split(snapped, LineString([vertex, end]))))
    _check_progress(poly, pieces)
    return pieces, end


def _barrier_face_fan(poly: Polygon, r: int, end: np.ndarray) -> list[Polygon]:
    """Pieces from diagonals joining reflex vertex r to both ends of the edge that holds ``end``.

    ``end`` is where the bisector of r meets that edge, so the two diagonals
    lie either side of the bisector and leave every corner at r below 180
    degrees, without a new vertex.
    """
    xy = _ring(poly)
    n = len(xy)
    tolerance = _tolerance(poly)
    edges = [k for k in range(n) if LineString([xy[k], xy[(k + 1) % n]]).distance(Point(end)) <= tolerance]
    if not edges:
        raise SplitError("the bisector of a reflex vertex ends on no edge")
    # Unwrapped ring positions r < a < a + 1 < r + n of the edge's ends.
    a = edges[0] if edges[0] > r else edges[0] + n
    shrunk = poly.buffer(-tolerance)
    for j in (a, a + 1):
        diagonal = LineString([xy[r], xy[j % n]])
        is_edge = j - r < 2 or j - r > n - 2
        if not is_edge and not (shrunk.intersects(diagonal) and poly.covers(diagonal)):
            raise SplitError("the bisector of a reflex vertex ends inside a barrier face it cannot cut off")
    rings = ([xy[k % n] for k in range(r, a + 1)], [xy[r], xy[a % n], xy[(a + 1) % n]],
             [xy[k % n] for k in range(a + 1, r + n + 1)])
    pieces = [Polygon(ring) for ring in rings if len(ring) >= 3]
    _check_progress(poly, pieces)
    return pieces


def _inside_barrier_face(point: np.ndarray, xy: np.ndarray, barriers, tolerance: float) -> bool:
    """True when ``point`` lies on ``barriers`` but on no vertex of the ring ``xy``."""
    if barriers is None or barriers.distance(Point(point)) > tolerance:
        return False
    return bool(np.hypot(*(xy - point).T).min() > tolerance)


def _check_progress(poly: Polygon, pieces: list[Polygon]) -> None:
    """Raise SplitError unless ``pieces`` are two or more valid polygons that tile ``poly``."""
    if len(pieces) < 2 or not all(p.geom_type == "Polygon" and p.is_valid and p.area > 0 for p in pieces):
        raise SplitError("a cut did not split the cell")
    if abs(sum(p.area for p in pieces) - poly.area) > _tolerance(poly) * poly.length:
        raise SplitError("a cut changed the cell area")


def convex_pieces(poly: Polygon, barriers=None) -> tuple[list[Polygon], list[np.ndarray]]:
    """Convex pieces of ``poly`` and the boundary vertices its neighbours need.

    Those are the ends of bisector cuts and the bases of removed spikes. No
    cut ends inside a face on the ``barriers`` lines, so each barrier face
    stays one face. Raises SplitError for a polygon with holes or when no cut
    makes progress.
    """
    if poly.geom_type != "Polygon" or poly.is_empty or len(poly.interiors) > 0:
        raise SplitError("only single polygons without holes can be split")
    poly, added = _drop_spikes(poly)
    stack, pieces = [poly], []
    # Every cut resolves one reflex vertex; the limit only guards against a bug.
    cuts_left = 4 * len(poly.exterior.coords)
    while stack:
        if cuts_left == 0:
            raise SplitError("too many cuts")
        piece = stack.pop()
        reflex = reflex_vertices(piece)
        if not reflex:
            pieces.append(piece)
            continue
        candidates = [c for c in (_best_diagonal(piece, r) for r in reflex) if c is not None]
        cuts_left -= 1
        if candidates:
            stack.extend(min(candidates, key=lambda c: c[1])[0])
        else:
            cut, end = _first_bisector_cut(piece, reflex, barriers)
            if end is not None:
                added.append(end)
            stack.extend(cut)
    return pieces, added


def _first_bisector_cut(poly: Polygon, reflex: list[int], barriers=None) -> tuple[list[Polygon], np.ndarray]:
    """The bisector cut of the first reflex vertex that has one; re-raise the last failure."""
    for r in reflex:
        try:
            return _bisector_cut(poly, r, barriers)
        except SplitError as error:
            last = error
    raise last


def _insert_vertex(poly: Polygon, point: np.ndarray, tolerance: float) -> Polygon:
    """``poly`` with ``point`` inserted into the exterior edge it lies inside, keeping the orientation."""
    coords = np.asarray(poly.exterior.coords)[:-1]
    for k in range(len(coords)):
        a, b = coords[k], coords[(k + 1) % len(coords)]
        on_edge = LineString([a, b]).distance(Point(point)) < tolerance
        if on_edge and min(np.hypot(*(point - a)), np.hypot(*(point - b))) > tolerance:
            return Polygon(np.insert(coords, k + 1, point, axis=0), poly.interiors)
    return poly


def split_concave_cells(grid, targets: np.ndarray, barriers=None) -> tuple:
    """Split the cells at row positions ``targets`` into convex pieces.

    The largest piece covering the generator (else the largest) keeps the
    row and its ``node_id``; the others are new rows with fresh IDs above the
    maximum. Every piece of a split cell takes its centroid as x/y and is
    marked in a ``concave_split`` column; the mesh node stays in
    ``source_x``/``source_y``, which are added (equal to x/y) when the grid
    lacks them. No cut ends inside a face on the ``barriers`` lines (a
    shapely geometry, or None). A cell that cannot be split stays as it is,
    with a warning. Returns the new grid
    (fresh RangeIndex) and a report DataFrame with one row per split cell:
    ``node_id``, ``convexity``, ``n_pieces``, ``piece_ids`` and
    ``min_piece_fraction`` (smallest piece over the cell area).
    """
    grid = grid.reset_index(drop=True).copy()
    grid["concave_split"] = False
    if "source_x" not in grid.columns:
        grid["source_x"], grid["source_y"] = grid["x"].astype(float), grid["y"].astype(float)
    if len(targets) == 0:
        return grid, _report_frame([])
    geometry = grid.geometry.name
    max_id = int(grid["node_id"].max())
    new_rows, report, added, failed = [], [], [], []
    for position in targets:
        row = grid.iloc[position]
        cell = row[geometry]
        try:
            pieces, cut_ends = convex_pieces(cell, barriers)
        except SplitError as error:
            failed.append(f"{row['node_id']} ({error})")
            continue
        added.extend(cut_ends)
        # Keep the cell's vertex order (MODFLOW 6 wants clockwise rings).
        sign = 1.0 if cell.exterior.is_ccw else -1.0
        pieces = [orient(p, sign) for p in pieces]
        generator = Point(float(row["x"]), float(row["y"]))
        covering = [k for k, p in enumerate(pieces) if p.buffer(SPLIT_TOLERANCE * p.length).covers(generator)]
        primary = max(covering or range(len(pieces)), key=lambda k: pieces[k].area)
        others = sorted((p for k, p in enumerate(pieces) if k != primary), key=lambda p: p.area, reverse=True)

        grid.at[position, geometry] = pieces[primary]
        grid.at[position, "concave_split"] = True
        grid.at[position, "x"] = pieces[primary].centroid.x
        grid.at[position, "y"] = pieces[primary].centroid.y
        piece_ids = [int(row["node_id"])]
        for piece in others:
            max_id += 1
            new = row.copy()
            new[geometry] = piece
            new["node_id"] = max_id
            new["concave_split"] = True
            new["x"], new["y"] = piece.centroid.x, piece.centroid.y
            new_rows.append(new)
            piece_ids.append(max_id)
        report.append({
            "node_id": int(row["node_id"]),
            "convexity": cell.area / cell.convex_hull.area,
            "n_pieces": len(pieces),
            "piece_ids": piece_ids,
            "min_piece_fraction": min(p.area for p in pieces) / cell.area,
        })

    if new_rows:
        node_dtype = grid["node_id"].dtype
        grid = pd.concat([grid, type(grid)(new_rows, crs=grid.crs)], ignore_index=True)
        grid["node_id"] = grid["node_id"].astype(node_dtype)
    if added:
        grid = _insert_cut_ends(grid, added)
    if failed:
        logger.warning(f"Left {len(failed)} concave cells unsplit, by node_id: {', '.join(failed)}.")
    return grid, _report_frame(report)


def _report_frame(rows: list[dict]) -> pd.DataFrame:
    """The split report, with the same column types whether or not it has rows."""
    dtypes = {"node_id": "int64", "convexity": "float64", "n_pieces": "int64",
              "piece_ids": "object", "min_piece_fraction": "float64"}
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)


def _insert_cut_ends(grid, points: list[np.ndarray]):
    """Grid with each bisector cut end inserted into the neighbouring cell edge it lies on."""
    bounds = np.asarray(grid.total_bounds).reshape(2, 2)
    tolerance = (SPLIT_TOLERANCE * max(float(np.ptp(bounds, axis=0).max()), 1.0)
                 + COORDINATE_TOLERANCE * float(np.abs(bounds).max()))
    geometry = grid.geometry.name
    tree = shapely.STRtree(grid.geometry.to_numpy())
    for point in points:
        for k in tree.query(Point(point).buffer(tolerance)):
            grid.at[k, geometry] = _insert_vertex(grid.geometry.iloc[k], point, tolerance)
    return grid
