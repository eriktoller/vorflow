"""Line-line junctions: snap near-miss ends, trim short overshoots, pin junction vertices.

Each embedded line is densified from its own vertices, and Gmsh ``fragment``
then inserts a node wherever two lines meet. Without help that node lands an
arbitrary 0..spacing from the nearest densified vertex, so T-junctions and
crossings get two generators a few metres apart and a tiny Voronoi cell
(issue #31). ``ConceptualMesh`` fixes this before Gmsh, in four steps, on the
lines that take part (embedded, not straddle/barrier, not quad-buffer):

1. **Snap ends.** A line end within ``tolerance`` of another line but not on
   it moves onto that line: onto the other line's nearest end when that is
   also within ``tolerance``, else onto its nearest point, so near misses
   become T-junctions. An end whose line already meets the other line within
   ``2 * tolerance`` of it is an overshoot and is not moved: pulling it back
   would fold the line across the other one. Nor is an end whose gap to the
   other line points more than ``SNAP_MAX_GAP_ANGLE_DEG`` away from the
   line's own direction at that end: that end runs alongside the other line
   rather than towards it, as for two parallel lines closer than
   ``tolerance``, and snapping would lay one line onto the other (guard
   after @d-strom's #35).
2. **Find junctions.** Every point where two lines cross or touch. A junction
   shared by several lines gets one coordinate, so all of them carry the same
   vertex and ``fragment`` makes a single node there.
3. **Trim stubs.** A piece of line shorter than ``tolerance`` beyond its last
   junction (a short overshoot) is cut off at the junction.
4. **Pin and thin.** Each junction becomes a vertex of every line through it,
   and the line's other vertices closer than ``tolerance`` to a junction are
   dropped. Densification then starts at the junction, so the nearest line
   vertex sits at least ``tolerance`` from it.

``tolerance`` is ``fraction`` times the line's vertex spacing (its numeric
``densify``, else its ``resolution``). Line ends and vertices on a polygon
boundary (``anchors``) are never moved, trimmed or dropped, so the line keeps
its connection to the zone it was snapped to.
"""
from __future__ import annotations

import math

from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import substring
from shapely.strtree import STRtree

# Distances at or below max(connectivity_tolerance, TOUCH_FLOOR) count as
# touching, so float noise from projecting onto a line is not a near miss.
TOUCH_FLOOR = 1e-6
# An end snaps only if the gap to the other line lies within this angle of the
# line's outward direction at that end. 80 degrees joins lines that meet at
# 10 degrees or more and leaves parallel or nearly parallel lines apart.
SNAP_MAX_GAP_ANGLE_DEG = 80.0


def snap_line_ends(lines: list, tolerances: list, eps: float, anchors=None) -> tuple[list, list]:
    """Move each line end that misses another line by at most its tolerance onto that line.

    Returns (lines, moves); each move is ``(line position, end, from_xy, to_xy)``
    with ``end`` 0 (start) or -1 (end).
    """
    lines = list(lines)
    tree = STRtree(lines)
    # Lines already processed may have moved by up to their own tolerance.
    slack = max(tolerances, default=0.0)
    moves = []
    for i, line in enumerate(lines):
        coords = list(line.coords)
        reach = tolerances[i] + eps
        for end in (0, -1):
            point = Point(coords[end])
            if anchors is not None and anchors.distance(point) <= eps:
                continue
            others = [int(j) for j in tree.query(point, predicate="dwithin", distance=reach + slack) if j != i]
            gaps = [(lines[j].distance(point), j) for j in others]
            if not gaps or min(gaps)[0] <= eps:
                continue
            gap, j = min(gaps)
            if gap > tolerances[i] or _meets_near_end(line, lines[j], end, 2 * tolerances[i]):
                continue
            target = _snap_target(lines[j], point, tolerances[i])
            if not _gap_is_ahead(coords, end, target):
                continue
            candidate = list(coords)
            candidate[end] = target
            if LineString(candidate).length <= eps or candidate[0] == candidate[-1]:
                continue
            moves.append((i, end, tuple(coords[end]), tuple(target)))
            coords = candidate
        lines[i] = LineString(coords)
    return lines, moves


def _gap_is_ahead(coords: list, end: int, target: tuple,
                  max_angle_deg: float = SNAP_MAX_GAP_ANGLE_DEG) -> bool:
    """True if ``target`` lies within ``max_angle_deg`` of the line's outward direction at ``end``."""
    ordered = coords if end == 0 else coords[::-1]
    tip = ordered[0]
    inner = next((xy for xy in ordered[1:] if math.dist(xy, tip) > 0), None)
    gap = (target[0] - tip[0], target[1] - tip[1])
    if inner is None or math.hypot(*gap) == 0:
        return False
    outward = (tip[0] - inner[0], tip[1] - inner[1])
    cos_angle = (outward[0] * gap[0] + outward[1] * gap[1]) / (math.hypot(*outward) * math.hypot(*gap))
    return cos_angle >= math.cos(math.radians(max_angle_deg))


def _meets_near_end(line: LineString, other: LineString, end: int, reach: float) -> bool:
    """True if ``line`` touches ``other`` within ``reach`` (along the line) of its ``end`` (0 or -1)."""
    hits = line.intersection(other)
    if hits.is_empty:
        return False
    parts = getattr(hits, "geoms", [hits])
    for part in parts:
        for xy in part.coords:
            s = line.project(Point(xy))
            if (s if end == 0 else line.length - s) <= reach:
                return True
    return False


def _snap_target(line: LineString, point: Point, tolerance: float) -> tuple:
    """Where ``point`` snaps onto ``line``: its nearest end if within ``tolerance``, else its nearest point."""
    ends = [Point(line.coords[0]), Point(line.coords[-1])]
    nearest_end = min(ends, key=point.distance)
    if nearest_end.distance(point) <= tolerance:
        return (nearest_end.x, nearest_end.y)
    target = line.interpolate(line.project(point))
    return (target.x, target.y)


def find_junctions(lines: list, eps: float) -> list[list[tuple]]:
    """Junction coordinates on each line; a junction shared by several lines has one coordinate."""
    tree = STRtree(lines)
    canonical = []
    junctions = [[] for _ in lines]
    for i, line in enumerate(lines):
        for j in tree.query(line, predicate="dwithin", distance=eps):
            j = int(j)
            if j <= i:
                continue
            for point in _pair_junctions(line, lines[j], eps):
                point = _canonical_point(canonical, point, eps)
                _add_unique(junctions[i], point, eps)
                _add_unique(junctions[j], point, eps)
    return junctions


def _pair_junctions(line_a: LineString, line_b: LineString, eps: float) -> list[tuple]:
    """Points where two lines touch: ends on the other line first, then proper crossings."""
    points = []
    for line, other in ((line_a, line_b), (line_b, line_a)):
        for xy in (line.coords[0], line.coords[-1]):
            if other.distance(Point(xy)) <= eps:
                points.append(tuple(xy))
    # Collinear overlaps come back as line parts; only their ends (found above) count.
    hits = line_a.intersection(line_b)
    for part in getattr(hits, "geoms", [hits]):
        if isinstance(part, MultiPoint):
            points.extend((p.x, p.y) for p in part.geoms)
        elif isinstance(part, Point) and not part.is_empty:
            points.append((part.x, part.y))
    return points


def _canonical_point(canonical: list, point: tuple, eps: float) -> tuple:
    """The already-seen junction within ``eps`` of ``point``, else ``point`` (recorded)."""
    for known in canonical:
        if math.dist(known, point) <= eps:
            return known
    canonical.append(point)
    return point


def _add_unique(points: list, point: tuple, eps: float) -> None:
    """Append ``point`` unless one within ``eps`` is already there."""
    if all(math.dist(known, point) > eps for known in points):
        points.append(point)


def trim_short_stubs(lines: list, junctions: list, tolerances: list, eps: float,
                     anchors=None) -> tuple[list, list]:
    """Cut off line ends shorter than the line's tolerance beyond a junction.

    Returns (lines, trims); each trim is ``(line position, end, cut length)``
    with ``end`` 0 (start) or -1 (end).
    """
    lines = list(lines)
    trims = []
    for i, line in enumerate(lines):
        length = line.length
        tolerance = tolerances[i]
        if length <= 2 * tolerance:
            continue
        cut_start, cut_end = None, None
        for xy in junctions[i]:
            s = line.project(Point(xy))
            if eps < s < tolerance and (cut_start is None or s > cut_start[0]):
                cut_start = (s, xy)
            if eps < length - s < tolerance and (cut_end is None or s < cut_end[0]):
                cut_end = (s, xy)
        if cut_start is not None and _is_anchored(line.coords[0], anchors, eps):
            cut_start = None
        if cut_end is not None and _is_anchored(line.coords[-1], anchors, eps):
            cut_end = None
        if cut_start is None and cut_end is None:
            continue
        start = cut_start[0] if cut_start else 0.0
        end = cut_end[0] if cut_end else length
        coords = list(substring(line, start, end).coords)
        # Use the junction's own coordinate so the other line's vertex matches exactly.
        if cut_start:
            coords[0] = cut_start[1]
        if cut_end:
            coords[-1] = cut_end[1]
        lines[i] = LineString(coords)
        if cut_start:
            trims.append((i, 0, start))
        if cut_end:
            trims.append((i, -1, length - end))
    return lines, trims


def _is_anchored(xy, anchors, eps: float) -> bool:
    """True if ``xy`` lies on the anchor geometry (polygon boundaries)."""
    return anchors is not None and anchors.distance(Point(xy)) <= eps


def pin_junctions(line: LineString, junctions: list, radius: float, eps: float,
                  anchors=None) -> LineString:
    """``line`` with each junction as a vertex and other interior vertices within ``radius`` of one dropped."""
    if not junctions:
        return line
    coords = _insert_vertices(list(line.coords), junctions, eps)
    pinned = {tuple(xy) for xy in junctions}
    kept = []
    for k, xy in enumerate(coords):
        keep = (
            k in (0, len(coords) - 1)
            or xy in pinned
            or _is_anchored(xy, anchors, eps)
            or all(math.dist(xy, junction) >= radius for junction in junctions)
        )
        if keep:
            kept.append(xy)
    return LineString(kept)


def _insert_vertices(coords: list, points: list, eps: float) -> list:
    """``coords`` with each point inserted on its nearest segment, or replacing a vertex within ``eps``."""
    coords = [tuple(xy) for xy in coords]
    for point in points:
        nearest_vertex = min(range(len(coords)), key=lambda k: math.dist(coords[k], point))
        if math.dist(coords[nearest_vertex], point) <= eps:
            coords[nearest_vertex] = tuple(point)
            continue
        segments = [LineString(coords[k:k + 2]) for k in range(len(coords) - 1)]
        k = min(range(len(segments)), key=lambda k: segments[k].distance(Point(point)))
        coords.insert(k + 1, tuple(point))
    return coords
