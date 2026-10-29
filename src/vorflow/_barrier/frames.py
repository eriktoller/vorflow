"""Glue between the clean ConceptualMesh frames, the barrier plan and the Gmsh mesh.

``MeshGenerator.generate()`` calls, in order:

1. ``prepare`` (before Gmsh): the barrier inputs in the local frame (domain,
   barrier lines, standard lines, feature nodes, vorflow's size field
   evaluated in Python) and the validated ``plan``; None without barriers.
2. ``gmsh_frames``: copies of the clean line and point frames for Gmsh, with
   the fine-field lines and the embeddable sites appended as new rows (the
   barrier rows themselves become field-only curves in the engine).
3. ``fit_sites`` (after Gmsh): remove the mesh nodes inside the circles or
   crowding a site (never a feature node; a node of a fixed line only inside
   a circle, and reported unless it is next to where the line meets a
   barrier), polish the free nodes within POLISH_BAND x their size of a
   circle (every other node keeps its Gmsh position exactly), and append
   the sites.
4. ``report``: ``MeshGenerator.barrier_report``; ``report_warnings`` sums
   up what the construction gave up, one warning per kind.

The size field: each sizing feature (polygon, standard line, point) adds
the sizes its fields ask for at each distance (``MeshField.python_size``),
polygons at distance 0 inside; the background lc caps it. Fields without a
Python size (custom ones) are left out of this estimate only, with a
warning.

A point feature (or a kept hex-ring seed) closer than POINT_CLEARANCE x lc
to a barrier raises BarrierMeshError: a cell centre cannot sit on a barrier
face.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.affinity
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
from shapely.validation import make_valid

from .. import _lloyd
from .._features import feature_lc, is_embedded, is_face_line, line_parts, row_bool, ring_seed_coords
from .._log import warn_user
from ..buffer import protection_epsilon
from . import pipeline as planning
from . import corridor as corridors
from . import polish, wedge
from .sizing import FEATURE_CAP, GapSizes, Obstacle, near_crossing, size_function

logger = logging.getLogger(__name__)

SIZE_SMOOTHING = 5        # passes of node-size smoothing behind the polish density (as lloyd_size_smoothing)
POINT_CLEARANCE = 2 * planning.MIN_H_FRACTION / FEATURE_CAP  # x barrier lc: closer points cannot be meshed
END_REACH = 1.0           # x barrier lc: a barrier end this close to a fixed line meets it (as a crossing)
ON_LINE_FRACTION = 1e-7   # x domain extent: a mesh node this close to a fixed line lies on it
POLISH_BAND = 3.0         # x node size: the polish moves only the free nodes this close to a barrier circle
POLISH_HALO = 2.0         # x node size beyond the band: the fixed nodes the moving nodes' cells can meet
GAP_NEIGHBOURS = 16       # circles looked at for a node's distance to the nearest one
CORRIDOR_WARN = 0.5       # x lc: a corridor with shorter cells is warned about (it adds many cells)


@dataclasses.dataclass
class Context:
    """What the plan needs from the clean frames, and what the post-Gmsh steps need again."""
    origin: np.ndarray
    domain: object              # local
    domain_global: object
    zones: list                 # embedded clean polygons (global)
    standard: list              # local standard lines
    features: np.ndarray        # local feature nodes (embedded points, hex-ring seeds)
    rings: list                 # (local xy, resolution) of kept hex rings
    size_fn: object             # local size function
    face_rows: dict             # clean-line index -> (line_id, is_barrier)
    events: list                # cleaning events from ConceptualMesh (global), with those of dropped lines
    rings_dropped: list         # (point_id, global xy) of hex rings dropped next to a barrier
    constraint_lines: object    # union of the embedded clean lines (global), or None
    obstacles: list             # local sizing.Obstacle of the fixed lines


def _local(geom, origin: np.ndarray):
    """``geom`` in the local frame."""
    return shapely.affinity.translate(geom, -origin[0], -origin[1])


def _embedded_rows(gdf) -> list:
    """(index, row) of the embedded rows of a clean frame."""
    if gdf is None or gdf.empty:
        return []
    return [(idx, row) for idx, row in gdf.iterrows() if is_embedded(row)]


def size_sources(polys, lines, points, background_lc: float, feature_fields, origin: np.ndarray) -> tuple:
    """(sources, ignored): (local geometry, is_area, ramp) of every non-barrier feature whose fields have a
    Python size, and the names of the fields without one."""
    out, ignored = [], []
    for gdf, geom_type in ((polys, 'surfaces'), (lines, 'lines'), (points, 'points')):
        if gdf is None or gdf.empty:
            continue
        for _, row in gdf.iterrows():
            if geom_type == 'lines' and is_face_line(row):
                continue
            lc = row.get('lc')
            lc = None if lc is None or pd.isna(lc) else float(lc)
            for field in feature_fields(row, geom_type, background_lc):
                if field.python_size(np.zeros(1), background_lc, lc) is None:
                    ignored.append(type(field).__name__)
                    continue
                geom = _local(row.geometry, origin)
                if geom_type == 'surfaces':
                    geom = make_valid(geom)
                out.append((geom, geom_type == 'surfaces',
                            lambda d, f=field, lc=lc: f.python_size(d, background_lc, lc)))
    return out, sorted(set(ignored))


def warn_ignored_fields(names: list) -> None:
    """Warn once that the barrier sizing leaves out the size fields ``names`` (no python_size)."""
    if names:
        warn_user(f"Barrier sizing ignores size fields without a python_size ({', '.join(names)}): Gmsh still "
                  "applies them, but the barrier faces are sized without them, so the cells along a barrier may "
                  "not follow them. Give the field a python_size (see MeshField.python_size).")


def _events(lines) -> list:
    """Cleaning events of the barrier rows, each with its line_id, and those of rows the cleaning dropped."""
    out = list(lines.attrs.get('dropped_barrier_events', []))
    if 'barrier_events' not in lines:
        return out
    for _, row in lines.iterrows():
        events = row.get('barrier_events')
        if isinstance(events, list):
            out += [dict(e, line_id=row.get('line_id')) for e in events]
    return out


def _dropped_rings(points) -> list:
    """(point_id, xy) of the hex_ring points whose ring ConceptualMesh dropped because a barrier is too close."""
    if points is None or points.empty or 'ring_near_barrier' not in points:
        return []
    return [(row.get('point_id'), (float(row.geometry.x), float(row.geometry.y)))
            for _, row in points.iterrows() if row_bool(row, 'ring_near_barrier', False)]


def check_points_off_barriers(points, items: list, face_rows: dict, origin: np.ndarray) -> None:
    """Raise BarrierMeshError for a point feature or kept hex-ring seed closer than POINT_CLEARANCE x lc to a barrier.

    ``items`` holds ``(clean-line index, local xy, lc)`` of the barrier parts.
    """
    lines = [(face_rows[idx][0], LineString(xy), lc) for idx, xy, lc in items]
    for _, row in _embedded_rows(points):
        pid = row.get('point_id')
        candidates = [(f"point {pid!r}", (row.geometry.x, row.geometry.y))]
        candidates += [(f"the hex-ring seed of point {pid!r}", seed) for seed in ring_seed_coords(row)]
        for label, xy in candidates:
            local = Point(np.asarray(xy, dtype=float) - origin)
            for line_id, line, lc in lines:
                d = line.distance(local)
                if d < POINT_CLEARANCE * lc:
                    raise planning.BarrierMeshError(
                        f"{label} at ({xy[0]:.2f}, {xy[1]:.2f}) is {d:.3g} from barrier {line_id!r}: a cell centre "
                        f"cannot sit on a barrier face. Move it off the barrier (at least {POINT_CLEARANCE * lc:.3g} "
                        "away; the faces next to it get shorter the closer it is), or, where the point marks the "
                        "barrier's end, end the barrier that far short of it.")


def _obstacle(label: str, geometry, half_width: float, barriers, ends: np.ndarray, end_lc: np.ndarray,
              tol: float) -> Obstacle | None:
    """An Obstacle crossing ``barriers`` where they intersect, next to the barrier ``ends`` within END_REACH x lc.

    The parts of ``geometry`` lying along a barrier are left out (the
    barrier faces are its faces there); None when nothing is left.
    """
    crossings = shapely.get_coordinates(shapely.intersection(geometry, barriers))
    if half_width == 0:
        geometry = geometry.difference(barriers)
    if geometry.is_empty or geometry.length <= tol:
        return None
    near_ends = np.empty((0, 2))
    if len(ends):
        pts = shapely.points(ends)
        close = shapely.distance(geometry, pts) <= END_REACH * end_lc
        near_ends = shapely.get_coordinates(shapely.get_point(shapely.shortest_line(geometry, pts[close]), 0))
    return Obstacle(label, geometry, float(half_width), crossings.reshape(-1, 2), near_ends.reshape(-1, 2))


def obstacles(polys, lines, domain, items: list, origin: np.ndarray, background_lc: float, tol: float) -> list:
    """The fixed lines (sizing.Obstacle, local) whose nodes the circles must not reach.

    Embedded standard lines, quad-buffer lines and polygons (by their strip
    half-width), and the zone edges inside the domain.
    """
    barriers = shapely.union_all([LineString(xy) for _, xy, _ in items])
    ends = np.array([xy[k] for _, xy, _ in items for k in (0, -1)]).reshape(-1, 2)
    end_lc = np.array([lc for _, _, lc in items for _ in (0, 1)])
    outer = domain.boundary.buffer(tol)
    out = []
    for _, row in _embedded_rows(polys):
        edges = _local(row.geometry.boundary, origin).difference(outer)
        if not edges.is_empty:
            out.append(_obstacle(f"edge of zone {row.get('zone_id')!r}", edges, 0.0, barriers, ends, end_lc, tol))
        if row_bool(row, 'quad_buffer', False):
            out.append(_obstacle(f"quad buffer of zone {row.get('zone_id')!r}", _local(row.geometry.boundary, origin),
                                 protection_epsilon(row, background_lc), barriers, ends, end_lc, tol))
    for _, row in _embedded_rows(lines):
        if is_face_line(row):
            continue
        half = protection_epsilon(row, background_lc) if row_bool(row, 'quad_buffer', False) else 0.0
        for part in line_parts(row.geometry):
            out.append(_obstacle(f"line {row.get('line_id')!r}", _local(part, origin), half, barriers, ends, end_lc,
                                 tol))
    return [ob for ob in out if ob is not None]


def prepare(polys, lines, points, background_lc: float, feature_fields, gradation: float,
            sharp_angle_deg: float, gap_estimate=None) -> tuple:
    """(plan, context) of the barriers in the clean frames, or (None, None) without any.

    ``gap_estimate`` (a ``_gap.GapEstimate``) is the GapField's estimate when
    ``MeshGenerator(gap_field=...)`` is set: the faces follow its sizes, and
    where it asks for more than one cell across a gap there is no corridor.
    Raises BarrierMeshError when the construction cannot be made valid.
    """
    face = [(idx, row) for idx, row in lines.iterrows() if is_face_line(row)] if lines is not None and \
        not lines.empty else []
    zones = [row.geometry for _, row in _embedded_rows(polys)]
    if not face or not zones:
        return None, None
    domain_global = make_valid(shapely.union_all(zones))
    origin = np.asarray(domain_global.bounds[:2], dtype=float)
    domain = _local(domain_global, origin)
    standard = [_local(part, origin) for _, row in _embedded_rows(lines)
                if not is_face_line(row) and not row_bool(row, 'quad_buffer', False)
                for part in line_parts(row.geometry)]
    features, rings = [], []
    for _, row in _embedded_rows(points):
        xy = np.array([row.geometry.x, row.geometry.y]) - origin
        features.append(xy[None])
        seeds = ring_seed_coords(row)
        if seeds:
            features.append(np.asarray(seeds, dtype=float) - origin)
            rings.append((xy, float(row.get('lc'))))
    features = np.vstack(features) if features else np.empty((0, 2))
    items = [(idx, np.asarray(part.coords, dtype=float)[:, :2] - origin, feature_lc(row, background_lc))
             for idx, row in face for part in line_parts(row.geometry)]
    face_rows = {idx: (row.get('line_id'), row_bool(row, 'is_barrier', False)) for idx, row in face}
    check_points_off_barriers(points, items, face_rows, origin)
    barrier_lines = planning.barrier_lines(items, standard, domain)
    sources, ignored = size_sources(polys, lines, points, background_lc, feature_fields, origin)
    warn_ignored_fields(ignored)
    gap = None if gap_estimate is None else GapSizes(gap_estimate, origin)
    size_fn = size_function(sources, background_lc, gap)
    embedded_lines = [row.geometry for _, row in _embedded_rows(lines)]
    context = Context(
        origin=origin, domain=domain, domain_global=domain_global, zones=zones, standard=standard,
        features=features, rings=rings, size_fn=size_fn, face_rows=face_rows,
        events=_events(lines),
        rings_dropped=_dropped_rings(points),
        constraint_lines=shapely.union_all(embedded_lines) if embedded_lines else None,
        obstacles=obstacles(polys, lines, domain, items, origin, background_lc, planning.touch_tolerance(domain)),
    )
    plan = planning.plan(barrier_lines, standard, domain, features, size_fn, origin, gradation,
                         math.radians(sharp_angle_deg), context.obstacles)
    return plan, context


def _append_rows(gdf, rows: list):
    """``gdf`` with ``rows`` appended under fresh integer indices above its own."""
    if not rows:
        return gdf
    start = int(gdf.index.max()) + 1 if len(gdf) else 0
    extra = gpd.GeoDataFrame(rows, index=pd.RangeIndex(start, start + len(rows)), crs=gdf.crs)
    if gdf.empty:
        columns = list(dict.fromkeys([*gdf.columns, *extra.columns]))
        return gpd.GeoDataFrame(extra.reindex(columns=columns), geometry=gdf.geometry.name, crs=gdf.crs)
    with warnings.catch_warnings():
        # pandas >= 2.1 warns that all-NA columns (``lc`` of the site rows,
        # unset optional columns of ``gdf``) will count towards the result
        # dtypes; the values are the same either way.
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat([gdf, extra])


def gmsh_frames(plan: planning.BarrierPlan, context: Context, lines, points) -> tuple:
    """(lines, points) for Gmsh: the clean frames plus fine-field lines and embeddable site points."""
    origin, layout = context.origin, plan.layout
    fine_rows = [{'geometry': shapely.affinity.translate(geom, origin[0], origin[1]),
                  'line_id': f"barrier_fine_{k}", 'lc': res, 'embed': False, 'is_barrier': False,
                  'growth_factor': planning.FINE_GROWTH}
                 for k, (geom, res) in enumerate(planning.fine_lines(layout, plan.net, context.size_fn))]
    spacing = planning.site_spacing(layout)
    edges = shapely.union_all([_local(z.boundary, origin) for z in context.zones] + context.standard)
    embed = planning.embeddable(layout.sites, spacing, edges, context.features, context.rings)
    sites = layout.sites + origin
    # The sites are no gap walls for GapField: only the barrier lines are.
    site_rows = [{'geometry': Point(sites[k]), 'point_id': f"barrier_site_{k}", 'lc': None, 'embed': True,
                  'hex_ring': False, 'ring_seeds': None, 'gap_wall': False}
                 for k in np.flatnonzero(embed)]
    return _append_rows(lines, fine_rows), _append_rows(points, site_rows)


def circle_gap(xy: np.ndarray, centres: np.ndarray, radii: np.ndarray, k: int = GAP_NEIGHBOURS) -> np.ndarray:
    """Distance from each point to the nearest of the circles (negative inside), over its k nearest centres."""
    k = min(k, len(centres))
    d, idx = cKDTree(centres).query(xy, k=k)
    d, idx = d.reshape(len(xy), k), idx.reshape(len(xy), k)
    return (d - radii[idx]).min(axis=1)


def polish_band(nodes: np.ndarray, keep: np.ndarray, is_free: np.ndarray, sizes: np.ndarray, circles: tuple,
                band: float = POLISH_BAND, halo: float = POLISH_HALO) -> tuple:
    """(moving, near): the kept free nodes within ``band`` x their size of a barrier circle, and the kept
    nodes within (band + halo) x their size (those the moving nodes' cells can meet)."""
    gap = circle_gap(nodes, *circles)
    sizes = np.asarray(sizes, dtype=float)
    near = keep & (gap < (band + halo) * sizes)
    return near & is_free & (gap < band * sizes), near


def _polish(plan, context, nodes, keep, is_free, sizes, edges, footprints, iterations) -> tuple:
    """(nodes, polish report): the kept free nodes near the circles Lloyd-polished around the fixed sites (global).

    Only the free nodes within POLISH_BAND x their size of a circle move;
    every other node is returned bit-identical. The Voronoi cells of the
    moving nodes are computed among the sites and the nodes within
    POLISH_HALO more sizes, which is all their cells can meet.
    """
    nodes = nodes.copy()
    if iterations == 0:
        return nodes, {}
    sites = plan.layout.sites + context.origin
    circles = (plan.layout.xy + context.origin, plan.layout.radius)
    is_free = np.asarray(is_free, dtype=bool)
    moving, near = polish_band(nodes, keep, is_free, sizes, circles)
    pieces = _lloyd.zone_pieces(context.zones, footprints)
    owners, moving = _lloyd.owner_polygons(nodes, moving, pieces)
    if not moving.any():
        return nodes, {"moved": 0}
    density = _lloyd.size_interpolator(nodes, _lloyd.smooth_log_sizes(sizes, is_free, edges, SIZE_SMOOTHING))
    fixed = near & ~moving
    free_xy, report = polish.lloyd(sites, nodes[moving], owners[moving], nodes[fixed], circles, density,
                                   context.domain_global, context.constraint_lines, iterations)
    free_xy, short = polish.open_short_faces(sites, free_xy, owners[moving], nodes[fixed], circles,
                                             context.constraint_lines)
    nodes[moving] = free_xy
    report.update(short_initial=short["initial"], short_final=short["final"], moved=int(moving.sum()))
    return nodes, report


def on_fixed_lines(xy: np.ndarray, obstacles: list, h: np.ndarray, tol: float) -> tuple:
    """(labels, near): the obstacle each point lies on (None: none), and whether that obstacle meets a barrier
    there (``sizing.near_crossing`` with a margin of one ``h``: the reach of a circle centred inside it)."""
    labels = np.full(len(xy), None, dtype=object)
    found = np.zeros(len(xy), dtype=bool)
    near = np.zeros(len(xy), dtype=bool)
    pts = shapely.points(xy)
    for ob in obstacles:
        on = ~found & (shapely.distance(ob.geometry, pts) <= ob.half_width + tol)
        labels[on] = ob.label
        near[on] = near_crossing(ob, xy[on], h[on], extra=1.0)
        found |= on
    return labels, near


def protected_nodes(local: np.ndarray, plan, context) -> tuple:
    """(protected, labels): nodes on a fixed line away from where it meets a barrier, and their line's label.

    Only the nodes the circles or the sites could remove are looked at.
    ``h`` there is the size the sizing started from (the barrier's lc,
    capped by vorflow's size field), as in ``sizing.obstacle_cap``.
    """
    labels = np.full(len(local), None, dtype=object)
    protected = np.zeros(len(local), dtype=bool)
    candidates = np.flatnonzero(planning.nodes_to_drop(local, plan.layout, context.features, protected)[0])
    if len(candidates) == 0 or not context.obstacles:
        return protected, labels
    layout = plan.layout
    mids = 0.5 * (layout.xy[layout.segments[:, 0]] + layout.xy[layout.segments[:, 1]])
    _, nearest = cKDTree(mids).query(local[candidates])
    h0 = np.minimum(plan.net.piece_lc[layout.seg_piece[nearest]], context.size_fn(local[candidates]))
    tol = ON_LINE_FRACTION * max(np.ptp(np.asarray(context.domain.bounds).reshape(2, 2), axis=0))
    found, near = on_fixed_lines(local[candidates], context.obstacles, h0, tol)
    labels[candidates] = found
    protected[candidates] = np.not_equal(found, None) & ~near
    return protected, labels


def _dropped_fixed(nodes: np.ndarray, dropped: np.ndarray, labels: np.ndarray) -> list:
    """Per fixed line, the number of its nodes in ``dropped`` and where the first one was (global)."""
    out = {}
    for k in np.flatnonzero(dropped):
        entry = out.setdefault(labels[k], {'feature': labels[k], 'nodes': 0,
                                           'location': (float(nodes[k, 0]), float(nodes[k, 1]))})
        entry['nodes'] += 1
    return list(out.values())


def fit_sites(plan, context, nodes, tags, is_free, sizes, edges, footprints, iterations) -> dict:
    """Mesh arrays with the barrier sites in: nodes, tags, is_free, sizes, edges, is_site, plus counts."""
    nodes = np.asarray(nodes, dtype=float)
    tags = np.asarray(tags).astype(np.int64)
    local = nodes - context.origin
    protected, labels = protected_nodes(local, plan, context)
    remove, copies = planning.nodes_to_drop(local, plan.layout, context.features, protected)
    keep = ~remove
    fixed = _dropped_fixed(nodes, remove & protected & ~copies, labels)
    nodes, polish_report = _polish(plan, context, nodes, keep, is_free, sizes, edges, footprints, iterations)
    sites = plan.layout.sites + context.origin
    first_tag = int(tags.max()) + 1 if len(tags) else 1
    position = np.full(len(nodes), -1, dtype=np.int64)
    position[keep] = np.arange(int(keep.sum()))
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = position[edges[keep[edges].all(axis=1)]] if len(edges) else edges
    return {
        "nodes": np.vstack([nodes[keep], sites]),
        "tags": np.concatenate([tags[keep], np.arange(first_tag, first_tag + len(sites), dtype=np.int64)]),
        "is_free": np.concatenate([np.asarray(is_free, dtype=bool)[keep], np.zeros(len(sites), dtype=bool)]),
        "sizes": np.concatenate([np.asarray(sizes, dtype=float)[keep], planning.site_spacing(plan.layout)]),
        "edges": edges,
        "is_site": np.concatenate([np.zeros(int(keep.sum()), dtype=bool), np.ones(len(sites), dtype=bool)]),
        "removed": int((remove & ~copies).sum()),
        "fixed": fixed,
        "polish": polish_report,
    }


def _nearest_line(xy: np.ndarray, plan) -> object:
    """Feature id of the barrier line nearest to ``xy`` (local)."""
    point = Point(xy)
    best = min(plan.net.lines, key=lambda b: LineString(b.xy).distance(point) if len(b.xy) > 1 else math.inf)
    return best.line_id


def barrier_lines_frame(plan, context, crs) -> gpd.GeoDataFrame:
    """The lines the faces follow (global), with feature_id, line_id and is_barrier."""
    rows = []
    for b in plan.net.lines:
        line_id, is_barrier = context.face_rows[b.line_id]
        rows.append({'feature_id': b.line_id, 'line_id': line_id, 'is_barrier': is_barrier,
                     'geometry': LineString(b.xy + context.origin)})
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)


def _corridor_entry(c, lay, plan, context) -> dict:
    """A corridor for ``barrier_report``: kind, the barriers on its sides, where it is, how long, its narrowest
    gap, how it ends, its cells (one row) and the resolution of its barriers."""
    mid = c.origin + 0.5 * (c.x0 + c.x1) * c.m + context.origin
    line_ids = []
    for side in c.sides:
        piece = side[0].piece
        xy = 0.5 * (plan.net.hard_xy[plan.net.pieces[piece][0]] + plan.net.hard_xy[plan.net.pieces[piece][1]])
        line_ids.append(context.face_rows[_nearest_line(xy, plan)][0])
    cells = len({tuple(np.round(site, 9)) for _, _, sites in lay.runs for site in sites})
    lc = min(float(plan.net.piece_lc[side[0].piece]) for side in c.sides)
    return {'kind': 'boundary' if len(c.sides) == 1 else 'barriers', 'line_ids': line_ids,
            'location': (float(mid[0]), float(mid[1])), 'length': float(c.x1 - c.x0), 'gap': float(c.gap),
            'ends': tuple(c.ends), 'cells': cells, 'resolution': lc}


def _collapse_entry(e: dict, context) -> dict:
    """A merged acute wedge for ``barrier_report`` (see ``wedge``)."""
    xy = e['location'] + context.origin
    return {'kind': e['kind'], 'line_ids': [context.face_rows[k][0] for k in e['lines']],
            'location': (float(xy[0]), float(xy[1])), 'angle': float(e['angle']), 'reach': float(e['reach']),
            'deviation': float(e['deviation'])}


def _clean_event(e: dict) -> dict:
    """A cleaning event for ``barrier_report``."""
    return {'line_id': e.get('line_id'), 'kind': e['kind'], 'distance': float(e['distance']),
            'location': (float(e['location'][0]), float(e['location'][1])), 'note': e['note']}


def _cleaning_entries(events: list) -> dict:
    """The ``barrier_report`` entries that come from the cleaning in ConceptualMesh."""
    return {
        'stubs': [{'line_id': e.get('line_id'), 'length': float(e['distance']),
                   'location': (float(e['location'][0]), float(e['location'][1]))}
                  for e in events if e['kind'] == 'stub'],
        'snaps': [_clean_event(e) for e in events if e['kind'] in ('snap', 'trim', 'simplify')],
        'noise': [_clean_event(e) for e in events if e['kind'] == 'noise'],
        'on_boundary': [_clean_event(e) for e in events if e['kind'] == 'boundary'],
    }


def report(plan, context, fitted: dict) -> dict:
    """``MeshGenerator.barrier_report`` (see its docstring)."""
    origin = context.origin
    cleaning = _cleaning_entries(context.events)
    tips, stubs = [], cleaning['stubs']
    for kind, length, xy, line in plan.net.uncovered:
        entry = {'line_id': context.face_rows[line][0], 'length': float(length),
                 'location': (float(xy[0] + origin[0]), float(xy[1] + origin[1]))}
        (tips if kind == "spike tip" else stubs).append(entry)
    return {
        'corridors': [_corridor_entry(c, lay, plan, context) for c, lay in zip(plan.net.corridors,
                                                                                  plan.field.corridors)],
        'collapsed': [_collapse_entry(e, context) for e in plan.net.collapsed],
        'tips': tips,
        'stubs': stubs,
        'snaps': cleaning['snaps'],
        'noise': cleaning['noise'],
        'on_boundary': cleaning['on_boundary'],
        'rings': [{'point_id': pid, 'location': xy} for pid, xy in context.rings_dropped],
        'fixed_nodes': fitted['fixed'],
        'repair_rounds': int(plan.rounds),
        'sites': int(len(plan.layout.sites)),
        'removed_nodes': fitted['removed'],
        'polish': fitted['polish'],
    }


def dropped_report(lines) -> dict | None:
    """``barrier_report`` when the cleaning dropped every barrier (None if there were none)."""
    events = list(lines.attrs.get('dropped_barrier_events', [])) if lines is not None else []
    if not events:
        return None
    return {'corridors': [], 'collapsed': [], 'tips': [], **_cleaning_entries(events), 'rings': [], 'fixed_nodes': [],
            'repair_rounds': 0, 'sites': 0, 'removed_nodes': 0, 'polish': {}}


def report_warnings(report: dict, sharp_angle: float) -> list:
    """One message per kind of thing the barrier construction gave up or moved (the details are in ``report``).

    Noise and boundary parts are warned about one by one when ConceptualMesh cleans the lines.
    """
    out = []
    dense = [e for e in report['corridors'] if e['length'] / e['cells'] < CORRIDOR_WARN * e['resolution']]
    if dense:
        cells = sum(e['cells'] for e in dense)
        usual = sum(e['length'] / e['resolution'] for e in dense)
        out.append(f"{len(dense)} narrow gap(s) between close barriers, or a barrier and the boundary, keep one "
                   f"row of cells, shorter than {CORRIDOR_WARN:g} x resolution there (at most "
                   f"{corridors.CORRIDOR_ASPECT:g} x the gap long, so flow along the gap is kept): {cells} cells "
                   f"where the resolution gives about {usual:.0f}, and the mesh around them is graded down to "
                   "match. Move the barriers further apart if the model allows (a barrier part within "
                   "simplify_tolerance of the boundary is dropped); see MeshGenerator.barrier_report['corridors'].")
    if report['collapsed']:
        moved = max(e['deviation'] for e in report['collapsed'])
        out.append(f"{len(report['collapsed'])} narrow barrier junction(s) or boundary exit(s) (legs closer than "
                   f"sharp_angle={sharp_angle:g} degrees, or fans of legs closer than "
                   f"{math.degrees(wedge.FAN_ANGLE):g} degrees) were merged over a short reach, so the barriers "
                   f"there share one line of faces for a short stretch (moved at most {moved:.3g}, within "
                   f"{wedge.DEVIATION:g} x resolution); see MeshGenerator.barrier_report['collapsed'].")
    if report['tips']:
        out.append(f"{len(report['tips'])} barrier spike tip(s) sharper than sharp_angle={sharp_angle:g} degrees "
                   f"are left without cell faces ({sum(t['length'] for t in report['tips']):.3g} of barrier "
                   "length); see MeshGenerator.barrier_report['tips'].")
    if report['stubs']:
        out.append(f"{len(report['stubs'])} short dangling barrier stub(s) were dropped "
                   f"({sum(t['length'] for t in report['stubs']):.3g} of barrier length); see "
                   "MeshGenerator.barrier_report['stubs'].")
    moved = [e for e in report['snaps'] if e['kind'] in ('snap', 'trim')]
    if moved:
        out.append(f"{len(moved)} barrier end(s) were snapped or trimmed onto another barrier or the domain "
                   "boundary; see MeshGenerator.barrier_report['snaps'].")
    if report['fixed_nodes']:
        names = ", ".join(e['feature'] for e in report['fixed_nodes'])
        out.append(f"{sum(e['nodes'] for e in report['fixed_nodes'])} mesh node(s) of {names} lie inside the "
                   "barrier circles and were removed, so the cells there do not follow that feature; see "
                   "MeshGenerator.barrier_report['fixed_nodes'].")
    return out
