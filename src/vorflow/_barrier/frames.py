"""Glue between the clean ConceptualMesh frames, the barrier plan and the Gmsh mesh.

``MeshGenerator.generate()`` calls, in order:

1. ``prepare`` (before Gmsh): the barrier inputs in the local frame (domain,
   barrier lines, standard lines, feature nodes, vorflow's size field
   evaluated in Python) and the validated ``plan``; None without barriers.
2. ``gmsh_frames``: copies of the clean line and point frames for Gmsh, with
   the fine-field lines and the embeddable sites appended as new rows (the
   barrier rows themselves become field-only curves in the engine).
3. ``fit_sites`` (after Gmsh): remove the mesh nodes inside the circles or
   crowding a site (never a feature node), polish the free nodes, and append
   the sites.
4. ``report``: ``MeshGenerator.barrier_report``.

The size field: each sizing feature (polygon, standard line, point) adds
the sizes its fields ask for at each distance (``MeshField.python_size``),
polygons at distance 0 inside; the background lc caps it. Fields without a
Python size (custom ones) are left out of this estimate only.
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
from shapely.validation import make_valid

from .. import _lloyd
from .._features import feature_lc, is_embedded, is_face_line, line_parts, row_bool, ring_seed_coords
from . import pipeline as planning
from . import polish
from .sizing import size_function

logger = logging.getLogger(__name__)

SIZE_SMOOTHING = 5        # passes of node-size smoothing behind the polish density (as lloyd_size_smoothing)
RING_CLEARANCE = 2.0      # x resolution: a hex ring this close to a barrier is reported as dropped


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
    events: list                # cleaning events from ConceptualMesh (global)
    rings_dropped: list         # (point_id, global xy) of hex rings dropped next to a barrier
    constraint_lines: object    # union of the embedded clean lines (global), or None


def _local(geom, origin: np.ndarray):
    """``geom`` in the local frame."""
    return shapely.affinity.translate(geom, -origin[0], -origin[1])


def _embedded_rows(gdf) -> list:
    """(index, row) of the embedded rows of a clean frame."""
    if gdf is None or gdf.empty:
        return []
    return [(idx, row) for idx, row in gdf.iterrows() if is_embedded(row)]


def size_sources(polys, lines, points, background_lc: float, feature_fields, origin: np.ndarray) -> list:
    """(local geometry, is_area, ramp) of every non-barrier feature whose fields have a Python size."""
    out = []
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
                    logger.debug(f"Barrier sizing ignores a {type(field).__name__} (no python_size).")
                    continue
                geom = _local(row.geometry, origin)
                if geom_type == 'surfaces':
                    geom = make_valid(geom)
                out.append((geom, geom_type == 'surfaces',
                            lambda d, f=field, lc=lc: f.python_size(d, background_lc, lc)))
    return out


def _events(lines) -> list:
    """Cleaning events of the barrier rows, each with its line_id."""
    out = []
    if 'barrier_events' not in lines:
        return out
    for _, row in lines.iterrows():
        events = row.get('barrier_events')
        if isinstance(events, list):
            out += [dict(e, line_id=row.get('line_id')) for e in events]
    return out


def _dropped_rings(points, barriers_global) -> list:
    """(point_id, xy) of hex_ring points without seeds that a barrier passes within RING_CLEARANCE x lc of."""
    out = []
    if points is None or points.empty or 'hex_ring' not in points or not barriers_global:
        return out
    union = shapely.union_all(barriers_global)
    for _, row in points.iterrows():
        if not row_bool(row, 'hex_ring', False) or ring_seed_coords(row):
            continue
        lc = float(row.get('lc'))
        if union.distance(row.geometry) < RING_CLEARANCE * lc:
            out.append((row.get('point_id'), (float(row.geometry.x), float(row.geometry.y))))
    return out


def prepare(polys, lines, points, background_lc: float, feature_fields, gradation: float,
            sharp_angle_deg: float) -> tuple:
    """(plan, context) of the barriers in the clean frames, or (None, None) without any.

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
    barrier_lines = planning.barrier_lines(items, standard, domain)
    size_fn = size_function(size_sources(polys, lines, points, background_lc, feature_fields, origin),
                            background_lc)
    embedded_lines = [row.geometry for _, row in _embedded_rows(lines)]
    context = Context(
        origin=origin, domain=domain, domain_global=domain_global, zones=zones, standard=standard,
        features=features, rings=rings, size_fn=size_fn,
        face_rows={idx: (row.get('line_id'), row_bool(row, 'is_barrier', False)) for idx, row in face},
        events=_events(lines),
        rings_dropped=_dropped_rings(points, [row.geometry for _, row in face]),
        constraint_lines=shapely.union_all(embedded_lines) if embedded_lines else None,
    )
    plan = planning.plan(barrier_lines, standard, domain, features, size_fn, origin, gradation,
                         math.radians(sharp_angle_deg))
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
    site_rows = [{'geometry': Point(sites[k]), 'point_id': f"barrier_site_{k}", 'lc': None, 'embed': True,
                  'hex_ring': False, 'ring_seeds': None}
                 for k in np.flatnonzero(embed)]
    return _append_rows(lines, fine_rows), _append_rows(points, site_rows)


def _polish(plan, context, nodes, keep, is_free, sizes, edges, footprints, iterations) -> tuple:
    """(nodes, polish report): the kept free nodes Lloyd-polished around the fixed sites (global)."""
    nodes = nodes.copy()
    if iterations == 0:
        return nodes, {}
    sites = plan.layout.sites + context.origin
    pieces = _lloyd.zone_pieces(context.zones, footprints)
    owners, free = _lloyd.owner_polygons(nodes, is_free & keep, pieces)
    if not free.any():
        return nodes, {}
    density = _lloyd.size_interpolator(nodes, _lloyd.smooth_log_sizes(sizes, is_free, edges, SIZE_SMOOTHING))
    circles = (plan.layout.xy + context.origin, plan.layout.radius)
    fixed = keep & ~free
    free_xy, report = polish.lloyd(sites, nodes[free], owners[free], nodes[fixed], circles, density,
                                   context.domain_global, context.constraint_lines, iterations)
    free_xy, short = polish.open_short_faces(sites, free_xy, owners[free], nodes[fixed], circles,
                                             context.constraint_lines)
    nodes[free] = free_xy
    report.update(short_initial=short["initial"], short_final=short["final"])
    return nodes, report


def fit_sites(plan, context, nodes, tags, is_free, sizes, edges, footprints, iterations) -> dict:
    """Mesh arrays with the barrier sites in: nodes, tags, is_free, sizes, edges, is_site, plus counts."""
    nodes = np.asarray(nodes, dtype=float)
    tags = np.asarray(tags).astype(np.int64)
    remove = planning.nodes_to_drop(nodes - context.origin, plan.layout, context.features)
    keep = ~remove
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
        "removed": int(remove.sum()),
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


def _corridor_entry(c, plan, context) -> dict:
    """A corridor for ``barrier_report``: kind, the barriers on its sides, where it is and how long."""
    mid = c.origin + 0.5 * (c.x0 + c.x1) * c.m + context.origin
    line_ids = []
    for side in c.sides:
        piece = side[0].piece
        xy = 0.5 * (plan.net.hard_xy[plan.net.pieces[piece][0]] + plan.net.hard_xy[plan.net.pieces[piece][1]])
        line_ids.append(context.face_rows[_nearest_line(xy, plan)][0])
    return {'kind': 'boundary' if len(c.sides) == 1 else 'barriers', 'line_ids': line_ids,
            'location': (float(mid[0]), float(mid[1])), 'length': float(c.x1 - c.x0)}


def report(plan, context, fitted: dict) -> dict:
    """``MeshGenerator.barrier_report`` (see its docstring)."""
    origin = context.origin
    tips, stubs = [], []
    for kind, length, xy, line in plan.net.uncovered:
        entry = {'line_id': context.face_rows[line][0], 'length': float(length),
                 'location': (float(xy[0] + origin[0]), float(xy[1] + origin[1]))}
        (tips if kind == "spike tip" else stubs).append(entry)
    def clean(e):
        return {'line_id': e.get('line_id'), 'kind': e['kind'], 'distance': float(e['distance']),
                'location': (float(e['location'][0]), float(e['location'][1])), 'note': e['note']}
    return {
        'corridors': [_corridor_entry(c, plan, context) for c in plan.net.corridors],
        'tips': tips,
        'stubs': stubs,
        'snaps': [clean(e) for e in context.events if e['kind'] in ('snap', 'trim', 'simplify')],
        'noise': [clean(e) for e in context.events if e['kind'] == 'noise'],
        'rings': [{'point_id': pid, 'location': xy} for pid, xy in context.rings_dropped],
        'repair_rounds': int(plan.rounds),
        'sites': int(len(plan.layout.sites)),
        'removed_nodes': fitted['removed'],
        'polish': fitted['polish'],
    }
