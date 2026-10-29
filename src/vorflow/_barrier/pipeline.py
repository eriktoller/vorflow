"""Barrier site planning with a validity gate, and the helpers that hand the plan to Gmsh.

``plan`` runs before Gmsh:

1. ``network.build_network``: drop stubs, truncate sharp spike tips, hard
   nodes and pieces, the sector plan of every node (rungs, shared, split);
   ``corridor.find_corridors``: narrow gaps that get one shared row of
   sites (their chord nodes are re-planned to take their radii from it).
2. ``sizing.capped_field``: target face length along the network from
   vorflow's size field, capped by the distance to other barrier parts, the
   boundary, feature nodes and fixed lines (``sizing.Obstacle``); a face
   length below MIN_H_FRACTION x lc fails here, before anything divides by
   it. ``sizing.grade_field``: Lipschitz; ladders; junction sizes;
   corridors.
3. ``sites.construct``: 1D nodes, radii, closed-form sites.
4. ``validate.defects``. A corridor that cannot be laid out, or has a
   defect in its region, is dropped and the plan starts again without it.
   Elsewhere, where the gate fails, the face length is set to REPAIR_FACTOR
   x the local face length there and steps 2-4 repeat, at most
   REPAIR_ROUNDS times. If a defect is left, the site count exceeds
   SITE_GROWTH_CAP x the first round's, or the face length would drop below
   MIN_H_FRACTION x lc, ``BarrierMeshError`` names the place, what set the
   face length there (the cap, or a junction; ``sizing.CAUSES``) and what
   helps for that cause (``advice``).

Gmsh gets a fine field along the barrier faces that are finer than the
size field (FINE_RATIO), at a size between the two (FINE_BLEND), or at the
face length along a corridor (CORRIDOR_BLEND), grouped in size classes.

After Gmsh (see ``MeshGenerator``): Gmsh nodes inside a circle (by
CIRCLE_MARGIN) or within CLEAR_FRACTION x the local spacing of a site are
removed, then the sites are added. Feature nodes (wells, hex-ring seeds,
embedded points) are never removed: the size caps keep them outside every
circle. Nodes of fixed lines away from where they meet a barrier are
removed only inside a circle (the caller reports those).

Everything here is in the local frame (``origin`` subtracted).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString

from . import corridor as corridors
from .._features import HEX_RING_CLEARANCE
from .lines import BarrierLine, must_keep
from .network import SHARP_ANGLE, Network, build_network, replan_chord_nodes
from .sites import Layout, construct
from .sizing import BARRIER_CAP, CAUSES, GRADATION, SizeField, capped_field, grade_field
from .validate import all_finite, defects

REPAIR_ROUNDS = 8
REPAIR_FACTOR = 0.5
MIN_H_FRACTION = 0.02    # x lc: smallest face length the construction makes before it gives up
SITE_GROWTH_CAP = 4.0    # x the first round's site count
TOL_FRACTION = 1e-9      # x domain extent: distance that counts as touching
FINE_RATIO = 0.87        # faces finer than this x the size field get a fine field
CLASS_RATIO = 1.25       # fine fields are grouped in size classes of this ratio
FINE_GROWTH = 1.3        # growth_factor of the fine fields
FINE_BLEND = 0.5         # fine field size = face^(1 - FINE_BLEND) x target^FINE_BLEND
CORRIDOR_BLEND = 0.0     # ... along a corridor, whose outer cells should match its row
EMBED_GAP = 0.05         # x spacing: sites this close to an edge, line or feature are not Gmsh points
CIRCLE_MARGIN = 0.02     # Gmsh nodes within R (1 + CIRCLE_MARGIN) of a circle centre are removed
FINER_MARGIN = 0.9       # an advised finer resolution is this x the one at which the faces just fit
FOLD_MARGIN = 1.25       # an advised simplify_tolerance is this x the shortest piece of the fold
CLEAR_FRACTION = 0.5     # ... and those within this x the local spacing of a site


GEOMETRY_ERRORS = (shapely.errors.GEOSException, ArithmeticError)  # degenerate geometry inside the construction


class BarrierMeshError(ValueError):
    """The barrier faces cannot be made exact at this resolution and tolerance."""


def internal_failure(step: str, err: Exception) -> BarrierMeshError:
    """A BarrierMeshError for a geometric failure inside ``step`` that the construction does not foresee."""
    return BarrierMeshError(
        f"the barrier {step} failed on a degenerate geometry ({type(err).__name__}: {err}). This is a bug in "
        "vorflow; please report it with the barrier lines. Until then, quad_buffer=True on the barrier meshes it "
        "with a structured strip of cells instead of exact faces.")


@dataclasses.dataclass
class BarrierPlan:
    """A validated barrier construction (local coordinates) and how it was reached."""
    origin: np.ndarray
    net: Network
    field: SizeField
    layout: Layout
    rounds: int


def where(points: np.ndarray, origin: np.ndarray, limit: int = 3) -> str:
    """Global coordinates of up to ``limit`` points, as text."""
    pts = np.unique(np.round(np.asarray(points, dtype=float).reshape(-1, 2) + origin, 2), axis=0)[:limit]
    return ", ".join(f"({x:.2f}, {y:.2f})" for x, y in pts)


QUAD_BUFFER_ADVICE = ("quad_buffer=True on the barrier meshes it with a structured strip of cells instead (its "
                      "faces are then not exact)")


def round_down(x: float, digits: int = 2) -> float:
    """``x`` rounded down to ``digits`` significant digits."""
    scale = 10.0 ** (math.floor(math.log10(x)) - digits + 1)
    return math.floor(x / scale + 1e-9) * scale


def round_up(x: float, digits: int = 2) -> float:
    """``x`` rounded up to ``digits`` significant digits."""
    scale = 10.0 ** (math.floor(math.log10(x)) - digits + 1)
    return math.ceil(x / scale - 1e-9) * scale


def fold_tolerance(net: Network, points: np.ndarray) -> float | None:
    """simplify_tolerance that removes a short fold near ``points``: FOLD_MARGIN x the shortest piece within
    lc of them, if one is shorter than lc (else None: the legs are long, a tolerance does not help)."""
    pts = shapely.points(np.asarray(points, dtype=float).reshape(-1, 2))
    best = None
    for (a, b), lc in zip(net.pieces, net.piece_lc):
        seg = LineString(net.hard_xy[[a, b]])
        length = float(seg.length)
        if length < lc and float(shapely.distance(seg, pts).min()) <= lc:
            best = length if best is None else min(best, length)
    return None if best is None else round_up(FOLD_MARGIN * best)


def advice(cause: str, h: float, fold: float | None = None, lc: float | None = None) -> str:
    """What makes the faces there possible, for a face length ``h`` set by ``cause`` (``sizing.CAUSES``);
    ``fold`` is the simplify_tolerance that removes a short fold there (``fold_tolerance``), ``lc`` the
    barrier resolution there."""
    finer = (f"a barrier resolution of {round_down(FINER_MARGIN * h / MIN_H_FRACTION):.3g} or finer there"
             if h > 0 else "a finer barrier resolution there")
    remedy = (f"If that is digitising noise, simplify_tolerance of at least {fold:.3g} on the barrier removes it; "
              f"otherwise {QUAD_BUFFER_ADVICE}") if fold is not None else \
        f"The legs there are long, so simplify_tolerance does not help; {QUAD_BUFFER_ADVICE}"
    return {
        "barrier": "a barrier end comes that close to another barrier without meeting it. If it should end on "
                   "the other, set simplify_tolerance on it to at least the gap, which snaps the end onto the "
                   "other; otherwise move them apart",
        "side by side": "two barrier parts run side by side that close without meeting. simplify_tolerance does "
                        "not join them: if they are one barrier drawn twice, keep one of them; otherwise move them "
                        f"apart, or {QUAD_BUFFER_ADVICE}",
        "boundary": "the barrier comes that close to the domain boundary or a hole without ending on it. End it on "
                    "the boundary or move it away: simplify_tolerance of at least the gap snaps a near-miss end "
                    f"onto the boundary and drops a part that runs along it. Or use {finer}",
        "feature": "a well, hex-ring seed or embedded point lies that close to the barrier. Move the point away "
                   f"from the barrier, or use {finer}",
        "fixed line": "an embedded line, a zone edge or a quad-buffer strip runs that close to the barrier "
                      f"without crossing it. Move it away from the barrier, or use {finer}",
        "short piece": "the barrier has vertices that close together (a densely digitised line or a short "
                       "zigzag). Set simplify_tolerance on the barrier, at least that distance, to remove them",
        "merge": "merging a narrow junction or boundary exit there (MeshGenerator.barrier_report['collapsed']) "
                 f"left barrier pieces that short; the input is not at fault. Until that is fixed, "
                 f"{QUAD_BUFFER_ADVICE}; please report the case",
        "gap field": "the GapField (MeshGenerator(gap_field=...)) asks for cells that small in the gap there "
                     "(gap / cells_across, but not below its min_size). "
                     + (f"Set GapField(min_size={round_up(MIN_H_FRACTION * lc / FINER_MARGIN):.3g}) or larger, "
                        if lc else "Set a larger GapField min_size, ")
                     + "or leave gap_field out",
        "size field": "the size field is that fine there (a finer zone, line or point next to the barrier). Use a "
                      "barrier resolution closer to it",
    }.get(cause, "barrier legs meet or fold back there at angles too narrow for cells between them (a narrow "
                 f"spike or zigzag, or many legs at one point). {remedy}")


def _fail(reason: str, points: np.ndarray, origin: np.ndarray) -> None:
    """Raise BarrierMeshError with the place, the cause and what to do about it."""
    raise BarrierMeshError(f"barrier faces cannot be made exact near {where(points, origin)}: {reason}.")


def touch_tolerance(domain) -> float:
    """Distance below which two things touch: TOL_FRACTION x the domain extent."""
    extent = float(max(np.ptp(np.asarray(domain.bounds).reshape(2, 2), axis=0)))
    return max(TOL_FRACTION * extent, 1e-12)


def barrier_lines(items: list, standard: list, domain) -> list:
    """BarrierLines (with must-keep vertices) from ``(line_id, xy, lc)`` of the cleaned barriers."""
    tol = touch_tolerance(domain)
    counts = {}
    for _, xy, _ in items:
        for p in map(tuple, xy):
            counts[p] = counts.get(p, 0) + 1
    std = shapely.union_all(standard) if standard else None
    out = []
    for line_id, xy, lc in items:
        keep = must_keep(xy, counts, domain, tol)
        if std is not None:
            keep |= shapely.distance(std, shapely.points(xy)) <= tol
        out.append(BarrierLine(line_id, xy, float(lc), 0.0, keep))
    return out


def floor_cause(net: Network, field: SizeField, k: int, lc: float, gradation: float = GRADATION) -> str:
    """What set the face length of sample k (``sizing.CAUSES``): a corridor (its sides), the cap of the sample
    its grading came from, else (a junction or ladder size) the tightest cap within ``lc`` of it, else
    "junction". A barrier cap from parts of one line only is a fold of that line ("junction")."""
    point = shapely.Point(field.xy[k])
    for c in net.corridors:
        if c.region.distance(point) <= 0.5 * lc:
            return "boundary" if len(c.sides) == 1 else "side by side"
    if field.cause is None or len(field.capped) != len(field.h):
        return "junction"
    d = np.hypot(*(field.xy - field.xy[k]).T)
    j = int(np.argmin(field.capped + gradation * d))
    if field.capped[j] + gradation * d[j] <= field.h[k] * (1 + 1e-6):
        cause = CAUSES[int(field.cause[j])]
    else:
        near = np.flatnonzero((d <= lc) & (field.cause != CAUSES.index("size field")))
        if len(near) == 0:
            return "junction"
        j = int(near[np.argmin(field.capped[near])])
        cause = CAUSES[int(field.cause[j])]
    if cause == "barrier" and not _other_line_near(net, field.xy[j], field.capped[j] / BARRIER_CAP):
        return "junction"
    if cause == "barrier" and not _free_end_near(net, field.xy[j], lc):
        return "side by side"
    return cause


def _free_end_near(net: Network, xy: np.ndarray, reach: float) -> bool:
    """True when a barrier end that is not on the domain boundary lies within ``reach`` of ``xy``."""
    ends = [v for v, pieces in enumerate(net.node_pieces) if len(pieces) == 1 and not net.plans[v].on_boundary]
    return bool(ends) and float(np.hypot(*(net.hard_xy[ends] - xy).T).min()) <= reach


def _other_line_near(net: Network, xy: np.ndarray, reach: float) -> bool:
    """True when a barrier line other than the one through ``xy`` comes within ``reach`` of it."""
    point = shapely.Point(xy)
    d = np.array([shapely.distance(LineString(b.xy), point) for b in net.lines])
    own = int(np.argmin(d))
    return bool(((d <= reach * (1 + 1e-6)) & (np.arange(len(d)) != own)).any())


def check_floor(net: Network, field: SizeField, origin: np.ndarray, drawn: set = frozenset()) -> None:
    """Fail when the target face length is below MIN_H_FRACTION x lc somewhere.

    ``drawn`` holds the vertices of the lines as they came in: a short piece
    that is not between two of them was made by a merge (``wedge``).
    """
    lc = net.piece_lc[field.piece]
    tiny = field.h < MIN_H_FRACTION * lc
    if tiny.any():
        k = int(np.argmin(field.h / lc))
        cause = floor_cause(net, field, k, lc[k])
        if cause == "short piece" and drawn and not {tuple(net.hard_xy[v]) for v in net.pieces[field.piece[k]]} <= drawn:
            cause = "merge"
        _fail(f"the faces would have to be {field.h[k]:.3g} long ({field.h[k] / lc[k]:.2g} x lc), because "
              f"{advice(cause, field.h[k], fold_tolerance(net, field.xy[[k]]), float(lc[k]))}", field.xy[tiny],
              origin)


def repair_bounds(field: SizeField, layout: Layout, bad: list, refine: np.ndarray | None) -> np.ndarray:
    """Per-sample upper bounds of h: REPAIR_FACTOR x the local face length at every defect."""
    out = np.full(len(field.h), np.inf) if refine is None else refine.copy()
    mids = 0.5 * (layout.xy[layout.segments[:, 0]] + layout.xy[layout.segments[:, 1]])
    points = np.array([p for _, p in bad])
    _, seg = cKDTree(mids).query(points)
    _, near = cKDTree(field.xy).query(points, k=min(3, len(field.h)))
    near = np.atleast_2d(near)
    if near.shape[0] != len(points):
        near = near.T
    for row, s in zip(near, np.atleast_1d(seg)):
        out[row] = np.minimum(out[row], REPAIR_FACTOR * layout.seg_size[s])
    return out


def network_with_corridors(lines: list, standard: list, domain, features: np.ndarray, size_fn,
                           sharp_angle: float, skip: set, fixed: list = ()) -> Network:
    """The barrier network with its corridors (except those in ``skip``) and their chord nodes re-planned.

    No corridor reaches across a standard line or a ``fixed`` line (zone
    edges, quad-buffer strips): its Gmsh nodes would fall inside the circles.
    """
    tol = touch_tolerance(domain)
    net = build_network(lines, standard, domain, tol, sharp_angle)
    net.corridors = corridors.find_corridors(net, size_fn, domain, list(standard) + list(fixed), features, skip)
    bent = corridors.bent_rungs(net, net.corridors)
    net.rungs = [r for r in net.rungs if not any(r is b for b in bent)]
    replan_chord_nodes(net, set().union(*[corridors.chord_nodes(net, c) for c in net.corridors]), domain, standard, tol)
    return net


def _troubled(net: Network, field: SizeField, bad: list) -> set:
    """Keys of the corridors that failed to lay out or have a defect in their region."""
    out = set(field.failed)
    for c in net.corridors:
        reach = c.region.buffer(0.5 * math.sqrt(max(c.region.area, 0.0)) + 1e-9)
        if any(reach.covers(shapely.Point(p)) for _, p in bad):
            out.add(c.key)
    return out


def plan(lines: list, standard: list, domain, features: np.ndarray, size_fn, origin: np.ndarray,
         gradation: float = GRADATION, sharp_angle: float = SHARP_ANGLE, obstacles: list = ()) -> BarrierPlan:
    """A barrier construction that passes the validity gate (module docstring), or BarrierMeshError.

    ``obstacles`` are the fixed lines (``sizing.Obstacle``) whose nodes the circles must not reach.
    A geometric failure inside the construction is raised as BarrierMeshError too.
    """
    try:
        return _repair_loop(lines, standard, domain, features, size_fn, origin, gradation, sharp_angle, obstacles)
    except GEOMETRY_ERRORS as err:
        raise internal_failure("construction", err) from err


def _repair_loop(lines: list, standard: list, domain, features: np.ndarray, size_fn, origin: np.ndarray,
                 gradation: float, sharp_angle: float, obstacles: list) -> BarrierPlan:
    """``plan``'s steps 1-4 with corridor drops and local repair rounds (module docstring)."""
    skip = set()
    tol = touch_tolerance(domain)
    drawn = {tuple(p) for b in lines for p in b.xy}
    fixed = [o.geometry.buffer(o.half_width) if o.half_width > 0 else o.geometry for o in obstacles]
    net = network_with_corridors(lines, standard, domain, features, size_fn, sharp_angle, skip, fixed)
    refine, first, rounds = None, None, 0
    while True:
        field = capped_field(net, size_fn, domain, features, obstacles, refine, tol)
        check_floor(net, field, origin, drawn)
        field = grade_field(net, field, gradation)
        if field.failed:
            skip |= set(field.failed)
            net = network_with_corridors(lines, standard, domain, features, size_fn, sharp_angle, skip, fixed)
            refine = None
            continue
        check_floor(net, field, origin, drawn)
        layout = construct(net, field)
        if not all_finite(layout):
            raise internal_failure("construction", FloatingPointError("non-finite radii or sites"))
        bad = defects(layout, features, domain)
        if not bad:
            return BarrierPlan(origin, net, field, layout, rounds)
        troubled = _troubled(net, field, bad)
        if troubled:
            skip |= troubled
            net = network_with_corridors(lines, standard, domain, features, size_fn, sharp_angle, skip, fixed)
            refine = None
            continue
        first = first or len(layout.sites)
        if rounds == REPAIR_ROUNDS or len(layout.sites) > SITE_GROWTH_CAP * first:
            kinds = sorted({k for k, _ in bad})
            points = np.array([p for _, p in bad])
            _fail(f"{len(bad)} defects ({', '.join(kinds)}) are left after {rounds} local refinements, because "
                  f"{advice('repair', 0.0, fold_tolerance(net, points))}", points, origin)
        refine = repair_bounds(field, layout, bad, refine)
        rounds += 1


# ---------------------------------------------------------------------------
# Handing the plan to Gmsh, and taking the mesh back
# ---------------------------------------------------------------------------
def fine_lines(layout: Layout, net: Network, size_fn, blend: float = FINE_BLEND,
               corridor_blend: float = CORRIDOR_BLEND) -> list:
    """(lines, resolution) of field-only lines where the barrier faces are finer than the size field."""
    mids = 0.5 * (layout.xy[layout.segments[:, 0]] + layout.xy[layout.segments[:, 1]])
    target = np.minimum(net.piece_lc[layout.seg_piece], size_fn(mids))
    in_corridor = np.zeros(len(mids), dtype=bool)
    for c in net.corridors:
        in_corridor |= shapely.covers(c.region.buffer(1e-9 * c.region.length), shapely.points(mids))
    power = np.where(in_corridor, corridor_blend, blend)
    size = layout.seg_size ** (1 - power) * target ** power
    classes = {}
    for k in np.flatnonzero(size < FINE_RATIO * target):
        d = float(size[k])
        key = int(math.floor(math.log(d) / math.log(CLASS_RATIO)))
        geoms, res = classes.get(key, ([], math.inf))
        classes[key] = (geoms + [layout.xy[layout.segments[k]]], min(res, d))
    return [(MultiLineString([LineString(g) for g in geoms]), res) for _, (geoms, res) in sorted(classes.items())]


def site_spacing(layout: Layout) -> np.ndarray:
    """Local face length at each site: the length of the nearest segment."""
    mids = 0.5 * (layout.xy[layout.segments[:, 0]] + layout.xy[layout.segments[:, 1]])
    _, idx = cKDTree(mids).query(layout.sites)
    return layout.seg_size[idx]


def embeddable(sites: np.ndarray, spacing: np.ndarray, edges, features: np.ndarray, rings: list) -> np.ndarray:
    """Sites that can be Gmsh points: clear of polygon edges and standard lines (``edges``), features and rings.

    ``rings`` holds ``(xy, resolution)`` of the kept hex-ring wells: Gmsh
    would split a ring with a point closer than HEX_RING_CLEARANCE x resolution.
    """
    ok = np.ones(len(sites), dtype=bool)
    if edges is not None and not edges.is_empty:
        ok &= shapely.distance(edges, shapely.points(sites)) > EMBED_GAP * spacing
    if len(features):
        d, _ = cKDTree(features).query(sites)
        ok &= d > EMBED_GAP * spacing
    for xy, res in rings:
        ok &= np.hypot(*(sites - xy).T) >= HEX_RING_CLEARANCE * res
    return ok


def nodes_to_drop(nodes: np.ndarray, layout: Layout, features: np.ndarray, protected: np.ndarray) -> tuple:
    """(remove, copies) masks of mesh nodes: remove those in a circle, crowding a site or on a site; ``copies``
    are those on a site (Gmsh's copies of the embedded sites).

    Feature nodes are never removed; ``protected`` nodes (on fixed lines)
    only in a circle or on a site, not for crowding.
    """
    scale = 1e-9 * max(1.0, float(np.abs(nodes).max()) if len(nodes) else 1.0)
    spacing = site_spacing(layout)
    d_site, nearest = cKDTree(layout.sites).query(nodes)
    copies = d_site <= scale
    remove = copies | ((d_site < CLEAR_FRACTION * spacing[nearest]) & ~protected)
    for idx in cKDTree(nodes).query_ball_point(layout.xy, layout.radius * (1 + CIRCLE_MARGIN)):
        remove[idx] = True
    if len(features):
        d_feat, _ = cKDTree(features).query(nodes)
        feature = d_feat <= 1e-6 * max(1.0, float(spacing.min()))
        remove &= ~feature
        copies &= ~feature
    return remove, copies
