"""Regression tests for MeshGenerator bookkeeping, logging and lazy exports."""
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

import vorflow
from vorflow.blueprint import ConceptualMesh
from vorflow import engine
from vorflow.engine import MeshGenerator, _SurvivorIndex, _match_heal_survivors

pytestmark = pytest.mark.slow  # gmsh-heavy end-to-end tests


def test_survivor_index_picks_nearest_then_same_tag_within_tolerance():
    survivors = _SurvivorIndex({(0, 1): (0.0, 0.0, 0.0), (0, 2): (4e-7, 0.0, 0.0),
                                (0, 3): (0.5, 0.0, 0.0), (0, 4): (0.5, 0.0, 0.0)},
                               tolerance=1e-6)
    # Two survivors round to the same 6-decimal key; the nearer one wins,
    # even over the same tag (healShapes reuses tags for other entities).
    assert survivors.match(0, 9, (3e-7, 0.0, 0.0)) == 2
    assert survivors.match(0, 1, (3e-7, 0.0, 0.0)) == 2
    # The same tag breaks a tie.
    assert survivors.match(0, 4, (0.5, 0.0, 0.0)) == 4
    # Offsets straddling a rounding boundary still match.
    assert survivors.match(0, 9, (0.5000004, 0.0, 0.0)) == 3
    assert survivors.match(0, 9, (0.25, 0.0, 0.0)) is None
    assert survivors.match(1, 9, (0.0, 0.0, 0.0)) is None


def _curve_signature(x0, y0, x1, y1):
    """Signature (bbox plus centre of mass) of a straight segment in the z=0 plane."""
    return (min(x0, x1), min(y0, y1), 0.0, max(x0, x1), max(y0, y1), 0.0,
            (x0 + x1) / 2, (y0 + y1) / 2, 0.0)


def test_heal_match_keeps_short_collinear_pieces_apart_and_prunes_deleted_ones():
    # Pieces near a three-line junction, all shorter than _HEAL_MATCH_TOL.
    # Healing renumbered 125 -> 124 and 126 -> 125, and deleted the 9e-6-long
    # piece 132. The old matcher kept 125 (now a different piece) because the
    # tag survived, and mapped 132 onto a neighbour instead of pruning it.
    left = _curve_signature(1.0, 1.0, 1.0001, 1.0)
    right = _curve_signature(1.0001, 1.0, 1.00018, 1.0)
    below = _curve_signature(1.0001, 0.98, 1.0001, 1.0)
    pre_heal = {(1, 125): left, (1, 126): right, (1, 131): below,
                (1, 132): _curve_signature(1.0001, 1.0, 1.0001, 1.000009)}
    drift = 1e-6
    survivors = _SurvivorIndex({(1, 124): left, (1, 125): right,
                                (1, 130): tuple(v + drift for v in below)}, tolerance=1e-4)
    assert _match_heal_survivors(pre_heal, survivors) == {
        (1, 125): 124, (1, 126): 125, (1, 131): 130, (1, 132): None}


def test_heal_match_is_one_to_one_for_curves_but_not_points():
    segment = _curve_signature(0.0, 0.0, 1.0, 0.0)
    pre_heal = {(1, 1): segment, (1, 2): tuple(v + 3e-6 for v in segment),
                (0, 1): (0.0, 0.0, 0.0), (0, 2): (2e-5, 0.0, 0.0)}
    survivors = _SurvivorIndex({(1, 7): tuple(v + 1e-6 for v in segment),
                                (0, 5): (1e-5, 0.0, 0.0)}, tolerance=1e-4)
    # The nearer curve claims the survivor; points merged by healing share it.
    assert _match_heal_survivors(pre_heal, survivors) == {
        (1, 1): 7, (1, 2): None, (0, 1): 5, (0, 2): 5}


def _barrier_model(with_point):
    """Domain with a barrier at x=20 and, optionally, a fine point far from it."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 100, 100), zone_id=1, resolution=10)
    cm.add_line(LineString([(20, 5), (20, 95)]), "barrier", resolution=10, is_barrier=True)
    if with_point:
        cm.add_point(Point(80, 80), "well", resolution=0.5)
    return cm.generate()


def test_point_field_does_not_leak_onto_barrier_with_same_feature_id():
    # Line 0's straddle points and point feature 0 used to share
    # gmsh_map['points'][0], so the point's fine field also refined the barrier.
    # Barrier sites are now point rows of their own, with fresh ids.
    def nodes_near_barrier(with_point):
        mg = MeshGenerator(background_lc=10, verbosity=0)
        assert mg.generate(*_barrier_model(with_point))
        return int(np.sum(np.abs(mg.nodes[:, 0] - 20) < 5))

    assert nodes_near_barrier(True) == nodes_near_barrier(False)


def test_barrier_curves_are_mapped_separately_from_point_features():
    # A barrier is a field-only curve under its own line id; point feature 0
    # keeps its single OCC point.
    polys, lines, points = _barrier_model(with_point=True)
    mg = MeshGenerator(background_lc=10, verbosity=0)
    mg._initialize_gmsh()
    try:
        gmsh_map = mg._add_geometry(polys, lines, points)
    finally:
        mg._finalize_gmsh()
    assert len(gmsh_map["points"][0]) == 1
    assert [dim for dim, _ in gmsh_map["lines"][0]] == [1]


def test_element_grid_is_built_lazily_and_cached():
    mg = MeshGenerator(background_lc=10, verbosity=0)
    assert mg.generate(*_barrier_model(with_point=False))
    assert mg.element_grid is None
    grid = mg.get_element_grid()
    assert not grid.empty
    assert mg.element_grid is not None
    assert len(mg.get_element_grid("triangles")) == len(grid)


def test_element_grid_zones_ignore_field_only_polygons():
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 20, 20), zone_id="domain", resolution=4)
    cm.add_polygon(box(5, 5, 15, 15), zone_id="sizing", resolution=1, z_order=5, embed=False)
    mg = MeshGenerator(background_lc=4, verbosity=0)
    assert mg.generate(*cm.generate())
    assert mg.get_element_grid()["zone_id"].eq("domain").all()


def test_mesh_generator_verbosity_does_not_change_package_level():
    logger = logging.getLogger("vorflow")
    before = logger.level
    MeshGenerator(background_lc=10, verbosity=0)
    assert logger.level == before

    mg = MeshGenerator(background_lc=10, verbosity=0)
    assert mg.generate(*_barrier_model(with_point=False))
    assert logger.level == before


def test_mesh_generator_verbosity_scopes_generate_output(caplog):
    model = _barrier_model(with_point=False)
    vorflow.set_verbosity(1, console=False)
    try:
        # caplog.records holds the whole test phase, and pytest >= 9.1 also
        # captures from non-propagating loggers, so drop the ConceptualMesh
        # progress messages logged while building the model.
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="vorflow"):
            MeshGenerator(background_lc=10, verbosity=0).generate(*model)
        assert not [r for r in caplog.records if r.levelno == logging.INFO]

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="vorflow"):
            MeshGenerator(background_lc=10).generate(*model)
        assert any("Generating Triangular Mesh" in r.getMessage() for r in caplog.records)
    finally:
        vorflow.set_verbosity(1)


def _legacy_zone_mesh(**legacy_kwargs):
    """Mesh a domain holding one zone added with pre-0.1 add_polygon keywords."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 60, 60), zone_id="domain", resolution=8)
    with pytest.warns(DeprecationWarning):
        cm.add_polygon(box(20, 20, 40, 40), zone_id="zone", resolution=4, z_order=1, **legacy_kwargs)
    mg = MeshGenerator(background_lc=8, verbosity=0)
    with warnings.catch_warnings():
        # dist_max (via dist_max_out) also warns from the engine's legacy path.
        warnings.simplefilter("ignore", DeprecationWarning)
        assert mg.generate(*cm.generate())
    return mg


def test_legacy_polygon_keywords_are_accepted_with_deprecation_warnings():
    mg = _legacy_zone_mesh(mesh_refinement=True, dist_max_out=10.0)
    assert len(mg.nodes) > 0


def test_legacy_border_density_grades_from_fine_border_to_resolution():
    # Densified boundary vertices alone force tiny edges on the border with an
    # abrupt jump inward (mean gamma ~0.77 there); the border field grades the
    # size smoothly into the zone.
    mg = _legacy_zone_mesh(border_density=1.0, dist_max_in=4.0)
    grid = mg.get_element_grid().merge(
        mg.get_triangular_quality()[["element_tag", "gamma"]], on="element_tag"
    )
    zone = box(20, 20, 40, 40)
    distance_to_border = grid.geometry.centroid.apply(zone.exterior.distance)
    transition = grid[(distance_to_border > 1) & (distance_to_border < 3)]
    assert transition.geometry.area.mean() < 1.5
    assert grid[distance_to_border < 3]["gamma"].mean() > 0.9


def test_generate_rejects_missing_background_lc_before_touching_gmsh():
    import gmsh

    mg = MeshGenerator(verbosity=0)
    with pytest.raises(ValueError, match="background_lc must be provided"):
        mg.generate(*_barrier_model(with_point=False))
    assert not gmsh.is_initialized()


def test_heal_remap_keeps_surfaces_with_identical_bounding_boxes_apart():
    # Two triangles tiling a square share a bounding box; matching healed
    # surfaces by bbox alone gave the fine triangle's field to both.
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10)]), zone_id="fine", resolution=1)
    cm.add_polygon(Polygon([(0, 0), (10, 10), (0, 10)]), zone_id="coarse", resolution=3)
    clean = cm.generate()
    maps = {}
    for heal in (False, True):
        mg = MeshGenerator(background_lc=3, heal_shapes=heal, verbosity=0)
        mg._initialize_gmsh()
        try:
            maps[heal] = mg._add_geometry(*clean)["surfaces"]
        finally:
            mg._finalize_gmsh()
    assert maps[True] == maps[False]
    assert len(maps[True][0]) == len(maps[True][1]) == 1


# Three lines meeting near a thin sliver polygon (cleaning_limitations_demo,
# Problem 4). While the sliver's hole in the domain surface was built inverted,
# healShapes duplicated its edges instead of sharing them, so line pieces ended
# on the domain surface's boundary without being in its topology; embedding
# them made Gmsh hang forever. With the hole built correctly the pieces bound
# the sliver instead; test_nonconforming_line_fragment_is_not_embedded covers
# the guard directly.
_HEALED_SLIVER_JUNCTION = """
import json, sys
import numpy as np
from shapely.geometry import LineString, Polygon, box
from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

line_kwargs = json.loads(sys.argv[1])
theta = np.deg2rad(5.0)
dx, dy = np.cos(theta) * 0.75, np.sin(theta) * 0.75
lines = [LineString([(0.25, 1), (1.75, 1)]),
         LineString([(1 - dx, 1 - dy), (1 + dx, 1 + dy)]),
         LineString([(1 + 1e-4, 0.35), (1 + 1e-4, 1.65)])]
sx, sy, w, length = 1.00018, 1.00002, 1e-3, 0.05
sliver = Polygon([(sx, sy - w / 2), (sx + length, sy - w / 2),
                  (sx + length, sy + w / 2), (sx, sy + w / 2)])
cm = ConceptualMesh(connectivity_tolerance=1e-12)
cm.add_polygon(box(0, 0, 2, 2), zone_id=1, resolution=0.45)
cm.add_polygon(sliver, zone_id=2, resolution=0.04, z_order=2, densify=False)
for i, line in enumerate(lines):
    cm.add_line(line, line_id=f"l{i}", resolution=0.04, **line_kwargs)
mg = MeshGenerator(background_lc=0.45, heal_shapes=True, heal_tolerance=5e-5,
                   optimization_cycles=0, smoothing_steps=0, verbosity=0)
assert mg.generate(*cm.generate())
grid = VoronoiTessellator(mg, cm).generate()
print(json.dumps({"cells": len(grid),
                  "skipped": mg.diagnostics["embedding"]["nonconforming_skip"]}))
"""


@pytest.mark.parametrize("line_kwargs", [{"growth_factor": 1.5}, {"dist_max": 0.12}],
                         ids=["growth_factor", "dist_max"])
def test_healed_sliver_at_line_junction_meshes_without_hanging(line_kwargs):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    try:
        result = subprocess.run(
            [sys.executable, "-c", _HEALED_SLIVER_JUNCTION, json.dumps(line_kwargs)],
            env=env, capture_output=True, text=True, timeout=120, check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Gmsh did not finish meshing the healed sliver junction within 120 s")
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["cells"] > 0
    assert report["skipped"] == 0


def _holed_domain_model():
    """The basic example plus a hole: a zone, a river, a barrier running into the hole and a well."""
    cm = ConceptualMesh()
    domain = Polygon(box(0, 0, 200, 200).exterior.coords, [box(80, 80, 120, 120).exterior.coords])
    cm.add_polygon(domain, zone_id=1)
    cm.add_polygon(box(50, 50, 75, 75), zone_id=2, resolution=2.0, z_order=1)
    river = LineString([(x, 75 + 20 * np.sin(0.1 * x)) for x in range(-10, 220, 10)])
    cm.add_line(river, line_id="river", resolution=2.0)
    cm.add_line(LineString([(100, 0), (100, 150)]), line_id="fault", resolution=2.0, is_barrier=True)
    cm.add_point(Point(25, 25), point_id="well", resolution=1.0)
    return cm.generate()


def test_polygon_holes_are_not_inverted_in_occ():
    # Shapely overlay output winds holes opposite to the shell; OCC
    # addPlaneSurface reverses hole loops itself, so passing them as is gave
    # a face of area shell + holes on which isInside() reported the hole as
    # inside and much of the domain as outside.
    import gmsh

    poly = box(0, 0, 200, 200).difference(box(80, 80, 120, 120)).difference(box(50, 50, 75, 75))
    assert poly.exterior.is_ccw != poly.interiors[0].is_ccw
    mg = MeshGenerator(background_lc=20, verbosity=0)
    mg._initialize_gmsh()
    try:
        s_tag, _ = mg._create_polygon_surface(poly)
        gmsh.model.occ.synchronize()
        assert gmsh.model.occ.getMass(2, s_tag) == pytest.approx(poly.area)
        inside = [gmsh.model.isInside(2, s_tag, [x, y, 0]) for x, y in
                  [(25, 25), (150, 150), (100, 100), (60, 60)]]
    finally:
        mg._finalize_gmsh()
    assert inside == [1, 1, 0, 0]


def test_features_in_a_holed_domain_are_embedded_as_triangle_vertices():
    # With the domain face inverted, 154 river, fault-straddle and well
    # entities matched no surface and were never embedded. Gmsh still meshed
    # them as free entities, so their nodes were in mg.nodes without being
    # triangle vertices and made sliver Voronoi cells along the lines. (The
    # fault is now a field-only line with embedded site points, so fewer
    # entities are embedded.)
    mg = MeshGenerator(background_lc=20.0, verbosity=0)
    assert mg.generate(*_holed_domain_model())
    embedding = mg.diagnostics["embedding"]
    assert embedding["skip_no_match"] == 0
    assert embedding["skip_no_cand"] == 0
    assert embedding["ok"] > 100
    vertices = {tuple(xy) for geom in mg.get_element_grid().geometry
                for xy in np.round(geom.exterior.coords, 9)}
    # Barrier sites are added and free nodes polished after Gmsh, so only
    # the fixed (feature and boundary) nodes are checked.
    fixed = mg.nodes[~mg.node_is_free & ~mg.node_is_site]
    loose = [xy for xy in np.round(fixed, 9) if tuple(xy) not in vertices]
    assert loose == []


def test_dedup_keeps_renumbered_curves_in_the_feature_map():
    # removeAllDuplicates renumbers curves (and reuses freed tags for other
    # entities); only points were remapped, so 163 of the river's 177 curves
    # were pruned from the map (no embedding, no refinement) and one entry
    # pointed at a piece of the domain boundary.
    import gmsh

    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 200, 200), zone_id=1)
    cm.add_polygon(box(50, 50, 75, 75), zone_id=2, resolution=2.0, z_order=1)
    river = LineString([(x, 75 + 20 * np.sin(0.1 * x)) for x in range(-10, 220, 10)])
    cm.add_line(river, line_id="river", resolution=2.0)
    cm.add_line(LineString([(100, 0), (100, 150)]), line_id="fault", resolution=2.0, is_barrier=True)
    polys, lines, points = cm.generate()
    mg = MeshGenerator(background_lc=20.0, verbosity=0)
    mg._initialize_gmsh()
    try:
        river_curves = mg._add_geometry(polys, lines, points)["lines"][0]
        centres = [Point(gmsh.model.occ.getCenterOfMass(1, tag)[:2]) for _dim, tag in river_curves]
        total_length = sum(gmsh.model.occ.getMass(1, tag) for _dim, tag in river_curves)
    finally:
        mg._finalize_gmsh()
    clean_river = lines.geometry.iloc[0]
    crossing = clean_river.intersection(lines.geometry.iloc[1])
    off_river = [c for c in centres if clean_river.distance(c) > 1e-6]
    # The barrier is a field-only line, so the river crosses it unbent and untrimmed.
    assert crossing.geom_type == "Point"
    assert off_river == []
    assert total_length == pytest.approx(clean_river.length, abs=1e-6)


def test_many_embedded_points_all_become_mesh_nodes():
    # Points are embedded with one embed() call per surface; their owner comes
    # from the fragmented topology instead of one isInside() per point, which
    # made embedding quadratic in the number of points.
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 200, 200), zone_id=1)
    cm.add_polygon(box(50, 50, 150, 150), zone_id=2, resolution=5.0, z_order=1)
    coords = [(x, y) for x in np.arange(4.0, 200.0, 8.0) for y in np.arange(4.0, 200.0, 8.0)]
    for i, xy in enumerate(coords):
        cm.add_point(Point(xy), point_id=f"p{i}", resolution=None)
    mg = MeshGenerator(background_lc=10.0, verbosity=0, diagnose=True)
    assert mg.generate(*cm.generate())
    embedding = mg.diagnostics["embedding"]
    assert embedding["ok"] == len(coords) == 625
    assert embedding["failed"] == embedding["skip_no_match"] == embedding["multi_match"] == 0
    assert len({record["surface"] for record in embedding["records"]}) == 2
    nodes = {tuple(xy) for xy in np.round(mg.nodes, 9)}
    assert all(xy in nodes for xy in coords)


def test_unembedded_entities_are_reported_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(MeshGenerator, "_surfaces_containing", staticmethod(lambda *args: []))
    monkeypatch.setattr(engine, "_interior_point_owners", lambda surface_tags: {})
    vorflow.set_verbosity(1, console=False)
    try:
        with caplog.at_level(logging.WARNING, logger="vorflow"):
            mg = MeshGenerator(background_lc=10, verbosity=0)
            assert mg.generate(*_barrier_model(with_point=True))
    finally:
        vorflow.set_verbosity(1)
    embedding = mg.diagnostics["embedding"]
    assert embedding["skip_no_match"] > 0
    assert len(embedding["unmatched_tags"]) == embedding["skip_no_match"]
    assert (0, embedding["unmatched_tags"][0][1], (80.0, 80.0)) in embedding["unmatched_tags"]
    assert any("no domain surface contains" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)


def test_nonconforming_line_fragment_is_not_embedded(monkeypatch):
    # A line piece ending on another surface's boundary without sharing a
    # vertex with the target surface made Gmsh hang (see the healed sliver
    # test above); such a piece must be skipped, not embedded.
    import vorflow.engine as engine

    monkeypatch.setattr(engine, "_endpoint_surfaces", lambda curve_tag: [{999}])
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 100, 100), zone_id=1, resolution=10)
    cm.add_line(LineString([(20, 20), (80, 80)]), "line", resolution=10)
    mg = MeshGenerator(background_lc=10, verbosity=0)
    assert mg.generate(*cm.generate())
    embedding = mg.diagnostics["embedding"]
    assert embedding["nonconforming_skip"] > 0
    assert embedding["ok"] == 0


def _face_crosses(face, line, tol=1e-6):
    """True if a face crosses ``line`` (touching it or lying on it does not count)."""
    # Faces on an oblique barrier lie on it only to round-off, so
    # face.crosses(line) flags most of them; require both ends off the line.
    ends = [Point(face.coords[0]), Point(face.coords[-1])]
    return face.intersects(line) and all(line.distance(p) > tol for p in ends)


OBLIQUE_BARRIER_DOMAINS = {
    # 60 degrees to the boundary at both ends, as in benchmark case v4_barrier.
    "outer_boundary": (box(0, 0, 200, 100), LineString([(128.87, 0), (71.13, 100)])),
    # Clipping splits the line at a hole whose edges it meets obliquely.
    "hole_edge": (
        Polygon(box(0, 0, 200, 100).exterior.coords,
                [[(80, 40), (120, 30), (130, 70), (90, 80)]]),
        LineString([(40, 0), (160, 100)]),
    ),
}


@pytest.mark.parametrize("domain_key", OBLIQUE_BARRIER_DOMAINS)
def test_oblique_barrier_end_sites_stay_inside_the_domain(domain_key):
    # A barrier meeting a boundary obliquely put one straddle point of each
    # end pair outside the domain: no surface embedded it, its node was not a
    # triangle vertex, and the boundary cell around the end straddled the line.
    # The boundary end now shares a site onto the boundary.
    from vorflow import VoronoiTessellator
    from vorflow.utils import build_connectivity

    domain, line = OBLIQUE_BARRIER_DOMAINS[domain_key]
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1)
    cm.add_line(line, line_id="barrier", resolution=2.0, is_barrier=True)
    polys, lines, points = cm.generate()
    mg = MeshGenerator(background_lc=10.0, verbosity=0)
    assert mg.generate(polys, lines, points)
    assert mg.diagnostics["embedding"]["unmatched_tags"] == []
    sites = mg.nodes[mg.node_is_site]
    assert domain.buffer(1e-9).covers(MultiPoint(sites))

    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()
    faces = build_connectivity(grid, center="generator").geometry
    barriers = list(lines.geometry)
    assert not any(_face_crosses(face, part) for face in faces for part in barriers)
    # The end pairs' bisector reaches the boundary, so no cell needs a mirror.
    assert vt.n_barrier_mirrors == 0


def _crossing_model(angle, offset):
    """A lc-2 line crossing a vertical lc-2 barrier at (50, 50.7), ``angle`` degrees from it.

    The crossing lies between two of the barrier's uniformly spaced pairs;
    ``offset`` shifts the line's densified vertices along it, so the
    crossing also falls between two of those.
    """
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 100, 100), zone_id=1)
    barrier = LineString([(50, 0), (50, 100)])
    cm.add_line(barrier, line_id="barrier", resolution=2.0, is_barrier=True)
    direction = np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    centre = np.array([50.0, 50.7])
    cm.add_line(LineString([centre - (30 + offset) * direction, centre + 30 * direction]),
                line_id="line", resolution=2.0)
    return cm, barrier


@pytest.mark.parametrize("angle, offset", [(90, 1.0), (30, 0.5)])
def test_line_crossing_a_barrier_shares_sites_with_it(angle, offset):
    # A line crossing a barrier was trimmed back by the barrier corridor, so
    # its end nodes sat at an arbitrary offset from the nearest straddle pair,
    # squeezed the cells there and needed 0-4 tessellator mirror cells. The
    # crossing is now a node of the barrier network: its circle holds no
    # mesh node (the line's nodes inside it are dropped) and the sites
    # around it make both the barrier and the line cell faces.
    from vorflow import VoronoiTessellator
    from vorflow.utils import build_connectivity

    cm, barrier = _crossing_model(angle, offset)
    mg = MeshGenerator(background_lc=10.0, verbosity=0)
    assert mg.generate(*cm.generate())
    assert mg.diagnostics["embedding"]["unmatched_tags"] == []
    crossing = np.array([50.0, 50.7])
    assert any(np.allclose(xy, crossing) for xy in mg.barrier_lines.geometry.iloc[0].coords)
    centres, radii = mg.barrier_circles
    k = int(np.argmin(np.hypot(*(centres - crossing).T)))
    assert np.allclose(centres[k], crossing)
    others = mg.nodes[~mg.node_is_site]
    assert np.hypot(*(others - crossing).T).min() > radii[k]

    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()
    faces = build_connectivity(grid, center="generator").geometry
    assert int(faces.crosses(barrier).sum()) == 0
    assert vt.n_barrier_mirrors == 0

    near = np.hypot(grid.x - 50.0, grid.y - 50.7) < 4.0
    compactness = 4 * np.pi * grid.geometry.area / grid.geometry.length ** 2
    assert compactness[near].min() > 0.65


@pytest.mark.parametrize("amplitude, wavelength, lc", [(8, 8, 2), (15, 5, 2), (8, 8, 5)])
def test_curved_barrier_leaves_no_sliver_cells(amplitude, wavelength, lc):
    # Straddle pairs' Voronoi faces are chords of a curved barrier, which
    # bulges across them by the sagitta. The post-hoc split made every bulge
    # a cell (80 on the 8/8/2 model, compactness < 0.3, areas down to 1e-15),
    # curvature tighter than lc put barrier mirrors beside straddle points
    # and outside the domain (15/5/2), and split() dropped pieces where the
    # barrier runs along a face to roundoff, leaving holes in the grid
    # (8/8/5). Bulges now join the cell on their side of the barrier.
    from vorflow import VoronoiTessellator
    from vorflow.utils import build_connectivity

    domain = box(0, 0, 100, 100)
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1)
    ys = np.linspace(0, 100, 60)
    curve = LineString(np.column_stack([50 + amplitude * np.sin(ys / wavelength), ys]))
    cm.add_line(curve, line_id="barrier", resolution=lc, is_barrier=True)
    polys, lines, points = cm.generate()
    mg = MeshGenerator(background_lc=10.0, verbosity=0)
    assert mg.generate(polys, lines, points)

    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()
    barrier = lines.geometry.iloc[0]
    # Every cell has a generator: no barrier fragments of its own.
    assert len(grid) == len(mg.nodes) + vt.n_barrier_mirrors
    assert grid.geometry.area.sum() == pytest.approx(domain.area, rel=1e-12)

    near = (grid.geometry.distance(barrier) < 3 * lc).to_numpy()
    area = grid.geometry.area.to_numpy()
    compactness = 4 * np.pi * area / grid.geometry.length.to_numpy() ** 2
    assert compactness[near].min() > 0.3
    assert area[near].min() > 1e-3 * lc ** 2

    faces = build_connectivity(grid, center="generator").geometry
    assert not any(_face_crosses(face, barrier) for face in faces)


@pytest.mark.parametrize("coords", [
    [(30, 40), (140, 170), (160, 150), (270, 260)],
    [(20, 150), (80, 210), (100, 190), (160, 250), (175, 235), (280, 280)],
    [(40, 60), (150, 250), (180, 60)],  # 34 degree spike
], ids=["single-kink", "zigzag", "spike"])
def test_kinked_barrier_cells_need_no_cut(coords):
    # A barrier vertex fell between two evenly spaced straddle pairs, so the
    # cell at the corner reached across the line. The tessellator then cut
    # it and merged the small piece into a neighbour, leaving cells that met
    # along two edges and faces up to 50 degrees off orthogonal. Each kink
    # is now a node of the barrier network with its own circle and sites.
    from vorflow import VoronoiTessellator
    from vorflow.utils import build_connectivity

    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 300, 300), zone_id=1, resolution=20)
    cm.add_line(LineString(coords), line_id="barrier", resolution=8, is_barrier=True)
    polys, lines, points = cm.generate()
    mg = MeshGenerator(background_lc=20, verbosity=0)
    assert mg.generate(polys, lines, points)
    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()

    assert vt.n_barrier_mirrors == 0
    # Every cell is a whole Voronoi cell: none was cut off by the barrier.
    assert len(grid) == len(mg.nodes)
    barrier = lines.geometry.iloc[0]
    connectivity = build_connectivity(grid, center="generator")
    assert not any(_face_crosses(face, barrier) for face in connectivity.geometry)
    near = connectivity.geometry.distance(barrier) < 16
    assert connectivity.loc[near, "ortho_error"].max() < 1e-6


def _multi_edge_pairs(grid, connectivity):
    """Neighbouring cell pairs whose shared boundary has more than one edge.

    A barrier cut whose small piece was merged into a neighbour leaves such a
    pair (a bent face), as does a straight face split by an extra vertex.
    """
    from shapely import MultiLineString, line_merge

    pairs = []
    for a, b in zip(connectivity["cell_id_1"], connectivity["cell_id_2"]):
        shared = grid.geometry.iloc[a].boundary.intersection(grid.geometry.iloc[b].boundary)
        segments = [part for part in getattr(shared, "geoms", [shared]) if part.geom_type == "LineString"]
        merged = line_merge(MultiLineString(segments))
        if sum(len(part.coords) - 1 for part in getattr(merged, "geoms", [merged])) > 1:
            pairs.append((a, b))
    return pairs


K67_BARRIER = [(30, 40), (140, 170), (160, 150), (270, 260)]
_SINE_X = np.linspace(40, 260, 61)


def _residual_grid(lines):
    """Mesh the 300 m square with ``lines`` (lc 8) and return (grid, mesh generator, tessellator)."""
    from vorflow import VoronoiTessellator

    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 300, 300), zone_id=1, resolution=20)
    for k, line in enumerate(lines):
        kwargs = {key: value for key, value in line.items() if key != "coords"}
        cm.add_line(LineString(line["coords"]), line_id=f"line{k}", resolution=8, **kwargs)
    polys, clean_lines, points = cm.generate()
    mg = MeshGenerator(background_lc=20, verbosity=0)
    assert mg.generate(polys, clean_lines, points)
    vt = VoronoiTessellator(mg, cm)
    return vt.generate(), mg, vt


@pytest.mark.filterwarnings("ignore:straddle_width is deprecated:FutureWarning")
@pytest.mark.parametrize("lines", [
    # Legs shorter than KINK_MIN_LEG * resolution kept the even spacing.
    [dict(coords=list(zip(_SINE_X, 150 + 40 * np.sin((_SINE_X - 40) / 220 * 2 * np.pi))),
          is_barrier=True)],
    # Pairs further apart than the resolution let mesh nodes in between
    # (straddle_width is now ignored).
    [dict(coords=K67_BARRIER, is_barrier=True, straddle_width=10.0)],
    # One barrier ending on another.
    [dict(coords=[(40, 150), (260, 150)], is_barrier=True),
     dict(coords=[(150, 150), (150, 270)], is_barrier=True)],
    # A standard line crossing the barrier next to a kink.
    [dict(coords=K67_BARRIER, is_barrier=True), dict(coords=[(100, 250), (180, 100)])],
], ids=["fine-curve", "wide-straddle", "t-junction", "crossing-near-kink"])
def test_barrier_residuals_need_no_cut(lines):
    # Layouts where the tessellator had to cut cells along a barrier with
    # straddle pairs, leaving cells without a generator or pairs of cells
    # that met along several edges.
    from vorflow.utils import build_connectivity

    grid, mg, vt = _residual_grid(lines)
    assert vt.n_barrier_mirrors == 0
    assert len(grid) == len(mg.nodes)
    assert _multi_edge_pairs(grid, build_connectivity(grid, center="generator")) == []


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="quad buffers on barriers keep the mirror pass and cut (out of scope)")
@pytest.mark.parametrize("lines", [
    # Transfinite points are spread per side, so the two node rows drift apart after a kink.
    [dict(coords=K67_BARRIER, is_barrier=True, quad_buffer=True)],
    # An even thickness puts a node row on the barrier itself.
    [dict(coords=[(40, 40), (260, 200)], is_barrier=True, quad_buffer=True, quad_buffer_thickness=2)],
], ids=["quad-kink", "quad-thickness-2"])
def test_quad_buffer_barrier_residuals_need_no_cut(lines):
    # Quad buffers on barriers still use the mirror pass and the post-hoc
    # cut, which leave cells without a generator or bent faces.
    from vorflow.utils import build_connectivity

    grid, mg, vt = _residual_grid(lines)
    assert len(grid) == len(mg.nodes) + vt.n_barrier_mirrors
    assert _multi_edge_pairs(grid, build_connectivity(grid, center="generator")) == []
