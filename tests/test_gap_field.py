"""Tests for the gap estimate (vorflow._gap) and the model-wide GapField."""
import logging
import re

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

import vorflow
from vorflow import ConceptualMesh, GapField, MeshGenerator, VoronoiTessellator
from vorflow._gap import GapWalls, SizeSource, estimate_gaps
from vorflow.engine import _gap_size_sources, _gap_walls

BG = 20.0
K = 3.0
GRADIENT = 0.2


def _estimate(polygons, lines=(), points=(), exempt=(), sources=(), min_size=0.1 * BG):
    """Gap estimate at BG and K for the given walls."""
    walls = GapWalls(list(polygons), list(lines), list(points), list(exempt))
    return estimate_gaps(walls, BG, K, GRADIENT, min_size, size_sources=list(sources))


def _parallel_lines(w):
    """Two 300 m lines w apart, far from the domain edge."""
    return [LineString([(150, 200), (450, 200)]), LineString([(150, 200 + w), (450, 200 + w)])]


# --- Gap estimate on analytic shapes ---


@pytest.mark.parametrize("w", [15.0, 6.0])
def test_parallel_lines_gap_matches_separation(w):
    est = _estimate([box(0, 0, 600, 400)], _parallel_lines(w))
    act = est.active
    mid = act & (est.xy[:, 0] > 160) & (est.xy[:, 0] < 440)
    assert mid.sum() > 50
    assert np.median(est.gap[mid]) == pytest.approx(w, rel=0.02)
    # Every active sample lies in the gap or just beyond the line ends.
    x, y = est.xy[act].T
    assert np.all((x > 150 - w) & (x < 450 + w) & (y > 200) & (y < 200 + w))


def test_circle_in_square_follows_medial_parabola():
    half, radius = 100.0, 80.0
    domain = box(-half, -half, half, half).difference(Point(0, 0).buffer(radius, quad_segs=64))
    est = _estimate([domain])
    act = est.active
    assert est.gap[act].min() == pytest.approx(half - radius, rel=0.05)
    # Rotate every sample into the sector facing the side x = half.
    xy = est.xy[act]
    turn = -np.round(np.arctan2(xy[:, 1], xy[:, 0]) / (np.pi / 2)) * np.pi / 2
    y = xy[:, 0] * np.sin(turn) + xy[:, 1] * np.cos(turn)
    x_exact = ((half + radius) ** 2 - y ** 2) / (2 * (half + radius))
    rel = est.r[act] / (half - x_exact) - 1
    assert abs(np.median(rel)) < 0.03


def test_curved_gap_between_concentric_arcs():
    radius, w = 200.0, 15.0
    arcs = [LineString(Point(0, 0).buffer(rad, quad_segs=64).exterior.coords[:129]) for rad in (radius, radius + w)]
    est = _estimate([box(-300, -300, 300, 300)], arcs)
    rad = np.hypot(*est.xy.T)
    inner = est.active & (rad > radius) & (rad < radius + w) & (est.xy[:, 1] < -10)
    assert inner.sum() > 50
    assert np.median(est.gap[inner]) == pytest.approx(w, rel=0.01)


@pytest.mark.parametrize("angle", [90.0, 45.0])
def test_junction_is_not_a_gap(angle):
    t = np.radians(angle)
    line = LineString([(400, 0), (400 + 150 * np.cos(t), 150 * np.sin(t))])
    est = _estimate([box(0, 0, 800, 600)], [line])
    assert not est.active.any()


def test_t_junction_between_lines_is_not_a_gap():
    lines = [LineString([(100, 300), (700, 300)]), LineString([(400, 300), (400, 450)])]
    est = _estimate([box(0, 0, 800, 600)], lines)
    assert not est.active.any()


def test_reflex_corner_is_not_a_gap():
    arm = 120.0
    domain = Polygon([(0, 0), (400, 0), (400, arm), (arm, arm), (arm, 400), (0, 400)])
    assert not _estimate([domain]).active.any()


def test_narrow_l_shape_arms_are_gaps():
    arm = 40.0
    domain = Polygon([(0, 0), (400, 0), (400, arm), (arm, arm), (arm, 400), (0, 400)])
    est = _estimate([domain])
    assert est.gap[est.active].min() == pytest.approx(arm, rel=0.01)


def test_dense_vertex_run_is_not_a_gap():
    x = np.arange(150, 450, 0.01)
    line = LineString(np.column_stack([x, np.full_like(x, 200.0)]))
    assert not _estimate([box(0, 0, 600, 400)], [line]).active.any()


@pytest.mark.parametrize("amplitude, period", [(1.0, 2.0), (5.0, 5.0)])
def test_zigzag_line_is_not_a_gap(amplitude, period):
    x = np.arange(150, 450 + 1e-9, period / 2)
    y = 200 + amplitude * (np.arange(len(x)) % 2)
    assert not _estimate([box(0, 0, 600, 400)], [LineString(np.column_stack([x, y]))]).active.any()


def test_polygon_neck_counts_as_gap():
    domain, strip = box(0, 0, 600, 400), box(0, 190, 600, 200)
    est = _estimate([domain.difference(strip), strip])
    assert np.median(est.gap[est.active]) == pytest.approx(10.0, rel=0.02)


def test_point_near_line_is_a_gap():
    est = _estimate([box(0, 0, 600, 400)], [LineString([(150, 200), (450, 200)])], [Point(300, 210)])
    assert est.active.any()
    assert est.gap[est.active].min() == pytest.approx(10.0, rel=0.05)


# --- Opt-out ---


def test_exempt_line_is_not_a_wall():
    lines = _parallel_lines(15.0)
    assert not _estimate([box(0, 0, 600, 400)], lines, exempt=[lines[1]]).active.any()


def test_exempt_polygon_boundary_is_not_a_wall():
    domain, strip = box(0, 0, 600, 400), box(0, 190, 600, 200)
    est = _estimate([domain.difference(strip), strip], exempt=[strip.boundary])
    assert not est.active.any()


def _two_line_model(gap_wall_b=True, embed_b=True, res=BG):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 600, 400), zone_id=1)
    lines = _parallel_lines(15.0)
    cm.add_line(lines[0], line_id="a", resolution=res)
    cm.add_line(lines[1], line_id="b", resolution=res, gap_wall=gap_wall_b, embed=embed_b)
    return cm


def test_gap_wall_false_line_is_exempt_in_engine_walls():
    walls = _gap_walls(*_two_line_model(gap_wall_b=False).generate())
    est = estimate_gaps(walls, BG, K, GRADIENT, 2.0)
    assert len(walls.exempt) == 1
    assert not est.active.any()


def test_field_only_line_is_not_a_wall():
    walls = _gap_walls(*_two_line_model(embed_b=False).generate())
    assert len(walls.lines) == 1
    assert not estimate_gaps(walls, BG, K, GRADIENT, 2.0).active.any()


def test_gap_wall_false_point_is_dropped():
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 600, 400), zone_id=1)
    cm.add_point(Point(300, 210), point_id="p", resolution=BG, gap_wall=False)
    assert _gap_walls(*cm.generate()).points == []


def test_gap_wall_column_survives_field_only_polygons():
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 600, 400), zone_id=1, gap_wall=False)
    cm.add_polygon(box(10, 10, 50, 50), zone_id=2, resolution=5, embed=False)
    polygons, _, _ = cm.generate()
    assert polygons.loc[polygons["zone_id"] == 1, "gap_wall"].tolist() == [False]


# --- Field size ---


def test_field_caps_size_only_near_gap():
    w = 15.0
    est = _estimate([box(0, 0, 600, 400)], _parallel_lines(w))
    h = est.sizes_at([[300, 200 + w / 2], [300, 200], [300, 200 + w + 10], [300, 350], [50, 50]])
    assert h[0] == pytest.approx(w / K, rel=0.03)
    assert h[1] == pytest.approx(w / K, rel=0.03)
    # Grows at the gradient beyond the ball, capped at the background far away.
    assert h[2] == pytest.approx(w / K + GRADIENT * 10, rel=0.05)
    assert h[3] == BG
    assert h[4] == BG


def test_fine_feature_sizes_leave_nothing_to_refine():
    lines = _parallel_lines(15.0)
    sources = [SizeSource(line, 3.0, GRADIENT) for line in lines]
    est = _estimate([box(0, 0, 600, 400)], lines, sources=sources)
    assert len(est.xy) > 0
    assert not est.active.any()


def test_size_sources_skip_explicit_and_coarse_features():
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 600, 400), zone_id=1)
    cm.add_line(LineString([(100, 100), (500, 100)]), line_id="fine", resolution=4)
    cm.add_line(LineString([(100, 200), (500, 200)]), line_id="coarse", resolution=40)
    cm.add_line(LineString([(100, 300), (500, 300)]), line_id="explicit", resolution=4,
                fields=[vorflow.GeometricGrowthField(1.1)])
    sources = _gap_size_sources(*cm.generate(), background_lc=BG)
    assert [s.size for s in sources] == [4.0]


def test_min_size_clips_size():
    est = _estimate([box(0, 0, 600, 400)], _parallel_lines(6.0), min_size=3.0)
    assert est.size[est.active].min() == 3.0
    assert est.floor_bound.any()


# --- Validation ---


@pytest.mark.parametrize("kwargs", [
    {"cells_across": 0}, {"cells_across": -2}, {"cells_across": True}, {"cells_across": "three"},
    {"cells_across": float("nan")}, {"growth_factor": 1.0}, {"growth_factor": "fast"},
    {"min_size": 0}, {"min_size": -1.0},
])
def test_gap_field_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        GapField(**kwargs)


def test_gap_field_min_size_must_be_below_background():
    with pytest.raises(ValueError, match="min_size"):
        GapField(min_size=BG).resolved_min_size(BG)


def test_gap_field_defaults():
    field = GapField()
    assert field.cells_across == 3.0
    assert field.growth_factor == 1.2
    assert field.resolved_min_size(40.0) == pytest.approx(4.0)


def test_mesh_generator_rejects_non_gap_field():
    with pytest.raises(ValueError, match="gap_field"):
        MeshGenerator(background_lc=BG, gap_field=3)


def test_gap_field_is_not_a_feature_field():
    cm = ConceptualMesh()
    with pytest.raises(ValueError, match="model-wide"):
        cm.add_line(LineString([(0, 0), (1, 1)]), line_id="a", resolution=1, fields=[GapField()])
    with pytest.raises(ValueError, match="model-wide"):
        GapField().create(None, {"lines": [1]}, BG)


@pytest.mark.parametrize("method, geometry", [
    ("add_polygon", box(0, 0, 1, 1)), ("add_line", LineString([(0, 0), (1, 1)])), ("add_point", Point(0, 0)),
])
def test_gap_wall_must_be_bool(method, geometry):
    cm = ConceptualMesh()
    with pytest.raises(ValueError, match="gap_wall"):
        getattr(cm, method)(geometry, "id", 1, gap_wall="no")


# --- Meshing ---


def _mesh(cm, gap_field, background_lc=BG):
    mesher = MeshGenerator(background_lc=background_lc, gap_field=gap_field, verbosity=0, conceptual_mesh=cm)
    mesher.generate()
    return mesher


def _single_line_model():
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 600, 400), zone_id=1)
    cm.add_line(LineString([(150, 200), (450, 200)]), line_id="a", resolution=BG)
    return cm


def test_no_narrow_gap_creates_no_field_and_same_mesh():
    plain = _mesh(_single_line_model(), None)
    gapped = _mesh(_single_line_model(), GapField(3))
    assert gapped.diagnostics["gap_field"]["field_created"] is False
    np.testing.assert_array_equal(plain.nodes, gapped.nodes)
    np.testing.assert_array_equal(plain.node_tags, gapped.node_tags)


def test_opted_out_gap_creates_no_field_and_same_mesh():
    plain = _mesh(_two_line_model(gap_wall_b=False), None)
    gapped = _mesh(_two_line_model(gap_wall_b=False), GapField(3))
    assert gapped.diagnostics["gap_field"]["field_created"] is False
    np.testing.assert_array_equal(plain.nodes, gapped.nodes)


def test_narrow_gap_creates_field():
    mesher = _mesh(_two_line_model(), GapField(3))
    diag = mesher.diagnostics["gap_field"]
    assert diag["field_created"] is True
    assert diag["min_gap"] == pytest.approx(15.0, rel=0.05)
    assert diag["n_floor_bound"] == 0


def test_min_size_warning_names_location_and_cells_across(caplog):
    vorflow.set_verbosity(1, console=False)
    try:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="vorflow"):
            _mesh(_two_line_model(), GapField(3, min_size=10.0))
        # pytest >= 9.1 can record a propagating message twice (once on the
        # vorflow logger, once on the root), so compare distinct messages.
        # Older Gmsh adds an unrelated quality-measure warning; keep GapField's.
        warnings = sorted({r.getMessage() for r in caplog.records
                           if r.levelno == logging.WARNING and r.getMessage().startswith("GapField")})
    finally:
        vorflow.set_verbosity(1)
    assert len(warnings) == 1
    assert "min_size=10" in warnings[0]
    assert "in 1 place(s)" in warnings[0]
    # The estimate samples every min_size / 2, so the 15 m gap reads slightly wide.
    assert re.search(r"\(\d+\.?\d*, 20\d\.?\d*\): gap 1[56]\.\d+, 1\.[56] cells across", warnings[0])


def _cells_between(grid, x, y0, y1):
    """Cells a transect strictly between two walls crosses."""
    return int(grid.intersects(LineString([(x, y0 + 1e-6), (x, y1 - 1e-6)])).sum())


@pytest.mark.slow
def test_narrow_gap_gets_cells_across():
    """Two lines 15 m apart in a 1000 x 600 domain at background 40."""
    def model():
        cm = ConceptualMesh()
        cm.add_polygon(box(0, 0, 1000, 600), zone_id=1)
        cm.add_line(LineString([(200, 300), (800, 300)]), line_id="a", resolution=40)
        cm.add_line(LineString([(200, 315), (800, 315)]), line_id="b", resolution=40)
        return cm

    transects = [300.0, 400.0, 500.0, 600.0, 700.0]
    plain = VoronoiTessellator(_mesh(model(), None, background_lc=40)).generate()
    gapped = VoronoiTessellator(_mesh(model(), GapField(3), background_lc=40)).generate()
    assert min(_cells_between(plain, x, 300, 315) for x in transects) < 3
    # The transect also crosses half of each cell centred on a line; the rest
    # lie wholly between the lines.
    interior = [_cells_between(gapped, x, 300, 315) - 2 for x in transects]
    assert min(interior) >= 3
    assert len(gapped) > len(plain)
