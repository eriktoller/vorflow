"""Line-line junctions get one node, not two close ones (issue #31)."""
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box

from vorflow import ConceptualMesh, MeshGenerator
from vorflow import _junctions

RES = 30.0
LINE_A = LineString([(0, 1000), (2000, 1150)])
# Line B variants from the issue: on A, 5 m short of A, 5 m past A, crossing A.
B_ENDS_ON_A = LineString([(500, 300), (1200, 1090)])
B_SHORT = LineString([(500, 300), (1200, 1085)])
B_OVERSHOOT = LineString([(500, 300), (1200, 1100)])
B_CROSSES = LineString([(500, 300), (1500, 1800)])
EPS = 1e-6


def _model(line_b, resolution=RES, line_junction_fraction=0.5, **line_b_kwargs):
    cm = ConceptualMesh(line_junction_fraction=line_junction_fraction)
    cm.add_polygon(box(-500, 0, 2500, 2000), zone_id=1, resolution=160)
    cm.add_line(LINE_A, "A", resolution=resolution)
    cm.add_line(line_b, "B", resolution=resolution, **line_b_kwargs)
    return cm


def _junction(line_a, line_b):
    hits = line_a.intersection(line_b)
    assert hits.geom_type == "Point", hits.wkt
    return hits.coords[0]


def _closest_vertex_gap(line, junction):
    """Distance from the junction to the nearest line vertex other than the junction itself."""
    gaps = [math.dist(xy, junction) for xy in line.coords]
    return min(g for g in gaps if g > EPS)


class TestBlueprint:
    @pytest.mark.parametrize("line_b", [B_ENDS_ON_A, B_SHORT, B_OVERSHOOT, B_CROSSES])
    def test_junction_is_a_shared_vertex_with_no_close_neighbours(self, line_b):
        _, lines, _ = _model(line_b).generate()
        line_a, line_b = lines.geometry
        junction = _junction(line_a, line_b)
        assert junction in line_a.coords
        assert junction in line_b.coords
        for line in (line_a, line_b):
            assert _closest_vertex_gap(line, junction) >= 0.5 * RES

    def test_near_miss_end_is_snapped_onto_line_a(self):
        _, lines, _ = _model(B_SHORT).generate()
        line_b = lines.geometry.iloc[1]
        assert Point(line_b.coords[-1]).distance(LINE_A) < EPS
        assert Point(line_b.coords[0]).equals(Point(500, 300))

    @pytest.mark.parametrize("line_b", [
        B_OVERSHOOT,
        # Overshoot through an interior vertex: snapping the end would fold B across A.
        LineString([(500, 300), (1196, 1095), (1197, 1097)]),
    ])
    def test_overshoot_is_trimmed_at_the_crossing(self, line_b):
        crossing = LINE_A.intersection(line_b)
        _, lines, _ = _model(line_b).generate()
        line_a, trimmed = lines.geometry
        assert Point(trimmed.coords[-1]).distance(crossing) < EPS
        # Nothing of B is left on the far side of A.
        assert trimmed.difference(line_a.buffer(1e-3)).geom_type == "LineString"

    def test_crossing_lines_share_a_single_junction(self):
        _, lines, _ = _model(B_CROSSES).generate()
        line_a, line_b = lines.geometry
        junction = _junction(line_a, line_b)
        assert sum(math.dist(xy, junction) < 0.5 * RES for xy in line_a.coords) == 1
        assert sum(math.dist(xy, junction) < 0.5 * RES for xy in line_b.coords) == 1

    def test_lines_far_apart_are_unchanged(self):
        far = LineString([(500, 300), (1200, 700)])
        _, joined, _ = _model(far).generate()
        _, plain, _ = _model(far, line_junction_fraction=0).generate()
        for got, expected in zip(joined.geometry, plain.geometry):
            assert got.equals_exact(expected, 0)

    def test_fraction_zero_keeps_the_old_near_miss(self):
        _, lines, _ = _model(B_SHORT, line_junction_fraction=0).generate()
        assert Point(lines.geometry.iloc[1].coords[-1]).distance(LINE_A) > 1.0

    @pytest.mark.parametrize("kwargs", [{"is_barrier": True}, {"straddle_width": 2.0},
                                        {"quad_buffer": True}, {"embed": False}])
    def test_special_lines_are_left_alone(self, kwargs):
        _, joined, _ = _model(B_SHORT, **kwargs).generate()
        _, plain, _ = _model(B_SHORT, line_junction_fraction=0, **kwargs).generate()
        for got, expected in zip(joined.geometry, plain.geometry):
            assert got.equals_exact(expected, 0)

    def test_index_and_order_are_kept_and_generate_is_repeatable(self):
        cm = _model(B_CROSSES)
        _, first, _ = cm.generate()
        _, second, _ = cm.generate()
        assert list(first.index) == [0, 1]
        assert list(first.line_id) == ["A", "B"]
        for a, b in zip(first.geometry, second.geometry):
            assert a.equals_exact(b, 0)
        assert cm.raw_lines[1]["geometry"].equals(B_CROSSES)

    @pytest.mark.parametrize("offset", [10.0, 14.0])
    def test_parallel_lines_closer_than_the_tolerance_stay_apart(self, offset):
        # Each end is within 0.5 x 30 = 15 of the other line, but runs alongside it.
        cm = ConceptualMesh()
        cm.add_polygon(box(-500, -500, 1500, 1500), zone_id=1, resolution=160)
        cm.add_line(LineString([(0, 0), (1000, 0)]), "A", resolution=RES)
        cm.add_line(LineString([(0, offset), (1000, offset)]), "B", resolution=RES)
        _, lines, _ = cm.generate()
        line_a, line_b = lines.geometry
        assert line_a.distance(line_b) == pytest.approx(offset)
        assert cm.junction_diagnostics["snapped_ends"] == []

    def test_shallow_tributary_still_snaps(self):
        # Meets A at about 15 degrees and stops 5 m short.
        tributary = LineString([(200, 1015 - 5 - 800 * math.tan(math.radians(15))), (1000, 1070)])
        _, lines, _ = _model(tributary).generate()
        assert Point(lines.geometry.iloc[1].coords[-1]).distance(LINE_A) < EPS

    def test_diagnostics_report_junctions_snaps_and_trims(self):
        cm = _model(B_SHORT)
        cm.add_line(LineString([(500, 1800), (1600, 1300), (1601.5, 1112)]), "C", resolution=RES)
        cm.generate()
        report = cm.junction_diagnostics
        assert sorted(sorted(j["line_ids"]) for j in report["junctions"]) == [["A", "B"], ["A", "C"]]
        assert [(s["line_id"], s["end"]) for s in report["snapped_ends"]] == [("B", "end")]
        assert report["snapped_ends"][0]["from"] == (1200.0, 1085.0)
        assert [(t["line_id"], t["end"]) for t in report["trimmed_ends"]] == [("C", "end")]

        cm.line_junction_fraction = 0
        cm.generate()
        assert cm.junction_diagnostics == {"junctions": [], "snapped_ends": [], "trimmed_ends": []}

    @pytest.mark.parametrize("value", [-0.1, True, "0.5", float("nan")])
    def test_invalid_fraction_is_rejected(self, value):
        with pytest.raises(ValueError, match="line_junction_fraction"):
            ConceptualMesh(line_junction_fraction=value)


class TestHelpers:
    def test_snap_prefers_the_other_lines_end(self):
        a = LineString([(0, 0), (100, 0)])
        b = LineString([(103, 4), (200, 50)])
        (a, b), moved = _junctions.snap_line_ends([a, b], [10.0, 10.0], EPS)
        # A's end moves onto B's end; B's end is then already on A.
        assert len(moved) == 1
        assert a.coords[-1] == b.coords[0] == (103.0, 4.0)

    def test_snap_skips_anchored_ends(self):
        a = LineString([(0, 0), (100, 0)])
        b = LineString([(50, 5), (50, 100)])
        anchors = LineString([(0, 5), (100, 5)])
        (_, kept), moved = _junctions.snap_line_ends([a, b], [10.0, 10.0], EPS, anchors)
        assert moved == []
        assert kept.equals(b)

    def test_three_lines_through_one_point_share_one_coordinate(self):
        lines = [LineString([(-10, 0), (10, 0)]), LineString([(0, -10), (0, 10)]),
                 LineString([(-10, -10), (10, 10)])]
        junctions = _junctions.find_junctions(lines, EPS)
        assert all(len(points) == 1 for points in junctions)
        assert len({points[0] for points in junctions}) == 1

    def test_stub_is_trimmed_at_the_junction(self):
        line = LineString([(0, 0), (100, 0)])
        (trimmed,), count = _junctions.trim_short_stubs([line], [[(97.0, 0.0)]], [5.0], EPS)
        assert count == [(0, -1, 3.0)]
        assert trimmed.coords[-1] == (97.0, 0.0)

    def test_anchored_stub_is_kept(self):
        line = LineString([(0, 0), (100, 0)])
        anchors = LineString([(100, -1), (100, 1)])
        (kept,), count = _junctions.trim_short_stubs([line], [[(97.0, 0.0)]], [5.0], EPS, anchors)
        assert count == []
        assert kept.equals(line)

    def test_pin_inserts_the_junction_and_drops_close_vertices(self):
        line = LineString([(0, 0), (48, 0), (53, 0), (70, 0), (100, 0)])
        pinned = _junctions.pin_junctions(line, [(50.0, 0.0)], 15.0, EPS)
        assert list(pinned.coords) == [(0, 0), (50, 0), (70, 0), (100, 0)]

    def test_pin_keeps_line_ends_and_anchored_vertices(self):
        line = LineString([(0, 0), (8, 0), (100, 0)])
        anchors = LineString([(8, -1), (8, 1)])
        pinned = _junctions.pin_junctions(line, [(5.0, 0.0)], 15.0, EPS, anchors)
        assert list(pinned.coords) == [(0, 0), (5, 0), (8, 0), (100, 0)]


def _min_node_spacing(mesh):
    distances, _ = cKDTree(mesh.nodes).query(mesh.nodes, k=2)
    return distances[:, 1].min()


@pytest.mark.slow
@pytest.mark.parametrize("line_b,resolution", [
    (B_ENDS_ON_A, RES), (B_SHORT, RES), (B_OVERSHOOT, RES), (B_CROSSES, RES), (B_CROSSES, 10.0),
])
def test_mesh_nodes_near_junction_are_at_least_half_a_cell_apart(line_b, resolution):
    # Before the fix the minimum spacing was 6.0, 7.5, 3.4, 2.6 and 2.6 m.
    polys, lines, points = _model(line_b, resolution=resolution).generate()
    mesh = MeshGenerator(background_lc=160, verbosity=0)
    mesh.generate(polys, lines, points)

    assert _min_node_spacing(mesh) >= 0.5 * resolution
    junction = np.array(_junction(*lines.geometry))
    near = cKDTree(mesh.nodes).query_ball_point(junction, 0.5 * resolution)
    assert len(near) == 1
    assert np.linalg.norm(mesh.nodes[near[0]] - junction) < 1e-6
