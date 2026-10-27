"""Unit checks (no Gmsh) of the barrier helpers: circle-aware Lloyd steps, line cleaning, sizing caps, warnings.

The end-to-end properties of the barrier faces are in ``test_barrier_faces.py``.
"""
import types
import warnings

import numpy as np
import pytest
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point, box

from vorflow import ConceptualMesh, _lloyd
from vorflow._barrier import frames
from vorflow._barrier import lines as barrier_lines
from vorflow._barrier import sizing

UNIT_SQUARE_OWNER = np.array([box(-5, -5, 5, 5)], dtype=object)
CIRCLE = (np.array([[0.0, 0.0]]), np.array([1.0]))


def _constant(value):
    """Size function returning ``value`` everywhere."""
    return lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), float(value))


def _outside(points, circles) -> bool:
    """True when no point lies inside a circle."""
    return not _lloyd.inside_circles(points, *circles).any()


# ---------------------------------------------------------------------------
# Circle helpers
# ---------------------------------------------------------------------------
def test_inside_circles_uses_the_radius_and_margin():
    points = np.array([[0.5, 0.0], [1.0, 0.0], [1.05, 0.0], [3.0, 0.0]])
    assert _lloyd.inside_circles(points, *CIRCLE).tolist() == [True, False, False, False]
    assert _lloyd.inside_circles(points, *CIRCLE, margin=0.1).tolist() == [True, True, True, False]
    assert not _lloyd.inside_circles(np.empty((0, 2)), *CIRCLE).any()


def test_push_out_of_circles_moves_radially_past_the_margin():
    pushed = _lloyd.push_out_of_circles(np.array([[0.3, 0.4], [2.0, 2.0]]), *CIRCLE)
    assert np.hypot(*pushed[0]) == pytest.approx(1.0 + _lloyd.CIRCLE_PUSH_MARGIN)
    assert np.allclose(pushed[0] / np.hypot(*pushed[0]), [0.6, 0.8])
    assert np.array_equal(pushed[1], [2.0, 2.0])


def test_push_out_of_overlapping_circles_ends_outside_both():
    circles = (np.array([[0.0, 0.0], [1.5, 0.0]]), np.array([1.0, 1.0]))
    pushed = _lloyd.push_out_of_circles(np.array([[0.75, 0.1]]), *circles)
    assert _outside(pushed, circles)


def test_keep_out_of_circles_pushes_accepted_moves_out():
    old, new = np.array([[0.0, 3.0]]), np.array([[0.0, 0.5]])
    moved, ok = _lloyd.keep_out_of_circles(old, new, np.array([True]), UNIT_SQUARE_OWNER, CIRCLE)
    assert ok[0] and _outside(moved, CIRCLE)
    assert moved[0, 1] > 1.0


def test_keep_out_of_circles_rejects_a_push_across_a_constraint_line():
    # The move itself does not cross y = 0.5, but the radial push out of the circle does.
    old, new = np.array([[0.1, 0.2]]), np.array([[0.1, 0.3]])
    line = LineString([(-5, 0.5), (5, 0.5)])
    assert _lloyd.accept_moves(old, new, UNIT_SQUARE_OWNER, line)[0]
    moved, ok = _lloyd.keep_out_of_circles(old, new, np.array([True]), UNIT_SQUARE_OWNER, CIRCLE, line)
    assert not ok[0]
    assert np.array_equal(moved[0], old[0])


def test_keep_out_of_circles_rejects_a_push_out_of_the_owner():
    owner = np.array([box(-2.0, 0.3, 2.0, 1.0)], dtype=object)
    old, new = np.array([[0.6, 0.9]]), np.array([[0.0, 0.5]])  # pushed to (0, 1.05), above the owner
    moved, ok = _lloyd.keep_out_of_circles(old, new, np.array([True]), owner, CIRCLE)
    assert not ok[0] and np.array_equal(moved[0], old[0])


def test_relax_with_circles_keeps_every_free_node_out():
    x, y = np.meshgrid(np.linspace(-3, 3, 13), np.linspace(-3, 3, 13))
    nodes = np.column_stack([x.ravel(), y.ravel()])
    circles = (np.array([[0.1, 0.0]]), np.array([0.8]))
    nodes = nodes[~_lloyd.inside_circles(nodes, *circles)]
    free = (np.abs(nodes) < 2.9).all(axis=1)
    owners = np.array([box(-3, -3, 3, 3) if f else None for f in free], dtype=object)
    moved, report = _lloyd.relax(nodes, free, _constant(0.5), box(-3, -3, 3, 3), owners, None, 20,
                                 circles=circles)
    assert report["iterations"] > 0
    assert _outside(moved, circles)
    assert np.array_equal(moved[~free], nodes[~free])


def test_relax_allows_over_relaxation_only_when_asked():
    nodes, free = np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([True, False])
    owners = np.array([box(-2, -2, 2, 2), None], dtype=object)
    with pytest.raises(ValueError, match="damping"):
        _lloyd.relax(nodes, free, _constant(1.0), box(-2, -2, 2, 2), owners, None, 1, damping=1.3)
    _lloyd.relax(nodes, free, _constant(1.0), box(-2, -2, 2, 2), owners, None, 1, damping=1.3, max_damping=2.0)


def test_remove_short_faces_with_circles_keeps_every_free_node_out():
    x, y = np.meshgrid(np.arange(8.0), np.arange(8.0))
    nodes = np.column_stack([x.ravel(), y.ravel()])  # a square lattice: every face is short
    circles = (np.array([[3.5, 3.5]]), np.array([1.0]))
    nodes = nodes[~_lloyd.inside_circles(nodes, *circles)]
    free = (nodes > 0).all(axis=1) & (nodes < 7).all(axis=1)
    owners = np.array([box(0, 0, 7, 7) if f else None for f in free], dtype=object)
    moved, report = _lloyd.remove_short_faces(nodes, free, owners, None, 0.2, circles=circles)
    assert report["final"] < report["initial"]
    assert _outside(moved, circles)


# ---------------------------------------------------------------------------
# Line cleaning
# ---------------------------------------------------------------------------
DOMAIN = box(0, 0, 300, 300)


def _barrier(xy, line_id="b", lc=8.0, tol=0.0, source=0):
    """A BarrierLine in the local frame of DOMAIN."""
    return barrier_lines.BarrierLine(line_id, np.asarray(xy, dtype=float), lc, tol, source=source)


@pytest.mark.parametrize("length, tol", [(4.0, 5.0), (3.0, 3.5)])
def test_branch_shorter_than_its_tolerance_is_dropped_as_a_stub(length, tol):
    lines = [_barrier([(40, 150), (260, 150)]), _barrier([(150, 150), (150, 150 + length)], "c", tol=tol, source=1)]
    kept, dropped = barrier_lines.clean_barrier_lines(lines, [], DOMAIN, 1e-3)
    assert [b.line_id for b in kept] == ["b"]
    assert [b.line_id for b in dropped] == ["c"]
    stub = [e for e in dropped[0].events if e["kind"] == "stub"]
    assert len(stub) == 1 and stub[0]["distance"] == pytest.approx(length)


def test_overshoot_trim_ignores_the_junction_a_branch_starts_on():
    branch = _barrier([(150, 150), (150, 154)], "c", tol=5.0)
    trimmed = barrier_lines._trim_overshoot(branch, -1, LineString([(40, 150), (260, 150)]), 5.0, 1e-9)
    assert trimmed is None
    assert np.array_equal(branch.xy, [(150, 150), (150, 154)])


def test_barrier_on_the_boundary_is_dropped_whole_or_in_part():
    lines = [_barrier([(40, 0), (260, 0)], "edge"),
             _barrier([(40, 100), (150, 0), (260, 0)], "partial", source=1),
             _barrier([(0, 40), (0, 80), (0, 120), (100, 120), (100, 200), (0, 200), (0, 260)], "inside", source=2)]
    kept, dropped = barrier_lines.drop_boundary_parts(lines, DOMAIN, 1e-3)
    assert [b.line_id for b in dropped] == ["edge"]
    assert dropped[0].events[0]["kind"] == "boundary" and dropped[0].events[0]["distance"] == pytest.approx(220)
    parts = {b.line_id: [] for b in kept}
    for b in kept:
        parts[b.line_id].append(b.xy.tolist())
    assert parts["partial"] == [[[40, 100], [150, 0]]]
    # A part on the boundary inside the line splits it; its ends stay on the boundary.
    assert parts["inside"] == [[[0, 120], [100, 120], [100, 200], [0, 200]]]


def test_multi_part_standard_line_crossings_become_barrier_vertices():
    cm = ConceptualMesh()
    cm.add_polygon(DOMAIN, zone_id=1, resolution=20.0)
    cm.add_line(LineString([(40, 150), (260, 150)]), line_id="b", resolution=8.0, is_barrier=True)
    river = MultiLineString([[(100, 100), (100, 200)], [(200, 100), (200, 200)]])
    cm.add_line(river, line_id="river", resolution=8.0)
    _, lines, _ = cm.generate()
    barrier = lines.geometry[lines["line_id"] == "b"].iloc[0]
    assert {(100.0, 150.0), (200.0, 150.0)} <= set(barrier.coords)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def test_near_pairs_matches_a_global_radius_search():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 100, (600, 2))
    piece = rng.integers(0, 5, 600)
    h0 = rng.choice([0.5, 2.0, 15.0], 600)
    field = types.SimpleNamespace(xy=xy, piece=piece)
    i, j = sizing.near_pairs(field, h0, chunk=97)
    pairs = cKDTree(xy).query_pairs(sizing.CAP_REACH * h0.max(), output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    e = np.hypot(*(xy[a] - xy[b]).T)
    keep = (piece[a] != piece[b]) & (e < sizing.CAP_REACH * np.maximum(h0[a], h0[b]))
    found = set(zip(i.tolist(), j.tolist()))
    assert set(zip(a[keep].tolist(), b[keep].tolist())) <= found
    assert all(sizing.CAP_REACH * max(h0[p], h0[q]) >= np.hypot(*(xy[p] - xy[q])) for p, q in found)


def test_obstacle_cap_spares_crossings_barrier_ends_and_the_line_itself():
    xy = np.array([[0.0, 2.0], [40.0, 2.0], [50.0, 0.0], [80.0, 4.0], [95.0, 1.0]])
    field = types.SimpleNamespace(xy=xy, s=np.zeros(len(xy)))
    h0 = np.full(len(xy), 4.0)
    river = sizing.Obstacle("line 'r'", LineString([(0, 0), (100, 0)]), 0.0, np.array([[0.0, 0.0]]),
                            np.array([[100.0, 0.0]]))
    cap = sizing.obstacle_cap(field, [river], h0, 1e-9)
    assert np.isinf(cap[0])          # within CROSSING_REACH x h of the crossing
    assert cap[1] == pytest.approx(sizing.OBSTACLE_CAP * 2.0)
    assert np.isinf(cap[2])          # on the line
    assert np.isinf(cap[4])          # within PROTECT_FRACTION x h of a barrier end next to the line
    strip = sizing.Obstacle("quad", LineString([(0, 0), (100, 0)]), 3.0, np.empty((0, 2)), np.empty((0, 2)))
    assert sizing.obstacle_cap(field, [strip], h0, 1e-9)[3] == pytest.approx(sizing.OBSTACLE_CAP * 1.0)


# ---------------------------------------------------------------------------
# Warnings point at the caller
# ---------------------------------------------------------------------------
def test_noise_warning_is_attributed_to_the_caller():
    cm = ConceptualMesh()
    cm.add_polygon(DOMAIN, zone_id=1, resolution=20.0)
    cm.add_line(LineString([(40, 100), (150, 150), (150.3, 150.4), (260, 210)]), line_id="b", resolution=8.0,
                is_barrier=True)
    with pytest.warns(UserWarning, match="digitising noise") as record:
        cm.generate()
    assert {w.filename for w in record} == {__file__}


def test_boundary_barrier_warning_is_attributed_to_the_caller():
    cm = ConceptualMesh()
    cm.add_polygon(DOMAIN, zone_id=1, resolution=20.0)
    cm.add_line(LineString([(40, 0), (260, 0)]), line_id="b", resolution=8.0, is_barrier=True)
    with pytest.warns(UserWarning, match="domain boundary") as record:
        _, lines, _ = cm.generate()
    assert {w.filename for w in record} == {__file__}
    assert lines.empty
    assert [e["kind"] for e in lines.attrs["dropped_barrier_events"]] == ["boundary"]


def test_straddle_width_warns_future_warning_at_the_caller():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ConceptualMesh().add_line(LineString([(0, 0), (1, 1)]), line_id="s", resolution=1.0, straddle_width=1.0)
    assert [(w.category, w.filename) for w in record] == [(FutureWarning, __file__)]
    message = str(record[0].message)
    assert "value is ignored" in message
    assert "remove it there" in message  # on a barrier: nothing to do
    assert "only way to get that without is_barrier=True" in message  # on another line: what still gives faces


def test_barrier_without_resolution_is_cleaned_at_its_zone_resolution():
    cm = ConceptualMesh()
    cm.add_polygon(DOMAIN, zone_id=1, resolution=20.0)
    cm.add_line(LineString([(40, 100), (150, 150), (150.3, 150.4), (260, 210)]), line_id="b", resolution=None,
                is_barrier=True)
    with pytest.warns(UserWarning, match="digitising noise"):
        _, lines, _ = cm.generate()
    assert len(lines.geometry.iloc[0].coords) == 3


def test_ring_dropped_by_a_river_is_not_reported_as_near_a_barrier():
    cm = ConceptualMesh()
    cm.add_polygon(DOMAIN, zone_id=1, resolution=20.0)
    cm.add_line(LineString([(40, 50), (260, 50)]), line_id="b", resolution=8.0, is_barrier=True)
    cm.add_line(LineString([(40, 152), (260, 152)]), line_id="river", resolution=8.0)
    cm.add_point(geometry=Point(150, 150), point_id="w", resolution=4.0, hex_ring=True)
    with pytest.warns(UserWarning, match="hex_ring ignored"):
        _, _, points = cm.generate()
    assert points.iloc[0]["ring_seeds"] is None and not points.iloc[0]["ring_near_barrier"]


def test_removed_fixed_line_nodes_are_counted_per_line_and_warned_about():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    labels = np.array(["line 'r'", "line 'r'", None, "edge of zone 2"], dtype=object)
    dropped = np.array([True, True, False, True])
    fixed = frames._dropped_fixed(nodes, dropped, labels)
    assert fixed == [{"feature": "line 'r'", "nodes": 2, "location": (0.0, 0.0)},
                     {"feature": "edge of zone 2", "nodes": 1, "location": (3.0, 0.0)}]
    report = {"tips": [], "stubs": [], "snaps": [], "fixed_nodes": fixed}
    [message] = frames.report_warnings(report, 18.0)
    assert message.startswith("3 mesh node(s) of line 'r', edge of zone 2 lie inside the barrier circles")
