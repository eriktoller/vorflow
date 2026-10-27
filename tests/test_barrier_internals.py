"""Unit checks (no Gmsh) of the barrier construction's modules: validate, network, rung, sites, corridor, the
plan's repair and failure paths, the sparse node distances and the near-linear line cleaning.

Plans are made with ``pipeline.plan`` on a 300 x 300 domain with a uniform
size field, so every check runs in milliseconds.
"""
import dataclasses
import math

import numpy as np
import pytest
import shapely
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, box

from vorflow._barrier import corridor, lines, network, pipeline, rung, sizing, validate

DOMAIN = box(0, 0, 300, 300)
LC = 8.0
ORIGIN = np.zeros(2)


def _plan(coords: list, standard: list = (), features=None, lc: float = LC, background: float = 20.0,
          domain=DOMAIN):
    """``pipeline.plan`` of barrier lines given as coordinate lists (all with resolution ``lc``)."""
    items = [(k, np.asarray(xy, dtype=float), lc) for k, xy in enumerate(coords)]
    std = [LineString(s) for s in standard]
    barriers = pipeline.barrier_lines(items, std, domain)
    features = np.empty((0, 2)) if features is None else np.asarray(features, dtype=float)
    return pipeline.plan(barriers, std, domain, features, sizing.size_function([], background), ORIGIN)


def _network(coords: list, standard: list = (), domain=DOMAIN) -> network.Network:
    """The barrier network (with node plans, no corridors) of coordinate lists."""
    items = [(k, np.asarray(xy, dtype=float), LC) for k, xy in enumerate(coords)]
    std = [LineString(s) for s in standard]
    barriers = pipeline.barrier_lines(items, std, domain)
    return network.build_network(barriers, std, domain, pipeline.touch_tolerance(domain))


STRAIGHT = [[(40, 150), (260, 150)]]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def test_a_planned_layout_has_no_defects():
    plan = _plan(STRAIGHT)
    assert validate.defects(plan.layout, np.empty((0, 2)), DOMAIN) == []
    assert validate.all_finite(plan.layout)


def test_defects_find_a_site_or_feature_inside_a_circle():
    layout = _plan(STRAIGHT).layout
    centre = layout.xy[len(layout.xy) // 2]
    inside = layout.sites.copy()
    inside[0] = centre
    kinds = {k for k, _ in validate.defects(dataclasses.replace(layout, sites=inside), np.empty((0, 2)), DOMAIN)}
    assert "site in circle" in kinds
    kinds = {k for k, _ in validate.defects(layout, centre[None] + [0.01, 0.0], DOMAIN)}
    assert kinds == {"feature in circle"}


def test_defects_find_a_bad_pair_and_a_site_outside_the_domain():
    layout = _plan(STRAIGHT).layout
    bad = layout.pair_ok.copy()
    bad[3] = False
    assert [k for k, _ in validate.defects(dataclasses.replace(layout, pair_ok=bad), np.empty((0, 2)), DOMAIN)] \
        == ["pair"]
    small = box(0, 0, 300, 150.5)  # the sites above the barrier lie outside
    kinds = {k for k, _ in validate.defects(layout, np.empty((0, 2)), small)}
    assert kinds == {"site outside"}


def test_arc_defect_when_a_site_on_a_node_circle_lies_inside_a_pair_arc():
    layout = _plan(STRAIGHT).layout
    seg = 5
    node, other = layout.segments[seg]
    direction = layout.xy[other] - layout.xy[node]
    on_circle = layout.xy[node] + layout.radius[node] * direction / np.hypot(*direction)
    broken = dataclasses.replace(layout, extra=np.vstack([layout.extra, on_circle]),
                                 extra_node=np.append(layout.extra_node, node))
    assert ("arc", pytest.approx(layout.xy[node])) in [(k, p) for k, p in validate.arc_defects(broken)]


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
def test_nodes_and_pieces_share_junction_vertices():
    net = _network([[(40, 150), (150, 150), (260, 150)], [(150, 150), (150, 260)]])
    assert len(net.hard_xy) == 4 and len(net.pieces) == 3
    junction = int(np.flatnonzero((net.hard_xy == [150, 150]).all(axis=1))[0])
    assert np.bincount(net.pieces.ravel())[junction] == 3
    assert net.plans[junction].prescribes


def test_straight_vertex_is_dropped_and_a_free_end_gets_a_tip():
    net = _network([[(40, 150), (100, 150), (260, 150)]])
    assert len(net.hard_xy) == 2
    assert all(p.tip_direction is not None and not p.prescribes for p in net.plans)


def test_short_stub_past_a_junction_is_dropped_and_recorded():
    net = _network([[(40, 150), (150, 150), (260, 150)], [(150, 150), (150, 260)],
                    [(150, 150), (150, 150 - 0.05 * LC)]])
    assert [u[0] for u in net.uncovered] == ["stub"]
    assert net.uncovered[0][1] == pytest.approx(0.05 * LC)


def test_sharp_spike_tip_is_truncated_and_joined_by_a_ladder():
    apex = np.array([150.0, 250.0])
    legs = [apex + 190 * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]) for a in (-95, -85)]
    net = _network([[legs[0], apex, legs[1]]])
    assert [u[0] for u in net.uncovered] == ["spike tip"]
    assert len(net.rungs) == 1 and net.rungs[0].t0 > 0
    assert net.rungs[0].theta == pytest.approx(math.radians(10.0))


def test_narrow_wedge_at_a_junction_gets_a_rung():
    net = _network([[(40, 150), (150, 150), (260, 150)], [(150, 150), (260, 190)]])  # a 20 degree splay
    assert len(net.rungs) == 1
    assert net.rungs[0].t0 == 0 and net.rungs[0].theta == pytest.approx(math.atan2(40, 110))


# ---------------------------------------------------------------------------
# rung
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [5.0, 20.0, 50.0])
def test_ladder_sites_alternate_with_their_chords(theta_deg):
    y, sigma = rung.ladder(math.radians(theta_deg), 0.0, None, lambda t: LC, 200.0)
    assert len(y) >= 1 and np.all(np.diff(y) > 0) and np.all(np.diff(sigma) > 0)
    for k in range(1, len(sigma)):  # chord k lies between sites k-1 and k, half-way
        assert sigma[k - 1] < y[k - 1] < sigma[k]
        assert y[k - 1] == pytest.approx(0.5 * (sigma[k - 1] + sigma[k]))


def test_truncated_ladder_starts_at_its_chord():
    y, sigma = rung.ladder(math.radians(10.0), 3.0, None, lambda t: LC, 200.0)
    assert y[0] == pytest.approx(3.0 * math.cos(math.radians(5.0)))
    assert sigma[0] < y[0] < sigma[1]


# ---------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------
def test_pair_sites_are_mirror_images_on_both_end_circles():
    layout = _plan([[(40, 150), (120, 150), (180, 200), (260, 200)]]).layout
    i, j = layout.segments[:, 0], layout.segments[:, 1]
    for end in (i, j):
        for side in (0, 1):
            r = np.hypot(*(layout.pairs[:, side] - layout.xy[end]).T)
            assert np.allclose(r, layout.radius[end], rtol=1e-9)
    mid = 0.5 * (layout.pairs[:, 0] + layout.pairs[:, 1])
    seg = shapely.linestrings(np.stack([layout.xy[i], layout.xy[j]], axis=1))
    assert shapely.distance(seg, shapely.points(mid)).max() < 1e-9


def test_face_lengths_follow_the_size_field():
    layout = _plan(STRAIGHT, background=LC).layout
    assert layout.seg_size.max() <= LC * 1.000001
    assert layout.seg_size.sum() == pytest.approx(220.0)


# ---------------------------------------------------------------------------
# corridor
# ---------------------------------------------------------------------------
def test_collinear_pieces_have_no_bisector_and_no_corridor():
    a0, a1, b0, b1 = map(np.array, ([40.0, 150.0], [147.0, 150.0], [153.0, 150.0], [260.0, 150.0]))
    assert corridor.collinear(a0, a1, b0, b1)
    origin, m = corridor._bisector(a0, a1, b0, b1)
    assert np.isfinite(origin).all() and np.isfinite(m).all()
    assert not corridor.collinear(a0, a1, b0 + [0, 4], b1 + [0, 4])


@pytest.mark.parametrize("gap", [0.5, 2.0, 6.0, 9.0])
def test_two_lines_crossing_a_straight_barrier_close_together_plan(gap):
    standard = [[(150 - gap / 2, 100), (150 - gap / 2, 200)], [(150 + gap / 2, 100), (150 + gap / 2, 200)]]
    plan = _plan(STRAIGHT, standard=standard)
    assert plan.net.corridors == []


def test_t_junction_next_to_a_crossing_plans():
    plan = _plan([[(40, 150), (150, 150), (153, 150), (260, 150)], [(150, 150), (150, 260)]],
                 standard=[[(153, 100), (153, 200)]])
    assert validate.defects(plan.layout, np.empty((0, 2)), DOMAIN) == []


def test_close_parallel_barriers_make_a_corridor_with_sites_on_the_midline():
    plan = _plan([[(40, 150), (260, 150)], [(60, 154), (240, 154)]])
    [c] = plan.net.corridors
    assert len(c.sides) == 2 and c.ends == ("stagger", "stagger")
    on_midline = np.abs(plan.layout.sites[:, 1] - 152.0) < 1e-6
    assert on_midline.sum() >= 10


# ---------------------------------------------------------------------------
# pipeline: repair and failure paths
# ---------------------------------------------------------------------------
def test_defects_left_after_the_repair_rounds_raise(monkeypatch):
    calls = []

    def always_bad(layout, features, domain):
        calls.append(len(layout.sites))
        return [("pair", layout.xy[1])]

    monkeypatch.setattr(pipeline, "defects", always_bad)
    monkeypatch.setattr(pipeline, "SITE_GROWTH_CAP", 1e9)
    with pytest.raises(pipeline.BarrierMeshError,
                       match=f"1 defects \\(pair\\) left after {pipeline.REPAIR_ROUNDS} local refinements"):
        _plan(STRAIGHT)
    assert len(calls) == pipeline.REPAIR_ROUNDS + 1
    assert calls[-1] > calls[0]  # each round refines the faces at the defect


def test_site_growth_cap_stops_the_repair(monkeypatch):
    monkeypatch.setattr(pipeline, "defects", lambda layout, features, domain: [("arc", layout.xy[1])])
    monkeypatch.setattr(pipeline, "SITE_GROWTH_CAP", 1.0)
    with pytest.raises(pipeline.BarrierMeshError, match="left after") as info:
        _plan(STRAIGHT)
    rounds = int(str(info.value).split("left after ")[1].split()[0])
    assert 0 < rounds < pipeline.REPAIR_ROUNDS  # stopped by the growth cap, not the round limit


def test_troubled_names_the_corridor_with_a_defect_in_its_region():
    plan = _plan([[(40, 150), (260, 150)], [(60, 154), (240, 154)]])
    [c] = plan.net.corridors
    inside = [("site in circle", np.array([150.0, 152.0]))]
    far = [("site in circle", np.array([150.0, 40.0]))]
    assert pipeline._troubled(plan.net, plan.field, inside) == {c.key}
    assert pipeline._troubled(plan.net, plan.field, far) == set()


def test_a_corridor_with_a_defect_is_dropped_and_the_plan_retried(monkeypatch):
    real = pipeline.defects
    seen = []

    def corridor_defect(layout, features, domain):
        seen.append(len(seen))
        return [("site in circle", np.array([150.0, 152.0]))] if len(seen) == 1 else real(layout, features, domain)

    monkeypatch.setattr(pipeline, "defects", corridor_defect)
    plan = _plan([[(40, 150), (260, 150)], [(60, 154), (240, 154)]])
    assert plan.net.corridors == [] and len(seen) == 2


def test_geometric_failure_inside_the_plan_is_a_barrier_mesh_error(monkeypatch):
    def broken(net, field):
        raise shapely.errors.GEOSException("IllegalArgumentException: Points of LinearRing do not form a closed")

    monkeypatch.setattr(pipeline, "construct", broken)
    with pytest.raises(pipeline.BarrierMeshError, match="bug in vorflow.*quad_buffer=True") as info:
        _plan(STRAIGHT)
    assert isinstance(info.value.__cause__, shapely.errors.GEOSException)


def test_non_finite_sites_are_a_barrier_mesh_error(monkeypatch):
    real = pipeline.construct

    def nan_sites(net, field):
        layout = real(net, field)
        layout.sites[0] = np.nan
        return layout

    monkeypatch.setattr(pipeline, "construct", nan_sites)
    with pytest.raises(pipeline.BarrierMeshError, match="non-finite"):
        _plan(STRAIGHT)


def test_faces_below_the_floor_raise_before_grading():
    with pytest.raises(pipeline.BarrierMeshError, match="faces would have to be"):
        _plan([[(40, 150), (260, 150)], [(40, 150.01), (260, 150.01)]])


# ---------------------------------------------------------------------------
# sizing: sparse node distances
# ---------------------------------------------------------------------------
def test_node_distances_match_the_dense_matrix():
    net = _network([[(40, 150), (100, 160), (150, 150), (260, 150)], [(150, 150), (150, 260), (40, 270)],
                    [(100, 160), (100, 40)]])
    graph = sizing.node_graph(net)
    limit = 150.0
    dense = dijkstra(graph, directed=False, limit=limit)
    n = len(net.hard_xy)
    src, dst = np.repeat(np.arange(n), n), np.tile(np.arange(n), n)
    for chunk in (1, 3, 256):
        assert np.array_equal(sizing.node_distances(graph, limit, src, dst, chunk=chunk), dense.ravel())


# ---------------------------------------------------------------------------
# lines: end snapping, boundary parts, near-linear simplification and noise
# ---------------------------------------------------------------------------
def _line(xy, line_id="b", tol=0.0, lc=LC, source=0):
    return lines.BarrierLine(line_id, np.asarray(xy, dtype=float), lc, tol, source=source)


@pytest.mark.parametrize("coords", [[(100, 154), (260, 154)], [(100, 154), (180, 154.5), (255, 154)]])
def test_parallel_neighbour_is_not_snapped_onto_the_barrier(coords):
    fault, other = _line([(40, 150), (260, 150)]), _line(coords, "p", tol=5.0, source=1)
    lines.snap_ends([fault, other], DOMAIN, 1e-3, 1e-9)
    assert np.array_equal(other.xy, coords) and other.events == []


def test_near_miss_and_collinear_gap_still_snap():
    fault = _line([(40, 150), (260, 150)])
    branch = _line([(150, 154), (150, 260)], "t", tol=5.0, source=1)
    lines.snap_ends([fault, branch], DOMAIN, 1e-3, 1e-9)
    assert branch.xy[0].tolist() == [150, 150]
    continuation = _line([(264, 150), (290, 150)], "c", tol=5.0, source=2)
    lines.snap_ends([fault, continuation], DOMAIN, 1e-3, 1e-9)
    assert continuation.xy[0].tolist() == [260, 150]


def test_both_ends_snapping_onto_one_barrier_it_lies_along_are_undone():
    fault = _line([(40, 150), (260, 150)])
    arch = _line([(100, 153), (103, 154.5), (106, 153)], "a", tol=5.0, source=1)  # ends at 27 degrees
    lines.snap_ends([fault, arch], DOMAIN, 1e-3, 1e-9)
    assert arch.xy.tolist() == [[100, 153], [103, 154.5], [106, 153]] and arch.events == []
    lens = _line([(100, 153), (130, 175), (160, 153)], "l", tol=5.0, source=2)
    lines.snap_ends([fault, lens], DOMAIN, 1e-3, 1e-9)
    assert lens.xy[0].tolist() == [100, 150] and lens.xy[-1].tolist() == [160, 150]


@pytest.mark.parametrize("tol, last_y", [(3.0, 298.0), (6.0, 295.0), (1.0, 299.5), (3.0, 299.99)])
def test_end_left_by_a_dropped_boundary_part_moves_onto_the_boundary(tol, last_y):
    u = _line([(100, 300), (100, last_y), (100, 200), (200, 200), (200, last_y), (200, 300)], "u", tol=tol)
    kept, dropped = lines.drop_boundary_parts([u], DOMAIN, 1e-3)
    assert dropped == [] and len(kept) == 1
    assert kept[0].xy[0].tolist() == [100, 300] and kept[0].xy[-1].tolist() == [200, 300]
    assert [e["kind"] for e in kept[0].events] == ["boundary", "snap", "snap"]


def _fewest_chords_quadratic(xy, keep, tol, lc, short_penalty=lines.SHORT_PENALTY):
    """The original O(n^2)-O(n^3) ``fewest_chords``, as the reference."""
    n = len(xy)
    best, prev = np.full(n, np.inf), np.full(n, -1)
    best[0] = 0.0
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            if not lines._chord_ok(xy, i, j, tol):
                break
            length = np.hypot(*(xy[j] - xy[i]))
            cost = best[i] + 1.0 + short_penalty * max(0.0, 1.0 - length / lc) ** 2
            if cost < best[j]:
                best[j], prev[j] = cost, i
            if keep[i]:
                break
    chain = [n - 1]
    while chain[-1] != 0:
        chain.append(prev[chain[-1]])
    return np.array(chain[::-1])


def _clean_noise_quadratic(xy, keep, lc, fraction=lines.NOISE_FRACTION):
    """The original repeated-scan noise cleaning, as the reference: (xy, keep, dropped, largest, where)."""
    dropped, largest, where = 0, 0.0, None
    while len(xy) > 2:
        seg = np.hypot(*np.diff(xy, axis=0).T)
        options = [(lines._offset(xy[v], xy[v - 1], xy[v + 1]), v)
                   for s in np.flatnonzero(seg < fraction * lc) for v in (s, s + 1)
                   if 0 < v < len(xy) - 1 and not keep[v]]
        if not options:
            break
        move, v = min(options)
        if move >= largest:
            largest, where = move, xy[v].copy()
        xy, keep = np.delete(xy, v, axis=0), np.delete(keep, v)
        dropped += 1
    return xy, keep, dropped, largest, where


def _wiggly_lines():
    rng = np.random.default_rng(7)
    for n in (4, 40, 300):
        s = np.arange(n, dtype=float)
        yield np.column_stack([s, 3 * np.sin(s / 20)])
        yield np.column_stack([s, rng.normal(0, 0.5, n)])
        yield np.column_stack([s, np.where(np.arange(n) % 2, 1.0, -1.0)])
        yield np.column_stack([np.cumsum(rng.normal(0.3, 1, n)), np.cumsum(rng.normal(0, 1, n))])
        yield np.round(np.column_stack([s, 2 * np.sin(s / 7)]) * 2) / 2


@pytest.mark.parametrize("tol", [0.0, 0.3, 1.0, 2.0, 5.0])
def test_fewest_chords_equals_the_quadratic_search(tol):
    rng = np.random.default_rng(11)
    for xy in _wiggly_lines():
        xy = lines._dedupe(xy)
        keep = rng.random(len(xy)) < 0.05
        keep[[0, -1]] = True
        for lc in (1.0, 8.0, np.inf):
            assert np.array_equal(lines.fewest_chords(xy, keep, tol, lc), _fewest_chords_quadratic(xy, keep, tol, lc))


def test_fewest_chords_bounds_the_span_and_keeps_the_tolerance():
    xy = np.column_stack([np.arange(2000.0), 0.2 * np.sin(np.arange(2000.0))])
    keep = np.zeros(len(xy), dtype=bool)
    keep[[0, -1]] = True
    idx = lines.fewest_chords(xy, keep, 1.0, 50.0, max_span=300)
    assert idx[0] == 0 and idx[-1] == len(xy) - 1
    assert np.diff(idx).max() <= 300 and len(idx) == 8
    assert all(lines._chord_ok(xy, i, j, 1.0) for i, j in zip(idx[:-1], idx[1:]))


@pytest.mark.parametrize("lc", [5.0, 20.0, 80.0])
def test_clean_noise_equals_the_repeated_scan(lc):
    rng = np.random.default_rng(5)
    for xy in _wiggly_lines():
        xy = lines._dedupe(xy)
        keep = rng.random(len(xy)) < 0.05
        keep[[0, -1]] = True
        b = _line(xy.copy(), lc=lc)
        b.keep = keep.copy()
        lines.clean_noise([b])
        ref_xy, ref_keep, dropped, largest, where = _clean_noise_quadratic(xy, keep, lc)
        assert np.array_equal(b.xy, ref_xy) and np.array_equal(b.keep, ref_keep)
        noise = [e for e in b.events if e["kind"] == "noise"]
        assert (len(noise) == 1) == (dropped > 0)
        if dropped:
            assert noise[0]["distance"] == largest and noise[0]["location"] == tuple(map(float, where))
