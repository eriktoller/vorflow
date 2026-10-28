"""Exact barrier faces (``vorflow._barrier``): every barrier segment is one Voronoi cell face.

Each model is meshed end to end and checked against the properties of the
construction (``_assert_exact``): no mirror generators and no cut cells, one
cell per mesh node, the barrier sites among the mesh nodes, no other node or
generator inside an empty circle, one single-edge, orthogonal face per
barrier segment (from circle centre to circle centre), and faces covering
the cleaned barrier lines except the reported spike tips and stubs. The
focused tests check one rule each (snapping, noise, stubs, spike tips,
corridors, options, errors, fixed lines next to a barrier, the domain
boundary, scale); ``REGRESSION_CASES`` are the held-out and stress shapes of
the barrier benchmark.
"""
import math
import warnings

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, Point, box

import vorflow
from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow.utils import build_connectivity

pytestmark = pytest.mark.slow  # gmsh-heavy end-to-end tests

DOMAIN = box(0, 0, 300, 300)
BACKGROUND_LC = 20.0
LC = 8.0
B = {"is_barrier": True}
CENTRE = (150.0, 150.0)
K67 = [(30, 40), (140, 170), (160, 150), (270, 260)]
UTM_SHIFT = (500000.0, 6500000.0)
TOUCH = 1e-6  # x domain extent: a face end this close to a barrier lies on it


def _ray(origin: tuple, direction_deg: float, length: float) -> tuple:
    """The point ``length`` from ``origin`` in direction ``direction_deg`` (counter-clockwise from +x)."""
    a = math.radians(direction_deg)
    return (origin[0] + length * math.cos(a), origin[1] + length * math.sin(a))


def _spike(angle_deg: float, apex: tuple = (150.0, 250.0), leg: float = 190.0) -> list:
    """A symmetric spike: two legs meeting at ``apex`` at ``angle_deg``."""
    return [_ray(apex, -90.0 - angle_deg / 2.0, leg), apex, _ray(apex, -90.0 + angle_deg / 2.0, leg)]


def _star(directions: list, arm: float = 110.0) -> list:
    """Separate barriers, each running from CENTRE in one of ``directions`` (deg)."""
    return [([CENTRE, _ray(CENTRE, d, arm)], B) for d in directions]


def _model(lines: list, points: list = (), polygons: list = (), domain=DOMAIN, background_lc: float = BACKGROUND_LC,
           **cm_kwargs) -> ConceptualMesh:
    """A conceptual model of ``domain`` with ``lines`` as (coords, add_line kwargs), resolution LC by default."""
    cm = ConceptualMesh(**cm_kwargs)
    cm.add_polygon(domain, zone_id=1, resolution=background_lc)
    for spec in polygons:
        cm.add_polygon(**spec)
    for k, (coords, kwargs) in enumerate(lines):
        cm.add_line(LineString(coords), **{"line_id": f"line{k}", "resolution": LC, **kwargs})
    for spec in points:
        cm.add_point(**spec)
    return cm


def _mesh(cm: ConceptualMesh, background_lc: float = BACKGROUND_LC, **mg_kwargs) -> tuple:
    """(mesh generator, tessellator, grid) of ``cm``."""
    mg = MeshGenerator(background_lc=background_lc, verbosity=0, **mg_kwargs)
    assert mg.generate(*cm.generate())
    vt = VoronoiTessellator(mg, cm)
    return mg, vt, vt.generate()


def _barriers(mg: MeshGenerator):
    """Union of the lines the barrier faces follow."""
    lines = mg.barrier_lines
    return shapely.union_all(lines.geometry[lines["is_barrier"].astype(bool)].to_numpy())


def _barrier_faces(grid, mg: MeshGenerator, tol: float):
    """Connectivity rows of the faces lying on a barrier, and of all faces."""
    connectivity = build_connectivity(grid, center="generator")
    faces = connectivity.geometry.to_numpy()
    barriers = _barriers(mg)
    on = np.ones(len(faces), dtype=bool)
    for point in (shapely.get_point(faces, 0), shapely.get_point(faces, -1), shapely.line_interpolate_point(
            faces, 0.5, normalized=True)):
        on &= shapely.distance(point, barriers) < tol
    return connectivity[on], connectivity


def _crossing_faces(connectivity, barriers, tol: float) -> int:
    """Faces that cross a barrier (touching it or lying on it does not count)."""
    faces = connectivity.geometry.to_numpy()
    ends_off = (shapely.distance(shapely.get_point(faces, 0), barriers) > tol) & (
        shapely.distance(shapely.get_point(faces, -1), barriers) > tol)
    return int((shapely.intersects(faces, barriers) & ends_off).sum())


def _segment_count(mg: MeshGenerator, tol: float) -> int:
    """Barrier segments: pairs of consecutive circle centres along the pieces of the lines the faces follow
    (a stretch shared by several lines, or crossed twice by one, counts once)."""
    centres = mg.barrier_circles[0]
    points = shapely.points(centres)
    pairs = set()
    for line in mg.barrier_lines.geometry[mg.barrier_lines["is_barrier"].astype(bool)]:
        xy = np.asarray(line.coords)
        for a, b in zip(xy[:-1], xy[1:]):
            on = np.flatnonzero(shapely.distance(LineString([a, b]), points) < tol)
            on = on[np.argsort(np.hypot(*(centres[on] - a).T))]
            pairs.update(frozenset(pair) for pair in zip(on[:-1], on[1:]))
    return len(pairs)


def _cleaned_barrier_length(cm: ConceptualMesh) -> float:
    """Length of the cleaned barrier input: the clean barrier rows plus the rows the cleaning dropped as stubs."""
    lines = cm.clean_lines
    barrier = lines["is_barrier"].fillna(False).astype(bool)
    dropped = sum(e["distance"] for e in lines.attrs.get("dropped_barrier_events", []) if e["kind"] == "stub")
    return float(lines.geometry[barrier].length.sum()) + dropped


def _assert_clear_of_circles(mg: MeshGenerator, points: np.ndarray) -> None:
    """No point other than a barrier site lies inside a circle."""
    centres, radii = mg.barrier_circles
    sites = mg.nodes[mg.node_is_site]
    is_site = cKDTree(sites).query(points)[0] <= 1e-9 * max(1.0, float(np.abs(points).max()))
    nearest = cKDTree(points[~is_site]).query(centres)[0]
    assert (nearest >= radii * (1 - 1e-9)).all()


def _assert_exact(mg: MeshGenerator, vt: VoronoiTessellator, grid) -> None:
    """The properties every exact barrier construction must have (module docstring)."""
    assert vt.n_barrier_mirrors == 0
    assert vt.n_exact_barrier_mirrors == 0 and vt.n_exact_barrier_cuts == 0
    assert len(grid) == len(mg.nodes)
    assert int(mg.node_is_site.sum()) == mg.barrier_report["sites"]
    _assert_clear_of_circles(mg, mg.nodes)
    _assert_clear_of_circles(mg, grid[["x", "y"]].to_numpy(dtype=float))  # after the tessellator's own moves

    tol = TOUCH * max(np.ptp(np.asarray(grid.total_bounds).reshape(2, 2), axis=0))
    on, connectivity = _barrier_faces(grid, mg, tol)
    assert _crossing_faces(connectivity, _barriers(mg), tol) == 0
    assert all(len(face.coords) == 2 for face in on.geometry)
    assert on["ortho_error"].max() < 1e-6
    # One face per segment, each from one circle centre to the next.
    ends = np.vstack([shapely.get_coordinates(shapely.get_point(on.geometry.to_numpy(), k)) for k in (0, -1)])
    assert (cKDTree(mg.barrier_circles[0]).query(ends)[0] < tol).all()
    assert len(on) == _segment_count(mg, tol)
    # Each pair of cells meets along at most one barrier face, and the faces cover the barriers.
    assert not on.duplicated(["cell_id_1", "cell_id_2"]).any()
    assert on.geometry.length.sum() == pytest.approx(_barriers(mg).length, rel=1e-6)
    # ... which are the cleaned input lines less what the report says was left without faces. A merged
    # wedge moves its legs (by at most its reported deviation) and they share a stem, so there the faces
    # are checked against the input by distance, both ways, instead of by length.
    report = mg.barrier_report
    uncovered = sum(entry["length"] for entry in report["tips"] + report["stubs"])
    if report["collapsed"]:
        domain = shapely.union_all(vt.cm.clean_polygons.geometry.to_numpy())
        bound = max(entry["deviation"] for entry in report["collapsed"])
        assert _moved(mg, vt.cm, domain, report["tips"] + report["stubs"]) <= bound * (1 + 1e-3) + tol
    else:
        assert on.geometry.length.sum() == pytest.approx(_cleaned_barrier_length(vt.cm) - uncovered, rel=1e-6)


# ---------------------------------------------------------------------------
# Exactness on the basic shapes
# ---------------------------------------------------------------------------
EXACT_CASES = {
    "straight": [([(40, 60), (260, 240)], B)],
    "kink": [(K67, B)],
    "zigzag": [([(20, 150), (80, 210), (100, 190), (160, 250), (175, 235), (280, 280)], B)],
    "t_junction": [([(40, 150), (260, 150)], B), ([(150, 150), (150, 270)], B)],
    "x_crossing": [([(40, 150), (260, 150)], B), ([_ray(CENTRE, 220, 110), _ray(CENTRE, 40, 110)], B)],
    "y_junction": _star([90.0, 210.0, 330.0]),
    "standard_crossing": [([(40, 150), (260, 150)], B), ([_ray(CENTRE, 210, 110), _ray(CENTRE, 30, 110)], {})],
    "standard_end": [([(40, 150), (260, 150)], B), ([(90, 262), CENTRE], {})],
    "boundary_end": [([(245, 40), (300, 191.0)], B)],
    "corner_end": [([(150, 180), (300, 300)], B)],
    "ring": [([(90, 100), (200, 80), (240, 170), (180, 240), (100, 210), (70, 150), (90, 100)], B)],
}


@pytest.mark.parametrize("name", EXACT_CASES)
def test_barrier_segments_are_exact_faces(name):
    mg, vt, grid = _mesh(_model(EXACT_CASES[name]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["tips"] == [] and mg.barrier_report["stubs"] == []


def test_meshing_is_deterministic():
    lines = EXACT_CASES["kink"] + [([(110, 280), K67[1]], B)]
    first, _, grid_a = _mesh(_model(lines))
    second, _, grid_b = _mesh(_model(lines))
    assert np.array_equal(first.nodes, second.nodes)
    assert np.array_equal(first.node_is_site, second.node_is_site)
    assert grid_a.geometry.geom_equals_exact(grid_b.geometry, 0.0).all()


def test_utm_offset_does_not_change_the_barrier_sites():
    # The construction works in a frame whose origin is the domain's lower-left corner.
    lines = EXACT_CASES["kink"] + [([(110, 280), K67[1]], B)]
    shifted = [([(x + UTM_SHIFT[0], y + UTM_SHIFT[1]) for x, y in coords], kwargs) for coords, kwargs in lines]
    near, _, _ = _mesh(_model(lines))
    far, vt, grid = _mesh(_model(shifted, domain=affinity.translate(DOMAIN, *UTM_SHIFT)))
    _assert_exact(far, vt, grid)

    def sites(mg, shift):
        xy = mg.nodes[mg.node_is_site] - np.asarray(shift)
        return xy[np.lexsort(np.round(xy, 4).T[::-1])]

    assert near.barrier_report["sites"] == far.barrier_report["sites"]
    assert np.allclose(sites(near, (0, 0)), sites(far, UTM_SHIFT), atol=1e-6)


# ---------------------------------------------------------------------------
# Spike tips, stubs, snapping and noise
# ---------------------------------------------------------------------------
def test_sharp_spike_tip_is_left_inside_cells_and_reported():
    coords = _spike(12.0)
    mg, vt, grid = _mesh(_model([(coords, B)]))
    _assert_exact(mg, vt, grid)
    tips = mg.barrier_report["tips"]
    assert len(tips) == 1 and tips[0]["line_id"] == "line0"
    assert 0 < tips[0]["length"] < LC
    assert Point(tips[0]["location"]).distance(Point(coords[1])) < LC
    # The faces stop short of the apex.
    assert _barriers(mg).length < LineString(coords).length - tips[0]["length"]


def test_spike_tip_is_kept_above_sharp_angle():
    coords = _spike(25.0)
    mg, vt, grid = _mesh(_model([(coords, B)]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["tips"] == []
    assert _barriers(mg).length == pytest.approx(LineString(coords).length)

    mg, vt, grid = _mesh(_model([(coords, B)]), sharp_angle=30.0)
    _assert_exact(mg, vt, grid)
    assert len(mg.barrier_report["tips"]) == 1


def test_short_stub_past_a_junction_is_dropped_and_reported():
    # The branch runs 0.5 m (0.06 lc) past the barrier it crosses.
    lines = [([(40, 150), (260, 150)], B), ([(150, 270), (150, 149.5)], B)]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    stubs = mg.barrier_report["stubs"]
    assert len(stubs) == 1 and stubs[0]["line_id"] == "line1"
    assert stubs[0]["length"] == pytest.approx(0.5)
    assert _barriers(mg).length == pytest.approx(220.0 + 120.0)


@pytest.mark.parametrize("end, kind", [(150.5, "snap"), (149.5, "trim")], ids=["near-miss", "overshoot"])
def test_end_within_simplify_tolerance_joins_the_other_barrier(end, kind):
    lines = [([(40, 150), (260, 150)], B), ([(150, 270), (150, end)], {**B, "simplify_tolerance": 1.0})]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    snaps = mg.barrier_report["snaps"]
    assert [(s["line_id"], s["kind"]) for s in snaps] == [("line1", kind)]
    assert snaps[0]["distance"] == pytest.approx(0.5)
    assert mg.barrier_report["stubs"] == []
    branch = mg.barrier_lines.geometry[mg.barrier_lines["line_id"] == "line1"].iloc[0]
    assert Point(branch.coords[-1]).distance(Point(150, 150)) < 1e-9


def test_near_miss_without_tolerance_raises_barrier_mesh_error():
    # An end 0.1 m (0.0125 lc) from another barrier would need faces shorter than 0.02 lc.
    lines = [([(40, 150), (260, 150)], B), ([(150, 270), (150, 150.1)], B)]
    with pytest.raises(vorflow.BarrierMeshError, match="simplify_tolerance") as info:
        _mesh(_model(lines))
    assert isinstance(info.value, ValueError)
    assert "(150.00, 150." in str(info.value)


def test_digitising_noise_is_cleaned_with_a_warning():
    noise = _ray(CENTRE, 60.0, 0.05 * LC)
    cm = _model([([(40, 100), CENTRE, noise, (260, 210)], B)])
    with pytest.warns(UserWarning, match="digitising noise"):
        mg, vt, grid = _mesh(cm)
    _assert_exact(mg, vt, grid)
    noise_events = mg.barrier_report["noise"]
    assert len(noise_events) == 1 and noise_events[0]["line_id"] == "line0"
    assert 0 < noise_events[0]["distance"] < 0.05 * LC
    assert len(mg.barrier_lines.geometry.iloc[0].coords) == 3


# ---------------------------------------------------------------------------
# Corridors: one shared row of cells across a narrow gap
# ---------------------------------------------------------------------------
def test_close_parallel_barriers_share_one_row_of_cells():
    lines = [([(40, 150), (260, 150)], B), ([(40, 154), (260, 154)], B)]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    corridors = mg.barrier_report["corridors"]
    assert len(corridors) == 1 and corridors[0]["kind"] == "barriers"
    assert sorted(corridors[0]["line_ids"]) == ["line0", "line1"]
    assert corridors[0]["length"] > 150.0
    # Every cell between the barriers has its site on the midline and touches both barriers.
    between = (grid.x > 60) & (grid.x < 240) & (grid.y > 150) & (grid.y < 154)
    assert between.sum() > 10
    assert np.allclose(grid.y[between], 152.0, atol=1e-6)
    for line in (LineString(lines[0][0]), LineString(lines[1][0])):
        assert (grid.geometry[between].distance(line) < 1e-9).all()


def test_barrier_along_the_boundary_shares_its_sites_onto_the_boundary():
    lines = [([(40, 0.4 * LC), (260, 0.4 * LC)], B)]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    corridors = mg.barrier_report["corridors"]
    assert len(corridors) == 1 and corridors[0]["kind"] == "boundary"
    assert corridors[0]["line_ids"] == ["line0"]
    # The cells between the barrier and the boundary have their sites on the boundary.
    below = (grid.x > 60) & (grid.x < 240) & (grid.y < 0.4 * LC)
    assert below.sum() > 10
    assert np.allclose(grid.y[below], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Options, report and the tessellator's safety net
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"barrier_gradation": 0.0}, {"barrier_gradation": -0.3}, {"sharp_angle": 60.0}, {"sharp_angle": 0.0},
    {"barrier_polish_iterations": -1},
])
def test_barrier_options_are_validated(kwargs):
    with pytest.raises(ValueError):
        MeshGenerator(background_lc=BACKGROUND_LC, verbosity=0, **kwargs)


def test_keep_close_rings_must_be_a_bool():
    with pytest.raises(ValueError, match="keep_close_rings"):
        ConceptualMesh(keep_close_rings="yes")


def test_straddle_width_is_deprecated_and_ignored():
    lines = EXACT_CASES["kink"]
    plain, _, _ = _mesh(_model(lines))
    with pytest.warns(FutureWarning, match="straddle_width"):
        cm = _model([(lines[0][0], {**B, "straddle_width": 10.0})])
    mg, vt, grid = _mesh(cm)
    _assert_exact(mg, vt, grid)
    assert np.array_equal(plain.nodes, mg.nodes)


WELL = {"geometry": Point(150, 150), "point_id": "well", "resolution": 4.0, "hex_ring": True}


def test_hex_ring_next_to_a_barrier_is_dropped_unless_kept():
    lines = [([(40, 150 - 0.6 * LC), (260, 150 - 0.6 * LC)], B)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mg, vt, grid = _mesh(_model(lines, points=[WELL]))
    _assert_exact(mg, vt, grid)
    assert [ring["point_id"] for ring in mg.barrier_report["rings"]] == ["well"]

    mg, vt, grid = _mesh(_model(lines, points=[WELL], keep_close_rings=True))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["rings"] == []


def test_barrier_report_lists_every_entry():
    mg, _, _ = _mesh(_model(EXACT_CASES["t_junction"]))
    assert set(mg.barrier_report) == {"corridors", "collapsed", "tips", "stubs", "snaps", "noise",
                                      "on_boundary", "rings", "fixed_nodes", "repair_rounds", "sites",
                                      "removed_nodes", "polish"}
    assert mg.barrier_report["sites"] > 0


def _broken_site(mg: MeshGenerator) -> None:
    """Move the barrier site nearest (150, 152) across the y = 150 barrier, next to its partner."""
    sites = np.flatnonzero(mg.node_is_site)
    k = sites[np.argmin(np.hypot(*(mg.nodes[sites] - [150.0, 152.0]).T))]
    mg.nodes[k] = [mg.nodes[k, 0] + 0.5, 300.0 - mg.nodes[k, 1]]


def test_safety_net_warns_or_raises_when_a_barrier_face_is_not_exact(monkeypatch):
    cm = _model([([(40, 150), (260, 150)], B)])
    mg = MeshGenerator(background_lc=BACKGROUND_LC, verbosity=0)
    assert mg.generate(*cm.generate())
    _broken_site(mg)
    with pytest.raises(vorflow.BarrierMeshError, match="safety net"):
        VoronoiTessellator(mg, cm).generate()

    monkeypatch.setattr(vorflow.tessellator, "_STRICT_BARRIERS", False)
    vt = VoronoiTessellator(mg, cm)
    with pytest.warns(UserWarning, match="safety net"):
        vt.generate()
    assert vt.n_exact_barrier_mirrors + vt.n_exact_barrier_cuts > 0


# ---------------------------------------------------------------------------
# Regression set: the held-out and stress shapes of the barrier benchmark
# ---------------------------------------------------------------------------
def _zone(resolution: float, bounds: tuple) -> dict:
    """A refinement polygon over the domain polygon."""
    return {"geometry": box(*bounds), "zone_id": 2, "resolution": resolution, "z_order": 1}


def _digitised(control: list, rng, spacing: tuple, wobble: float) -> list:
    """Vertices at irregular spacing along ``control``, jittered sideways; the ends stay exact."""
    guide = LineString(control)
    distances = [0.0]
    while distances[-1] < guide.length:
        distances.append(distances[-1] + rng.uniform(*spacing))
    out = [tuple(control[0])]
    for d in distances[1:-1]:
        p = np.asarray(guide.interpolate(d).coords[0])
        q = np.asarray(guide.interpolate(min(d + 1.0, guide.length)).coords[0])
        normal = np.array([p[1] - q[1], q[0] - p[0]]) / max(np.hypot(*(q - p)), 1e-12)
        out.append(tuple(p + rng.uniform(-wobble, wobble) * normal))
    return out + [tuple(control[-1])]


def _relay_network() -> tuple:
    """Two overlapping digitised faults 0.8 lc apart and a splay fanning off at 10 degrees (lc 30, tolerance lc/8)."""
    rng = np.random.default_rng(15)
    size, lc = 3000.0, 30.0
    main = _digitised([(300, 1500), (1050, 1503), (1800, 1500)], rng, (0.3 * lc, 1.5 * lc), 0.1 * lc)
    echelon = _digitised([(1350, 1524), (2100, 1527), (2760, 1530)], rng, (0.3 * lc, 1.5 * lc), 0.1 * lc)
    start = LineString(main).interpolate(LineString(main).project(Point(900, 1500))).coords[0]
    splay = _digitised([start, _ray(start, -10.0, 600.0)], rng, (0.3 * lc, 1.5 * lc), 0.1 * lc)
    # At tolerance 0 the main fault's vertex 0.19 lc past the splay junction pinches the 10 degree wedge.
    lines = [(trace, {**B, "resolution": lc, "simplify_tolerance": lc / 8}) for trace in (main, echelon, splay)]
    return lines, box(0, 0, size, size), 4.0 * lc


def _hairpin(gap: float) -> list:
    """A U-turn whose two legs are ``gap`` apart."""
    half = gap / 2.0
    return [([(40, 150 - half), (220, 150 - half), (220, 150 + half), (40, 150 + half)], B)]


REGRESSION_CASES = {
    "spike10": dict(lines=[(_spike(10.0), B)]),
    "spike15": dict(lines=[(_spike(15.0), B)]),
    "spike20": dict(lines=[(_spike(20.0), B)]),
    "x20": dict(lines=[([(40, 150), (260, 150)], B), ([_ray(CENTRE, 200, 110), _ray(CENTRE, 20, 110)], B)]),
    "x70": dict(lines=[([(40, 150), (260, 150)], B), ([_ray(CENTRE, 250, 110), _ray(CENTRE, 70, 110)], B)]),
    "y_asym": dict(lines=_star([30.0, 150.0, 180.0])),
    "t30": dict(lines=[([(40, 150), (260, 150)], B), ([CENTRE, _ray(CENTRE, 30, 110)], B)]),
    "star5": dict(lines=_star([0, 60, 130, 200, 280])),
    "end_on_kink": dict(lines=[(K67, B), ([(110, 280), K67[1]], B)]),
    "u_turn": dict(lines=_hairpin(1.2 * LC)),
    "close_parallel": dict(lines=[([(40, 150), (260, 150)], B), ([(40, 154), (260, 154)], B)]),
    "along_boundary": dict(lines=[([(40, 0.4 * LC), (260, 0.4 * LC)], B)]),
    "along_boundary_taper": dict(lines=[([(40, 0.5 * LC), (260, 0.1 * LC)], B)]),
    "corner_t": dict(lines=[([(0, 250), (50, 300)], B), ([(25, 275), (150, 190)], B)]),
    "std_cross10": dict(lines=[([(40, 150), (260, 150)], B), ([_ray(CENTRE, 190, 110), _ray(CENTRE, 10, 110)], {})]),
    "zone_crossing": dict(lines=[([(40, 70), (260, 230)], B)], polygons=[_zone(5.0, (110, 110, 210, 210))]),
    "well_close": dict(lines=[([(40, 145), (260, 145)], B)], points=[WELL], keep_close_rings=True),
    "short_barrier": dict(lines=[([_ray(CENTRE, 200, 0.35 * LC), _ray(CENTRE, 20, 0.35 * LC)], B)]),
    "short_leg": dict(lines=[([K67[0], K67[1], _ray(K67[1], -45, 0.2 * LC), K67[3]], B)]),
    "noise_vertex": dict(lines=[([(40, 100), CENTRE, _ray(CENTRE, 60, 0.05 * LC), (260, 210)], B)]),
}


@pytest.mark.parametrize("name", REGRESSION_CASES)
def test_benchmark_shapes_mesh_exactly(name):
    case = dict(REGRESSION_CASES[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # noise cleaning warns
        mg, vt, grid = _mesh(_model(case.pop("lines"), **case))
    _assert_exact(mg, vt, grid)


def test_relay_ramp_network_meshes_exactly():
    lines, domain, background_lc = _relay_network()
    mg, vt, grid = _mesh(_model(lines, domain=domain, background_lc=background_lc), background_lc=background_lc)
    _assert_exact(mg, vt, grid)
    assert any(c["kind"] == "barriers" for c in mg.barrier_report["corridors"])


# ---------------------------------------------------------------------------
# Deviation, scale and the tessellator's own generator moves
# ---------------------------------------------------------------------------
def test_faces_stay_within_simplify_tolerance_of_the_input_line():
    rng = np.random.default_rng(7)
    raw = _digitised([(30, 60), (150, 180), (270, 230)], rng, (0.3 * LC, 1.5 * LC), 0.15 * LC)
    tolerance = 1.5
    mg, vt, grid = _mesh(_model([(raw, {**B, "simplify_tolerance": tolerance})]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["noise"] == []
    on, _ = _barrier_faces(grid, mg, TOUCH * 300.0)
    faces = shapely.union_all(on.geometry.to_numpy())
    assert shapely.hausdorff_distance(LineString(raw), faces, densify=0.05) <= tolerance
    assert len(mg.barrier_lines.geometry.iloc[0].coords) < len(raw)  # the tolerance did simplify it


@pytest.mark.parametrize("factor", [0.01, 100.0])
def test_scaled_model_has_the_same_barrier_topology(factor):
    lines = EXACT_CASES["kink"] + [([(110, 280), K67[1]], B)]

    def scaled(k):
        coords = [([(x * k, y * k) for x, y in c], {**kw, "resolution": LC * k}) for c, kw in lines]
        cm = _model(coords, domain=affinity.scale(DOMAIN, k, k, origin=(0, 0)), background_lc=BACKGROUND_LC * k)
        return _mesh(cm, background_lc=BACKGROUND_LC * k)

    base, _, base_grid = scaled(1.0)
    mg, vt, grid = scaled(factor)
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["sites"] == base.barrier_report["sites"]
    assert len(mg.barrier_circles[0]) == len(base.barrier_circles[0])
    sites = np.sort(np.round(mg.nodes[mg.node_is_site] / factor, 6).view("f8,f8"), axis=0)
    base_sites = np.sort(np.round(base.nodes[base.node_is_site], 6).view("f8,f8"), axis=0)
    assert np.allclose(sites.view("f8").reshape(-1, 2), base_sites.view("f8").reshape(-1, 2), atol=1e-5)
    on, _ = _barrier_faces(grid, mg, TOUCH * 300.0 * factor)
    base_on, _ = _barrier_faces(base_grid, base, TOUCH * 300.0)
    assert len(on) == len(base_on)


def test_tessellator_lloyd_short_faces_and_inset_mirror_keep_the_circles_empty():
    # One barrier ends on the boundary, where inset_mirror shifts generators and adds ghosts.
    cm = _model([([(150, 150), (300, 200)], B), ([(0, 100), (120, 140)], B)])
    mg = MeshGenerator(background_lc=BACKGROUND_LC, verbosity=0)
    assert mg.generate(*cm.generate())
    vt = VoronoiTessellator(mg, cm, lloyd_iterations=30, min_face_fraction=0.1, boundary_centering="inset_mirror")
    grid = vt.generate()
    assert vt.lloyd_report["iterations"] > 0 and vt.short_face_report["initial"] > 0
    assert (grid["boundary_centering"] == "inset_mirror").any()
    _assert_exact(mg, vt, grid)


# ---------------------------------------------------------------------------
# Fixed lines, points and the domain boundary next to a barrier
# ---------------------------------------------------------------------------
def _nodes_on(mg: MeshGenerator, line, x_range=(70.0, 230.0)) -> np.ndarray:
    """Mesh nodes lying on ``line`` within ``x_range``."""
    nodes = mg.nodes[(mg.nodes[:, 0] > x_range[0]) & (mg.nodes[:, 0] < x_range[1])]
    return nodes[shapely.distance(line, shapely.points(nodes)) < 1e-6]


@pytest.mark.parametrize("gap", [0.25 * LC, 0.5 * LC])
def test_standard_line_next_to_a_barrier_keeps_its_nodes(gap):
    river = LineString([(40, 150 + gap), (260, 150 + gap)])
    mg, vt, grid = _mesh(_model([([(40, 150), (260, 150)], B), (list(river.coords), {})]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["fixed_nodes"] == []
    xs = np.sort(_nodes_on(mg, river)[:, 0])
    assert len(xs) > 0 and np.diff(np.r_[70.0, xs, 230.0]).max() <= LC


@pytest.mark.parametrize("gap", [0.25 * LC, 0.5 * LC])
def test_zone_edge_next_to_a_barrier_keeps_its_nodes(gap):
    zone = box(60, 150 + gap, 240, 250)
    mg, vt, grid = _mesh(_model([([(40, 150), (260, 150)], B)], polygons=[_zone(LC, zone.bounds)]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["fixed_nodes"] == []
    edge = LineString([(60, 150 + gap), (240, 150 + gap)])
    xs = np.sort(_nodes_on(mg, edge)[:, 0])
    assert len(xs) > 0 and np.diff(np.r_[70.0, xs, 230.0]).max() <= LC


def test_quad_buffer_strip_next_to_a_barrier_keeps_its_nodes():
    lines = [([(40, 150), (260, 150)], B), ([(40, 156), (260, 156)], {"quad_buffer": True, "resolution": 6.0})]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the strip is trimmed next to the barrier
        mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["fixed_nodes"] == []
    footprint = mg.buffer_footprints
    lower = LineString([(40, footprint.bounds[1]), (260, footprint.bounds[1])])
    assert len(_nodes_on(mg, lower)) >= 20


def test_standard_line_crossing_a_barrier_is_not_reported():
    mg, vt, grid = _mesh(_model(REGRESSION_CASES["std_cross10"]["lines"]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["fixed_nodes"] == []


def test_barrier_ending_just_short_of_a_standard_line_meshes():
    # The end is 0.1 m (0.0125 lc) from the line: as close as a crossing, so it is treated as one.
    lines = [([(40, 150), (149.9, 150)], B), ([(150, 40), (150, 260)], {})]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["fixed_nodes"] == []


def test_barrier_along_the_boundary_is_dropped_with_a_warning():
    cm = _model([([(40, 0), (260, 0)], B)])
    with pytest.warns(UserWarning, match="domain boundary"):
        mg, vt, grid = _mesh(cm)
    assert [e["line_id"] for e in mg.barrier_report["on_boundary"]] == ["line0"]
    assert mg.barrier_report["on_boundary"][0]["distance"] == pytest.approx(220.0)
    assert mg.barrier_circles is None and vt.n_barrier_mirrors == 0 and len(grid) == len(mg.nodes)


def test_barrier_partly_on_the_boundary_keeps_the_part_inside():
    cm = _model([([(40, 100), (150, 0), (260, 0)], B)])
    with pytest.warns(UserWarning, match="domain boundary"):
        mg, vt, grid = _mesh(cm)
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["on_boundary"][0]["distance"] == pytest.approx(110.0)
    assert list(mg.barrier_lines.geometry.iloc[0].coords) == [(40.0, 100.0), (150.0, 0.0)]


@pytest.mark.parametrize("length, tolerance", [(4.0, 5.0), (3.0, 3.5)])
def test_branch_within_its_tolerance_is_dropped_as_a_stub(length, tolerance):
    lines = [([(40, 150), (260, 150)], B), ([(150, 150), (150, 150 + length)], {**B, "simplify_tolerance": tolerance})]
    with pytest.warns(UserWarning, match="stub"):
        mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    stubs = mg.barrier_report["stubs"]
    assert len(stubs) == 1 and stubs[0]["line_id"] == "line1"
    assert stubs[0]["length"] == pytest.approx(length)


@pytest.mark.parametrize("x, y", [(150, 150.0), (150, 150.05), (260, 150.0)], ids=["on", "next to", "at the end"])
def test_point_on_a_barrier_raises_a_clear_error(x, y):
    well = {"geometry": Point(x, y), "point_id": "w", "resolution": 4.0}
    with pytest.raises(vorflow.BarrierMeshError,
                       match="point 'w' .* from barrier 'line0'.*Move it off the barrier.*end the barrier"):
        _mesh(_model([([(40, 150), (260, 150)], B)], points=[well]))


def test_size_field_without_python_size_is_warned_about():
    class OpaqueField(vorflow.fields.ThresholdField):
        """A ThresholdField the barrier sizing cannot read."""

        def python_size(self, distance, background_lc, feature_lc=None):
            return None

    river = ([(40, 170), (260, 170)], {"fields": [OpaqueField(size_min=4.0, dist_min=2.0, dist_max=20.0)]})
    with pytest.warns(UserWarning, match="python_size.*OpaqueField"):
        mg, vt, grid = _mesh(_model([([(40, 150), (260, 150)], B), river]))
    _assert_exact(mg, vt, grid)


def test_warnings_point_at_the_caller_through_mesh_generator():
    cm = _model([(_spike(12.0), B), ([(40, 30), CENTRE, _ray(CENTRE, 60, 0.05 * LC), (260, 60)], B)])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        MeshGenerator(background_lc=BACKGROUND_LC, verbosity=0, conceptual_mesh=cm).generate()
    messages = {str(w.message).split(":")[0]: w.filename for w in record if w.category is UserWarning}
    assert any("digitising noise" in str(w.message) for w in record)
    assert any("spike tip" in str(w.message) for w in record)
    assert set(messages.values()) == {__file__}


# ---------------------------------------------------------------------------
# PR #73 review: crossings close together, sealing at the boundary, parallel
# snapping, straddle_width, barriers on zone edges
# ---------------------------------------------------------------------------
def _sealed(mg: MeshGenerator, grid, a: tuple, b: tuple) -> bool:
    """True when the cells at ``a`` and ``b`` are not connected through any face off the barrier faces."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    tol = TOUCH * max(np.ptp(np.asarray(grid.total_bounds).reshape(2, 2), axis=0))
    on, connectivity = _barrier_faces(grid, mg, tol)
    open_faces = connectivity.drop(on.index)
    i, j = open_faces["cell_id_1"].to_numpy(), open_faces["cell_id_2"].to_numpy()
    _, label = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(grid),) * 2), directed=False)
    cells = grid[["x", "y"]].to_numpy(dtype=float)
    first, second = (label[int(np.argmin(np.hypot(*(cells - p).T)))] for p in (a, b))
    return bool(first != second)


@pytest.mark.parametrize("kind", ["standard", "barrier"])
@pytest.mark.parametrize("gap", [0.5, 2.0, 5.0, 9.0])
def test_two_lines_crossing_a_barrier_close_together_mesh_exactly(gap, kind):
    other = {} if kind == "standard" else B
    lines = [([(40, 150), (260, 150)], B), ([(150 - gap / 2, 100), (150 - gap / 2, 200)], other),
             ([(150 + gap / 2, 100), (150 + gap / 2, 200)], other)]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)


def test_t_junction_next_to_a_crossing_meshes_exactly():
    lines = [([(40, 150), (260, 150)], B), ([(150, 150), (150, 260)], B), ([(153, 100), (153, 200)], {})]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)


@pytest.mark.parametrize("tolerance, last_y", [(0.0, 298.0), (3.0, 298.0), (6.0, 295.0), (1.0, 299.5),
                                               (3.0, 299.9), (3.0, 299.99)])
def test_barrier_crossing_the_boundary_keeps_its_compartment_sealed(tolerance, last_y):
    # A U closing a compartment against the top boundary; each leg's last vertex is just inside it.
    coords = [(100, 320), (100, last_y), (100, 200), (200, 200), (200, last_y), (200, 320)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mg, vt, grid = _mesh(_model([(coords, {**B, "simplify_tolerance": tolerance or None})]))
    _assert_exact(mg, vt, grid)
    ends = [line.coords[k] for line in mg.barrier_lines.geometry for k in (0, -1)]
    assert sorted(ends) == [(100.0, 300.0), (200.0, 300.0)]
    assert _sealed(mg, grid, (150, 250), (150, 100))


@pytest.mark.parametrize("coords", [[(100, 154), (260, 154)], [(100, 154), (180, 154.5), (255, 154)]])
def test_parallel_barrier_within_the_tolerance_is_not_merged_into_its_neighbour(coords):
    lines = [([(40, 150), (260, 150)], B), (coords, {**B, "simplify_tolerance": 5.0})]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    assert [e for e in mg.barrier_report["snaps"] if e["kind"] in ("snap", "trim")] == []
    neighbour = mg.barrier_lines.geometry[mg.barrier_lines["line_id"] == "line1"].iloc[0]
    assert shapely.hausdorff_distance(neighbour, LineString(coords)) < 0.98 * 5.0


def test_straddle_width_value_does_not_change_the_mesh():
    # A non-barrier face line crossed by a quad-buffer line: the value used to set the trimming width.
    nodes = []
    for width in (0.5, 20.0):
        with pytest.warns(FutureWarning, match="straddle_width"):
            cm = _model([([(40, 150), (260, 150)], {"straddle_width": width}),
                         ([(150, 60), (150, 240)], {"quad_buffer": True})])
        mg, _, _ = _mesh(cm)
        nodes.append(mg.nodes)
        assert mg.barrier_lines["is_barrier"].tolist() == [False]
    assert np.array_equal(*nodes)


def test_barrier_on_a_zone_edge_reports_no_fixed_nodes():
    left = {"geometry": box(0, 0, 150, 300), "zone_id": 2, "resolution": BACKGROUND_LC, "z_order": 1}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mg, vt, grid = _mesh(_model([([(150, 40), (150, 260)], B)], polygons=[left]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["fixed_nodes"] == []
    assert not [w for w in record if "fixed" in str(w.message) or "removed" in str(w.message)]


# ---------------------------------------------------------------------------
# Acute wedges: legs meeting below sharp_angle are merged over a short reach
# ---------------------------------------------------------------------------
DEVIATION = 0.1  # x resolution: the largest move of a barrier by a merge


def _wedge_cases() -> dict:
    """(lines, probe points in separate compartments, domain) per acute shape and angle."""
    out = {}
    for angle in (3, 5, 8, 12):
        j = (60.0, 150.0)
        splay_end = (300.0, 150.0 + 240.0 * math.tan(math.radians(angle)))
        out[f"splay{angle}"] = ([([(0, 150), j, (300, 150)], B), ([j, splay_end], B)],
                                [(150, 100), (150, 200), _ray(j, angle / 2, 225)], DOMAIN)
        h = angle / 2
        out[f"x{angle}"] = ([([_ray(CENTRE, 180 - h, 300), _ray(CENTRE, -h, 300)], B),
                             ([_ray(CENTRE, 180 + h, 300), _ray(CENTRE, h, 300)], B)],
                            [(150, 220), (150, 80), _ray(CENTRE, 0, 140), _ray(CENTRE, 180, 140)], DOMAIN)
        t = (100.0, 150.0)
        out[f"t{angle}"] = ([([(0, 150), (300, 150)], B), ([t, _ray(t, angle, 400)], B)],
                            [(150, 100), (60, 200), _ray(t, angle / 2, 190)], DOMAIN)
        f = (40.0, 150.0)
        fan = [([(0, 150), (300, 150)], B)] + [([f, _ray(f, angle * k, 400)], B) for k in (1, 2, 3)]
        out[f"fan{angle}"] = (fan, [(150, 100), (40, 250)] + [_ray(f, angle * (k + 0.5), 250) for k in range(3)],
                              DOMAIN)
    for angle in (5, 10):
        s = (40.0, 0.0)
        out[f"boundary{angle}"] = ([([s, _ray(s, angle, 2000)], B)], [(250, 5), (100, 250)], DOMAIN)
        hole = box(120, 120, 180, 180)
        out[f"hole{angle}"] = ([([(180, 150), _ray((180, 150), 90 - angle, 400)], B),
                                ([(120, 150), _ray((120, 150), 270 - angle, 400)], B)],
                               [(260, 60), (40, 240)], DOMAIN.difference(hole))
    return out


WEDGE_CASES = _wedge_cases()


def _moved(mg: MeshGenerator, cm: ConceptualMesh, domain, uncovered: list = ()) -> float:
    """The largest distance between a barrier as cleaned and as its faces follow it (both ways; the input
    may also be matched by the domain boundary it was merged onto, and need not be within 2 x the reported
    length of an ``uncovered`` tip or stub)."""
    cleaned = cm.clean_lines.set_index("line_id").geometry
    left = shapely.union_all([Point(e["location"]).buffer(2 * e["length"]) for e in uncovered]) if uncovered \
        else None
    out = 0.0
    for line_id, line in zip(mg.barrier_lines["line_id"], mg.barrier_lines.geometry):
        drawn = cleaned[line_id]
        drawn = drawn if isinstance(drawn, LineString) else shapely.union_all(drawn.to_numpy())
        new_points = shapely.points(shapely.get_coordinates(shapely.segmentize(line, 0.05)))
        old_points = shapely.points(shapely.get_coordinates(shapely.segmentize(drawn, 0.05)))
        if left is not None:
            old_points = old_points[~shapely.covers(left, old_points)]
        out = max(out, float(shapely.distance(drawn, new_points).max()),
                  float(shapely.distance(shapely.union(line, domain.boundary), old_points).max(initial=0.0)))
    return out


@pytest.mark.parametrize("name", WEDGE_CASES)
def test_acute_wedges_are_merged_over_a_short_reach(name):
    lines, probes, domain = WEDGE_CASES[name]
    cm = _model(lines, domain=domain)
    with pytest.warns(UserWarning, match="were merged over a short reach"):
        mg, vt, grid = _mesh(cm)
    _assert_exact(mg, vt, grid)
    report = mg.barrier_report
    assert report["collapsed"] and report["tips"] == [] and report["stubs"] == []
    assert all(e["deviation"] <= DEVIATION * LC * (1 + 1e-6) for e in report["collapsed"])
    assert {i for e in report["collapsed"] for i in e["line_ids"]} <= set(mg.barrier_lines["line_id"])
    assert _moved(mg, cm, domain) <= DEVIATION * LC * (1 + 1e-3)
    for a, b in zip(probes[:-1], probes[1:]):
        assert _sealed(mg, grid, a, b)


def test_self_crossing_barrier_is_noded_where_it_crosses_itself():
    coords = [(40, 100), (260, 200), (260, 100), (40, 200)]
    mg, vt, grid = _mesh(_model([(coords, B)]))
    _assert_exact(mg, vt, grid)
    line = mg.barrier_lines.geometry.iloc[0]
    assert (150.0, 150.0) in line.coords  # the crossing is a vertex, so its faces meet there
    for outside in [(150, 120), (150, 180), (60, 150)]:  # the loop closed by the crossing is a compartment
        assert _sealed(mg, grid, (250, 150), outside)


def test_polish_moves_only_the_nodes_near_the_barrier():
    from vorflow._barrier import frames

    cm = _model([([(100, 150), (200, 150)], B)], domain=box(0, 0, 600, 300))
    fixed, _, _ = _mesh(cm, barrier_polish_iterations=0)
    polished, vt, grid = _mesh(cm)
    _assert_exact(polished, vt, grid)
    assert len(fixed.nodes) == len(polished.nodes)
    moved = (fixed.nodes != polished.nodes).any(axis=1)
    gap = frames.circle_gap(fixed.nodes, *fixed.barrier_circles)
    assert 0 < moved.sum() <= polished.barrier_report["polish"]["moved"]
    assert (gap[moved] < frames.POLISH_BAND * fixed.node_sizes[moved]).all()
    far = gap > frames.POLISH_BAND * fixed.node_sizes
    assert far.sum() > len(fixed.nodes) / 2
    assert np.array_equal(fixed.nodes[far], polished.nodes[far])  # bit-identical to Gmsh's


# ---------------------------------------------------------------------------
# Narrow gaps keep one row of cells, so flow along the gap stays open
# ---------------------------------------------------------------------------
ASPECT = 3.0  # corridor.CORRIDOR_ASPECT: a corridor's cells are at most this many times longer than the gap


def _parallel(gap: float, x0: float = 40.0, x1: float = 260.0) -> list:
    """Two barriers ``gap`` apart, from x0 to x1."""
    return [([(x0, 150), (x1, 150)], B), ([(x0, 150 + gap), (x1, 150 + gap)], B)]


def _components(mg: MeshGenerator, grid) -> int:
    """Number of groups of cells connected across faces that are not barrier faces."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    tol = TOUCH * max(np.ptp(np.asarray(grid.total_bounds).reshape(2, 2), axis=0))
    on, connectivity = _barrier_faces(grid, mg, tol)
    open_faces = connectivity.drop(on.index)
    i, j = open_faces["cell_id_1"].to_numpy(), open_faces["cell_id_2"].to_numpy()
    count, _ = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(grid),) * 2), directed=False)
    return int(count)


@pytest.mark.parametrize("gap", [0.4, 0.8, 2.0, 4.0])
def test_close_parallel_barriers_keep_one_row_of_cells(gap):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mg, vt, grid = _mesh(_model(_parallel(gap)))
    _assert_exact(mg, vt, grid)
    [corridor] = mg.barrier_report["corridors"]
    assert corridor["kind"] == "barriers" and corridor["gap"] == pytest.approx(gap)
    assert corridor["ends"] == ("tip", "tip") and corridor["resolution"] == LC
    # One row: as many cells as the spacing min(lc, ASPECT x gap) gives, each with its site on the midline.
    spacing = min(LC, ASPECT * gap)
    assert abs(corridor["cells"] - corridor["length"] / spacing) <= 2
    between = (grid.x > 60) & (grid.x < 240) & (grid.y > 150) & (grid.y < 150 + gap)
    assert between.sum() == pytest.approx(180 / spacing, abs=2)
    assert np.allclose(grid.y[between], 150 + gap / 2, atol=1e-6)
    widths = grid.geometry[between].bounds
    assert ((widths.maxx - widths.minx) <= ASPECT * gap * (1 + 1e-6)).all()
    assert _components(mg, grid) == 1  # the gap is open along its length and at both ends
    dense = [w for w in record if "keep one row of cells" in str(w.message)]
    assert len(dense) == int(spacing < 0.5 * LC)
    if dense:
        assert f"{corridor['cells']} cells" in str(dense[0].message)


@pytest.mark.parametrize("distance", [0.8, 2.0])
def test_barrier_close_to_the_boundary_keeps_cells_at_most_three_to_one(distance):
    # The boundary cuts the row's cells to the distance, so that is the gap their length is held to.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mg, vt, grid = _mesh(_model([([(40, distance), (260, distance)], B)]))
    _assert_exact(mg, vt, grid)
    [corridor] = mg.barrier_report["corridors"]
    assert corridor["kind"] == "boundary" and corridor["gap"] == pytest.approx(distance)
    between = (grid.x > 60) & (grid.x < 240) & (grid.y < distance)
    widths = grid.geometry[between].bounds
    assert between.sum() == pytest.approx(180 / min(LC, ASPECT * distance), abs=2)
    assert ((widths.maxx - widths.minx) <= ASPECT * distance * (1 + 1e-6)).all()


def test_corridor_cells_grow_smoothly_as_the_gap_narrows():
    # No cliff anywhere: from 0.05 to 1.3 lc the row has length / min(lc, ASPECT x gap) cells, the total
    # never drops as the gap narrows and rises by at most the step's share of 1 / gap. The one step up with
    # a wider gap (11%) is at 1.2 lc, where the corridor gives way to a row of faces on each barrier.
    fractions = np.round(np.arange(0.05, 1.31, 0.1), 3)
    cells = []
    for fraction in fractions:
        gap = fraction * LC
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mg, vt, grid = _mesh(_model(_parallel(gap)))
        _assert_exact(mg, vt, grid)
        assert _components(mg, grid) == 1
        corridors = mg.barrier_report["corridors"]
        assert len(corridors) == int(fraction <= 1.2)
        if corridors:
            assert abs(corridors[0]["cells"] - corridors[0]["length"] / min(LC, ASPECT * gap)) <= 2
        cells.append(len(grid))
    cells = np.array(cells, dtype=float)
    density = 1.0 / np.minimum(fractions, 1.0 / ASPECT)
    assert (cells[1:] <= 1.15 * cells[:-1]).all()
    assert (cells[:-1] / cells[1:] <= density[:-1] / density[1:] + 0.15).all()


def test_relay_ramp_to_the_boundary_stays_connected():
    # Two overlapping faults 0.2 lc apart, each ending on the boundary: the ramp between them joins north and
    # south (a single line of faces there would seal them apart).
    gap = 0.2 * LC
    lines = [([(0, 150), (200, 150)], B), ([(120, 150 + gap), (300, 150 + gap)], B)]
    mg, vt, grid = _mesh(_model(lines))
    _assert_exact(mg, vt, grid)
    assert not _sealed(mg, grid, (60, 200), (240, 100))
    assert [c["ends"] for c in mg.barrier_report["corridors"]] == [("stagger", "stagger")]


@pytest.mark.parametrize("name, coords", [
    ("double dip", [(40, 156), (90, 151.6), (130, 151.6), (150, 154), (170, 151.6), (210, 151.6), (260, 156)]),
    ("wavy", [(x, 150 + LC * (0.25 + 0.06 * math.sin(x / 9))) for x in np.linspace(40, 260, 111)]),
], ids=["double_dip", "wavy"])
def test_a_gap_that_narrows_and_widens_leaves_no_pocket(name, coords):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mg, vt, grid = _mesh(_model([([(40, 150), (260, 150)], B), (coords, B)]))
    _assert_exact(mg, vt, grid)
    assert _components(mg, grid) == 1
    [corridor] = mg.barrier_report["corridors"]  # one row along the whole bending strand
    assert corridor["length"] == pytest.approx(220.0, abs=1.0)


@pytest.mark.parametrize("y", [150.8, 151.2])
def test_well_between_two_close_barriers_meshes(y):
    well = {"geometry": Point(150, y), "point_id": "w", "resolution": 2.0}
    mg, vt, grid = _mesh(_model(_parallel(1.6), points=[well]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["corridors"] == []  # the barriers keep their own faces, capped by the well
    assert np.hypot(*(mg.nodes - [150, y]).T).min() < 1e-9  # the well is a cell centre, where it was put
    assert _components(mg, grid) == 1


def test_zone_edge_between_two_close_barriers_keeps_its_nodes():
    zone = {"geometry": box(0, 0, 300, 150.8), "zone_id": 2, "resolution": BACKGROUND_LC, "z_order": 1}
    mg, vt, grid = _mesh(_model(_parallel(1.6), polygons=[zone]))
    _assert_exact(mg, vt, grid)
    assert mg.barrier_report["corridors"] == [] and mg.barrier_report["fixed_nodes"] == []


def _hairpin_shapes(gap: float) -> dict:
    """Joined near-parallel legs ``gap`` apart: a U-turn, a thin closed loop and a closed lens with tapered ends."""
    half, taper = gap / 2.0, gap / math.tan(math.radians(40.0))
    return {
        "u_turn": [([(40, 150 - half), (260, 150 - half), (260, 150 + half), (40, 150 + half)], B)],
        "thin_loop": [([(40, 150 - half), (260, 150 - half), (260, 150 + half), (40, 150 + half),
                        (40, 150 - half)], B)],
        "lens": [([(60, 150), (60 + taper, 150 + half), (240 - taper, 150 + half), (240, 150),
                   (240 - taper, 150 - half), (60 + taper, 150 - half), (60, 150)], B)],
    }


@pytest.mark.parametrize("gap", [0.1 * LC, 0.25 * LC])
@pytest.mark.parametrize("name", ["u_turn", "thin_loop", "lens"])
def test_joined_close_legs_keep_one_row_of_cells(name, gap):
    lines = _hairpin_shapes(gap)[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mg, vt, grid = _mesh(_model(lines))
        _, _, plain = _mesh(_model([(coords, {}) for coords, _ in lines]))  # the legs as standard lines
    _assert_exact(mg, vt, grid)
    [corridor] = mg.barrier_report["corridors"]
    assert "bend" in corridor["ends"]
    # Cost bound: each row cell beyond the length / lc the resolution gives brings its two mirrored outer
    # cells and the grading from the row's spacing back to lc on both sides, about 2 / 0.3 (Gmsh's growth)
    # more: about 10 cells; 14 leaves 40% for Gmsh, and 1.2 x the legs meshed as standard lines covers the
    # turns. Measured at 0.1 lc: 1298 / 1360 / 1061 cells against bounds of 1591 / 1513 / 1300 (develop, which
    # cuts cells instead: 673 / 680 / 622; this branch with two rows of capped faces: 2649 / 2649 / 2329).
    extra = max(0.0, corridor["cells"] - corridor["length"] / LC)
    assert len(grid) <= 1.2 * len(plain) + 14 * extra


def test_splay_crossing_a_bent_fault_just_past_the_junction_is_merged_onto_one_line():
    # The fault bends 0.8 m towards the splay 1.5 m past the junction, so the splay crosses it again at 4.5 m.
    junction = (100.0, 150.0)
    fault = [(20, 150), junction, (101.52, 150.8), (280, 150)]
    splay = [junction, _ray(junction, 10.0, 170.0)]
    cm = _model([(fault, B), (splay, B)])
    with pytest.warns(UserWarning, match="were merged over a short reach"):
        mg, vt, grid = _mesh(cm)
    _assert_exact(mg, vt, grid)
    assert len(mg.barrier_report["collapsed"]) == 2  # the thin loop, then the 10 degree wedge past it
    assert _moved(mg, cm, DOMAIN) <= DEVIATION * LC * (1 + 1e-3)
