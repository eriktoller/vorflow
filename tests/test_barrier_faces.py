"""Exact barrier faces (``vorflow._barrier``): every barrier segment is one Voronoi cell face.

Each model is meshed end to end and checked against the properties of the
construction (``_assert_exact``): no mirror generators and no cut cells, one
cell per mesh node, the barrier sites among the mesh nodes, no other node
inside an empty circle, and the barrier lines covered by single-edge,
orthogonal faces. The focused tests check one rule each (snapping, noise,
stubs, spike tips, corridors, options, errors); ``REGRESSION_CASES`` are the
held-out and stress shapes of the barrier benchmark.
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


def _assert_exact(mg: MeshGenerator, vt: VoronoiTessellator, grid) -> None:
    """The properties every exact barrier construction must have (module docstring)."""
    assert vt.n_barrier_mirrors == 0
    assert vt.n_exact_barrier_mirrors == 0 and vt.n_exact_barrier_cuts == 0
    assert len(grid) == len(mg.nodes)
    assert int(mg.node_is_site.sum()) == mg.barrier_report["sites"]
    centres, radii = mg.barrier_circles
    others = mg.nodes[~mg.node_is_site]
    nearest = cKDTree(others).query(centres)[0]
    assert (nearest >= radii * (1 - 1e-9)).all()

    tol = TOUCH * max(np.ptp(np.asarray(grid.total_bounds).reshape(2, 2), axis=0))
    on, connectivity = _barrier_faces(grid, mg, tol)
    assert _crossing_faces(connectivity, _barriers(mg), tol) == 0
    assert all(len(face.coords) == 2 for face in on.geometry)
    assert on["ortho_error"].max() < 1e-6
    # Each pair of cells meets along at most one barrier face, and the faces cover the barriers.
    assert not on.duplicated(["cell_id_1", "cell_id_2"]).any()
    assert on.geometry.length.sum() == pytest.approx(_barriers(mg).length, rel=1e-6)


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
    assert set(mg.barrier_report) == {"corridors", "tips", "stubs", "snaps", "noise", "rings", "repair_rounds",
                                      "sites", "removed_nodes", "polish"}
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

    monkeypatch.setattr(vorflow.tessellator, "STRICT_BARRIERS", False)
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
