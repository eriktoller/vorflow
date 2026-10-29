import logging

import geopandas as gpd
import numpy as np
import pytest
import shapely
import shapely.affinity
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

import vorflow
import vorflow._convex as convex
import vorflow.tessellator
from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow._convex import (
    SplitError,
    centroid_fan_fraction,
    convex_pieces,
    reflex_vertices,
    split_concave_cells,
)
from vorflow.utils import calculate_mesh_quality, grid_to_disv, summarize_quality


class DummyMeshGenerator:
    def __init__(self, nodes, tags, zones_gdf):
        self.nodes = nodes
        self.node_tags = tags
        self.zones_gdf = zones_gdf


# Reflex corner at (1, 1); every face is visible from the centroid (5/6, 5/6),
# the smallest fan triangle being 1/36 of the area.
L_SHAPE = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
# Thin L: the centroid lies outside, so faces are hidden from it.
THIN_L = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])
# A slit from the top down to (5, 1): no diagonal from its tip resolves it.
SLIT = Polygon([(0, 0), (10, 0), (10, 10), (5.1, 10), (5, 1), (4.9, 10), (0, 10)])
# L shape with a hole on the reflex corner's bisector.
HOLED_L = Polygon(
    [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)],
    [[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]],
)
# A zero-width spike, which made the bisector of a reflex vertex meet the
# boundary only along an edge (found by fuzzing).
SPIKE = Polygon([
    (0.3821345956502424, 0.1182080826645358), (0.0636634815009326, 0.8977454879436491),
    (-0.4708008938042767, 0.6467971230556722), (-0.5296510055298113, 0.7276467634376311),
    (-0.2942505586276729, 0.404248201909795), (-0.6536436208636119, -0.7568024953079283),
    (0.141733954858252, -0.1411080651140784), (0.3102263514040999, -0.2525066551489287),
    (0.4173563924195799, -0.2753427712988188), (0.2989626291069652, -0.0249268208452489),
])
# A barrier end left this cell: (3.1, 2.9) lies on the edge from the first
# vertex to the last, so the ring runs out to the first vertex and straight back.
NEEDLE = Polygon([
    (3.084569574545748, 2.886039138874723), (3.1, 2.9),
    (3.052984262536075, 2.9197960999848283), (3.0166893065287557, 3.9203211970614844),
    (3.4325303010616546, 3.9254651571092167), (3.7710503730426863, 3.5071408137052877),
])
# Three teeth: diagonals alone make it convex.
COMB = Polygon([
    (0, 0), (6, 0), (6, 3), (5, 3), (5, 1), (4, 1), (4, 3), (3, 3),
    (3, 1), (2, 1), (2, 3), (1, 3), (1, 1), (0, 1),
])
UTM_OFFSET = (5e5, 5e6)


def _is_convex(poly):
    return not reflex_vertices(poly)


def test_centroid_fan_fraction_of_square_is_a_quarter():
    """Catches a wrong normalisation of the fan triangles."""
    assert centroid_fan_fraction(np.array([box(0, 0, 2, 2)]))[0] == pytest.approx(0.25)


def test_centroid_fan_fraction_ignores_ring_orientation():
    """Catches a sign error on clockwise rings (vorflow cells are clockwise for MF6)."""
    ccw = shapely.orient_polygons(L_SHAPE)
    cw = shapely.orient_polygons(L_SHAPE, exterior_cw=True)
    assert centroid_fan_fraction(np.array([ccw, cw])) == pytest.approx([1 / 36] * 2)


def test_centroid_fan_fraction_is_negative_when_a_face_is_hidden():
    """Catches a fan test that only looks at convexity."""
    fan = centroid_fan_fraction(np.array([L_SHAPE, THIN_L, None]))
    assert fan[0] > 0
    assert fan[1] < 0
    assert np.isnan(fan[2])


def test_centroid_fan_fraction_is_nan_where_undefined():
    """Catches empty polygons shifting centroids onto the wrong cells, and inf/garbage for degenerate cells."""
    holed = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], [[(1, 1), (3, 1), (3, 3), (1, 3)]])
    flat = Polygon([(0, 0), (1, 0), (2, 0)])
    multi = shapely.MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
    geoms = np.array([box(0, 0, 2, 2), Polygon(), L_SHAPE, holed, flat, multi], dtype=object)
    fan = centroid_fan_fraction(geoms)
    assert fan[[0, 2]] == pytest.approx([0.25, 1 / 36])
    assert np.isnan(fan[[1, 3, 4, 5]]).all()


def test_repeated_vertex_neither_hides_a_face_nor_a_reflex_corner():
    """Catches a zero-length edge giving a zero fan triangle or masking a reflex vertex."""
    square = Polygon([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1)])
    l_shape = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 1), (1, 2), (0, 2)])
    assert centroid_fan_fraction(np.array([square]))[0] == pytest.approx(0.25)
    assert len(reflex_vertices(l_shape)) == 1
    pieces, _ = convex_pieces(l_shape)
    assert len(pieces) == 2 and all(_is_convex(p) for p in pieces)


def test_convex_pieces_raise_on_a_hole_instead_of_looping():
    """Catches a cut that returns a holed cell unchanged and loops forever."""
    with pytest.raises(SplitError):
        convex_pieces(HOLED_L)


@pytest.mark.parametrize("poly", [SPIKE, NEEDLE], ids=["spike", "barrier-needle"])
def test_convex_pieces_drop_zero_width_spikes(poly):
    """Catches a spike tip blocking every cut, leaving the cell unsplit."""
    pieces, added = convex_pieces(poly)
    assert all(_is_convex(p) and p.is_valid for p in pieces)
    assert sum(p.area for p in pieces) == pytest.approx(poly.area, rel=1e-9)
    assert added


def test_needle_tip_is_dropped_and_its_base_kept():
    """Catches the spike tip staying in the cell or its base not reaching the neighbour."""
    pieces, added = convex_pieces(NEEDLE)
    tip = NEEDLE.exterior.coords[0]
    assert len(pieces) == 1
    assert tip not in pieces[0].exterior.coords
    assert any(np.array_equal(point, (3.1, 2.9)) for point in added)


@pytest.mark.parametrize("scale", [1.0, 0.01], ids=["metre", "centimetre"])
def test_convex_pieces_do_not_depend_on_the_coordinate_offset(scale):
    """Catches tolerances that reject diagonals or cuts at UTM-sized coordinates."""
    for poly in (COMB, SLIT, THIN_L):
        small = shapely.affinity.scale(poly, scale, scale, origin=(0, 0))
        far = shapely.affinity.translate(small, *UTM_OFFSET)
        near_pieces, near_added = convex_pieces(small)
        far_pieces, far_added = convex_pieces(far)
        assert len(far_pieces) == len(near_pieces)
        assert len(far_added) == len(near_added)
        assert all(_is_convex(p) for p in far_pieces)
        assert sum(p.area for p in far_pieces) == pytest.approx(far.area, rel=1e-9)
    assert len(convex_pieces(shapely.affinity.translate(COMB, *UTM_OFFSET))[1]) == 0


def test_small_random_cells_split_the_same_at_utm_coordinates():
    """Catches roundoff at UTM northings failing the area check on centimetre cells."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        angles = np.sort(rng.uniform(0, 2 * np.pi, 9))
        radii = rng.uniform(0.003, 0.01, 9)
        cell = Polygon(np.c_[radii * np.cos(angles), radii * np.sin(angles)])
        if not cell.is_valid:
            continue
        near_pieces, _ = convex_pieces(cell)
        far_pieces, _ = convex_pieces(shapely.affinity.translate(cell, *UTM_OFFSET))
        assert len(far_pieces) == len(near_pieces)


def test_bisector_cut_tries_the_next_reflex_vertex(monkeypatch):
    """Catches one reflex vertex without a bisector cut blocking the whole cell."""
    real = convex._bisector_cut
    tried = []

    def first_fails(poly, r, barriers=None):
        tried.append(r)
        if len(tried) == 1:
            raise SplitError("no cut")
        return real(poly, r, barriers)

    monkeypatch.setattr(convex, "_bisector_cut", first_fails)
    reflex = reflex_vertices(THIN_L) + reflex_vertices(THIN_L)
    pieces, _ = convex._first_bisector_cut(THIN_L, reflex)
    assert tried == reflex[:2]
    assert len(pieces) >= 2


def test_bisector_cut_raises_when_no_reflex_vertex_has_one(monkeypatch):
    """Catches the fallback swallowing the failure of every reflex vertex."""
    def always_fails(poly, r, barriers=None):
        raise SplitError(f"no cut {r}")

    monkeypatch.setattr(convex, "_bisector_cut", always_fails)
    with pytest.raises(SplitError, match="no cut 1"):
        convex._first_bisector_cut(THIN_L, [0, 1])


def test_check_progress_rejects_a_cut_that_changes_nothing_or_the_area():
    """Catches a no-op or lossy cut being accepted and looping or changing the grid."""
    with pytest.raises(SplitError, match="did not split"):
        convex._check_progress(L_SHAPE, [L_SHAPE])
    with pytest.raises(SplitError, match="area"):
        convex._check_progress(L_SHAPE, [box(0, 0, 1, 1), box(1, 0, 2, 1)])
    convex._check_progress(L_SHAPE, [box(0, 0, 2, 1), box(0, 1, 1, 2)])


def test_split_concave_cells_keeps_unsplittable_cell_with_warning(caplog):
    """Catches one unsplittable cell aborting the whole split."""
    grid = _grid([HOLED_L, shapely.affinity.translate(L_SHAPE, 10)])
    vorflow.set_verbosity(1, console=False)
    try:
        with caplog.at_level(logging.WARNING, logger="vorflow"):
            split, report = split_concave_cells(grid, np.array([0, 1]))
    finally:
        vorflow.set_verbosity(1)
    assert split.geometry.iloc[0].equals(HOLED_L)
    assert report["node_id"].tolist() == [11]
    assert len(split) == 3
    messages = {r.getMessage() for r in caplog.records if r.levelno == logging.WARNING}
    assert any("1 concave cells unsplit" in m and "10 (" in m for m in messages)


def test_split_concave_cells_with_no_targets_returns_empty_report():
    """Catches int(NaN) on an empty grid and a report without its columns."""
    empty = _grid([])
    split, report = split_concave_cells(empty, np.array([], dtype=int))
    assert split.empty
    assert report.empty
    assert list(report.columns) == ["node_id", "convexity", "n_pieces", "piece_ids", "min_piece_fraction"]
    assert report.dtypes.to_dict() == {
        "node_id": np.int64, "convexity": np.float64, "n_pieces": np.int64,
        "piece_ids": object, "min_piece_fraction": np.float64,
    }


def test_summarize_quality_without_star_column_does_not_raise():
    """Catches a KeyError on frames from older calculate_mesh_quality output."""
    quality = calculate_mesh_quality(gpd.GeoDataFrame(geometry=[L_SHAPE]))
    summarize_quality(quality.drop(columns="star_from_centroid"))


def test_mesh_quality_flags_concave_and_hidden_face_cells():
    """Catches is_concave/star_from_centroid missing or computed on the wrong cells."""
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), L_SHAPE, THIN_L])
    quality = calculate_mesh_quality(gdf)
    assert quality["is_concave"].tolist() == [False, True, True]
    assert quality["star_from_centroid"].tolist() == [True, True, False]


def test_mesh_quality_does_not_flag_near_collinear_vertices():
    """Catches roundoff from clipping (a vertex 1e-12 off its edge) being flagged as concave."""
    almost_square = Polygon([(0, 0), (0.5, -1e-12), (1, 0), (1, 1), (0, 1)])
    quality = calculate_mesh_quality(gpd.GeoDataFrame(geometry=[almost_square]))
    assert not quality["is_concave"].iloc[0]
    assert quality["star_from_centroid"].iloc[0]


def test_convex_pieces_cut_l_shape_along_a_diagonal():
    """Catches a split that adds vertices or loses area when a diagonal exists."""
    pieces, added = convex_pieces(L_SHAPE)
    assert len(pieces) == 2
    assert added == []
    assert all(_is_convex(p) for p in pieces)
    assert sum(p.area for p in pieces) == pytest.approx(L_SHAPE.area, rel=1e-12)
    vertices = {tuple(c) for c in np.asarray(L_SHAPE.exterior.coords)}
    assert all({tuple(c) for c in np.asarray(p.exterior.coords)} <= vertices for p in pieces)


def test_convex_pieces_fall_back_to_a_bisector_cut():
    """Catches a split that gives up when no diagonal resolves the reflex vertex."""
    pieces, added = convex_pieces(SLIT)
    assert all(_is_convex(p) for p in pieces)
    assert sum(p.area for p in pieces) == pytest.approx(SLIT.area, rel=1e-12)
    assert len(added) == 1
    assert added[0] == pytest.approx([5.0, 0.0])


def test_bisector_ending_on_a_barrier_face_cuts_the_face_off_whole():
    """Catches a bisector cut adding a vertex inside a barrier face, which splits it into two faces."""
    barrier = LineString([(0, 0), (10, 0)])
    pieces, added = convex_pieces(SLIT, barrier)

    assert added == []
    assert len(pieces) == 3
    assert all(_is_convex(p) for p in pieces)
    assert sum(p.area for p in pieces) == pytest.approx(SLIT.area, rel=1e-12)
    vertices = {tuple(c) for c in np.asarray(SLIT.exterior.coords)}
    assert all({tuple(c) for c in np.asarray(p.exterior.coords)} <= vertices for p in pieces)
    assert sum(p.boundary.intersection(barrier).length > 0 for p in pieces) == 1


def test_split_leaves_the_neighbour_across_a_barrier_face_unchanged():
    """Catches the neighbour across a barrier face gaining the bisector end as a vertex."""
    below = box(0, -5, 10, 0)
    grid = _grid([SLIT, below])

    split, report = split_concave_cells(grid, np.array([0]), LineString([(0, 0), (10, 0)]))

    assert split.geometry.iloc[1].equals_exact(below, 0)
    assert report["n_pieces"].iloc[0] == 3


def test_split_piece_holding_the_generator_keeps_it():
    """Catches a piece replacing a generator well inside it with its centroid, which skews its old faces."""
    grid = _grid([L_SHAPE])
    grid.loc[0, ["x", "y"]] = [0.5, 0.2]

    split, _ = split_concave_cells(grid, np.array([0]))

    assert split[["x", "y"]].iloc[0].tolist() == [0.5, 0.2]
    other = split.geometry.iloc[1].centroid
    assert split[["x", "y"]].iloc[1].tolist() == pytest.approx([other.x, other.y])


def test_split_piece_with_a_barrier_face_is_centred_on_its_normal():
    """Catches a piece cut off with a barrier face taking its centroid, which skews that face."""
    grid = _grid([SLIT])
    grid.loc[0, ["x", "y"]] = [5.0, 1.0]

    split, _ = split_concave_cells(grid, np.array([0]), LineString([(0, 0), (10, 0)]))

    triangle = split[np.isclose(split.geometry.area, 5.0)]
    assert len(triangle) == 1
    assert triangle[["x", "y"]].iloc[0].tolist() == pytest.approx([5.0, 0.5])


def _grid(cells):
    return gpd.GeoDataFrame(
        {
            "node_id": np.arange(10, 10 + len(cells), dtype=np.int64),
            "x": [shapely.get_coordinates(c)[0, 0] for c in cells],
            "y": [shapely.get_coordinates(c)[0, 1] for c in cells],
            "zone_id": 7,
        },
        geometry=cells,
    )


def test_split_concave_cells_gives_fresh_ids_and_centroid_centres():
    """Catches duplicated node_ids, generator x/y on split pieces, or lost attributes."""
    neighbour = box(2, 0, 3, 2)
    grid = _grid([shapely.orient_polygons(L_SHAPE, exterior_cw=True), neighbour])

    split, report = split_concave_cells(grid, np.array([0]))

    assert len(split) == 3
    assert split["node_id"].is_unique
    assert split["node_id"].dtype == np.int64
    assert split["node_id"].iloc[0] == 10
    assert split["node_id"].iloc[2] == 12
    assert (split["zone_id"] == 7).all()
    pieces = split.iloc[[0, 2]]
    centroids = shapely.get_coordinates(pieces.geometry.centroid.to_numpy())
    assert np.allclose(pieces[["x", "y"]].to_numpy(), centroids)
    # Clockwise cells stay clockwise; the neighbour is untouched.
    assert not shapely.is_ccw(shapely.get_exterior_ring(pieces.geometry.to_numpy())).any()
    assert split.geometry.iloc[1].equals_exact(neighbour, 0)
    assert report.to_dict("list")["piece_ids"] == [[10, 12]]
    assert report["n_pieces"].iloc[0] == 2


def test_split_concave_cells_inserts_bisector_end_into_neighbour():
    """Catches a hanging vertex left on the neighbour's face after a bisector cut."""
    below = box(0, -5, 10, 0)
    grid = _grid([SLIT, below])

    split, _ = split_concave_cells(grid, np.array([0]))

    assert split.geometry.area.sum() == pytest.approx(SLIT.area + below.area, rel=1e-12)
    below_coords = {tuple(np.round(c, 9)) for c in shapely.get_coordinates(split.geometry.iloc[1])}
    assert (5.0, 0.0) in below_coords


def _hole_grid(**kwargs):
    """Grid of a square domain with an irregular hole, clipped at the hole's reflex corners."""
    hole = [(3, 3), (7, 3.5), (6, 5), (7.5, 7), (4, 6.5), (4.5, 5)]
    domain = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [hole])
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1, densify=False)
    clean_polys, _, _ = cm.generate()
    ring = domain.interiors[0]
    hole_nodes = [ring.interpolate(d).coords[0] for d in np.arange(0.0, ring.length, 0.5)]
    hole_nodes += list(hole)
    outer_nodes = [domain.exterior.interpolate(d).coords[0] for d in np.arange(0.0, 40.0, 1.0)]
    lattice = np.mgrid[0.5:10:1.0, 0.5:10:1.0].reshape(2, -1).T
    inner = domain.buffer(-0.3)
    interior_nodes = lattice[shapely.contains_xy(inner, lattice[:, 0], lattice[:, 1])]
    nodes = np.unique(np.round(np.vstack([hole_nodes, outer_nodes, interior_nodes]), 12), axis=0)
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)
    tessellator = VoronoiTessellator(mesh_gen, cm, **kwargs)
    return tessellator, tessellator.generate(), domain


def _unshared_interior_edges(grid, domain):
    """Cell edges used by one cell only that do not lie on the domain boundary (MF6 disconnected faces)."""
    counts = {}
    for geom in grid.geometry:
        coords = np.round(np.asarray(geom.exterior.coords), 9)
        for a, b in zip(map(tuple, coords[:-1]), map(tuple, coords[1:])):
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
    boundary = domain.boundary.buffer(1e-9)
    return [edge for edge, count in counts.items() if count == 1 and not boundary.covers(LineString(edge))]


def test_split_concave_all_leaves_a_valid_convex_grid():
    """Catches a split that loses area, duplicates node_ids or disconnects faces."""
    _, plain, domain = _hole_grid()
    concave = calculate_mesh_quality(plain)["is_concave"]
    assert concave.sum() > 0

    tessellator, grid, _ = _hole_grid(split_concave="all")

    quality = calculate_mesh_quality(grid)
    assert not quality["is_concave"].any()
    assert grid["node_id"].is_unique
    assert len(grid) > len(plain)
    assert grid.geometry.area.sum() == pytest.approx(domain.area, rel=1e-12)
    assert domain.symmetric_difference(unary_union(list(grid.geometry))).area < 1e-9
    assert _unshared_interior_edges(grid, domain) == []
    assert grid_to_disv(grid)["ncpl"] == len(grid)
    report = tessellator.concave_split_report
    assert set(report["node_id"]) == set(plain.loc[concave, "node_id"])
    pieces = grid[grid["node_id"].isin([i for ids in report["piece_ids"] for i in ids])]
    assert np.allclose(pieces[["x", "y"]].to_numpy(), pieces[["centroid_x", "centroid_y"]].to_numpy())
    # Other cells keep their shape; a bisector cut may add a vertex on a face.
    untouched = grid[~grid["node_id"].isin(pieces["node_id"])]
    assert untouched.set_index("node_id").geometry.geom_equals(
        plain.set_index("node_id").geometry.loc[untouched["node_id"]]
    ).all()


def test_split_concave_hidden_face_splits_only_cells_hidden_from_centroid():
    """Catches the hidden_face mode splitting concave cells that PRT handles."""
    _, plain, _ = _hole_grid()
    quality = calculate_mesh_quality(plain)
    hidden = plain.loc[~quality["star_from_centroid"], "node_id"]
    assert 0 < len(hidden) < quality["is_concave"].sum()

    tessellator, grid, _ = _hole_grid(split_concave="hidden_face")

    assert set(tessellator.concave_split_report["node_id"]) == set(hidden)
    assert calculate_mesh_quality(grid)["star_from_centroid"].all()


def test_split_concave_off_by_default(monkeypatch):
    """Catches the default grid running the split or gaining its columns."""

    def fail(*args, **kwargs):
        raise AssertionError("the split ran with split_concave off")

    monkeypatch.setattr(vorflow.tessellator, "split_concave_cells", fail)
    monkeypatch.setattr(vorflow.tessellator, "centroid_fan_fraction", fail)
    tessellator, grid, _ = _hole_grid()
    assert tessellator.concave_split_report is None
    assert not {"concave_split", "source_x", "source_y"} & set(grid.columns)


def test_summarize_quality_counts_split_pieces_at_their_mesh_node(caplog):
    """Catches split pieces, which carry their centroid as x/y, moving to the internal statistics."""
    _, base, _ = _hole_grid()
    _, split, _ = _hole_grid(split_concave="all")
    assert len(split) > len(base)

    def boundary_cells(grid):
        caplog.clear()
        summarize_quality(grid)
        line = next(r.getMessage() for r in caplog.records if "Boundary Cells" in r.getMessage())
        return int(line.split(":")[1])

    vorflow.set_verbosity(1, console=False)
    try:
        with caplog.at_level(logging.INFO, logger="vorflow"):
            base_count, split_count = boundary_cells(base), boundary_cells(split)
            # The same pieces counted by their centroid, as before.
            split_count_by_centroid = boundary_cells(split.drop(columns="concave_split"))
    finally:
        vorflow.set_verbosity(1)
    assert split_count >= base_count > split_count_by_centroid


@pytest.mark.parametrize("value", [False, np.False_])
def test_split_concave_accepts_numpy_false(value):
    """Catches np.False_ being rejected as an off setting."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 1, 1), zone_id=1)
    mesh_gen = DummyMeshGenerator(nodes=np.empty((0, 2)), tags=[], zones_gdf=None)
    assert VoronoiTessellator(mesh_gen, cm, split_concave=value).split_concave is False


@pytest.mark.parametrize("value", [True, None, "yes", 1, 0, np.True_])
def test_split_concave_rejects_unknown_settings(value):
    """Catches a truthy setting that silently picks a mode."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 1, 1), zone_id=1)
    mesh_gen = DummyMeshGenerator(nodes=np.empty((0, 2)), tags=[], zones_gdf=None)
    with pytest.raises(ValueError, match="split_concave"):
        VoronoiTessellator(mesh_gen, cm, split_concave=value)


@pytest.mark.slow
def test_split_concave_on_gmsh_mesh_around_l_shaped_hole():
    """Catches a split that fails on a real mesh, where Gmsh puts a node on every reflex corner."""
    hole = [(4, 4), (6, 4), (6, 5), (5, 5), (5, 6), (4, 6)]
    domain = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [hole])
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1, resolution=0.5)
    mg = MeshGenerator(background_lc=0.5, verbosity=0)
    mg.generate(*cm.generate())
    plain = VoronoiTessellator(mg, cm).generate()
    concave = calculate_mesh_quality(plain)["is_concave"]
    assert concave.sum() > 0

    tessellator = VoronoiTessellator(mg, cm, split_concave="all")
    grid = tessellator.generate()

    assert len(grid) == len(plain) + int(tessellator.concave_split_report["n_pieces"].sum()) - concave.sum()
    assert not calculate_mesh_quality(grid)["is_concave"].any()
    assert grid["node_id"].is_unique
    assert grid.geometry.area.sum() == pytest.approx(domain.area, rel=1e-12)
    assert grid_to_disv(grid)["ncpl"] == len(grid)


# A square and a triangle joined at (1, 1); the second copy of that vertex lies
# 4e-14 away, as clipping a cell around a notch tip computes it.
PINCHED = Polygon([(0, 0), (1, 0), (1, 1), (2, 1.5), (1.5, 2), (1 + 4e-14, 1 + 4e-14), (0, 1)])


def test_split_pinches_separates_lobes_joined_through_a_roundoff_neck():
    """Catches a pinched cell surviving as one ring that passes twice through one point."""
    assert PINCHED.is_valid
    out = convex.split_pinches(np.array([PINCHED, L_SHAPE, None], dtype=object), 1e-12)

    assert out[0].geom_type == "MultiPolygon"
    assert sorted(p.area for p in out[0].geoms) == pytest.approx([0.375, 1.0])
    assert out[1] is L_SHAPE
    assert out[2] is None


def test_explode_gives_a_pinched_lobe_its_own_cell():
    """Catches a pinched cell reaching DISV with a vertex listed twice."""
    grid = gpd.GeoDataFrame({"node_id": [5], "x": [0.5], "y": [0.5]}, geometry=[PINCHED])

    exploded = vorflow.tessellator._explode_with_unique_ids(grid)

    assert list(exploded["node_id"]) == [5, 6]
    assert exploded.geometry.iloc[0].covers(shapely.Point(0.5, 0.5))
    assert exploded[["x", "y"]].iloc[1].tolist() == pytest.approx([1.5, 1.5])


@pytest.mark.slow
def test_barrier_from_a_notch_tip_leaves_no_pinched_cell():
    """Catches the cell clipped around a notch tip pinching and the barrier safety net cutting it."""
    domain = Polygon([(0, 0), (0, 300), (228.26, 300), (241.46, 189.16), (240.15, 300), (300, 300), (300, 0)])
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1, resolution=20.0)
    cm.add_line(LineString([(241.46, 189.16), (109.94977304865061, 152.87715312809004)]), line_id="b", resolution=8.0, is_barrier=True)
    mg = MeshGenerator(background_lc=20.0, verbosity=0)
    mg.generate(*cm.generate())

    tessellator = VoronoiTessellator(mg, cm, split_concave="hidden_face")
    grid = tessellator.generate()

    assert (centroid_fan_fraction(grid.geometry.to_numpy()) > 0).all()
    cell2d = grid_to_disv(grid)["cell2d"]
    assert all(len(set(row[4:])) == row[3] for row in cell2d)
    assert grid["node_id"].is_unique
    assert grid.geometry.area.sum() == pytest.approx(domain.area, rel=1e-12)


# Seed 17 of the reviewer's random barrier cases: a narrow notch whose tip lies
# 0.8 above a barrier, so the tip cell's bisector cut ends on its barrier face.
SEED17_DOMAIN = shapely.from_wkt(
    "POLYGON ((0 0, 0 300, 156.23 300, 166.53 150.69, 168.15 300, 300 300, 300 0, 0 0), "
    "(81.36 181.83, 55.42 163.85, 69.47 143.57, 95.41 161.55, 81.36 181.83), "
    "(159.42 176.62, 153.33 191.07, 108.97 172.37, 115.07 157.92, 159.42 176.62), "
    "(144.53 30.51, 204.33 33.97, 201.3 86.34, 141.5 82.88, 144.53 30.51))"
)
SEED17_BARRIERS = [
    [(62.87977551686288, 140.1090564788047), (167.42094021909605, 137.49639819765943)],
    [(108.97, 172.37), (231.88029378598097, 124.30317416998481)],
    [(72.82200468007599, 113.16464001399362), (244.12910942171544, 110.69204419156041)],
]


def _barrier_faces(grid, barriers):
    """Connectivity rows of the faces that lie on ``barriers``."""
    from vorflow.utils import build_connectivity

    faces = build_connectivity(grid, center="generator")
    lines = faces.geometry.to_numpy()
    on = np.ones(len(lines), dtype=bool)
    for point in (shapely.get_point(lines, 0), shapely.get_point(lines, -1),
                  shapely.line_interpolate_point(lines, 0.5, normalized=True)):
        on &= shapely.distance(point, barriers) < 1e-6
    return faces[on]


@pytest.mark.slow
def test_split_keeps_every_barrier_segment_one_face():
    """Catches the split cutting an exact barrier face in two at a notch tip."""
    cm = ConceptualMesh()
    cm.add_polygon(SEED17_DOMAIN, zone_id=1, resolution=20.0)
    for k, coords in enumerate(SEED17_BARRIERS):
        cm.add_line(LineString(coords), line_id=f"b{k}", resolution=8.0, is_barrier=True)
    mg = MeshGenerator(background_lc=20.0, verbosity=0)
    mg.generate(*cm.generate())
    barriers = shapely.MultiLineString(SEED17_BARRIERS)

    plain = _barrier_faces(VoronoiTessellator(mg, cm).generate(), barriers)
    tessellator = VoronoiTessellator(mg, cm, split_concave="all")
    split = _barrier_faces(tessellator.generate(), barriers)

    assert len(tessellator.concave_split_report) > 0
    assert len(split) == len(plain)
    assert split.geometry.length.sum() == pytest.approx(plain.geometry.length.sum(), rel=1e-9)
    assert split["ortho_error"].max() < 1e-6


@pytest.mark.slow
def test_split_keeps_barrier_faces_orthogonal_at_a_notch_tip():
    """Catches the split piece with a barrier face taking its centroid instead of the generator."""
    domain = Polygon([(0, 0), (0, 300), (156.23, 300), (166.53, 150.69), (168.15, 300), (300, 300), (300, 0)])
    barrier = LineString([(108.97, 172.37), (231.88, 124.30)])
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1, resolution=20.0)
    cm.add_line(barrier, line_id="b", resolution=8.0, is_barrier=True)
    mg = MeshGenerator(background_lc=20.0, verbosity=0)
    mg.generate(*cm.generate())

    tessellator = VoronoiTessellator(mg, cm, split_concave="all")
    faces = _barrier_faces(tessellator.generate(), barrier)

    assert len(tessellator.concave_split_report) == 1
    assert faces["ortho_error"].max() < 1e-6
