"""Unit tests for the shapely-only quad-buffer planning in vorflow.buffer."""
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

from vorflow import buffer
from vorflow._features import (
    feature_lc,
    is_embedded,
    line_parts,
    polygon_parts,
    positive_number,
    row_bool,
    sanitize_coords,
    unpinch_polygons,
)


def _row(name=0, **values):
    return pd.Series(values, name=name)


def _lines(*rows):
    return gpd.GeoDataFrame(list(rows), geometry="geometry")


def _polys(*rows):
    return gpd.GeoDataFrame(list(rows), geometry="geometry")


EMPTY = gpd.GeoDataFrame({"geometry": []}, geometry="geometry")


# --- row accessors -----------------------------------------------------------

def test_row_accessors_resolve_missing_and_nan_values():
    assert is_embedded(_row()) is True
    assert is_embedded(_row(embed=np.nan)) is True
    assert is_embedded(_row(embed=False)) is False
    assert row_bool(_row(flag="yes"), "flag") is True
    assert row_bool(_row(flag=np.nan), "flag", default=True) is True
    assert row_bool(_row(), "flag") is False
    assert positive_number("2.5") == 2.5
    assert positive_number(-1) is None
    assert positive_number("abc") is None
    assert feature_lc(_row(lc=np.nan), 4.0) == 4.0
    assert feature_lc(_row(), None) == 10.0
    assert feature_lc(_row(lc=1e-6), 4.0) == 0.001


def test_quad_buffer_thickness_and_z_order_defaults():
    assert buffer.quad_buffer_thickness(_row()) == 1
    assert buffer.quad_buffer_thickness(_row(quad_buffer_thickness=np.nan)) == 1
    assert buffer.quad_buffer_thickness(_row(quad_buffer_thickness=2.0)) == 2
    with pytest.raises(ValueError, match="either 1 or 2"):
        buffer.quad_buffer_thickness(_row(quad_buffer_thickness=3))
    assert buffer.feature_z_order(_row()) == 0.0
    assert buffer.feature_z_order(_row(z_order=np.nan)) == 0.0
    assert buffer.feature_z_order(_row(z_order="top")) == 0.0
    assert buffer.feature_z_order(_row(z_order=3)) == 3.0


def test_sanitize_coords_drops_duplicates_and_closing_point():
    ring = [(0, 0), (0, 0), (1, 0), (1, 1), (float("nan"), 2), (0, 0)]
    assert sanitize_coords(ring, require_closed=True, min_points=3) == [(0, 0), (1, 0), (1, 1)]
    assert sanitize_coords([(0, 0), (0, 0)], min_points=2) == []


def test_unpinch_polygons_turns_a_near_touching_notch_into_a_touching_hole():
    # The ring visits (5, 0) twice, 1e-13 apart: simple to GEOS, pinched to OCC.
    ring = [(0, 0), (5, 0), (4, 3), (6, 3), (5 + 1e-13, 0), (10, 0), (10, 10), (0, 10)]
    pinched = Polygon(ring)
    assert pinched.is_valid and pinched.exterior.is_simple

    parts = polygon_parts(unpinch_polygons(pinched, 1e-5))
    assert len(parts) == 1
    assert len(parts[0].interiors) == 1
    assert parts[0].is_valid
    assert parts[0].area == pytest.approx(pinched.area)


def test_unpinch_polygons_splits_a_near_touching_figure_eight():
    ring = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (1, 2), (1, 1 + 1e-13), (0, 1)]
    pinched = Polygon(ring)
    assert pinched.is_valid

    parts = polygon_parts(unpinch_polygons(pinched, 1e-5))
    assert sorted(p.area for p in parts) == pytest.approx([1.0, 1.0])
    assert all(not p.interiors for p in parts)


def test_unpinch_polygons_leaves_clean_geometry_untouched():
    # Close consecutive vertices are sanitize_coords' job, not a pinch.
    poly = Polygon([(0, 0), (1, 0), (1 + 1e-7, 0), (1, 1), (0, 1)])
    multi = box(0, 0, 1, 1).union(box(1 + 1e-7, 0, 2, 1))
    assert unpinch_polygons(poly, 1e-5) is poly
    assert unpinch_polygons(multi, 1e-5) is multi


def test_part_helpers_flatten_collections():
    mp = box(0, 0, 1, 1).union(box(3, 3, 4, 4))
    assert len(polygon_parts(mp)) == 2
    assert polygon_parts(LineString([(0, 0), (1, 1)])) == []
    mls = MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]])
    assert len(line_parts(mls)) == 2


def test_coerce_offset_line_merges_or_keeps_longest_part():
    line = LineString([(0, 0), (1, 0)])
    assert buffer.coerce_offset_line(line) is line
    touching = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])
    assert buffer.coerce_offset_line(touching).length == pytest.approx(2.0)
    disjoint = MultiLineString([[(0, 0), (1, 0)], [(5, 0), (8, 0)]])
    assert buffer.coerce_offset_line(disjoint).length == pytest.approx(3.0)
    assert buffer.coerce_offset_line(Point(0, 0)) is None


def test_domain_union_ignores_field_only_polygons():
    polys = _polys(
        {"geometry": box(0, 0, 10, 10), "embed": True},
        {"geometry": box(20, 0, 30, 10), "embed": False},
    )
    domain = buffer.domain_union_geometry(polys)
    assert domain.area == pytest.approx(100.0)
    assert buffer.domain_union_geometry(EMPTY) is None
    assert buffer.domain_union_geometry(None) is None


def test_strip_cells_across_keeps_barrier_node_rows_off_the_line():
    assert buffer.strip_cells_across(_row(quad_buffer_thickness=2)) == 2
    assert buffer.strip_cells_across(_row(is_barrier=True)) == 1
    # An odd quad count: thickness cell rows on each side of the barrier.
    assert buffer.strip_cells_across(_row(is_barrier=True, quad_buffer_thickness=2)) == 3
    row = _row(lc=2.0, is_barrier=True, quad_buffer=True, quad_buffer_thickness=2)
    assert buffer.protection_epsilon(row, 5) == 3.0


# --- footprint planning ------------------------------------------------------

def test_plan_line_strip_builds_strip_with_ordered_corners():
    parts = buffer.plan_line_strip(LineString([(0, 5), (10, 5)]), lc=1.0, thickness=2,
                                   domain=None, feature_name=0)
    assert len(parts) == 1
    part = parts[0]
    assert part.strip.area == pytest.approx(10.0 * 2.0)
    # neg start, neg end, pos end, pos start (setTransfiniteSurface "Left").
    assert part.corners == [(0.0, 4.0), (10.0, 4.0), (10.0, 6.0), (0.0, 6.0)]
    pos, neg = part.side_lines
    assert pos.coords[0][1] == pytest.approx(6.0)
    assert neg.coords[0][1] == pytest.approx(4.0)


def test_plan_line_strip_clips_offsets_to_the_domain():
    parts = buffer.plan_line_strip(LineString([(-5, 5), (15, 5)]), lc=1.0, thickness=1,
                                   domain=box(0, 0, 10, 10), feature_name=0)
    assert len(parts) == 1
    minx, _, maxx, _ = parts[0].strip.bounds
    assert minx == pytest.approx(0.0) and maxx == pytest.approx(10.0)


def test_plan_polygon_band_is_an_annulus_around_the_outline():
    band = buffer.plan_polygon_band(box(0, 0, 10, 10), lc=1.0, thickness=2, feature_name=0)
    assert band.area == pytest.approx(12.0 ** 2 - 8.0 ** 2)
    assert band.contains(Point(0, 5))  # the outline runs along the band's midline
    assert not band.contains(Point(5, 5))


def test_plan_polygon_band_warns_and_skips_narrow_polygons():
    with pytest.warns(UserWarning, match="too narrow for a quad_buffer band"):
        assert buffer.plan_polygon_band(box(0, 0, 0.8, 0.8), lc=1.0, thickness=1,
                                        feature_name=3) is None


def test_priority_key_orders_z_order_then_lc_then_width_then_kind_then_order():
    base = buffer.priority_key(0.0, 1.0, 1, 'line', 5)
    assert buffer.priority_key(1.0, 1.0, 1, 'line', 9) < base  # higher z_order wins
    assert buffer.priority_key(0.0, 0.5, 1, 'line', 9) < base  # finer lc wins
    assert buffer.priority_key(0.0, 1.0, 2, 'line', 9) < base  # wider strip wins
    assert base < buffer.priority_key(0.0, 1.0, 1, 'poly', 0)  # line beats polygon
    assert buffer.priority_key(0.0, 1.0, 1, 'line', 4) < base  # then input order


def test_plan_quad_buffers_keys_lines_then_embedded_polygons():
    lines = _lines(
        {"geometry": LineString([(0, 5), (20, 5)]), "lc": 1.0, "quad_buffer": True},
        {"geometry": LineString([(0, 8), (20, 8)]), "lc": 1.0, "quad_buffer": False},
    )
    polys = _polys(
        {"geometry": box(0, 0, 20, 20), "lc": 2.0, "quad_buffer": False, "embed": True},
        {"geometry": box(5, 10, 15, 18), "lc": 1.0, "quad_buffer": True, "embed": True},
        {"geometry": box(5, 10, 15, 18), "lc": 1.0, "quad_buffer": True, "embed": False},
    )
    plans = buffer.plan_quad_buffers(polys, lines, background_lc=2.0, domain=box(0, 0, 20, 20))
    assert list(plans) == [('line', 0), ('poly', 1)]
    line_plan, poly_plan = plans[('line', 0)], plans[('poly', 1)]
    assert line_plan.kind == 'line' and len(line_plan.parts) == 1 and line_plan.band is None
    assert poly_plan.kind == 'poly' and poly_plan.parts == [] and poly_plan.band is not None
    assert line_plan.width == 1.0
    assert line_plan.priority_key[-1] == 0 and poly_plan.priority_key[-1] == 1


# --- corridors and obstacles -------------------------------------------------

def test_protection_epsilon_by_feature_kind():
    assert buffer.protection_epsilon(_row(lc=2.0, quad_buffer=True, quad_buffer_thickness=2), 5) == 2.0
    # The deprecated straddle_width is ignored.
    assert buffer.protection_epsilon(_row(lc=2.0, straddle_width=0.6), 5) == pytest.approx(0.4)
    assert buffer.protection_epsilon(_row(lc=2.0, is_barrier=True), 5) == pytest.approx(0.4)


def test_protected_corridors_select_barrier_straddle_and_quad_features():
    lines = _lines(
        {"geometry": LineString([(0, 1), (10, 1)]), "lc": 1.0, "is_barrier": True},
        {"geometry": LineString([(0, 2), (10, 2)]), "lc": 1.0},
        {"geometry": LineString([(0, 3), (10, 3)]), "lc": 1.0, "straddle_width": 0.5},
        {"geometry": LineString([(0, 4), (10, 4)]), "lc": 1.0, "quad_buffer": True},
    )
    polys = _polys(
        {"geometry": box(0, 0, 10, 10), "quad_buffer": True, "embed": False},
        {"geometry": box(2, 2, 8, 8), "quad_buffer": True, "embed": True, "lc": 1.0},
    )
    corridors = buffer.protected_corridors(polys, lines, background_lc=1.0)
    assert list(corridors) == [('line', 0), ('line', 2), ('line', 3), ('poly', 1)]
    basis, eps = corridors[('poly', 1)]
    assert basis.geom_type == 'LineString' and eps == 0.5
    zone = buffer.barrier_zone(corridors)
    assert zone.contains(Point(5, 1)) and not zone.contains(Point(5, 9.5))
    assert buffer.barrier_zone({}) is None


def test_corridor_geometry_uses_flat_caps_and_min_half_width():
    corridor = buffer.corridor_geometry(LineString([(0, 0), (10, 0)]), eps=1.0)
    assert corridor.bounds == pytest.approx((0.0, -1.2, 10.0, 1.2))
    wide = buffer.corridor_geometry(LineString([(0, 0), (10, 0)]), eps=1.0, min_half_width=3.0)
    assert wide.bounds[1] == pytest.approx(-3.0)


def _crossing_plans():
    lines = _lines(
        {"geometry": LineString([(0, 10), (20, 10)]), "lc": 1.0, "quad_buffer": True, "z_order": 0},
        {"geometry": LineString([(10, 0), (10, 20)]), "lc": 1.0, "quad_buffer": True, "z_order": 1},
    )
    plans = buffer.plan_quad_buffers(EMPTY, lines, background_lc=2.0, domain=None)
    corridors = buffer.protected_corridors(EMPTY, lines, background_lc=2.0)
    return plans, corridors


def test_higher_priority_obstacles_only_trim_the_loser():
    plans, corridors = _crossing_plans()
    assert buffer.higher_priority_obstacles(('line', 1), plans, corridors) is None
    obstacles = buffer.higher_priority_obstacles(('line', 0), plans, corridors)
    # Winner footprint (x in 9.5..10.5) grown by the half-cell crossing gap.
    assert obstacles.bounds[0] == pytest.approx(9.0) and obstacles.bounds[2] == pytest.approx(11.0)
    assert buffer.higher_priority_obstacles(('line', 9), plans, corridors) is None


def test_higher_priority_obstacles_include_non_quad_corridors():
    lines = _lines(
        {"geometry": LineString([(0, 10), (20, 10)]), "lc": 1.0, "quad_buffer": True},
        {"geometry": LineString([(10, 0), (10, 20)]), "lc": 1.0, "is_barrier": True},
    )
    plans = buffer.plan_quad_buffers(EMPTY, lines, background_lc=2.0, domain=None)
    corridors = buffer.protected_corridors(EMPTY, lines, background_lc=2.0)
    obstacles = buffer.higher_priority_obstacles(('line', 0), plans, corridors)
    # Barrier corridor half-width max(0.2 * 1.2, 0.6 * lc) = 0.6.
    assert obstacles.bounds == pytest.approx((9.4, 0.0, 10.6, 20.0))


def test_find_crossings_records_a_disk_from_the_losers_side_only():
    plans, _ = _crossing_plans()
    assert buffer.find_crossings(('line', 1), plans) == []
    crossings = buffer.find_crossings(('line', 0), plans)
    assert len(crossings) == 1
    c = crossings[0]
    assert (c.x, c.y) == pytest.approx((10.0, 10.0))
    assert c.size == 1.0
    assert c.radius == pytest.approx(0.5 * math.hypot(1.0, 1.0) + 0.5 * (1.0 + 1.0) + 1.0)
    assert buffer.find_crossings(('poly', 0), plans) == []
    assert buffer.find_all_crossings(plans) == crossings


# --- trimming ----------------------------------------------------------------

def test_clean_trimmed_pieces_drops_slivers_with_a_warning():
    with pytest.warns(UserWarning, match=r"dropped 1 sliver piece"):
        kept = buffer.clean_trimmed_pieces(
            box(0, 0, 10, 1).union(box(0, 3, 10, 3.2)), lc=1.0, feature_label="line feature 0"
        )
    assert len(kept) == 1 and kept[0].area == pytest.approx(10.0)


def test_trim_against_obstacles():
    strip = box(0, 0, 10, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert buffer.trim_against_obstacles(strip, None, 1.0, "x") == (strip, False)
        assert buffer.trim_against_obstacles(strip, box(20, 0, 21, 1), 1.0, "x") == (strip, False)
    with pytest.warns(UserWarning, match="crosses a higher-priority protected feature"):
        trimmed, was_trimmed = buffer.trim_against_obstacles(strip, box(4, -1, 6, 2), 1.0, "x")
    assert was_trimmed and trimmed.area == pytest.approx(8.0)
    with pytest.warns(UserWarning):
        assert buffer.trim_against_obstacles(strip, box(-1, -1, 11, 2), 1.0, "x") == (None, True)
