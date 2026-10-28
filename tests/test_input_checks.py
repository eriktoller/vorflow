"""Tests for ConceptualMesh.check(): pre-meshing input checks (pure shapely, no Gmsh)."""
import copy
import logging

import pytest
from shapely.geometry import LineString, Point, box

import vorflow
from vorflow import ConceptualMesh
from vorflow._checks import COLUMNS, deduplicate, severity

BACKGROUND_LC = 50.0
LC = 10.0  # resolution of every seeded feature, so h = LC on the defect


def _domain_mesh():
    """A 1 km square domain at the background resolution."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 1000, 1000), zone_id="domain", resolution=BACKGROUND_LC)
    return cm


def _of_kind(findings, kind):
    """Findings rows of one kind."""
    return findings[findings["kind"] == kind]


def _assert_finding(findings, kind, expected):
    """One finding of ``kind`` with severity ``expected``, or none when expected is None."""
    rows = _of_kind(findings, kind)
    if expected is None:
        assert rows.empty, rows.drop(columns="geometry")
        return
    assert len(rows) == 1, findings.drop(columns="geometry")
    assert rows.iloc[0]["severity"] == expected
    return rows.iloc[0]


# ---------------------------------------------------------------- one test per kind

@pytest.mark.parametrize("ratio, expected", [(0.05, "error"), (0.2, "warning"), (0.3, None)])
def test_short_segment(ratio, expected):
    cm = _domain_mesh()
    x, y = 100.0, 500.0
    coords = [(x, y), (x + 50, y), (x + 50 + ratio * LC, y), (x + 100, y + LC)]
    cm.add_line(LineString(coords), line_id="seg", resolution=LC)

    findings = cm.check(background_lc=BACKGROUND_LC)

    row = _assert_finding(findings, "short_segment", expected)
    if row is not None:
        assert row["feature_type"] == "line"
        assert row["feature_id"] == "seg"
        assert row["h"] == pytest.approx(LC)
        assert row["ratio"] == pytest.approx(ratio)


@pytest.mark.parametrize("ratio, expected", [(0.05, "error"), (0.3, "warning"), (0.7, None)])
def test_close_vertices_crossing_beside_a_corner(ratio, expected):
    cm = _domain_mesh()
    cm.add_polygon(box(400, 400, 600, 600), zone_id="zone", resolution=LC, z_order=1)
    # The line crosses the zone's top edge ratio * LC from its corner (600, 600).
    x0 = 600.0 - ratio * LC
    cm.add_line(LineString([(x0 + 30, 630), (x0 - 100, 500)]), line_id="cross", resolution=LC)

    findings = cm.check(background_lc=BACKGROUND_LC)

    row = _assert_finding(findings, "close_vertices", expected)
    if row is not None:
        assert {row["feature_id"], row["other_id"]} == {"zone", "cross"}
        assert row["distance"] == pytest.approx(ratio * LC)


@pytest.mark.parametrize(
    "ratio, expected", [(0.3, "error"), (0.8, "warning"), (1.0, "warning"), (1.2, None)]
)
def test_line_end_near_miss(ratio, expected):
    cm = _domain_mesh()
    cm.add_polygon(box(400, 400, 600, 600), zone_id="zone", resolution=LC, z_order=1)
    cm.add_line(LineString([(500, 800), (500, 600 + ratio * LC)]), line_id="short", resolution=LC)

    findings = cm.check(background_lc=BACKGROUND_LC)

    row = _assert_finding(findings, "line_end_near_miss", expected)
    if row is not None:
        assert (row["feature_id"], row["other_id"]) == ("short", "zone")
        assert (row["x"], row["y"]) == pytest.approx((500, 600 + ratio * LC))


def test_line_end_within_connectivity_tolerance_is_snapped_not_reported():
    cm = _domain_mesh()
    cm.add_polygon(box(400, 400, 600, 600), zone_id="zone", resolution=LC, z_order=1)
    cm.add_line(LineString([(500, 800), (500, 600.0005)]), line_id="snapped", resolution=LC)

    assert cm.check(background_lc=BACKGROUND_LC).empty


@pytest.mark.parametrize("ratio, expected", [(0.3, "error"), (0.8, "warning"), (1.2, None)])
def test_point_near_miss(ratio, expected):
    cm = _domain_mesh()
    cm.add_line(LineString([(100, 500), (300, 500)]), line_id="river", resolution=LC)
    cm.add_point(Point(200, 500 + ratio * LC), point_id="well", resolution=LC)

    findings = cm.check(background_lc=BACKGROUND_LC)

    row = _assert_finding(findings, "point_near_miss", expected)
    if row is not None:
        assert (row["feature_type"], row["feature_id"]) == ("point", "well")
        assert (row["other_type"], row["other_id"]) == ("line", "river")


def _add_sliver(cm, width):
    """Zone A, mostly covered by higher zone B, leaving a strip of A ``width`` wide."""
    cm.add_polygon(box(150, 350, 350, 500), zone_id="A", resolution=LC, z_order=1)
    cm.add_polygon(box(150 + width, 330, 370, 520), zone_id="B", resolution=LC, z_order=2)


@pytest.mark.parametrize("ratio, expected", [(0.2, "error"), (0.7, "warning"), (1.5, None)])
def test_sliver_polygon(ratio, expected):
    cm = _domain_mesh()
    _add_sliver(cm, ratio * LC)

    findings = cm.check(background_lc=BACKGROUND_LC)

    row = _assert_finding(findings, "sliver_polygon", expected)
    if row is not None:
        assert row["feature_id"] == "A"
        assert row["distance"] == pytest.approx(ratio * LC, rel=1e-2)


@pytest.mark.parametrize("ratio, expected", [(0.3, "warning"), (1.0, "warning"), (1.5, None)])
def test_narrow_gap_is_warning_only(ratio, expected):
    cm = _domain_mesh()
    cm.add_line(LineString([(100, 500), (500, 500)]), line_id="straight", resolution=LC)
    # Bows towards the straight line along its middle; both ends stay well clear.
    gap = 500 + ratio * LC
    cm.add_line(LineString([(100, 560), (250, gap), (350, gap), (500, 560)]), line_id="bowed",
                resolution=LC)

    findings = cm.check(background_lc=BACKGROUND_LC)

    row = _assert_finding(findings, "narrow_gap", expected)
    if row is not None:
        assert {row["feature_id"], row["other_id"]} == {"straight", "bowed"}
        assert row["ratio"] == pytest.approx(ratio)
    assert set(findings["kind"]) <= {"narrow_gap"}


@pytest.mark.parametrize(
    "kind, ratio, expected",
    [
        ("short_segment", 0.099, "error"),
        ("short_segment", 0.25, None),
        ("close_vertices", 0.5, None),
        ("line_end_near_miss", 0.5, "warning"),
        ("point_near_miss", 1.0, "warning"),
        ("sliver_polygon", 1.0, None),
        ("narrow_gap", 0.01, "warning"),
        ("narrow_gap", 1.0, "warning"),
    ],
)
def test_severity_thresholds(kind, ratio, expected):
    assert severity(kind, ratio) == expected


# ---------------------------------------------------------------- behaviour

def test_clean_inputs_give_an_empty_frame():
    cm = _domain_mesh()
    cm.add_polygon(box(200, 200, 400, 400), zone_id="zone", resolution=LC, z_order=1)
    cm.add_line(LineString([(500, 100), (500, 800)]), line_id="river", resolution=LC)
    cm.add_line(LineString([(300, 300), (300, 700)]), line_id="crossing", resolution=LC)
    cm.add_point(Point(800, 800), point_id="well", resolution=2.0)

    findings = cm.check(background_lc=BACKGROUND_LC)

    assert findings.empty
    assert list(findings.columns) == COLUMNS


def test_densification_vertices_are_not_reported():
    # Densifying the river every 10 puts a vertex 1 from where it crosses the
    # zone edge at y = 401: vorflow's own vertex, not an input defect.
    cm = _domain_mesh()
    cm.add_polygon(box(200, 200, 400, 401), zone_id="zone", resolution=LC, z_order=1)
    cm.add_line(LineString([(300, 100), (300, 800)]), line_id="river", resolution=LC)

    assert cm.check(background_lc=BACKGROUND_LC).empty


def test_one_defect_gives_one_row():
    # The strip trips sliver_polygon and, on the domain's hole ring, short_segment.
    cm = _domain_mesh()
    _add_sliver(cm, 0.2 * LC)

    findings = cm.check(background_lc=BACKGROUND_LC)

    assert len(findings) == 1
    row = findings.iloc[0]
    assert (row["kind"], row["severity"]) == ("sliver_polygon", "error")
    assert set(row["site_kinds"].split(", ")) == {"sliver_polygon", "short_segment"}


def test_deduplicate_keeps_the_most_severe_per_site():
    def finding(kind, level, ratio, x):
        return {"kind": kind, "severity": level, "ratio": ratio, "h": 10.0,
                "geometry": Point(x, 0)}

    findings = [
        finding("narrow_gap", "warning", 0.2, 0.0),
        finding("short_segment", "error", 0.05, 15.0),  # within 2 h: same site
        finding("close_vertices", "warning", 0.3, 15.0),
        finding("close_vertices", "warning", 0.4, 100.0),  # a separate site
    ]

    kept = sorted(deduplicate(findings), key=lambda f: f["geometry"].x)

    assert [(f["kind"], f["site_kinds"]) for f in kept] == [
        ("short_segment", "close_vertices, narrow_gap, short_segment"),
        ("close_vertices", "close_vertices"),
    ]


def test_field_only_features_are_not_checked():
    cm = _domain_mesh()
    cm.add_polygon(box(400, 400, 600, 600), zone_id="zone", resolution=LC, z_order=1)
    cm.add_line(LineString([(500, 800), (500, 603)]), line_id="sizing", resolution=LC, embed=False)
    cm.add_point(Point(500, 602), point_id="sizing-point", resolution=LC, embed=False)
    cm.add_polygon(box(150, 350, 152, 500), zone_id="sizing-strip", resolution=LC, embed=False)

    assert cm.check(background_lc=BACKGROUND_LC).empty


def test_field_only_features_still_set_the_local_size():
    cm = _domain_mesh()
    coords = [(100, 500), (150, 500), (152, 500), (200, 510)]  # a 2-long segment
    cm.add_line(LineString(coords), line_id="seg", resolution=LC)
    assert _of_kind(cm.check(background_lc=BACKGROUND_LC), "short_segment")["ratio"].tolist() == \
        pytest.approx([0.2])

    # A field-only zone at resolution 2 makes h = 2 there, so the segment is fine.
    cm.add_polygon(box(140, 490, 160, 510), zone_id="fine", resolution=2.0, embed=False)
    assert cm.check(background_lc=BACKGROUND_LC).empty


def test_check_leaves_inputs_and_generate_results_unchanged():
    cm = _domain_mesh()
    _add_sliver(cm, 0.2 * LC)
    cm.add_line(LineString([(500, 800), (500, 603)]), line_id="short", resolution=LC,
                simplify_tolerance=0.5)
    raw_before = copy.deepcopy((cm.raw_polygons, cm.raw_lines, cm.raw_points))
    clean = cm.generate()
    clean_geometry = [frame.geometry.copy() for frame in clean]
    diagnostics = cm.junction_diagnostics

    cm.check(background_lc=BACKGROUND_LC)

    assert (cm.raw_polygons, cm.raw_lines, cm.raw_points) == raw_before
    for frame, kept, geometry in zip(clean, (cm.clean_polygons, cm.clean_lines, cm.clean_points),
                                     clean_geometry):
        assert kept is frame
        assert frame.geometry.geom_equals_exact(geometry, tolerance=0).all()
    assert cm.junction_diagnostics is diagnostics


def test_background_lc_defaults_to_the_coarsest_resolution():
    cm = _domain_mesh()
    _add_sliver(cm, 0.7 * LC)

    default = cm.check()
    explicit = cm.check(background_lc=BACKGROUND_LC)

    assert default.drop(columns="geometry").equals(explicit.drop(columns="geometry"))


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf"), "coarse", True])
def test_bad_background_lc_raises(value):
    with pytest.raises(ValueError, match="background_lc"):
        _domain_mesh().check(background_lc=value)


def test_missing_background_lc_without_resolutions_raises():
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 100, 100), zone_id="domain")

    with pytest.raises(ValueError, match="background_lc"):
        cm.check()


def test_check_logs_a_one_line_summary(caplog):
    cm = _domain_mesh()
    _add_sliver(cm, 0.2 * LC)
    vorflow.set_verbosity(1, console=False)
    try:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="vorflow"):
            cm.check(background_lc=BACKGROUND_LC)
        messages = [r.getMessage() for r in caplog.records]
    finally:
        vorflow.set_verbosity(1)

    # pytest >= 9.1 can record a propagating message twice (once on the
    # vorflow logger, once on the root), so compare distinct messages.
    assert set(messages) == {"Input check: 1 error(s), 0 warning(s)."}
