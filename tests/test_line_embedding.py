"""
Tests for line embedding in the presence of polygons.

Regression tests for the bug where lines were not properly embedded into
the triangular mesh when polygons were also present. Root causes:
  1. getEntitiesInBoundingBox requires containment, not intersection — tiny
     entity bboxes could never contain large domain surfaces.
  2. Boundary lines (created by fragment when a line crosses a polygon edge)
     were re-embedded into their own surface, corrupting the mesh.
"""

import pytest
from shapely.geometry import Polygon, LineString, Point

from vorflow.blueprint import ConceptualMesh
from vorflow.engine import MeshGenerator
from vorflow.tessellator import VoronoiTessellator



def _nodes_near_line(nodes, line, tolerance):
    """Count mesh nodes that lie within `tolerance` of a LineString."""
    count = 0
    for x, y in nodes:
        pt = Point(x, y)
        if line.distance(pt) < tolerance:
            count += 1
    return count


class TestLineEmbeddingWithPolygons:
    """
    Core regression: a line crossing through a domain that also contains
    interior polygons must produce mesh nodes along the full line path,
    not just at polygon boundaries.
    """

    def test_line_embedded_with_polygon_present(self):
        """
        A line crossing through a domain with an interior polygon must
        produce nodes along the line — the defining symptom of the bug
        was that zero nodes appeared along interior line segments.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        # Large domain
        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)

        # Small interior square that the line crosses through
        square = Polygon([(40, 15), (60, 15), (60, 35), (40, 35)])
        cm.add_polygon(square, zone_id=2, resolution=5.0, z_order=1)

        # Horizontal line crossing through the square
        line = LineString([(10, 25), (90, 25)])
        cm.add_line(line, line_id="crossing_line", resolution=3.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=10.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success, "Mesh generation failed"

        # Count nodes near the line — before the fix this was ~0
        nodes_on_line = _nodes_near_line(mg.nodes, line, tolerance=1.0)

        # With resolution=3.0 on an 80-unit line, we expect ~25+ nodes
        assert nodes_on_line >= 10, (
            f"Only {nodes_on_line} nodes found near line — "
            f"line embedding likely broken (expected >= 10)"
        )

    def test_line_embedded_without_polygon(self):
        """
        Baseline: line embedding works correctly without interior polygons.
        This should always pass — it's the control case.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)

        line = LineString([(10, 25), (90, 25)])
        cm.add_line(line, line_id="simple_line", resolution=3.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=10.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success

        nodes_on_line = _nodes_near_line(mg.nodes, line, tolerance=1.0)
        assert nodes_on_line >= 10, (
            f"Only {nodes_on_line} nodes near line in no-polygon case"
        )

    def test_line_node_count_comparable_with_and_without_polygons(self):
        """
        The number of nodes near a line should be roughly similar whether
        or not an interior polygon exists. The original bug caused a near-
        total loss of line nodes when polygons were present.
        """
        line = LineString([(10, 25), (90, 25)])

        # Case A: without polygon
        cm_a = ConceptualMesh(crs="EPSG:3857")
        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm_a.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)
        cm_a.add_line(line, line_id="line_a", resolution=3.0)
        polys_a, lines_a, pts_a = cm_a.generate()
        mg_a = MeshGenerator(background_lc=10.0, verbosity=0)
        mg_a.generate(polys_a, lines_a, pts_a)
        nodes_a = _nodes_near_line(mg_a.nodes, line, tolerance=1.0)

        # Case B: with polygon
        cm_b = ConceptualMesh(crs="EPSG:3857")
        cm_b.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)
        square = Polygon([(40, 15), (60, 15), (60, 35), (40, 35)])
        cm_b.add_polygon(square, zone_id=2, resolution=5.0, z_order=1)
        cm_b.add_line(line, line_id="line_b", resolution=3.0)
        polys_b, lines_b, pts_b = cm_b.generate()
        mg_b = MeshGenerator(background_lc=10.0, verbosity=0)
        mg_b.generate(polys_b, lines_b, pts_b)
        nodes_b = _nodes_near_line(mg_b.nodes, line, tolerance=1.0)

        # Case B should have at least 50% of Case A's nodes
        # (it may have more due to the finer polygon resolution)
        ratio = nodes_b / max(nodes_a, 1)
        assert ratio >= 0.5, (
            f"With-polygon case has {nodes_b} nodes vs {nodes_a} without — "
            f"ratio {ratio:.2f} < 0.5, embedding likely broken"
        )

    def test_line_crossing_multiple_polygons(self):
        """
        A line that crosses through multiple interior polygons must still
        produce nodes along its full length.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        domain = Polygon([(0, 0), (200, 0), (200, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=15.0, z_order=0)

        # Three squares along the line path
        for i, x_start in enumerate([30, 80, 140]):
            sq = Polygon([
                (x_start, 15), (x_start + 20, 15),
                (x_start + 20, 35), (x_start, 35)
            ])
            cm.add_polygon(sq, zone_id=10 + i, resolution=5.0, z_order=1)

        line = LineString([(10, 25), (190, 25)])
        cm.add_line(line, line_id="multi_cross", resolution=5.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=15.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success

        nodes_on_line = _nodes_near_line(mg.nodes, line, tolerance=1.5)
        # 180-unit line at resolution 5 → ~36 segments → ~30+ nodes expected
        assert nodes_on_line >= 15, (
            f"Only {nodes_on_line} nodes on line crossing 3 polygons"
        )

    def test_mesh_area_conservation_with_embedded_line(self):
        """
        Total mesh area must match the domain area, ensuring no
        garbage triangles extend outside the domain.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)

        square = Polygon([(40, 15), (60, 15), (60, 35), (40, 35)])
        cm.add_polygon(square, zone_id=2, resolution=5.0, z_order=1)

        line = LineString([(10, 25), (90, 25)])
        cm.add_line(line, line_id="area_test", resolution=3.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=10.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success

        vt = VoronoiTessellator(mg, cm, clip_to_boundary=True)
        grid = vt.generate()

        total_area = grid.geometry.area.sum()
        expected_area = domain.area  # 5000
        assert pytest.approx(total_area, rel=0.02) == expected_area, (
            f"Area mismatch: {total_area:.1f} vs {expected_area:.1f} — "
            f"possible out-of-domain triangles"
        )
