"""Minimal end-to-end vorflow example used by release smoke tests.

Run with ``--plot PATH`` to also save a figure of the grid (needs Matplotlib,
from the ``examples`` extra). The README's Basic Usage figure is made with
``python examples/basic_usage.py --plot docs/images/basic_usage.png``.
"""

import argparse

from shapely.geometry import LineString, Point, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

DOMAIN = box(0, 0, 200, 200)
WELL_POINT = Point(25, 25)
FAULT_LINE = LineString([(100, 0), (100, 150)])


def main():
    blueprint = ConceptualMesh(crs="EPSG:3857")
    blueprint.add_polygon(DOMAIN, zone_id=1)
    blueprint.add_point(
        WELL_POINT,
        point_id="Well-A",
        resolution=2,
        growth_factor=1.2,
    )
    blueprint.add_line(
        FAULT_LINE,
        line_id="Fault-1",
        resolution=1,
        is_barrier=True,
    )

    clean_polys, clean_lines, clean_pts = blueprint.generate()
    mesher = MeshGenerator(background_lc=100)
    mesher.generate(clean_polys, clean_lines, clean_pts)

    tessellator = VoronoiTessellator(mesher, blueprint, clip_to_boundary=True)
    grid_gdf = tessellator.generate()
    if grid_gdf.empty:
        raise RuntimeError("Basic Usage generated an empty Voronoi grid")
    return grid_gdf


def plot_grid(grid_gdf, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    grid_gdf.plot(ax=ax, facecolor="whitesmoke", edgecolor="0.4", linewidth=0.3)
    ax.plot(*FAULT_LINE.xy, color="tab:red", linewidth=1.5, label="Fault-1 (barrier)")
    ax.plot(WELL_POINT.x, WELL_POINT.y, "o", color="tab:blue", markersize=4, label="Well-A")
    ax.set_aspect("equal")
    ax.set_title(f"Basic Usage: {len(grid_gdf)} Voronoi cells")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", metavar="PATH", help="save a figure of the grid to PATH")
    args = parser.parse_args()
    grid = main()
    print(f"Generated {len(grid)} Voronoi cells")
    if args.plot:
        plot_grid(grid, args.plot)
        print(f"Saved figure to {args.plot}")
