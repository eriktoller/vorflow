#%%
"""Structured quad-buffer example.

Demonstrates the four canonical quad-buffer behaviors (gmshflow parity). The
band/strip width is ``quad_buffer_thickness`` cells, and the feature outline
itself never becomes mesh edges:

- line, thickness=1  -> one quad row straddling the line, no mesh nodes on it:
  the Voronoi FACES align with the line (sharp barrier).
- line, thickness=2  -> two quad rows, a node row lands ON the line: a row of
  ~square Voronoi cells centered on it.
- polygon, thickness=1 -> a quad band straddling the outline: the Voronoi
  faces trace the shapefile shape.
- polygon, thickness=2 -> a ring of ~square cells centered on the outline.

Crossing buffers are a vorflow extension (gmshflow never supported them). When
two quad buffers cross, the higher-priority one stays continuous and only the
other is trimmed at the crossing -- priority is z_order, then finer resolution,
then wider strip, then line over polygon, then insertion order. A local size
field keeps the small trimmed gap meshed at the feature size instead of jumping
to the background size, so the cell size and quality stay smooth across the
crossing. Here the inner zone (z_order=1) wins over the fault, which is trimmed.

Two more features round out the comparison: a ``tee`` line that ends ON the
fault (a T-junction, trimmed because the fault was added first) and a plain
``barrier`` line (no quad buffer), whose Voronoi faces are placed exactly on
the line by generator pairs rather than by a structured strip -- the default
treatment to compare against the quad buffers.
"""

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, box
from shapely.affinity import translate

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow.utils import build_connectivity, calculate_mesh_quality
from vorflow.fields import GeometricGrowthField

#%%
field = GeometricGrowthField(growth_factor=1.5)
domain = box(0, 0, 12, 22)
fault = LineString([(0, 5), (12, 5)])    # crosses inner_zone's band
drain = LineString([(0, 10), (12, 10)])  # passes between the two zones
tee = LineString([(10, 5), (10, 9)])     # T-junction: ends on the fault
barrier = LineString([(0, 20), (12, 20)])  # exact barrier faces, clear of all bands
inner_zone = Polygon([(4, 3), (8, 3), (8, 7), (4, 7)])
sup_inner_zone = translate(inner_zone, xoff=0, yoff=10)

mesh = ConceptualMesh(crs=None)
mesh.add_polygon(domain, zone_id="domain", resolution=1, densify=False)

# thickness=1: Voronoi faces trace the outline.
mesh.add_polygon(
    inner_zone,
    zone_id="inner-zone",
    resolution=1.0,
    z_order=1,
    densify=True,
    fields=[field],
    quad_buffer=True,
    quad_buffer_thickness=1,
)

# thickness=2: a ring of ~square cells centered on the outline.
mesh.add_polygon(
    sup_inner_zone,
    zone_id="sup-inner-zone",
    resolution=1.0,
    z_order=1,
    densify=True,
    fields=[field],
    quad_buffer=True,
    quad_buffer_thickness=2,
)

# thickness=1: sharp Voronoi faces on the line (barrier-like).
mesh.add_line(
    fault,
    line_id="fault-buffer",
    resolution=1.0,
    quad_buffer=True,
    quad_buffer_thickness=1,
    fields=[field],
)

# thickness=2: square cells centered on the line.
mesh.add_line(
    drain,
    line_id="drain-buffer",
    resolution=1.0,
    quad_buffer=True,
    quad_buffer_thickness=2,
    fields=[field],
)

# T-junction: this quad buffer ends on the fault. With equal priority the fault
# (added first) wins, so the tee is trimmed where the two meet.
mesh.add_line(
    tee,
    line_id="tee-buffer",
    resolution=1.0,
    quad_buffer=True,
    quad_buffer_thickness=1,
    fields=[field],
)

# Plain barrier (no quad_buffer) for contrast: its Voronoi faces are placed
# exactly on the line by generator pairs rather than by a structured strip.
mesh.add_line(
    barrier,
    line_id="barrier",
    resolution=1.0,
    is_barrier=True,
    fields=[field],
)

clean_polys, clean_lines, clean_points = mesh.generate()

mesher = MeshGenerator(
    background_lc=1.5,
    verbosity=0,
    smoothing_steps=0,
    optimization_cycles=0,
)
mesher.generate(clean_polys, clean_lines, clean_points)

element_grid = mesher.get_element_grid()
quads = mesher.get_element_grid("quads")
triangles = mesher.get_element_grid("triangles")

print(f"2D elements: {len(element_grid)}")
print(f"  triangles: {len(triangles)}")
print(f"  quads:     {len(quads)}")


#%%
tessellator = VoronoiTessellator(mesher, mesh, clip_to_boundary=True)
voronoi_grid = tessellator.generate()

# Behavior diagnostics: count cells centered on each feature (thickness=2
# should have a full row, thickness=1 none) and, for the thickness=1 zone,
# cells straddling the outline (faces should trace it). The inner zone wins
# the crossing with the fault (z_order=1), so it stays continuous and its
# straddle count is ~0 -- the fault is the one trimmed there instead.
centroids = voronoi_grid.geometry.centroid
def centered_cells(geom, tol=0.05):
    return voronoi_grid[centroids.distance(geom) < tol]

fault_centered = centered_cells(fault)
drain_centered = centered_cells(drain)
inner_centered = centered_cells(inner_zone.exterior)
sup_centered = centered_cells(sup_inner_zone.exterior)
inner_frac = voronoi_grid.geometry.intersection(inner_zone).area / voronoi_grid.geometry.area
inner_crossing = ((inner_frac > 0.03) & (inner_frac < 0.97)).sum()

print("\nBehavior checks (cells centered on each feature):")
print(f"  fault  t=1: {len(fault_centered)} (expect 0 - faces align with the line)")
print(f"  drain  t=2: {len(drain_centered)} ~square cells, "
      f"area mean={drain_centered.geometry.area.mean():.2f}")
print(f"  inner  t=1: {len(inner_centered)} on outline, "
      f"{inner_crossing} cells straddle it (~0 - the zone wins, faces trace it)")
print(f"  sup    t=2: {len(sup_centered)} ~square cells on outline, "
      f"area mean={sup_centered.geometry.area.mean():.2f}")

connectivity = build_connectivity(voronoi_grid, center="centroid")
quality = calculate_mesh_quality(
    voronoi_grid,
    calc_ortho=True,
    calc_skewness=True,
    connectivity=connectivity,
)

# Note: calculate_mesh_quality aggregates per cell with the MAX over the
# cell's faces, so these per-cell numbers read higher than the face-level
# distribution. In this small, gradient-heavy domain boundary cells also
# dominate the stats; the faces along the buffered features are the best ones.
print("\nPer-cell quality (worst face per cell):")
print(quality[["compactness", "drift_ratio", "ortho_error", "skewness"]].describe().T)
print(f"\nFace-level ortho_error mean: {connectivity['ortho_error'].mean():.2f} deg "
      f"(per-cell worst-face mean: {quality['ortho_error'].mean():.2f} deg)")


#%%
fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), constrained_layout=True)

triangles.plot(ax=axes[0], facecolor="#d7e3f4", edgecolor="#8aa0bd", linewidth=0.35)
quads.plot(ax=axes[0], facecolor="#f3c97a", edgecolor="#9b6b1f", linewidth=0.55)
axes[0].set_title("Element grid: quad strips/bands")

voronoi_grid.plot(
    column="zone_id", ax=axes[1], categorical=True, legend=True,
    cmap="Pastel2", edgecolor="#666666", linewidth=0.4,
)
axes[1].set_title("Voronoi cells: faces trace the t=1 outlines,\nsquares sit on the t=2 features")

quality.plot(
    column="ortho_error",
    ax=axes[2],
    cmap="magma_r",
    edgecolor="#555555",
    linewidth=0.25,
    legend=True,
)
axes[2].set_title("Centroid orthogonality error (deg)")

for ax in axes:
    ax.plot(*fault.xy, color="black", linewidth=1.2)
    ax.plot(*drain.xy, color="black", linewidth=1.2, linestyle="--")
    ax.plot(*tee.xy, color="black", linewidth=1.2)
    ax.plot(*barrier.xy, color="crimson", linewidth=1.2, linestyle=":")
    ax.plot(*inner_zone.exterior.xy, color="black", linewidth=1.0)
    ax.plot(*sup_inner_zone.exterior.xy, color="black", linewidth=1.0, linestyle="--")
    ax.set_aspect("equal")
    ax.axis("off")

plt.show()

# %%
