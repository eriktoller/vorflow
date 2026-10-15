# Milestone 5 — Triangular-grid output option

**Status:** planned · **Risk:** low–medium · **Behavior change:** opt-in
**Back to** [design notes](README.md)

## Goal

Allow the final product to be a plain **triangular grid** (the gmsh 2D triangle
mesh) instead of the Voronoi dual — useful when a triangular FE/FV grid is wanted
directly. Captured now (rather than later) because it **interacts with the quad
recombination** of Milestone 4 and we want to avoid an architectural clash.

## Why

The current pipeline always produces Voronoi cells via `VoronoiTessellator`. Some
workflows want the triangular mesh itself as the deliverable. The MeshGenerator
already extracts node coordinates; this path additionally needs the **triangle
element connectivity**.

## Approach to explore (decide during execution)

Two viable shapes — pick during implementation:

- **(a) A flag on the existing pipeline** — e.g. `tessellate=False` or
  `output="triangular"` — that short-circuits before the Voronoi step and returns
  a triangle-cell GeoDataFrame.
- **(b) A separate lightweight exporter** that reads `gmsh.model.mesh` triangle
  elements + node coordinates and builds the triangle-cell GeoDataFrame, bypassing
  `VoronoiTessellator` entirely.

Both need: `gmsh.model.mesh.getElements(dim=2)` for triangle connectivity +
`getNodes` for coordinates, assembled into shapely triangles → GeoDataFrame
(carrying zone ids the same way the Voronoi path does).

## Files to touch

- `src/vorflow/engine.py` and/or `src/vorflow/tessellator.py` — depending on
  approach (a) vs (b).
- `tests/` — new tests for the triangular path.

## Architectural watch-outs (the reason this is captured now)

- **Clash with Milestone 4:** quad recombination / transfinite buffers produce a
  **mixed** tri/quad mesh. A "triangular-only" mode must either:
  - disable recombination (no structured buffers in this mode), **or**
  - explicitly support a mixed-element grid (triangles + quads) in the output.
  Define and document the chosen behavior; design jointly with Milestone 4.
- **Skip Voronoi-only steps:** mirror points (Milestone 3) are Voronoi-specific
  and must be **skipped** on the triangular path. Active-domain filtering from
  Milestone 2 remains an external GeoPandas workflow and can be applied to any
  polygonal grid after export.
- **Quality reporting differs:** the triangular grid uses Milestone 1's
  `get_triangular_quality()` (element metrics), **not** the Voronoi
  connectivity/orthogonality/skewness report.

## Keep it additive

This must be opt-in and additive so the default Voronoi pipeline is unchanged.

## Verification

- New test: with the triangular option enabled, the output GeoDataFrame contains
  triangle polygons matching the gmsh element count, with zone ids assigned.
- With Milestone 4 buffers present, assert the documented mixed-vs-disabled
  behavior holds (no silent loss of quad regions, or a clear error/te disable).
- Default Voronoi output unchanged when the option is off.
