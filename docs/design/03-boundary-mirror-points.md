# Milestone 3 — Boundary mirror points (opt-in)

**Status:** done · **Risk:** medium · **Behavior change:** none by default
**Back to** [design notes](README.md)

## Outcome

Implemented as `VoronoiTessellator(boundary_centering="inset_mirror")`. Each
non-corner boundary node moves inward by `boundary_inset_fraction` x its
nearest boundary-node spacing, and a ghost goes at the mirror position
outside. Nodes at ring vertices sharper than `boundary_corner_angle` (135
degrees), nodes whose ghost would land back inside the domain, and nodes
whose ghost's Voronoi cell would cover part of the domain stay in place and
are clipped (`boundary_centering` is `"clip"` in their row). This insets the generator rather than leaving it on the boundary, so
the plan below (a ghost for a node that stays on the edge) changed during
implementation.

The inset is what decides whether the mode helps. A boundary cell reaches
from the boundary to about halfway to the first interior row, so its generator
sits near its centroid when the inset is about a third of that row's depth.
Gmsh places the first row about 0.87 x the boundary spacing inside
(near-equilateral triangles), which puts the best inset at 0.2-0.3 x spacing.
The first default, 0.5, overshot: boundary nodes ended up closer to the
interior row than to each other, and orthogonality got worse than clip. The
default is now 0.25.

Median centroid-to-centroid `ortho_error` (degrees) of connections touching
the boundary, `boundary_connectivity_report(grid, domain, center="centroid")`,
`background_lc=20`:

| Mesh | clip | inset 0.25 | inset 0.5 |
|------|------|------------|-----------|
| box(0, 0, 200, 200) | 6.36 | 1.39 | 12.02 |
| same box, well at (25, 25), resolution 1 | 6.42 | 1.37 | 8.61 |
| pentagon | 5.29 | 1.66 | 10.69 |
| `examples/comprehensive_demo.ipynb` model | 6.85 | 1.28 | 8.29 |

Limits:

- Corner nodes are not moved, so the worst connections (about 12 degrees on
  the box and pentagon) sit next to corners and are the same as with clip.
  Away from corners the maximum drops from about 11 to 5 degrees.
- The best fraction depends on the depth of the first interior row. On the
  synthetic staggered lattice in
  `tests/test_voronoi_tessellator.py`, where that row is one boundary spacing
  in, 0.3 does slightly better than 0.25. Both beat clip there.
- `tests/test_integration_gmsh.py::test_inset_mirror_improves_boundary_orthogonality_on_gmsh_mesh`
  checks the improvement on real Gmsh meshes.
- At a reflex vertex (the boundary bends into the domain) the ghost's face
  follows the averaged tangent, which cuts into the domain on both sides of
  the vertex. Ghost cells are discarded, so that strip was left uncovered and
  MODFLOW 6 saw disconnected faces
  ([#71](https://github.com/rhugman/vorflow/issues/71)). The tessellator now
  builds the diagram once with the ghosts, drops every ghost whose cell covers
  more than `GHOST_LEAK_FRACTION` x inset^2 of the domain, and puts its node
  back on the boundary; it repeats until no ghost cell leaks. Mirroring across
  both segments at the vertex does not help, because both reflections land
  inside the domain. Leaving the node inset without a ghost also covers the
  domain, but gives worse skewness on finely faceted curves: p95 skewness of
  faces touching a boundary cell on a 720-sided circular hole is 2.29 inset,
  1.58 clipped (plain clip: 1.57).
- Every hole vertex of a polygonised circle is reflex, so a circular hole's
  nodes all end up clipped (a 64-sided hole: 64 clipped, 260 nodes elsewhere
  mirrored).

The rest of this document is the original plan.

## Goal

Give boundary Voronoi cells properly-centered generators and perpendicular
boundary faces, **adapting to variable cell sizes** along the domain edge. This
is the principled replacement for `gmshflow`'s empirical global `cs_dom/3`
pre-buffer.

The current `gpd.clip`-to-domain stays the **default**; mirror points are
strictly **opt-in**.

## Why

When a boundary generator node sits exactly on the domain edge, `gpd.clip` chops
its cell into a half-cell with the generator on a face — poor for CVFD
(cell center on the face). A single global outward buffer (`cs_dom/3`) softens
this but is wrong when cell size varies along the boundary (too much where cells
are small, too little where large).

**Mirror/reflection points** fix this in a size-adaptive way: for each boundary
generator node, add a ghost node mirrored just outside the local boundary. The
Voronoi edge between the node and its mirror then lands exactly on the boundary,
producing a centered boundary cell with a clean perpendicular face. Because each
node is mirrored by its **own local offset**, variable sizes are handled
automatically.

`vorflow` already densifies and embeds the domain boundary, so the boundary nodes
are known and mirroring is tractable.

## Files to touch

- `src/vorflow/tessellator.py` — mirror-point generation + integration with the
  raw Voronoi build (`_build_raw_voronoi` adds the 4 far ghost nodes today; mirror
  points are an additional, opt-in ghost set).
- `src/vorflow/engine.py` — expose boundary generator nodes / boundary polyline if
  not already available to the tessellator.
- `tests/test_voronoi_tessellator.py` — extend.

## Detail

Add an opt-in flag (e.g. `boundary_mirror: bool = False`, plus tuning params).
When enabled:

1. Identify boundary generator nodes and the local boundary segment/normal at
   each (from the embedded, densified domain boundary).
2. For each boundary node, place a reflected ghost node across the local boundary
   at an offset clamped to the **local node spacing** (so the Voronoi face lands
   on the boundary).
3. Build the Voronoi with these extra ghosts; the mirror ghosts' cells are
   discarded (like the existing far ghost nodes).

### Corner-aware handling (important)

The team's concern is **weird angles among adjacent boundary cells**, not interior
cells. Mirror reflection misbehaves at sharp convex/concave corners and where
boundary spacing is irregular (mirrors can overlap or skew). So:

- Detect sharp corners (turn angle threshold) and **limit or skip** mirrors there.
- **Clamp** the mirror offset to local node spacing.
- Optionally fall back to plain clipping in flagged corner neighborhoods.

### Diagnostics

Reuse Milestone 1's `build_connectivity` helper, restricted to boundary cells, to
report the boundary-cell **angle distribution before vs after** enabling mirror
points — so the effect is measurable and corner regressions are visible.

## Verification

- `tests/test_voronoi_tessellator.py`:
  - Default (mirror off) reproduces current output exactly.
  - On a straight-edged domain with uniform spacing, boundary-cell orthogonality
    improves toward 90° with mirror on.
  - On a graded-size boundary, assert mirror offsets scale with local spacing.
  - On a domain with a sharp corner, assert no degenerate/overlapping boundary
    cells are produced (corner handling regression test).
- Visual check via an example: overlay boundary cells with mirror on/off and
  inspect the angle-distribution diagnostic.
