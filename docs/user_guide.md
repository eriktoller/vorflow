# User guide

```{include} ../README.md
:start-after: <!-- docs:installation-start -->
:end-before: <!-- docs:usage-end -->
```

## Checking inputs before meshing

`ConceptualMesh.check()` looks for input geometry that tends to produce badly
shaped or very small Voronoi cells, and reports it before you mesh. It needs
no Gmsh and takes a fraction of a second, so run it whenever the inputs
change:

```python
findings = blueprint.check(background_lc=50.0)
print(findings[["feature_id", "other_id", "kind", "severity", "ratio"]])
findings.to_file("input_findings.gpkg")  # inspect them in a GIS
```

Pass the `background_lc` you will give `MeshGenerator`. Without it, the
coarsest feature `resolution` stands in, so defects in coarse areas far from
any feature may go unreported.

Each defect is measured against the local mesh size `h` at its location: the
size the mesh will aim for there, from `background_lc` and each feature's
`resolution` and `growth_factor`. `ratio = distance / h` sets the severity.
An `error` very likely gives bad cells; a `warning` may. The result is a
GeoDataFrame with one row per defect site, errors first. `feature_id` and
`other_id` are the `zone_id`, `line_id` or `point_id` involved, `x`/`y` mark
the narrowest point, and `geometry` shows the segment, gap or polygon part.
It is empty when nothing is found.

| kind | what it means | error if ratio < | warning if ratio < | what to do |
|---|---|---|---|---|
| `short_segment` | two consecutive vertices of a polygon or line are very close | 0.1 | 0.25 | Remove the vertex, or set `simplify_tolerance` on the feature. |
| `close_vertices` | two mesh points are very close: vertices of different features, a point, or a line crossing a polygon edge just beside a vertex | 0.1 | 0.5 | Move the crossing or the vertex apart, or make them coincide. |
| `line_end_near_miss` | a line end touches nothing but lies close to another feature | 0.5 | 1.0 (inclusive) | Extend the line onto the feature, pull it well back, or raise `connectivity_tolerance` so it snaps. |
| `point_near_miss` | a point lies close to, but not on, another feature | 0.5 | 1.0 (inclusive) | Move the point onto the feature or away from it. |
| `sliver_polygon` | a polygon part, or a strip of one, is narrower than `h`; often a leftover of overlap resolution | 0.5 | 1.0 | Remove the strip, change `z_order`, or snap the polygon outlines together. Below 0.5 the strip may also take the wrong zone. |
| `narrow_gap` | a line runs close to another feature, or to itself, away from where they meet | never | 1.0 (inclusive) | Only one or two cells fit across. Accept it, refine the area, or move the line. |

Snapped features are not defects: a line end or point within
`connectivity_tolerance` of another feature is joined to it by `generate()`
and is not reported.

One defect often trips several checks; a thin strip, for example, also has
short edges. Findings within `2 * h` of each other are treated as one site,
and only the most severe is kept. Its `site_kinds` column lists every kind
found there, so a site may need more than one fix. Run `check()` again after
editing.

What is checked:

- The features as `generate()` cleans them (simplification, overlap
  resolution, snapping, clipping and line junctions), but before
  densification. The vertices densification adds are vorflow's own, so they
  are not reported.
- Embedded features only. Field-only (`embed=False`) features add no mesh
  points or edges, so they are not checked; they still set `h`.
- Barrier, straddle and quad-buffer features as the plain lines and
  polygons they are drawn from. The extra nodes and quad strips the mesher
  builds around them are not modelled. One exception: a barrier end that
  runs alongside another barrier, or along the domain outline, is not a near
  miss. The mesher gives that gap one row of cells, so it is reported as a
  narrow gap.

`check()` changes nothing: the registered features and any earlier
`generate()` result are left as they were. It does not catch every bad cell.
Many come from grading and from clipping at the domain boundary, not from the
inputs; use `vorflow.utils.calculate_mesh_quality` on the finished grid for
those.
