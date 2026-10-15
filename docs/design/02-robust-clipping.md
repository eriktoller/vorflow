# Milestone 2 - Active-Domain Filtering Workflow

**Status:** replaced by example workflow | **Risk:** low | **Behavior change:** none
**Back to** [design notes](README.md)

## Decision

Do **not** add `min_cell_overlap`, `boundary_buffer`, or `active_domain` options
to `VoronoiTessellator` for now.

The better default is to keep core tessellation simple:

1. Build a mesh on the geometry the user wants as the mesh envelope.
2. Generate the Voronoi grid with the existing clipping behavior.
3. If the model's active domain is smaller than the mesh envelope, select or tag
   active cells externally with GeoPandas.

This keeps cell removal visible to the user instead of hiding it inside a
tessellator parameter.

## Why

Boundary sliver filtering is easy to do externally and can be risky if it happens
silently inside the core API. A low-overlap cell might be a real modeling choice,
not just a bad artifact. Removing it automatically can change the model topology
without enough user inspection.

The larger-envelope workflow is also a different use case from clipping cleanup.
Users may deliberately mesh a convex hull, square extent, or buffered envelope so
the cells around the real active model boundary stay healthy. In that case the
right operation is not clipping cells to the active boundary; it is selecting or
tagging active cells after grid generation.

## Example Workflow

Add `examples/active_domain_quality_example.py` showing:

- a baseline mesh whose mesh boundary equals the active model boundary;
- an improved mesh generated on a larger envelope;
- active-cell selection from the larger-envelope grid by centroid and overlap;
- centroid-based connectivity/quality comparisons for the two outputs.

The example should use ordinary GeoPandas operations such as:

```python
active_overlap = grid.geometry.intersection(active_domain).area / grid.geometry.area
grid["active_overlap_ratio"] = active_overlap
grid["is_active_by_overlap"] = active_overlap >= 0.5
grid["is_active_by_centroid"] = grid.geometry.centroid.within(active_domain)
active_grid = grid[grid["is_active_by_centroid"]].copy()
```

## Verification

- Run the new example with `MPLBACKEND=Agg` to confirm it executes headlessly.
- Keep full `pytest` green.
- No new unit tests are required for a core API because this milestone adds no
  core API.

## Future Revisit Criteria

Promote active-domain filtering into core code only if the external workflow
becomes repetitive, error-prone, or needs library-managed metadata for downstream
export. Until then, prefer transparent GeoPandas selection in user code and
examples.
