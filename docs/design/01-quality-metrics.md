# Milestone 1 — Quality metrics

**Status:** planned · **Risk:** low · **Behavior change:** additive only
**Back to** [design notes](README.md)

## Goal

Give `vorflow` three quality capabilities that `gmshflow` had (some more complete
than others):

1. **Triangular-element quality** on the gmsh mesh (gmsh's built-in metrics).
2. A **completed CVFD skewness** metric for the Voronoi grid (gmshflow's version
   is buggy).
3. A **cell-pair connectivity report** (per shared face: angle + skewness) for
   diagnostics/QA export.

All additions reuse `vorflow`'s **generator-point** based pair logic, which is
the correct CVFD center — not `gmshflow`'s centroid-Delaunay reconstruction.

## Why

- `vorflow`'s `utils.py` already has `calculate_orthogonality`,
  `calculate_mesh_quality` (compactness/convexity/drift), and
  `summarize_quality`, but **no triangular-element quality** and **no skewness**.
- `gmshflow.calculate_cvfd_quality` (`../gmshflow/src/gmshflow.py:20-92`) computes
  angle + skewness but:
  - rebuilds a Delaunay from polygon **centroids** (an approximation — the
    correct connector uses the generator points), and
  - has a bug at `gmshflow.py:50-51` where `voronoi_index_1` and
    `voronoi_index_2` are computed identically, so every "pair" is a cell with
    itself.
- `gmshflow.GmshModel.get_triangular_quality` (`gmshflow.py:211-250`) is a clean,
  useful wrapper worth porting as-is.

## Files to touch

- `src/vorflow/engine.py` — add `MeshGenerator.get_triangular_quality()`.
- `src/vorflow/utils.py` — add new connectivity/skewness functions; **leave
  `calculate_orthogonality` untouched**.
- `examples/field_capabilities_example.py` — demonstrate the new reports.
- `tests/test_quality_metrics.py` (new) — unit tests.

## Detail

### 1. `MeshGenerator.get_triangular_quality()`

Port of `gmshflow.py:211-250`. Wrap `gmsh.model.mesh.getElementQualities` over
the 2D elements and return a `pandas.DataFrame` with one row per element and
columns: `minSICN`, `minDetJac`, `maxDetJac`, `minSJ`, `minSIGE`, `gamma`,
`innerRadius`, `outerRadius`, `minIsotropy`, `angleShape`, `minEdge`, `maxEdge`.

Constraints:
- Must run **while the gmsh model is still live** — i.e. after
  `MeshGenerator.generate()` and before gmsh is finalized. Document this in the
  docstring (mirrors how nodes are extracted today).
- Get element tags via `gmsh.model.mesh.getElements(dim=2)` and pass the
  triangle tag array to `getElementQualities(tags, measure)` per measure name.

### 2. New Voronoi connectivity + skewness functions (in `utils.py`)

**Do not change `calculate_orthogonality`.** It keeps the same signature and
returns the same result (backward-compatible). Add new, separate functions:

- `build_connectivity(gdf)` → GeoDataFrame of neighbor pairs with columns
  `cell_id_1`, `cell_id_2`, `shared_edge` (geometry), `angle`, `skewness`.
  - Find touching pairs and their shared edge using the same robust approach as
    `calculate_orthogonality` (spatial self-join on touching cells, then
    `intersection` to get the shared edge). Implement it within the new function
    so the existing function's behavior is guaranteed untouched.
  - **Connector = line between the two generator points** (`x`, `y` columns that
    `VoronoiTessellator` already stores), not centroids.
  - `angle` = orthogonality error in degrees between the connector and the shared
    edge (the connector should be perpendicular to the face).
  - `skewness` = fractional position along the connector where it crosses the
    shared face, ideal `0.5`. Compute the connector∩face intersection; if empty,
    project the face midpoint onto the connector (as gmshflow did at
    `gmshflow.py:75-89`) — but using generator points and correct pair indices.
- A public **connectivity report** wrapper that returns the per-pair GeoDataFrame
  for export (supersedes `gmshflow.calculate_cvfd_quality`).
- `calculate_mesh_quality` / `summarize_quality` gain an **optional** `skewness`
  column (max `|skewness − 0.5|` per cell) + summary line, behind a flag so the
  default output is unchanged.

### Note on "angle between centroids"

What the team referred to as "angle between centroids" is exactly orthogonality:
the cell-center connector should be perpendicular to the shared face. `vorflow`
already measures this from **generator points** (the proper CVFD cell center), so
no change is needed there — `build_connectivity` simply exposes the same quantity
per pair, plus skewness.

## Verification

- `tests/test_quality_metrics.py`:
  - Build a small known mesh; assert `get_triangular_quality()` returns one row
    per triangle with all expected columns and values in valid ranges
    (`0 ≤ gamma ≤ 1`, `minSICN ∈ [-1, 1]`, etc.).
  - On a regular grid of generators, assert orthogonality angle ≈ 90° and
    skewness ≈ 0.5; assert `build_connectivity` produces **distinct** cell ids per
    pair (regression against the gmshflow index bug).
  - Assert `calculate_orthogonality` output is byte-for-byte unchanged vs the
    current implementation (golden test).
- Extend `examples/field_capabilities_example.py` to print a triangular-quality
  summary and a skewness summary.
