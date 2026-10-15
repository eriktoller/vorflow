# Milestone 4 — Structured-quad transfinite buffer (OCC)

**Status:** planned · **Risk:** high · **Behavior change:** opt-in
**Back to** [design notes](README.md)

## Goal

Port `gmshflow`'s transfinite **structured-quad buffer-surface** system for
**line and polygon** features into `vorflow`'s OCC pipeline, as an opt-in robust
alternative to the current straddle approach. It produces sharp, regular boundary
alignment of Voronoi cells along a feature.

The existing lightweight **straddle** path stays as-is; this buffer is the
heavier, higher-fidelity option. **Keep both.**

## Why

- `gmshflow` builds a thin transfinite quad strip hugging each feature:
  - Lines: `create_surfacegrid_from_buffer_line` (`../gmshflow/src/gmshflow.py:992-1123`).
  - Polygons: `create_surfacegrid_from_buffer_poly` (`gmshflow.py:776-868`).
  - `offset_curve(±cs_thick*cs/2)` on each side, build curve loops, a plane
    surface, set transfinite curves/surface, `setRecombine`, mesh algorithm 8
    (frontal-Delaunay for quads).
  - `cs_thick = 1` ⇒ a sharp boundary (two node rows `cs/2` apart ⇒ Voronoi edge
    lands exactly on the feature); `cs_thick = 2` ⇒ a quasi-rectangular zone.
  - Asserts `cs_thick ∈ {1, 2}` because thicker strips give bad quads.
- `vorflow`'s straddle (in `src/vorflow/engine.py`) is the simpler dual-row of
  offset points. It is conceptually the same idea but without structured control
  of along-feature spacing.

## Scope (agreed)

- **Lines and polygons**, `cs_thick ∈ {1, 2}` only.
- **Per-feature `cs` preserved** (each feature row carries its own cell size;
  transfinite divisions computed from it). Graded widening **away** from the
  feature is left to the field system (`fields.py`) — do **not** extend
  `cs_thick > 2`.
- Straddle remains the lightweight default path; the buffer is opt-in.

## Files to touch

- `src/vorflow/buffer.py` (new) **or** methods on `MeshGenerator` in
  `src/vorflow/engine.py`.
- `src/vorflow/engine.py` — reuse the existing post-`removeAllDuplicates` /
  `healShapes` **coordinate/bbox tag-remap** machinery; factor out the straddle's
  corridor-protection logic (below).
- `tests/test_buffer.py` (new).

## Detail

### OCC integration (the hard part)

`vorflow` runs `occ.fragment` + `removeAllDuplicates`, which **re-tags and can
split surfaces** — this would destroy a 4-corner transfinite structure if applied
naively. Approach:

1. Create the buffer surfaces (offset curves → loops → plane surfaces).
2. Run the normal `occ.fragment` + `removeAllDuplicates` (+ optional
   `healShapes`).
3. **Relocate the buffer surfaces by coordinate/bbox match** after fragmentation,
   reusing the same remap logic `engine.py` already uses to track killed/remapped
   tags.
4. **Apply transfinite + recombine to the relocated surfaces** (post-fragment),
   so re-tagging cannot break the structured constraint.

`setTransfiniteCurve/Surface` and `setRecombine` are mesh-level ops and work on
OCC surfaces — the issue is purely ordering vs fragmentation.

### Shared feature-corridor protection (improvement over gmshflow)

The straddle path already **differences-out other features within a corridor**
around the barrier so nothing else injects nodes into that zone. The transfinite
strip needs the **same** protection to mesh cleanly. So:

- Factor that corridor-protection step out of the straddle code into a **shared
  helper** used by **both** straddle and the structured buffer.
- This is a genuine improvement over `gmshflow`, which had no such protection (it
  only warned that feature lines must not intersect — `gmshflow.py:1022`).

### offset_curve artifacts

`offset_curve` can occasionally emit a `MultiLineString` (GEOS). `gmshflow` worked
around this with `merge_many_multilinestring_into_one_linestring`
(`gmshflow.py:94-125`). Shapely 2.x is more robust but can still split on
self-intersections. **Only** add a localized merge if the port actually
encounters it — do not vendor it preemptively.

## Interaction with other milestones

- **Milestone 5 (triangular-grid output) must be designed together with this
  one:** quad recombination here produces a **mixed** tri/quad mesh, which a
  "triangular-only" mode must either disable or explicitly support. Define the
  behavior jointly.

## Verification

- `tests/test_buffer.py`:
  - A single straight line with `cs_thick=1` yields a Voronoi boundary that
    follows the line within tolerance (sharp alignment).
  - `cs_thick=2` yields a quasi-rectangular band of cells along the feature.
  - A polygon feature yields a clean buffered band around its boundary.
  - With two **crossing** features, the shared corridor-protection step prevents
    node injection / meshing failure (regression vs the gmshflow non-intersecting
    assumption).
  - Transfinite/recombine survive `occ.fragment` (surfaces correctly relocated
    and structured after fragmentation).
- Example: add a structured buffer on a line and a polygon to an example script
  and visually confirm alignment + quad structure.
