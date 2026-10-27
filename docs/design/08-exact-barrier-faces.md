# Milestone 8 — Exact barrier faces

**Status:** done (after 0.3.0, unreleased) · **Risk:** high · **Behavior change:** every model with barriers
**Back to** [design notes](README.md)

## Outcome

Barriers (`add_line(..., is_barrier=True)`) are now meshed so that every
barrier segment is exactly one Voronoi cell face. The faces are placed by the
empty-circle rule of the UPR method (Berge, Klemetsdal and Lie, 2019,
*Computational Geosciences* 23:169-188,
<https://doi.org/10.1007/s10596-018-9790-0>), before Gmsh runs. On the 73
barrier benchmark cases no cell is cut and no two cells share a bent face.
The orthogonality error on every barrier face is 0. Where the construction
cannot be made valid, `MeshGenerator.generate()` raises
`vorflow.BarrierMeshError` instead of returning a damaged grid.

This replaces the straddle pairs of 0.3.0 and closes
[#67](https://github.com/rhugman/vorflow/issues/67) (cut cells at barrier
corners). It supersedes [#69](https://github.com/rhugman/vorflow/pull/69),
which placed straddle pairs around each corner and documented the cases that
still left cut cells.

## Why

Up to 0.3.0, vorflow placed pairs of mesh nodes on either side of a barrier,
about one `resolution` apart. The tessellator then cut any cell that still
reached across the line. A cut left one of two defects. A **cut cell** has no
mesh node of its own, so its centre is its centroid and its connections are
not at right angles to its faces. A **face of several edges** is a bent face
shared by two cells, left where a small cut piece was merged into a
neighbour. MODFLOW 6 DISV takes the connection width from the first edge of a
shared face only, so it under-counts the flow area there.

These defects appeared at short legs between corners, on finely drawn curves,
at junctions and crossings of barriers, next to standard lines crossing near a
corner and along polygon edges. On two benchmark cases the flow through an HFB
on the barrier faces was 40-60% lower than on a finer reference grid
(fine_curve 0.14 against 0.37, real_faults 14.3 against 24.3, in units of
1e-4). A finer `resolution` reduced the defects but did not remove them.

## Method

The code is in the private package `src/vorflow/_barrier/`. It is pure
geometry (NumPy, SciPy, Shapely) and makes no Gmsh calls.

**The empty-circle rule.** Each barrier is cut into segments at points we
refer to as 1D nodes. Each 1D node gets a circle. The two circles of a
segment's end nodes cross at two points, which are mirror images across the
segment. A generator (site) goes at each point. If no other generator lies
inside any circle, each segment is exactly the Voronoi face between its two
sites, whatever the other generators are. The sites are therefore fixed in
closed form, and the rule is checked before meshing (`validate.py`).

**Nodes with several legs** (`network.py`, `rung.py`). At a corner, junction,
crossing, boundary end or standard-line crossing, the first site of each leg
sits on the node's circle at an angle from the leg. Neighbouring legs then
either share one site or keep a gap between their sites. In a wedge narrower
than 60 degrees, a row of cells is laid across the wedge, with sites on its
bisector. A spike sharper than `sharp_angle` (18 degrees by default) loses the
last 0.45 x `resolution` of each leg, or 0.3 of the leg if that is shorter. The
tip then lies inside regular cells instead of between long thin ones, and it
is reported.

**Face lengths** (`sizing.py`). The face length along the barrier follows
vorflow's size field, read before Gmsh runs through `MeshField.python_size()`.
It is capped by the distance to other barrier parts, to the domain boundary,
to point features and to standard lines, zone edges and quad-buffer strips,
so that no circle reaches them. A line that crosses a barrier is not capped
within 7 face lengths of the crossing, which the crossing's own sites handle.
It is then graded along the whole network at once (`barrier_gradation`, 0.3
by default), across junctions. A point feature on or next to a barrier
(closer than about 0.05 x `resolution`) raises `BarrierMeshError` naming the
point and the barrier, since a cell centre cannot sit on a barrier face.

**Narrow gaps** (`corridor.py`). Two nearly parallel barriers closer than
1.2 x the face length share one row of cells, with sites on the midline. A
barrier closer than 0.6 x the face length to the domain boundary gets one row
of cells too, with sites on the boundary. Without this, the face lengths would
be capped by the gap and the cells there would be small and thin.

**Repair and error** (`pipeline.py`). Where the check fails, the faces are
halved locally. If the construction is still invalid after a few rounds, the
plan raises `BarrierMeshError`, a `ValueError` naming the place and suggesting
`simplify_tolerance` or a finer `resolution`.

**Meshing** (`frames.py`, `polish.py`, `engine.py`). Gmsh meshes a copy of the
clean frames in which each barrier is a size field only, and the sites are
embedded points. Afterwards the mesh nodes inside the circles are removed
(feature nodes never are). A removed node of a standard line, zone edge or
quad-buffer strip away from where it crosses a barrier is reported in
`barrier_report["fixed_nodes"]`, with a warning. The free nodes are then moved
by Lloyd steps around the fixed sites, kept out of the circles
(`barrier_polish_iterations`, 20 by default); a step pushed out of a circle is
checked again against the embedded lines. The sites join
`MeshGenerator.nodes`.

**Tessellation.** The tessellator's barrier mirror generators and cell cutting
stay as a safety net. They found nothing to do on any of the 73 cases. If they
are ever needed on an exact barrier, `generate()` warns. With
`vorflow.tessellator.STRICT_BARRIERS = True`, as in the test suite, it raises
instead. Quad buffers on barriers still go through this path.

## Line cleaning

`ConceptualMesh.generate()` cleans the barrier lines before meshing
(`_barrier/lines.py`), in a local frame so that UTM-sized coordinates behave
like small ones:

1. An end within `max(connectivity_tolerance, simplify_tolerance)` of another
   barrier, a barrier vertex or the domain boundary is snapped onto it. A
   short overshoot past another barrier is trimmed back. A branch that
   collapses onto the barrier it starts on is dropped as a stub.
2. A part lying on the domain boundary (within the same distance) is dropped
   with a warning: a barrier there separates nothing. A barrier lying wholly
   on the boundary leaves no row in `clean_lines`.
3. Junctions and crossings with other barriers and with standard lines become
   shared vertices.
4. With `simplify_tolerance` above 0, the line is replaced by the fewest
   straight pieces between its own vertices that stay within the tolerance.
   Junctions, crossings and ends are kept. The old per-line Douglas-Peucker
   pass ran before junctions were joined and could pull a splay off its
   fault.
5. A vertex closer than 0.1 x `resolution` to its neighbour is dropped as
   digitising noise, with a warning, even when `simplify_tolerance` is 0.

Barrier lines are no longer densified. Each change is recorded and passed on
to `MeshGenerator.barrier_report`.

## Public API

- `MeshGenerator(barrier_gradation=0.3, sharp_angle=18, barrier_polish_iterations=20)`.
  These are mesh-wide options because faces are graded, and spikes found,
  across lines.
- `ConceptualMesh(keep_close_rings=False)`: keep a `hex_ring` next to a
  barrier by making the barrier faces there shorter.
- After `generate()`: `MeshGenerator.barrier_report` (`snaps`, `noise`,
  `stubs`, `tips`, `on_boundary`, `rings`, `fixed_nodes`, `corridors`, each
  entry with its place, plus `repair_rounds`, `sites`, `removed_nodes` and
  `polish`), `barrier_lines`, `barrier_circles` and `node_is_site`. Spike
  tips, stubs, snapped ends and removed fixed-line nodes each raise one
  `UserWarning` with their count.
- `vorflow.BarrierMeshError(ValueError)`.
- `add_line(straddle_width=...)` is deprecated with a `FutureWarning`, and
  its value is ignored. The internal `vorflow._straddle` module was removed.

## Results

The benchmark has three sets: 19 design cases, 43 held-out cases and 11
adversarial cases. Each case was meshed with 0.3.0 and with the new code.
Compactness is a cell's area relative to a circle of the same perimeter; it
is about 0.8 for a regular Voronoi cell. The cell ratio is the number of cells
near the barrier divided by the number when the same line is meshed as a
standard line.

| set | cut cells | faces of several edges | mean worst compactness next to barriers | median cell ratio |
|---|---|---|---|---|
| design (19) | 48 → 0 | 529 → 0 | 0.59 → 0.76 | 1.15 → 0.94 |
| held-out (43) | 27 → 0 | 2692 → 0 | 0.49 → 0.66 | 1.15 → 0.97 |
| adversarial (11) | 3 → 0 | 0 → 0 | 0.55 → 0.73 | 1.18 → 0.93 |

In the MODFLOW 6 check on the design set, the largest head RMSE fell from
0.0192 to 0.0180. Spikes of 10 and 15 degrees in the held-out and adversarial
sets each leave 7.2 m of tip without faces (0.9 x their 8 m resolution), as
intended. These are the only cases where the faces do not cover the whole
barrier.

## Limitations

- **Quad buffers on barriers** (`quad_buffer=True`) still use the mirror
  generators and the cut, so they can leave cut cells and faces of several
  edges. Two strict `xfail` tests in `tests/test_engine_regressions.py` record
  this.
- **Narrow splays next to a digitised vertex.** A splay leaving a fault at
  about 10 degrees, with a fault vertex about 0.2 x `resolution` past the
  junction, raises `BarrierMeshError` at `simplify_tolerance=0`, because the
  faces would be about 0.01 x `resolution` long. A tolerance of
  `resolution / 8` removes the vertex. The same cause leaves clusters of small
  cells at splay junctions on digitised networks that do mesh. A row of cells
  that follows a slightly bent leg would fix both.
- **Near-miss ends at tolerance 0.** An end that stops just short of another
  barrier, but further away than the snap distance, meshes exactly but with a
  cluster of small faces. The error message, and
  the docs, point to `simplify_tolerance`.
- The safety-net code in `tessellator.py` (`_straddled_pieces`,
  `_merge_barrier_fragments`) stays for quad buffers and is to be retired
  once those use exact faces too.

## Verification

- `tests/test_barrier_faces.py` meshes each case end to end and checks the
  properties of the construction: no mirror generators and no cut cells, one
  cell per mesh node, no other node or generator inside an empty circle, no
  face crossing a barrier, one single-edge face per barrier segment running
  from circle centre to circle centre with an orthogonality error below
  1e-6, and faces that cover the cleaned barrier lines less the reported
  tips and stubs. The cases are a straight line, kinks, a zigzag, T, X and Y
  junctions, standard-line crossings and ends, boundary and corner ends and
  a ring. Focused tests cover spike tips, stubs, snapping, noise, corridors,
  options, the report, the deprecation, the errors, the safety net, the
  deviation from the input line, x0.01 and x100 scale, the tessellator's
  Lloyd, short-face and `inset_mirror` steps, standard lines, zone edges and
  quad-buffer strips next to a barrier, barriers on the domain boundary and
  warnings that point at the caller. A `slow` regression set holds the
  held-out and adversarial shapes and a digitised relay-ramp fault network.
  `tests/test_barrier_units.py` checks the helpers without Gmsh.
- When this note was written the full suite gave 664 passed and 2 xfailed.
- `examples/barrier_meshing.ipynb` shows the cases that used to be cut, the
  report and the remaining limits.
