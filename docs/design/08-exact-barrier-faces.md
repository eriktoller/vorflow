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

**Merges** (`parallel.py`, `wedge.py`). Two inputs leave slivers that no
choice of face lengths can fill with regular cells, so the lines are moved a
bounded distance instead, and each move is reported:

- Two barriers closer than 0.25 x `resolution` over a stretch of at least
  one `resolution` are moved onto their midline there. One line of faces
  carries both barriers, and `barrier_lines` lists it under both `line_id`s,
  so the user gives the HFB there the combined properties of both. Where the
  lines part, each leaves the midline at 0.75 x `sharp_angle`. A line moves
  at most half the local gap (`barrier_report["merged"]`).
- Two barrier legs that meet at a junction (a splay, a T-branch, a crossing)
  at less than `sharp_angle` share a stem along their bisector, as long as it
  keeps every point within 0.1 x `resolution` of the lines as drawn, and
  part at 1.5 x `sharp_angle` at its end. A barrier leaving the domain
  boundary or a hole edge that narrowly slides along the boundary instead,
  so it still ends on it. Fans and stars are merged pair by pair, narrowest
  first; two adjacent sectors below 29 degrees are merged too, because the
  sector plan cannot place sites in both. Two legs that run to the same next
  vertex become its chord (`barrier_report["collapsed"]`).

A stretch or a junction where the move would exceed its bound, or would
reach another line, is left as it is.

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
1.2 x the face length, but not close enough to be merged, share one row of
cells, with sites on the midline. A barrier closer than 0.6 x the face length
to the domain boundary gets one row of cells too, with sites on the boundary.
Without this, the face lengths would be capped by the gap and the cells there
would be small and thin. The row's cells are at most 1.5 x the gap long, so a
narrow gap still adds cells: at 0.25 x `resolution` a 220 m pair of barriers
at `resolution` 8 has about twice the cells of one barrier. A row of cells
shorter than 0.5 x `resolution` raises a warning with its cell count.

**Repair and error** (`pipeline.py`). Where the check fails, the faces are
halved locally. If the construction is still invalid after a few rounds, or
the faces would have to be shorter than 0.02 x `resolution`, the plan raises
`BarrierMeshError`, a `ValueError` naming the place, what made the faces that
short there (the size caps record which one bound each face length: another
barrier part, the boundary, a point, a fixed line, closely spaced vertices,
a corridor, or a junction or fold) and what helps for that cause. A
geometric failure inside the construction is raised as `BarrierMeshError`
too, as a bug to report.

**Meshing** (`frames.py`, `polish.py`, `engine.py`). Gmsh meshes a copy of the
clean frames in which each barrier is a size field only, and the sites are
embedded points. Afterwards the mesh nodes inside the circles are removed
(feature nodes never are). A removed node of a standard line, zone edge or
quad-buffer strip away from where it crosses a barrier is reported in
`barrier_report["fixed_nodes"]`, with a warning. The free nodes are then moved
by Lloyd steps around the fixed sites, kept out of the circles
(`barrier_polish_iterations`, 20 by default); a step pushed out of a circle is
checked again against the embedded lines. Only the free nodes within 3 x their
size of a circle move; every other node keeps its Gmsh position bit for bit.
On a 118k-node model this cut the polish from 60 s to 10 s and moved 18k
instead of 110k nodes, with the same barrier-cell compactness. The sites join
`MeshGenerator.nodes`. `get_element_grid()` and `get_triangular_quality()`
describe Gmsh's mesh before this step.

**Tessellation.** The tessellator's barrier mirror generators and cell cutting
stay as a safety net. They found nothing to do on any of the 73 cases. If they
are ever needed on an exact barrier, `generate()` warns. The test suite sets
a private hook so that it raises instead. Quad buffers on barriers still go
through this path.

## Line cleaning

`ConceptualMesh.generate()` cleans the barrier lines before meshing
(`_barrier/lines.py`), in a local frame so that UTM-sized coordinates behave
like small ones:

1. An end within `max(connectivity_tolerance, simplify_tolerance)` of another
   barrier, a barrier vertex or the domain boundary is snapped onto it. A
   short overshoot past another barrier is trimmed back. A branch that
   collapses onto the barrier it starts on is dropped as a stub. An end is
   not snapped onto a barrier it runs alongside, so a close parallel barrier
   is not folded onto its neighbour.
2. A part lying on the domain boundary (within the same distance) is dropped
   with a warning: a barrier there separates nothing. The end left behind is
   moved onto the boundary, so a barrier that ended on it still does. A
   barrier lying wholly on the boundary leaves no row in `clean_lines`.
3. Junctions and crossings with other barriers and with standard lines become
   shared vertices, and so do points where a line crosses or touches itself.
4. With `simplify_tolerance` above 0, the line is replaced by the fewest
   straight pieces between its own vertices that stay within the tolerance.
   Junctions, crossings and ends are kept. The old per-line Douglas-Peucker
   pass ran before junctions were joined and could pull a splay off its
   fault.
5. A vertex closer than 0.1 x `resolution` to its neighbour is dropped as
   digitising noise, with a warning, even when `simplify_tolerance` is 0.

Barrier lines are no longer densified. Each change is recorded and passed on
to `MeshGenerator.barrier_report`. Steps 4 and 5 take time linear in the
vertex count: a 3000-vertex trace took 16-79 s to simplify and 35 s to clean
of noise, and now takes 0.3 s and 0.05 s; a 20 000-vertex trace takes 2.2 s
and 0.35 s.

## Public API

- `MeshGenerator(barrier_gradation=0.3, sharp_angle=18, barrier_polish_iterations=20)`.
  These are mesh-wide options because faces are graded, and spikes found,
  across lines.
- `ConceptualMesh(keep_close_rings=False)`: keep a `hex_ring` next to a
  barrier by making the barrier faces there shorter.
- After `generate()`: `MeshGenerator.barrier_report` (`snaps`, `noise`,
  `stubs`, `tips`, `on_boundary`, `rings`, `fixed_nodes`, `corridors`,
  `merged`, `collapsed`, each entry with its place, plus `repair_rounds`,
  `sites`, `removed_nodes` and `polish`), `barrier_lines`, `barrier_circles`
  and `node_is_site`. Spike tips, stubs, snapped ends, removed fixed-line
  nodes, merged stretches, merged junctions and dense corridors each raise
  one `UserWarning` with their count.
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
| design (19) | 48 → 0 | 529 → 0 | 0.59 → 0.76 | 1.15 → 0.93 |
| held-out (43) | 27 → 0 | 2692 → 0 | 0.49 → 0.66 | 1.15 → 0.96 |
| adversarial (11) | 3 → 0 | 0 → 0 | 0.55 → 0.73 | 1.18 → 0.93 |

These are with the polish limited to the nodes near the barriers; polishing
the whole domain gave the same compactness and cell ratios of 0.94 and 0.97.
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
- **Merges move the lines.** A merged stretch moves each barrier by up to
  half the gap (at most 0.125 x `resolution`), a merged junction by up to
  0.1 x `resolution`, and the faces there carry both barriers. Both are
  reported, but a model where two close barriers must stay separate needs a
  finer `resolution` there (the merge gap scales with it).
- **Inputs that still raise.** Two barriers that end together at a few
  degrees (a narrow V of two lines, which the spike rule leaves alone), a
  line folding back on itself over a fraction of `resolution`, 16 legs at
  one point at a fine `resolution`, and dense digitising noise at
  `simplify_tolerance=0` still raise `BarrierMeshError`. The message names
  the fold and gives `simplify_tolerance` of about `resolution / 8` for
  noise and `quad_buffer=True` otherwise, which meshes such a barrier with a
  structured strip (not exact faces).
- **Near-miss ends at tolerance 0.** An end that stops just short of another
  barrier, but further away than the snap distance, meshes exactly but with a
  cluster of small faces, or raises; the error then gives the
  `simplify_tolerance` that snaps it.
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
- `tests/test_barrier_faces.py` also meshes splays, crossings, T-branches and
  fans at 3, 5, 8 and 12 degrees, boundary and hole-edge exits at 5 and 10
  degrees and a self-crossing line, checking exactness, sealing and that no
  line moved more than 0.1 x `resolution`; and close parallel pairs at 0.5,
  1, 2 and 4 m at `resolution` 8, merged below 2 m. `tests/test_barrier_internals.py`
  checks the planning steps (validity gate, network, ladders, sites,
  corridors, repair loop, errors, merges, sizing, cleaning) without Gmsh.
- On the 73 benchmark cases the merges change nothing: the barrier sites are
  bit-identical with and without them.
- When this note was written the full suite gave 664 passed and 2 xfailed;
  after the review fixes it gives 769 passed and 2 xfailed.
- `examples/barrier_meshing.ipynb` shows the cases that used to be cut, the
  report and the remaining limits.
