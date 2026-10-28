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
on the barrier faces was 42-56% lower than on a 4 x finer grid built by the
same code (fine_curve 0.14 against 0.32, real_faults 12.7 against 21.7, in
units of 1e-4). A finer `resolution` reduced the defects but did not remove
them.

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

**Junction merges** (`wedge.py`). Two legs meeting at a narrow angle leave a
sliver that no choice of face lengths can fill with regular cells, so the
lines are moved a bounded distance instead, and each move is reported. Two
barrier legs that meet at a junction (a splay, a T-branch, a crossing) at less
than `sharp_angle` share a stem along their bisector, as long as it keeps
every point within 0.1 x `resolution` of the lines as drawn, and part at
1.5 x `sharp_angle` at its end. A barrier leaving the domain boundary or a
hole edge that narrowly slides along the boundary instead, so it still ends
on it. Fans and stars are merged pair by pair, narrowest first; two adjacent
sectors below 29 degrees are merged too, because the sector plan cannot place
sites in both. Two legs that run to the same next vertex become its chord
(`barrier_report["collapsed"]`). A junction where the move would exceed its
bound, or would reach another line, is left as it is. These are the only
moves of a barrier line. Close parallel barriers are not merged: an earlier
version moved barriers closer than 0.25 x `resolution` onto their midline,
which joined two barriers into one line of faces and changed the model. They
now keep a row of cells between them (see Narrow gaps).

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

**Narrow gaps** (`corridor.py`). Two nearly parallel barrier pieces closer
than 1.2 x the face length share one row of cells (a corridor), with sites on
the midline. Every such gap keeps its row, however narrow, so flow along the
gap stays connected and no barrier is moved. A barrier closer than 0.6 x the
face length to the domain boundary gets one row of cells too, with sites on
the boundary. Without this, the face lengths would be capped by the gap and
the cells there would be small and thin.

Along the gap the sites are spaced at min(h, 3 x gap), where h is the face
length the size field gives. The spacing therefore shrinks in proportion to
the gap below h / 3, with no jump at any gap width. The spacing may change by
at most the gradation from one cell to the next, both where it grows and
where it shrinks. A 3:1 limit was chosen from measurements at `resolution` 8
on a 220 m pair, a wavy gap and a U-turn:

| cells at most ... x the gap long | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| compactness of the row's cells | 0.75 | 0.70 | 0.64 | 0.59 | 0.54 | 0.50 | 0.44 |
| cells, pair at 0.05 x `resolution` | 4312 | | | 2288 | | 1775 | |
| cells, pair at 0.1 x `resolution` | 2260 | | | 1228 | | 980 | |

A 3:1 rectangle has a compactness of 0.59, about the worst cell next to a
barrier in 0.3.0 (0.59 mean over the design set). The worst cell touching the
row of a U-turn stays at 0.63 at 3:1 and drops to 0.54-0.56 at 3.5-4:1. Going
from 1.5:1 to 3:1 halves the cells of a narrow gap; going to 4:1 saves only
another 20%.

The two sides of a corridor are strands: chains of barrier pieces joined at
kinks within the corridor angle, so a corridor follows a wavy gap instead of
ending at each vertex. Each end of a corridor is closed in one of these ways:
a tip (one side ends), a stagger (the sides end at different places), a mouth
(the gap widens), the boundary, a bend (the sides are joined by a short path
of the same line, as at a U-turn, a thin loop or the ends of a lens; the
corridor turns round it and replaces the ladder there) or a fork (both sides
carry on as other pieces past aligned kinks). A corridor may not contain a
point feature, another barrier or a standard line, zone edge or quad-buffer
strip; there the faces are made short instead. A corridor whose plan fails is
dropped, and its pieces get the capped face lengths instead.

A narrow gap still costs cells. The row has about `length / (3 x gap)` cells,
and the mesh around it is graded down to match, which adds about ten cells for
each cell the row has beyond the resolution. Rows of cells shorter than
0.5 x `resolution` raise one warning with their cell count and the count the
resolution alone would give, and `barrier_report["corridors"]` lists each
corridor with its gap, its ends and its cells.

**Repair and error** (`pipeline.py`). Where the check fails, the faces are
halved locally. If the construction is still invalid after a few rounds, or
the faces would have to be shorter than 0.02 x `resolution`, the plan raises
`BarrierMeshError`, a `ValueError` naming the place, what made the faces that
short there (the size caps record which one bound each face length: another
barrier part, the boundary, a point, a fixed line, closely spaced vertices,
a corridor, or a junction or fold) and what helps for that cause. A finer
`resolution` is given with a 10% margin and rounded down, so the value
printed meshes. For a fold, the message gives the `simplify_tolerance` that
removes it, 1.25 x the shortest piece there, or says that the legs are too
long for a tolerance to help. Where a junction merge made the pieces short,
the message says the input is not at fault. A geometric failure inside the
construction is raised as `BarrierMeshError` too, as a bug to report.

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
   digitising noise, with a warning, even when `simplify_tolerance` is 0. A
   vertex is kept where dropping it would leave a fold sharper than
   18 degrees, so the corners of a narrow U-turn are not turned into a
   spike.

Barrier lines are no longer densified. Each change is recorded and passed on
to `MeshGenerator.barrier_report`. Steps 4 and 5 take time linear in the
vertex count: a 3000-vertex trace took 16-79 s to simplify and 35 s to clean
of noise, and now takes 0.3 s and 0.07 s; a 20 000-vertex trace takes 2.2 s
and 0.45 s.

## Public API

- `MeshGenerator(barrier_gradation=0.3, sharp_angle=18, barrier_polish_iterations=20)`.
  These are mesh-wide options because faces are graded, and spikes found,
  across lines.
- `ConceptualMesh(keep_close_rings=False)`: keep a `hex_ring` next to a
  barrier by making the barrier faces there shorter.
- After `generate()`: `MeshGenerator.barrier_report` (`snaps`, `noise`,
  `stubs`, `tips`, `on_boundary`, `rings`, `fixed_nodes`, `corridors`,
  `collapsed`, each entry with its place, plus `repair_rounds`, `sites`,
  `removed_nodes` and `polish`), `barrier_lines`, `barrier_circles` and
  `node_is_site`. Spike tips, stubs, snapped ends, removed fixed-line nodes,
  merged junctions and dense corridors each raise one `UserWarning` with
  their count.
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
0.0192 to 0.0180.

The flow through an HFB on the barrier faces was compared with a 4 x finer
grid built by the same code (in units of 1e-4):

| case | 0.3.0 | 0.3.0, 4 x finer | this change | this change, 4 x finer |
|---|---|---|---|---|
| real_faults | 12.68 | 21.73 | 24.84 | 24.97 |
| fine_curve | 0.14 | 0.32 | 0.414 | 0.416 |

So 0.3.0 under-counts the flow by 42-56%, and the exact faces are within 1%
of their own finer grid.

Meshing a model with barriers takes about 1.7 x as long as in 0.3.0: 12.7 s
against 7.4 s over 23 benchmark cases, 1.5-2.3 x per case. The polish band
keeps the extra cost tied to the barrier length.

Spikes of 10 and 15 degrees in the held-out and adversarial sets each leave 7.2 m of tip without faces (0.9 x their 8 m resolution), as
intended. These are the only cases where the faces do not cover the whole
barrier.

## Limitations

- **Quad buffers on barriers** (`quad_buffer=True`) still use the mirror
  generators and the cut, so they can leave cut cells and faces of several
  edges. Two strict `xfail` tests in `tests/test_engine_regressions.py` record
  this. Field-only barriers (`embed=False`) are not embedded and go through
  the same path.
- **Junction merges move the lines and change meshes that worked.** A merged
  junction moves each line by up to 0.1 x `resolution`, and the stem carries
  both barriers. This also applies to junctions that meshed without it: a
  crossing at 10 degrees went from 568 to 733 cells, and other narrow
  junctions gained 10-30% cells near the merged legs. A barrier leaving the
  boundary at 3 degrees slides along it and now starts up to
  1.9 x `resolution` further along the boundary.
- **Narrow gaps cost cells.** A corridor's row has about `length / (3 x gap)`
  cells, and the mesh around it is graded down to match. A 220 m pair at
  `resolution` 8 has 2312 cells at a gap of 0.05 x `resolution`, 1228 at 0.1
  and about 490 above 0.35; a U-turn, thin loop or lens at 0.1 x `resolution`
  has 2.3-2.7 x the cells of the same shape as a standard line. At a gap of
  1.2 x `resolution` the corridor stops and the count rises by about 10%
  (488 to 540), since the faces are then capped by the gap.
- **Lenses and loops with junction ends.** Two lines that part at one
  junction and meet again at another (a fault lens, a box lens, a thin loop
  from a junction) get no corridor, since a corridor can only turn round the
  bend of one line. Their faces are capped by the gap instead. A lens 1.5 or
  2 m wide at `resolution` 8 raises `BarrierMeshError`; 4 m and wider
  meshes.
- **Densely digitised lines at tolerance 0.** Every input vertex is a face
  end, so the faces are no longer than the input pieces. On `net13_collinear`
  (vertices 0.1-0.5 x `resolution` apart, median 0.3) there are 2.87 x as
  many cells near the barriers as on 0.3.0 (2281 against 794). 0.3.0 ignored
  the vertices and left 287 bent faces and an orthogonality error of 24
  degrees. With `simplify_tolerance` of `resolution / 8` the same network
  has 0.80 x the cells of 0.3.0 (639 against 799).
- **Inputs that still raise.** Two barriers that end together at a few
  degrees (a narrow V of two lines, which the spike rule leaves alone), a
  line folding back on itself over a fraction of `resolution`, 16 legs at
  one point at a fine `resolution`, and dense digitising noise at
  `simplify_tolerance=0` still raise `BarrierMeshError`. The message names
  the fold and gives the `simplify_tolerance` that removes it, measured from
  the shortest piece there, or `quad_buffer=True` where the legs are long,
  which meshes such a barrier with a structured strip (not exact faces).
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
  line moved more than 0.1 x `resolution`.
- Narrow gaps: close parallel pairs at 0.4, 0.8, 2 and 4 m at `resolution` 8
  keep one row of cells no longer than 3 x the gap; a sweep of the gap from
  0.05 to 1.25 x `resolution` checks that the cell count changes smoothly
  and the mesh stays connected; a relay ramp to the boundary, a gap that
  narrows and widens (a double dip, a wavy gap), a well and a zone edge
  between two close barriers, and the joined legs of a U-turn, a thin loop
  and a lens, whose cell count is held to a bound. `tests/test_barrier_internals.py`
  checks the planning steps (validity gate, network, ladders, sites,
  corridors and their ends, repair loop, errors and their advice, merges,
  sizing, cleaning) without Gmsh.
- The coverage check runs on every case, including merged junctions, where
  the faces must stay within the reported deviation of the line.
- When this note was written the full suite gave 664 passed and 2 xfailed;
  after the first review fixes it gave 769 passed and 2 xfailed, and after
  the second, merged with develop, 843 passed, 2 skipped and 2 xfailed.
- `examples/barrier_meshing.ipynb` shows the cases that used to be cut, the
  report and the remaining limits.
