# Benchmark plan — vorflow vs mf6Voronoi, FloPy (Triangle + VoronoiGrid), VOROGRIDGEN

**Status:** draft · **Scope:** benchmark + demo material, no package changes ·
**Audience:** README/demo, and helping users choose the right tool

## Goal

Help a MODFLOW 6 user decide which free Voronoi grid generator fits their
problem, and show where vorflow does and does not add value. Compare four
tools on a shared set of test cases, measuring:

1. **Cost** — cell count, run time, memory.
2. **Spec conformance** — does the tool deliver the cell sizes and feature
   alignment it was asked for?
3. **Geometric quality** — CVFD orthogonality, skewness, short edges, cell shape.
4. **Numerical accuracy** — MF6 head error against analytical solutions, and
   solver behaviour on real models.
5. **Capability/usability** — which features each tool can represent at all.

## Deliverables

1. **A tool-choice guide for the README.** This is a "which tool when" table
   keyed on user needs: holes, barriers/faults, many wells, regional scale,
   GUI, Windows-only acceptable, pure-pip install. It is backed by the
   capability matrix and the measured results, and states where another tool
   is the better choice.
2. **Four to six figures:**
   - the same case meshed by each tool, side by side;
   - cell size vs distance from a well, against the spec;
   - MF6-seen non-orthogonality distributions;
   - V1/V2 head error, with generator vs centroid centres;
   - run time vs cell count;
   - the barrier demo.
3. **A reproducible `tool-benchmark/` folder,** so users can rerun it on their own
   geometry.

This is not a paper. Anything that doesn't feed the guide or a figure is
optional. That covers the Tier 2 fine-reference models and most of Tier 3.

## Tools under test

| Tool | Version to pin | Platform | Mesh engine | Sizing control | Cell centre in DISV |
|---|---|---|---|---|---|
| **vorflow** | this repo, tagged commit | any | Gmsh (OCC) → dual | per-feature `resolution` + Distance/Threshold fields, `growth_factor`, `background_lc` | generator (`x`,`y`) — adapter must write DISV (vorflow has no DISV writer) |
| **mf6Voronoi** | 0.0.38 @ `38a29849` | any | point rings around features → `shapely.voronoi_diagram` | `maxRef`, `multiplier`, per-layer `layerRef` | polygon centroid |
| **FloPy** | flopy ≥ 3.10, Triangle 1.6 from `get-modflow` | any | Triangle (conforming Delaunay, `-q` angle) → `VoronoiGrid` | piecewise-constant `maximum_area` per region; lines/points only as fixed `nodes` | *verify* (`get_disv_gridprops`) |
| **VOROGRIDGEN** | Fortran 2026-02-09 binaries | Windows only | seed placement → Delaunay (GEOMPACK2) → damped Lloyd | per-vertex spacing, `max_centroid_separation`, `poly_growth_rate` | centroid |

Optional fifth entry: the Python port in `../vorogridgen` (`vorogridgen.core.generate.generate`),
reported separately from the Fortran reference, never merged with it.

### Capability matrix (to confirm in phase 1)

| Capability | vorflow | mf6Voronoi | FloPy | VOROGRIDGEN |
|---|---|---|---|---|
| Domain holes | yes | limit must be one Polygon; holes are clipped out, not seeded | yes (`add_hole`, polygon order matters) | yes (`inner_boundary`) |
| Multiple zones / zone IDs | yes (`z_order` partition) | no (layers are refinement only) | region attributes | no |
| Internal line refinement | yes | yes | only as densified `nodes` (centres on line) | yes (`inner_line`) |
| Faces aligned to a line | yes (embedded lines) | no | no | no (seeds on line) |
| Flow barrier | yes (exact faces, empty-circle rule) | no | no | no |
| Well is a generator | yes | point is seeded | if passed as node | yes, with hexagonal ring |
| Smooth size grading | fields, `growth_factor` | geometric rings | nested buffer regions only | `poly_growth_rate` |
| Output | GeoDataFrame | shapefile → `get_gridprops_disv` | DISV gridprops | DISV + minimal MF6 model |

## Cell centres: generator vs centroid

Each DISV cell has a centre (`xc`, `yc` in `cell2d`). How MF6 uses it, from
`DisvGeom%cprops` in the MF6 source:

- `hwva` is the length of the shared edge.
- `cl1` and `cl2` are the **perpendicular distances** from each cell's `xc`,`yc`
  to the line through that shared edge.

Without XT3D, the flow across the face is computed as
`K · hwva · (h₂ − h₁) / (cl1 + cl2)`. That formula treats the head difference
between the two centres as if it were entirely the gradient normal to the face.
This holds only if the line joining the two centres is **perpendicular to the
shared face**.

Voronoi generators satisfy this by construction: each Voronoi edge lies on the
perpendicular bisector of the two generators it separates. That property is
the reason to use a Voronoi grid with MF6 at all.

The centroid of a Voronoi cell is generally *not* its generator. The two
coincide only for regular hexagons, or for a fully converged centroidal Voronoi
tessellation. Elsewhere they differ, especially:

- in grading zones, where neighbouring cells change size;
- at clipped boundary cells;
- around features.

If the DISV stores centroids, the line joining two neighbouring centroids is
tilted by some angle θ from the face normal. MF6 still divides by the
perpendicular distances, so part of the head difference that it counts as
normal flow is really caused by the gradient *along* the face. The flux error
is roughly `tan θ` times that along-face gradient.

The error doesn't show up as an obvious failure. The model converges and the
budget closes, but heads are biased wherever flow runs oblique to faces, and
more so where cells are graded or irregular. Uniform flow on a mixed grid
exposes it most directly. It also affects:

- where heads are reported and plotted, and where observations are matched;
- raster sampling of top/bottom, which is usually done at `xc`,`yc`;
- XT3D, which also uses the centres to rebuild the full gradient.

XT3D reduces the error but costs more solver time.

**How the tools differ:**

| Tool | Writes to `xc`,`yc` | Notes |
|---|---|---|
| vorflow | generator | keeps `x`,`y` = generator and `centroid_x`,`centroid_y` separately |
| FloPy `VoronoiGrid` | generator | `get_disv_gridprops` passes `xcyc=self.points`, the Triangle vertices; boundary generators sit on the domain edge (half-cells) |
| mf6Voronoi | centroid | `meshShape.get_gridprops_disv` uses the polygon centroid; the generators exist only in `modelDis` before `shapely.voronoi_diagram` |
| VOROGRIDGEN (Fortran) | centroid | damped Lloyd (default `lloyd_fac` 0.2, 30 iterations) moves generators towards centroids but doesn't converge; the golden example has 3.25° median MF6-seen non-orthogonality. The vorogridgen Python port writes generators |

This one choice can dominate the results. On the same V1 geometry, the
VOROGRIDGEN golden grid (centroids) gives L2 9.9e-4, and the Python port
(generators) gives 3.1e-8.

**How the benchmark handles it:**

- Every quality metric is computed **as MF6 sees the grid**, from the DISV
  `xc`,`yc` the tool writes. This is what a user actually gets.
- A **centre ablation** reruns V1–V3 with each tool's DISV rewritten to use
  generators, wherever they can be recovered:
  - mf6Voronoi: point-in-polygon match against `modelDis` points;
  - FloPy: Triangle vertices;
  - VOROGRIDGEN: only if an output lists the seeds (to check).

  This separates "the mesher placed points badly" from "the tool wrote the
  wrong centre". The fix for the second is a one-line change on the user's
  side, and the guide should say so.
- A new metric, **centre offset** = `|generator − xc,yc| / √area`, is reported
  per cell.

## Fairness protocol

We maintain vorflow, so the setup has to be defensible to the other maintainers.

- **Three run modes:**
  - **Native:** each tool runs its own published cases with the author's own
    parameters, untouched. This shows what the authors intended.
  - **Matched spec:** one tool-neutral sizing spec per case (below), translated
    to each tool by a documented rule. This tests spec conformance.
  - **Matched count:** one global scale factor per tool is tuned until NCPL is
    within ±5 % of a target. This is the headline comparison for quality and
    accuracy, since it compares them at equal cost.
- Use documented defaults for every parameter not in the spec. Record every
  non-default value in the case file.
- Record failures, crashes, timeouts (30 min) and invalid grids as results.
  Don't drop them.
- Before publishing, offer the Hatari Labs and Hydrosymple authors a chance to
  review their adapters and parameters.

### Tool-neutral sizing spec

Each case defines:

- `h_max`: the background cell size.
- For each feature: a target size `h_f` and a growth ratio `g` (default 1.2).

The implied size field is `h(x) = min(h_max, min_f h_f · g^(d_f(x)/h_f))`, or the
equivalent linear growth form. Translation rules:

- **vorflow:** `resolution = h_f`, `growth_factor = g`, `background_lc = h_max`.
- **mf6Voronoi:** `layerRef = h_f`, `maxRef = h_max`. `multiplier` is calibrated
  once per `g` so that its ring sizes `s_i = s_{i-1} + mult^i · layerRef` fit
  `h(d)` best (least squares over the growth zone).
- **FloPy:** `maximum_area = (√3/4) · h²` (equilateral triangle with edge h).
  Grading uses nested buffer regions at the distances where `h` crosses
  `h_f · g^k`. Lines are densified at spacing `h_f` and passed as `nodes`.
- **VOROGRIDGEN:** per-vertex spacing `h_f`, `max_centroid_separation = h_max`,
  `poly_growth_rate = g`.

## Test cases

Cases are grouped into four tiers. Every case is stored once in a tool-neutral
form: a GeoPackage with layers `domain`, `holes`, `polygons`, `lines` and
`points`, plus a `case.yml` holding the sizing spec, the native parameters per
tool, and the MF6 problem if there is one. Adapters read only this form.

### Tier 0 — verification (analytical solutions, MF6 runs)

Small domains where the exact head is known, so accuracy can be measured.

| ID | Problem | Origin | Geometry | What it probes |
|---|---|---|---|---|
| V1 | Uniform 1D flow, `h = 1 − x/L` | FloPy `dis_voronoi_example` (2000×1000 m, area 1000 → ~2 000 cells); vorogridgen `mesh_accuracy.py` "linear" | rectangle, uniform size | baseline: a good Voronoi grid should reproduce linear head almost exactly (vorogridgen Python port: L2 3e-8) |
| V2 | Manufactured solution `h = cos(ax)cos(ay)`, source `W = 2Ta²h` as WEL | vorogridgen `examples/mesh_accuracy.py` "mms" | square, with and without a graded refinement patch | error across grading transitions; XT3D off vs on |
| V3 | Steady radial flow to a well (Thiem), `h = h₀ − Q/(2πT)·ln(R/r)` | new; mf6Voronoi c20 `theisNash` is the transient analogue | circle radius R, point refinement at centre | point refinement quality near the singularity |
| V4 | 1D flow across a horizontal flow barrier (series resistance, head jump at the barrier) | new | rectangle crossed by a straight barrier at 30° to the axes (a kinked line has no closed-form solution, so it was dropped) | barrier representation: vorflow has faces on the line; others get HFB on the zig-zag of faces nearest the line |
| V5 | Pure advection at 45°, analytical rotated inflow profile | MF6 `ex-gwt-adv-schemes` (100×100 cm) | square, uniform size | optional GWT check; TVD/UTVD on each grid (not yet implemented) |

MF6 setup follows the vorogridgen harness: confined single layer, K = 1,
T = 100, CHD at the exact head on edge cells. The error is area-weighted L2 over
interior cells, plus the maximum error. Run each problem with XT3D off and on.

V1–V3 also get the centre ablation described in
[Cell centres](#cell-centres-generator-vs-centroid).

### Tier 1 — feature regression (synthetic, small)

| ID | Case | Origin | Features | Reference |
|---|---|---|---|---|
| F1 | `voronoi_circle`, `voronoi_nested_circles` | FloPy `autotest/test_grid_cases.py` | circle r=100; circle with r=30 hole | FloPy ncpl 538 / 300 (±10) |
| F2 | `voronoi_polygons`, `voronoi_many_polygons` | FloPy `test_grid_cases.py` | square with nested refinement squares, a hole, a conforming circle, a diagonal line | FloPy ncpl 410 / 1305 (±10) |
| F3 | `voronoi_polygon` | FloPy `test_grid_cases.py` | irregular 17-vertex polygon about 9×7 km | FloPy ncpl 3803 |
| F4 | vorflow comprehensive demo | `examples/comprehensive_demo.ipynb` | 200×200 domain with hole, refinement zone, river, fault barrier, 2 wells | — |
| F5 | vorflow field capabilities | `examples/field_capabilities_example.py` | 400×200 domain, nested polygons, several lines and points | — |
| F6 | VOROGRIDGEN example | `../vorogridgen/2026-02-09_vorogridgen/example/vg.in` | 58×48 km outer boundary with variable spacing, 2 holes, 2 lines, 3 points, 2 refinement polygons | golden DISV NCPL 6 440 |

### Tier 2 — real-world geometry

These come mostly from mf6Voronoi's `tests/json/meshCasesNormal.json`, with native
parameters taken from that file.

| ID | Case | Size | Features | Why |
|---|---|---|---|---|
| R1 | c2 riverAquifer | 3.1×1.7 km | 4 GHB lines, river polygon, 3 wells | small, mixed features; native mesh about 4 350 cells |
| R2 | c5 siteDewatering | 302×290 m | MultiLine GHB, 2 drains, 2 wells | very fine sizes (1–2 m) |
| R3 | c7 trenchExcavation | 2.2×1.35 km | 2 GHB polygons, 4 wells, 4 drains | native mesh about 3 300 cells |
| R4 | c20 theisNash | 592×530 m | refinement polygon **with a hole**, river polygon, 4 lines | hole inside a refinement layer |
| R5 | c1 regionalAngascancha | 22.5×27.1 km | 17 river lines | regional scale; MF6 runner exists (5 layers) |
| R6 | c6 stibniteMine | 13.5×14.8 km | 35 water lines, 8 pit/dump polygons, **139 fault lines** | many intersecting lines; native mesh about 49 900 cells. Also run the faults as barriers in vorflow only (demo) |
| R7 | c19 regionalSeawaterIntrusion | 31×30 km | 34 streams, 95 points, sea polygon | many points; fix the `.prj` CRS defect on load |
| R8 | MF6 `ex-gwt-synthetic-valley` | 6.1×3.8 km | river polyline, lake, 3 wells | USGS reference Triangle/Voronoi grid (6 343 cells per layer) and a full GWF+GWT model |
| R9 | Groundwater 2023 watershed (Hughes et al. 2024) | 180×100 km | boundary + 4 stream segments | USGS paper case; also compare against the GRIDGEN quadtree grid from the same paper |

Tier 2 cases have no analytical solution. For the guide, each grid is run in a
simple steady-state recharge + river/drain model and reported on:

- whether the grid is valid and runs;
- budget percent discrepancy;
- outer and inner solver iterations;
- MF6 wall time.

Optional: head error against a fine structured DIS reference, using the
existing runners (c1, c14, c19, synthetic valley).

### Tier 3 — scale and stress

S1 and S2 are enough for the run-time figure and one "does it survive messy
regional data" row in the guide.

| ID | Case | Purpose |
|---|---|---|
| S1 | c8 lowerOuachitaSmackover (one of mf6Voronoi's HUC-8 riparian cases; c9–c13 optional) | robustness with many holes and vertices |
| S2 | R5 (c1) scaled by `0.9^k`, k = 0…10 (mf6Voronoi's `caseGrid` pattern) | run time vs cell count; target 10³–10⁶ cells |
| S3 (optional) | synthetic square with N wells and M random lines, N,M ∈ {10, 100, 1000} | feature-count scaling |

Report mf6Voronoi single-threaded in the main runs. Its Dask `nproc` scaling
goes in a separate figure, as its own tests do.

## Metrics

All outputs are normalised to one form: DISV gridprops (`vertices`, `cell2d`
with `xc`,`yc`) plus a GeoDataFrame of cells with `x`,`y` = generator where the
tool exposes one. Metrics are computed **as MF6 sees the grid** (from `xc`,`yc`)
first, and generator-based second.

| Group | Metric | Implementation |
|---|---|---|
| Cost | NCPL, NVERT, wall time (median of 5, I/O timed separately), peak RSS | `time.perf_counter`, `psutil` |
| Validity | self-intersections, multipart cells, duplicate vertices, hanging nodes, coverage error `|Σ area − domain area| / domain area` | shapely + FloPy `VertexGrid` |
| Spec conformance | ratio `√(cell area)/h_spec(x)` (distribution); distance from each well to its cell's centre; line-to-nearest-face distance; cells per unit river length | new, in `tool-benchmark/bench/metrics.py` |
| CVFD quality | orthogonality error, skewness, drift ratio, compactness, convexity | `vorflow.utils.calculate_mesh_quality`, `build_connectivity` |
| Short edges | minimum edge length; count below 0.001 × local h (mf6Voronoi's `checkVoronoiQuality` criterion) | new |
| Topology | neighbour area ratio, hexagon fraction, neighbour-distance CoV | port from `../vorogridgen/docs/topological-parity-findings.md` |
| Accuracy | L2 and max head error (Tier 0); head/budget/iterations against the fine reference (Tier 2) | `tool-benchmark/bench/models.py` |
| Usability | lines of adapter code per case, manual steps, unsupported features | recorded by hand |

## Layout

The benchmark lives in the repo but outside the package, with its own
environment so that mf6Voronoi, FloPy and Triangle don't become vorflow
dependencies.

```
tool-benchmark/
  environment.yml        # conda env; or a venv over the vorflow env (see README)
  workflow.py            # entry point: flags in __main__ (calibrate, match, mf6, plot)
  cases/<id>.yml         # inline WKT geometry, sizing spec, target NCPL, MF6 problems
  bench/case.py          # case loading, spec size field
  bench/grid.py          # common Grid form, DISV read/write, centre conventions
  bench/adapters/        # vorflow, mf6voronoi, flopy, vorogridgen: build(case, scale, ws) -> Grid
  bench/metrics.py       # metrics from DISV topology, as MF6 reads it
  bench/models.py        # V1-V5 verification problems
  bench/matching.py      # matched-count scale search
  bench/isolate.py       # one spawned process per build: timing, peak RSS, timeout
  bench/plots.py
  results/               # metrics.csv, mf6_verification.csv, figures
```

Real-world cases (Tier 2-3) will add a `fetch.py` that downloads inputs at
pinned commits and a GeoPackage-based case variant.

- **VOROGRIDGEN on CI:** it runs on a Windows job, reusing
  `../vorogridgen/.github/workflows/golden.yml`. The adapter writes BLN + `vg.in`
  there, and the DISV comes back as an artifact. All metrics run on
  Linux/macOS.
- **MF6 and Triangle:** both come from the MODFLOW-ORG executables release (`get-modflow`).

## Phases

1. **Scaffold.** Build the case format, all four adapters, and the metrics on V1
   and F4. Confirm the capability matrix (mf6Voronoi limit holes, FloPy centre
   convention).
2. **Verification.** Run Tier 0 with MF6, including the centre ablation and
   XT3D on/off.
3. **Feature and real cases.** Run Tiers 1–2 in all three modes, plus the
   Tier 2 reference models.
4. **Scale.** Run Tier 3 scaling runs.
5. **Report.** Write the notebook, the README tool-choice guide and figures,
   and the vorflow-only barrier demo on R6 and V4. Then send the results to the
   other tools' maintainers for review.

### Phase 1 status (2026-09-28)

Done on macOS for vorflow, mf6Voronoi and FloPy. The cases are V1, F4 and a
calibration case `c0_point_grading`, which has one refined point. Matched
count is within 5 % for every buildable tool; outputs are in
`tool-benchmark/results/`.

- **VOROGRIDGEN:** the adapter writes BLN + `vg.in`, but the Windows CI job
  isn't built yet. The DISV reader and orthogonality metric reproduce the
  golden grid's 6 440 cells and 3.25° median.
- **mf6Voronoi calibration:** `multiplier = 1.05` fits growth 1.2 best. The
  common example value, 1.5, grades much faster. Even at 1.05 its sizes reach
  `h_max` at about 40 % of the spec's growth distance.
- **Cell centres:** the V1-style linear problem, now oblique at 30°, confirms
  the ablation on every tool. Centroid centres give L2 ≈ 4e-4 to 1.3e-3 without
  XT3D; generators give about 1e-10. XT3D removes the centroid error except
  where the faces are disconnected.
- **Disconnected faces**, i.e. neighbours MF6 won't connect, are found and do
  change heads:
  - *mf6Voronoi:* its `meshShape.get_gridprops_disv` merges vertices only on
    exact equality, so vertices that differ in the 16th digit break edges
    (16 faces on c0).
  - *FloPy:* `VoronoiGrid` duplicates the vertex where a right-angled corner
    triangle's circumcentre falls on a boundary-edge midpoint (6 faces on c0,
    11 on F4). Worth reporting upstream to both projects.
- **vorflow F4:** the error stays at about 3e-5 even with XT3D. It is
  concentrated on the barrier-split cells, whose centres are centroids.
  This was before exact barrier faces (design note 08), which leave no
  split cells; rerun F4 to update it.
- **FloPy adapter fairness:** band boundaries leave clusters of small cells
  along the rings. Densifying the rings to the band size should fix this;
  do it before Phase 3.

### Phase 2 status (2026-09-28)

Verification problems V1–V4 run for every tool, centre convention and XT3D
setting. The new cases are `v2_mms_uniform` and `v2_mms_graded` (each a
three-point convergence sweep), `v3_thiem` and `v4_barrier`. The linear and
MMS fields are rotated 30° so they don't align with square lattices. Figures
are in `tool-benchmark/results/figures`:

- `verification_summary.png`
- `v2_*_convergence.png`
- mesh and size plots per case

Headline L2 errors (centres as written, XT3D off, first target):

| Problem | vorflow | mf6Voronoi | FloPy |
|---|---|---|---|
| V1 linear | 6e-11 | 8e-11 | 8e-11 |
| V2 MMS uniform (2 000 cells) | 3.9e-4 | 5.5e-4 | 3.9e-3 |
| V2 MMS graded (1 000 / 9 000 cells) | 4.7e-3 / 5.2e-4 | 2.7e-3 / 5.2e-4 | 2.2e-3 / 4.2e-4 |
| V3 Thiem | 1.6e-4 | 4.0e-4 | 1.5e-4 |
| V4 barrier | **5e-10** | 1.3e-2 | 1.4e-2 |

- **Barrier:** V4 is the clearest vorflow result. Faces on the line plus HFB
  are exact to roundoff; HFB on a zig-zag of faces is off by about 1 % of the
  head drop.
- **Smooth fields (V2/V3):** the tools are within about 2× of each other
  where their grids are sound. vorflow is not better on graded MMS, and the
  guide should say so.
- **FloPy's large V2 uniform error** comes from `VoronoiGrid`'s duplicate
  vertices (disconnected faces on the domain edge), not from cell shape.
  Phase 3 should add a "repaired" variant that merges duplicate vertices, so
  users see what the fix buys.
- **mf6Voronoi on `v2_mms_graded` at 3 000 cells:** MF6 6.7.0 crashes
  (SIGILL) on the grid, which has 90 zero-length edges from its exact-equality
  vertex merge. This is a user-facing failure, recorded as `mf6_ok = False`.
- **Centroids with XT3D:** on the smooth MMS field, centroid centres with XT3D
  beat generator centres slightly (vorflow uniform: 9.5e-4 vs 1.6e-3 at
  500 cells). Without XT3D, generators always win.
- **FloPy adapter fix:** FloPy passes Triangle either a global `-a<area>` or a
  bare `-a`. Only the bare flag enables regional areas, so the adapter now
  sets `maximum_area=None` and adds a background region. Before this fix the
  bands were silently ignored.
- **FloPy `to_cvfd` hang:** the hanging-node check does not terminate on
  vorflow's V4 grid. The benchmark skips it (it is meant for quadtree grids),
  so any hanging nodes appear as disconnected faces instead.

CI: `.github/workflows/benchmark.yml` runs three jobs:

- **vorogridgen** (Windows): downloads the freeware at run time and checks its
  shipped example.
- **others** (Linux).
- **report**: merges the per-(case, tool, target) JSON rows, and fails if
  there are none.

First full run (run 36466904249): all green, and VOROGRIDGEN's shipped example
gives NCPL 6440. Findings from VOROGRIDGEN's first results:

- **Builds:** 7 of 11 finished. The other four stopped during the Lloyd
  iterations (`c0` n3000, `v2_mms_uniform` n2000 and n8000, `v2_mms_graded`
  n9000). Three failed with "consider reducing the LLOYD_FAC" and one with
  "Missing triangle local neighbour". The adapter now retries with
  `lloyd_fac` 0.1, then 0.05, the program's documented remedy, and records
  the value used. `v3_thiem` also missed the matched count (3 851 against
  3 000 after 8 builds).
- **Second run (36479497679):** the `lloyd_fac` retry fixed the "missing
  triangle" case (0.2 failed, 0.1 built). The other three still failed at all
  three values. In each, the first build (`build_00`, 0.2) succeeded; the
  second, at the finer matching scale, failed. A lower `lloyd_fac` only made
  it fail sooner. The failures are specific to particular spacings, not to the
  cases. Matching (`bench/matching.py`) now retries a failed step at 1.02×,
  0.98× and 1.05× its planned scale. If no build lands within tolerance, it
  keeps the successful build closest to the target (`matched = False`).
  `n_failed_builds` is recorded for every tool.
- **Third run (36485251402):** VOROGRIDGEN now returns a grid for all 11
  targets, but four fall short of the matched count. Every finer attempt
  failed, 13 builds in all; the other tools had none.

  | Case | Target | VOROGRIDGEN cells | Failed builds |
  |---|---|---|---|
  | `c0_point_grading` | 3 000 | 1 807 | 5 |
  | `v2_mms_uniform` | 8 000 | 6 560 | 4 |
  | `v2_mms_graded` | 9 000 | 6 103 | 4 |
  | `v3_thiem` | 3 000 | 2 826 (6 % short) | 0 |

  On these cases VOROGRIDGEN seems unable to build much finer than the
  counts shown. This is a result for the guide, not a harness bug.
  `mf6_verification.csv` now carries each grid's `matched` flag.
  `verification_summary.png` leaves out unmatched grids. The convergence
  plots keep them, since they plot error against actual cell count.
- **VOROGRIDGEN with XT3D stays competitive even with fewer cells:**
  `v2_mms_uniform` gives 5.8e-5 at 6 560 cells against vorflow's 9.1e-5 at
  8 039, and `v2_mms_graded` 2.0e-4 at 6 103 cells against 2.2e-4 to 5.1e-4
  for the others at about 9 000. Without XT3D it is the worst tool in every
  case (1e-2 to 6.5e-2).
- **Centres:** written centres give exactly the same errors as centroids, which
  confirms the DISV carries centroids. Without XT3D, V1 linear gives 4.3e-3,
  ten times worse than the other tools' centroid rows, because the unconverged
  Lloyd iterations leave 19° p95 non-orthogonality.
- **With XT3D** it is exact on every linear case and best or near-best on
  smooth fields: V2 graded 1 000 cells 9.6e-4 against 1.2e-3 to 1.7e-3 for the
  others; V3 Thiem 8.9e-5. For VOROGRIDGEN grids the guide should say to turn
  XT3D on.
- **Run time:** 1.5–100 s per grid on the Windows runner, against under 3 s for
  the other tools. That is not a like-for-like machine comparison.

Linux CI and local vorflow results differ on V4. CI gives 1.2e-6 with 4
zero-length edges; locally it was 5e-10 with 2. Presumably the conda-forge
Gmsh builds differ. XT3D makes every tool's V4 barrier error worse or no
better (vorflow 2.3e-4), so the barrier headline is XT3D off.

## Risks and caveats

- **Case bias.** Each tool's own cases suit that tool. This is why cases are
  pooled across all sources and matched count is the headline mode.
- **No true "same input".** The sizing controls differ in kind. The spec
  translation is itself a result and must be published with the numbers.
- **Centre convention.** Centroid vs generator can dominate accuracy (see the
  ablation above). Never compare accuracy without it.
- **Platform dependence.** FloPy/Triangle cell counts vary by platform (its
  tests allow ±10), and VOROGRIDGEN runs only on Windows. Pin the platform per
  tool and record it.
- **Barrier comparison is one-sided.** Only vorflow represents barriers. Present
  V4 and the R6 fault run as a capability demo, not a head-to-head.

## Licensing and data

- **FloPy, python-for-hydrology:** public domain / CC0. Their cases can be
  vendored.
- **MF6 examples:** there is no LICENSE file (presumably a USGS public-domain
  work). Fetch at run time.
- **mf6Voronoi:** the code is MIT, but `tests/data` has no provenance or data
  licence. Fetch at a pinned commit instead of vendoring, and ask Hatari before
  redistributing derived inputs.
- **VOROGRIDGEN:** freeware, closed source. Don't redistribute the binaries or
  the example; fetch or keep them local.
- **Triangle:** free for non-commercial use only. Use the `get-modflow` copy and
  don't bundle it.

## Decisions

- **Headline mode:** matched count.
- **Spec growth ratio:** `g = 1.2`.
- **Audience:** README/demo and a tool-choice guide (not a paper).

## Open decisions

- Include the vorogridgen Python port as a fifth entry, or keep it out of this
  benchmark?
