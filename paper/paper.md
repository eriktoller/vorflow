---
title: 'vorflow: feature-conforming Voronoi grids for MODFLOW 6'
tags:
  - Python
  - groundwater
  - MODFLOW
  - Voronoi
  - unstructured grids
  - mesh generation
authors:
  - name: Rui Hugman
    orcid: 0000-0003-0891-3886
    corresponding: true
    affiliation: 1
  - name: Oscar Sanchez
    affiliation: 1  # TODO: confirm affiliation; add ORCID
affiliations:
  - name: INTERA Geoscience Pty Ltd, Australia
    index: 1
date: 13 October 2026
bibliography: paper.bib
---

# Summary

`vorflow` is a Python package that generates two-dimensional unstructured
Voronoi grids for groundwater models built with MODFLOW 6
[@langevin2017; @hughes2017]. The user describes a conceptual model as
Shapely geometries [@shapely]: polygons for the domain and zones, lines for
rivers, drains and faults, and points for wells. Each feature has a target
cell size. `vorflow` meshes this conceptual model with Gmsh [@geuzaine2009],
which produces a triangular mesh that conforms to the features and grades
smoothly between them. The Voronoi grid is the dual of that mesh. The
resulting cells are returned as a GeoPandas [@geopandas] GeoDataFrame
carrying zone IDs, generator coordinates and quality metrics. They can be
exported directly as MODFLOW 6 DISV arrays or as a FloPy [@bakker2016]
package (\autoref{fig:example}).

![A `vorflow` grid refined along a stream network, around wells and inside
a circular zone, with a hole cut into the zone.\label{fig:example}](figure1.png)

# Statement of need

MODFLOW 6 solves groundwater flow on unstructured grids with a
control-volume finite-difference (CVFD) scheme
[@narasimhan1976; @panday2013]. The standard CVFD flux between two cells is
accurate only when the line joining the two cell centres is perpendicular
to their shared face and passes through its midpoint. Voronoi grids satisfy
the perpendicularity condition by construction when the generator points
are used as cell centres [@aurenhammer1991]. Other unstructured grids need
the more expensive XT3D formulation [@provost2017] to correct the error.
Voronoi cells also allow local refinement around wells and along streams
without the cell-count growth of structured or nested grids.

Building a Voronoi grid for a real model is the hard part. Cell sizes have
to grade smoothly from a few metres at a pumping well to hundreds of metres
across a regional aquifer. Rivers and drains need cells centred on or
aligned with them. Faults and cut-off walls need cell faces that lie along
the barrier, so that MODFLOW 6's horizontal flow barrier (HFB) package acts
on the right connections. Zones need cells that do not straddle their
boundaries. Without a tool, modellers do this by hand in GIS, or accept
grids with poorly aligned features, abrupt size jumps and skewed
connections.

`vorflow` makes grid design a scripted step in a Python modelling
workflow. A conceptual model of a few dozen lines of code produces a grid
that is reproducible and can be regenerated when the conceptual model
changes. This matters for workflows that rebuild models automatically,
such as model-structure sensitivity tests and ensemble-based uncertainty
analysis. The intended users are groundwater modellers who already build
MODFLOW 6 models with FloPy.

# State of the field

Several free tools build unstructured grids for MODFLOW 6.

- GRIDGEN [@lien2014] builds quadtree grids. They are locally refined but
  remain rectangular, are written for the DISU package rather than DISV, and
  do not align cell faces with features.
- FloPy [@hughes2024] wraps Triangle [@shewchuk1996] and builds the
  Voronoi dual with its `VoronoiGrid` class. Sizes are controlled by a
  maximum triangle area per region, so grading between regions is stepwise.
  Lines and points enter only as fixed nodes.
- mf6Voronoi [@mf6voronoi] places rings of points around features and
  passes them to Shapely's Voronoi routine. It writes cell centroids rather
  than generators as cell centres, which loses the perpendicularity property.
- Algomesh is a commercial graphical tool. <!-- TODO: cite or remove -->
  <!-- TODO: VOROGRIDGEN is in the benchmark; add a citable reference or drop it. -->

`vorflow` differs from these tools in four ways. Mesh size is controlled by
composable distance fields with geometric growth. Embedded lines become
Voronoi cell faces, not just rows of nodes. Flow barriers are represented
explicitly. Overlapping zones are resolved into a single partition with
zone IDs. We built a new package rather than extending FloPy's Triangle
workflow because these features rely on Gmsh's OpenCASCADE kernel and size
fields, which Triangle does not provide.

The `tool-benchmark/` folder in the repository compares `vorflow`, FloPy and
mf6Voronoi on 15 cases. These include grids reproduced from each tool's own
examples and MODFLOW 6 runs against analytical solutions.
<!-- TODO: re-run the benchmark on the release being reviewed and confirm
these figures; the numbers below are from a local run on 29 September 2026. -->
For uniform flow, which a CVFD scheme should reproduce exactly on an
orthogonal grid, `vorflow` grids gave head errors (L2) of about $10^{-10}$ m
in every case. Grids that use centroids as cell centres gave errors of
$10^{-4}$ to $10^{-3}$ m. In the barrier case, a barrier crosses the grid
at 30°. Only `vorflow` aligned cell faces with it and reproduced the
analytical head jump; the other tools' error was about $10^{-2}$ m.

# Software design

`vorflow` runs in three stages, each a class.

`ConceptualMesh` cleans the conceptual model with pure Shapely operations.
Higher-priority polygons (`z_order`) cut lower ones, lines are snapped to
polygon boundaries and points to lines, and everything is clipped to the
domain. Doing this before meshing keeps topological errors out of Gmsh,
where they are harder to diagnose, and makes this stage testable without
Gmsh.

`MeshGenerator` loads the cleaned features into Gmsh's OpenCASCADE kernel
and fragments them into one consistent model, so that shared edges and
intersections become shared mesh entities. Each feature contributes a
mesh-size field (threshold, exponential or geometric-growth). The fields are
combined with a `Min` field, and Gmsh's own size heuristics are disabled so
that the fields alone control sizing. Features can also be marked
field-only: they refine the mesh without embedding their geometry. Lines and
polygons can optionally be wrapped in a strip of structured quadrilaterals.

`VoronoiTessellator` builds the dual of the triangular mesh with SciPy
[@virtanen2020], which calls Qhull [@barber1996]. Ghost nodes bound the
diagram, and the cells are clipped to the domain. Zone IDs are assigned by
joining generator points, rather than cell polygons, to zones. This is much
cheaper than a polygon overlay. Generator coordinates are kept as the cell
centres. Lloyd relaxation [@du1999] and hexagonal rings around wells are
optional, for users who prefer more regular cells near wells at some cost
in feature alignment.

Barriers are handled in two independent ways. First, the barrier line is
replaced in the mesh by pairs of nodes offset either side of it, so that the
Voronoi face between each pair lies on the line. Second, any cell the line
still crosses is split along it. Together these give MODFLOW 6's HFB package
a continuous chain of faces along the barrier.

The main trade-off is the dependency on Gmsh. It is a large binary
dependency and needs system OpenGL libraries on Linux. In return, it
provides robust constrained meshing and a field-based sizing model that
would be costly to reimplement. The stages communicate through plain arrays
and GeoDataFrames. This let us test the tessellation stage with synthetic
meshes and keep geometry planning (such as quad-buffer crossings) in pure
Shapely.

`vorflow` also reports CVFD grid quality: orthogonality and face skewness
for each connection [@panday2013], computed from the generator points.
Users can check a grid before running a model.

# Research impact statement

<!-- TODO: replace with specific evidence. JOSS asks for realised impact
(projects, publications, external users) or concrete near-term
significance, not aspirations. Candidates:
- named consulting or research models built with vorflow (e.g. the Sunraysia
  model, if it can be named);
- presentations or reports that used it;
- downstream use by other groups. -->

`vorflow` is released on PyPI under the MIT license. Its documentation,
including executed example notebooks, is online, and it is tested on Linux,
Windows and macOS. Users outside the development team have reported issues
and contributed code: the handling of line junctions includes a
contribution from an external user. The reproducible benchmark lets users
compare `vorflow` with other free tools on their own geometry before
choosing one.

# AI usage disclosure

<!-- TODO: confirm this matches how AI tools were actually used. -->
The authors used generative AI tools (Anthropic's Claude, through Claude
Code) to help write code, tests and documentation, and to help draft this
paper. The authors directed the design and reviewed every change. All code
is covered by the automated test suite, which runs on three operating
systems. The numerical behaviour was checked against analytical solutions in
MODFLOW 6 through the benchmark described above.

# Acknowledgements

We thank David ([\@d-strom](https://github.com/d-strom)) for reporting the
line-junction problem and contributing the parallel-line guard and junction
diagnostics.
<!-- TODO: funding statement (JOSS requires acknowledgement of financial
support), e.g. "Development was supported by INTERA Geoscience Pty Ltd." -->

# References
