"""Exact barrier faces: Voronoi sites placed so every barrier segment is one cell face.

Pure geometry (NumPy, SciPy, Shapely), no Gmsh calls; ``ConceptualMesh``
cleans the barrier lines (``lines``) and ``MeshGenerator`` plans the sites
before Gmsh and fits the mesh around them (``pipeline``, ``frames``, ``polish``).

Methodology
-----------
*The empty-circle rule* (Berge, Klemetsdal & Lie 2019, the UPR method; also
VoroCrust). Cut every barrier into segments at *1D nodes* and give each node
a circle. The two circles of a segment's end nodes cross at two points,
mirror images across the segment; put a site at each. If no other site lies
inside any circle, each segment is exactly the Voronoi face between its two
sites, whatever the other generators are. The construction therefore fixes
the barrier sites in closed form, checks the condition (``validate``) and
lets Gmsh mesh everything outside the circles.

*Nodes with several legs* (``network``). At a kink, junction, crossing,
boundary end or standard-line crossing, the first pair of each leg sits on
the node's circle at +-phi from the leg. Neighbouring legs either share one
site (phi_a + phi_b = sector angle), keep an empty arc between their sites,
or, in a sector narrower than RUNG_ANGLE, get a *ladder* (``rung``): one row
of cells across the wedge with sites on its bisector. A spike sharper than
``sharp_angle`` loses the last bit of its tip, which lies inside a cell.

*Sizing* (``sizing``). The face length follows vorflow's size field, capped
by the distance to other barrier parts, the domain boundary and feature
nodes (so circles cannot reach them) and graded along the network.

*Repair* (``pipeline``). Where the gate fails, the faces are halved locally; a
construction that is still invalid after a few rounds raises
``BarrierMeshError`` naming the place, instead of returning a bad mesh.
"""
from .pipeline import BarrierMeshError, BarrierPlan, plan

__all__ = ["BarrierMeshError", "BarrierPlan", "plan"]
