# vorflow

Voronoi mesh generation for MODFLOW 6 using Gmsh and GeoPandas.

`vorflow` turns a conceptual model (Shapely polygons, lines and points) into a
2D unstructured Voronoi grid for MODFLOW 6. Gmsh builds a triangular mesh that
honours the features, and the Voronoi grid is its dual.

```{image} images/vorflow-banner-network.png
:alt: vorflow Voronoi grid refined along a stream network, around wells and inside a circular zone
```

```{include} ../README.md
:start-after: <!-- docs:overview-start -->
:end-before: <!-- docs:installation-start -->
```

```{toctree}
:hidden:
:maxdepth: 2

user_guide
examples
api
changelog
```
