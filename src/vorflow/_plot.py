"""Optional Matplotlib plotting behind the ``plot()`` methods.

Matplotlib is not a dependency of vorflow: it is imported only when a
``plot()`` method is called, and installed with the ``plot`` extra.
"""
from __future__ import annotations

import geopandas as gpd

from ._features import is_embedded, row_bool

# Columns the raw feature dicts are reduced to before plotting; the rest
# (fields, refinement settings) cannot go into a GeoDataFrame cleanly.
RAW_PLOT_COLUMNS = ("geometry", "zone_id", "embed", "is_barrier")


def require_matplotlib():
    """Import and return ``matplotlib.pyplot``, with an install hint if it is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "Plotting needs Matplotlib; install it with `pip install vorflow[plot]`."
        ) from err
    return plt


def get_ax(ax=None):
    """``ax``, or the axes of a new figure when None, set to an equal aspect ratio."""
    plt = require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect("equal")
    return ax


def features_frame(features, crs):
    """GeoDataFrame of raw feature dicts (as stored by ``ConceptualMesh.add_*``), reduced to ``RAW_PLOT_COLUMNS``."""
    if not features:
        return gpd.GeoDataFrame()
    rows = [{key: feature[key] for key in RAW_PLOT_COLUMNS if key in feature} for feature in features]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def split_embedded(gdf):
    """(embedded, field_only) rows of ``gdf``."""
    if gdf.empty:
        return gdf, gdf
    mask = gdf.apply(is_embedded, axis=1).astype(bool)
    return gdf[mask], gdf[~mask]


def plot_features(ax, polygons, lines, points, column="zone_id", **kwargs):
    """Draw conceptual-model features on ``ax``; returns ``ax``.

    Embedded polygons are filled by ``column`` (``kwargs`` go to their
    ``GeoDataFrame.plot``); lines and points are drawn on top, barrier lines
    in red. Field-only (``embed=False``) features are dashed outlines, dashed
    lines and hollow points, since they only size the mesh.
    """
    embedded, field_only = split_embedded(polygons)
    if not embedded.empty:
        polygon_kwargs = {"edgecolor": "black", "linewidth": 0.8, "alpha": 0.6, **kwargs}
        if column is not None and column in embedded.columns:
            polygon_kwargs.setdefault("categorical", True)
            polygon_kwargs.setdefault("cmap", "Pastel1")
            embedded.plot(ax=ax, column=column, **polygon_kwargs)
        else:
            embedded.plot(ax=ax, **polygon_kwargs)
    if not field_only.empty:
        field_only.boundary.plot(ax=ax, color="grey", linestyle="--", linewidth=0.8)

    embedded, field_only = split_embedded(lines)
    if not embedded.empty:
        colors = ["tab:red" if row_bool(row, "is_barrier") else "tab:blue"
                  for _, row in embedded.iterrows()]
        embedded.plot(ax=ax, color=colors, linewidth=1.5)
    if not field_only.empty:
        field_only.plot(ax=ax, color="tab:blue", linestyle="--", linewidth=1.0)

    embedded, field_only = split_embedded(points)
    if not embedded.empty:
        embedded.plot(ax=ax, color="black", markersize=12)
    if not field_only.empty:
        field_only.plot(ax=ax, facecolor="none", edgecolor="black", markersize=12)
    return ax
