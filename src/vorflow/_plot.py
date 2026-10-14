"""Optional Matplotlib plotting behind the ``plot()`` methods.

Matplotlib is not a dependency of vorflow: it is imported only when a
``plot()`` method is called, and installed with the ``plot`` extra.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from ._features import is_embedded, row_bool

# Columns the raw feature dicts are reduced to before plotting; the rest
# (fields, refinement settings) cannot go into a GeoDataFrame cleanly.
RAW_PLOT_COLUMNS = ("geometry", "zone_id", "embed", "is_barrier", "label")


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


def label_groups(gdf):
    """``(label, positions)`` of ``gdf`` rows grouped by their ``label`` column.

    Unlabelled rows come first with label None; labelled groups follow in
    order of first appearance, so legend entries follow insertion order.
    """
    labels = gdf["label"] if "label" in gdf.columns else pd.Series(None, index=gdf.index)
    groups = {}
    for position, label in enumerate(labels):
        groups.setdefault(None if pd.isna(label) else str(label), []).append(position)
    unlabelled = groups.pop(None, [])
    return ([(None, unlabelled)] if unlabelled else []) + list(groups.items())


def plot_by_label(gdf, ax, colors=None, **kwargs):
    """Plot ``gdf`` with one artist per label group, so each label gets a legend entry.

    ``colors`` (optional) holds one colour per row of ``gdf``.
    """
    for label, positions in label_groups(gdf):
        group_kwargs = dict(kwargs)
        if colors is not None:
            group_kwargs["color"] = [colors[i] for i in positions]
        if label is not None:
            group_kwargs["label"] = label
        gdf.iloc[positions].plot(ax=ax, **group_kwargs)


def row_colors(collection, gdf):
    """Face colour of each ``gdf`` row in the polygon ``collection`` drawn from it, or None.

    GeoPandas draws one patch per polygon part, so rows are matched to their
    first part. None when the patches cannot be matched to rows (e.g. rows
    left out for a missing ``column`` value).
    """
    facecolors = collection.get_facecolor()
    if len(facecolors) == 1:
        return [facecolors[0]] * len(gdf)
    part_counts = [len(geom.geoms) if hasattr(geom, "geoms") else 1 for geom in gdf.geometry]
    if len(facecolors) != sum(part_counts):
        return None
    first_parts = pd.Series(part_counts).cumsum() - part_counts
    return [facecolors[i] for i in first_parts]


def add_polygon_legend_entries(ax, collection, gdf):
    """Add a legend entry per label of the polygons ``gdf`` drawn as ``collection``.

    Matplotlib cannot use a polygon ``PatchCollection`` as a legend handle,
    so each label gets an empty patch styled like the label's first polygon.
    """
    colors = row_colors(collection, gdf)
    edgecolors = collection.get_edgecolor()
    linewidths = collection.get_linewidth()
    for label, positions in label_groups(gdf):
        if label is None:
            continue
        patch_kwargs = {"label": label, "linewidth": linewidths[0] if len(linewidths) else None}
        if colors is not None:
            patch_kwargs["facecolor"] = colors[positions[0]]
        if len(edgecolors):
            patch_kwargs["edgecolor"] = edgecolors[0]
        ax.fill([], [], **patch_kwargs)


def plot_features(ax, polygons, lines, points, column="zone_id", **kwargs):
    """Draw conceptual-model features on ``ax``; returns ``ax``.

    Embedded polygons are filled by ``column`` (``kwargs`` go to their
    ``GeoDataFrame.plot``); lines and points are drawn on top, barrier lines
    in red. Field-only (``embed=False``) features are dashed outlines, dashed
    lines and hollow points, since they only size the mesh. Features with a
    ``label`` get one legend entry per label, shown by ``ax.legend()``.
    """
    embedded, field_only = split_embedded(polygons)
    if not embedded.empty:
        polygon_kwargs = {"edgecolor": "black", "linewidth": 0.8, "alpha": 0.6, **kwargs}
        if column is not None and column in embedded.columns:
            polygon_kwargs.setdefault("categorical", True)
            polygon_kwargs.setdefault("cmap", "Pastel1")
            polygon_kwargs["column"] = column
        n_collections = len(ax.collections)
        embedded.plot(ax=ax, **polygon_kwargs)
        if len(ax.collections) > n_collections:
            collection = ax.collections[n_collections]
            # Some GeoPandas versions label the collection with a category
            # value; the proxy patches are the polygon legend entries.
            collection.set_label("_nolegend_")
            add_polygon_legend_entries(ax, collection, embedded)
    if not field_only.empty:
        outlines = field_only.assign(geometry=field_only.boundary)
        plot_by_label(outlines, ax, color="grey", linestyle="--", linewidth=0.8)

    embedded, field_only = split_embedded(lines)
    if not embedded.empty:
        colors = ["tab:red" if row_bool(row, "is_barrier") else "tab:blue"
                  for _, row in embedded.iterrows()]
        plot_by_label(embedded, ax, colors=colors, linewidth=1.5)
    if not field_only.empty:
        plot_by_label(field_only, ax, color="tab:blue", linestyle="--", linewidth=1.0)

    embedded, field_only = split_embedded(points)
    if not embedded.empty:
        plot_by_label(embedded, ax, color="black", markersize=12)
    if not field_only.empty:
        plot_by_label(field_only, ax, facecolor="none", edgecolor="black", markersize=12)
    return ax
