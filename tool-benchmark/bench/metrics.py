"""Grid metrics computed from the DISV geometry, as MODFLOW 6 reads it.

Faces are the vertex pairs of ``iverts``: an edge used by two cells is a
connection, an edge used by one cell is on the model boundary. This is the
same topology MF6 builds (``DisvGeom%shared_edge``), so hanging nodes and
unmatched vertices show up here exactly as MF6 would see them.

Per face, with centres c1, c2 and face endpoints a, b:

* orthogonality error = |90 - angle(c2 - c1, b - a)| in degrees. Without XT3D,
  MF6 divides the head difference by the perpendicular distances from each
  centre to the face; this is exact only when the error is 0.
* skewness = distance from the face midpoint to where the line c1-c2 crosses
  the face line, divided by the face length.
* short face = face length below ``SHORT_FACE`` x connection length.

A one-cell edge away from the domain boundary is a *disconnected face*: two
cells touch there but share no DISV edge (a hanging node or two vertices that
differ in the last digits), so MF6 treats it as a no-flow boundary. Edges of
zero length (two distinct vertices at one point) are counted separately; MF6
6.7 can crash on them.

Per cell: size h = sqrt(2 A / sqrt(3)) (the spacing of a regular hexagon of
area A, i.e. generator spacing), its ratio to the spec size at the centre,
compactness 4 pi A / P^2, convexity A / A(convex hull), and the centre offset
|generator - xc| / sqrt(A). Size and shape statistics use interior cells only:
boundary cells are clipped (half-cells where generators sit on the boundary).
Face statistics are also split into faces of two interior cells (``_inner``)
and faces with a boundary cell on either side (``_bnd``), where the worst
faces usually are.

A case's transects (lines across narrow gaps between features) report how
many cells each one passes through; the minimum is the number of cells across
the narrowest gap.
"""

import numpy as np
import shapely

from .case import Case, size_field
from .grid import Grid, cell_polygons, open_rings

SHORT_FACE = 0.01
NEAR_DUPLICATE = 1e-9
CONCAVE = 1e-6          # a cell is concave when its convexity is below 1 - CONCAVE
TRANSECT_MIN_FRACTION = 1e-6  # a cell counts as crossed when it holds this fraction of the transect


def face_table(grid: Grid) -> dict:
    """Interior faces (two cells) and boundary faces (one cell) of a grid."""
    owners = {}
    for c, ring in enumerate(open_rings(grid.iverts)):
        for a, b in zip(ring, ring[1:] + ring[:1]):
            owners.setdefault((min(a, b), max(a, b)), []).append(c)
    interior = [(k, v) for k, v in owners.items() if len(v) == 2]
    boundary = [(k, v) for k, v in owners.items() if len(v) == 1]
    return {
        "ci": np.array([v[0] for _, v in interior], dtype=int),
        "cj": np.array([v[1] for _, v in interior], dtype=int),
        "va": np.array([k[0] for k, _ in interior], dtype=int),
        "vb": np.array([k[1] for k, _ in interior], dtype=int),
        "boundary_cell": np.array([v[0] for _, v in boundary], dtype=int),
        "boundary_va": np.array([k[0] for k, _ in boundary], dtype=int),
        "boundary_vb": np.array([k[1] for k, _ in boundary], dtype=int),
        "n_nonmanifold": sum(1 for v in owners.values() if len(v) > 2),
    }


def disconnected_faces(grid: Grid, faces: dict, case: Case, rel_tol: float = NEAR_DUPLICATE) -> np.ndarray:
    """True for one-cell edges of non-zero length off the domain boundary: neighbours MF6 will not connect."""
    tol = rel_tol * np.ptp(grid.vertices, axis=0).max()
    a, b = grid.vertices[faces["boundary_va"]], grid.vertices[faces["boundary_vb"]]
    off_boundary = shapely.distance(case.domain.boundary, shapely.points(0.5 * (a + b))) > tol
    return off_boundary & (np.hypot(*(b - a).T) > tol)


def zero_length_edges(grid: Grid, rel_tol: float = NEAR_DUPLICATE) -> int:
    """Cell edges shorter than rel_tol x grid extent (distinct vertices at one point)."""
    tol = rel_tol * np.ptp(grid.vertices, axis=0).max()
    count = 0
    for ring in open_rings(grid.iverts):
        pts = grid.vertices[ring]
        count += int(np.sum(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T) < tol))
    return count


def near_duplicate_vertices(grid: Grid, rel_tol: float = NEAR_DUPLICATE) -> int:
    """Vertices within rel_tol x grid extent of another vertex: distinct to MF6, so their edges don't connect."""
    tol = rel_tol * np.ptp(grid.vertices, axis=0).max()
    rounded = np.round(grid.vertices / tol).astype(np.int64)
    return int(len(grid.vertices) - len(np.unique(rounded, axis=0)))


def boundary_cells(grid: Grid, faces: dict, case: Case | None = None) -> np.ndarray:
    """True for cells owning a one-cell edge; with a case, only edges on the domain boundary."""
    on_domain = np.ones(len(faces["boundary_cell"]), dtype=bool)
    if case is not None:
        on_domain = ~disconnected_faces(grid, faces, case)
    mask = np.zeros(grid.ncpl, dtype=bool)
    mask[faces["boundary_cell"][on_domain]] = True
    return mask


def face_metrics(grid: Grid, faces: dict, centres: str) -> dict:
    """Orthogonality error (deg), skewness and short-face flag per interior face."""
    xy = grid.centres(centres)
    a, b = grid.vertices[faces["va"]], grid.vertices[faces["vb"]]
    c1, c2 = xy[faces["ci"]], xy[faces["cj"]]
    face, conn = b - a, c2 - c1
    face_len = np.hypot(*face.T)
    conn_len = np.hypot(*conn.T)
    cos = np.abs(np.sum(face * conn, axis=1)) / (face_len * conn_len)
    ortho = np.degrees(np.arcsin(np.clip(cos, 0.0, 1.0)))
    return {
        "ortho_deg": ortho,
        "skewness": _skewness(a, face, c1, conn, face_len),
        "short": face_len < SHORT_FACE * conn_len,
    }


def _skewness(a, face, c1, conn, face_len) -> np.ndarray:
    """|crossing point - face midpoint| / face length; NaN where c1-c2 is parallel to the face."""
    denom = face[:, 0] * conn[:, 1] - face[:, 1] * conn[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((c1[:, 0] - a[:, 0]) * conn[:, 1] - (c1[:, 1] - a[:, 1]) * conn[:, 0]) / denom
    return np.where(np.abs(denom) > 0, np.abs(t - 0.5), np.nan)


def cell_metrics(grid: Grid, case: Case, scale: float) -> dict:
    """Size, spec ratio, compactness and centre offset per cell."""
    polys = cell_polygons(grid)
    area = shapely.area(polys)
    perimeter = shapely.length(polys)
    h_cell = np.sqrt(2.0 * area / np.sqrt(3.0))
    metrics = {
        "area": area,
        "h": h_cell,
        "h_ratio": h_cell / size_field(case, grid.xc, scale),
        "compactness": 4.0 * np.pi * area / perimeter**2,
        "convexity": area / shapely.area(shapely.convex_hull(polys)),
        "valid": shapely.is_valid(polys),
        "clockwise": np.array([not p.exterior.is_ccw for p in polys]),
        "centre_inside": shapely.intersects_xy(polys, grid.xc[:, 0], grid.xc[:, 1]),
    }
    if grid.generators is not None:
        metrics["centre_offset"] = np.hypot(*(grid.generators - grid.xc).T) / np.sqrt(area)
    return metrics


def cells_across(grid: Grid, transects: list) -> np.ndarray:
    """Number of cells each transect passes through (ignoring touches at its ends)."""
    polys = np.asarray(cell_polygons(grid), dtype=object)
    counts = []
    for line in transects:
        candidates = np.flatnonzero(shapely.intersects(polys, line))
        lengths = shapely.length(shapely.intersection(polys[candidates], line))
        counts.append(int(np.sum(lengths > TRANSECT_MIN_FRACTION * line.length)))
    return np.array(counts, dtype=int)


def summarize(grid: Grid, case: Case, scale: float) -> dict:
    """One row of scalar metrics for a grid (written centres, plus generators if known)."""
    faces = face_table(grid)
    cells = cell_metrics(grid, case, scale)
    inner = ~boundary_cells(grid, faces, case)  # boundary cells are clipped half-cells
    disconnected = disconnected_faces(grid, faces, case)
    row = {
        "ncpl": grid.ncpl,
        "nvert": grid.nvert,
        "n_invalid_cells": int(np.sum(~cells["valid"])),
        "frac_clockwise": float(np.mean(cells["clockwise"])),
        "n_centre_outside_cell": int(np.sum(~cells["centre_inside"])),
        "coverage_error": abs(cells["area"].sum() - case.domain.area) / case.domain.area,
        "n_disconnected_faces": int(disconnected.sum()),
        "n_nonmanifold_edges": faces["n_nonmanifold"],
        "n_near_duplicate_vertices": near_duplicate_vertices(grid),
        "n_zero_length_edges": zero_length_edges(grid),
        "h_ratio_p05": np.quantile(cells["h_ratio"][inner], 0.05),
        "h_ratio_p25": np.quantile(cells["h_ratio"][inner], 0.25),
        "h_ratio_p50": np.median(cells["h_ratio"][inner]),
        "h_ratio_p75": np.quantile(cells["h_ratio"][inner], 0.75),
        "h_ratio_p95": np.quantile(cells["h_ratio"][inner], 0.95),
        "compactness_p05": np.quantile(cells["compactness"][inner], 0.05),
        "convexity_min": cells["convexity"].min(),
        "n_concave_cells": int(np.sum(cells["convexity"] < 1.0 - CONCAVE)),
    }
    if case.transects:
        across = cells_across(grid, case.transects)
        row["cells_across_min"] = int(across.min())
        row["cells_across_mean"] = float(across.mean())
    if "centre_offset" in cells:
        row["centre_offset_p50"] = np.median(cells["centre_offset"])
        row["centre_offset_p95"] = np.quantile(cells["centre_offset"], 0.95)
    tags = {"written": "mf6", "centroid": "cen"}
    if grid.generators is not None:
        tags["generator"] = "gen"
    inner_face = inner[faces["ci"]] & inner[faces["cj"]]
    for centres, tag in tags.items():
        fm = face_metrics(grid, faces, centres)
        row[f"ortho_{tag}_p50"] = np.median(fm["ortho_deg"])
        row[f"ortho_{tag}_p95"] = np.quantile(fm["ortho_deg"], 0.95)
        row[f"ortho_{tag}_max"] = fm["ortho_deg"].max()
        row[f"skew_{tag}_p50"] = np.nanmedian(fm["skewness"])
        row[f"skew_{tag}_p95"] = np.nanquantile(fm["skewness"], 0.95)
        row[f"frac_short_faces_{tag}"] = float(np.mean(fm["short"]))
        for part, mask in (("inner", inner_face), ("bnd", ~inner_face)):
            if mask.any():
                row[f"ortho_{tag}_p95_{part}"] = np.quantile(fm["ortho_deg"][mask], 0.95)
                row[f"skew_{tag}_p95_{part}"] = np.nanquantile(fm["skewness"][mask], 0.95)
    return row
