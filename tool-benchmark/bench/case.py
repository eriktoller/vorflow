"""Tool-neutral benchmark case: geometry, sizing spec and MF6 problems.

A case is one YAML file in ``tool-benchmark/cases``. Geometry is inline WKT (small
synthetic cases), ``domain: {circle: [x, y, r], segments: n}`` for a circle, or
``{file: <path>, layer: <name>}`` for a vector file (relative to
``tool-benchmark/``; the other tools' examples, which ``fetch.py`` downloads into
``data/``). Every feature is clipped to the domain on load, so all tools see
exactly the same inputs.

A feature read from a file becomes one feature per point or polygon part (ids
``<id>-0``, ``<id>-1``, ...); its lines are merged into one. Without ``h`` in
the YAML, each row takes its size from the file's ``h`` column. A line with
``boundary: true`` lies on the domain boundary and sets the size there
(VOROGRIDGEN's per-vertex boundary spacing); adapters apply it to the boundary
instead of adding it as an interior line.

``target_ncpl`` is one cell count or a list (a convergence sweep); the first
target is the one plotted. ``mf6_problems`` lists problem names, or
``{name: ..., <parameter>: ...}`` mappings for problems that take parameters.
Optional ``transects`` (WKT lines) cross narrow gaps between features; the
metrics count the cells each one passes through.

Sizing spec (see docs/benchmark-plan.md): a background size ``h_max``, a target
size ``h`` per feature and one cell-to-cell growth ratio ``growth``. Geometric
growth of successive cells by ``g`` means the size grows *linearly* with
distance d from a feature:

    h(x) = min(h_max, min_f(h_f + (g - 1) * d_f(x)))

``scale`` multiplies every length in the spec (the matched-count knob); the
growth ratio is never scaled.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
import yaml
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point, Polygon

FEATURE_KINDS = ("point", "line", "polygon")


@dataclass
class Feature:
    id: str
    kind: str
    geometry: object
    h: float
    barrier: bool = False
    boundary: bool = False


@dataclass
class Case:
    id: str
    tier: int
    description: str
    source: str
    crs: str
    domain: Polygon
    h_max: float
    growth: float
    targets: tuple
    features: list = field(default_factory=list)
    mf6_problems: list = field(default_factory=list)
    transects: list = field(default_factory=list)

    def features_of(self, kind: str) -> list:
        """Features of one kind, in file order."""
        assert kind in FEATURE_KINDS, f"unknown feature kind {kind!r}"
        return [f for f in self.features if f.kind == kind]

    def feature(self, feature_id: str) -> Feature:
        """The feature with this id."""
        matches = [f for f in self.features if f.id == feature_id]
        assert len(matches) == 1, f"{self.id}: expected one feature {feature_id!r}, found {len(matches)}"
        return matches[0]


def load_case(path: Path) -> Case:
    """Read a case YAML file and clip its features to the domain."""
    spec = yaml.safe_load(Path(path).read_text())
    base_d = Path(path).resolve().parent.parent
    domain = _load_domain(spec["domain"], base_d)
    assert isinstance(domain, Polygon) and domain.is_valid, f"{path}: domain must be one valid Polygon"
    assert spec["growth"] > 1.0, f"{path}: growth must be > 1"
    features = [f for item in spec.get("features", []) for f in _load_features(item, domain, base_d)]
    targets = spec["target_ncpl"]
    targets = tuple(int(t) for t in (targets if isinstance(targets, list) else [targets]))
    return Case(
        id=spec["id"],
        tier=spec["tier"],
        description=spec["description"].strip(),
        source=spec["source"],
        crs=spec["crs"],
        domain=domain,
        h_max=float(spec["h_max"]),
        growth=float(spec["growth"]),
        targets=targets,
        features=features,
        mf6_problems=[_load_problem(item) for item in spec.get("mf6_problems", [])],
        transects=[wkt.loads(item) for item in spec.get("transects", [])],
    )


def _load_domain(item, base_d: Path) -> Polygon:
    """Domain polygon from WKT, a {circle: [x, y, r], segments: n} mapping or a {file:, layer:} mapping."""
    if isinstance(item, str):
        return wkt.loads(item)
    if "file" in item:
        return shapely.union_all(list(_read_file(item, base_d).geometry))
    x, y, r = item["circle"]
    n = int(item.get("segments", 128))
    assert n % 4 == 0, "circle segments must be a multiple of 4"
    return Point(x, y).buffer(r, quad_segs=n // 4)


def _load_problem(item) -> dict:
    """An MF6 problem entry as {'name': ..., parameters...}."""
    return {"name": item} if isinstance(item, str) else dict(item)


def _load_features(item: dict, domain: Polygon, base_d: Path) -> list:
    """Parse one feature entry into features clipped to the domain (none if it falls outside)."""
    kind = item["kind"]
    assert kind in FEATURE_KINDS, f"feature {item['id']}: unknown kind {kind!r}"
    if "file" not in item:
        return _clipped(item["id"], kind, wkt.loads(item["wkt"]), float(item["h"]), item, domain)
    rows = _read_file(item, base_d)
    sizes = [float(item["h"])] * len(rows) if "h" in item else [float(h) for h in rows["h"]]
    if kind == "line" and len(set(sizes)) == 1:
        geom = shapely.line_merge(shapely.union_all(list(rows.geometry)))
        return _clipped(item["id"], kind, geom, sizes[0], item, domain)
    parts = [(part, h) for geom, h in zip(rows.geometry, sizes) for part in getattr(geom, "geoms", [geom])]
    features = []
    for i, (part, h) in enumerate(parts):
        features += _clipped(f"{item['id']}-{i}", kind, part, h, item, domain)
    return features


def _clipped(feature_id: str, kind: str, geom, h: float, item: dict, domain: Polygon) -> list:
    """Features of one geometry clipped to the domain: one per polygon part, else one."""
    geom = geom.intersection(domain)
    if geom.is_empty:
        return []
    flags = {"barrier": bool(item.get("barrier", False)), "boundary": bool(item.get("boundary", False))}
    if kind == "polygon":
        parts = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon) and g.area > 0]
        ids = [feature_id] if len(parts) == 1 else [f"{feature_id}.{i}" for i in range(len(parts))]
        return [Feature(id=i, kind=kind, geometry=g, h=h, **flags) for i, g in zip(ids, parts)]
    if kind == "line":
        lines = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, (LineString, MultiLineString))]
        geom = shapely.line_merge(shapely.union_all(lines))
        assert isinstance(geom, (LineString, MultiLineString)), f"feature {feature_id}: clipped to {geom.geom_type}"
    if kind == "point":
        assert isinstance(geom, Point), f"feature {feature_id}: points must be single Points"
    return [Feature(id=str(feature_id), kind=kind, geometry=geom, h=h, **flags)]


def _read_file(item: dict, base_d: Path):
    """GeoDataFrame of a {file:, layer:} entry; tells the user to fetch it if missing."""
    import geopandas as gpd

    path = base_d / item["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python fetch.py` in tool-benchmark/")
    return gpd.read_file(path, layer=item.get("layer"))


def line_parts(geom) -> list:
    """Single LineStrings of a (Multi)LineString."""
    return list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]


def size_field(case: Case, xy: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Target cell size of the spec at points xy (N, 2)."""
    pts = shapely.points(np.asarray(xy, dtype=float))
    h = np.full(len(pts), case.h_max * scale)
    for feature in case.features:
        d = shapely.distance(feature.geometry, pts)
        h = np.minimum(h, feature.h * scale + (case.growth - 1.0) * d)
    return h
