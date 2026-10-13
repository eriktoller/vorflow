from __future__ import annotations

import logging
import math
import operator

import shapely

logger = logging.getLogger(__name__)


DEFAULT_GROWTH_FACTOR = 1.2
_GROWTH_MODELS = {"edge_ratio", "continuous_metric"}


def _positive_finite(value, name):
    """Return ``value`` as a positive finite float or raise a clear error."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number. Got {value!r}.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number. Got {value!r}.")
    return value


def _positive_integer(value, name):
    """Return ``value`` as a positive integer without silently truncating it."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer. Got {value!r}.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer. Got {value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {value!r}.")
    return int(value)


def _growth_gradient(growth_factor, growth_model):
    """Convert an adjacent-size growth factor to a spatial size gradient."""
    try:
        growth_factor = float(growth_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"growth_factor must be a finite number greater than 1.0. Got {growth_factor!r}."
        ) from exc
    if not math.isfinite(growth_factor) or growth_factor <= 1.0:
        raise ValueError(
            f"growth_factor must be a finite number greater than 1.0. Got {growth_factor!r}."
        )
    if not isinstance(growth_model, str) or growth_model not in _GROWTH_MODELS:
        choices = ", ".join(sorted(_GROWTH_MODELS))
        raise ValueError(f"growth_model must be one of: {choices}. Got {growth_model!r}.")
    if growth_model == "edge_ratio":
        return growth_factor, growth_factor - 1.0
    return growth_factor, math.log(growth_factor)


def _hashable(value):
    """Return ``value`` if it can be hashed, else its repr."""
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class MeshField:
    """
    Base class for all mesh size fields.
    """
    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        """
        Creates the Gmsh field(s) and returns the field ID.
        
        Args:
            gmsh_api: The gmsh module.
            tags_dict (dict): Dictionary of tags {'points': [], 'lines': [], 'surfaces': []}.
                The engine also splits polygon surfaces into 'embedded_surfaces'
                and 'field_only_surfaces', and passes the field-only polygons'
                shapely geometries as 'field_only_polygons'.
            background_lc (float): Global background mesh size.
            feature_lc (float, optional): The target resolution of the specific feature group.
        """
        raise NotImplementedError("Subclasses must implement create()")

    def __eq__(self, other):
        """Equality check for grouping."""
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __hash__(self):
        """Hash for dictionary keys (unhashable attributes hash by repr)."""
        return hash((self.__class__.__name__, tuple(
            (name, _hashable(value)) for name, value in sorted(self.__dict__.items())
        )))


class DistanceField(MeshField):
    """Creates a Gmsh Distance field from points/lines/surfaces tags.

    Internal — not part of the public API. This is a raw building block
    (distance-to-feature, not a mesh size) combined by the size fields via
    _distance_tags_for_growth(). Its create() signature differs from
    MeshField's, so it cannot be passed as a user field via ``fields=``.
    """

    def __init__(self, include_surfaces=True, sampling=20):
        self.include_surfaces = bool(include_surfaces)
        self.sampling = int(sampling)

    def create(self, gmsh_api, tags_dict):
        f_dist = gmsh_api.model.mesh.field.add("Distance")

        has_entities = False
        if tags_dict.get('points'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "PointsList", tags_dict['points'])
            has_entities = True
        if tags_dict.get('lines'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "CurvesList", tags_dict['lines'])
            gmsh_api.model.mesh.field.setNumber(f_dist, 'Sampling', self.sampling)
            has_entities = True
        if self.include_surfaces and tags_dict.get('surfaces'):#TODO check if loops needed
            gmsh_api.model.mesh.field.setNumbers(f_dist, "SurfacesList", tags_dict['surfaces'])
            gmsh_api.model.mesh.field.setNumber(f_dist, 'Sampling', self.sampling)
            has_entities = True

        if not has_entities:
            gmsh_api.model.mesh.field.remove(f_dist)
            return None

        return f_dist


def _surface_boundary_curves(gmsh_api, surface_tags):
    """Return boundary curve tags for the given surface tags."""
    curves = []
    seen = set()
    for tag in surface_tags:
        try:
            boundary = gmsh_api.model.getBoundary(
                [(2, int(tag))],
                combined=False,
                oriented=False,
                recursive=False,
            )
        except Exception:
            boundary = []

        for dim, curve_tag in boundary:
            if int(dim) != 1:
                continue
            curve_tag = int(curve_tag)
            if curve_tag not in seen:
                seen.add(curve_tag)
                curves.append(curve_tag)
    return curves


def _embedded_surface_tags(tags_dict):
    """Embedded polygon surface tags (domain partitions that hold mesh nodes)."""
    embedded_surfaces = tags_dict.get("embedded_surfaces", None)
    if embedded_surfaces is None:
        embedded_surfaces = tags_dict.get("surfaces", [])
    return list(dict.fromkeys(int(t) for t in embedded_surfaces))


def _polygon_surface_tags(tags_dict):
    """Embedded and field-only polygon surface tags, embedded first, deduplicated."""
    field_only_surfaces = [int(t) for t in tags_dict.get("field_only_surfaces", [])]
    return list(dict.fromkeys(_embedded_surface_tags(tags_dict) + field_only_surfaces))


def _distance_tags_for_growth(gmsh_api, tags_dict, sampling):
    """Build tags for distance growth while keeping polygon interiors flat."""
    polygon_surfaces = _polygon_surface_tags(tags_dict)
    boundary_curves = _surface_boundary_curves(gmsh_api, polygon_surfaces)

    growth_tags = {
        "points": list(tags_dict.get("points", [])),
        "lines": list(tags_dict.get("lines", [])) + boundary_curves,
        "surfaces": [],
    }

    # If no boundary curves could be recovered, fall back to the old surface
    # distance behavior instead of dropping the field.
    if polygon_surfaces and not boundary_curves:
        logger.warning(
            "Warning: could not recover boundary curves for a polygon size "
            "field; falling back to surface-distance growth."
        )
        growth_tags["surfaces"].extend(polygon_surfaces)

    return DistanceField(include_surfaces=True, sampling=sampling).create(
        gmsh_api, growth_tags
    )


def _polygon_surface_constant(gmsh_api, surface_tags, size, background_lc):
    """Constant ``size`` on the given embedded surfaces, scoped by entity."""
    if not surface_tags:
        return None

    const = gmsh_api.model.mesh.field.add("Constant")
    gmsh_api.model.mesh.field.setNumber(const, "VIn", float(size))
    gmsh_api.model.mesh.field.setNumber(const, "VOut", float(background_lc))
    # SurfacesList matches the entity being meshed, so this only reaches
    # surfaces that are domain partitions (embedded polygons).
    gmsh_api.model.mesh.field.setNumbers(
        const, "SurfacesList", [float(t) for t in surface_tags]
    )
    return const


def _view_triangles(geometry, value):
    """List-data ("ST") entries for the constrained Delaunay triangles of ``geometry``."""
    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(geometry))
    data = []
    for triangle in triangles:
        (x0, y0), (x1, y1), (x2, y2) = list(triangle.exterior.coords)[:3]
        data.extend([x0, x1, x2, y0, y1, y2, 0.0, 0.0, 0.0, value, value, value])
    return len(triangles), data


def _field_only_interior(gmsh_api, polygons, size, background_lc):
    """Field holding field-only polygons at ``size``, located by position.

    A field-only polygon's gmsh surface is not fragmented into the domain,
    so domain nodes never lie on it and entity-scoped fields (Constant or
    Restrict with SurfacesList) never apply to them. Instead, a list-based
    view stores ``size`` on a triangulation of the polygons and
    ``background_lc`` on a triangulation of the rest of the model's
    bounding box, and a PostView field evaluates it. Covering the bounding
    box keeps gmsh from warning about every point outside the polygons.
    """
    polygons = shapely.union_all([p for p in polygons if p is not None and not p.is_empty])
    if polygons.is_empty or polygons.area <= 0.0:
        return None

    xmin, ymin, _, xmax, ymax, _ = gmsh_api.model.getBoundingBox(-1, -1)
    pxmin, pymin, pxmax, pymax = polygons.bounds
    xmin, ymin = min(xmin, pxmin), min(ymin, pymin)
    xmax, ymax = max(xmax, pxmax), max(ymax, pymax)
    margin = 0.01 * math.hypot(xmax - xmin, ymax - ymin) + float(background_lc)
    outside = shapely.box(xmin - margin, ymin - margin, xmax + margin, ymax + margin).difference(polygons)

    n_in, data_in = _view_triangles(polygons, float(size))
    n_out, data_out = _view_triangles(outside, float(background_lc))
    view = gmsh_api.view.add("vorflow field-only polygon interior")
    gmsh_api.view.addListData(view, "ST", n_in + n_out, data_in + data_out)

    field = gmsh_api.model.mesh.field
    f_view = field.add("PostView")
    field.setNumber(f_view, "ViewTag", view)
    # Outside the view (beyond the margin), return 0 -> no constraint rather
    # than the closest node's value.
    field.setNumber(f_view, "CropNegativeValues", 1)
    field.setNumber(f_view, "UseClosest", 0)
    return f_view


def _polygon_interior_size(gmsh_api, tags_dict, size, background_lc):
    """Field holding polygon interiors at ``size``, or None if there are none."""
    interiors = [
        f for f in (
            _polygon_surface_constant(
                gmsh_api, _embedded_surface_tags(tags_dict), size, background_lc
            ),
            _field_only_interior(
                gmsh_api, tags_dict.get("field_only_polygons", []), size, background_lc
            ),
        )
        if f is not None
    ]
    if len(interiors) < 2:
        return interiors[0] if interiors else None
    f_min = gmsh_api.model.mesh.field.add("Min")
    gmsh_api.model.mesh.field.setNumbers(f_min, "FieldsList", [float(f) for f in interiors])
    return f_min


def _combine_with_polygon_interior(
    gmsh_api, growth_field, tags_dict, size, background_lc
):
    interior = _polygon_interior_size(gmsh_api, tags_dict, size, background_lc)
    if interior is None:
        return growth_field
    if growth_field is None:
        return interior

    f_min = gmsh_api.model.mesh.field.add("Min")
    gmsh_api.model.mesh.field.setNumbers(
        f_min, "FieldsList", [float(growth_field), float(interior)]
    )
    return f_min

# --- Manual Fields ---

class ConstantField(MeshField):
    """Internal — not part of the public API.

    Used by the engine to set the global background size (which users control
    through ``background_lc``). Not useful as a per-feature field: create()
    ignores ``tags_dict``, so it cannot scope a size to a feature. Polygon
    interiors are held at their size by the internal
    _polygon_interior_size() helper, which scopes embedded surfaces with
    SurfacesList and locates field-only polygons geometrically.
    """

    def __init__(self, size):
        self.size = float(size)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        const = gmsh_api.model.mesh.field.add("Constant")
        gmsh_api.model.mesh.field.setNumber(const, "VIn", self.size)
        gmsh_api.model.mesh.field.setNumber(const, "VOut", background_lc)
        return const


class ThresholdField(MeshField):
    """Linear transition from ``size_min`` to ``size_max`` with distance.

    The size is ``size_min`` within ``dist_min`` of the feature, rises
    linearly to ``size_max`` at ``dist_max`` and stays there beyond it. This
    is the field behind the deprecated ``dist_min``/``dist_max`` keywords;
    ``GeometricGrowthField`` is the default.

    Args:
        size_min (float): Size at and near the feature.
        dist_min (float): Distance over which ``size_min`` is held.
        dist_max (float): Distance at which ``size_max`` is reached.
        size_max (float, optional): Far-field size. Defaults to the
            background size.
        sampling (int): Number of distance sample points per curve.
    """

    def __init__(self, size_min, dist_min, dist_max, size_max=None, sampling=20):
        self.size_min = float(size_min)
        self.dist_min = float(dist_min)
        self.dist_max = float(dist_max)
        self.size_max = float(size_max) if size_max is not None else None
        self.sampling = int(sampling)
    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        # 1. Distance field for growth away from features. For polygon
        # surfaces, use boundary curves for growth and add an interior
        # field below so the polygon interior remains flat.
        f_dist = _distance_tags_for_growth(gmsh_api, tags_dict, self.sampling)
        if f_dist is None:
            return _combine_with_polygon_interior(
                gmsh_api, None, tags_dict, self.size_min, background_lc
            )

        # 2. Threshold Field
        f_thresh = gmsh_api.model.mesh.field.add("Threshold")
        gmsh_api.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "SizeMin", self.size_min)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "SizeMax", self.size_max if self.size_max else background_lc)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "DistMin", self.dist_min)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "DistMax", self.dist_max)

        return _combine_with_polygon_interior(
            gmsh_api, f_thresh, tags_dict, self.size_min, background_lc
        )

class ExponentialField(MeshField):
    """Exponential transition controlled by a physical decay length.

    The field is ``H - (H - size_min) * exp(-d / decay_length)``. Its maximum
    gradient occurs at the feature and equals ``(H - size_min) / decay_length``.
    To keep that gradient below an edge-ratio bound ``r - 1``, choose
    ``decay_length >= (H - size_min) / (r - 1)``.
    """

    def __init__(self, size_min, decay_length, size_max=None, sampling=20):
        self.size_min = _positive_finite(size_min, "size_min")
        self.decay_length = _positive_finite(decay_length, "decay_length")
        self.size_max = (
            _positive_finite(size_max, "size_max") if size_max is not None else None
        )
        if self.size_max is not None and self.size_max < self.size_min:
            raise ValueError("size_max must be greater than or equal to size_min.")
        self.sampling = _positive_integer(sampling, "sampling")

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        background_lc = _positive_finite(background_lc, "background_lc")
        s_max = self.size_max if self.size_max is not None else background_lc
        if s_max < self.size_min:
            raise ValueError(
                "background_lc must be greater than or equal to size_min when size_max is omitted."
            )

        f_dist = _distance_tags_for_growth(gmsh_api, tags_dict, self.sampling)
        if f_dist is None:
            return _combine_with_polygon_interior(
                gmsh_api, None, tags_dict, self.size_min, background_lc
            )

        f_math = gmsh_api.model.mesh.field.add("MathEval")
        expr = f"{s_max} - ({s_max} - {self.size_min}) * Exp(-F{f_dist} / {self.decay_length})"
        gmsh_api.model.mesh.field.setString(f_math, "F", expr)
        return _combine_with_polygon_interior(
            gmsh_api, f_math, tags_dict, self.size_min, background_lc
        )

# --- Growth fields ---


class GeometricGrowthField(MeshField):
    """Grade target edge length away from a feature with a bounded gradient.

    ``growth_factor`` describes geometric characteristic-length growth by
    element index. The Gmsh field is deliberately linear in physical distance:

    - ``edge_ratio`` uses gradient ``growth_factor - 1``;
    - ``continuous_metric`` uses gradient ``log(growth_factor)``.

    Normal ``MeshGenerator`` use caps the result at ``background_lc`` through
    the engine's global ``Min`` field. A direct low-level call to ``create()``
    returns the uncapped growth field.
    """

    def __init__(
        self,
        growth_factor=DEFAULT_GROWTH_FACTOR,
        growth_model="edge_ratio",
        sampling=20,
    ):
        self.growth_factor, self.gradient = _growth_gradient(
            growth_factor, growth_model
        )
        self.growth_model = growth_model
        self.sampling = _positive_integer(sampling, "sampling")

    def create(
        self,
        gmsh_api,
        tags_dict,
        background_lc,
        feature_lc=None,
        sampling=None,
    ):
        if feature_lc is None:
            return None

        cs = _positive_finite(feature_lc, "feature_lc")
        cs_dom = _positive_finite(background_lc, "background_lc")
        if cs >= cs_dom:
            return None

        distance_sampling = (
            self.sampling
            if sampling is None
            else _positive_integer(sampling, "sampling")
        )

        f_dist = _distance_tags_for_growth(gmsh_api, tags_dict, distance_sampling)
        if f_dist is None:
            return _combine_with_polygon_interior(
                gmsh_api, None, tags_dict, cs, background_lc
            )

        f_math = gmsh_api.model.mesh.field.add("MathEval")
        gradient = format(self.gradient, ".15g")
        expr = f"{cs} + {gradient} * F{f_dist}"

        gmsh_api.model.mesh.field.setString(f_math, "F", expr)
        return _combine_with_polygon_interior(
            gmsh_api, f_math, tags_dict, cs, background_lc
        )


class _BorderGradingField(MeshField):
    """Fine polygon border graded up to the polygon resolution.

    Internal — backs the deprecated ``add_polygon(border_density=...)``,
    reproducing its pre-0.1 sizing. Inside the polygon the size ramps
    linearly from ``border_size`` at the boundary to the polygon resolution
    between ``dist_min`` and ``dist_max``; outside it grows from
    ``border_size`` at DEFAULT_GROWTH_FACTOR, so the border is fine on both
    sides.
    """

    def __init__(self, border_size, dist_min=0.0, dist_max=None, sampling=20):
        self.border_size = _positive_finite(border_size, "border_size")
        self.dist_min = float(dist_min)
        self.dist_max = None if dist_max is None else float(dist_max)
        self.sampling = _positive_integer(sampling, "sampling")

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        interior = float(background_lc) if feature_lc is None else min(float(feature_lc), float(background_lc))
        surfaces = _polygon_surface_tags(tags_dict)
        if self.border_size >= interior or not surfaces:
            return None
        curves = _surface_boundary_curves(gmsh_api, surfaces)
        if not curves:
            return None

        dist_max = self.dist_max
        if dist_max is None or dist_max <= self.dist_min:
            dist_max = max(
                self.dist_min + 5.0 * self.border_size,
                self.dist_min + 0.2 * (interior - self.border_size),
            )

        field = gmsh_api.model.mesh.field
        f_dist = DistanceField(include_surfaces=False, sampling=self.sampling).create(
            gmsh_api, {"lines": curves}
        )
        f_thresh = field.add("Threshold")
        field.setNumber(f_thresh, "InField", f_dist)
        field.setNumber(f_thresh, "SizeMin", self.border_size)
        field.setNumber(f_thresh, "SizeMax", interior)
        field.setNumber(f_thresh, "DistMin", self.dist_min)
        field.setNumber(f_thresh, "DistMax", dist_max)

        f_inside = field.add("Restrict")
        field.setNumber(f_inside, "InField", f_thresh)
        field.setNumbers(f_inside, "SurfacesList", [float(t) for t in surfaces])

        gradient = format(DEFAULT_GROWTH_FACTOR - 1.0, ".15g")
        f_outside = field.add("MathEval")
        field.setString(f_outside, "F", f"{self.border_size} + {gradient} * F{f_dist}")

        f_min = field.add("Min")
        field.setNumbers(f_min, "FieldsList", [float(f_inside), float(f_outside)])
        return f_min
