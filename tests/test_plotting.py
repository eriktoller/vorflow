import sys

import pytest
from shapely.geometry import LineString, Point, Polygon

from vorflow.blueprint import ConceptualMesh
from vorflow.tessellator import VoronoiTessellator

from test_pipeline import FakeMeshGenerator

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _model():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, resolution=2)
    cm.add_polygon(Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), zone_id=2, resolution=1, z_order=1)
    cm.add_polygon(Polygon([(6, 6), (8, 6), (8, 8), (6, 8)]), zone_id=3, resolution=1, embed=False)
    cm.add_line(LineString([(1, 8), (9, 8)]), line_id="river", resolution=1)
    cm.add_line(LineString([(5, 1), (5, 9)]), line_id="fault", resolution=1, embed=False)
    cm.add_point(Point(7, 3), point_id="well", resolution=0.5)
    return cm


@pytest.mark.parametrize("clean", [False, True])
def test_conceptual_mesh_plot_draws_all_feature_types(clean):
    cm = _model()
    if clean:
        cm.generate()

    ax = cm.plot(clean=clean)

    assert ax.collections  # polygons, lines and points are all collections
    assert ax.get_aspect() == 1.0


def test_conceptual_mesh_plot_clean_before_generate_raises():
    with pytest.raises(RuntimeError, match="generate"):
        _model().plot(clean=True)


def test_conceptual_mesh_plot_uses_given_axes():
    _, ax = plt.subplots()
    assert _model().plot(ax=ax, column=None) is ax


def _labelled_model():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, label="Region")
    cm.add_polygon(Polygon([(6, 6), (8, 6), (8, 8), (6, 8)]), zone_id=2, embed=False, label="Refine")
    cm.add_line(LineString([(5, 0), (5, 8)]), line_id="fault", resolution=1, is_barrier=True,
                label="Fault")
    cm.add_line(LineString([(0, 2), (10, 2)]), line_id="left", resolution=1, label="Boundary")
    cm.add_line(LineString([(0, 4), (10, 4)]), line_id="right", resolution=1, label="Boundary")
    cm.add_line(LineString([(0, 9), (10, 9)]), line_id="unlabelled", resolution=1)
    cm.add_point(Point(2, 6), point_id="well", resolution=0.5, label="Well")
    return cm


@pytest.mark.parametrize("clean", [False, True])
def test_conceptual_mesh_plot_gives_one_legend_entry_per_label(clean):
    cm = _labelled_model()
    if clean:
        cm.generate()

    legend = cm.plot(clean=clean).legend()

    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["Region", "Refine", "Fault", "Boundary", "Well"]


def test_conceptual_mesh_plot_polygon_legend_entry_matches_fill():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, label="A")
    cm.add_polygon(Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), zone_id=2, z_order=1, label="B")

    ax = cm.plot()

    fills = {tuple(fill) for collection in ax.collections for fill in collection.get_facecolor()}
    handles, labels = ax.get_legend_handles_labels()
    assert labels == ["A", "B"]
    handle_fills = {tuple(handle.get_facecolor()) for handle in handles}
    assert len(handle_fills) == 2
    assert handle_fills == fills


@pytest.mark.parametrize("plot_kwargs, expected", [
    ({"column": None}, "C0"),
    ({"column": None, "color": "wheat"}, "wheat"),
    ({"facecolor": "none"}, "none"),
])
def test_conceptual_mesh_plot_single_colour_polygon_legend_entry(plot_kwargs, expected):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, label="A")

    handles, _ = cm.plot(alpha=1, **plot_kwargs).get_legend_handles_labels()

    assert tuple(handles[0].get_facecolor()) == pytest.approx(matplotlib.colors.to_rgba(expected))


@pytest.mark.parametrize("clean", [False, True])
def test_conceptual_mesh_plot_uses_line_colour(clean):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1)
    cm.add_line(LineString([(0, 2), (10, 2)]), line_id="river", resolution=1, label="River",
                color="navy")
    cm.add_line(LineString([(0, 6), (10, 6)]), line_id="drain", resolution=1, label="Drain",
                color=(0.0, 0.5, 0.0))
    cm.add_line(LineString([(5, 0), (5, 9)]), line_id="fault", resolution=1, is_barrier=True,
                label="Fault")
    if clean:
        cm.generate()

    handles, labels = cm.plot(clean=clean).get_legend_handles_labels()

    colors = {label: tuple(handle.get_color()[0]) for handle, label in zip(handles, labels)}
    to_rgba = matplotlib.colors.to_rgba
    assert colors == {
        "River": pytest.approx(to_rgba("navy")),
        "Drain": pytest.approx(to_rgba((0.0, 0.5, 0.0))),
        "Fault": pytest.approx(to_rgba("tab:red")),
    }


def test_conceptual_mesh_plot_without_labels_has_no_legend_entries():
    _, labels = _model().plot().get_legend_handles_labels()
    assert labels == []


def test_tessellator_plot_draws_cells_and_generators():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), zone_id=99, densify=0.5)
    clean_polys, _, _ = cm.generate()
    tessellator = VoronoiTessellator(FakeMeshGenerator(clean_polys), cm)
    tessellator.generate()

    ax = tessellator.plot(show_generators=True)

    assert len(ax.collections) == 2
    assert len(ax.collections[1].get_offsets()) == len(tessellator.grid)


def test_tessellator_plot_before_generate_raises():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), zone_id=99)
    clean_polys, _, _ = cm.generate()
    with pytest.raises(RuntimeError, match="generate"):
        VoronoiTessellator(FakeMeshGenerator(clean_polys), cm).plot()


def test_plot_without_matplotlib_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"vorflow\[plot\]"):
        _model().plot()


@pytest.mark.slow
def test_mesh_generator_plot_draws_elements():
    from vorflow.engine import MeshGenerator

    mesher = MeshGenerator(background_lc=2, verbosity=0, conceptual_mesh=_model())
    mesher.generate()

    ax = mesher.plot(column="zone_id")

    assert ax.collections
