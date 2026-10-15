"""Sphinx configuration for the vorflow documentation."""

import os
import shutil
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent
REPO_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = REPO_ROOT / "examples"
EXAMPLES_DST = DOCS_DIR / "examples"

# Import the checkout, not another installed vorflow. PYTHONPATH carries the
# same choice into the kernels that execute the notebooks.
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(None, [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH")])
)

import vorflow  # noqa: E402


def _copy_notebooks(src, dst):
    """Copy the example notebooks into the source tree.

    Sphinx only builds files under ``docs/``; the notebooks stay in
    ``examples/`` so they remain runnable from the repository root.
    """
    dst.mkdir(exist_ok=True)
    for notebook in sorted(src.glob("*.ipynb")):
        shutil.copy2(notebook, dst / notebook.name)


_copy_notebooks(EXAMPLES_SRC, EXAMPLES_DST)

project = "vorflow"
author = "Rui Hugman and Oscar Sanchez"
copyright = "vorflow developers"
release = vorflow.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = [
    "_build",
    "design",
    "benchmark-plan.md",
    "make_splash.py",
    "**.ipynb_checkpoints",
]
templates_path = ["_templates"]

# -- API reference ----------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_custom_sections = [("Deprecated parameters", "params_style")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

# -- Notebooks --------------------------------------------------------------
# Notebooks are committed without outputs, so they run during the build.
nb_execution_mode = "cache"
nb_execution_timeout = 900
nb_execution_raise_on_error = True
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

# -- HTML -------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"vorflow {release}"
html_logo = "images/vorflow-banner-network.png"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/rhugman/vorflow",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/vorflow/",
            "icon": "fa-brands fa-python",
        },
    ],
}

# The Pages deploy builds develop into /dev/ with VORFLOW_DOCS_CHANNEL=dev.
if os.environ.get("VORFLOW_DOCS_CHANNEL") == "dev":
    html_theme_options["announcement"] = (
        "Development docs for the unreleased <code>develop</code> branch. "
        '<a href="https://rhugman.github.io/vorflow/">Stable docs</a>'
    )
