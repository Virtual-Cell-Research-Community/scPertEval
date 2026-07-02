# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from sphinxcontrib import katex

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

# -- Project information -----------------------------------------------------

info = metadata("scperteval")
project = info["Name"]
author = info.get("Author") or "scPertEval authors"
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
_project_urls = info.get_all("Project-URL") or []
urls = dict(pu.split(", ", 1) for pu in _project_urls)
repository_url = urls.get("Source", "https://github.com/Virtual-Cell-Research-Community/scPertEval")

release = info["Version"]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
templates_path = ["_templates"]
nitpicky = True
needs_sphinx = "4.0"

html_context = {
    "display_github": True,
    "github_user": "Virtual-Cell-Research-Community",
    "github_repo": "scPertEval",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"
always_use_bars_union = True

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "notebooks/README.md"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": True,
    "path_to_docs": "docs/",
    # Tutorials are committed as executed .ipynb (outputs saved), so a launch button makes the
    # runnable notebook downloadable and gives a valid "Open in Colab" link.
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "navigation_with_keys": False,
    "show_navbar_depth": 1,
}

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [  # type: ignore
    # Add exceptions here for links outside your control that fail to resolve
    ("py:class", "Context"),
    ("py:class", "Dataset"),
    # Internal classes referenced in type hints but not given their own API page.
    ("py:class", "scperteval.reference.Reference"),
]
