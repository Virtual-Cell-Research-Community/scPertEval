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
_credited = (
    (info.get_all("Author") or []) + (info.get_all("Maintainer") or []) + (info.get_all("Maintainer-email") or [])
)
_names = dict.fromkeys(entry.split("<")[0].strip() for entry in _credited)
author = ", ".join(_names) or "scPertEval authors"
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
    "sphinxcontrib.mermaid",
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
    "alert",
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
mermaid_d3_zoom = True
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
nb_render_markdown_format = "myst"
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
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/logo/scPertEval-favicon.svg"

html_title = project

html_theme_options = {
    "logo": {
        "image_light": "_static/logo/scPertEval-logo.svg",
        "image_dark": "_static/logo/scPertEval-dark-logo.svg",
    },
    "repository_url": repository_url,
    "repository_branch": "main",
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "navigation_with_keys": False,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
}

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [  # type: ignore
    # Add exceptions here for links outside your control that fail to resolve
    ("py:class", "Context"),
    ("py:class", "Dataset"),
    # Internal classes referenced in type hints but not given their own API page.
    ("py:class", "scperteval.reference.Reference"),
    ("py:class", "CacheStore"),
    ("py:class", "scperteval.context.CacheStore"),
    # NumPy-docstring type modifiers that are text, not resolvable classes.
    ("py:class", "optional"),
    ("py:class", "sequence"),
    # The `pd` alias in annotations doesn't resolve to the `pandas.DataFrame` intersphinx target.
    ("py:class", "pd.DataFrame"),
]
