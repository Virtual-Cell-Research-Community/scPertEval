"""Collapse a heading and everything nested under it into a closed sphinx-design dropdown.

Docutils/MyST already nest a heading's following content into one ``nodes.section`` regardless
of which notebook cell it came from (markdown prose, code cells, cell outputs...), so marking the
*section* is enough -- no per-cell tagging needed (unlike notebook_cell_tabs.py's tags, which only
survive into the doctree on code cells; a tag on a markdown cell leaves no trace at all, since
myst-nb only attaches ``cell_metadata`` to code-cell container nodes).

Mark a section by writing a plain HTML comment right after its heading, in a markdown cell::

    ### A. Section title
    <!-- collapse-section: Click to expand -->

    ... everything below, up to the next same-or-higher heading, becomes the dropdown's body.

The heading itself ("### A. Section title") stays a normal, visible heading; only the content
after the marker collapses. In the raw notebook this renders as nothing at all (Jupyter/Colab/
JupyterLab ignore an HTML comment), so the notebook itself stays untouched.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from docutils import nodes
from sphinx.util import logging
from sphinx_design.shared import create_component

if TYPE_CHECKING:
    from sphinx.application import Sphinx

LOGGER = logging.getLogger(__name__)
_MARKER = re.compile(r"<!--\s*collapse-section(?::\s*(?P<label>.+?))?\s*-->")


def _marker_label(node: nodes.Node) -> str | None:
    if not (isinstance(node, nodes.raw) and node.get("format") == "html"):
        return None
    m = _MARKER.fullmatch(node.astext().strip())
    if not m:
        return None
    return m.group("label") or "Details"


def collapse_marked_sections(_app: Sphinx, doctree: nodes.document) -> None:
    """Replace each section's marker + trailing content with a closed dropdown."""
    for section in list(doctree.findall(nodes.section)):
        children = section.children
        marker_idx = next((i for i, c in enumerate(children) if _marker_label(c) is not None), None)
        if marker_idx is None:
            continue
        label = _marker_label(children[marker_idx])
        body = children[marker_idx + 1 :]

        dropdown = create_component(
            "dropdown",
            opened=False,
            type="dropdown",
            has_title=True,
            icon=None,
            chevron=None,
            container_classes=["sd-mb-3"],
            title_classes=[],
            body_classes=[],
        )
        dropdown += nodes.rubric(label, "", nodes.Text(label))
        dropdown.extend(body)

        del section.children[1:]  # keep the section + its own (visible) title only
        section += dropdown


def setup(app: Sphinx) -> dict:
    """App setup hook."""
    app.connect("doctree-read", collapse_marked_sections)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
