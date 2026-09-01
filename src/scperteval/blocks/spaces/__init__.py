"""Feature spaces: which features a protocol scores on, chosen before the metric runs.

Every space is a decorated rule in ``catalog.py`` — the file to open to see what exists, or to
add one. ``registry.py`` holds the machinery: the catalog of definitions, and the named
instances built from them.
"""

from __future__ import annotations

from . import catalog  # imported for its side effect: decorating the rules defines the catalog
from .registry import OPS, SPACES, SetOps, Space, SpaceRegistry

__all__ = ["OPS", "SPACES", "SetOps", "Space", "SpaceRegistry", "catalog"]
