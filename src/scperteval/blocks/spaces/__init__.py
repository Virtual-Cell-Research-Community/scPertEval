"""Feature spaces: which features a protocol scores on, chosen before the metric runs.

Every space is a decorated rule in ``catalog.py`` — the file to open to see what exists, or to
add one. ``registry.py`` holds the machinery: the catalog of definitions, and the named
instances built from them.
"""

from __future__ import annotations

from typing import cast

from . import catalog  # imported for its side effect: decorating the rules defines the catalog
from .registry import (
    OPS,
    SPACES,
    CombinedSubset,
    FeatureSpace,
    PipedTransform,
    RuleSubset,
    RuleTransform,
    SetOps,
    Space,
    SpaceRegistry,
    SubsetSpace,
    TransformSpace,
    UndefinedSpace,
    resolve_space,
)


def full() -> SubsetSpace:
    """All genes, no transform."""
    return cast(SubsetSpace, SPACES.space("full"))


def hvg(k: int = 2000) -> SubsetSpace:
    """Top ``k`` highly variable genes in control cells."""
    return cast(SubsetSpace, SPACES.space("hvg", k))


def heg(k: int = 1000) -> SubsetSpace:
    """Top ``k`` highest-expressed genes in control cells."""
    return cast(SubsetSpace, SPACES.space("heg", k))


def top(k: int = 50, *, reference: str = "all_perturbed") -> SubsetSpace:
    """Top ``k`` genes by absolute ground-truth DE statistic."""
    return cast(SubsetSpace, SPACES.space("top", k, reference=reference))


def degs(
    padj: float | None = 0.05,
    *,
    top_k: int | None = None,
    rank_by: str = "significance",
    reference: str = "all_perturbed",
) -> SubsetSpace:
    """Ground-truth DEGs with optional ``padj`` threshold, ``top_k`` cap, ranking, and reference."""
    if padj is not None and not 0 < padj <= 1:
        raise ValueError("padj must be in (0, 1]")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if rank_by not in ("significance", "absolute_effect"):
        raise ValueError(f"unknown rank_by {rank_by!r}; expected ['significance', 'absolute_effect']")
    return cast(SubsetSpace, SPACES.space("degs", padj, top_k=top_k, rank_by=rank_by, reference=reference))


def perturbed_genes() -> SubsetSpace:
    """All genes targeted by any retained perturbation in the dataset."""
    return cast(SubsetSpace, SPACES.space("perturbed_genes"))


def own_target() -> SubsetSpace:
    """The gene(s) targeted by the current perturbation (compose via ``|`` or ``-``)."""
    return cast(SubsetSpace, SPACES.space("own_target"))


def pca(k: int = 50, *, on: SubsetSpace | None = None) -> TransformSpace:
    """Top ``k`` principal components, optionally fit on a dataset-wide ``on`` subset."""
    base = cast(TransformSpace, SPACES.space("pca", k))
    return (on >> base) if on is not None else base


__all__ = [
    "OPS",
    "SPACES",
    "CombinedSubset",
    "FeatureSpace",
    "PipedTransform",
    "RuleSubset",
    "RuleTransform",
    "SetOps",
    "Space",
    "SpaceRegistry",
    "SubsetSpace",
    "TransformSpace",
    "UndefinedSpace",
    "catalog",
    "degs",
    "full",
    "heg",
    "hvg",
    "own_target",
    "pca",
    "perturbed_genes",
    "resolve_space",
    "top",
]
