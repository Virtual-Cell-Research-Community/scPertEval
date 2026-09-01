"""Every feature space, one decorated rule each.

To add one, write a rule and decorate it:

- A **subset** rule returns a column selection into the *full* gene axis: an integer array, or a
  slice. Never positions into some earlier subset, so selections from different spaces can be
  folded together. It takes one of four shapes — ``(ctx)``, ``(ctx, k)``, ``(ctx, pert)``, or
  ``(ctx, pert, k)``. Naming ``pert`` is how the space says its genes vary by perturbation; omit
  it and the space is dataset-wide, and the reference is projected once and shared.
- A **transform** rule returns the finished dense array, for a space that replaces the gene axis
  rather than narrowing it. Same shapes, with the cells first: ``(X, ctx, k)`` and so on.

The rule runs once per perturbation per protocol, so anything computed over the whole dataset
belongs behind a :class:`~scperteval.context.Context` cache — ``ctx.control_mean()`` and friends —
rather than recomputed here.
"""

from __future__ import annotations

import numpy as np

from ...dataset import to_dense
from .helpers import control_dispersion, pca_for, targeted_genes
from .registry import OPS, SPACES


@SPACES.subset("full", description="all genes, no transform")
def full(ctx):
    """Every gene. Returns a slice, so applying the identity space is a view rather than a copy."""
    return slice(None)


@SPACES.subset("top", default=50, description="top {v} genes by ground-truth effect size")
def top(ctx, pert, k):
    """The k strongest ground-truth effect sizes for this perturbation, by absolute value."""
    return np.argsort(-np.abs(ctx.de(pert, ctx.cfg.truth).statistic))[:k]


@SPACES.subset("degs", default=0.05, description="ground-truth DEGs at adjusted p < {v}")
def degs(ctx, pert, padj):
    """Ground-truth differentially expressed genes for this perturbation."""
    return np.where(ctx.de(pert, ctx.cfg.truth).pvalue_adj < padj)[0]


@SPACES.subset("heg", default=1000, description="top {v} genes by control-condition expression")
def heg(ctx, k):
    """The k highest-expressed genes in the control cells — the criterion of Ahlmann-Eltze 2025.

    Dataset-wide, so the same panel serves every perturbation, unlike ``top``/``degs``.
    """
    return np.argsort(-ctx.control_mean())[:k]


@SPACES.subset("hvg", default=2000, description="top {v} genes by control-condition normalized dispersion")
def hvg(ctx, k):
    """The k most variable genes in the control cells, by scanpy's ``"seurat"`` dispersion."""
    return np.argsort(-control_dispersion(ctx))[:k]


@SPACES.subset("perturbed_genes", description="genes targeted by a perturbation in the dataset")
def perturbed_genes(ctx):
    """The genes the perturbations target — for a knockdown screen, the knocked-down genes.

    Their own expression is the most direct readout that a perturbation took effect, and they
    aren't necessarily variable, so this is meant to be unioned with another subset.
    """
    return targeted_genes(ctx)


@SPACES.transform(
    "pca", default=50, precompute=pca_for, description="top {v} principal components (fit on the dataset)"
)
def pca(X, ctx, k):
    """The top k principal components, from a PCA fit once on the dataset and shared."""
    return pca_for(ctx, k).transform(to_dense(X))[:, :k]


# A space is created when a protocol or a `Param` asks for it. `full` is created here because
# `Protocol.space` defaults to it, so that name has to resolve before any run.
SPACES.instance("full")


# Composed panels. `per_pert` is derived from the operands, so a composition cannot claim to be
# dataset-wide while computing something that varies by perturbation.
SPACES.combine_subsets(
    OPS.union,
    SPACES.instance("hvg", 8192),
    SPACES.instance("perturbed_genes"),
    name="perturbed_and_hvgs",
    description="HVG union perturbed genes — a panel introduced in Miller et al. 2025",
)
