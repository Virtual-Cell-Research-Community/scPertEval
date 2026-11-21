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
from .helpers import control_dispersion, pca_for, subset_pca_for, targeted_genes
from .registry import OPS, SPACES, UndefinedSpace

_DEG_RANKINGS = ("significance", "absolute_effect")


@SPACES.subset("full", description="all genes, no transform")
def full(ctx):
    """Every gene. Returns a slice, so applying the identity space is a view rather than a copy."""
    return slice(None)


@SPACES.subset("top", default=50, description="top {v} genes by ground-truth effect size")
def top(ctx, pert, k, *, reference: str = "all_perturbed"):
    """The k strongest ground-truth effect sizes for this perturbation, by absolute value."""
    return np.argsort(-np.abs(ctx.de(pert, ctx.cfg.truth, reference).statistic))[:k]


@SPACES.subset("degs", default=0.05, description="ground-truth DEGs at adjusted p < {v}")
def degs(
    ctx,
    pert,
    padj,
    *,
    top_k: int | None = None,
    rank_by: str = "significance",
    reference: str = "all_perturbed",
):
    """Ground-truth differentially expressed genes for this perturbation.

    Optionally cap at ``top_k`` genes ordered deterministically by ``"significance"`` or
    ``"absolute_effect"``, and choose the DE ``reference`` population.
    """
    if padj is not None and not 0 < padj <= 1:
        raise ValueError("padj must be in (0, 1]")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if rank_by not in _DEG_RANKINGS:
        raise ValueError(f"unknown rank_by {rank_by!r}; expected {list(_DEG_RANKINGS)}")
    de = ctx.de(pert, ctx.cfg.truth, reference)
    keep = np.isfinite(de.statistic) & np.isfinite(de.pvalue_adj)
    if padj is not None:
        keep &= de.pvalue_adj < padj
    candidates = np.flatnonzero(keep)
    if top_k is None:
        return candidates
    if rank_by == "significance":
        order = np.lexsort((candidates, -np.abs(de.statistic[candidates]), de.pvalue_adj[candidates]))
    else:
        order = np.lexsort((candidates, de.pvalue_adj[candidates], -np.abs(de.statistic[candidates])))
    return candidates[order[:top_k]]


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


@SPACES.subset("own_target", description="gene(s) targeted by the current perturbation")
def own_target(ctx, pert):
    """The gene(s) targeted by ``pert`` alone — compose via ``|`` or ``-`` with any subset space."""
    idx = ctx.ds.target_gene_indices(pert)
    if idx.size == 0:
        raise UndefinedSpace(
            "target_gene_not_measured",
            f"target gene(s) for {pert!r} are not present in var_names.",
        )
    return idx


def _pca_subset_apply(X, ctx, subset, k):
    keep, fitted = subset_pca_for(ctx, subset, k)
    return fitted.transform(to_dense(X[:, keep]))[:, :k]


def _pca_subset_precompute(ctx, subset, k):
    subset_pca_for(ctx, subset, k)


@SPACES.transform(
    "pca",
    default=50,
    precompute=pca_for,
    subset_apply=_pca_subset_apply,
    subset_precompute=_pca_subset_precompute,
    description="top {v} principal components (fit on the dataset)",
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
