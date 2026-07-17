"""Feature spaces: a transform applied to the gene axis before a protocol runs.

A space receives raw (possibly sparse) cells and returns a dense array over a gene subset.
Two registration patterns:

- **Fixed space** — one decorated function (:func:`space_full`).
- **Parameterised family** — a factory that registers ``name_<value>`` on demand:
  ``top_<k>`` (:func:`top_space`), ``degs_<padj>`` (:func:`degs_space`),
  ``pca_<k>`` (:func:`pca_space`).

Default instances (``top_50``, ``degs_0.05``, ``pca_50``) are created at import;
these are what ``scperteval list spaces`` shows.
"""

from __future__ import annotations

import numpy as np

from ..dataset import to_dense
from ..registry import Registry

SPACES = Registry("space")
"""Registry of feature-space transforms; keys are space names (e.g. ``"top_50"``).

Use :meth:`~scperteval.registry.Registry.register` to add a custom space::

    from scperteval.blocks.spaces import SPACES, to_dense

    @SPACES.register("hvg_100", global_space=True, description="100 highest-variance genes")
    def space_hvg(X, ctx, pert):
        keep = ...                    # indices of the 100 genes to keep
        return to_dense(X[:, keep])

Pass ``global_space=True`` if the transform does not depend on the perturbation
(so it can be computed once and shared across all perturbations in a run).

**Optional ``prepare`` hook.** A space family whose transform depends on some expensive, shared
structure (a fitted basis, a trained embedding model) can register a ``prepare`` hook to build
that structure once, up front, instead of lazily inside the per-perturbation loop::

    @SPACES.register("pca_50", global_space=True, prepare=my_prepare, description="…")
    def space(X, ctx, pert): ...

- Signature ``prepare(ctx, names) -> None`` — ``names`` is the *set* of that family's variant
  space names requested in the run (e.g. ``{"pca_30", "pca_50", "pca_100"}``).
- :meth:`~scperteval.context.Context.warm` calls each distinct hook **once** with all its
  variants, before any transform runs — so the family sees every variant at once and can build
  each variant's shared structure eagerly instead of on first use (e.g. fit each requested PCA
  size; a learned embedding might fit one model per variant). Store the result on ``ctx`` (e.g.
  ``ctx.pca(...)``, which caches on the shared store).
- It is **purely an optimisation and must be idempotent**: the transform has to stay correct if
  the hook never runs (a space run without being declared to ``prepare`` computes lazily), and the
  hook may be invoked again on an already-warm context. Do no per-perturbation work here — that
  belongs in the transform.
"""


# --- Fixed spaces: one registered function each ---


@SPACES.register("full", global_space=True, description="all genes, no transform")
def space_full(X, ctx, pert):
    """Identity space: all genes, densified, no transform."""
    return to_dense(X)


# --- Parameterised families: a factory registers name_<value> on demand ---


def _field(de, name):
    return de.extra[name.split(":", 1)[1]] if name.startswith("extra:") else getattr(de, name)


def register_de_space(name, field, top=None, threshold=None, description=""):
    r"""Register a DE-derived gene subset selected from a field of the GT PerturbationDEResult.

    Exactly one of ``top`` (select top-k by \|value\|) or ``threshold`` (a callable
    returning a boolean mask) must be provided.

    Parameters
    ----------
    name : str
        Registry key for the new space.
    field : str
        Attribute of :class:`~scperteval.types.PerturbationDEResult` to read
        (e.g. ``"statistic"``, ``"pvalue_adj"``).
    top : int or None
        If given, keep the top-k genes by absolute value of ``field``.
    threshold : Callable or None
        If given, a function ``(values) -> bool mask`` selecting genes to keep.
    description : str
        Human-readable description shown by ``scperteval list spaces``.

    Returns
    -------
    str
        The registered space name (same as ``name``).
    """

    def space(X, ctx, pert):
        values = _field(ctx.de(pert, ctx.cfg.truth), field)
        if top is not None:
            keep = np.argsort(-np.abs(values))[:top]
        else:
            assert threshold is not None  # register_de_space takes exactly one of top/threshold
            keep = np.where(threshold(values))[0]
        return to_dense(X[:, keep])

    SPACES.add(name, space, description=description)
    return name


def top_space(k: int) -> str:
    r"""top-k genes by absolute ground-truth effect size (registered on demand).

    Parameters
    ----------
    k : int
        Number of genes to keep (selected by \|ground-truth effect size\| per perturbation).

    Returns
    -------
    str
        Space name ``"top_<k>"`` (e.g. ``"top_50"``).
    """
    name = f"top_{k}"
    if name not in SPACES:
        register_de_space(
            name, field="statistic", top=k, description=f"top {k} genes by ground-truth effect size, per perturbation"
        )
    return name


def degs_space(padj: float) -> str:
    """ground-truth DEGs at adjusted p < padj (registered on demand).

    Parameters
    ----------
    padj : float
        Adjusted p-value threshold (e.g. 0.05).

    Returns
    -------
    str
        Space name ``"degs_<padj>"`` (e.g. ``"degs_0.05"``).
    """
    name = f"degs_{padj:g}"
    if name not in SPACES:
        register_de_space(
            name,
            field="pvalue_adj",
            threshold=(lambda v, p=padj: v < p),
            description=f"ground-truth DEGs at adjusted p < {padj:g}, per perturbation",
        )
    return name


def _pca_prepare(ctx, names):
    """Prepare hook for the ``pca_*`` family: fit each requested ``pca_<k>`` up front.

    ``names`` is the set of requested ``pca_<k>`` space names. Each distinct fit-size is fit once
    and cached independently (see :meth:`~scperteval.context.Context.pca`): sklearn's PCA is not
    basis-stable across ``n_components``, so a smaller ``pca_k`` cannot be sliced from a larger
    fit without changing its result. Set iteration order does not matter — every size is fit.
    """
    for name in names:
        ctx.pca(int(name.rsplit("_", 1)[1]))


def pca_space(k: int) -> str:
    """top-k principal components (registered on demand).

    PCA is fit once on (up to 50 000) cells from the full dataset, then applied
    to each cell population. The fitted transform is shared across perturbations.

    Parameters
    ----------
    k : int
        Number of principal components to retain.

    Returns
    -------
    str
        Space name ``"pca_<k>"`` (e.g. ``"pca_50"``).
    """
    name = f"pca_{k}"
    if name not in SPACES:

        def transform(X, ctx, pert):
            return ctx.pca(k).transform(to_dense(X))[:, :k]

        SPACES.add(
            name,
            transform,
            global_space=True,
            prepare=_pca_prepare,
            description=f"top {k} principal components (fit on the dataset)",
        )
    return name


# Default instances — also what `scperteval list spaces` shows.
top_space(50)
pca_space(50)
degs_space(0.05)
