"""Composable, truth-derived gene panels for concrete evaluation protocols.

The small public interface is :class:`GenePanel` plus :func:`panel_space`.  It hides deterministic
DEG selection, target-gene policy, dataset-wide HVG/HEG selection, runtime space registration, and
subset-then-PCA fitting.  Frozen analysis manifests can therefore describe panels as data rather
than adding bespoke entries to scPertEval's curated protocol table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..caching import _once
from ..dataset import to_dense
from .spaces import SPACES
from .spaces.helpers import PCA_FIT_CAP, control_dispersion

FEATURE_SETS = {"all", "deg", "hvg", "heg"}
RANKINGS = {"significance", "absolute_effect"}
TARGET_POLICIES = {"no_special", "force_include_target", "exclude_own_target"}


class UndefinedPanel(ValueError):
    """A truth-derived panel cannot be constructed for one evaluation unit."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class GenePanel:
    """A deterministic gene panel selected exclusively from observed ground truth.

    ``padj_max``, ``rank_by`` and ``top_k`` are the deliberately compact DEG selector knobs.
    Non-DEG panels reject them, preventing configurations whose names imply semantics they do not
    execute.  Target handling is applied after the base panel is selected.
    """

    feature_set: str = "all"
    top_k: int | None = None
    padj_max: float | None = None
    rank_by: str | None = None
    target_policy: str = "no_special"
    reference: str = "control"

    def __post_init__(self) -> None:
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature_set {self.feature_set!r}; expected {sorted(FEATURE_SETS)}")
        if self.target_policy not in TARGET_POLICIES:
            raise ValueError(f"unknown target_policy {self.target_policy!r}; expected {sorted(TARGET_POLICIES)}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.feature_set == "deg":
            if self.rank_by not in RANKINGS:
                raise ValueError(f"DEG panels require rank_by in {sorted(RANKINGS)}")
            if self.padj_max is not None and not 0 < self.padj_max <= 1:
                raise ValueError("padj_max must be in (0, 1]")
        elif self.rank_by is not None or self.padj_max is not None:
            raise ValueError("rank_by and padj_max apply only to DEG panels")

    @property
    def per_perturbation(self) -> bool:
        """Whether the selected genes can differ between perturbations."""
        return self.feature_set == "deg" or self.target_policy != "no_special"

    @property
    def name(self) -> str:
        """Canonical, fully resolved panel name used in protocol identities and caches."""
        parts = [self.feature_set]
        if self.feature_set == "deg":
            parts += [f"padj-{self.padj_max:g}" if self.padj_max is not None else "padj-none", f"rank-{self.rank_by}"]
        if self.top_k is not None:
            parts.append(f"top-{self.top_k}")
        if self.target_policy != "no_special":
            parts.append(self.target_policy.replace("_", "-"))
        if self.feature_set == "deg":
            parts.append(f"ref-{self.reference.replace('_', '-')}")
        return "panel__" + "__".join(parts)

    def select(self, ctx, pert: str | None) -> np.ndarray:
        """Return integer positions into the full dataset gene axis."""
        n_genes = len(ctx.ds.var_names)
        if self.feature_set == "all":
            selected = np.arange(n_genes)
        elif self.feature_set == "hvg":
            selected = np.argsort(-control_dispersion(ctx))[: min(self.top_k or n_genes, n_genes)]
        elif self.feature_set == "heg":
            selected = np.argsort(-ctx.control_mean())[: min(self.top_k or n_genes, n_genes)]
        else:
            if pert is None:
                raise ValueError("a DEG panel requires a perturbation")
            de = ctx.de(pert, ctx.cfg.truth, self.reference)
            keep = np.isfinite(de.statistic) & np.isfinite(de.pvalue_adj)
            if self.padj_max is not None:
                keep &= de.pvalue_adj < self.padj_max
            candidates = np.flatnonzero(keep)
            if self.rank_by == "significance":
                order = np.lexsort((candidates, -np.abs(de.statistic[candidates]), de.pvalue_adj[candidates]))
            else:
                order = np.lexsort((candidates, de.pvalue_adj[candidates], -np.abs(de.statistic[candidates])))
            selected = candidates[order]
            if self.top_k is not None:
                selected = selected[: self.top_k]

        if self.target_policy != "no_special":
            if pert is None:
                raise ValueError("target-gene policy requires a perturbation")
            targets = _target_indices(ctx, pert)
            if targets.size == 0:
                raise UndefinedPanel("target_gene_not_measured")
            if self.target_policy == "force_include_target":
                selected = np.union1d(selected, targets)
            else:
                selected = np.setdiff1d(selected, targets)
        selected = np.asarray(selected, dtype=int)
        if selected.size == 0:
            raise UndefinedPanel("empty_truth_derived_gene_panel")
        return selected


def _target_indices(ctx, pert: str) -> np.ndarray:
    positions = {str(g): i for i, g in enumerate(ctx.ds.var_names)}
    return np.array(sorted({positions[token] for token in pert.split("+") if token in positions}), dtype=int)


def panel_space(panel: GenePanel) -> str:
    """Materialize ``panel`` as a runtime feature space for a concrete protocol."""

    def select(ctx, pert):
        return panel.select(ctx, pert)

    return SPACES.custom_subset(
        panel.name,
        select,
        per_pert=panel.per_perturbation,
        description=f"truth-derived {panel.name}",
    )


def panel_pca_space(panel: GenePanel, n_components: int = 50) -> str:
    """Fit PCA once on observed cells restricted to a dataset-wide panel, then reuse it."""
    if panel.per_perturbation:
        raise ValueError("subset-then-PCA requires a dataset-wide panel")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    name = f"{panel.name}__pca-{n_components}"

    def fitted(ctx):
        def compute():
            from sklearn.decomposition import PCA

            keep = panel.select(ctx, None)
            n = ctx.ds.adata.n_obs
            idx = np.arange(n)
            if n > PCA_FIT_CAP:
                idx = np.sort(np.random.default_rng(ctx.cfg.seed).choice(n, PCA_FIT_CAP, replace=False))
            X = to_dense(ctx.ds.adata.X[idx][:, keep]).astype(np.float64)
            return keep, PCA(n_components=min(n_components, *X.shape), random_state=ctx.cfg.seed).fit(X)

        return _once(ctx._store, ("panel_pca", name), compute)

    def transform(X, ctx, pert):
        keep, pca = fitted(ctx)
        return pca.transform(to_dense(X[:, keep]))[:, :n_components]

    def precompute(ctx, _value):
        fitted(ctx)

    return SPACES.custom_transform(
        name,
        transform,
        per_pert=False,
        description=f"PCA-{n_components} fit on {panel.name}",
        precompute=precompute,
    )
