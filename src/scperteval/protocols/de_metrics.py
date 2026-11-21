"""Signed differential-expression recovery metrics behind one composable interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score

from ..blocks.panels import GenePanel, UndefinedPanel
from ..types import MetricOutcome, PerturbationDEResult

DE_METRICS = {
    "average-precision",
    "auroc",
    "class-precision",
    "deg-identification-auprc",
    "deg-set-intersection",
    "direction-concordance",
    "f1",
    "jaccard",
    "macro-f1",
    "set-overlap-at-n",
    "set-overlap-at-truth-size",
    "spearman",
    "topk-overlap",
}


def _undefined(reason: str) -> MetricOutcome:
    return MetricOutcome(float("nan"), reason)


@dataclass(frozen=True)
class DEComparison:
    """Evaluate one DE-recovery family on a truth-derived gene panel.

    The panel is selected from observed truth only.  Predictions contribute signed statistics and
    adjusted p-values only after that panel is frozen, avoiding prediction-dependent eligibility.
    """

    family: str
    panel: GenePanel = field(default_factory=GenePanel)
    padj_max: float = 0.05
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.family not in DE_METRICS:
            raise ValueError(f"unknown DE metric {self.family!r}; expected {sorted(DE_METRICS)}")
        if not 0 < self.padj_max <= 1:
            raise ValueError("padj_max must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def __call__(self, truth: PerturbationDEResult, prediction: PerturbationDEResult, ctx) -> MetricOutcome:
        """Return one signed/recovery score or a structured undefined result."""
        try:
            idx = self.panel.select(ctx, ctx.current_pert)
        except UndefinedPanel as error:
            return _undefined(error.reason)
        t_stat = np.asarray(truth.statistic[idx], dtype=float)
        p_stat = np.asarray(prediction.statistic[idx], dtype=float)
        t_pos = np.asarray(truth.pvalue_adj[idx] < self.padj_max)
        p_pos = np.asarray(prediction.pvalue_adj[idx] < self.padj_max)
        if not (np.isfinite(t_stat).all() and np.isfinite(p_stat).all()):
            return _undefined("non_finite_de_statistic")

        if self.family in {"average-precision", "deg-identification-auprc"}:
            if not t_pos.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(average_precision_score(t_pos, np.abs(p_stat))))
        if self.family == "auroc":
            if t_pos.all() or not t_pos.any():
                return _undefined("single_ground_truth_class")
            return MetricOutcome(float(roc_auc_score(t_pos, np.abs(p_stat))))
        if self.family == "spearman":
            if np.ptp(t_stat) == 0:
                return _undefined("constant_ground_truth_de_statistic")
            if np.ptp(p_stat) == 0:
                return MetricOutcome(0.0)
            return MetricOutcome(float(spearmanr(t_stat, p_stat).statistic))
        if self.family == "direction-concordance":
            use = t_pos
            if not use.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(np.mean(np.sign(t_stat[use]) == np.sign(p_stat[use]))))
        if self.family in {"topk-overlap", "set-overlap-at-n"}:
            k = min(self.top_k or 50, len(idx))
            if k == 0:
                return _undefined("empty_truth_derived_gene_panel")
            t_set = set(np.argsort(-np.abs(t_stat), kind="stable")[:k].tolist())
            p_set = set(np.argsort(-np.abs(p_stat), kind="stable")[:k].tolist())
            return MetricOutcome(len(t_set & p_set) / k)
        if self.family == "set-overlap-at-truth-size":
            n = int(t_pos.sum())
            if n == 0:
                return _undefined("no_ground_truth_positive_degs")
            p_top = set(np.argsort(-np.abs(p_stat), kind="stable")[:n].tolist())
            return MetricOutcome(len(set(np.flatnonzero(t_pos)) & p_top) / n)
        if self.family == "deg-set-intersection":
            if not t_pos.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(np.logical_and(t_pos, p_pos).sum() / t_pos.sum()))
        if self.family == "jaccard":
            union = np.logical_or(t_pos, p_pos).sum()
            return MetricOutcome(1.0 if union == 0 else float(np.logical_and(t_pos, p_pos).sum() / union))
        if self.family == "class-precision":
            if not t_pos.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(precision_score(t_pos, p_pos, zero_division=0)))
        if self.family == "f1":
            if not t_pos.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(f1_score(t_pos, p_pos, zero_division=0)))
        if self.family == "macro-f1":
            if not t_pos.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(f1_score(t_pos, p_pos, average="macro", zero_division=0)))
        raise AssertionError(f"unhandled DE metric {self.family}")
