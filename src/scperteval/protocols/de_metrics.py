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


def _ranking_scores(result: PerturbationDEResult, idx: np.ndarray, rank_by: str | None) -> np.ndarray:
    """Return deterministic continuous scores whose descending order is the protocol ranking."""
    statistic = np.asarray(result.statistic[idx], dtype=float)
    if rank_by == "significance":
        padj = np.asarray(result.pvalue_adj[idx], dtype=float)
        order = np.lexsort((idx, -np.abs(statistic), padj))
        scores = np.empty(len(idx), dtype=float)
        scores[order] = np.arange(len(idx), 0, -1, dtype=float)
        return scores
    return np.abs(statistic)


def _top_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Select exactly the first ``k`` genes under a stable descending ranking."""
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


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
            selected_idx = self.panel.select(ctx, ctx.current_pert)
        except UndefinedPanel as error:
            return _undefined(error.reason)

        # A DEG panel defines the observed positive/relevant genes, not the whole classification
        # universe.  In particular, AUROC/AUPRC need the non-DE genes as negatives.  Dataset-wide
        # panels (all/HVG/HEG) already define that universe directly.
        if self.panel.feature_set == "deg":
            universe_idx = np.arange(len(ctx.ds.var_names), dtype=int)
            truth_positive = np.isin(universe_idx, selected_idx)
        else:
            universe_idx = selected_idx
            truth_positive = np.asarray(truth.pvalue_adj[universe_idx] < self.padj_max)

        t_stat = np.asarray(truth.statistic[universe_idx], dtype=float)
        p_stat = np.asarray(prediction.statistic[universe_idx], dtype=float)
        t_padj = np.asarray(truth.pvalue_adj[universe_idx], dtype=float)
        p_padj = np.asarray(prediction.pvalue_adj[universe_idx], dtype=float)
        if not (np.isfinite(t_stat).all() and np.isfinite(t_padj).all()):
            return _undefined("non_finite_ground_truth_de_statistic")
        if not (np.isfinite(p_stat).all() and np.isfinite(p_padj).all()):
            return _undefined("non_finite_prediction_de_statistic")

        rank_by = self.panel.rank_by if self.panel.feature_set == "deg" else None
        truth_scores = _ranking_scores(truth, universe_idx, rank_by)
        prediction_scores = _ranking_scores(prediction, universe_idx, rank_by)
        if self.panel.feature_set == "deg" and self.panel.top_k is not None:
            prediction_positive = _top_mask(prediction_scores, min(self.panel.top_k, len(universe_idx)))
        else:
            prediction_positive = p_padj < self.padj_max

        if self.family in {"average-precision", "deg-identification-auprc"}:
            if not truth_positive.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(average_precision_score(truth_positive, prediction_scores)))
        if self.family == "auroc":
            if truth_positive.all() or not truth_positive.any():
                return _undefined("single_ground_truth_class")
            return MetricOutcome(float(roc_auc_score(truth_positive, prediction_scores)))
        if self.family == "spearman":
            use = truth_positive if self.panel.feature_set == "deg" else np.ones(len(universe_idx), dtype=bool)
            if np.ptp(t_stat[use]) == 0:
                return _undefined("constant_ground_truth_de_statistic")
            if np.ptp(p_stat[use]) == 0:
                return MetricOutcome(0.0)
            return MetricOutcome(float(spearmanr(t_stat[use], p_stat[use]).statistic))
        if self.family == "direction-concordance":
            use = truth_positive
            if not use.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(np.mean(np.sign(t_stat[use]) == np.sign(p_stat[use]))))
        if self.family in {"topk-overlap", "set-overlap-at-n"}:
            k = min(self.top_k or self.panel.top_k or 50, len(universe_idx))
            if k == 0:
                return _undefined("empty_truth_derived_gene_panel")
            truth_top = _top_mask(truth_scores, k)
            prediction_top = _top_mask(prediction_scores, k)
            return MetricOutcome(float(np.logical_and(truth_top, prediction_top).sum() / k))
        if self.family == "set-overlap-at-truth-size":
            n = int(truth_positive.sum())
            if n == 0:
                return _undefined("no_ground_truth_positive_degs")
            prediction_top = _top_mask(prediction_scores, n)
            return MetricOutcome(float(np.logical_and(truth_positive, prediction_top).sum() / n))
        if self.family == "deg-set-intersection":
            if not truth_positive.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(
                float(np.logical_and(truth_positive, prediction_positive).sum() / truth_positive.sum())
            )
        if self.family == "jaccard":
            union = np.logical_or(truth_positive, prediction_positive).sum()
            return MetricOutcome(
                1.0 if union == 0 else float(np.logical_and(truth_positive, prediction_positive).sum() / union)
            )
        if self.family == "class-precision":
            if not truth_positive.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(precision_score(truth_positive, prediction_positive, zero_division=0)))
        if self.family == "f1":
            if not truth_positive.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(f1_score(truth_positive, prediction_positive, zero_division=0)))
        if self.family == "macro-f1":
            if not truth_positive.any():
                return _undefined("no_ground_truth_positive_degs")
            return MetricOutcome(float(f1_score(truth_positive, prediction_positive, average="macro", zero_division=0)))
        raise AssertionError(f"unhandled DE metric {self.family}")
