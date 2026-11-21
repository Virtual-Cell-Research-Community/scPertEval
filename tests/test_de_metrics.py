from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scperteval.blocks.panels import GenePanel
from scperteval.protocols.de_metrics import DEComparison
from scperteval.types import PerturbationDEResult


class _Context:
    current_pert = "geneA"
    ds = type("Dataset", (), {"var_names": np.array(["g0", "g1", "g2", "g3"])})()


class _DEGContext(_Context):
    cfg = type("Config", (), {"truth": "gt_all_cells"})()

    def __init__(self, truth):
        self.truth = truth

    def de(self, pert, source, reference):
        assert pert == self.current_pert
        assert source == self.cfg.truth
        assert reference == "control"
        return self.truth


def _de(stat, padj):
    stat = np.asarray(stat, dtype=float)
    padj = np.asarray(padj, dtype=float)
    return PerturbationDEResult(stat, padj, padj)


@pytest.mark.parametrize(
    "family",
    [
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
    ],
)
def test_de_comparison_is_perfect_on_identical_results(family):
    result = _de([4, -3, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    metric = DEComparison(family, GenePanel("all"), top_k=2)
    outcome = metric(result, result, _Context())
    assert outcome.undefined_reason is None
    assert outcome.value == pytest.approx(1.0)


@pytest.mark.parametrize("family", ["average-precision", "direction-concordance", "f1"])
def test_de_recovery_is_undefined_when_truth_has_no_positive_degs(family):
    result = _de([0.2, -0.1, 0.3, -0.4], [0.8, 0.9, 0.7, 0.6])
    outcome = DEComparison(family)(result, result, _Context())
    assert np.isnan(outcome.value)
    assert outcome.undefined_reason == "no_ground_truth_positive_degs"


@pytest.mark.parametrize("family", ["average-precision", "auroc"])
def test_deg_panel_keeps_non_selected_genes_as_classification_negatives(family):
    result = _de([4, -3, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    panel = GenePanel("deg", top_k=2, padj_max=0.05, rank_by="significance")
    outcome = DEComparison(family, panel)(result, result, _DEGContext(result))
    assert outcome.undefined_reason is None
    assert outcome.value == pytest.approx(1.0)


def test_deg_panel_direction_uses_only_truth_selected_genes():
    truth = _de([4, -3, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    prediction = _de([4, -3, -8, 9], [0.001, 0.01, 0.001, 0.001])
    panel = GenePanel("deg", top_k=2, padj_max=0.05, rank_by="significance")
    outcome = DEComparison("direction-concordance", panel)(truth, prediction, _DEGContext(truth))
    assert outcome.undefined_reason is None
    assert outcome.value == pytest.approx(1.0)


class _TargetContext(_DEGContext):
    current_pert = "geneA"
    ds = type(
        "Dataset",
        (),
        {
            "var_names": np.array(["geneA", "g1", "g2", "g3"]),
            "adata": type("AnnData", (), {"obs": pd.DataFrame(index=[0])})(),
        },
    )()


def test_force_include_target_does_not_turn_target_into_de_positive():
    truth = _de([8, 4, 0.2, -0.1], [0.8, 0.001, 0.8, 0.9])
    prediction = _de([-8, 4, 0.2, -0.1], [0.001, 0.001, 0.8, 0.9])
    panel = GenePanel(
        "deg",
        top_k=1,
        padj_max=0.05,
        rank_by="significance",
        target_policy="force_include_target",
    )
    outcome = DEComparison("direction-concordance", panel)(truth, prediction, _TargetContext(truth))
    assert outcome.value == pytest.approx(1.0)


def test_exclude_target_removes_it_from_de_classification_universe():
    truth = _de([8, 4, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    panel = GenePanel(
        "deg",
        top_k=2,
        padj_max=0.05,
        rank_by="significance",
        target_policy="exclude_own_target",
    )
    outcome = DEComparison("auroc", panel)(truth, truth, _TargetContext(truth))
    assert outcome.value == pytest.approx(1.0)
