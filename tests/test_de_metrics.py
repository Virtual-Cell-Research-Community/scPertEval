from __future__ import annotations

import numpy as np
import pytest

from scperteval.blocks.panels import GenePanel
from scperteval.protocols.de_metrics import DEComparison
from scperteval.types import PerturbationDEResult


class _Context:
    current_pert = "geneA"
    ds = type("Dataset", (), {"var_names": np.array(["g0", "g1", "g2", "g3"])})()


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
