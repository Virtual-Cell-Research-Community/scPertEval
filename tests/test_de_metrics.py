"""Tests for symmetric DE views, DE gene-universe restriction via Protocol.space, and pure DE metrics."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from conftest import make_dataset

import scperteval as sp
from scperteval.blocks import spaces
from scperteval.protocols import metrics as M
from scperteval.types import MetricOutcome, PerturbationDEResult, Protocol


def _de(stat, padj):
    stat = np.asarray(stat, dtype=float)
    padj = np.asarray(padj, dtype=float)
    return PerturbationDEResult(stat, padj, padj)


@pytest.mark.parametrize(
    "metric",
    [
        M.de_auprc,
        M.de_auroc,
        partial(M.de_overlap, k=2),
        M.de_overlap_at_truth_size,
        M.de_direction_concordance,
        M.de_spearman,
        partial(M.de_spearman, degs_only=True),
        M.de_recall,
        M.de_precision,
        M.de_f1,
        M.de_macro_f1,
        M.de_jaccard,
    ],
)
def test_de_metrics_attain_perfect_score_on_identical_results(metric):
    result = _de([4.0, -3.0, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    got = metric(result, result, None)
    assert isinstance(got, float)
    assert got == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metric",
    [
        M.de_auprc,
        M.de_overlap_at_truth_size,
        M.de_direction_concordance,
        M.de_recall,
        M.de_precision,
        M.de_f1,
        M.de_macro_f1,
    ],
)
def test_de_metrics_are_undefined_when_truth_has_no_positive_degs(metric):
    result = _de([0.2, -0.1, 0.3, -0.4], [0.8, 0.9, 0.7, 0.6])
    outcome = metric(result, result, None)
    assert isinstance(outcome, MetricOutcome)
    assert np.isnan(outcome.value)
    assert outcome.undefined_reason == "no_ground_truth_positive_degs"


def test_de_auroc_reports_single_ground_truth_class():
    all_neg = _de([0.2, -0.1, 0.3, -0.4], [0.8, 0.9, 0.7, 0.6])
    all_pos = _de([4.0, -3.0, 5.0, -2.0], [0.001, 0.002, 0.001, 0.01])
    for res in (all_neg, all_pos):
        outcome = M.de_auroc(res, res, None)
        assert isinstance(outcome, MetricOutcome)
        assert outcome.undefined_reason == "single_ground_truth_class"


def test_de_spearman_reports_constant_ground_truth_and_handles_constant_prediction():
    varying = _de([4.0, -3.0, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    constant = _de([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5])
    out = M.de_spearman(constant, varying, None)
    assert isinstance(out, MetricOutcome)
    assert out.undefined_reason == "constant_ground_truth_de_statistic"
    assert M.de_spearman(varying, constant, None) == 0.0


def test_de_direction_concordance_uses_signed_statistics():
    truth = _de([4.0, -3.0, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    flipped = _de([-4.0, 3.0, 0.2, -0.1], [0.001, 0.01, 0.8, 0.9])
    partial_match = _de([4.0, 3.0, -9.0, 9.0], [0.001, 0.01, 0.001, 0.001])
    assert M.de_direction_concordance(truth, flipped, None) == pytest.approx(0.0)
    assert M.de_direction_concordance(truth, partial_match, None) == pytest.approx(0.5)


def test_de_protocol_applies_subset_space_to_gene_universe(predictions_factory):
    adata = make_dataset()
    adata.var_names = ["pertA", "pertB", "pertC", "pertD", *[f"g{i}" for i in range(56)]]
    pred = predictions_factory(adata, kind="perfect")

    no_target = Protocol(
        "de_f1_no_target",
        partial(M.de_f1, padj=0.05),
        "de",
        space=spaces.full() - spaces.own_target(),
        reference="control",
    )
    hvg_with_target = Protocol(
        "de_dir_hvg_target",
        M.de_direction_concordance,
        "de",
        space=spaces.hvg(20) | spaces.own_target(),
        reference="control",
    )
    prep = sp.prepare(adata, [no_target, hvg_with_target], subsample=400, seed=0, min_cells=10, workers=1)
    scored = sp.score(prep, no_target, pred)
    assert scored.aggregate["mean"] == pytest.approx(1.0)
    avail = sp.truth_availability(prep, [no_target, hvg_with_target])
    assert len(avail) == 8


def test_de_protocol_rejects_transform_space(dataset_adata):
    bad = Protocol("de_pca", M.de_auprc, "de", space=spaces.pca(5))
    prep = sp.prepare(dataset_adata, [], subsample=400, seed=0, min_cells=10, workers=1)
    with pytest.raises(TypeError, match="gene-subset space"):
        sp.truth_availability(prep, bad)
