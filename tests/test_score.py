"""Prediction-scoring mode: score predictions against ground truth, and PredictionSet."""

from __future__ import annotations

import numpy as np
import pytest

from scperteval.calibrators import CALIBRATORS
from scperteval.cli import _concrete
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.predictions import PredictionSet
from scperteval.protocols.table import PROTOCOLS
from scperteval.runner import run_protocol


def _score(name, dataset_adata, pred_adata, cfg):
    ds = Dataset(dataset_adata, cfg)
    ctx = Context(ds, cfg)
    ctx.predictions = PredictionSet(pred_adata, ds, cfg)
    return run_protocol(_concrete(PROTOCOLS[name]), ctx, CALIBRATORS["score"])


def _score_cfg(cfg_factory):
    return cfg_factory(truth="gt_all_cells", calibrator="score")


def test_score_rows_have_prediction_column(dataset_adata, predictions_factory, cfg_factory):
    pred = predictions_factory(dataset_adata, kind="perfect")
    agg, rows, _ = _score("mse", dataset_adata, pred, _score_cfg(cfg_factory))
    assert {"protocol", "perturbation", "raw_prediction", "score"} <= set(rows[0])
    assert {"mean", "median"} <= set(agg)


def test_perfect_prediction_is_optimal(dataset_adata, predictions_factory, cfg_factory):
    # an exact replica of the real cells must score optimally on every representation,
    # even with the prediction's gene columns shuffled (name-based alignment).
    pred = predictions_factory(dataset_adata, kind="perfect", shuffle_genes=True)
    cfg = _score_cfg(cfg_factory)
    assert _score("pearson", dataset_adata, pred, cfg)[0]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert _score("mse", dataset_adata, pred, cfg)[0]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert _score("de_auprc", dataset_adata, pred, cfg)[0]["mean"] == pytest.approx(1.0, abs=1e-6)


def test_degraded_prediction_scores_worse(dataset_adata, predictions_factory, cfg_factory):
    cfg = _score_cfg(cfg_factory)
    perfect = predictions_factory(dataset_adata, kind="perfect")
    degraded = predictions_factory(dataset_adata, kind="degraded")

    def mean(name, pred):
        return _score(name, dataset_adata, pred, cfg)[0]["mean"]

    assert mean("mse", degraded) > mean("mse", perfect)  # error up
    assert mean("de_auprc", degraded) < mean("de_auprc", perfect)  # auprc down


def test_gene_set_mismatch_raises(dataset_adata, predictions_factory, cfg_factory):
    ds = Dataset(dataset_adata, cfg_factory())
    pred_missing = predictions_factory(dataset_adata, kind="perfect")[:, :-1].copy()
    with pytest.raises(ValueError, match="gene mismatch"):
        PredictionSet(pred_missing, ds, cfg_factory())


def test_missing_perturbation_raises(dataset_adata, predictions_factory, cfg_factory):
    ds = Dataset(dataset_adata, cfg_factory())
    pred = predictions_factory(dataset_adata, kind="perfect")
    only_a = pred[np.asarray(pred.obs["perturbation"]) == "pertA"].copy()
    ps = PredictionSet(only_a, ds, cfg_factory())
    with pytest.raises(ValueError, match="no cells for perturbation"):
        ps.cells("pertB")


def test_gene_alignment_reorders_by_name(dataset_adata, predictions_factory, cfg_factory):
    # a shuffled-gene prediction is reordered to the dataset's gene order
    ds = Dataset(dataset_adata, cfg_factory())
    pred = predictions_factory(dataset_adata, kind="perfect", shuffle_genes=True)
    ps = PredictionSet(pred, ds, cfg_factory())
    cells = np.asarray(ps.cells("pertA"))
    assert cells.shape[1] == len(ds.var_names)
    # pertA's DE block (genes 0-5) should be the high-expression columns after realignment
    col_means = cells.mean(0)
    assert col_means[list(range(0, 6))].min() > col_means[10:].max()
