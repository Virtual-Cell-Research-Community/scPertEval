"""Pure-function tests for the new metric family: r2, l2, weighted_pearson, weighted_r2, nir."""

from __future__ import annotations

import numpy as np
import pytest

from scperteval.protocols import metrics as M


def test_r2_is_perfect_for_identical_profiles():
    gt = np.array([1.0, 2.0, 3.0, -1.0, 0.5])
    assert M.r2(gt, gt.copy(), None) == 1.0


class _WeightCtx:
    """Fake Context serving a fixed DE statistic, which is what mejia_weights turns into weights."""

    current_pert = "p"

    class cfg:
        truth = "truth"

    def __init__(self, statistic=None):
        self.statistic = np.ones(10) if statistic is None else np.asarray(statistic, dtype=float)

    def de(self, pert, truth, ref):
        return type("_DE", (), {"statistic": self.statistic})()


#: A constant DE statistic makes mejia_weights fall back to uniform weights.
_UniformWeightCtx = _WeightCtx


def test_weighted_pearson_matches_pearson_with_uniform_weights():
    rng = np.random.default_rng(0)
    gt, pred = rng.normal(size=10), rng.normal(size=10)
    assert M.weighted_pearson(gt, pred, _UniformWeightCtx(), exp=2.0) == pytest.approx(M.pearson(gt, pred, None))


def test_weighted_r2_matches_r2_with_uniform_weights():
    rng = np.random.default_rng(0)
    gt, pred = rng.normal(size=10), rng.normal(size=10)
    assert M.weighted_r2(gt, pred, _UniformWeightCtx(), exp=2.0) == pytest.approx(M.r2(gt, pred, None))


def test_nir_is_one_minus_transpose_rank():
    rng = np.random.default_rng(0)
    gt = [rng.normal(size=5) for _ in range(4)]
    pred = [rng.normal(size=5) for _ in range(4)]
    assert np.allclose(M.nir(gt, pred, None), 1.0 - M.rank_retrieval(gt, pred, None, transpose=True))


def test_r2_penalises_scale_error_where_pearson_does_not():
    """The reason to use R² over Pearson: correlation cannot see magnitude."""
    gt = np.array([2.0, 4.0, 6.0, 8.0])
    scaled = 3 * gt - 10  # perfectly correlated, badly off-scale
    assert M.pearson(gt, scaled, None) == pytest.approx(1.0)
    # SS_res = 80, SS_tot = 20 -> 1 - 80/20
    assert M.r2(gt, scaled, None) == pytest.approx(-3.0)


def test_r2_is_unbounded_below():
    """Mejia's Appendix B property: how far below 0 says how badly a prediction failed.

    Flooring would collapse these to one value, and with both controls floored the DRF
    numerator would go to 0 -- reporting a well-separated metric as uncalibrated.
    """
    gt = np.array([2.0, 4.0, 6.0, 8.0])
    assert M.r2(gt, 3 * gt - 10, None) < -1.0
    assert M.r2(gt, 10 * gt, None) < M.r2(gt, 3 * gt - 10, None)


def test_r2_is_zero_for_a_prediction_that_matches_the_mean():
    """A prediction carrying no perturbation-specific signal scores 0, not something flattering."""
    gt = np.array([2.0, 4.0, 6.0, 8.0])
    assert M.r2(gt, np.full_like(gt, gt.mean()), None) == pytest.approx(0.0)


def test_l2_is_zero_for_identical_profiles_and_matches_the_norm():
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 8.0])
    assert M.l2(gt, gt.copy(), None) == 0.0
    assert M.l2(gt, pred, None) == pytest.approx(4.0)  # sqrt((8-4)^2)


def test_weighted_metrics_follow_the_weights():
    """The path that distinguishes them from their unweighted forms: non-uniform weights.

    Gene 0 is a far stronger DEG than the rest, so it dominates. A prediction wrong only on
    gene 0 must score worse than one wrong only on an unweighted gene.
    """
    ctx = _WeightCtx([10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    wrong_where_it_counts = gt.copy()
    wrong_where_it_counts[0] += 3.0
    wrong_elsewhere = gt.copy()
    wrong_elsewhere[3] += 3.0

    assert M.weighted_r2(gt, wrong_where_it_counts, ctx) < M.weighted_r2(gt, wrong_elsewhere, ctx)
    assert M.weighted_mse(gt, wrong_where_it_counts, ctx) > M.weighted_mse(gt, wrong_elsewhere, ctx)


def test_weighted_pearson_matches_an_independent_weighted_correlation():
    """Checked against the general form, which divides through by the weight sum."""
    from scperteval.blocks.de import mejia_weights

    rng = np.random.default_rng(3)
    stat = rng.normal(size=40)
    gt, pred = rng.normal(size=40), rng.normal(size=40)
    w = mejia_weights(stat, exp=2.0)

    wn = w / w.sum()
    mx, my = np.sum(wn * gt), np.sum(wn * pred)
    cov = np.sum(wn * (gt - mx) * (pred - my))
    expected = cov / np.sqrt(np.sum(wn * (gt - mx) ** 2) * np.sum(wn * (pred - my) ** 2))

    ctx = _WeightCtx(stat)
    assert M.weighted_pearson(gt, pred, ctx, exp=2.0) == pytest.approx(expected)


def test_weighted_metrics_return_nan_when_no_gene_carries_weight():
    """An all-non-finite DE statistic gives zero weights; there is nothing to correlate."""
    ctx = _WeightCtx(np.full(6, np.nan))
    gt, pred = np.arange(6.0), np.arange(6.0) + 1
    assert np.isnan(M.weighted_pearson(gt, pred, ctx))
    assert np.isnan(M.weighted_r2(gt, pred, ctx))
