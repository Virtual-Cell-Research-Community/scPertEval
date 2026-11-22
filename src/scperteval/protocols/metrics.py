r"""Evaluation-protocol metrics — pure functions, one per protocol.

Each metric receives ``(gt, prediction, ctx)`` and returns a scalar score:

- ``gt`` — ground-truth datapoint (representation-dependent, see below).
- ``prediction`` — the control or predicted datapoint in the same format as ``gt``.
- ``ctx`` — :class:`~scperteval.context.Context` providing dataset access and cached computations.

Two protocol fields control what the metric receives:

**representation** — the shape of each datapoint:

- ``"centroid"`` — 1-D pseudobulk vector, shape ``(G,)``.
- ``"population"`` — cell matrix, shape ``(n, G)``.
- ``"de"`` — :class:`~scperteval.types.PerturbationDEResult` for ground truth; \|statistic\| ranking for predictions.

**scope** — how many perturbations the metric sees at once:

- ``"perturbation"`` (default) — called once per perturbation; returns a scalar.
- ``"dataset"`` — called once with every perturbation's ``(gt, prediction)``; returns one score per perturbation (e.g. :func:`rank_retrieval`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, explained_variance_score, r2_score, roc_auc_score

from ..blocks.de import mejia_weights
from ..types import MetricOutcome

# --- shared parameter blocks, substituted into docstrings at decoration time ---

_CENTROID = """\
gt : numpy.ndarray
    Ground-truth pseudobulk profile, shape ``(G,)``.
prediction : numpy.ndarray
    Predicted pseudobulk profile, shape ``(G,)``.
ctx : Context
    Unused; present for signature compatibility."""

_CENTROID_W = """\
gt : numpy.ndarray
    Ground-truth pseudobulk profile, shape ``(G,)``.
prediction : numpy.ndarray
    Predicted pseudobulk profile, shape ``(G,)``.
ctx : Context
    Per-run context; used to read the ground-truth DE effect sizes for the per-gene weights."""

_POPULATION = """\
gt : numpy.ndarray
    Ground-truth cell matrix, shape ``(n, G)``.
prediction : numpy.ndarray
    Predicted cell matrix, shape ``(m, G)``.
ctx : Context
    Unused; present for signature compatibility."""

_DATASET = """\
gt : list of numpy.ndarray
    Ground-truth centroids, one per perturbation, each shape ``(G,)``.
prediction : list of numpy.ndarray
    Predicted centroids, one per perturbation, each shape ``(G,)``.
ctx : Context
    Unused; present for signature compatibility."""

_DE = """\
gt : ~scperteval.types.PerturbationDEResult
    Ground-truth DE result with signed ``statistic`` and ``pvalue_adj``.
prediction : ~scperteval.types.PerturbationDEResult
    Candidate DE result with signed ``statistic`` and ``pvalue_adj``.
ctx : Context
    Unused; present for signature compatibility."""


def _doc(**subs):
    """Decorator that substitutes %(key)s placeholders, propagating surrounding indentation.

    Python's ``%`` substitution only indents the first line of a multi-line value.
    This decorator detects the column position of each placeholder and re-indents all
    continuation lines to match, so the substituted text stays inside the RST section.
    """

    def deco(fn):
        doc = fn.__doc__
        for key, value in subs.items():
            placeholder = f"%({key})s"
            while placeholder in doc:
                idx = doc.index(placeholder)
                line_start = doc.rfind("\n", 0, idx) + 1
                indent = " " * (idx - line_start)
                indented = ("\n" + indent).join(value.split("\n"))
                doc = doc[:idx] + indented + doc[idx + len(placeholder) :]
        fn.__doc__ = doc
        return fn

    return deco


def _scale(better: str, perfect: float):
    """Attach intrinsic ``(better, perfect)`` scale metadata to a built-in metric."""

    def deco(fn):
        fn._scperteval_scale = (better, perfect)
        return fn

    return deco


def _sq_dists(X, Y):
    """Pairwise squared euclidean distances via ||x||^2 + ||y||^2 - 2 x.y.

    Routed through a BLAS matrix product, which releases the GIL so the
    per-perturbation thread pool actually parallelises.
    """
    xx = np.einsum("ij,ij->i", X, X)
    yy = np.einsum("ij,ij->i", Y, Y)
    sq = xx[:, None] + yy[None, :] - 2.0 * (X @ Y.T)
    return np.maximum(sq, 0.0)


def _within_unbiased(sq, n):
    """Unbiased (U-statistic) mean within-population euclidean distance."""
    if n <= 1:
        return 0.0
    return float(np.sqrt(sq).sum() / (n * (n - 1)))


@_scale("higher", 1.0)
@_doc(params=_CENTROID)
def pearson(gt, prediction, ctx):
    r"""Pearson correlation between pseudobulk profiles.

    .. math::

        r = \frac{\sum_g (gt_g - \bar{gt})(pred_g - \bar{pred})}{
                  \sqrt{\sum_g (gt_g - \bar{gt})^2 \cdot \sum_g (pred_g - \bar{pred})^2}}

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Pearson r in [-1, 1]; 1 is perfect.
    """
    return float(np.corrcoef(gt, prediction)[0, 1])


@_scale("higher", 1.0)
@_doc(params=_CENTROID)
def spearman(gt, prediction, ctx):
    """Spearman rank correlation between pseudobulk profiles.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Spearman ρ in [-1, 1]; 1 is perfect. Returns ``nan`` for constant ground truth
        and ``0.0`` for a constant prediction against varying ground truth.
    """
    if np.ptp(gt) == 0:
        return float("nan")
    if np.ptp(prediction) == 0:
        return 0.0
    result: Any = spearmanr(gt, prediction)
    return float(result.statistic)


@_scale("higher", 1.0)
@_doc(params=_CENTROID)
def cosine(gt, prediction, ctx):
    """Cosine similarity between pseudobulk profiles.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Cosine similarity in [-1, 1]; 1 is perfect. Returns ``nan`` for a zero
        ground-truth vector and ``0.0`` for a zero prediction against non-zero ground truth.
    """
    gt_norm = float(np.linalg.norm(gt))
    if gt_norm == 0.0:
        return float("nan")
    pred_norm = float(np.linalg.norm(prediction))
    if pred_norm == 0.0:
        return 0.0
    return float(np.dot(gt, prediction) / (gt_norm * pred_norm))


@_scale("higher", 1.0)
@_doc(params=_CENTROID)
def squared_pearson(gt, prediction, ctx):
    """Squared Pearson correlation between pseudobulk profiles.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Squared Pearson r² in [0, 1]; 1 is perfect.
    """
    if np.ptp(gt) == 0:
        return float("nan")
    if np.ptp(prediction) == 0:
        return 0.0
    value = pearson(gt, prediction, ctx)
    return value * value


@_scale("higher", 1.0)
@_doc(params=_CENTROID)
def r2(gt, prediction, ctx):
    r"""Coefficient of determination between pseudobulk profiles.

    .. math::

        R^2 = 1 - \frac{\sum_g (gt_g - pred_g)^2}{\sum_g (gt_g - \bar{gt})^2}

    Unlike :func:`pearson`, this penalises bias and scale errors, not just linear
    association — a prediction that is perfectly correlated with, but off-scale from, the
    ground truth still loses R² :cite:p:`Miller_2025`.

    Deliberately unbounded below, matching :cite:t:`Mejia_2025`'s reference implementation:
    a prediction that ignores the perturbation is guaranteed :math:`\leq 0` (their Appendix B),
    and how far below tells you how badly it failed. Clipping that range would erase the
    property R² is chosen for, and would distort the DRF of both controls at once.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        R² ≤ 1; 1 is perfect, 0 matches the mean baseline, negative is worse than it.
    """
    return float(r2_score(gt, prediction))


@_scale("higher", 1.0)
@_doc(params=_CENTROID)
def clipped_r2(gt, prediction, ctx):
    """Coefficient of determination clipped to ``[0, 1]``.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Clipped R² in [0, 1]; 1 is perfect.
    """
    return float(np.clip(r2(gt, prediction, ctx), 0.0, 1.0))


@_scale("higher", 1.0)
@_doc(params=_CENTROID)
def explained_variance(gt, prediction, ctx):
    """Explained-variance regression score between pseudobulk profiles.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Explained variance ≤ 1; 1 is perfect.
    """
    return float(explained_variance_score(gt, prediction))


@_scale("lower", 0.0)
@_doc(params=_CENTROID)
def mse(gt, prediction, ctx):
    r"""Mean squared error between pseudobulk profiles.

    .. math::

        \text{MSE} = \frac{1}{G}\sum_{g=1}^G (gt_g - pred_g)^2

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Non-negative MSE; 0 is perfect.
    """
    return float(np.mean((gt - prediction) ** 2))


@_scale("lower", 0.0)
@_doc(params=_CENTROID)
def mae(gt, prediction, ctx):
    """Mean absolute error between pseudobulk profiles.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Non-negative MAE; 0 is perfect.
    """
    return float(np.mean(np.abs(gt - prediction)))


@_scale("lower", 0.0)
@_doc(params=_CENTROID)
def l2(gt, prediction, ctx):
    r"""Euclidean distance between pseudobulk profiles.

    .. math::

        \ell_2 = \sqrt{\sum_{g=1}^G (gt_g - pred_g)^2}

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Non-negative Euclidean distance; 0 is perfect.
    """
    return float(np.sqrt(np.sum((gt - prediction) ** 2)))


@_scale("lower", 0.0)
@_doc(params=_CENTROID_W)
def weighted_mse(gt, prediction, ctx, exp=2.0):
    r"""MSE weighted by ground-truth effect size raised to ``exp``.

    Weights are min-max normalised per-gene; high-effect genes contribute more.

    .. math::

        \text{wMSE} = \sum_g w_g \,(gt_g - pred_g)^2, \quad
        w_g \propto |s_g|^{\text{exp}} / \sum_{g'} |s_{g'}|^{\text{exp}}

    where :math:`s_g` is the ground-truth DE t-statistic for gene :math:`g`.

    Parameters
    ----------
    %(params)s
    exp : float
        Exponent applied to the effect-size weights (default 2.0).

    Returns
    -------
    float
        Non-negative weighted MSE; 0 is perfect.
    """
    # Mejia DEG weights: min-max normalised absolute GT effect size (the ground-truth DE is cached
    # in ctx.de, keyed by GT source and method), raised to `exp`. Factored into the pure
    # blocks.de.mejia_weights helper so WMSE stays self-contained (no Context weighting method).
    s = ctx.de(ctx.current_pert, ctx.cfg.truth, "all_perturbed").statistic
    w = mejia_weights(s, exp=exp)
    return float(np.sum(w * (gt - prediction) ** 2))


@_scale("higher", 1.0)
@_doc(params=_CENTROID_W)
def weighted_pearson(gt, prediction, ctx, exp=2.0):
    r"""Pearson correlation weighted by ground-truth effect size raised to ``exp``.

    Same per-gene weights as :func:`weighted_mse`, applied as a weighted covariance instead of a
    weighted squared error — the "weighted Pearson Delta" of :cite:t:`Vollenweider_2026`.

    Assumes the weights sum to 1, which :func:`~scperteval.blocks.de.mejia_weights` guarantees
    (:cite:t:`Mejia_2025` normalises them as step (v) of its weight definition, and its reference
    implementation divides by the weight sum at point of use). That is what lets the weighted
    means below be written as ``sum(w * x)``. Weights that do not sum to 1 need every term
    divided by ``w.sum()`` — the general form in :cite:t:`Vollenweider_2026` — otherwise the
    means are scaled wrongly and the correlation is not bounded to [-1, 1].

    .. math::

        r_w = \frac{\sum_g w_g (gt_g - \bar{gt}_w)(pred_g - \bar{pred}_w)}
                   {\sqrt{\sum_g w_g (gt_g - \bar{gt}_w)^2 \cdot \sum_g w_g (pred_g - \bar{pred}_w)^2}}

    Parameters
    ----------
    %(params)s
    exp : float
        Exponent applied to the effect-size weights (default 2.0).

    Returns
    -------
    float
        Weighted Pearson r in [-1, 1]; 1 is perfect. ``nan`` if either weighted variance is 0.
    """
    s = ctx.de(ctx.current_pert, ctx.cfg.truth, "all_perturbed").statistic
    w = mejia_weights(s, exp=exp)
    d_gt, d_pred = gt - np.sum(w * gt), prediction - np.sum(w * prediction)
    var_gt, var_pred = np.sum(w * d_gt**2), np.sum(w * d_pred**2)
    if var_gt <= 0 or var_pred <= 0:
        return float("nan")
    return float(np.sum(w * d_gt * d_pred) / np.sqrt(var_gt * var_pred))


@_scale("higher", 1.0)
@_doc(params=_CENTROID_W)
def weighted_r2(gt, prediction, ctx, exp=2.0):
    r"""Coefficient of determination weighted by ground-truth effect size raised to ``exp``.

    Same per-gene weights as :func:`weighted_mse`, passed straight to scikit-learn's
    weighted :math:`R^2` — the :math:`R_w^2(\Delta)` of :cite:t:`Mejia_2025`, which
    :cite:t:`Miller_2025` adopts as the "continuous weighting" variant of :func:`r2`.
    Unbounded below, as :func:`r2` is.

    Parameters
    ----------
    %(params)s
    exp : float
        Exponent applied to the effect-size weights (default 2.0).

    Returns
    -------
    float
        Weighted R² ≤ 1; 1 is perfect. ``nan`` if every weight is 0.
    """
    s = ctx.de(ctx.current_pert, ctx.cfg.truth, "all_perturbed").statistic
    w = mejia_weights(s, exp=exp)
    if w.sum() <= 0:
        return float("nan")
    return float(r2_score(gt, prediction, sample_weight=w))


@_scale("lower", 0.0)
@_doc(params=_POPULATION)
def energy_distance(gt, prediction, ctx):
    r"""Székely–Rizzo energy distance with bias-corrected within-population terms.

    .. math::

        E(X, Y) = 2\,\mathbb{E}[\|X - Y\|]
                  - \mathbb{E}[\|X - X'\|] - \mathbb{E}[\|Y - Y'\|]

    Within-population terms use the unbiased (U-statistic) estimator.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        Energy distance >= 0; 0 is perfect (identical distributions).
        Returns ``nan`` if either population is empty.
    """
    if len(gt) == 0 or len(prediction) == 0:
        return float("nan")
    X = gt.astype(np.float64)
    Y = prediction.astype(np.float64)
    cross = np.sqrt(_sq_dists(X, Y)).mean()
    xx = _within_unbiased(_sq_dists(X, X), len(X))
    yy = _within_unbiased(_sq_dists(Y, Y), len(Y))
    return float(2.0 * cross - xx - yy)


@_scale("lower", 0.0)
@_doc(params=_POPULATION)
def unbiased_mmd_median(gt, prediction, ctx):
    r"""Unbiased RBF-MMD² with median-heuristic bandwidth (Gretton 2012).

    .. math::

        \widehat{\text{MMD}}^2(X, Y)
        = \frac{1}{n(n-1)} \sum_{i \neq j} k(x_i, x_j)
        + \frac{1}{m(m-1)} \sum_{i \neq j} k(y_i, y_j)
        - \frac{2}{nm} \sum_{i,j} k(x_i, y_j)

    with :math:`k(x,y) = \exp(-\|x-y\|^2 / 2\sigma^2)` and :math:`\sigma` the median
    pairwise Euclidean distance over the pooled sample.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        MMD² (may be slightly negative due to estimation variance); 0 is perfect.
        Returns ``nan`` if either population has fewer than 2 cells.
    """
    if len(gt) < 2 or len(prediction) < 2:
        return float("nan")
    X = gt.astype(np.float64)
    Y = prediction.astype(np.float64)
    nx, ny = len(X), len(Y)
    pooled = np.vstack([X, Y])
    euc = np.sqrt(_sq_dists(pooled, pooled))
    n = euc.shape[0]
    sigma = float(np.median(euc[~np.eye(n, dtype=bool)]))
    if sigma <= 0:
        return 0.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = np.exp(-gamma * _sq_dists(X, X))
    k_yy = np.exp(-gamma * _sq_dists(Y, Y))
    k_xy = np.exp(-gamma * _sq_dists(X, Y))
    xx = (k_xx.sum() - np.trace(k_xx)) / (nx * (nx - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (ny * (ny - 1))
    return float(xx + yy - 2.0 * k_xy.mean())


_geomloss_cache: dict = {}


@_scale("lower", 0.0)
@_doc(params=_POPULATION)
def sinkhorn_w2(gt, prediction, ctx, blur=0.05):
    r"""Debiased Sinkhorn 2-Wasserstein distance (geomloss, p=2).

    .. math::

        W_2(X, Y) = \sqrt{2\,S_\varepsilon(X, Y)}

    where :math:`S_\varepsilon` is the debiased Sinkhorn divergence with blur
    :math:`\varepsilon`. Requires ``geomloss`` and ``torch``.

    Parameters
    ----------
    %(params)s
    blur : float
        Sinkhorn entropic regularisation parameter (default 0.05).

    Returns
    -------
    float
        W2 distance >= 0; 0 is perfect.
        Returns ``nan`` if either population is empty.
    """
    if len(gt) == 0 or len(prediction) == 0:
        return float("nan")
    try:
        import torch  # pyright: ignore[reportMissingImports]  # optional `sinkhorn` extra
        from geomloss import SamplesLoss  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as e:  # torch/geomloss are the optional `sinkhorn` extra
        raise ModuleNotFoundError(
            "The 'sinkhorn_w2' protocol needs the optional Sinkhorn/optimal-transport dependencies "
            "(torch, geomloss). Install them with:  pip install 'scperteval[sinkhorn]'"
        ) from e

    loss = _geomloss_cache.get(blur)
    if loss is None:
        torch.set_num_threads(1)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True, backend="tensorized")
        _geomloss_cache[blur] = loss
    Xt = torch.as_tensor(np.ascontiguousarray(gt), dtype=torch.float32)
    Yt = torch.as_tensor(np.ascontiguousarray(prediction), dtype=torch.float32)
    a = torch.full((len(gt),), 1.0 / len(gt), dtype=torch.float32)
    b = torch.full((len(prediction),), 1.0 / len(prediction), dtype=torch.float32)
    with torch.no_grad():
        val = float(loss(a, Xt, b, Yt))
    return float(np.sqrt(max(2.0 * val, 0.0)))


@_scale("lower", 0.0)
@_doc(params=_DATASET)
def rank_retrieval(gt, prediction, ctx, transpose=False):
    r"""Cross-perturbation retrieval rank — dataset-scope metric, lower is better.

    Builds the ``(n x n)`` squared-distance matrix between all predicted and ground-truth
    centroids, then reads off the diagonal rank (column-wise by default).

    .. math::

        \text{rank}(a) = \frac{\text{rank}_{\text{col}}(D_{aa})}{n - 1}, \quad
        D_{ij} = \|P_i - G_j\|^2

    where :math:`P_i` and :math:`G_j` are the predicted and ground-truth centroids.
    ``transpose_rank`` transposes the matrix first (each prediction ranked among all GTs).
    Tie-breaking noise (seed 42) matches the DRF calibration convention.

    Parameters
    ----------
    %(params)s
    transpose : bool
        If ``True``, rank row-wise (each prediction vs all GTs) instead of column-wise.

    Returns
    -------
    np.ndarray
        Per-perturbation normalised rank in [0, 1]; 0 is a perfect top-1 retrieval.
    """
    G = np.vstack(gt)
    P = np.vstack(prediction)
    sq = _sq_dists(P, G)
    if transpose:
        sq = sq.T
    n = sq.shape[0]
    noise = np.random.default_rng(42).uniform(0, 1e-12, size=sq.shape)
    ranks = np.argsort(np.argsort(sq + noise, axis=0), axis=0)
    return np.diag(ranks).astype(np.float64) / max(n - 1, 1)


@_scale("higher", 1.0)
@_doc(params=_DATASET)
def nir(gt, prediction, ctx):
    r"""Normalized Inverse Rank — dataset-scope, higher is better.

    Introduced by :cite:t:`Wu_2024` as the "transposed-rank" metric, and adopted under this name
    by :cite:t:`Miller_2025`. Exactly ``1 - transpose_rank``: :func:`rank_retrieval` with
    ``transpose=True`` (each prediction ranked among all ground-truth centroids), inverted so 1
    is a perfect top-1 retrieval and 0 is worst. Under DRF and BDS this reparameterisation
    cancels, so it scores identically to ``transpose_rank``; it is kept as its own metric for the
    published name and sign convention, and because other calibrators need not be invariant to
    it.

    .. math::

        \text{NIR} = 1 - \text{transpose\_rank}

    Parameters
    ----------
    %(params)s

    Returns
    -------
    np.ndarray
        Per-perturbation NIR in [0, 1]; 1 is a perfect top-1 retrieval.
    """
    return 1.0 - rank_retrieval(gt, prediction, ctx, transpose=True)


_DE_RANKINGS = ("absolute_effect", "significance")


def _de_check_finite(gt, prediction) -> MetricOutcome | None:
    """Validate finiteness of DE statistics/p-values before computing a DE metric."""
    t_stat = np.asarray(gt.statistic, dtype=float)
    t_padj = np.asarray(gt.pvalue_adj, dtype=float)
    if not (np.isfinite(t_stat).all() and np.isfinite(t_padj).all()):
        return MetricOutcome(float("nan"), "non_finite_ground_truth_de_statistic")
    if hasattr(prediction, "statistic"):
        p_stat = np.asarray(prediction.statistic, dtype=float)
        p_padj = np.asarray(prediction.pvalue_adj, dtype=float)
        if not (np.isfinite(p_stat).all() and np.isfinite(p_padj).all()):
            return MetricOutcome(float("nan"), "non_finite_prediction_de_statistic")
    elif not np.isfinite(np.asarray(prediction, dtype=float)).all():
        return MetricOutcome(float("nan"), "non_finite_prediction_de_statistic")
    return None


def _de_scores(result, rank_by: str = "absolute_effect") -> np.ndarray:
    """Continuous per-gene scores whose descending order is the DE ranking."""
    if rank_by not in _DE_RANKINGS:
        raise ValueError(f"unknown rank_by {rank_by!r}; expected {list(_DE_RANKINGS)}")
    if not hasattr(result, "statistic"):
        return np.asarray(result, dtype=float)
    statistic = np.asarray(result.statistic, dtype=float)
    if rank_by == "significance":
        padj = np.asarray(result.pvalue_adj, dtype=float)
        idx = np.arange(len(statistic))
        order = np.lexsort((idx, -np.abs(statistic), padj))
        scores = np.empty(len(statistic), dtype=float)
        scores[order] = np.arange(len(statistic), 0, -1, dtype=float)
        return scores
    return np.abs(statistic)


def _de_top_mask(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def _de_positives(
    result,
    *,
    padj: float | None = 0.05,
    top_k: int | None = None,
    rank_by: str = "significance",
) -> np.ndarray:
    """Boolean mask of positive DEGs under optional ``padj`` threshold and ``top_k`` cap."""
    if padj is not None and not 0 < padj <= 1:
        raise ValueError("padj must be in (0, 1]")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if rank_by not in _DE_RANKINGS:
        raise ValueError(f"unknown rank_by {rank_by!r}; expected {list(_DE_RANKINGS)}")
    stat = np.asarray(result.statistic, dtype=float)
    pvals = np.asarray(result.pvalue_adj, dtype=float)
    mask = np.ones(len(stat), dtype=bool) if padj is None else (pvals < padj)
    if top_k is None:
        return mask
    candidates = np.flatnonzero(mask)
    if candidates.size <= top_k:
        return mask
    if rank_by == "significance":
        order = np.lexsort((candidates, -np.abs(stat[candidates]), pvals[candidates]))
    else:
        order = np.lexsort((candidates, pvals[candidates], -np.abs(stat[candidates])))
    out = np.zeros(len(stat), dtype=bool)
    out[candidates[order[:top_k]]] = True
    return out


def _de_pred_positives(
    prediction,
    *,
    padj: float | None = 0.05,
    top_k: int | None = None,
    rank_by: str = "significance",
) -> np.ndarray:
    """Boolean mask of predicted positive DEGs: top-k when ``top_k`` is set, else ``pvalue_adj < padj``."""
    if top_k is not None:
        scores = _de_scores(prediction, rank_by)
        return _de_top_mask(scores, min(top_k, len(scores)))
    if padj is None:
        return np.ones(len(prediction.pvalue_adj), dtype=bool)
    return np.asarray(prediction.pvalue_adj, dtype=float) < padj


def _binary_f1(truth: np.ndarray, prediction: np.ndarray, *, positive: bool = True) -> float:
    expected = truth if positive else ~truth
    observed = prediction if positive else ~prediction
    tp = int(np.logical_and(expected, observed).sum())
    fp = int(np.logical_and(~expected, observed).sum())
    fn = int(np.logical_and(expected, ~observed).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_auprc(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "absolute_effect"
):
    """Area under the precision-recall curve for DEG recovery.

    Parameters
    ----------
    %(params)s
    padj : float or None, optional
        Adjusted p-value cutoff defining ground-truth positive DEGs (default 0.05).
    top_k : int or None, optional
        Cap the ground-truth positive set to the top ``top_k`` genes.
    rank_by : {"absolute_effect", "significance"}, optional
        Ranking criterion for prediction scores (default ``"absolute_effect"``).

    Returns
    -------
    float or ~scperteval.types.MetricOutcome
        AUPRC in [0, 1]; higher is better. Returns an undefined outcome when there are no
        ground-truth positive DEGs.
    """
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    labels = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    if not labels.any():
        return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
    scores = _de_scores(prediction, rank_by)
    return float(average_precision_score(labels, scores))


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_auroc(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "absolute_effect"
):
    """Area under the ROC curve for DEG recovery.

    Parameters
    ----------
    %(params)s
    padj : float or None, optional
        Adjusted p-value cutoff defining ground-truth positive DEGs (default 0.05).
    top_k : int or None, optional
        Cap the ground-truth positive set to the top ``top_k`` genes.
    rank_by : {"absolute_effect", "significance"}, optional
        Ranking criterion for prediction scores (default ``"absolute_effect"``).

    Returns
    -------
    float or ~scperteval.types.MetricOutcome
        AUROC in [0, 1]; higher is better. Returns an undefined outcome when ground-truth
        labels contain only a single class.
    """
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    labels = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    if not labels.any() or labels.all():
        return MetricOutcome(float("nan"), "single_ground_truth_class")
    scores = _de_scores(prediction, rank_by)
    return float(roc_auc_score(labels, scores))


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_overlap(gt, prediction, ctx, k: int = 50, *, rank_by: str = "absolute_effect"):
    r"""Top-k overlap between ground-truth and predicted DE gene rankings.

    .. math::

        \text{Overlap}_k
        = \frac{|\text{top-}k(gt) \cap \text{top-}k(pred)|}{k}

    Parameters
    ----------
    %(params)s
    k : int
        Number of top genes to intersect (default 50).
    rank_by : {"absolute_effect", "significance"}, optional
        Ranking criterion (default ``"absolute_effect"``).

    Returns
    -------
    float or ~scperteval.types.MetricOutcome
        Fraction of top-k genes shared, in [0, 1]; higher is better.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_scores = _de_scores(gt, rank_by)
    pred_scores = _de_scores(prediction, rank_by)
    eff_k = min(k, truth_scores.size)
    if eff_k == 0:
        return MetricOutcome(float("nan"), "empty_gene_space")
    top_truth = _de_top_mask(truth_scores, eff_k)
    top_pred = _de_top_mask(pred_scores, eff_k)
    return float(np.logical_and(top_truth, top_pred).sum() / eff_k)


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_overlap_at_truth_size(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "significance"
):
    """Overlap between ground-truth positive DEGs and the top-N predicted DEGs (N = truth DEG count)."""
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_pos = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    n = int(truth_pos.sum())
    if n == 0:
        return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
    pred_top = _de_top_mask(_de_scores(prediction, rank_by), n)
    return float(np.logical_and(truth_pos, pred_top).sum() / n)


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_direction_concordance(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "significance"
):
    """Fraction of ground-truth positive DEGs with matching signed DE direction."""
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_pos = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    if not truth_pos.any():
        return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
    t_stat = np.asarray(gt.statistic, dtype=float)
    p_stat = np.asarray(prediction.statistic, dtype=float)
    return float(np.mean(np.sign(t_stat[truth_pos]) == np.sign(p_stat[truth_pos])))


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_spearman(
    gt,
    prediction,
    ctx,
    *,
    degs_only: bool = False,
    padj: float | None = 0.05,
    top_k: int | None = None,
    rank_by: str = "significance",
):
    """Spearman rank correlation between signed ground-truth and predicted DE statistics.

    Pass ``degs_only=True`` (or restrict ``Protocol.space`` to ``spaces.degs(...)``) to correlate
    over ground-truth positive DEGs only.
    """
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    t_stat = np.asarray(gt.statistic, dtype=float)
    p_stat = np.asarray(prediction.statistic, dtype=float)
    if degs_only:
        use = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
        if not use.any():
            return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
        t_stat = t_stat[use]
        p_stat = p_stat[use]
    if np.ptp(t_stat) == 0:
        return MetricOutcome(float("nan"), "constant_ground_truth_de_statistic")
    if np.ptp(p_stat) == 0:
        return 0.0
    result: Any = spearmanr(t_stat, p_stat)
    return float(result.statistic)


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_recall(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "significance"
):
    """Recall (true-positive rate) of predicted positive DEGs against ground-truth positive DEGs."""
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_pos = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    if not truth_pos.any():
        return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
    pred_pos = _de_pred_positives(prediction, padj=padj, top_k=top_k, rank_by=rank_by)
    return float(np.logical_and(truth_pos, pred_pos).sum() / truth_pos.sum())


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_precision(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "significance"
):
    """Precision of predicted positive DEGs against ground-truth positive DEGs."""
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_pos = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    if not truth_pos.any():
        return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
    pred_pos = _de_pred_positives(prediction, padj=padj, top_k=top_k, rank_by=rank_by)
    predicted = int(pred_pos.sum())
    if predicted == 0:
        return 0.0
    return float(np.logical_and(truth_pos, pred_pos).sum() / predicted)


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_f1(gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "significance"):
    """Binary F1 score of predicted positive DEGs against ground-truth positive DEGs."""
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_pos = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    if not truth_pos.any():
        return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
    pred_pos = _de_pred_positives(prediction, padj=padj, top_k=top_k, rank_by=rank_by)
    return _binary_f1(truth_pos, pred_pos)


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_macro_f1(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "significance"
):
    """Macro-averaged F1 across positive and negative DEG classes."""
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_pos = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    if not truth_pos.any():
        return MetricOutcome(float("nan"), "no_ground_truth_positive_degs")
    pred_pos = _de_pred_positives(prediction, padj=padj, top_k=top_k, rank_by=rank_by)
    return (_binary_f1(truth_pos, pred_pos, positive=True) + _binary_f1(truth_pos, pred_pos, positive=False)) / 2.0


@_scale("higher", 1.0)
@_doc(params=_DE)
def de_jaccard(
    gt, prediction, ctx, *, padj: float | None = 0.05, top_k: int | None = None, rank_by: str = "significance"
):
    """Jaccard similarity between ground-truth and predicted positive DEG sets."""
    invalid = _de_check_finite(gt, prediction)
    if invalid is not None:
        return invalid
    truth_pos = _de_positives(gt, padj=padj, top_k=top_k, rank_by=rank_by)
    pred_pos = _de_pred_positives(prediction, padj=padj, top_k=top_k, rank_by=rank_by)
    union = int(np.logical_or(truth_pos, pred_pos).sum())
    return 1.0 if union == 0 else float(np.logical_and(truth_pos, pred_pos).sum() / union)
