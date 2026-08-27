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

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..blocks.de import mejia_weights

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
    Ground-truth DE result; ``gt.pvalue_adj`` defines the positive class.
prediction : numpy.ndarray
    Per-gene absolute DE score ranking from the candidate source, shape ``(G,)``.
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


@_doc(params=_DE)
def de_auprc(gt, prediction, ctx):
    """Area under the precision-recall curve for DEG recovery.

    Positive class: ground-truth DEGs with ``gt.pvalue_adj < 0.05``.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        AUPRC in [0, 1]; higher is better.
        Returns ``nan`` if all genes fall in the same class.
    """
    labels = gt.pvalue_adj < 0.05
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(average_precision_score(labels, prediction))


@_doc(params=_DE)
def de_auroc(gt, prediction, ctx):
    """Area under the ROC curve for DEG recovery.

    Positive class: ground-truth DEGs with ``gt.pvalue_adj < 0.05``.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    float
        AUROC in [0, 1]; higher is better.
        Returns ``nan`` if all genes fall in the same class.
    """
    labels = gt.pvalue_adj < 0.05
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(roc_auc_score(labels, prediction))


@_doc(params=_DE)
def de_overlap(gt, prediction, ctx, k=50):
    r"""Top-k overlap between ground-truth and predicted DE gene rankings.

    .. math::

        \text{Overlap}_k
        = \frac{|\text{top-}k(|gt.statistic|) \cap \text{top-}k(pred)|}{k}

    Parameters
    ----------
    %(params)s
    k : int
        Number of top genes to intersect (default 50).

    Returns
    -------
    float
        Fraction of top-k genes shared, in [0, 1]; higher is better.
        Returns ``nan`` if k >= number of genes.
    """
    truth = np.abs(gt.statistic)
    if k >= truth.size:
        return float("nan")
    top_truth = np.argpartition(-truth, k - 1)[:k]
    top_prediction = np.argpartition(-prediction, k - 1)[:k]
    return float(np.intersect1d(top_truth, top_prediction).size) / k
