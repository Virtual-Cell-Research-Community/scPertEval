"""The declarative table of evaluation protocols — every protocol in one list (``TABLE``).

Each row is one ``Protocol(...)``. A protocol that takes a knob — a feature-space size or a
metric argument supplied at the CLI, e.g. ``-p mse_top_k=30`` — just adds ``param=<family>``;
with no value the family default is used. To add a protocol, write a metric in
``metrics.py`` and add one row below.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from ..blocks.spaces import degs_space, pca_space, top_space
from ..types import Param, Protocol
from . import metrics as M

# --- parameter families: a CLI value selects a feature space (or feeds the metric) ---
top_k = Param("k", int, 50, space=top_space)  # top-k DEGs by effect size
pca_k = Param("k", int, 50, space=pca_space)  # k principal components
degs_padj = Param("padj", float, 0.05, space=degs_space)  # DEGs at adjusted p < padj
overlap_k = Param("k", int, 50)  # passed straight to de_overlap's k


# --- shared wiring bundles (score scale + declared control deviations), splatted with ** ---
# Controls are resolved at runtime from generic-by-representation defaults (see runner.GENERIC_CONTROLS);
# only rows that deviate declare a default_positive/default_negative below.
_PB: dict[str, Any] = dict(group="pseudobulk")
_LOWER: dict[str, Any] = dict(better="lower", perfect=0.0)
_DIST: dict[str, Any] = dict(group="distributional", better="lower", perfect=0.0)
_DE: dict[str, Any] = dict(
    group="de",
    reference="all_perturbed",
    neg_reference="control",
    better="higher",
    perfect=1.0,
)
# rank retrieval is dataset-scope: the constant global mean is its correct chance-level baseline.
_RANK: dict[str, Any] = dict(group="pseudobulk", default_negative="global_mean", better="lower", perfect=0.0)


TABLE = [
    # --- pseudobulk: correlation & error (positive = interpolated duplicate) ---
    Protocol("pearson", M.pearson, representation="centroid", **_PB),
    Protocol("pearson_ctrl", M.pearson, representation="centroid", centering="ctrl", **_PB),
    # allpert-centred: all_perturbed_mean would be ~0 after centring, so control is the declared negative.
    Protocol(
        "pearson_pert", M.pearson, representation="centroid", centering="allpert", default_negative="control", **_PB
    ),
    Protocol("mse", M.mse, representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp1", partial(M.weighted_mse, exp=1.0), representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp2", partial(M.weighted_mse, exp=2.0), representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp4", partial(M.weighted_mse, exp=4.0), representation="centroid", **_PB, **_LOWER),
    Protocol("mse_top_k", M.mse, representation="centroid", param=top_k, **_PB, **_LOWER),
    Protocol("mse_degs_padj", M.mse, representation="centroid", param=degs_padj, **_PB, **_LOWER),
    Protocol(
        "pearson_pert_top_k",
        M.pearson,
        representation="centroid",
        centering="allpert",
        param=top_k,
        default_negative="control",
        **_PB,
    ),
    Protocol(
        "pearson_pert_degs_padj",
        M.pearson,
        representation="centroid",
        centering="allpert",
        param=degs_padj,
        default_negative="control",
        **_PB,
    ),
    # --- cross-perturbation retrieval rank (dataset-wide over centroids) ---
    Protocol("rank", partial(M.rank_retrieval, transpose=False), representation="centroid", scope="dataset", **_RANK),
    Protocol(
        "transpose_rank", partial(M.rank_retrieval, transpose=True), representation="centroid", scope="dataset", **_RANK
    ),
    # --- distributional: distances between cell populations (positive = technical duplicate) ---
    Protocol("unbiased_mmd_median_top_k", M.unbiased_mmd_median, representation="population", param=top_k, **_DIST),
    Protocol("unbiased_mmd_median_pca_k", M.unbiased_mmd_median, representation="population", param=pca_k, **_DIST),
    Protocol("energy_distance_top_k", M.energy_distance, representation="population", param=top_k, **_DIST),
    Protocol("energy_distance_pca_k", M.energy_distance, representation="population", param=pca_k, **_DIST),
    Protocol("sinkhorn_w2_top_k", M.sinkhorn_w2, representation="population", param=top_k, **_DIST),
    Protocol("sinkhorn_w2_pca_k", M.sinkhorn_w2, representation="population", param=pca_k, **_DIST),
    # --- differential expression: GT DEGs vs prediction ranking ---
    Protocol("de_auprc", M.de_auprc, representation="de", **_DE),
    Protocol("de_auroc", M.de_auroc, representation="de", **_DE),
    Protocol("de_overlap_k", M.de_overlap, representation="de", param=overlap_k, **_DE),
]
"""Every registered :class:`~scperteval.types.Protocol`, one row per protocol.

To add a protocol, write a metric in ``metrics.py`` and add one row here.
"""

PROTOCOLS = {p.name: p for p in TABLE}
"""``{name: Protocol}`` — fast lookup for the CLI's ``-p <name>`` and ``scperteval list protocols``."""

GROUPS = sorted({p.group for p in TABLE})
"""Sorted protocol group names (e.g. ``"pseudobulk"``, ``"distributional"``, ``"de"``), selectable as ``-p <group>``."""
