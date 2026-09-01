"""The declarative table of evaluation protocols — every protocol in one list (``TABLE``).

Each row is one ``Protocol(...)``. A protocol that takes a knob — a feature-space size or a
metric argument supplied at the CLI, e.g. ``-p mse_top_k=30`` — just adds ``param=<family>``;
with no value the family default is used. To add a protocol, write a metric in
``metrics.py`` and add one row below.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from ..blocks.spaces import SPACES
from ..types import Param, Protocol
from . import metrics as M

# --- parameter families: a CLI value selects a feature space (or feeds the metric) ---
top_k = Param("k", int, 50, space=partial(SPACES.instance, "top"))  # top-k DEGs by effect size
pca_k = Param("k", int, 50, space=partial(SPACES.instance, "pca"))  # k principal components
degs_padj = Param("padj", float, 0.05, space=partial(SPACES.instance, "degs"))  # DEGs at adjusted p < padj
overlap_k = Param("k", int, 50)  # passed straight to de_overlap's k
heg_k = Param(
    "k", int, 1000, space=partial(SPACES.instance, "heg")
)  # top-k highly-expressed genes (Ahlmann-Eltze et al. 2025)


# --- shared wiring bundles (score scale + declared control deviations), splatted with ** ---
# Controls are resolved at runtime from generic-by-representation defaults (see runner.GENERIC_CONTROLS);
# only rows that deviate declare a default_positive/default_negative below.
_PB: dict[str, Any] = dict(group="pseudobulk")
_LOWER: dict[str, Any] = dict(better="lower", perfect=0.0)
_DIST: dict[str, Any] = dict(group="distributional", better="lower", perfect=0.0)
# Distributional, but backed by geomloss/torch — only available with the `sinkhorn` extra.
_OT: dict[str, Any] = dict(**_DIST, requires_extra="sinkhorn")
# de: GT tested vs all-perturbed; the negative candidate is tested vs control (hybrid reference).
_DE: dict[str, Any] = dict(
    group="de",
    reference="all_perturbed",
    neg_reference="control",
    better="higher",
    perfect=1.0,
)
# rank retrieval is dataset-scope: the constant global mean is its correct chance-level baseline.
_RANK: dict[str, Any] = dict(group="pseudobulk", default_negative="global_mean", better="lower", perfect=0.0)
# nir is 1 - transpose_rank, hence the inverted direction.
_NIR: dict[str, Any] = dict(group="pseudobulk", default_negative="global_mean", better="higher", perfect=1.0)


TABLE = [
    # --- pseudobulk: correlation & error (positive = interpolated duplicate) ---
    Protocol("pearson", M.pearson, representation="centroid", **_PB),
    # control-centred: each profile relative to the control mean.
    Protocol("pearson_ctrl", M.pearson, representation="centroid", centering="control_mean", **_PB),
    # allpert-centred: all_perturbed_mean would be ~0 after centring, so control is the declared negative.
    Protocol(
        "pearson_pert",
        M.pearson,
        representation="centroid",
        centering="all_perturbed_mean",
        default_negative="control",
        **_PB,
    ),
    Protocol("mse", M.mse, representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp1", partial(M.weighted_mse, exp=1.0), representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp2", partial(M.weighted_mse, exp=2.0), representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp4", partial(M.weighted_mse, exp=4.0), representation="centroid", **_PB, **_LOWER),
    Protocol("mse_top_k", M.mse, representation="centroid", param=top_k, **_PB, **_LOWER),
    Protocol("mse_degs_padj", M.mse, representation="centroid", param=degs_padj, **_PB, **_LOWER),
    # allpert-centred pearson on the top-k DEGs (control negative, as pearson_pert).
    Protocol(
        "pearson_pert_top_k",
        M.pearson,
        representation="centroid",
        centering="all_perturbed_mean",
        param=top_k,
        default_negative="control",
        **_PB,
    ),
    # allpert-centred pearson on the DEG subset (control negative, as pearson_pert).
    Protocol(
        "pearson_pert_degs_padj",
        M.pearson,
        representation="centroid",
        centering="all_perturbed_mean",
        param=degs_padj,
        default_negative="control",
        **_PB,
    ),
    # --- Ahlmann-Eltze et al. 2025: top-1000 control-expressed (highly-expressed) genes ---
    Protocol("pearson_heg_k", M.pearson, representation="centroid", param=heg_k, **_PB),
    Protocol("pearson_ctrl_heg_k", M.pearson, representation="centroid", centering="control_mean", param=heg_k, **_PB),
    Protocol("l2_heg_k", M.l2, representation="centroid", param=heg_k, **_PB, **_LOWER),
    # --- Miller et al. 2025 / Vollenweider & Bühlmann 2026: R2 and DE-weighted delta
    # correlations, each in three gene-set variants (all genes / DEG top-k / DEG padj / DE
    # effect-size-weighted) — MSE's own three variants are mse / mse_top_k+mse_degs_padj /
    # wmse_exp1-4 above; PearsonDeltaPerturbMean's are pearson_pert / pearson_pert_top_k+
    # pearson_pert_degs_padj / weighted_pearson_pert_exp2 below.
    Protocol("pearson_ctrl_top_k", M.pearson, representation="centroid", centering="control_mean", param=top_k, **_PB),
    Protocol(
        "pearson_ctrl_degs_padj", M.pearson, representation="centroid", centering="control_mean", param=degs_padj, **_PB
    ),
    Protocol(
        "weighted_pearson_ctrl_exp2",
        partial(M.weighted_pearson, exp=2.0),
        representation="centroid",
        centering="control_mean",
        **_PB,
    ),
    Protocol(
        "weighted_pearson_pert_exp2",
        partial(M.weighted_pearson, exp=2.0),
        representation="centroid",
        centering="all_perturbed_mean",
        default_negative="control",
        **_PB,
    ),
    Protocol("r2_ctrl", M.r2, representation="centroid", centering="control_mean", **_PB),
    Protocol("r2_ctrl_top_k", M.r2, representation="centroid", centering="control_mean", param=top_k, **_PB),
    Protocol("r2_ctrl_degs_padj", M.r2, representation="centroid", centering="control_mean", param=degs_padj, **_PB),
    Protocol(
        "weighted_r2_ctrl_exp2",
        partial(M.weighted_r2, exp=2.0),
        representation="centroid",
        centering="control_mean",
        **_PB,
    ),
    Protocol(
        "r2_pert",
        M.r2,
        representation="centroid",
        centering="all_perturbed_mean",
        default_negative="control",
        **_PB,
    ),
    Protocol(
        "r2_pert_top_k",
        M.r2,
        representation="centroid",
        centering="all_perturbed_mean",
        param=top_k,
        default_negative="control",
        **_PB,
    ),
    Protocol(
        "r2_pert_degs_padj",
        M.r2,
        representation="centroid",
        centering="all_perturbed_mean",
        param=degs_padj,
        default_negative="control",
        **_PB,
    ),
    Protocol(
        "weighted_r2_pert_exp2",
        partial(M.weighted_r2, exp=2.0),
        representation="centroid",
        centering="all_perturbed_mean",
        default_negative="control",
        **_PB,
    ),
    # --- cross-perturbation retrieval rank (dataset-wide over centroids) ---
    Protocol("rank", partial(M.rank_retrieval, transpose=False), representation="centroid", scope="dataset", **_RANK),
    Protocol(
        "transpose_rank", partial(M.rank_retrieval, transpose=True), representation="centroid", scope="dataset", **_RANK
    ),
    # --- Miller et al. 2025: Normalized Inverse Rank — transpose_rank, oriented higher-is-better ---
    Protocol("nir", M.nir, representation="centroid", scope="dataset", **_NIR),
    # --- distributional: distances between cell populations (positive = technical duplicate) ---
    Protocol("unbiased_mmd_median_top_k", M.unbiased_mmd_median, representation="population", param=top_k, **_DIST),
    Protocol("unbiased_mmd_median_pca_k", M.unbiased_mmd_median, representation="population", param=pca_k, **_DIST),
    Protocol("energy_distance_top_k", M.energy_distance, representation="population", param=top_k, **_DIST),
    Protocol("energy_distance_pca_k", M.energy_distance, representation="population", param=pca_k, **_DIST),
    Protocol("sinkhorn_w2_top_k", M.sinkhorn_w2, representation="population", param=top_k, **_OT),
    Protocol("sinkhorn_w2_pca_k", M.sinkhorn_w2, representation="population", param=pca_k, **_OT),
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
