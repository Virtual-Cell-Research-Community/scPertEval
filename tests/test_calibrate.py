"""Calibration mode: DRF/BDS over built-in positive/negative controls."""

from __future__ import annotations

import numpy as np

from scperteval.calibrators import CALIBRATORS
from scperteval.cli import _concrete
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.protocols.table import PROTOCOLS
from scperteval.runner import run_protocol


def _run(name, calibrator, dataset_adata, cfg):
    ctx = Context(Dataset(dataset_adata, cfg), cfg)
    return run_protocol(_concrete(PROTOCOLS[name]), ctx, CALIBRATORS[calibrator])


def test_calibrators_registered():
    assert {"drf", "bds", "score"} <= set(CALIBRATORS)
    # drf/bds need both controls; score needs only the prediction
    assert CALIBRATORS["drf"].requires == ("positive", "negative")
    assert CALIBRATORS["score"].requires == ("prediction",)


def test_drf_rows_have_control_columns(dataset_adata, cfg_factory):
    agg, rows, seconds = _run("pearson_ctrl", "drf", dataset_adata, cfg_factory())
    assert seconds >= 0.0
    assert len(rows) == 4  # one row per perturbation
    cols = set(rows[0])
    assert {"protocol", "perturbation", "raw_positive", "raw_negative", "drf"} <= cols
    assert {"mean", "median"} <= set(agg)


def test_drf_positive_for_real_signal(dataset_adata, cfg_factory):
    # the positive control (held-out replicate) should beat the uninformative baseline,
    # so mean DRF is clearly positive on a dataset with strong perturbation signal.
    for name in ("pearson_ctrl", "mse"):
        agg, _, _ = _run(name, "drf", dataset_adata, cfg_factory())
        assert agg["mean"] > 0.0, name


def test_bds_is_a_fraction(dataset_adata, cfg_factory):
    agg, rows, _ = _run("pearson_ctrl", "bds", dataset_adata, cfg_factory())
    assert 0.0 <= agg["bds"] <= 1.0
    assert all(r["bds"] in (0.0, 1.0) for r in rows)  # per-perturbation BDS is binary


def test_de_protocol_calibrates(dataset_adata, cfg_factory):
    # the de representation should produce a finite, well-defined auprc (distinct DE blocks)
    agg, _, _ = _run("de_auprc", "drf", dataset_adata, cfg_factory())
    assert np.isfinite(agg["mean"])


# --- Ahlmann-Eltze et al. 2025: heg_space + l2 (const-ae's protocol, ported natively) ---


def test_heg_k_protocols_resolve_and_calibrate(dataset_adata, cfg_factory):
    for name in ("pearson_heg_k", "pearson_ctrl_heg_k", "l2_heg_k"):
        agg, rows, _ = _run(name, "drf", dataset_adata, cfg_factory())
        assert np.isfinite(agg["mean"]), name
        assert len(rows) == 4


# --- Miller et al. 2025 / Vollenweider & Bühlmann 2026: R2 and DE-weighted delta correlations ---


def test_miller_r2_and_weighted_protocols_resolve_and_calibrate(dataset_adata, cfg_factory):
    names = (
        "pearson_ctrl_top_k",
        "pearson_ctrl_degs_padj",
        "weighted_pearson_ctrl_exp2",
        "weighted_pearson_pert_exp2",
        "r2_ctrl",
        "r2_ctrl_top_k",
        "r2_ctrl_degs_padj",
        "weighted_r2_ctrl_exp2",
        "r2_pert",
        "r2_pert_top_k",
        "r2_pert_degs_padj",
        "weighted_r2_pert_exp2",
        "nir",
    )
    for name in names:
        agg, rows, _ = _run(name, "drf", dataset_adata, cfg_factory())
        assert np.isfinite(agg["mean"]), name
        assert len(rows) == 4, name
