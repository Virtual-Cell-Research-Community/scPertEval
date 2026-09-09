"""The pseudobulk memo: what it caches, what it refuses to cache, and that it stays immutable."""

from __future__ import annotations

import numpy as np
import pytest

import scperteval as sp
from scperteval.context import Context
from scperteval.dataset import Dataset


def _pseudobulk_keys(prepared):
    """Every ``("pseudobulk", (source, pert, de_method))`` key currently in the handle's store."""
    return [k for k in prepared._store.memo if isinstance(k[0], str) and k[0] == "pseudobulk"]


def test_pseudobulk_is_cached_and_reused(dataset_adata, predictions_factory):
    pred = predictions_factory(dataset_adata, kind="degraded")
    prepared = sp.prepare(dataset_adata, ["pearson_ctrl"], min_cells=10)
    assert _pseudobulk_keys(prepared) == []

    first = sp.score(prepared, "pearson_ctrl", pred).per_perturbation["raw_prediction"].to_numpy()
    truth_keys = [k for k in _pseudobulk_keys(prepared) if k[1][0] == "gt_all_cells"]
    assert len(truth_keys) == len(prepared._ds.perturbations)  # one per perturbation, not one per call

    cached = {k: prepared._store.memo[k] for k in truth_keys}
    second = sp.score(prepared, "pearson_ctrl", pred).per_perturbation["raw_prediction"].to_numpy()

    np.testing.assert_array_equal(first, second)
    for key, value in cached.items():
        assert prepared._store.memo[key] is value  # served from the cache, not recomputed


def test_prediction_pseudobulk_is_never_cached(dataset_adata, predictions_factory):
    """A per-call source must not leak into a store shared by every later call."""
    prepared = sp.prepare(dataset_adata, ["mse"], min_cells=10)
    sp.score(prepared, "mse", predictions_factory(dataset_adata, kind="degraded"))

    assert not [k for k in _pseudobulk_keys(prepared) if k[1][0] == "prediction"]


def test_user_source_pseudobulk_is_never_cached(dataset_adata):
    """`prepare(sources=...)` vectors are per-handle and non-cacheable, like predictions."""
    vector = np.asarray(dataset_adata.X).mean(0)
    prepared = sp.prepare(dataset_adata, ["pearson"], min_cells=10, sources={"myvec": vector})
    sp.calibrate(prepared, "pearson", negative="myvec")

    assert not [k for k in _pseudobulk_keys(prepared) if k[1][0] == "myvec"]


def test_cached_pseudobulk_is_read_only(dataset_adata, predictions_factory):
    prepared = sp.prepare(dataset_adata, ["pearson_ctrl"], min_cells=10)
    sp.score(prepared, "pearson_ctrl", predictions_factory(dataset_adata, kind="perfect"))

    values = [prepared._store.memo[k] for k in _pseudobulk_keys(prepared)]
    assert values and all(not v.flags.writeable for v in values)
    with pytest.raises(ValueError):
        values[0][0] = 0.0


def test_centering_leaves_the_cached_array_untouched(dataset_adata, cfg_factory):
    """Centering must return a new vector rather than subtracting into the shared one."""
    cfg = cfg_factory(truth="gt_all_cells")
    ctx = Context(Dataset(dataset_adata, cfg), cfg)
    pert = ctx.perturbations[0]

    raw = ctx.centroid(pert, "gt_all_cells", None).copy()
    centered = ctx.centroid(pert, "gt_all_cells", "control_mean")

    assert not np.shares_memory(raw, centered)
    np.testing.assert_array_equal(ctx.centroid(pert, "gt_all_cells", None), raw)


def test_de_method_is_part_of_the_key(dataset_adata):
    """`interpolated` blends by DE weights, so its pseudobulk differs per DE method."""
    prepared = sp.prepare(dataset_adata, ["pearson"], min_cells=10)
    for method in ("t-test", "MWU"):
        sp.calibrate(prepared, "pearson", de_method=method, positive="interpolated")

    methods = {k[1][2] for k in _pseudobulk_keys(prepared) if k[1][0] == "interpolated"}
    assert methods == {"t-test", "MWU"}


def test_scores_match_an_uncached_context(dataset_adata, cfg_factory):
    """A fresh Context per perturbation (no shared store) must agree with the cached path."""
    cfg = cfg_factory(truth="gt_all_cells")
    ds = Dataset(dataset_adata, cfg)
    shared = Context(ds, cfg)

    for pert in ds.perturbations:
        for centering in (None, "control_mean", "all_perturbed_mean"):
            expected = Context(ds, cfg).centroid(pert, "gt_all_cells", centering)
            np.testing.assert_array_equal(shared.centroid(pert, "gt_all_cells", centering), expected)
