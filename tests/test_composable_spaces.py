"""Tests for first-class composable FeatureSpace objects, subset-to-PCA piping, and target_key."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_cfg, make_dataset

import scperteval as sp
from scperteval.blocks import spaces
from scperteval.blocks.spaces import SPACES, UndefinedSpace, resolve_space
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.protocols import metrics
from scperteval.types import Protocol


def _context(target_key: str | None = None):
    adata = make_dataset()
    adata.var_names = ["pertA", "pertB", *[f"g{i}" for i in range(58)]]
    cfg = make_cfg(target_key=target_key)
    return Context(Dataset(adata, cfg), cfg)


def test_composable_spaces_do_not_mutate_global_registry():
    before_names = set(SPACES.names())
    before_catalog = {s.name for s in SPACES.catalog()}

    deg = spaces.degs(padj=0.05, top_k=5, rank_by="significance", reference="control") - spaces.own_target()
    hvg_pca = spaces.pca(4).on(spaces.hvg(10))

    assert deg.cache_key != hvg_pca.cache_key
    assert set(SPACES.names()) == before_names
    assert {s.name for s in SPACES.catalog()} == before_catalog


def test_degs_space_is_deterministic_and_preserves_distinct_float_keys():
    ctx = _context()
    panel = spaces.degs(padj=0.05, top_k=3, rank_by="significance", reference="control")
    selected = panel.select(ctx, "pertA")
    de = ctx.de("pertA", ctx.cfg.truth, "control")
    assert len(selected) == 3
    assert np.all(de.pvalue_adj[selected] < 0.05)
    assert panel.select(ctx, "pertA").tolist() == selected.tolist()

    left = spaces.degs(padj=0.05, rank_by="significance")
    right = spaces.degs(padj=0.05000001, rank_by="significance")
    assert left != right
    assert left.cache_key != right.cache_key


def test_degs_supports_absolute_effect_ranking_and_unthresholded_top_k():
    ctx = _context()
    # Stub DE with distinct orderings under significance vs absolute effect.
    fake_de = type(
        "DE",
        (),
        {
            "statistic": np.array([2.0, 9.0, 5.0, 0.5] + [0.0] * 56, dtype=float),
            "pvalue_adj": np.array([0.001, 0.04, 0.01, 0.8] + [0.9] * 56, dtype=float),
        },
    )()
    ctx.de = lambda pert, source, reference="all_perturbed": fake_de  # type: ignore[method-assign]

    by_sig = spaces.degs(padj=0.05, top_k=2, rank_by="significance").select(ctx, "pertA")
    by_eff = spaces.degs(padj=0.05, top_k=2, rank_by="absolute_effect").select(ctx, "pertA")
    no_gate = spaces.degs(padj=None, top_k=4, rank_by="absolute_effect").select(ctx, "pertA")

    assert by_sig.tolist() == [0, 2]
    assert by_eff.tolist() == [1, 2]
    assert no_gate.tolist() == [1, 2, 0, 3]


def test_own_target_composes_with_union_and_difference():
    ctx = _context()
    included = (spaces.hvg(2) | spaces.own_target()).select(ctx, "pertA")
    excluded = (spaces.full() - spaces.own_target()).select(ctx, "pertA")
    assert 0 in included
    assert 0 not in excluded


def test_own_target_reports_unmeasured_target():
    ctx = _context()
    with pytest.raises(UndefinedSpace, match="pertC") as exc_info:
        (spaces.full() - spaces.own_target()).select(ctx, "pertC")
    assert exc_info.value.reason == "target_gene_not_measured"


def test_union_with_own_target_survives_empty_base_subset():
    ctx = _context()
    # Unrealistically tight padj selects 0 DEGs, but unioning own_target still yields {pertA}.
    empty_degs = spaces.degs(padj=1e-300, reference="control")
    with pytest.raises(UndefinedSpace) as exc_info:
        empty_degs.select(ctx, "pertA")
    assert exc_info.value.reason == "empty_gene_space"

    with_target = (empty_degs | spaces.own_target()).select(ctx, "pertA")
    assert with_target.tolist() == [0]


def test_target_key_resolves_namespaced_perturbations():
    adata = make_dataset()
    adata.var_names = ["pertA", "pertB", *[f"g{i}" for i in range(58)]]
    original = adata.obs["perturbation"].astype(str)
    adata.obs["target_gene"] = original
    adata.obs["perturbation"] = original.map(
        {
            "pertA": "dataset-a::pertA",
            "pertB": "dataset-a::pertB",
            "pertC": "dataset-a::pertC",
            "pertD": "dataset-a::pertD",
            "control": "control",
        }
    )
    proto = Protocol(
        "pearson_no_target",
        metrics.pearson,
        "centroid",
        space=spaces.full() - spaces.own_target(),
    )
    prep = sp.prepare(adata, [proto], subsample=400, seed=0, min_cells=10, workers=1, target_key="target_gene")
    avail = sp.truth_availability(prep, [proto]).set_index("perturbation")
    assert bool(avail.loc["dataset-a::pertA", "ground_truth_defined"]) is True
    assert bool(avail.loc["dataset-a::pertB", "ground_truth_defined"]) is True
    assert bool(avail.loc["dataset-a::pertC", "ground_truth_defined"]) is False
    assert avail.loc["dataset-a::pertC", "undefined_reason"] == "target_gene_not_measured"


def test_missing_target_key_raises_clean_error(dataset_adata):
    with pytest.raises(ValueError, match="target_key 'nope' not found"):
        sp.prepare(dataset_adata, [], subsample=400, seed=0, min_cells=10, workers=1, target_key="nope")


def test_transform_on_subset_fits_on_subset_and_warms():
    ctx = _context()
    hvg_pca = spaces.pca(4).on(spaces.hvg(10))
    full_pca = spaces.pca(4)

    proto = Protocol("ed_hvg_pca", metrics.energy_distance, "population", space=hvg_pca, better="lower", perfect=0.0)
    ctx.warm([proto])
    assert any(k[0] == "subset_pca" for k in ctx._store.memo)
    assert ("reference_projection", (hvg_pca.cache_key,)) in ctx._store.memo

    cells = ctx.ds.cells("pertA")
    projected = hvg_pca.apply(cells, ctx, "pertA")
    projected_full = full_pca.apply(cells, ctx, "pertA")
    assert projected.shape == (120, 4)
    assert not np.allclose(projected, projected_full)


def test_transform_on_rejects_per_perturbation_subset():
    with pytest.raises(ValueError, match="dataset-wide subset"):
        spaces.pca(5).on(spaces.degs(padj=0.05))


def test_empty_space_records_undefined_outcome_in_runner(dataset_adata):
    # Both composable spaces and registered string spaces produce NaN + undefined_reason, not a crash.
    empty_proto = Protocol(
        "mse_empty_degs",
        metrics.mse,
        "centroid",
        space=spaces.degs(padj=1e-300),
        better="lower",
        perfect=0.0,
    )
    prepared = sp.prepare(dataset_adata, [empty_proto], subsample=400, seed=0, min_cells=10, workers=1)
    avail = sp.truth_availability(prepared, [empty_proto])
    assert not avail["ground_truth_defined"].any()
    assert set(avail["undefined_reason"]) == {"empty_gene_space"}

    cal = sp.calibrate(prepared, empty_proto, calibrator="bds")
    assert np.isnan(cal.aggregate["bds"])
    assert set(cal.per_perturbation["undefined_reason"]) == {"empty_gene_space"}


def test_resolve_space_preserves_string_cache_key():
    sp_full = resolve_space("full")
    assert sp_full.cache_key == "full"
    assert sp_full.is_subset is True
    assert sp_full.per_pert is False
