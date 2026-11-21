from __future__ import annotations

import numpy as np
import pytest
from conftest import make_cfg, make_dataset

from scperteval.blocks.panels import GenePanel, UndefinedPanel, panel_pca_space, panel_space
from scperteval.blocks.spaces import SPACES
from scperteval.context import Context
from scperteval.dataset import Dataset


def _context():
    adata = make_dataset()
    adata.var_names = ["pertA", "pertB", *[f"g{i}" for i in range(58)]]
    cfg = make_cfg()
    return Context(Dataset(adata, cfg), cfg)


def test_deg_panel_is_deterministic_and_significance_gated():
    ctx = _context()
    panel = GenePanel("deg", top_k=3, padj_max=0.05, rank_by="significance")
    selected = panel.select(ctx, "pertA")
    de = ctx.de("pertA", ctx.cfg.truth, "control")
    assert len(selected) == 3
    assert np.all(de.pvalue_adj[selected] < 0.05)
    assert panel.select(ctx, "pertA").tolist() == selected.tolist()


def test_panel_names_preserve_distinct_float_thresholds():
    left = GenePanel("deg", padj_max=0.05, rank_by="significance")
    right = GenePanel("deg", padj_max=0.05000001, rank_by="significance")
    assert left.name != right.name
    assert panel_space(left) != panel_space(right)


def test_target_policy_can_force_include_or_exclude_own_target():
    ctx = _context()
    included = GenePanel("hvg", top_k=2, target_policy="force_include_target").select(ctx, "pertA")
    excluded = GenePanel("all", target_policy="exclude_own_target").select(ctx, "pertA")
    assert 0 in included
    assert 0 not in excluded


def test_target_policy_reports_unmeasured_target():
    ctx = _context()
    with pytest.raises(UndefinedPanel, match="target_gene_not_measured"):
        GenePanel("all", target_policy="exclude_own_target").select(ctx, "pertC")


def test_target_policy_prefers_explicit_target_gene_for_namespaced_conditions():
    adata = make_dataset()
    adata.var_names = ["pertA", "pertB", *[f"g{i}" for i in range(58)]]
    original = adata.obs["perturbation"].astype(str)
    adata.obs["target_gene"] = original
    adata.obs["perturbation"] = original.map(
        {"pertA": "dataset-a::pertA", "pertB": "dataset-a::pertB", "control": "control"}
    )
    cfg = make_cfg()
    ctx = Context(Dataset(adata, cfg), cfg)
    selected = GenePanel("all", target_policy="exclude_own_target").select(ctx, "dataset-a::pertA")
    assert 0 not in selected


def test_panel_space_applies_the_same_selection_to_cells():
    ctx = _context()
    panel = GenePanel("deg", top_k=4, padj_max=0.05, rank_by="absolute_effect")
    space = panel_space(panel)
    got = SPACES[space](ctx.ds.cells("pertA"), ctx, "pertA")
    assert got.shape == (120, 4)


def test_subset_then_pca_uses_panel_and_requested_dimension():
    ctx = _context()
    panel = GenePanel("hvg", top_k=20)
    space = panel_pca_space(panel, 5)
    got = SPACES[space](ctx.ds.cells("pertA"), ctx, "pertA")
    assert got.shape == (120, 5)
