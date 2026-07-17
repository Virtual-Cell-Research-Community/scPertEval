"""Runtime control resolution: two-tier defaults, overrides, recording, and validation."""

from __future__ import annotations

import warnings

import pytest

from scperteval.calibrators import CALIBRATORS
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.protocols.resolve import _concrete
from scperteval.protocols.table import PROTOCOLS
from scperteval.runner import _resolve_candidates, _warn_declared_overrides, run_protocol

# The intended default (positive, negative) for every catalog protocol — doubles as documentation.
# Generic by representation (centroid: interpolated/all_perturbed_mean, population & de:
# tech_dup/all_perturbed), with two declared deviations: pearson_pert* -> control (allpert-centred),
# rank/transpose_rank -> global_mean (dataset-scope chance level).
EXPECTED = {
    "pearson": ("interpolated", "all_perturbed_mean"),
    "pearson_ctrl": ("interpolated", "all_perturbed_mean"),
    "pearson_pert": ("interpolated", "control"),
    "mse": ("interpolated", "all_perturbed_mean"),
    "wmse_exp1": ("interpolated", "all_perturbed_mean"),
    "wmse_exp2": ("interpolated", "all_perturbed_mean"),
    "wmse_exp4": ("interpolated", "all_perturbed_mean"),
    "mse_top_k": ("interpolated", "all_perturbed_mean"),
    "mse_degs_padj": ("interpolated", "all_perturbed_mean"),
    "pearson_pert_top_k": ("interpolated", "control"),
    "pearson_pert_degs_padj": ("interpolated", "control"),
    "rank": ("interpolated", "global_mean"),
    "transpose_rank": ("interpolated", "global_mean"),
    "unbiased_mmd_median_top_k": ("tech_dup", "all_perturbed"),
    "unbiased_mmd_median_pca_k": ("tech_dup", "all_perturbed"),
    "energy_distance_top_k": ("tech_dup", "all_perturbed"),
    "energy_distance_pca_k": ("tech_dup", "all_perturbed"),
    "sinkhorn_w2_top_k": ("tech_dup", "all_perturbed"),
    "sinkhorn_w2_pca_k": ("tech_dup", "all_perturbed"),
    "de_auprc": ("tech_dup", "all_perturbed"),
    "de_auroc": ("tech_dup", "all_perturbed"),
    "de_overlap_k": ("tech_dup", "all_perturbed"),
}


def test_expectation_table_covers_catalog():
    assert set(EXPECTED) == set(PROTOCOLS)  # guard against a new protocol slipping past this table


@pytest.mark.parametrize("name", list(EXPECTED))
def test_default_controls_match_catalog(name, cfg_factory):
    # No override (cfg controls are "auto"): the resolved defaults must equal today's controls exactly.
    c = _resolve_candidates(_concrete(PROTOCOLS[name]), cfg_factory())
    assert (c["positive"], c["negative"]) == EXPECTED[name]


def test_override_replaces_generic_default(cfg_factory):
    c = _resolve_candidates(PROTOCOLS["mse"], cfg_factory(negative="control"))
    assert c["negative"] == "control"  # override wins over the generic all_perturbed_mean
    assert c["positive"] == "interpolated"  # untouched controls still resolve to their default


def test_overriding_declared_default_warns(cfg_factory):
    rank = _concrete(PROTOCOLS["rank"])  # declares negative=global_mean
    with pytest.warns(UserWarning, match="declared default"):
        _warn_declared_overrides(cfg_factory(negative="control"), [rank])


def test_overriding_generic_default_is_silent(cfg_factory):
    pearson = PROTOCOLS["pearson"]  # negative is a generic default, not declared
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        _warn_declared_overrides(cfg_factory(negative="control"), [pearson])


def test_resolved_controls_recorded_in_rows(dataset_adata, cfg_factory):
    cfg = cfg_factory()
    ctx = Context(Dataset(dataset_adata, cfg), cfg)
    _, rows, _ = run_protocol(_concrete(PROTOCOLS["pearson_pert"]), ctx, CALIBRATORS["drf"])
    assert rows[0]["positive"] == "interpolated"
    assert rows[0]["negative"] == "control"  # the declared deviation, recorded per row


def test_unknown_control_source_raises(dataset_adata, cfg_factory):
    cfg = cfg_factory(negative="does_not_exist")
    ctx = Context(Dataset(dataset_adata, cfg), cfg)
    with pytest.raises(ValueError, match="unknown negative control source"):
        run_protocol(_concrete(PROTOCOLS["mse"]), ctx, CALIBRATORS["drf"])


def test_centroid_source_rejected_for_population_protocol(dataset_adata, cfg_factory):
    cfg = cfg_factory(negative="all_perturbed_mean")  # a centroid source; population needs cells
    ctx = Context(Dataset(dataset_adata, cfg), cfg)
    with pytest.raises(ValueError, match="needs cells"):
        run_protocol(_concrete(PROTOCOLS["unbiased_mmd_median_top_k"]), ctx, CALIBRATORS["drf"])
