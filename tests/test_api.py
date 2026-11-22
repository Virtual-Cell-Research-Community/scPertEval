"""The native Python API: prepare -> calibrate/score/de (single protocol/method), and hardening."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import get_args

import numpy as np
import pandas as pd
import pytest

import scperteval as sp
from scperteval.api import DatasetDEResults, DEMethodName, EvalResult, Prepared
from scperteval.blocks.de import DE_METHODS
from scperteval.cli import main

PREP = dict(subsample=400, seed=0, min_cells=10, workers=1)


def prep(adata, protocols="all"):
    return sp.prepare(adata, protocols, **PREP)


# --- surface -----------------------------------------------------------------


def test_public_surface():
    assert set(sp.__all__) == {
        "DatasetDEResults",
        "EvalResult",
        "Prepared",
        "__version__",
        "calibrate",
        "de",
        "prepare",
        "score",
    }
    for gone in ("differential_expression", "available_protocols", "DEResults", "DEMethodResult"):
        assert not hasattr(sp, gone)
    assert sp.__version__


def test_de_method_literal_matches_registry():
    assert set(get_args(DEMethodName)) == set(DE_METHODS.names())


# --- prepare + verbs ---------------------------------------------------------


def test_prepare_returns_reusable_handle(dataset_adata):
    p = prep(dataset_adata, ["de_auprc", "de_overlap_k"])
    assert isinstance(p, Prepared)
    a = sp.calibrate(p, "de_auprc")
    b = sp.calibrate(p, "de_overlap_k")  # reuses the shared DE cache on the handle
    assert np.isfinite(a.aggregate["mean"]) and np.isfinite(b.aggregate["mean"])


def test_calibrate_single_protocol(dataset_adata):
    r = sp.calibrate(prep(dataset_adata, "pearson_ctrl"), "pearson_ctrl")
    assert isinstance(r, EvalResult)
    assert set(r.aggregate) == {"mean", "median"}
    assert r.aggregate["mean"] > 0.0
    assert len(r.per_perturbation) == 4
    assert {"protocol", "perturbation", "raw_positive", "raw_negative", "drf"} <= set(r.per_perturbation.columns)


def test_calibrate_rejects_multiple_protocols(dataset_adata):
    with pytest.raises(ValueError, match="one protocol per call"):
        sp.calibrate(prep(dataset_adata), "all")


def test_calibrate_rejects_score_output(dataset_adata):
    with pytest.raises(ValueError, match="score"):
        sp.calibrate(prep(dataset_adata, "mse"), "mse", calibrator="score")


def test_calibrate_bds(dataset_adata):
    r = sp.calibrate(prep(dataset_adata, "mse"), "mse", calibrator="bds")
    assert set(r.aggregate) == {"bds"}
    assert 0.0 <= r.aggregate["bds"] <= 1.0


def test_score_single_protocol(dataset_adata, predictions_factory):
    pred = predictions_factory(dataset_adata, kind="degraded")
    r = sp.score(prep(dataset_adata, "pearson"), "pearson", pred)
    assert isinstance(r, EvalResult)
    assert "score" in r.per_perturbation.columns
    assert np.isfinite(r.aggregate["mean"])


def test_de_single_method(dataset_adata):
    d = sp.de(prep(dataset_adata, []), "t-test")
    assert isinstance(d, DatasetDEResults)
    assert d.statistic.shape == (4, 60)
    assert list(d.statistic.columns) == [f"g{i}" for i in range(60)]
    assert np.isfinite(d.statistic.to_numpy()).all()
    stat, padj = d  # NamedTuple unpacks
    assert stat.shape == padj.shape == (4, 60)


@pytest.mark.parametrize("target", ["gt_half", "gt_all_cells", "tech_dup"])
@pytest.mark.parametrize("reference", ["all_perturbed", "control"])
def test_de_supports_cell_source_combinations(dataset_adata, target, reference):
    result = sp.de(prep(dataset_adata, []), "t-test", target=target, reference=reference)
    assert result.statistic.shape == result.pvalue_adj.shape == (4, 60)
    assert np.isfinite(result.statistic.to_numpy()).all()


def test_de_target_and_reference_select_requested_cell_populations(dataset_adata):
    from scperteval.blocks.de import de_ttest

    prepared = prep(dataset_adata, [])
    all_vs_ctrl = sp.de(prepared, "t-test", target="gt_all_cells", reference="control")
    half_vs_ctrl = sp.de(prepared, "t-test", target="gt_half", reference="control")
    all_vs_pert = sp.de(prepared, "t-test", target="gt_all_cells", reference="all_perturbed")

    ctrl_cells = prepared._ds.control_cells(PREP["subsample"])
    for pert in prepared._ds.perturbations:
        expected_all = de_ttest(prepared._ds.cells(pert), ctrl_cells)
        expected_half = de_ttest(prepared._ds.cells(pert, half="first"), ctrl_cells)
        np.testing.assert_allclose(all_vs_ctrl.statistic.loc[pert].to_numpy(), expected_all.statistic)
        np.testing.assert_allclose(all_vs_ctrl.pvalue_adj.loc[pert].to_numpy(), expected_all.pvalue_adj)
        np.testing.assert_allclose(half_vs_ctrl.statistic.loc[pert].to_numpy(), expected_half.statistic)

    assert not np.allclose(all_vs_ctrl.statistic.to_numpy(), half_vs_ctrl.statistic.to_numpy())
    assert not np.allclose(all_vs_ctrl.statistic.to_numpy(), all_vs_pert.statistic.to_numpy())


def test_de_accepts_per_handle_user_cell_source(dataset_adata):
    from scperteval.blocks.de import de_ttest

    ctrl_cells = np.asarray(dataset_adata[dataset_adata.obs["perturbation"] == "control"].X, dtype=np.float64)
    custom_ref = ctrl_cells[:25]
    prepared = sp.prepare(dataset_adata, [], sources={"custom_ref": custom_ref}, **PREP)
    result = sp.de(prepared, "t-test", target="gt_all_cells", reference="custom_ref")
    expected = de_ttest(prepared._ds.cells("pertA"), custom_ref)
    np.testing.assert_allclose(result.statistic.loc["pertA"].to_numpy(), expected.statistic)
    np.testing.assert_allclose(result.pvalue_adj.loc["pertA"].to_numpy(), expected.pvalue_adj)


def test_de_accepts_custom_registered_cell_source(dataset_adata):
    from scperteval.sources import SOURCES

    @SOURCES.register("test_de_custom_cells", provides="cells", description="test double")
    def _custom_cells(ctx, pert):
        return ctx.ds.cells(pert)

    try:
        prepared = prep(dataset_adata, [])
        custom = sp.de(prepared, "t-test", target="test_de_custom_cells", reference="control")
        expected = sp.de(prepared, "t-test", target="gt_all_cells", reference="control")
        pd.testing.assert_frame_equal(custom.statistic, expected.statistic)
        pd.testing.assert_frame_equal(custom.pvalue_adj, expected.pvalue_adj)
    finally:
        SOURCES._items.pop("test_de_custom_cells", None)


def test_de_selects_requested_perturbations_in_order(dataset_adata):
    prepared = prep(dataset_adata, [])
    complete = sp.de(prepared, "t-test")
    selected = sp.de(prepared, "t-test", perturbations=["pertC", "pertA"])
    expected = ["pertC", "pertA"]
    assert list(selected.statistic.index) == expected
    assert selected.statistic.shape == (2, 60)
    pd.testing.assert_frame_equal(selected.statistic, complete.statistic.loc[expected])
    pd.testing.assert_frame_equal(selected.pvalue_adj, complete.pvalue_adj.loc[expected])


def test_de_rejects_invalid_perturbation_selections(dataset_adata):
    prepared = prep(dataset_adata, [])
    with pytest.raises(ValueError, match="unknown perturbation"):
        sp.de(prepared, "t-test", perturbations=["missing"])
    with pytest.raises(ValueError, match="duplicate perturbation"):
        sp.de(prepared, "t-test", perturbations=["pertA", "pertA"])
    with pytest.raises(ValueError, match="empty"):
        sp.de(prepared, "t-test", perturbations=[])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": "nope"}, "unknown DE target source"),
        ({"reference": "nope"}, "unknown DE reference source"),
        ({"target": "control_mean"}, "expected 'cells'"),
        ({"reference": "global_mean"}, "expected 'cells'"),
        ({"target": "prediction"}, "requires predictions"),
    ],
)
def test_de_rejects_invalid_sources(dataset_adata, kwargs, message):
    with pytest.raises(ValueError, match=message):
        sp.de(prep(dataset_adata, []), "t-test", **kwargs)


def test_de_rejects_unknown_method(dataset_adata):
    with pytest.raises(ValueError, match="unknown DE method"):
        sp.de(prep(dataset_adata, []), "nope")


# --- hardening: mandatory prepare, per-call method, concurrency --------------


@pytest.mark.parametrize(
    "call",
    [
        lambda ad: sp.calibrate(ad, "mse"),
        lambda ad: sp.score(ad, "mse", ad),
        lambda ad: sp.de(ad, "t-test"),
    ],
)
def test_verbs_require_prepared(dataset_adata, call):
    with pytest.raises(TypeError, match="prepare"):
        call(dataset_adata)


def test_per_call_de_method_no_reload(dataset_adata):
    p = prep(dataset_adata, [])
    t = sp.de(p, "t-test").statistic.to_numpy()
    m = sp.de(p, "MWU").statistic.to_numpy()
    assert not np.allclose(t, m)  # distinct methods, one prepared dataset, both cached


def test_concurrent_mixed_verbs_match_sequential(dataset_adata, predictions_factory):
    p = prep(dataset_adata, "all")
    pred = predictions_factory(dataset_adata, kind="degraded")
    jobs = [
        lambda: sp.calibrate(p, "pearson_ctrl").aggregate["mean"],
        lambda: sp.calibrate(p, "mse", calibrator="bds").aggregate["bds"],
        lambda: sp.calibrate(p, "de_auprc").aggregate["mean"],
        lambda: sp.score(p, "pearson", pred).aggregate["mean"],
        lambda: float(sp.de(p, "MWU").statistic.to_numpy().sum()),
    ]
    seq = [j() for j in jobs]
    with ThreadPoolExecutor(max_workers=5) as ex:
        par = list(ex.map(lambda j: j(), jobs))
    assert np.allclose(seq, par, equal_nan=True)


def test_wmse_weights_not_contaminated_across_truths(dataset_adata, predictions_factory):
    # `calibrate` (truth=gt_half) then `score` (truth=gt_all_cells) on ONE handle must NOT let the
    # WMSE weights bleed between the two ground-truth sources (the old per-pert `_weights` bug).
    pred = predictions_factory(dataset_adata, kind="degraded")
    clean = sp.score(prep(dataset_adata, "wmse_exp2"), "wmse_exp2", pred).aggregate
    shared = prep(dataset_adata, "wmse_exp2")
    sp.calibrate(shared, "wmse_exp2")  # would poison a truth-agnostic weights cache
    after = sp.score(shared, "wmse_exp2", pred).aggregate
    assert clean == pytest.approx(after)


def test_score_different_predictions_not_contaminated(dataset_adata, predictions_factory):
    # Scoring a DE-representation protocol on TWO different predictions through ONE handle must not
    # reuse the first prediction's DE (the shared-store `prediction` source is per-call, not cached).
    perfect = predictions_factory(dataset_adata, kind="perfect")
    degraded = predictions_factory(dataset_adata, kind="degraded")
    base_p = sp.score(prep(dataset_adata, "de_auprc"), "de_auprc", perfect).aggregate
    base_d = sp.score(prep(dataset_adata, "de_auprc"), "de_auprc", degraded).aggregate
    assert base_p != pytest.approx(base_d)  # the two predictions genuinely score differently
    shared = prep(dataset_adata, "de_auprc")
    got_p = sp.score(shared, "de_auprc", perfect).aggregate
    got_d = sp.score(shared, "de_auprc", degraded).aggregate  # must NOT reuse `perfect`'s cached DE
    assert got_p == pytest.approx(base_p)
    assert got_d == pytest.approx(base_d)


def test_calibrate_validates_de_method(dataset_adata):
    with pytest.raises(ValueError, match="unknown DE method"):
        sp.calibrate(prep(dataset_adata, "mse"), "mse", de_method="nope")


def test_protocol_must_be_a_string(dataset_adata):
    with pytest.raises(TypeError, match="protocol must be"):
        sp.calibrate(prep(dataset_adata, "mse"), ["pearson", "mse"])


def test_out_dir_no_collision_across_protocols(dataset_adata, tmp_path):
    p = prep(dataset_adata, ["mse", "pearson_ctrl"])
    sp.calibrate(p, "mse", out_dir=str(tmp_path))
    sp.calibrate(p, "pearson_ctrl", out_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*__drf.csv"))) == 2  # distinct filenames, no overwrite


def test_out_dir_same_protocol_no_overwrite(dataset_adata, tmp_path):
    # Two writes of the SAME protocol to one out_dir must not clobber each other (sub-second stamp).
    p = prep(dataset_adata, "mse")
    sp.calibrate(p, "mse", out_dir=str(tmp_path))
    sp.calibrate(p, "mse", out_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*__drf.csv"))) == 2


def test_undeclared_protocol_on_demand(dataset_adata):
    # A PCA protocol not declared to prepare still runs (space computed on first use).
    r = sp.calibrate(prep(dataset_adata, []), "energy_distance_pca_k")
    assert np.isfinite(r.aggregate["mean"])


def test_in_memory_anndata_not_mutated(dataset_adata):
    before = dataset_adata.X.copy()
    p = prep(dataset_adata, ["de_auprc"])
    sp.calibrate(p, "de_auprc")
    sp.de(p, "t-test")
    assert np.array_equal(np.asarray(before), np.asarray(dataset_adata.X))


def test_out_dir_writes_files(dataset_adata, tmp_path):
    p = prep(dataset_adata, "mse")
    sp.calibrate(p, "mse", out_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*__drf.csv"))) == 1
    sp.de(p, "t-test", out_dir=str(tmp_path))
    h5 = list(tmp_path.glob("*__de.h5"))
    assert len(h5) == 1
    import h5py

    with h5py.File(h5[0]) as f:
        assert f["t-test"]["statistic"].shape == (4, 60)
        assert f.attrs["target"] == "gt_half"
        assert f.attrs["reference"] == "all_perturbed"
        assert f.attrs["subsample"] == PREP["subsample"]
        assert f.attrs["seed"] == PREP["seed"]
        assert f["genes"].shape == (60,)


def test_api_matches_cli(dataset_path, tmp_path):
    """The API's per-perturbation table equals the CLI's CSV for the same single protocol."""
    cli_dir = tmp_path / "cli"
    main(
        [
            "calibrate",
            dataset_path,
            "-p",
            "pearson_ctrl",
            "--subsample",
            "400",
            "--seed",
            "0",
            "--min-cells",
            "10",
            "--workers",
            "1",
            "--out-dir",
            str(cli_dir),
            "--quiet",
        ]
    )
    cli_df = pd.read_csv(next(cli_dir.glob("*__drf.csv")))
    api = sp.calibrate(sp.prepare(dataset_path, "pearson_ctrl", **PREP), "pearson_ctrl")
    pd.testing.assert_frame_equal(
        api.per_perturbation.reset_index(drop=True), cli_df.reset_index(drop=True), check_dtype=False
    )
