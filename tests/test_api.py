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
        "PROTOCOLS",
        "DatasetDEResults",
        "EvalResult",
        "MetricOutcome",
        "Prepared",
        "Protocol",
        "__version__",
        "calibrate",
        "de",
        "metrics",
        "prepare",
        "score",
        "spaces",
        "truth_availability",
    }
    for gone in ("differential_expression", "available_protocols", "DEResults", "DEMethodResult"):
        assert not hasattr(sp, gone)
    assert sp.__version__
    from scperteval.blocks import spaces as blocks_spaces
    from scperteval.protocols import metrics as proto_metrics
    from scperteval.protocols.table import PROTOCOLS as table_protocols
    from scperteval.types import MetricOutcome, Protocol

    assert sp.Protocol is Protocol
    assert sp.MetricOutcome is MetricOutcome
    assert sp.spaces is blocks_spaces
    assert sp.metrics is proto_metrics
    assert sp.PROTOCOLS is table_protocols


def test_top_level_import_stays_lazy():
    import subprocess
    import sys

    code = (
        "import sys, scperteval; "
        "_ = scperteval.__version__; "
        "loaded = [m for m in ('anndata', 'pandas', 'sklearn', 'torch', 'scperteval.api') if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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


def test_concrete_protocol_objects_are_supported(dataset_adata, predictions_factory):
    protocol = sp.Protocol("external_mae", sp.metrics.mae, representation="centroid")
    assert protocol.better == "lower"
    assert protocol.perfect == 0.0
    prepared = prep(dataset_adata, protocol)
    cal = sp.calibrate(prepared, protocol)
    assert cal.per_perturbation["protocol"].unique().tolist() == ["external_mae"]
    pred = predictions_factory(dataset_adata, kind="degraded")
    scored = sp.score(prepared, protocol, pred)
    assert scored.per_perturbation["protocol"].unique().tolist() == ["external_mae"]


def test_builtin_metric_scale_inference_and_catalog_consistency(dataset_adata):
    import inspect
    from functools import partial

    public_metrics = [
        fn
        for name, fn in inspect.getmembers(sp.metrics, inspect.isfunction)
        if not name.startswith("_") and fn.__module__ == sp.metrics.__name__
    ]
    assert public_metrics
    for fn in public_metrics:
        scale = getattr(fn, "_scperteval_scale", None)
        assert scale is not None, f"built-in metric {fn.__name__} missing _scperteval_scale"
        proto = sp.Protocol(f"test_{fn.__name__}", fn, representation="centroid")
        assert (proto.better, proto.perfect) == scale

    for proto in sp.PROTOCOLS.values():
        fn = proto.metric
        while isinstance(fn, partial):
            fn = fn.func
        assert (proto.better, proto.perfect) == getattr(fn, "_scperteval_scale", None)

    # Inference unwraps functools.partial and lets explicit overrides win.
    wrapped = sp.Protocol("wmse_p", partial(sp.metrics.weighted_mse, exp=1.0), representation="centroid")
    assert (wrapped.better, wrapped.perfect) == ("lower", 0.0)
    de_wrapped = sp.Protocol("auroc_p", partial(sp.metrics.de_auroc, top_k=20), representation="de")
    assert (de_wrapped.better, de_wrapped.perfect) == ("higher", 1.0)
    overridden = sp.Protocol("override", sp.metrics.mse, representation="centroid", better="higher", perfect=2.0)
    assert (overridden.better, overridden.perfect) == ("higher", 2.0)

    # Calibrating an inferred lower-is-better metric matches the curated catalog protocol.
    inferred_mse = sp.Protocol("my_mse", sp.metrics.mse, representation="centroid")
    prepared = prep(dataset_adata, ["mse", inferred_mse])
    for cal_name in ("drf", "bds"):
        expected = sp.calibrate(prepared, "mse", calibrator=cal_name).aggregate
        actual = sp.calibrate(prepared, inferred_mse, calibrator=cal_name).aggregate
        assert actual == pytest.approx(expected)


def test_custom_metric_scale_validation(dataset_adata, predictions_factory):
    def custom_metric(gt, pred, ctx):
        return float(np.mean(gt - pred))

    proto_unset = sp.Protocol("custom_unset", custom_metric, representation="centroid")
    assert proto_unset.better is None
    assert proto_unset.perfect is None

    # score() and truth_availability() work without better/perfect.
    prepared = prep(dataset_adata, [proto_unset])
    pred = predictions_factory(dataset_adata, kind="degraded")
    assert np.isfinite(sp.score(prepared, proto_unset, pred).aggregate["mean"])
    assert sp.truth_availability(prepared, [proto_unset])["ground_truth_defined"].all()

    # calibrate() fails fast for missing scale metadata.
    with pytest.raises(ValueError, match=r"better=.*perfect="):
        sp.calibrate(prepared, proto_unset, calibrator="drf")
    with pytest.raises(ValueError, match="better="):
        sp.calibrate(prepared, proto_unset, calibrator="bds")

    # better-only custom protocol can run BDS, but DRF still requires perfect.
    proto_bds = sp.Protocol("custom_bds", custom_metric, representation="centroid", better="higher")
    assert 0.0 <= sp.calibrate(prepared, proto_bds, calibrator="bds").aggregate["bds"] <= 1.0
    with pytest.raises(ValueError, match="perfect="):
        sp.calibrate(prepared, proto_bds, calibrator="drf")

    # Invalid better / non-finite perfect fail eagerly on Protocol construction.
    with pytest.raises(ValueError, match="better must be"):
        sp.Protocol("bad_dir", custom_metric, representation="centroid", better="up")
    with pytest.raises(ValueError, match="perfect must be finite"):
        sp.Protocol("bad_perf", custom_metric, representation="centroid", better="higher", perfect=float("nan"))


def test_concrete_protocol_cannot_shadow_curated_name(dataset_adata):
    from scperteval.protocols import metrics
    from scperteval.protocols.table import PROTOCOLS
    from scperteval.types import Protocol

    misleading = Protocol("mse", metrics.mae, representation="centroid", better="lower", perfect=0.0)
    with pytest.raises(ValueError, match="conflicts with a curated protocol"):
        prep(dataset_adata, misleading)
    # Passing the exact curated Protocol object itself is allowed.
    prepared = prep(dataset_adata, PROTOCOLS["mse"])
    assert sp.calibrate(prepared, PROTOCOLS["mse"]).per_perturbation["protocol"].unique().tolist() == ["mse"]
    # Unresolved parameterised templates are rejected.
    with pytest.raises(ValueError, match="still parameterised"):
        prep(dataset_adata, PROTOCOLS["mse_top_k"])


def test_concrete_protocol_name_collisions_are_rejected(dataset_adata):
    from scperteval.protocols import metrics
    from scperteval.types import Protocol

    left = Protocol("same", metrics.mse, representation="centroid")
    right = Protocol("same", metrics.pearson, representation="centroid")
    with pytest.raises(ValueError, match="two different concrete protocols"):
        prep(dataset_adata, [left, right])


def test_truth_availability_self_evaluates_without_predictions(dataset_adata):
    from scperteval.protocols import metrics
    from scperteval.types import Protocol

    protocols = [
        Protocol("external_mse", metrics.mse, representation="centroid", better="lower", perfect=0.0),
        Protocol("external_pearson", metrics.pearson, representation="centroid"),
    ]
    prepared = prep(dataset_adata, protocols)
    availability = sp.truth_availability(prepared, protocols)
    assert len(availability) == 8
    assert availability["ground_truth_defined"].all()
    assert set(availability["protocol"]) == {"external_mse", "external_pearson"}


def test_truth_availability_supports_truth_source_and_dataset_scope(dataset_adata):
    prepared = prep(dataset_adata, "rank")
    for truth in ("gt_all_cells", "gt_half"):
        availability = sp.truth_availability(prepared, "rank", truth=truth)
        assert len(availability) == 4
        assert availability["ground_truth_defined"].all()
        assert availability["self_score"].eq(0.0).all()
    with pytest.raises(ValueError, match="unknown truth source"):
        sp.truth_availability(prepared, "rank", truth="missing")


def test_truth_availability_reports_undefined_perturbations(dataset_adata):
    from scperteval.types import MetricOutcome, Protocol

    def sometimes_undefined(gt, pred, ctx):
        if ctx.current_pert == "pertA":
            return MetricOutcome(float("nan"), "constant_ground_truth")
        return 1.0

    proto = Protocol("partial_defined", sometimes_undefined, representation="centroid")
    prepared = prep(dataset_adata, [proto])
    avail = sp.truth_availability(prepared, [proto]).set_index("perturbation")
    assert bool(avail.loc["pertA", "ground_truth_defined"]) is False
    assert avail.loc["pertA", "undefined_reason"] == "constant_ground_truth"
    assert np.isnan(avail.loc["pertA", "self_score"])
    assert bool(avail.loc["pertB", "ground_truth_defined"]) is True
    assert pd.isna(avail.loc["pertB", "undefined_reason"])
    assert avail.loc["pertB", "self_score"] == 1.0


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
