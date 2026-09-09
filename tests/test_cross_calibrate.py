"""cross_calibrate(): the cross-pairing execution path and the calibrator contract it enforces.

scPertEval ships no cross calibrator — only the mechanism — so these tests register throwaway ones
and remove them again, the way a caller would register their own.
"""

from __future__ import annotations

import numpy as np
import pytest

import scperteval as sp
from scperteval.calibrators import CALIBRATORS
from scperteval.dataset import to_dense
from scperteval.types import Calibrator, CrossRow

PREP = dict(subsample=400, seed=0, min_cells=10, workers=1)

#: One protocol per representation, plus a weighted one and a per-perturbation space.
REPRESENTATIVE = ["pearson_ctrl", "mse_top_k=50", "energy_distance_pca_k=50", "de_auprc", "wmse_exp2"]


@pytest.fixture
def prep(dataset_adata):
    return sp.prepare(dataset_adata, "all", **PREP)


@pytest.fixture
def registered():
    """Register throwaway cross calibrators for one test, then restore the registry."""
    before = dict(CALIBRATORS)

    def count_references(row, p):
        """Deliberately trivial: how many references the row holds."""
        return float(len(row))

    def target_value(row, p):
        """Reads `target` — the reason CrossRow carries it rather than per_pert taking a third arg."""
        return float("nan") if row.target is None else float(row[row.target])

    CALIBRATORS["_count"] = Calibrator(
        "_count", (), count_references, lambda v: {"mean": float(np.nanmean(v))}, pairing="cross"
    )
    CALIBRATORS["_target"] = Calibrator(
        "_target", (), target_value, lambda v: {"mean": float(np.nanmean(v))}, pairing="cross"
    )
    yield
    CALIBRATORS.clear()
    CALIBRATORS.update(before)


def cells_of(prepared, pert):
    return np.asarray(to_dense(prepared._ds.cells(pert)), dtype=np.float64)


# --- the path runs end to end -------------------------------------------------


def test_returns_one_row_per_query(prep, registered):
    res = sp.cross_calibrate(prep, "pearson_ctrl", ["pertA", "pertB"], calibrator="_count")
    assert list(res.per_perturbation["query"]) == ["pertA", "pertB"]
    assert res.aggregate == {"mean": float(len(prep._ds.perturbations))}


def test_row_records_what_the_score_was_computed_over(prep, registered):
    res = sp.cross_calibrate(prep, "pearson_ctrl", ["pertA"], calibrator="_count", references=["pertB", "pertC"])
    (row,) = res.per_perturbation.to_dict("records")
    assert row["protocol"] == "pearson_ctrl"
    assert row["origin"] == "pertA"  # a query named from the handle defaults to its own name
    assert row["references"] == 2
    assert row["_count"] == 2.0


def test_raw_reference_values_are_not_columns(prep, registered):
    """One column per reference would make the frame grow with the library; compare() is for those."""
    res = sp.cross_calibrate(prep, "pearson_ctrl", ["pertA"], calibrator="_count")
    assert not set(res.per_perturbation.columns) & set(prep._ds.perturbations)


@pytest.mark.parametrize("protocol", REPRESENTATIVE)
def test_every_representation_runs(prep, registered, protocol):
    res = sp.cross_calibrate(prep, protocol, ["pertA"], calibrator="_count")
    assert np.isfinite(res.per_perturbation["_count"]).all()


def test_repr_counts_queries_not_perturbations(prep, registered):
    assert "querys=1" in repr(sp.cross_calibrate(prep, "mse", ["pertA"], calibrator="_count"))


# --- the calibrator sees the values, keyed by reference -----------------------


def test_scores_match_the_compare_row(prep, registered):
    """The reduction is applied to exactly the row compare() returns raw."""
    refs = ["pertB", "pertC", "pertD"]
    row = sp.compare(prep, "pearson_ctrl", ["pertA"], references=refs)
    res = sp.cross_calibrate(prep, "pearson_ctrl", ["pertA"], calibrator="_target", references=[*refs, "pertA"])
    expected = float(sp.compare(prep, "pearson_ctrl", ["pertA"], references=["pertA"]).iloc[0, 0])
    assert res.per_perturbation["_target"].iloc[0] == expected
    assert row.shape == (1, 3)


def test_target_is_none_for_an_external_query(prep, registered):
    """A bare array has no counterpart among the references, so `target` is None."""
    res = sp.cross_calibrate(prep, "pearson_ctrl", cells_of(prep, "pertA"), calibrator="_target")
    assert res.per_perturbation["origin"].iloc[0] is None
    assert np.isnan(res.per_perturbation["_target"].iloc[0])


def test_cross_row_is_a_plain_mapping():
    row = CrossRow({"a": 1.0, "b": 2.0}, target="a")
    assert dict(row) == {"a": 1.0, "b": 2.0}
    assert row["b"] == 2.0 and len(row) == 2 and list(row) == ["a", "b"]
    assert row.target == "a"
    assert "target='a'" in repr(row)


# --- pairing is enforced in both directions -----------------------------------


def test_cross_calibrate_refuses_a_same_label_calibrator(prep):
    for name in ("drf", "bds", "score"):
        with pytest.raises(ValueError, match="pairing='same-label'"):
            sp.cross_calibrate(prep, "pearson_ctrl", ["pertA"], calibrator=name)


def test_calibrate_refuses_a_cross_calibrator(prep, registered):
    with pytest.raises(ValueError, match="cross_calibrate"):
        sp.calibrate(prep, "pearson_ctrl", calibrator="_count")


def test_unknown_calibrator_is_named(prep):
    with pytest.raises(ValueError, match="unknown calibrator 'nope'"):
        sp.cross_calibrate(prep, "pearson_ctrl", ["pertA"], calibrator="nope")


def test_dataset_scope_protocols_are_refused(prep, registered):
    with pytest.raises(ValueError, match="dataset-scope"):
        sp.cross_calibrate(prep, "rank", ["pertA"], calibrator="_count")


# --- the same-label calibrators are untouched ---------------------------------


def test_builtin_calibrators_declare_same_label():
    assert {c.pairing for c in CALIBRATORS.values()} == {"same-label"}


def test_calibrate_and_score_are_unchanged(prep, dataset_adata):
    """Nothing about the existing paths may shift when a pairing axis is added."""
    from conftest import make_predictions

    assert sp.calibrate(prep, "pearson_ctrl").aggregate == sp.calibrate(prep, "pearson_ctrl").aggregate
    drf = sp.calibrate(prep, "pearson_ctrl", calibrator="drf")
    assert set(drf.per_perturbation.columns) >= {"perturbation", "positive", "negative", "drf"}
    scored = sp.score(prep, "pearson_ctrl", make_predictions(dataset_adata, "perfect"))
    assert "raw_prediction" in scored.per_perturbation


# --- out_dir ------------------------------------------------------------------


def test_out_dir_writes_the_per_query_csv(prep, registered, tmp_path):
    import pandas as pd

    res = sp.cross_calibrate(prep, "pearson_ctrl", ["pertA", "pertB"], calibrator="_count", out_dir=str(tmp_path))
    (written,) = list(tmp_path.glob("*.csv"))
    assert "_count" in written.name  # the calibrator names the file, as it does for drf/bds
    back = pd.read_csv(written)
    assert list(back["query"]) == ["pertA", "pertB"]
    np.testing.assert_allclose(back["_count"], res.per_perturbation["_count"])


def test_no_out_dir_writes_nothing(prep, registered, tmp_path):
    sp.cross_calibrate(prep, "mse", ["pertA"], calibrator="_count")
    assert list(tmp_path.iterdir()) == []
