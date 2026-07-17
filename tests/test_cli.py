"""End-to-end CLI dispatch for the calibrate / score / de subcommands."""

from __future__ import annotations

import pytest

from scperteval.cli import main


def test_calibrate_writes_drf_csv(dataset_path, tmp_path):
    main(["calibrate", dataset_path, "-p", "pearson_ctrl,mse", "--out-dir", str(tmp_path), "--quiet"])
    assert len(list(tmp_path.glob("*__drf.csv"))) == 1


def test_calibrate_bds_output(dataset_path, tmp_path):
    main(["calibrate", dataset_path, "-p", "mse", "--calibrator", "bds", "--out-dir", str(tmp_path), "--quiet"])
    assert len(list(tmp_path.glob("*__bds.csv"))) == 1


def test_score_writes_score_csv(dataset_path, dataset_adata, predictions_factory, tmp_path):
    pred_path = tmp_path / "pred.h5ad"
    predictions_factory(dataset_adata, kind="degraded").write_h5ad(pred_path)
    main(["score", dataset_path, str(pred_path), "-p", "pearson,mse,de_auprc", "--out-dir", str(tmp_path), "--quiet"])
    assert len(list(tmp_path.glob("*__score.csv"))) == 1


def test_de_writes_h5(dataset_path, tmp_path):
    main(["de", dataset_path, "--method", "t-test", "--out-dir", str(tmp_path), "--quiet"])
    assert len(list(tmp_path.glob("*__de.h5"))) == 1


def test_calibrate_rejects_score_output(dataset_path, tmp_path):
    # `score` is a scoring-mode calibrator, not selectable from `calibrate --calibrator`
    with pytest.raises(SystemExit):
        main(["calibrate", dataset_path, "-p", "mse", "--calibrator", "score", "--out-dir", str(tmp_path)])
