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


def test_missing_dataset_is_a_clean_error(tmp_path):
    """A path that does not exist fails as a CLI error, not an h5py traceback."""
    missing = str(tmp_path / "nope.h5ad")
    with pytest.raises(SystemExit) as e:
        main(["calibrate", missing, "-p", "mse", "--out-dir", str(tmp_path)])
    assert "dataset file not found" in str(e.value)
    assert missing in str(e.value)


def test_missing_predictions_names_the_predictions_file(dataset_path, tmp_path):
    """The message says which of the two inputs was unreadable."""
    missing = str(tmp_path / "nope.h5ad")
    with pytest.raises(SystemExit) as e:
        main(["score", dataset_path, missing, "-p", "mse", "--out-dir", str(tmp_path)])
    assert "predictions file not found" in str(e.value)


def test_unreadable_h5ad_is_a_clean_error(tmp_path):
    """A file that exists but is not an .h5ad reports as a CLI error too."""
    bogus = tmp_path / "bogus.h5ad"
    bogus.write_text("definitely not HDF5")
    with pytest.raises(SystemExit) as e:
        main(["calibrate", str(bogus), "-p", "mse", "--out-dir", str(tmp_path)])
    assert "as an .h5ad file" in str(e.value)


def test_directory_instead_of_file_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["calibrate", str(tmp_path), "-p", "mse", "--out-dir", str(tmp_path)])
    assert "is a directory" in str(e.value)
