"""Dataset indexing and cell-selection regressions."""

from unittest.mock import patch

import anndata as ad
import numpy as np

from scperteval.dataset import Dataset
from scperteval.types import RunConfig


def test_all_cells_uses_precomputed_indices_without_rescanning_labels():
    matrix = np.arange(30, dtype=np.float32).reshape(6, 5)
    data = ad.AnnData(matrix)
    data.obs["perturbation"] = ["pertA", "control", "pertA", "pertB", "pertA", "pertB"]
    config = RunConfig(dataset="-", protocols=[], seed=0, min_cells=1)
    dataset = Dataset(data, config)

    with patch("scperteval.dataset.np.where", side_effect=AssertionError("unexpected full-label scan")):
        observed = dataset.cells("pertA")

    np.testing.assert_array_equal(observed, matrix[[0, 2, 4]])


def test_indexing_groups_interleaved_cells_once_in_sorted_perturbation_order():
    matrix = np.arange(40, dtype=np.float32).reshape(8, 5)
    data = ad.AnnData(matrix)
    data.obs["perturbation"] = ["pertB", "control", "pertA", "pertB", "rare", "pertA", "control", "pertB"]
    config = RunConfig(dataset="-", protocols=[], seed=0, min_cells=2)

    dataset = Dataset(data, config)

    assert dataset.perturbations == ["pertA", "pertB"]
    np.testing.assert_array_equal(dataset.indices["pertA"], [2, 5])
    np.testing.assert_array_equal(dataset.indices["pertB"], [0, 3, 7])
    assert "rare" not in dataset.indices
