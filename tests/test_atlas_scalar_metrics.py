from __future__ import annotations

import numpy as np
import pytest

from scperteval.protocols import metrics


@pytest.mark.parametrize(
    ("metric", "perfect"),
    [
        (metrics.mae, 0.0),
        (metrics.spearman, 1.0),
        (metrics.cosine, 1.0),
        (metrics.squared_pearson, 1.0),
        (metrics.explained_variance, 1.0),
        (metrics.clipped_r2, 1.0),
    ],
)
def test_atlas_scalar_metrics_have_expected_self_score(metric, perfect):
    values = np.array([0.0, 2.0, -1.0, 4.0])
    assert metric(values, values, None) == pytest.approx(perfect)


@pytest.mark.parametrize("metric", [metrics.spearman, metrics.cosine])
def test_correlation_metrics_use_agreed_constant_conventions(metric):
    varying = np.array([0.0, 1.0, 2.0])
    constant = np.ones(3)
    assert metric(varying, constant, None) == 0.0
    assert np.isnan(metric(constant, varying, None))
