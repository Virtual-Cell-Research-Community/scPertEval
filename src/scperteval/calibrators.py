"""Calibrators turn raw control metric values into a final per-metric score.

Each declares the calibrator inputs (candidate names) it needs, a per-perturbation combine, and a
cross-perturbation aggregate.
"""

from __future__ import annotations

import numpy as np

from .types import Calibrator, MetricOutcome


def _validate_drf(p) -> None:
    missing = []
    if p.better is None:
        missing.append("better=('higher' | 'lower')")
    if p.perfect is None:
        missing.append("perfect=<float>")
    if missing:
        raise ValueError(
            f"Protocol {p.name!r} cannot be calibrated with 'drf': specify {' and '.join(missing)} on Protocol(...)."
        )


def _validate_bds(p) -> None:
    if p.better is None:
        raise ValueError(
            f"Protocol {p.name!r} cannot be calibrated with 'bds': specify better=('higher' | 'lower') on Protocol(...)."
        )


def _drf_per_pert(raws, p):
    _validate_drf(p)
    pos, neg = raws["positive"], raws["negative"]
    if not np.isfinite(pos) or not np.isfinite(neg):
        return MetricOutcome(float("nan"), "non_finite_raw_metric")
    beyond_perfect = neg > p.perfect if p.better == "higher" else neg < p.perfect
    if beyond_perfect:
        return MetricOutcome(float("nan"), "negative_beyond_protocol_perfect")
    if p.better == "higher":
        num, den = pos - neg, p.perfect - neg
    else:
        num, den = neg - pos, neg - p.perfect
    return MetricOutcome(float(np.clip(num / (den + 1e-6), -1.0, 1.0)))


def _bds_per_pert(raws, p):
    _validate_bds(p)
    pos, neg = raws["positive"], raws["negative"]
    if not np.isfinite(pos) or not np.isfinite(neg):
        return MetricOutcome(float("nan"), "non_finite_raw_metric")
    wins = pos < neg if p.better == "lower" else pos > neg
    return MetricOutcome(float(wins))


def _score_per_pert(raws, p):
    value = float(raws["prediction"])
    return MetricOutcome(value, None if np.isfinite(value) else "non_finite_raw_metric")


def _finite(values):
    """Return finite entries only, avoiding warnings when every perturbation is undefined."""
    return values[np.isfinite(values)]


def _mean_median(values):
    finite = _finite(values)
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(finite)), "median": float(np.median(finite))}


def _mean_bds(values):
    finite = _finite(values)
    return {"bds": float(np.mean(finite)) if finite.size else float("nan")}


#: ``{name: Calibrator}`` dict — ``drf`` and ``bds`` for calibration mode, ``score`` for prediction-scoring mode.
CALIBRATORS = {
    "drf": Calibrator(
        "drf",
        ("positive", "negative"),
        _drf_per_pert,
        _mean_median,
        description="Dynamic Range Fraction — mean/median over perturbations (Miller et al. 2025)",
        validate=_validate_drf,
    ),
    "bds": Calibrator(
        "bds",
        ("positive", "negative"),
        _bds_per_pert,
        _mean_bds,
        description="Bound Discrimination Score — fraction of perturbations the positive control wins (SBB 2026)",
        validate=_validate_bds,
    ),
    "score": Calibrator(
        "score",
        ("prediction",),
        _score_per_pert,
        _mean_median,
        description="raw metric of a prediction vs ground truth — mean/median over perturbations (prediction-scoring mode)",
    ),
}
