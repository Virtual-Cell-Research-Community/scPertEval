"""Core dataclasses shared across the package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np


@dataclass
class RunConfig:
    """Resolved options for a single run."""

    #: Path to the preprocessed ``.h5ad`` file, or a display name when the API is given an
    #: in-memory AnnData (used only to label output files and the summary).
    dataset: str
    #: Names of the resolved (concrete) protocols to run.
    protocols: list[str]
    #: DE backend for every DE-dependent unit (default ``"t-test"``).
    de_method: str = "t-test"
    #: Number of cells in the all-perturbed reference sample (default 8192).
    subsample: int = 8192
    #: Random seed for subsampling and reproducibility (default 42).
    seed: int = 42
    #: Override the positive control source; ``"auto"`` defers to the protocol.
    positive: str = "auto"
    #: Override the negative control source; ``"auto"`` defers to the protocol.
    negative: str = "auto"
    #: Label of the ground-truth condition used for DE (the perturbation key whose cells serve as the ground-truth target; default ``"gt_half"``).
    truth: str = "gt_half"
    #: Path to a model-predictions ``.h5ad`` for prediction-scoring mode; ``None`` selects calibration mode (default).
    predictions: str | None = None
    #: Calibrator to apply — ``"drf"`` or ``"bds"`` (default ``"drf"``).
    calibrator: str = "drf"
    #: Directory for output CSV files (default ``"results"``).
    out_dir: str = "results"
    #: Number of worker threads; 0 auto-detects (default 0).
    workers: int = 0
    #: Minimum cells required to evaluate a perturbation (default 30).
    min_cells: int = 30
    #: ``adata.obs`` column holding perturbation labels (default ``"perturbation"``).
    perturbation_key: str = "perturbation"
    #: Label in ``perturbation_key`` that identifies control cells (default ``"control"``).
    control_label: str = "control"
    #: If ``True``, also write a per-protocol wall-clock timing CSV.
    profile: bool = False


@dataclass(frozen=True)
class PerturbationDEResult:
    """Per-gene differential expression for one perturbation (one target-vs-reference comparison)."""

    #: Per-gene test statistic (e.g. t-statistic or Cliff's delta), shape ``(G,)``.
    statistic: np.ndarray
    #: Raw per-gene p-values, shape ``(G,)``.
    pvalue: np.ndarray
    #: Benjamini-Hochberg adjusted p-values, shape ``(G,)``.
    pvalue_adj: np.ndarray
    #: Optional method-specific extras (e.g. ``{"u": u_statistic}`` for MWU).
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Param:
    """A protocol's tunable knob: how a CLI value is cast, defaulted, and applied.

    ``space`` maps the value (``k=30``, ``padj=0.05``) to a feature-space name; when it is
    ``None`` the value is passed straight to the metric as a keyword argument.
    """

    #: Keyword argument name passed to the metric or space factory (e.g. ``"k"``).
    name: str
    #: Type cast applied to the CLI string (e.g. ``int``, ``float``).
    cast: Callable
    #: Default value used when no value is given on the CLI.
    default: float
    #: Factory mapping the value to a feature-space name (e.g. ``top_space``); ``None`` passes the value directly to the metric.
    space: Callable | None = None


@dataclass(frozen=True)
class Protocol:
    """An evaluation protocol: a pure metric plus its data and control wiring.

    ``representation`` and ``scope`` are independent and together decide what the metric
    receives:

    - ``representation`` — the shape of one perturbation's datapoint: ``"centroid"`` (a 1-D
      pseudobulk vector), ``"population"`` (a cells × genes matrix), or ``"de"`` (a PerturbationDEResult).
    - ``scope`` — ``"perturbation"`` (default): the metric is called once per perturbation,
      gets that perturbation's ``(gt, prediction)`` datapoints, and returns a scalar.
      ``"dataset"``: the metric is called once, gets the list of *every* perturbation's
      ``gt`` and ``prediction`` datapoints, and returns one score per perturbation (e.g. a
      cross-perturbation retrieval rank).

    Set ``param`` to make the protocol *tunable* — its feature space (or a metric argument)
    is then chosen per run from a CLI value, e.g. ``-p mse_top_k=30``; with no value the
    parameter's default is used. Leave ``param`` unset for a fully-specified protocol.

    ``better`` and ``perfect`` describe the metric's score scale and are independent:

    - ``better`` — which direction is an improvement, ``"higher"`` or ``"lower"``.
      Correlations and overlaps improve as they go up (``"higher"``); errors and
      distances improve as they go down (``"lower"``). This is the metric's *sense*,
      and it is not implied by ``perfect`` — e.g. perplexity has ``perfect=1.0`` yet
      ``better="lower"``, and a log-likelihood has ``perfect=0.0`` yet ``better="higher"``.
    - ``perfect`` — the value a flawless prediction attains (1.0 for a correlation,
      0.0 for an error). It anchors the top of the DRF scale.

    Together they let the calibrator orient the score: DRF measures how far the positive
    control moves from the negative-control floor toward ``perfect`` in the ``better``
    direction, and BDS counts the perturbations where the positive control is ``better``
    than the negative.
    """

    #: Protocol identifier used on the CLI (``-p name``) and in output CSVs.
    name: str
    #: Pure metric function ``(gt, prediction, ctx) -> float``.
    metric: Callable
    #: One of ``"centroid"`` (pseudobulk vector), ``"population"`` (cells × genes matrix), or ``"de"`` (PerturbationDEResult).
    representation: str
    #: ``"perturbation"`` (default) or ``"dataset"`` — how many perturbations the metric sees at once.
    scope: str = "perturbation"
    #: Feature space applied before scoring (default ``"full"``).
    space: str = "full"
    #: A centroid **source name** whose centroid is subtracted before scoring
    #: (e.g. ``"control_mean"``, ``"all_perturbed_mean"``), or ``None`` for no centering.
    centering: str | None = None
    #: Source used as the reference for the GT DE computation (default ``"all_perturbed"``).
    reference: str = "all_perturbed"
    #: Reference for the negative-control DE computation; ``None`` uses ``reference``.
    neg_reference: str | None = None
    #: ``"higher"`` or ``"lower"`` — which direction improves the score.
    better: str = "higher"
    #: Score a flawless prediction attains (e.g. 1.0 for correlations, 0.0 for errors).
    perfect: float = 1.0
    #: Declared default positive control; ``None`` uses the representation's generic default.
    default_positive: str | None = None
    #: Declared default negative control; ``None`` uses the representation's generic default.
    default_negative: str | None = None
    #: Display group for ``scperteval list protocols`` (e.g. ``"pseudobulk"``).
    group: str = ""
    #: If set, makes the protocol tunable from the CLI; ``None`` for fixed protocols.
    param: Param | None = None
    #: Optional-dependency extra this protocol's metric needs (e.g. ``"sinkhorn"``); ``None``
    #: for protocols that run on the base install. Bulk selectors (``all``, a group) skip
    #: protocols whose extra is not installed; naming one explicitly is an error.
    requires_extra: str | None = None

    @property
    def parameterised(self) -> bool:
        """Whether this protocol takes a CLI-supplied parameter."""
        return self.param is not None

    def resolve(self, value) -> Protocol:
        """Concrete protocol for a tunable one at ``value`` (sets the space or metric arg)."""
        assert self.param is not None  # resolve() is only called on parameterised protocols
        suffix = f"{value:g}" if isinstance(value, float) else str(value)
        name = f"{self.name}={suffix}"
        if self.param.space is not None:
            return replace(self, name=name, space=self.param.space(value), param=None)
        metric = partial(self.metric, **{self.param.name: value})
        return replace(self, name=name, metric=metric, param=None)


@dataclass(frozen=True)
class Calibrator:
    """Turns per-control raw metric values into per-perturbation and aggregate scores."""

    #: Registry key and output column name (e.g. ``"drf"``).
    name: str
    #: Calibrator inputs (candidate names) it needs — e.g. ``("positive", "negative")``.
    requires: tuple[str, ...]
    #: ``(raws: dict, protocol: Protocol) -> float`` — combines raw control values into one per-perturbation calibrated score.
    per_pert: Callable
    #: ``(values: numpy.ndarray) -> dict`` — reduces per-perturbation scores into summary statistics (e.g. ``{"mean": …, "median": …}``).
    aggregate: Callable
    #: Human-readable description shown by ``scperteval list calibrators``.
    description: str = ""
