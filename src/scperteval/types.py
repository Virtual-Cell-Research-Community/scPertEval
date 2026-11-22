"""Core dataclasses shared across the package."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .blocks.spaces.registry import FeatureSpace


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
    #: Optional ``adata.obs`` column holding target gene symbols (when perturbation labels are not gene symbols).
    target_key: str | None = None
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
class MetricOutcome:
    """A scalar metric value with an optional machine-readable undefined reason."""

    value: float
    undefined_reason: str | None = None


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
    #: Maps the value to a feature-space name (e.g. ``partial(SPACES.instance, "top")``); ``None`` passes the value directly to the metric.
    space: Callable | None = None


_VALID_REPRESENTATIONS = ("centroid", "population", "de")
_VALID_SCOPES = ("perturbation", "dataset")


def _space_de_references(space) -> list[tuple[str, str]]:
    """Collect ``(rule_name, reference)`` pairs from any DE-backed subsets inside ``space``."""
    out: list[tuple[str, str]] = []
    left, right = getattr(space, "left", None), getattr(space, "right", None)
    if left is not None and right is not None:
        out.extend(_space_de_references(left))
        out.extend(_space_de_references(right))
        return out
    params = getattr(space, "params", None)
    name = getattr(space, "name", None)
    if isinstance(params, tuple) and isinstance(name, str):
        ref = dict(params).get("reference")
        if isinstance(ref, str):
            out.append((name, ref))
    return out


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

    Built-in metrics in :mod:`scperteval.protocols.metrics` (including ``functools.partial``
    wrappers) declare their intrinsic ``(representation, scope, better, perfect)`` contract:
    ``representation`` and ``scope`` are inferred automatically and strictly enforced against
    contradictions, while ``better`` and ``perfect`` act as overridable defaults. Custom user
    metrics remain plain functions — specify ``representation`` explicitly (``scope`` defaults
    to ``"perturbation"``), and specify ``better`` / ``perfect`` when calling ``calibrate()``.
    """

    #: Protocol identifier used on the CLI (``-p name``) and in output CSVs.
    name: str
    #: Pure metric function ``(gt, prediction, ctx) -> float``.
    metric: Callable
    #: One of ``"centroid"``, ``"population"``, or ``"de"``; inferred for built-in metrics.
    representation: str | None = None
    #: ``"perturbation"`` (default) or ``"dataset"``; inferred for built-in metrics.
    scope: str | None = None
    #: Feature space applied before scoring — a registered space name or a composable
    #: :class:`~scperteval.blocks.spaces.FeatureSpace` object (default ``"full"``).
    space: str | FeatureSpace = "full"
    #: A centroid **source name** whose centroid is subtracted before scoring
    #: (e.g. ``"control_mean"``, ``"all_perturbed_mean"``), or ``None`` for no centering.
    centering: str | None = None
    #: Reference source for ``representation="de"`` (defaults to ``"all_perturbed"`` for DE protocols; invalid otherwise).
    reference: str | None = None
    #: Reference for the negative-control DE computation; ``None`` uses ``reference`` (or ``"control"`` when hybrid).
    neg_reference: str | None = None
    #: ``"higher"`` or ``"lower"`` — which direction improves the score; inferred for built-in metrics.
    better: str | None = None
    #: Score a flawless prediction attains (e.g. 1.0 for correlations, 0.0 for errors); inferred for built-in metrics.
    perfect: float | None = None
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

    def __post_init__(self) -> None:
        fn = self.metric
        while isinstance(fn, partial):
            fn = fn.func
        meta = getattr(fn, "_scperteval_metric", None)

        if self.representation is None:
            if meta is not None:
                object.__setattr__(self, "representation", meta[0])
            else:
                raise ValueError(
                    f"Protocol {self.name!r}: representation is required for custom metrics "
                    f"(choose one of {', '.join(repr(r) for r in _VALID_REPRESENTATIONS)})."
                )
        if self.representation not in _VALID_REPRESENTATIONS:
            raise ValueError(
                f"Protocol {self.name!r}: representation must be one of "
                f"{', '.join(repr(r) for r in _VALID_REPRESENTATIONS)}, got {self.representation!r}"
            )
        if meta is not None and self.representation != meta[0]:
            metric_name = getattr(fn, "__name__", repr(fn))
            raise ValueError(
                f"Protocol {self.name!r}: metric {metric_name!r} requires "
                f"representation={meta[0]!r}, got {self.representation!r}"
            )

        if self.scope is None:
            object.__setattr__(self, "scope", meta[1] if meta is not None else "perturbation")
        if self.scope not in _VALID_SCOPES:
            raise ValueError(
                f"Protocol {self.name!r}: scope must be one of "
                f"{', '.join(repr(s) for s in _VALID_SCOPES)}, got {self.scope!r}"
            )
        if meta is not None and self.scope != meta[1]:
            metric_name = getattr(fn, "__name__", repr(fn))
            raise ValueError(
                f"Protocol {self.name!r}: metric {metric_name!r} requires scope={meta[1]!r}, got {self.scope!r}"
            )

        if meta is not None:
            if self.better is None:
                object.__setattr__(self, "better", meta[2])
            if self.perfect is None:
                object.__setattr__(self, "perfect", meta[3])
        if self.better is not None and self.better not in ("higher", "lower"):
            raise ValueError(f"Protocol {self.name!r}: better must be 'higher' or 'lower', got {self.better!r}")
        if self.perfect is not None and not np.isfinite(self.perfect):
            raise ValueError(f"Protocol {self.name!r}: perfect must be finite, got {self.perfect!r}")

        if self.centering is not None and self.representation != "centroid":
            # TODO: consider supporting cell-level centering on representation="population" for non-translation-invariant metrics.
            raise ValueError(
                f"Protocol {self.name!r}: centering={self.centering!r} only applies to "
                f"representation='centroid', got {self.representation!r}"
            )

        if self.representation != "de":
            if self.reference is not None or self.neg_reference is not None:
                raise ValueError(
                    f"Protocol {self.name!r}: reference/neg_reference only apply to representation='de' "
                    f"(for DEG spaces or DE-weighted metrics on {self.representation!r} protocols, pass "
                    f"reference=... to spaces.degs(), spaces.top(), or partial(weighted_*, reference=...))."
                )
        else:
            if self.reference is None:
                object.__setattr__(self, "reference", "all_perturbed")
            for space_name, space_ref in _space_de_references(self.space):
                if space_ref != self.reference:
                    warnings.warn(
                        f"Protocol {self.name!r}: space {space_name!r} uses reference={space_ref!r}, "
                        f"which differs from Protocol.reference={self.reference!r}.",
                        UserWarning,
                        stacklevel=3,
                    )

    @property
    def parameterised(self) -> bool:
        """Whether this protocol takes a CLI-supplied parameter."""
        return self.param is not None

    def effective_neg_reference(self, negative_source: str) -> str:
        """Reference population for ``negative_source`` in a DE protocol.

        Uses explicit ``neg_reference`` when set; otherwise avoids testing ``all_perturbed``
        against itself by falling back to ``"control"`` (the hybrid DE setup).
        """
        if self.neg_reference is not None:
            return self.neg_reference
        ref = self.reference or "all_perturbed"
        if negative_source == "all_perturbed" and ref == "all_perturbed":
            return "control"
        return ref

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
    #: ``(raws: dict, protocol: Protocol) -> float | MetricOutcome`` — combines raw control values
    #: into one per-perturbation score; structured outcomes preserve undefined reasons.
    per_pert: Callable
    #: ``(values: numpy.ndarray) -> dict`` — reduces per-perturbation scores into summary statistics (e.g. ``{"mean": …, "median": …}``).
    aggregate: Callable
    #: Human-readable description shown by ``scperteval list calibrators``.
    description: str = ""
    #: Optional ``(protocol: Protocol) -> None`` pre-run check (e.g. requiring ``better``/``perfect``).
    validate: Callable[[Protocol], None] | None = None
