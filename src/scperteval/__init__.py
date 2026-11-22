"""Evaluation Protocols for Perturbation Studies."""

import os as _os
from typing import TYPE_CHECKING

# Pin BLAS/OMP threads BEFORE anything imports numpy/torch — this must stay above the API
# re-export below, which (lazily) pulls in the heavy numeric stack.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_v, "1")

if TYPE_CHECKING:  # let type checkers and IDEs see the lazily re-exported names (no runtime import)
    from .api import (
        DatasetDEResults,
        EvalResult,
        Prepared,
        calibrate,
        de,
        prepare,
        score,
        truth_availability,
    )
    from .blocks import spaces
    from .protocols import metrics
    from .protocols.table import PROTOCOLS
    from .types import MetricOutcome, Protocol

    __version__: str

_API_EXPORTS = frozenset(
    {
        "DatasetDEResults",
        "EvalResult",
        "Prepared",
        "calibrate",
        "de",
        "prepare",
        "score",
        "truth_availability",
    }
)

#: The public Python API, lazily re-exported at the package root.
__all__ = [
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
]


def __getattr__(name: str):
    """Lazily resolve public names so ``import scperteval`` stays cheap.

    The API (and its numeric dependencies: numpy/torch/geomloss/sklearn) load only on first
    access to a public symbol; ``scperteval.__version__`` never triggers that import.
    """
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("scperteval")
        except PackageNotFoundError:  # running from a source tree without an installed dist
            return "0.0.0+unknown"
    if name in _API_EXPORTS:
        from . import api

        return getattr(api, name)
    if name in ("Protocol", "MetricOutcome"):
        from . import types

        return getattr(types, name)
    if name == "spaces":
        from .blocks import spaces

        return spaces
    if name == "metrics":
        from .protocols import metrics

        return metrics
    if name == "PROTOCOLS":
        from .protocols.table import PROTOCOLS

        return PROTOCOLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
