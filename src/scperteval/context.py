"""The per-call engine that turns a (perturbation, source) into a protocol's view.

A fresh :class:`Context` is built for each verb call, carrying that call's options; the expensive
building blocks (DE, reference sample, PCA/space projections) live in a shared, thread-safe
:class:`CacheStore` so many calls against one prepared dataset reuse them safely.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .blocks.de import DE_METHODS, _moments
from .blocks.spaces import SubsetSpace, resolve_space
from .caching import DatasetScope, _once, cached
from .dataset import Dataset, to_dense
from .reference import Reference
from .sources import SOURCES
from .types import PerturbationDEResult, Protocol, RunConfig

if TYPE_CHECKING:
    from .predictions import PredictionSet


class CacheStore:
    """The shared, thread-safe memoization for one prepared dataset.

    Held by a :class:`~scperteval.api.Prepared` handle and shared by reference across the
    ephemeral per-call :class:`Context` objects the verbs build. All entries are dataset-level
    or per-(perturbation, comparison) — none depend on a single call's ``calibrator``/``truth``, so
    concurrent calls reuse them safely (distinct keys, or idempotent recompute under ``lock``).
    """

    def __init__(self):
        # Reentrant because cache bookkeeping can be nested even though computations run outside the lock.
        self.lock = threading.RLock()
        self.memo: dict = {}  # (function, params) -> value, for every @cached/_once computation
        self.inflight: dict = {}  # key -> Future for computations currently owned by another thread
        self.scope: DatasetScope | None = None  # the settings every value here was computed under


@cached
def _control_mean(scope: DatasetScope) -> np.ndarray:
    return scope.ds.control_mean()


@cached
def _control_cells(scope: DatasetScope):
    """Materialize the immutable subsampled control reference once per prepared dataset."""
    return scope.ds.control_cells(scope.subsample)


@cached
def _reference(scope: DatasetScope) -> Reference:
    idx = scope.ds.all_perturbed_indices(scope.subsample)
    cells = to_dense(scope.ds.adata.X[idx]).astype(np.float64)
    return Reference(cells, scope.ds.pert[idx])


class Context:
    """Per-call engine: turns a (perturbation, source) into a protocol's view.

    A fresh, cheap ``Context`` is built for each verb call, carrying that call's ``cfg`` (the
    per-call options: DE method, ground-truth source, calibrator, controls) and its own
    ``predictions``/``current_pert``. The expensive caches live in a shared :class:`CacheStore`
    (``store``) so many calls against one prepared dataset reuse them without recomputation or
    cross-call mutation. ``current_pert`` is thread-local so the runner can fan out across threads.

    Parameters
    ----------
    dataset : ~scperteval.dataset.Dataset
        Loaded and indexed AnnData wrapper.
    cfg : ~scperteval.types.RunConfig
        Resolved options for this call (DE method, subsample size, seed, …).
    store : CacheStore, optional
        Shared cache to reuse (from a prepared handle). A fresh one is created if omitted.
    """

    def __init__(
        self,
        dataset: Dataset,
        cfg: RunConfig,
        store: CacheStore | None = None,
        user_sources: dict[str, tuple[Callable, dict]] | None = None,
    ):
        self.ds = dataset
        self.cfg = cfg
        self.predictions: PredictionSet | None = None  # set in prediction-scoring mode
        self.candidates: dict[str, str] = {}  # the running protocol's resolved controls (set by the runner)
        # Per-handle runtime user sources ({name: (callable, meta)}); read-only, resolved ahead of
        # the global SOURCES registry so different handles/vectors never leak across each other.
        self._user_sources = user_sources or {}
        self._local = threading.local()
        self._store = store if store is not None else CacheStore()

    @property
    def perturbations(self):
        """The list of perturbations evaluated in this run."""
        return self.ds.perturbations

    @property
    def current_pert(self):
        """The perturbation the current worker thread is processing."""
        return getattr(self._local, "pert", None)

    @current_pert.setter
    def current_pert(self, value):
        self._local.pert = value

    @property
    def current_space(self):
        """The feature space of the protocol the current worker thread is evaluating."""
        return getattr(self._local, "space", None)

    @current_space.setter
    def current_space(self, value):
        self._local.space = value

    # -- source resolution: per-handle user sources first, else the global registry ----

    def source(self, name: str) -> Callable:
        """The callable ``(ctx, pert) -> cells|centroid`` for ``name`` (user source or built-in)."""
        us = self._user_sources.get(name)
        return us[0] if us is not None else SOURCES[name]

    def source_meta(self, name: str) -> dict:
        """Metadata (``provides``/``cacheable``) for ``name`` (user source or built-in)."""
        us = self._user_sources.get(name)
        return us[1] if us is not None else SOURCES.meta(name)

    def has_source(self, name: str) -> bool:
        """Whether ``name`` resolves to a user source or a built-in."""
        return name in self._user_sources or name in SOURCES

    def source_names(self) -> list[str]:
        """Sorted names of every resolvable source (user sources + built-ins), for error messages."""
        return sorted(set(SOURCES.names()) | set(self._user_sources))

    def warm(self, protocols):
        """Precompute shared singletons before the parallel loop.

        So per-perturbation threads only ever write per-perturbation cache keys.
        """
        self.control_mean()
        if any(p.representation in ("population", "de") for p in protocols):
            self.reference()
        if any(p.representation == "de" for p in protocols):
            self._ensure_reference_sums()
            self._moments("control", None)
        resolved_spaces = [resolve_space(p.space) for p in protocols]
        for sp in resolved_spaces:
            sp.precompute(self)
        seen_keys = set()
        for p, sp in zip(protocols, resolved_spaces):
            if p.representation == "population" and not sp.per_pert and sp.cache_key not in seen_keys:
                seen_keys.add(sp.cache_key)
                self.reference_projection(sp)

    def view(self, pert: str, source: str, p: Protocol):
        """Return ``source``'s datapoint for ``pert`` in the shape ``p`` consumes."""
        if p.representation == "population":
            if source == "all_perturbed":
                return self._reference_population(p.space, pert)
            return resolve_space(p.space).apply(self.source(source)(self, pert), self, pert)
        if p.representation == "centroid":
            v = self.centroid(pert, source, p.centering)
            return resolve_space(p.space).apply(v[None, :], self, pert).ravel()
        if p.representation == "de":
            return self._de_view(pert, source, p)
        raise ValueError(f"unknown protocol representation {p.representation!r}")

    def centroid(self, pert, source, centering):
        """Pseudobulk centroid of ``source`` for ``pert``, optionally centered."""
        arr = self.source(source)(self, pert)
        if self.source_meta(source).get("provides") == "centroid":
            v = np.asarray(arr, dtype=np.float64).ravel()
        else:
            v = np.asarray(to_dense(arr), dtype=np.float64).mean(0)
        if centering is not None:  # subtract a named centroid source; None = no centering
            v = v - self._centering_vector(centering, pert)
        return v

    def _centering_vector(self, name, pert):
        """Centroid of a named centering source for ``pert`` (the ``center_on`` baseline)."""
        if self.source_meta(name).get("provides") != "centroid":
            raise ValueError(
                f"centering source {name!r} provides {self.source_meta(name).get('provides')!r}, "
                f"but a centering baseline must be a centroid (1-D)"
            )
        return np.asarray(self.source(name)(self, pert), dtype=np.float64).ravel()

    def _de_view(self, pert, source, p):
        """Return symmetric signed DE results for truth and candidate, restricted to ``p.space``.

        The negative candidate is tested against ``neg_reference`` (e.g. control) rather than
        ``reference`` (the all-perturbed sample), the hybrid DE setup. When ``p.space`` narrows the
        gene axis (e.g. ``hvg(2000)`` or ``full() - own_target()``), both truth and candidate
        results are sliced to that gene universe without altering biological DE p-values.
        """
        sp = resolve_space(p.space)
        if not isinstance(sp, SubsetSpace):
            raise TypeError(f"DE protocol {p.name!r} requires a gene-subset space, got {p.space!r}")
        is_negative = source == self.candidates.get("negative")
        reference = p.effective_neg_reference(source) if is_negative else (p.reference or "all_perturbed")
        res = self.de(pert, source, reference)
        keep = sp.select(self, pert)
        if isinstance(keep, slice) and keep == slice(None):
            return res
        return PerturbationDEResult(
            statistic=res.statistic[keep],
            pvalue=res.pvalue[keep],
            pvalue_adj=res.pvalue_adj[keep],
            extra=res.extra,
        )

    def de(self, pert, source, reference="all_perturbed"):
        """Differential expression for one (source vs reference) comparison, cached.

        The reference moments are leave-one-out, so a perturbation is never compared against
        a sample of itself. Comparisons involving a non-cacheable source (e.g. per-call
        ``prediction`` cells) are computed fresh and never stored, so a shared handle can score
        different predictions without cross-call contamination.
        """
        method = self.cfg.de_method

        def compute():
            # A method may declare a `from_moments` capability (see DE_METHODS); if it does,
            # dispatch through the shared moment cache instead of recomputing from cells.
            from_moments = DE_METHODS.meta(method).get("from_moments")
            if from_moments is not None:
                return from_moments(*self._moments(source, pert), *self._moments(reference, pert))
            return DE_METHODS[method](self._de_cells(source, pert), self._de_cells(reference, pert))

        if not (self._cacheable(source) and self._cacheable(reference)):
            return compute()
        key = (self._moment_key(source, pert), self._moment_key(reference, pert), method)
        return _once(self._store, ("de", key), compute)

    def _moments(self, source, pert):
        if source == "all_perturbed":
            return self._reference_moments(pert)
        if not self._cacheable(source):  # per-call source (e.g. prediction) — compute fresh, don't cache
            return _moments(self._de_cells(source, pert))
        key = self._moment_key(source, pert)
        return _once(self._store, ("moments", key), lambda: _moments(self._de_cells(source, pert)))

    def _cacheable(self, source):
        """Whether a source's cells are dataset-derived (shareable) rather than per-call (predictions/user sources)."""
        return self.source_meta(source).get("cacheable", True)

    def _de_cells(self, source, pert):
        if source == "all_perturbed":
            return self.reference().subset(pert)
        if source == "control":
            return _control_cells(self)
        return self.source(source)(self, pert)

    @staticmethod
    def _moment_key(source, pert):
        return source if source == "control" else (source, pert)

    # -- the all-perturbed reference: one sample, served leave-one-out -------------

    def reference(self) -> Reference:
        """The all-perturbed sample, subsampled and densified once.

        Each cell's perturbation is recorded so the sample can be served leave-one-out.
        """
        return _reference(self)

    def _reference_population(self, space, pert):
        """The reference in a feature space with the target perturbation removed.

        Project the whole sample (cached for global spaces) then drop its rows.
        """
        ref = self.reference()
        sp = resolve_space(space)
        proj = sp.apply(ref.cells, self, pert) if sp.per_pert else self.reference_projection(sp)
        keep = ref.keep(pert)
        return proj if keep is None else proj[keep]

    def reference_projection(self, space):
        """The reference projected to a perturbation-independent space, cached once."""
        sp = resolve_space(space)
        return _once(
            self._store, ("reference_projection", (sp.cache_key,)), lambda: sp.apply(self.reference().cells, self, None)
        )

    def _ensure_reference_sums(self):
        """Cache the reference's column sums and sums-of-squares once.

        Leave-one-out moments are then an O(target cells) subtraction rather than a
        re-densify per perturbation.
        """

        def compute():
            X = self.reference().cells
            return (X.sum(0), np.einsum("ij,ij->j", X, X), len(X))

        return _once(self._store, ("reference_sums", ()), compute)

    def _reference_moments(self, pert):
        total, totalsq, n = self._ensure_reference_sums()
        keep = self.reference().keep(pert)
        if keep is None:
            s, sq, k = total, totalsq, n
        else:
            Xp = self.reference().cells[~keep]
            s = total - Xp.sum(0)
            sq = totalsq - np.einsum("ij,ij->j", Xp, Xp)
            k = int(keep.sum())
        mean = s / k
        var = np.maximum((sq / k - mean * mean) * (k / max(k - 1, 1)), 0.0)
        return mean, var, k

    def scope(self):
        """The prepare-scoped view a ``@cached`` dataset computation is given.

        Only settings fixed when the handle was prepared — never per-call config, which the
        shared cache outlives.
        """
        from .runner import _n_workers  # local: runner imports this module

        scope = DatasetScope(self.ds, self.cfg.seed, _n_workers(self.cfg), self.cfg.subsample)
        if self._store.scope is None:
            self._store.scope = scope
        else:
            # `threads` is excluded: it changes how fast a value is computed, never what it is.
            was = self._store.scope
            if was.ds is not scope.ds or (was.seed, was.subsample) != (scope.seed, scope.subsample):
                raise ValueError(
                    "this cache holds values computed for a different dataset, seed, or subsample. "
                    "A CacheStore belongs to one prepare() call -- build a new handle rather than "
                    "sharing one across settings."
                )
        return scope

    def control_mean(self):
        """The control centroid (cached). Kept as a method: sources and centering both read it."""
        return _control_mean(self)
