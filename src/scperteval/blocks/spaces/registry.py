"""The feature-space registry: the catalog of space definitions, and the instances built from them.

Two levels:

- A **definition** is one entry in the catalog — a rule, plus what it takes and how to
  describe it. Each is declared by decorating a rule in ``catalog.py``, and
  ``scperteval list spaces`` lists them.
- An **instance** is a definition at one parameter value, registered under a concrete name
  (``"heg_1000"``). A protocol names its space as a string, so an instance must exist before a
  run can resolve it. :meth:`SpaceRegistry.instance` creates one when a protocol or a ``Param``
  asks for it.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...caching import _once
from ...dataset import to_dense
from ...registry import Registry

_UNSET = object()


class UndefinedSpace(ValueError):
    """A feature space could not be constructed for one evaluation unit (e.g. empty DEG panel)."""

    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason


@dataclass(frozen=True)
class SetOps:
    """The set operations :func:`combine_subsets` folds with, named as Python's own set methods.

    Exposed as the singleton ``OPS``, so composing a space reads
    ``combine_subsets(ctx, OPS.union, ...)``.
    """

    #: Genes in either selection.
    union: Callable = np.union1d
    #: Genes in both selections.
    intersection: Callable = np.intersect1d
    #: Genes in the first selection but not the rest.
    difference: Callable = np.setdiff1d


OPS = SetOps()

_OP_NAMES = {OPS.union: "union", OPS.intersection: "intersection", OPS.difference: "difference"}
"""The set operations available to :func:`combine_subsets` — ``OPS.union``, ``OPS.intersection``,
``OPS.difference``."""


def _signature_of(rule: Callable, lead: int) -> tuple[bool, str | None, tuple[tuple[str, Any], ...]]:
    """What a rule asks for: whether it wants ``pert``, its primary parameter, and keyword-only options.

    Both are read off the signature, so neither can be declared wrong. ``lead`` counts the fixed
    leading arguments — 1 for a selection rule ``(ctx, …)``, 2 for a transform ``(X, ctx, …)``.

    A rule's positional parameters must take one of four shapes. Naming ``pert`` is how it asks
    for the perturbation, and *is* the declaration that its result varies by perturbation::

        (ctx)                (ctx, k)                dataset-wide
        (ctx, pert)          (ctx, pert, k)          per-perturbation

    Additional optional parameters may be declared as keyword-only with defaults (used by Python
    composable spaces while keeping single-parameter CLI specs unambiguous).
    """
    all_params = list(inspect.signature(rule).parameters.values())
    pos_params = [
        p for p in all_params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    kw_params = [p for p in all_params if p.kind is inspect.Parameter.KEYWORD_ONLY]
    params = [p.name for p in pos_params]
    expected_lead = ["ctx"] if lead == 1 else ["X", "ctx"]
    if params[:lead] != expected_lead:
        raise TypeError(
            f"rule {rule.__name__}({', '.join(params)}) must start with "
            f"({', '.join(expected_lead)}, …). A `@cached` helper decorated as a space will look "
            f"like this: `@cached` belongs on the helper the rule calls, not on the rule."
        )
    tail = params[lead:]
    takes_pert = bool(tail) and tail[0] == "pert"
    rest = tail[1:] if takes_pert else tail
    if len(rest) > 1 or "pert" in rest:
        shape = f"({', '.join(params)})"
        lead_args = "ctx" if lead == 1 else "X, ctx"
        raise TypeError(
            f"rule {rule.__name__}{shape} does not match a space rule's shape. Expected one of "
            f"({lead_args}), ({lead_args}, <param>), ({lead_args}, pert), or "
            f"({lead_args}, pert, <param>) — pert comes first when present."
        )
    kw_defaults: list[tuple[str, Any]] = []
    for p in kw_params:
        if p.name in ("pert", "ctx", "X"):
            raise TypeError(f"rule {rule.__name__}: {p.name!r} cannot be a keyword-only parameter")
        if p.default is inspect.Parameter.empty:
            raise TypeError(f"rule {rule.__name__}: keyword-only parameter {p.name!r} must have a default")
        kw_defaults.append((p.name, p.default))
    return takes_pert, (rest[0] if rest else None), tuple(kw_defaults)


@dataclass(frozen=True)
class Space:
    """One entry in the space catalog — what a decorated rule becomes."""

    #: Catalog name (``"heg"``). Instances are ``"<name>_<value>"``, or ``"<name>"`` unparameterised.
    name: str
    #: The rule: ``(ctx, …)`` for a subset, ``(X, ctx, …)`` for a transform.
    rule: Callable
    #: The rule's primary parameter name (``"k"``), read from its signature; ``None`` if it takes none.
    parameter: str | None
    #: Parameter value used when a caller doesn't supply one; ``None`` iff unparameterised.
    default: Any
    #: Human-readable, with ``{v}`` standing in for the parameter.
    description: str
    #: Whether the rule is passed ``pert`` — it named the argument, so it must be given one.
    takes_pert: bool = False
    #: Whether the selection actually varies by perturbation, which decides whether the reference
    #: projection can be computed once and shared. Derived, never declared: from the signature for
    #: a decorated rule, from the operands for a composed one.
    per_pert: bool = False
    #: ``False`` for a space that replaces the gene axis rather than narrowing it.
    is_subset: bool = True
    #: Optional ``precompute(ctx, value)`` run during :meth:`~scperteval.context.Context.warm`,
    #: before the per-perturbation loop, so heavy setup happens while the machine is idle rather
    #: than inside the loop (transforms only). Purely an optimisation: the rule must stay correct
    #: if it never runs.
    precompute: Callable | None = None
    #: Optional keyword-only parameters and their defaults, configurable from Python.
    kw_defaults: tuple[tuple[str, Any], ...] = ()
    #: Optional ``(X, ctx, subset, **params) -> ndarray`` hook for transforms that fit natively on a
    #: dataset-wide :class:`SubsetSpace` (e.g. ``pca(50).on(hvg(2000))``).
    subset_apply: Callable | None = None
    #: Optional ``(ctx, subset, **params) -> None`` warmup hook paired with ``subset_apply``.
    subset_precompute: Callable | None = None

    @property
    def label(self) -> str:
        """How the space is written in listings and docs — ``"heg_<k>"`` or ``"full"``."""
        return f"{self.name}_<{self.parameter}>" if self.parameter else self.name

    def describe(self, value=None) -> str:
        """The description with ``{v}`` filled in — by ``value``, or by the parameter's name."""
        return self.description.format(v=f"{value:g}" if value is not None else self.parameter or "")


class FeatureSpace:
    """Base type for composable feature spaces accepted by :attr:`~scperteval.types.Protocol.space`."""

    @property
    def cache_key(self) -> Hashable:
        """Hashable identity for dataset-scoped caches."""
        raise NotImplementedError

    @property
    def is_subset(self) -> bool:
        """Whether this space selects a subset of genes rather than transforming the axis."""
        raise NotImplementedError

    @property
    def per_pert(self) -> bool:
        """Whether the selected features or transform vary by perturbation."""
        raise NotImplementedError

    def apply(self, X, ctx, pert: str | None) -> np.ndarray:
        """Project ``X`` (cells × genes) into this feature space."""
        raise NotImplementedError

    def precompute(self, ctx) -> None:
        """Optional warmup hook run during :meth:`~scperteval.context.Context.warm`."""


class SubsetSpace(FeatureSpace):
    """A composable gene-subset space supporting ``|`` (union), ``&`` (intersection), and ``-`` (difference)."""

    @property
    def is_subset(self) -> bool:
        """Gene-subset spaces narrow the gene axis."""
        return True

    def _select_raw(self, ctx, pert: str | None) -> np.ndarray | slice:
        raise NotImplementedError

    def select(self, ctx, pert: str | None) -> np.ndarray | slice:
        """Return integer positions (or a slice) into the full gene axis, rejecting empty panels."""
        keep = self._select_raw(ctx, pert)
        if isinstance(keep, np.ndarray) and keep.size == 0:
            raise UndefinedSpace(
                "empty_gene_space",
                f"space {self!r} selected no genes for {pert!r}.",
            )
        return keep

    def apply(self, X, ctx, pert: str | None) -> np.ndarray:
        """Slice ``X`` to the selected genes and return a dense array."""
        return to_dense(X[:, self.select(ctx, pert)])

    def __or__(self, other: SubsetSpace) -> CombinedSubset:
        if not isinstance(other, SubsetSpace):
            return NotImplemented
        return CombinedSubset(OPS.union, self, other)

    def __and__(self, other: SubsetSpace) -> CombinedSubset:
        if not isinstance(other, SubsetSpace):
            return NotImplemented
        return CombinedSubset(OPS.intersection, self, other)

    def __sub__(self, other: SubsetSpace) -> CombinedSubset:
        if not isinstance(other, SubsetSpace):
            return NotImplemented
        return CombinedSubset(OPS.difference, self, other)


class TransformSpace(FeatureSpace):
    """A feature space that replaces the gene axis (e.g. PCA), optionally restricted via :meth:`on`."""

    @property
    def is_subset(self) -> bool:
        """Transforms replace the gene axis rather than subsetting it."""
        return False

    def on(self, subset: SubsetSpace) -> PipedTransform:
        """Apply (and, where supported, fit) this transform restricted to a dataset-wide ``subset``."""
        if not isinstance(subset, SubsetSpace):
            raise TypeError(f"on() expects a SubsetSpace, got {type(subset).__name__}")
        return PipedTransform(subset, self)


@dataclass(frozen=True)
class RuleSubset(SubsetSpace):
    """A subset space instantiated from a catalog rule without mutating the global registry."""

    name: str
    rule: Callable
    params: tuple[tuple[str, Any], ...]
    takes_pert: bool
    _per_pert: bool

    @property
    def per_pert(self) -> bool:
        """Whether this subset varies by perturbation."""
        return self._per_pert

    @property
    def cache_key(self) -> Hashable:
        """Structural hashable identity preserving exact parameter values."""
        return ("subset", self.name, self.params)

    def _select_raw(self, ctx, pert: str | None) -> np.ndarray | slice:
        kwargs = dict(self.params)

        def call():
            return self.rule(ctx, **({"pert": pert} if self.takes_pert else {}), **kwargs)

        if self._per_pert:
            return call()
        return _once(ctx._store, ("select", self.cache_key), call)


@dataclass(frozen=True)
class CombinedSubset(SubsetSpace):
    """Set-algebra composition (union, intersection, difference) of two :class:`SubsetSpace` objects."""

    op: Callable
    left: SubsetSpace
    right: SubsetSpace

    @property
    def per_pert(self) -> bool:
        """Derived from operands: varies by perturbation iff either operand does."""
        return self.left.per_pert or self.right.per_pert

    @property
    def cache_key(self) -> Hashable:
        """Structural hashable identity for the composed expression."""
        return ("combine", _OP_NAMES[self.op], self.left.cache_key, self.right.cache_key)

    def _select_raw(self, ctx, pert: str | None) -> np.ndarray:
        def call() -> np.ndarray:
            genes = np.arange(len(ctx.ds.var_names))
            lhs = genes[self.left._select_raw(ctx, pert)]
            rhs = genes[self.right._select_raw(ctx, pert)]
            return np.asarray(self.op(lhs, rhs), dtype=int)

        if self.per_pert:
            return call()
        return _once(ctx._store, ("select", self.cache_key), call)


@dataclass(frozen=True)
class RuleTransform(TransformSpace):
    """A transform space instantiated from a catalog rule without mutating the global registry."""

    name: str
    rule: Callable
    params: tuple[tuple[str, Any], ...]
    takes_pert: bool
    _per_pert: bool
    primary_value: Any = None
    precompute_fn: Callable | None = None
    subset_apply: Callable | None = None
    subset_precompute: Callable | None = None

    @property
    def per_pert(self) -> bool:
        """Whether this transform varies by perturbation."""
        return self._per_pert

    @property
    def cache_key(self) -> Hashable:
        """Structural hashable identity preserving exact parameter values."""
        return ("transform", self.name, self.params)

    def precompute(self, ctx) -> None:
        """Run the catalog transform's warmup hook if declared."""
        if self.precompute_fn is not None:
            self.precompute_fn(ctx, self.primary_value)

    def apply(self, X, ctx, pert: str | None) -> np.ndarray:
        """Apply the transform rule to ``X``."""
        kwargs = dict(self.params)
        return self.rule(X, ctx, **({"pert": pert} if self.takes_pert else {}), **kwargs)


@dataclass(frozen=True)
class PipedTransform(TransformSpace):
    """A dataset-wide :class:`SubsetSpace` paired with a :class:`TransformSpace` (``transform.on(subset)``)."""

    subset: SubsetSpace
    transform: TransformSpace

    def __post_init__(self) -> None:
        if self.subset.per_pert:
            raise ValueError("subset-then-transform requires a dataset-wide subset space")

    @property
    def per_pert(self) -> bool:
        """Derived from the downstream transform."""
        return self.transform.per_pert

    @property
    def cache_key(self) -> Hashable:
        """Structural hashable identity combining the subset and transform."""
        return ("pipe", self.subset.cache_key, self.transform.cache_key)

    def precompute(self, ctx) -> None:
        """Warm the piped transform on the restricted gene subset."""
        if isinstance(self.transform, RuleTransform) and self.transform.subset_precompute is not None:
            self.transform.subset_precompute(ctx, self.subset, **dict(self.transform.params))
            return
        self.transform.precompute(ctx)

    def apply(self, X, ctx, pert: str | None) -> np.ndarray:
        """Apply the transform restricted to ``subset``."""
        if isinstance(self.transform, RuleTransform) and self.transform.subset_apply is not None:
            kwargs = dict(self.transform.params)
            if self.transform.takes_pert:
                kwargs["pert"] = pert
            return self.transform.subset_apply(X, ctx, self.subset, **kwargs)
        keep = self.subset.select(ctx, pert)
        return self.transform.apply(X[:, keep], ctx, pert)


@dataclass(frozen=True)
class _RegisteredSubset(SubsetSpace):
    """Thin adapter for a named subset space looked up in :data:`SPACES`."""

    name: str
    _select: Callable
    _per_pert: bool

    @property
    def per_pert(self) -> bool:
        return self._per_pert

    @property
    def cache_key(self) -> Hashable:
        return self.name

    def _select_raw(self, ctx, pert: str | None) -> np.ndarray | slice:
        return self._select(ctx, pert)


@dataclass(frozen=True)
class _RegisteredTransform(TransformSpace):
    """Thin adapter for a named transform space looked up in :data:`SPACES`."""

    name: str
    _apply: Callable
    _per_pert: bool
    _precompute: Callable | None = None
    _value: Any = None

    @property
    def per_pert(self) -> bool:
        return self._per_pert

    @property
    def cache_key(self) -> Hashable:
        return self.name

    def precompute(self, ctx) -> None:
        if self._precompute is not None:
            self._precompute(ctx, self._value)

    def apply(self, X, ctx, pert: str | None) -> np.ndarray:
        return self._apply(X, ctx, pert)


class SpaceRegistry(Registry):
    """A :class:`~scperteval.registry.Registry` that also holds the catalog spaces are built from.

    Add a space by decorating its rule with :meth:`SpaceRegistry.subset` or
    :meth:`SpaceRegistry.transform`, the same way ``DE_METHODS`` and ``SOURCES`` are extended.
    :meth:`SpaceRegistry.instance` then builds a named instance from a definition on demand. The
    inherited ``__getitem__`` / ``meta`` / ``names`` see instances only, never definitions.
    """

    def __init__(self, kind: str):
        super().__init__(kind)
        self._catalog: dict[str, Space] = {}
        # Definitions are only ever added at import time (single-threaded); instance() is the one
        # place this registry is mutated at runtime, so it's the one place that needs a lock.
        self._lock = threading.Lock()

    # -- defining ------------------------------------------------------------------

    def register(self, name: str, **meta):
        """Not the way to define a space — use :meth:`subset` or :meth:`transform`."""
        raise TypeError(
            f"a {self.kind} is defined with @SPACES.subset or @SPACES.transform, not "
            f"@SPACES.register. One registered the inherited way scores correctly but never "
            f"appears in `scperteval list spaces`, because the listing reads the catalog."
        )

    def add(self, name: str, fn, **meta):
        """Not the way to define a space — use :meth:`subset` or :meth:`transform`."""
        raise TypeError(
            f"a {self.kind} is defined with @SPACES.subset or @SPACES.transform, then "
            f"instantiated with SPACES.instance(...); adding one directly skips the catalog."
        )

    def subset(self, name: str, *, default=None, description="") -> Callable:
        """Decorator: define a space that keeps a subset of the genes.

        The rule returns a column selection into the full gene axis — an integer array, or a
        slice. Name a parameter ``pert`` to be given the perturbation, which is also how the space
        declares that its genes vary by perturbation; omit it and the space is dataset-wide. Give
        the trailing argument a default to declare that the space takes no parameter. Arguments
        are passed by keyword, so their order doesn't matter::

            def heg(ctx, k): ...  # dataset-wide, takes k
            def top(ctx, pert, k): ...  # per-perturbation, takes k
            def full(ctx, value=None): ...  # dataset-wide, no parameter

        Parameters
        ----------
        name : str
            Catalog name; instances are ``"<name>_<value>"``.
        default : optional
            Parameter value used when a caller doesn't supply one. Required if the rule takes a
            parameter, and must be omitted if it doesn't.
        description : str
            Shown by ``scperteval list spaces``; ``{v}`` stands in for the parameter.
        """

        def deco(rule: Callable) -> Callable:
            takes_pert, parameter, kw_defaults = _signature_of(rule, 1)
            self._define(
                Space(name, rule, parameter, default, description, takes_pert, takes_pert, kw_defaults=kw_defaults)
            )
            return rule

        return deco

    def transform(
        self,
        name: str,
        *,
        default=None,
        description="",
        precompute=None,
        subset_apply=None,
        subset_precompute=None,
    ) -> Callable:
        """Decorator: define a space that replaces the gene axis instead of narrowing it.

        The rule is ``(X, ctx, pert, value) -> dense cells × features array``, built directly, so
        the space has no gene selection and cannot be composed. ``precompute(ctx, value)``
        optionally runs the space's heavy setup during
        :meth:`~scperteval.context.Context.warm`, before the per-perturbation loop; it is purely
        an optimisation, so the rule must stay correct without it.
        """

        def deco(rule: Callable) -> Callable:
            takes_pert, parameter, kw_defaults = _signature_of(rule, 2)
            self._define(
                Space(
                    name,
                    rule,
                    parameter,
                    default,
                    description,
                    takes_pert,
                    takes_pert,
                    False,
                    precompute,
                    kw_defaults=kw_defaults,
                    subset_apply=subset_apply,
                    subset_precompute=subset_precompute,
                )
            )
            return rule

        return deco

    def _define(self, space: Space) -> None:
        if space.name in self._catalog:
            raise ValueError(
                f"{self.kind} {space.name!r} is already defined. Names are unique: redefining one "
                f"would leave instances already registered from the old definition in place, "
                f"scoring on genes the catalog no longer describes."
            )
        if (space.parameter is None) != (space.default is None):
            raise TypeError(
                f"{self.kind} {space.name!r}: a rule taking a parameter needs a default and one "
                f"taking none must not have one (parameter={space.parameter!r}, default={space.default!r})"
            )
        self._catalog[space.name] = space

    def catalog(self) -> list[Space]:
        """Every defined space, by name — the palette, not the registered instances."""
        return [self._catalog[n] for n in sorted(self._catalog)]

    # -- instantiating -------------------------------------------------------------

    def combine_subsets(self, op: Callable, *names: str, name: str, description: str | None = None) -> str:
        """Register the union, intersection, or difference of already-registered subset spaces.

        ``per_pert`` is derived from the operands — the result varies by perturbation if any
        operand does — so a composed space cannot claim to be dataset-wide while computing
        something that isn't. That mistake is invisible at runtime: the reference projection would
        be built once from one perturbation's genes and reused for all of them, producing
        plausible scores that are simply wrong.

        Parameters
        ----------
        op : Callable
            One of ``OPS`` — ``OPS.union``, ``OPS.intersection``, or ``OPS.difference``,
            applied left to right, so ``OPS.difference`` subtracts the rest from the first.
        *names : str
            Two or more registered gene-subset instance names (e.g. ``"hvg_8192"``), in the order
            ``op`` should apply them. A transform (``"pca_50"``) has no genes and is rejected.
        name : str
            What to register the result under. Required rather than derived: joining operator
            symbols made ``(a-b)+c`` and ``a-(b+c)`` collide, and silently aliasing onto whichever
            registered first is worse than asking.
        description : str, optional
            Shown by ``scperteval list spaces``. Defaults to naming the operation and operands.

        Returns
        -------
        str
            ``name``, for symmetry with :meth:`instance`.

        Notes
        -----
        The operands' selections are read once, here; re-registering an operand afterwards does
        not change an already-composed space.

        Examples
        --------
        The HVG ∪ perturbed-genes panel of :cite:t:`Miller_2025`::

            SPACES.combine_subsets(OPS.union, "hvg_8192", "perturbed_genes", name="perturbed_and_hvgs")
        """
        if len(names) < 2:
            raise ValueError(f"combine_subsets needs at least two spaces to combine, got {len(names)}")
        if op not in _OP_NAMES:
            raise ValueError(f"unknown op {op!r}; expected one of OPS: {sorted(_OP_NAMES.values())}")
        unknown = [n for n in names if n not in self]
        if unknown:
            raise KeyError(f"unknown {self.kind}(s) {unknown}; available: {self.names()}")
        not_subsets = [n for n in names if "select" not in self.meta(n)]
        if not_subsets:
            raise ValueError(f"not gene subsets, so they have no genes to combine: {not_subsets}")
        selects = [self.meta(n)["select"] for n in names]
        per_pert = any(not self.meta(n)["global_space"] for n in names)

        def rule(ctx, pert):
            # Canonicalise each selection to integer positions -- a rule may return a slice, and
            # the set operations need real indices. Every rule indexes the same full gene axis.
            genes = np.arange(len(ctx.ds.var_names))
            result = genes[selects[0](ctx, pert)]
            for select in selects[1:]:
                result = op(result, genes[select(ctx, pert)])
            return result

        label = _OP_NAMES.get(op, getattr(op, "__name__", "combination"))
        self._define(Space(name, rule, None, None, description or f"{label} of {', '.join(names)}", True, per_pert))
        return self.instance(name)

    def instance(self, name: str, value=None) -> str:
        """Register one variant of a defined space and return its name, e.g. ``"heg_250"``.

        Idempotent: a variant already registered at the same value is reused. Omit ``value`` for
        the space's default. Thread-safe: two concurrent calls registering the same not-yet-seen
        value each build it at most once, guarded by a lock scoped to this registry alone.

        Registers the rule; nothing is computed until the space is applied, and that computation
        is cached (see :func:`~scperteval.caching.cached`). Call this at import time — from a
        module body, next to the protocol row or composition that needs the space — as the
        built-ins and the guides do. A run itself never registers from a worker: protocols
        resolve and :meth:`~scperteval.context.Context.warm` completes before the scoring pool
        opens, after which workers only read.
        """
        if name not in self._catalog:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._catalog)}")
        space = self._catalog[name]
        if space.parameter is None:
            if value is not None:
                raise TypeError(f"{self.kind} {name!r} takes no parameter, got {value!r}")
            key = name
        else:
            value = space.default if value is None else value
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"{self.kind} {name!r} takes a positive number, got {value!r}. Negative values "
                    f"do not mean the same thing across spaces -- top/heg/hvg would drop the "
                    f"strongest genes, degs would select none, pca would truncate its components."
                )
            key = f"{name}_{value:g}"

        def guard_existing():
            # Distinct values can format to the same name (0.05 and 0.05000001 are both "0.05").
            # Registering the second silently under the first's rule would score the wrong genes.
            registered = self.meta(key).get("value")
            if registered != value:
                raise ValueError(f"{key!r} is already registered with value {registered!r}, not {value!r}")

        if key in self:
            guard_existing()
            return key
        with self._lock:
            if key in self:  # another thread may have registered it while this one waited for the lock
                guard_existing()
                return key
            common = dict(description=space.describe(value), value=value)
            if space.is_subset:
                select = _bound_select(space, value, key)

                def apply(X, ctx, pert):
                    keep = select(ctx, pert)
                    if isinstance(keep, np.ndarray) and keep.size == 0:
                        raise UndefinedSpace(
                            "empty_gene_space",
                            f"space {key!r} selected no genes for {pert!r}. Every metric would return "
                            f"nan rather than fail, so this is refused: widen the space, or check that "
                            f"its criterion matches the dataset.",
                        )
                    return to_dense(X[:, keep])

                super().add(key, apply, select=select, global_space=not space.per_pert, **common)
            else:
                super().add(
                    key,
                    _bound_transform(space, value),
                    global_space=not space.per_pert,
                    precompute=space.precompute,
                    **common,
                )
        return key

    def space(self, name: str, value=_UNSET, **kwargs) -> FeatureSpace:
        """Build a composable :class:`FeatureSpace` value object from a catalog definition.

        Unlike :meth:`instance`, this does not register anything globally or mutate the catalog.
        """
        if name not in self._catalog:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._catalog)}")
        defn = self._catalog[name]
        allowed_kw = dict(defn.kw_defaults)
        unknown = sorted(set(kwargs) - set(allowed_kw))
        if unknown:
            raise TypeError(f"{self.kind} {name!r} got unexpected keyword argument(s): {', '.join(unknown)}")
        merged: dict[str, Any] = dict(allowed_kw)
        merged.update(kwargs)

        primary_value: Any = None
        if defn.parameter is None:
            if value is not _UNSET:
                raise TypeError(f"{self.kind} {name!r} takes no parameter, got {value!r}")
        else:
            primary_value = defn.default if value is _UNSET else value
            if primary_value is not None and (
                not isinstance(primary_value, (int, float)) or isinstance(primary_value, bool) or primary_value <= 0
            ):
                raise ValueError(f"{self.kind} {name!r} takes a positive number, got {primary_value!r}")
            merged[defn.parameter] = primary_value

        params = tuple(sorted(merged.items()))
        if defn.is_subset:
            return RuleSubset(defn.name, defn.rule, params, defn.takes_pert, defn.per_pert)
        return RuleTransform(
            defn.name,
            defn.rule,
            params,
            defn.takes_pert,
            defn.per_pert,
            primary_value=primary_value,
            precompute_fn=defn.precompute,
            subset_apply=defn.subset_apply,
            subset_precompute=defn.subset_precompute,
        )


def resolve_space(space: str | FeatureSpace) -> FeatureSpace:
    """Resolve a registered space name or composable :class:`FeatureSpace` to a :class:`FeatureSpace`."""
    if isinstance(space, FeatureSpace):
        return space
    if isinstance(space, str):
        fn = SPACES[space]
        meta = SPACES.meta(space)
        per_pert = not meta.get("global_space", True)
        if "select" in meta:
            return _RegisteredSubset(space, meta["select"], per_pert)
        return _RegisteredTransform(space, fn, per_pert, meta.get("precompute"), meta.get("value"))
    raise TypeError(f"space must be a registered name (str) or FeatureSpace, not {type(space).__name__}")


def _bind(space: Space, value):
    """The rule's arguments after ``ctx``, by keyword."""
    kwargs = {}
    if space.parameter is not None:
        kwargs[space.parameter] = value
    return kwargs


def _bound_select(space: Space, value, key: str):
    """``select(ctx, pert)`` for a subset rule, with its parameter bound.

    Runs the rule on every call -- once per perturbation per candidate per protocol. For a
    dataset-wide space (``not space.per_pert``) the answer is identical every time by
    construction (the rule never received ``pert``, so it cannot depend on it), so the ranking on
    top of the cached statistic (an argsort over all genes, ~0.7 ms at 20k genes) is computed once
    per prepared dataset and reused, the same way :func:`~scperteval.context.Context.reference`
    already trusts a global space's result to be reusable across every perturbation.
    """
    kwargs = _bind(space, value)

    def call(ctx, pert):
        return space.rule(ctx, **({"pert": pert} if space.takes_pert else {}), **kwargs)

    if space.per_pert:
        return call

    def select(ctx, pert):
        return _once(ctx._store, ("select", key), lambda: call(ctx, pert))

    return select


def _bound_transform(space: Space, value):
    """``apply(X, ctx, pert)`` for a transform rule, with its parameter bound."""
    kwargs = _bind(space, value)

    def apply(X, ctx, pert):
        return space.rule(X, ctx, **({"pert": pert} if space.takes_pert else {}), **kwargs)

    return apply


SPACES = SpaceRegistry("space")
"""The feature-space registry: the catalog, and the instances registered from it."""
