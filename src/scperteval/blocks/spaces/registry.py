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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...caching import _once
from ...dataset import to_dense
from ...registry import Registry


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


def _signature_of(rule: Callable, lead: int) -> tuple[bool, str | None]:
    """What a rule asks for: whether it wants ``pert``, and its parameter's name.

    Both are read off the signature, so neither can be declared wrong. ``lead`` counts the fixed
    leading arguments — 1 for a selection rule ``(ctx, …)``, 2 for a transform ``(X, ctx, …)``.

    A rule must take one of four shapes. Naming ``pert`` is how it asks for the perturbation, and
    *is* the declaration that its result varies by perturbation::

        (ctx)                (ctx, k)                dataset-wide
        (ctx, pert)          (ctx, pert, k)          per-perturbation

    Anything else is rejected here rather than mis-read: ``(ctx, k, pert)`` would otherwise
    register as dataset-wide and score every perturbation on one panel, silently.
    """
    params = [p.name for p in inspect.signature(rule).parameters.values()]
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
    return takes_pert, (rest[0] if rest else None)


@dataclass(frozen=True)
class Space:
    """One entry in the space catalog — what a decorated rule becomes."""

    #: Catalog name (``"heg"``). Instances are ``"<name>_<value>"``, or ``"<name>"`` unparameterised.
    name: str
    #: The rule: ``(ctx, …)`` for a subset, ``(X, ctx, …)`` for a transform.
    rule: Callable
    #: The rule's parameter name (``"k"``), read from its signature; ``None`` if it takes none.
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

    @property
    def label(self) -> str:
        """How the space is written in listings and docs — ``"heg_<k>"`` or ``"full"``."""
        return f"{self.name}_<{self.parameter}>" if self.parameter else self.name

    def describe(self, value=None) -> str:
        """The description with ``{v}`` filled in — by ``value``, or by the parameter's name."""
        return self.description.format(v=f"{value:g}" if value is not None else self.parameter or "")


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
            takes_pert, parameter = _signature_of(rule, 1)
            self._define(Space(name, rule, parameter, default, description, takes_pert, takes_pert))
            return rule

        return deco

    def transform(self, name: str, *, default=None, description="", precompute=None) -> Callable:
        """Decorator: define a space that replaces the gene axis instead of narrowing it.

        The rule is ``(X, ctx, pert, value) -> dense cells × features array``, built directly, so
        the space has no gene selection and cannot be composed. ``precompute(ctx, value)``
        optionally runs the space's heavy setup during
        :meth:`~scperteval.context.Context.warm`, before the per-perturbation loop; it is purely
        an optimisation, so the rule must stay correct without it.
        """

        def deco(rule: Callable) -> Callable:
            takes_pert, parameter = _signature_of(rule, 2)
            self._define(Space(name, rule, parameter, default, description, takes_pert, takes_pert, False, precompute))
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
                        raise ValueError(
                            f"space {key!r} selected no genes for {pert!r}. Every metric would return "
                            f"nan rather than fail, so this is refused: widen the space, or check that "
                            f"its criterion matches the dataset."
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
