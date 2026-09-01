# Building blocks

Spaces, DE methods, control sources, and calibrators are registered units — add one when
the palette is missing what a new protocol needs. Each is a small function (or object) plus
a one-line registration. To author the protocol or metric that draws on these blocks, see
[Create a protocol](protocols.md#create-a-protocol).

## Add a feature space

A space decides which features a protocol scores on. Add one by writing a rule and decorating
it, the same way DE methods and control sources are registered. Everything lives in
[`src/scperteval/blocks/spaces/catalog.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/spaces/catalog.py):

```python
@SPACES.subset("mito", default=20, description="top {v} mitochondrial genes by control expression")
def mito(ctx, k):
    """The k highest-expressed mitochondrial genes."""
    mt = np.flatnonzero([g.startswith("MT-") for g in ctx.ds.var_names])
    return mt[np.argsort(-ctx.control_mean()[mt])][:k]
```

`mito_<k>` now appears in `scperteval list spaces`, and a protocol can use it at any `k`.

**The rule** returns a column selection into the *full* gene axis — an integer array, or a
slice — never positions into some earlier subset, so selections from different spaces can be
folded together. It receives `(ctx, pert, value)`. The rule runs once per perturbation per
protocol, so anything computed over the whole dataset belongs behind a `Context` cache (as
`ctx.control_mean()` is), not recomputed here.

**Name a parameter `pert` to be given the perturbation.** That is also how a space says its
genes vary by perturbation, so scPertEval knows it can't compute the selection once and share it:

```python
def full(ctx): ...  # dataset-wide, no parameter
def heg(ctx, k): ...  # dataset-wide, takes k
def targets(ctx, pert): ...  # per-perturbation, no parameter
def top(ctx, pert, k): ...  # per-perturbation, takes k
```

There is no flag to set. A rule that doesn't name `pert` is never passed one, so reaching for it
raises `NameError` rather than silently scoring every perturbation on one panel. `pert` comes
first when present; any other shape is rejected at registration.

**The decorator** carries the metadata. `default` is the parameter value used when a caller
doesn't supply one; `{v}` in the description is filled in with it.

**Whether a space takes a parameter is read from the rule's signature.** A trailing argument
with a default means it takes none:

```python
@SPACES.subset("perturbed_genes", description="genes targeted by a perturbation")
def perturbed_genes(ctx):
    return targeted_genes(ctx)  # a @cached helper in helpers.py, see below
```

`scperteval list spaces` shows `mito_<k>` and `perturbed_genes` accordingly. Declaring a
parameter without a default (or a default without a parameter) is an error at import.

**Computations over the whole dataset** go in
[`helpers.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/spaces/helpers.py),
decorated with `@cached` so they run once per dataset instead of once per perturbation:

```python
@cached
def control_dispersion(scope: DatasetScope):
    """Per-gene normalized dispersion of the control cells."""
    ...  # scope.ds, scope.seed, scope.threads
```

Your rule then calls it as `control_dispersion(ctx)`. The body is handed a `DatasetScope`
([`scperteval/caching.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/caching.py)) —
the dataset plus the settings fixed at `prepare()` time — so a cached value can't accidentally
depend on per-call options like `--de-method`, which the cache would outlive.

### Composing subsets

`SPACES.combine_subsets` builds a new space from registered ones with a set operation from `OPS`
(`OPS.union`, `OPS.intersection`, `OPS.difference`, the last subtracting left to right):

```python
SPACES.combine_subsets(
    OPS.union,
    SPACES.instance("hvg", 8192),
    SPACES.instance("perturbed_genes"),
    name="perturbed_and_hvgs",
    description="HVG union perturbed genes",
)
```

The result is a space like any other: it appears in `scperteval list spaces`, resolves by name,
and can itself be composed. `name` is required rather than derived, because operator-symbol names
made `(a-b)+c` and `a-(b+c)` collide.

Whether the composite varies by perturbation is **derived from its operands** — union in a
per-perturbation space such as `top_50` and the result is per-perturbation too. Nothing to
declare, so nothing to get wrong.

### Spaces that aren't gene subsets

`pca_<k>` replaces the gene axis with components instead of narrowing it, so it has no gene
selection and can't be composed. Use `@SPACES.transform`, whose rule takes the cells and returns
the finished array:

```python
@SPACES.transform("pca", default=50, precompute=pca_for, description="top {v} principal components")
def pca(X, ctx, k):
    return pca_for(ctx, k).transform(to_dense(X))[:, :k]
```

**Advanced — `precompute`.** A `@cached` helper is computed the first time a rule asks for it,
which is inside the parallel scoring loop. For something heavy enough to want the machine's
threads to itself — PCA, for one — pass `precompute=<callable>` so it happens before the loop
instead. It is an optimisation only: the rule must still work if it never runs. `pca_<k>` does
this — see `pca_for` in
[`helpers.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/spaces/helpers.py).

## Add a DE method

A DE method maps `(target_cells, reference_cells) -> PerturbationDEResult(statistic, pvalue, pvalue_adj)`.
Register it with `@DE_METHODS.register` in [`src/scperteval/blocks/de.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/de.py) (the
`bh` helper there BH-adjusts p-values):

```python
@DE_METHODS.register("my_test", description="…")
def de_my_test(target, reference):
    statistic, pvalue = ...  # per-gene statistic and raw p-value
    return PerturbationDEResult(statistic=statistic, pvalue=pvalue, pvalue_adj=bh(pvalue))
```

Then `--de-method my_test` routes every DE-dependent unit through it.

A method whose statistic is expressible from per-gene moments (mean, variance, cell count) may
additionally declare `from_moments=<callable>` in its `register(...)` metadata to reuse
scPertEval's cached reference moments, as the built-in `t-test` does — the callable takes
`(mean_t, var_t, n_t, mean_r, var_r, n_r)` and returns a `PerturbationDEResult`. It's a pure
performance opt-in: correctness is identical without it, and the `(target, reference)` function
above is still required.

## Add a control source

A source maps `(ctx, pert) -> cells or a 1-D centroid`, declaring which with `provides`.
Register it with `@SOURCES.register` in [`src/scperteval/sources.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/sources.py):

```python
@SOURCES.register("my_baseline", provides="centroid", description="…")
def src_my_baseline(ctx, pert):
    return ...  # a 1-D centroid (or cells, if provides="cells")
```

Use it as a control at the CLI via `--positive`/`--negative`, or make it a row's default with
`default_positive=`/`default_negative=` (only where the row deviates from the representation's
generic default; controls are otherwise resolved at runtime — see
[Protocols → Control sources](protocols.md)).

## Add a calibrator

A calibrator declares the control roles it needs, a per-perturbation combine, and a
cross-perturbation aggregate. Add a `Calibrator` to the `CALIBRATORS` dict in
[`src/scperteval/calibrators.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/calibrators.py):

```python
CALIBRATORS["my_score"] = Calibrator(
    "my_score",
    ("positive", "negative"),
    per_pert=lambda raws, p: ...,  # raws["positive"], raws["negative"] -> one number
    aggregate=lambda v: {"my_score": float(np.nanmean(v))},
    description="…",
)
```

Then `--calibrator my_score` reports it.
