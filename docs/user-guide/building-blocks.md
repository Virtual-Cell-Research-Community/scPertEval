# Building blocks

Spaces, DE methods, control sources, and calibrators are registered units — add one when
the palette is missing what a new protocol needs. Each is a small function (or object) plus
a one-line registration. To author the protocol or metric that draws on these blocks, see
[Create a protocol](protocols.md#create-a-protocol).

## Add a feature space

A space is a function `(X, ctx, pert) -> dense (cells × genes) array` that transforms the
gene axis. Register it with `@SPACES.register` in
[`src/scperteval/blocks/spaces.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/spaces.py); pass `global_space=True` if it doesn't
depend on the perturbation (so it can be computed once and shared):

```python
@SPACES.register("hvg_100", global_space=True, description="100 highest-variance genes")
def space_hvg(X, ctx, pert):
    keep = ...  # indices of the genes to keep
    return to_dense(X[:, keep])
```

For a per-perturbation subset derived from the ground-truth DE (like `top_k` / `degs`), use
the `register_de_space(name, field=..., top=...)` helper in the same file instead.

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
