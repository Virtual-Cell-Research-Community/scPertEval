# Python API

scPertEval can be driven two ways: the [command-line interface](usage) (`scperteval …`) and
this native **Python API**. Both run the exact same engine over the same
[input data](usage.md#input-data); the CLI writes result files, while the Python API returns
in-memory [pandas](https://pandas.pydata.org/) objects — ideal for notebooks and scripts.

Install with `pip install scperteval` (or, from this repo,
`pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"`),
then:

```python
import scperteval as sp
```

Everything you need is at the package root; you don't build a `RunConfig` or touch
`Context`.

## Prepare first, then run

The API is always **prepare, then run**. You first call {func}`~scperteval.api.prepare` to
read and index a dataset once (and precompute the feature spaces the protocols you name will
need), which returns a reusable {class}`~scperteval.api.Prepared` handle. You then pass that
handle to {func}`~scperteval.api.calibrate`, {func}`~scperteval.api.score`, or
{func}`~scperteval.api.de` — each evaluates **one** protocol (or computes **one** DE method)
and returns its result in memory.

```python
prep = sp.prepare("data/wessels23.h5ad", ["pearson_ctrl", "mse", "de_auprc"])
```

`dataset` accepts either a path to a `.h5ad` **or** an in-memory
{class}`~anndata.AnnData`, so there's no need to write a temporary file in a notebook. The
second argument is the same protocol-spec language as the CLI's `-p`: `"all"`, a group
(`"pseudobulk"` / `"distributional"` / `"de"`), a name, or a tunable protocol as
`name=value` — either a list or a comma-separated string. Pass `[]` if you only need
{func}`~scperteval.api.de`. Prepare-time knobs (`subsample`, `seed`, `min_cells`,
`perturbation_key`, `control_label`, `workers`, `name`) are fixed on the handle.

One handle is **reusable and thread-safe**: run as many `calibrate` / `score` / `de` calls
against it as you like — they share its dataset and caches (no reload), and are safe to run
concurrently.

```python
a = sp.calibrate(prep, "pearson_ctrl")   # reuses the shared dataset + caches
b = sp.calibrate(prep, "mse")
c = sp.de(prep, "t-test")
```

## Calibrate

{func}`~scperteval.api.calibrate` calibrates **one** protocol against the built-in
positive/negative controls (DRF or BDS) — the programmatic equivalent of
{func}`scperteval calibrate <scperteval.api.calibrate>`:

```python
res = sp.calibrate(prep, "pearson_ctrl", de_method="t-test")

res.aggregate          # {"mean": …, "median": …} — the DRF summary for this protocol
res.per_perturbation   # DataFrame: raw control values + the calibrated DRF, one row per perturbation
```

`calibrate` takes a **single** protocol spec — a name (`"pearson_ctrl"`) or a tunable one
(`"mse_top_k=30"`); it does not accept `"all"` or a group. Pass `calibrator="bds"` for the Bound
Discrimination Score instead of DRF.

### Your own vectors as sources

{func}`~scperteval.api.prepare` accepts `sources={"name": array}` to register **runtime user
sources** on the handle (never on the global registry, so they don't leak across handles). A 1-D
`(G,)` array is a centroid; a 2-D `(n_cells, G)` array is a cell population. Arrays are validated
(numeric, all-finite, `G` = the dataset's gene count) and copied. **Columns are assumed to be in
`adata.var_names` order** — only the count is checked, so a mis-ordered vector silently compares the
wrong genes.

A registered source is reusable in any slot that accepts its shape:

```python
prep = sp.prepare(adata, "pearson", sources={"my_baseline": vec})   # vec is a (G,) centroid
sp.calibrate(prep, "pearson", negative="my_baseline")               # ... as a control
sp.calibrate(prep, "pearson", center_on="my_baseline")              # ... as a centering baseline
```

`center_on` centers an **un-centred, centroid** protocol on a named centroid source (user or
built-in, e.g. `"global_mean"`). Because centering is protocol identity, this **mints a named
variant** `<protocol>_center_<name>` — recorded in `EvalResult` and any CSV — rather than silently
overriding the catalog protocol. It also works on {func}`~scperteval.api.score`. (The CLI can't pass
arrays; centering on named sources there is deferred to the broader centering-source redesign.)

{func}`~scperteval.api.calibrate` returns an {class}`~scperteval.api.EvalResult`:

- `.aggregate` — a `dict` of the calibrator's summary stats (`mean`/`median` for DRF,
  `bds` for BDS).
- `.per_perturbation` — a {class}`~pandas.DataFrame`, one row per perturbation, identical to
  the CSV the CLI writes.

By default nothing is written; pass `out_dir="results"` to also write the CLI-style CSV.

## Score

{func}`~scperteval.api.score` scores a model's predictions against ground truth for **one**
protocol — the equivalent of {func}`scperteval score <scperteval.api.score>`. Note the
argument order: `(prepared, protocol, predictions)`. Predictions accept a path or an
in-memory {class}`~anndata.AnnData`, and must have the same genes and perturbation labels as
the dataset:

```python
res = sp.score(prep, "pearson", "predictions.h5ad")
res.aggregate          # {"mean": …, "median": …} — the raw metric summary
res.per_perturbation   # DataFrame with a `score` column per perturbation
```

## Differential expression

{func}`~scperteval.api.de` computes per-gene differential expression (ground truth vs
all-perturbed) for **one** method — the equivalent of
{func}`scperteval de <scperteval.api.de>`:

```python
d = sp.de(prep, "t-test")

d.statistic    # DataFrame: perturbations × genes (the test statistic)
d.pvalue_adj   # DataFrame: perturbations × genes (BH-adjusted p-values)
```

{func}`~scperteval.api.de` returns a {class}`~scperteval.api.DatasetDEResults`, a
`NamedTuple` of two DataFrames that also unpacks directly:

```python
statistic, pvalue_adj = sp.de(prep, "MWU")
```

Different DE methods reuse the same prepared dataset (each cached separately, no reload).

See the [Python API reference](../api/api) for full signatures.
