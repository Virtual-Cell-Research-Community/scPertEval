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
second argument accepts the same protocol-spec language as the CLI's `-p`: `"all"`, a group
(`"pseudobulk"` / `"distributional"` / `"de"`), a name, or a tunable protocol as
`name=value` — either a sequence or a comma-separated string. Python callers may also pass
concrete {class}`~scperteval.types.Protocol` objects directly. Pass `[]` if you only need
{func}`~scperteval.api.de`. Prepare-time knobs (`subsample`, `seed`, `min_cells`,
`perturbation_key`, `control_label`, `workers`, `name`) are fixed on the handle.

One handle is **reusable and thread-safe**: run as many `calibrate` / `score` / `de` calls
against it as you like — they share its dataset and caches (no reload), and are safe to run
concurrently.

```python
a = sp.calibrate(prep, "pearson_ctrl")  # reuses the shared dataset + caches
b = sp.calibrate(prep, "mse")
c = sp.de(prep, "t-test")
```

### Concrete protocols & `truth_availability()`

Python callers can pass concrete {class}`~scperteval.types.Protocol` objects to `prepare()`,
`calibrate()`, `score()`, and `truth_availability()` without registering them in the curated CLI
catalog:

```python
from scperteval.protocols import metrics
from scperteval.types import Protocol

proto = Protocol("my_mae", metrics.mae, representation="centroid", better="lower", perfect=0.0)
prep = sp.prepare(adata, [proto])
res = sp.score(prep, proto, predictions)
```

Concrete protocols must be fully resolved and cannot shadow a curated protocol name with different
semantics. Use {func}`~scperteval.api.truth_availability` to audit which `(protocol, perturbation)`
units are ground-truth-defined before loading predictions (default `truth="gt_all_cells"`; pass
`truth="gt_half"` to match calibration cohorts).

## Calibrate

{func}`~scperteval.api.calibrate` calibrates **one** protocol against the built-in
positive/negative controls (DRF or BDS) — the programmatic equivalent of
{func}`scperteval calibrate <scperteval.api.calibrate>`:

```python
res = sp.calibrate(prep, "pearson_ctrl", de_method="t-test")

res.aggregate  # {"mean": …, "median": …} — the DRF summary for this protocol
res.per_perturbation  # DataFrame: raw control values + the calibrated DRF, one row per perturbation
```

`calibrate` takes a **single** protocol — a curated name (`"pearson_ctrl"`), a tunable spec
(`"mse_top_k=30"`), or a concrete `Protocol` object; it does not accept `"all"` or a group. Pass
`calibrator="bds"` for the Bound Discrimination Score instead of DRF. Any perturbation whose raw or
calibrated score is undefined is recorded as `NaN` with a machine-readable `undefined_reason`.

### Your own vectors as sources

{func}`~scperteval.api.prepare` accepts `sources={"name": array}` to register **runtime user
sources** on the handle (never on the global registry, so they don't leak across handles). A 1-D
`(G,)` array is a centroid; a 2-D `(n_cells, G)` array is a cell population. Arrays are validated
(numeric, all-finite, `G` = the dataset's gene count) and copied. **Columns are assumed to be in
`adata.var_names` order** — only the count is checked, so a mis-ordered vector silently compares the
wrong genes.

A registered source is reusable in any slot that accepts its shape:

```python
prep = sp.prepare(adata, "pearson", sources={"my_baseline": vec})  # vec is a (G,) centroid
sp.calibrate(prep, "pearson", negative="my_baseline")  # ... as a control
sp.calibrate(prep, "pearson", center_on="my_baseline")  # ... as a centering baseline
```

`center_on` centers an **un-centred, centroid** protocol on a named centroid source (user or
built-in, e.g. `"global_mean"`). Because centering is protocol identity, this **mints a named
variant** `<protocol>_center_<name>` — recorded in `EvalResult` and any CSV — rather than silently
overriding the catalog protocol. It also works on {func}`~scperteval.api.score`. (The CLI can't pass
arrays, so `center_on` is Python-API-only.)

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
res.aggregate  # {"mean": …, "median": …} — the raw metric summary
res.per_perturbation  # DataFrame with a `score` column per perturbation
```

## Differential expression

{func}`~scperteval.api.de` computes per-gene differential expression for **one** method — the
equivalent of {func}`scperteval de <scperteval.api.de>` (defaulting to `target="gt_half"` vs
`reference="all_perturbed"`):

```python
d = sp.de(prep, "t-test")

# All observed cells vs non-targeting controls, for a chosen subset of perturbations
d_ctrl = sp.de(prep, "t-test", target="gt_all_cells", reference="control", perturbations=["pertA"])

d.statistic  # DataFrame: perturbations × genes (the test statistic)
d.pvalue_adj  # DataFrame: perturbations × genes (BH-adjusted p-values)
```

{func}`~scperteval.api.de` returns a {class}`~scperteval.api.DatasetDEResults`, a
`NamedTuple` of two DataFrames that also unpacks directly:

```python
statistic, pvalue_adj = sp.de(prep, "MWU")
```

Different DE methods reuse the same prepared dataset (each cached separately, no reload).

See the [Python API reference](../api/api) for full signatures.
