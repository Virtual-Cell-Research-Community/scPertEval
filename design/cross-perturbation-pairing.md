# Design note — cross-perturbation pairing

**Status:** draft. The mechanism is implemented; no calibrator uses it yet. This stays open
until one does — see "Why this is a draft" at the bottom.

## The assumption this removes

Every comparison the calibrators make is **same-label**. The inner loop is literally:

```python
gt   = ctx.view(pert, truth, p)   # runner.py, _run_per_perturbation
cand = ctx.view(pert, src,   p)
```

`pert` appears twice: ground truth for X against X's prediction, or against X's positive or
negative control. The assumption is not declared anywhere — it is simply what the loop does, and
`Context.view()` reinforces it by resolving a source *for the perturbation in hand*.

{func}`~scperteval.api.compare` already scores across labels: one query population against every
reference, each pair judged in the reference's own feature space. But its output is raw values.
A calibrator cannot run on that path, because `run_protocol` has no way to be told which pairing
a calibrator wants.

## The change

Pairing becomes an axis the calibrator declares:

```python
drf   → pairing = "same-label"   # unchanged
bds   → pairing = "same-label"   # unchanged
score → pairing = "same-label"   # unchanged
```

The runner dispatches on it, alongside the existing dispatch on `Protocol.scope`. Nothing about
`drf`, `bds` or `score` changes; the same-label path is what they already had, now named.

## Why a separate verb rather than a mode on `calibrate()`

The library answered this once already. `CALIBRATORS` holds three entries — `drf`, `bds` and
`score` — yet `score()` is its own verb, and `calibrate()` explicitly refuses it:

```python
if calibrator not in ("drf", "bds"):
    raise ValueError("... use score() for predictions")
```

Shared registry, shared machinery, separate verb — because the *arguments* differ, not the
concept. Three things make the same split necessary here:

1. **Arity.** `Calibrator.requires` names a fixed pair of candidates. A cross-paired reduction
   takes one value per reference, and the count varies per call.
2. **Output schema.** `_finalize` emits one `raw_<name>` column per candidate. N references would
   mean roughly 2N columns in the per-perturbation frame and the CSV.
3. **Pairing.** Every comparison routes through `view(pert, source, p)`, indexed by the current
   perturbation. "This query against that reference" cannot be expressed through a source,
   because a source is a function of the perturbation in hand.

There is also a vocabulary mismatch: `positive`/`negative`, `better`/`perfect` and DRF's dynamic
range all mean *"a positive control should beat a negative control for this perturbation."* That
sentence has no meaning under cross pairing, and reusing the words would put it in every
docstring and CSV column.

## One registry, not two

A cross-paired reduction and a calibrator are the same shape: reduce per unit, then aggregate.
Keeping both in `CALIBRATORS` behind a declared pairing is better than a parallel registry that
would drift.

Because every registry here is extensible from user code — `SOURCES`, `SPACES`, `DE_METHODS`,
`CALIBRATORS` — a reduction can be registered externally and tried without touching the library.

## Out of scope

`rank`, `transpose_rank` and `nir` stay exactly as they are. They are *metrics*: they score a
prediction against ground truth and return one value per perturbation. The cross-perturbation
matrix inside `rank_retrieval` is an implementation detail of one metric, not a generalisation
point.

## Why this is a draft

The reduction interface is being designed **ahead of its first consumer**, which is the usual way
to get an abstraction that fits nothing. In particular, how a reduction reaches the unit it should
be measured against is settled here by argument rather than by evidence, and questions like tie
handling and whether a reduction needs the whole matrix or only one row cannot be answered yet.

Merge when at least two independent reductions exist and have stopped changing. Until then this
branch records the mechanism and keeps it compiling against `main`.
