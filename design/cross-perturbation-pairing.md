# Design note — cross-perturbation pairing

**Status:** design note. Nothing here is implemented, and nothing should be until the
precondition at the bottom is met. This file exists so the reasoning is not lost.

## The problem

Every comparison scPertEval makes today is **same-label**. The whole inner loop is:

```python
gt   = ctx.view(pert, truth, p)   # runner.py, _run_per_perturbation
cand = ctx.view(pert, src,   p)
```

`pert` appears twice. Ground truth for perturbation X is compared against X's prediction, or
against X's positive/negative control. That assumption is baked into `Context.view()`, which
resolves a source *for the perturbation in hand*.

A family of methods now under development needs the two sides to be **different**
perturbations: score a query population against every reference in the library, then read out a
rank, a rank distance, a concordance, or a pass/fail. These evaluate the *protocols themselves*
rather than a model's predictions — a layer above what the library scores today.

There is no way to express that pairing, so callers encode it in the data: replicate one query
under every reference label and hand it to `score()`. The library then reasonably treats it as N
unrelated perturbations. `compare()` addresses the immediate need; this note is about what it
would take for such readouts to become first-class alongside `drf` and `bds`.

## Why this is not just a new calibrator

Three things genuinely do not fit inside `calibrate()`:

1. **Arity.** `Calibrator.requires` is a fixed tuple naming two candidates
   (`positive`, `negative`). A cross-perturbation readout needs N, named by reference
   perturbation, with N varying per call.
2. **Output schema.** `_finalize` emits one `raw_<name>` column per candidate plus the resolved
   source names. N references would mean roughly 2N columns in the per-perturbation frame and
   the CSV.
3. **Pairing.** Every comparison routes through `view(pert, source, p)`, indexed by the current
   perturbation. "Query i against reference j" cannot be expressed through a source, because a
   source is a function of the perturbation in hand.

There is also a vocabulary mismatch. `positive`/`negative`, `better`/`perfect` and DRF's dynamic
range all mean *"a positive control should beat a negative control for this perturbation."* A
meta-evaluation readout has a target and a field of distractors. Reusing the control vocabulary
would be a category error visible in every docstring and CSV column.

## Proposed shape

**Make pairing an explicit axis, declared by the calibrator.**

```python
drf   → pairing = "same-label"   # unchanged
bds   → pairing = "same-label"   # unchanged
score → pairing = "same-label"   # unchanged
<new> → pairing = "cross"
```

`run_protocol` already dispatches on `Protocol.scope` to choose between `_run_per_perturbation`
and `_run_dataset`, so "select an execution path" is an existing pattern; the cross path is a
third sibling. Nothing about `drf`/`bds`/`score` changes.

**A separate verb, not a mode flag on `calibrate()`.** The library has already answered this
question once: `CALIBRATORS` holds three entries — `drf`, `bds` and `score` — yet `score()` is
its own verb, and `calibrate()` explicitly refuses it:

```python
if calibrator not in ("drf", "bds"):
    raise ValueError("... use score() for predictions")
```

Shared registry, shared machinery, separate verb — because the *arguments* differ, not the
concept. The same answer applies here.

**One registry, not two.** A readout and a calibrator are the same shape: reduce per unit, then
aggregate. Keeping them in `CALIBRATORS` with a declared pairing is preferable to a parallel
`READOUTS` concept that would drift.

## What this buys

Readouts become first-class: they work with any protocol, return an `EvalResult` (whose
`.aggregate` / `.per_perturbation` split already matches "dataset-level interpretation with a
per-unit layer underneath"), get the existing CSV output and caching, and appear in
`scperteval list calibrators`.

Because the codebase is registry-driven throughout — `SOURCES`, `SPACES`, `DE_METHODS` and
`CALIBRATORS` are all extensible from user code — experimental readouts can be registered
externally and only upstreamed once they have survived contact with data.

## Explicitly out of scope

**`rank`, `transpose_rank` and `nir` stay exactly as they are.** They are *metrics*: they score a
prediction against ground truth and emit one value per perturbation. That is a different layer
from a meta-evaluation readout, which evaluates the protocol. The cross-perturbation distance
matrix inside `rank_retrieval` is an implementation detail, not a generalisation point.

## Where the line sits

The library should own: which queries, which references, which protocol, which readout.

It should **not** own how a query is constructed — degradation budgets, transforms, draws. That is
research surface, it will keep changing, and it is already expressible as "build the cells and
hand them in." Left unbounded it would freeze an evolving experiment design into a public
interface.

## Precondition

Do not implement this until **at least two or three readouts have stopped changing**. Designing
the reduction interface from a single example is how an abstraction ends up fitting nothing. The
open questions it would need to answer — does a readout need the target label, the full matrix or
one row, tie handling, a null distribution — are answerable from real readouts and not before.

Until then, callers get the raw matrix from `compare()` and own the reduction themselves.
