# Contributing to scPertEval

scPertEval is meant to be a shared catalog of evaluation protocols, so contributions are
welcome. There are a few paths, depending on what you're changing.

## New evaluation protocol implementations — open a Pull Request

If you're adding a protocol (a new metric, or a new combination of an existing metric with
a space / centering / controls), **open a PR directly.** This is the common case and the
whole point of the project. See [Create a protocol](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/docs/user-guide/protocols.md#create-a-protocol) for the
two-step pattern (a pure function in `src/scperteval/protocols/metrics.py` plus a row in
`src/scperteval/protocols/table.py`). Adding a new building block (feature space, DE method, control
source, calibrator) the same way is also welcome as a PR.

Please include:
- a one-line reference to the source paper/method the protocol comes from, where applicable;
- the protocol added to the table and runnable via `scperteval calibrate ... -p <name>`.

## Bugs or changes to core code — open an Issue first

If you've found a bug, or want to change shared/core behavior (the runner, the context
engine, the reference/sampling logic, the calibrators, or the scoring semantics),
**open an Issue and discuss it first** before sending a PR. Core changes affect every
protocol's results, so we want to agree on the approach before implementation.

## Tutorials and notebooks

For tutorials and more in-depth examples, consider adding a notebook under `docs/notebooks/`.
Commit it as an executed notebook (outputs saved) so the rendered docs show real output, then
add it to the toctree in `docs/tutorials.md`. The `Notebooks` CI workflow re-executes every
notebook against the current API to guard against tutorials that no longer run.
