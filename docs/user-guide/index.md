# User guide

scPertEval runs in two modes:

- [Scoring predictions](scoring) compares a model's output to ground truth
- [Calibration](calibration) measures whether a protocol can tell real signal from an uninformative baseline

Start with whichever matches your goal, then see [Usage](usage).

```{mermaid} ../_mermaid/user-guide-overview.mmd
```

A protocol pairs a **metric** (e.g. Pearson correlation, MSE) with a **representation** of
the cells it's given (a single pseudobulk centroid, the full population of cells, or a
differential-expression result) and a **feature space** (which genes it looks at) — see
[Building blocks](building-blocks). Scoring mode compares that metric against real held-out
cells; calibration mode compares it against built-in positive/negative controls to gauge
whether the protocol is sensitive at all.

For the full internal wiring — datasets, caching, registries, output files — see
[Architecture](../api/architecture).

You can drive scPertEval two ways over the same engine: the [command-line interface](usage)
(`scperteval …`, which writes result files) and the native [Python API](python-api)
(`import scperteval`, which returns in-memory pandas objects). Pick whichever suits your
workflow — the concepts below apply to both.

```{toctree}
:maxdepth: 1

scoring
calibration
usage
python-api
protocols
building-blocks
datasets
limitations
```
