# Tutorials

Most tutorials are runnable notebooks (use the download/Colab buttons at the top of the page);
one entry below is background reading plus a notebook. See
[Contributing](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/CONTRIBUTORS.md#tutorials-and-notebooks)
to add one.

```{toctree}
:maxdepth: 1

notebooks/01_cli_walkthrough
notebooks/02_preparing_a_dataset
notebooks/03_python_api
benchmark/index
```

- **CLI walkthrough** — run scPertEval end-to-end on a tiny synthetic dataset.
- **Preparing a dataset** — turn a raw perturb-seq `.h5ad` into scPertEval-ready form, worked
  through three real datasets. This notebook downloads large real files, so it is not run in CI.
- **Python API** — use scPertEval programmatically from a notebook or script.
- **Model Benchmark** — why {cite}`AhlmannEltze_2025` and {cite}`Miller_2025` disagree on whether deep-
  learning models beat simple baselines, which models and metrics we benchmark to test it, and a
  worked notebook that builds every model's and baseline's predictions, calibrates each protocol
  against positive/negative controls, and runs the pairwise statistical tests to reproduce both
  papers' verdicts end to end (see [Model Benchmark](benchmark/index.md)).
