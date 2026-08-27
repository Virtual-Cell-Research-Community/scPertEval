# scPertEval — Evaluation Protocols for Perturbation Sequencing

[![Stars][stars-badge]][stars-link]
[![PyPI][pypi-badge]][pypi-link]
[![PyPI Downloads][pepy-badge]][pepy-link]
[![Docs][docs-badge]][docs-link]
[![Lint][lint-badge]][lint-link]
[![Test][test-badge]][test-link]
[![Build][build-badge]][build-link]

scPertEval is a command-line tool for **experimenting with and sharing reference implementations of
evaluation protocols** in single-cell perturbation studies.

It is introduced in [*Towards Principled Evaluation of Single-Cell Perturbation Prediction
Models*](https://doi.org/10.64898/2026.07.23.740433) {cite}`Schafer_2026` by Philipp S. L.
Schäfer, Kendall A. Reid, Zach Boldyga, Ekin D. Aksu, Hugo Hakem, and Julio Saez-Rodriguez —
please cite it if you use this package.

Evaluating predictions across a dataset's perturbations reduces to a single question: how
different is one group of cells from another? To answer this, an **evaluation protocol** is
defined: a specific formulation of a metric, along with some representation of the
perturbation data fed to the metric. However, there are a multitude of possibilities — many
already reflected in the literature — and it can be challenging to compare and contrast
protocols across the field and ultimately choose the right approach for a given dataset and
problem space.

scPertEval renders each protocol as a short, readable building block to run, read, reuse,
and contribute back — a place for collaboration and alignment in the field. The same catalog
of protocols backs three commands:

- **`score`** — score a model's predictions against ground truth, one metric value per
  perturbation (see [Scoring predictions](user-guide/scoring.md)).
- **`calibrate`** — calibrate a protocol against built-in positive/negative controls, reporting
  the **Dynamic Range Fraction (DRF)** and **Bound Discrimination Score (BDS)** — how well it
  separates real signal from an uninformative baseline (see [Calibration](user-guide/calibration.md)).
- **`de`** — export per-gene differential expression to HDF5.

## Quick start

```bash
pip install scperteval
scperteval calibrate data/wessels23.h5ad -p all --de-method t-test
```

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} {octicon}`desktop-download;1em;` Installation
:link: installation
:link-type: doc
Get scPertEval installed and set up your development environment.
:::

:::{grid-item-card} {octicon}`book;1em;` User guide
:link: user-guide/index
:link-type: doc
Learn how to run protocols, interpret scores, and explore the building blocks.
:::

:::{grid-item-card} {octicon}`mortar-board;1em;` Tutorials
:link: tutorials
:link-type: doc
Step-by-step notebooks: CLI walkthrough, Python API, and extending the tool.
:::

:::{grid-item-card} {octicon}`code-square;1em;` API reference
:link: api
:link-type: doc
Full reference for the Python API.
:::

:::{grid-item-card} {octicon}`mark-github;1em;` GitHub
:link: https://github.com/Virtual-Cell-Research-Community/scPertEval
:link-type: url
Browse the source code, open issues, or contribute a pull request.
:::

::::

## Citation

If you use scPertEval, please cite {cite}`Schafer_2026`.

```bibtex
@unpublished{Schafer_2026,
    author = {Schäfer, Philipp S. L. and Reid, Kendall A. and Boldyga, Zach
              and Aksu, Ekin Deniz and Hakem, Hugo and Saez-Rodriguez, Julio},
    title  = {Towards a Principled Evaluation of Single-Cell Perturbation
              Response Prediction Models},
    note   = {In preparation},
    year   = {2026},
}
```

[stars-badge]: https://img.shields.io/github/stars/Virtual-Cell-Research-Community/scPertEval?style=flat&logo=GitHub&color=yellow
[stars-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/stargazers
[pypi-badge]: https://img.shields.io/pypi/v/scperteval.svg
[pypi-link]: https://pypi.org/project/scperteval
[pepy-badge]: https://static.pepy.tech/badge/scperteval
[pepy-link]: https://pepy.tech/project/scperteval
[docs-badge]: https://readthedocs.org/projects/scperteval/badge/?version=latest
[docs-link]: https://scperteval.readthedocs.io/
[lint-badge]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/lint.yaml/badge.svg
[lint-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/lint.yaml
[test-badge]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/test.yaml/badge.svg
[test-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/test.yaml
[build-badge]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/build.yaml/badge.svg
[build-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/build.yaml
<!-- Codecov badge intentionally omitted until the repo is activated in the Codecov org. -->

```{toctree}
:hidden: true
:maxdepth: 2

installation.md
user-guide/index
tutorials.md
api.md
changelog.md
Contributing <contributing.md>
references.md
```
