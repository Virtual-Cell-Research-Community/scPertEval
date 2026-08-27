<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Virtual-Cell-Research-Community/scPertEval/main/docs/_static/logo/scPertEval-dark-logo.svg">
  <img alt="scPertEval" src="https://raw.githubusercontent.com/Virtual-Cell-Research-Community/scPertEval/main/docs/_static/logo/scPertEval-logo.svg" width="400">
</picture>

# scPertEval — Evaluation Protocols for Perturbation Sequencing

[![Paper][paper-badge]][paper-link]
[![Stars][stars-badge]][stars-link]
[![PyPI][pypi-badge]][pypi-link]
[![PyPI Downloads][pepy-badge]][pepy-link]
[![Docs][docs-badge]][docs-link]
[![Lint][lint-badge]][lint-link]
[![Test][test-badge]][test-link]
[![Build][build-badge]][build-link]

scPertEval is a toolkit for **experimenting with and sharing reference implementations of
evaluation protocols** in single-cell perturbation studies, usable both as a **command-line
interface** and as a **native Python API**. The same catalog of protocols backs three actions:
**`score`** (score a model's predictions against ground truth), **`calibrate`** (calibrate a
protocol against empirical positive/negative controls per perturbation, reporting the **Dynamic
Range Fraction (DRF)** and **Bound Discrimination Score (BDS)**), and **`de`** (export per-gene
differential expression).

scPertEval is introduced in **[Towards Principled Evaluation of Single-Cell Perturbation
Prediction Models][paper-link]**, where we develop a taxonomy of evaluation protocols —
decomposing them into representation, metric, score transformation, and reporting strategy —
and use this package to assess protocol behavior across seven public perturbation datasets.

> Schäfer, P. S. L., Reid, K. A., Boldyga, Z., Aksu, E. D., Hakem, H., & Saez-Rodriguez, J.
> (2026). *Towards Principled Evaluation of Single-Cell Perturbation Prediction Models*.
> bioRxiv. <https://doi.org/10.64898/2026.07.23.740433>

If you use scPertEval in your work, please cite that paper (see [CITATION.cff](CITATION.cff)).

**→ Full documentation at <https://scperteval.readthedocs.io/>**

## Install

```bash
pip install scperteval
```

Or from this repo:

```bash
pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"
```

The Sinkhorn / optimal-transport metrics (the `sinkhorn_w2_*` protocols) need PyTorch and
[GeomLoss](https://www.kernel-operations.io/geomloss/), which are optional to keep the base
install light. Enable them with the `sinkhorn` extra:

```bash
pip install "scperteval[sinkhorn]"
```

## Quick start

From the command line:

```bash
# calibrate protocols against built-in controls (DRF/BDS)
scperteval calibrate data/wessels23.h5ad -p all --de-method t-test

# score a model's predictions against ground truth
scperteval score data/wessels23.h5ad predictions.h5ad -p all

scperteval list protocols   # also: de-methods | spaces | sources | calibrators
```

Or from Python — the same protocols, returning results in memory (see the
[Python API guide](https://scperteval.readthedocs.io/en/latest/user-guide/python-api.html)):

```python
import scperteval as sp

prep = sp.prepare("data/wessels23.h5ad", "pearson_ctrl")  # read + index once, reusable
result = sp.calibrate(prep, "pearson_ctrl", de_method="t-test")
result.aggregate  # {"mean": …, "median": …} — calibrated DRF summary
result.per_perturbation  # the per-perturbation detail table
```

Sample datasets are available at
`https://storage.googleapis.com/scperteval/processed/<dataset>_processed_complete.h5ad`.

---

## Citation

If you use scPertEval, please cite the paper it accompanies:

```bibtex
@article{Schafer_2026_scPertEval,
  author  = {Sch{\"a}fer, Philipp S. L. and Reid, Kendall A. and Boldyga, Zach and
             Aksu, Ekin D. and Hakem, Hugo and Saez-Rodriguez, Julio},
  title   = {Towards Principled Evaluation of Single-Cell Perturbation Prediction Models},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.07.23.740433},
}
```

## Authors

scPertEval originates with the authors of that paper — Philipp S. L. Schäfer, Kendall A. Reid,
Zach Boldyga, Ekin D. Aksu, Hugo Hakem, and Julio Saez-Rodriguez — and is maintained as a
community project under the [Virtual Cell Research Community][homepage-link].

**Contributing:** see [CONTRIBUTORS.md](CONTRIBUTORS.md).

[paper-badge]: https://img.shields.io/badge/paper-bioRxiv-b31b1b.svg
[paper-link]: https://www.biorxiv.org/content/10.64898/2026.07.23.740433v1
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
<!-- Codecov badge intentionally omitted until the repo is activated in the Codecov org --
     it renders "unknown" otherwise. See the upload step in .github/workflows/test.yaml.
[codecov-badge]: https://codecov.io/gh/Virtual-Cell-Research-Community/scPertEval/branch/main/graph/badge.svg
[codecov-link]: https://codecov.io/gh/Virtual-Cell-Research-Community/scPertEval
-->

[homepage-link]: https://github.com/Virtual-Cell-Research-Community
