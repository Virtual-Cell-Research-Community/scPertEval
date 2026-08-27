# Installation

## From PyPI

```bash
pip install scperteval
```

## Optional: the Sinkhorn metrics

The Sinkhorn / optimal-transport protocols (`sinkhorn_w2_top_k`, `sinkhorn_w2_pca_k`) are
backed by [PyTorch](https://pytorch.org/) and
[GeomLoss](https://www.kernel-operations.io/geomloss/). Those are kept out of the base install
because torch is large, so enable them with the `sinkhorn` extra:

```bash
pip install "scperteval[sinkhorn]"
```

Without the extra everything else works normally: bulk selections (`-p all`, `-p distributional`)
simply skip those two protocols and say so, and naming one explicitly raises an error pointing
back at this command. `scperteval list protocols` marks them with their install state.

## From source

```bash
pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"
```

Or, for an editable install from a local clone:

```bash
git clone https://github.com/Virtual-Cell-Research-Community/scPertEval.git
cd scPertEval
pip install -e .
```

## Development setup

Install all dev dependencies (linting + docs + tests):

```bash
uv sync --group dev
```

Run linters:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/scperteval
```

Build the docs locally with live reload:

```bash
uv sync --group docs
uv run sphinx-autobuild docs docs/_build/html
```
