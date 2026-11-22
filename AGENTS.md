# AGENTS.md

## Design Philosophy

scPertEval is a research tool first. Its primary goal is making it **as simple as possible for
users to experiment with the core domain entities** — protocols, feature spaces, metrics,
sources, DE methods, and calibrators.

- **Extend existing entities; don't build parallel ones.** When an experiment isn't possible
  today, generalize the existing entity rather than adding a side-door, wrapper, or competing
  abstraction around it.
- **Keep composition open and uniform.** Prefer small, composable building blocks (pure metric
  functions, first-class spaces, open registries) over closed enums, monolithic switchboards, or
  special-case flags. An entity's contract should hold consistently wherever it applies.
- **Keep built-in catalogs curated; keep Python composition open.** Built-in catalogs hold
  canonical presets for CLI and common use; exploratory combinations should be composed on the
  fly in Python without mutating global registries.
- **Put reusable methods in core, not analysis-repo plumbing.** Canonical protocols and metrics
  from the literature belong in core; one-off dataset naming conventions, implicit column
  assumptions, and downstream manifest aliases belong in downstream analysis repos.
- **Prefer simple, Pythonic APIs.** One obvious way to express an idea; plain, readable
  interfaces over clever syntax.
- **Follow PR best practices.** Keep PRs small, focused, and single-purpose (e.g., separate
  performance optimizations from API or design changes), with concise tests and matching doc
  updates.
