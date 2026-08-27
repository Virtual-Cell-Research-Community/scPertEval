# Releasing scPertEval

This is the maintainer runbook for cutting a release and the reference for how the
repository's release/CI/docs machinery is wired together.

---

## Quickstart

**To cut a release:**

1. Update `CHANGELOG.md`: rename the `(unreleased)` heading to the version you're about to
   release and start a fresh `(unreleased)` section above it. Commit + merge to `main`.
2. Create a GitHub Release with a `vX.Y.Z` tag from the latest `main`:

   ```bash
   gh release create v0.1.0 --target main --title "v0.1.0" --generate-notes
   ```

   (Or GitHub UI → *Releases* → *Draft a new release* → create the tag → *Generate release
   notes* → *Publish*.)
3. Approve the publish when the `Release` workflow pauses for the `pypi` environment review.

That's it — everything below step 3 is automatic.

**How to pick the version number (semver):**

- Pre-1.0 (where we are now): bump the **minor** (`0.2.0`) for new features *or* breaking
  changes; bump the **patch** (`0.1.1`) for bug fixes only.
- Post-1.0: **major** = breaking API change, **minor** = backward-compatible feature,
  **patch** = backward-compatible fix.
- When unsure, the `CHANGELOG.md` you just wrote is the tell: any *Removed*/*Changed* entry
  that breaks callers ⇒ minor (pre-1.0) or major (post-1.0).

**What happens when you publish the Release:**

- The **`Release` workflow** (`.github/workflows/release.yaml`) builds the package and
  publishes it to **PyPI** — after you approve the `pypi` environment gate.
- The package **version is taken from the tag** (`v0.1.0` → `0.1.0`); nothing to edit by hand.
- **ReadTheDocs** independently builds the tag and repoints `stable` at it, so the default
  docs now show the released version. `latest` keeps tracking `main`.

---

## How versioning works (hatch-vcs)

The version is **not stored in `pyproject.toml`** — it's derived from the latest git tag by
[`hatch-vcs`](https://github.com/ofek/hatch-vcs). The git tag is the single source of truth.

- Tag `v0.1.0` on a clean commit → the build is version `0.1.0`.
- Any build *between* tags (local dev, CI on a branch) gets a dev version like
  `0.1.dev12+gabc1234` — this is expected and only ever appears on non-release builds.
- `scperteval.__version__` reads the installed version back via `importlib.metadata`, so it
  always matches whatever was built.

Implications:
- **You never bump a version number in a file.** The tag *is* the bump.
- The commit you tag must be **clean** (no uncommitted changes) or the version picks up a
  local `.d<date>` "dirty" suffix. Tag from a merged `main`, not a working tree.
- CI checkouts use `fetch-depth: 0` so the tags are present for the version to resolve.
- `uv.lock` is intentionally gitignored (this is a library; we test against a range of
  dependency versions rather than pinning one).

---

## What runs automatically

| Workflow | File | Trigger | Does |
|---|---|---|---|
| **Lint** | `lint.yaml` | push/PR to `main` | ruff lint + format check, mypy, pyright |
| **Test** | `test.yaml` | push/PR to `main`, twice-monthly cron | hatch matrix on Python 3.11–3.14 (with the `sinkhorn` extra), coverage → Codecov |
| **Check Build** | `build.yaml` | push/PR to `main` | `uv build` + `twine check --strict` |
| **Notebooks** | `notebooks.yaml` | push/PR to `main` | re-executes tutorial notebooks against the current API |
| **Release** | `release.yaml` | GitHub Release *published* | `uv build` + publish to PyPI via Trusted Publishing |

The **required status checks** for merging into `main` (set in branch protection) are the job
names: `lint`, `package`, and `Tests pass in all hatch environments` (the alls-green aggregate
that is green only if the whole test matrix passed).

---

## PyPI publishing (Trusted Publishing — no tokens)

`release.yaml` uses PyPI **Trusted Publishing** (OIDC): there is **no API token** stored
anywhere. PyPI is configured to trust exactly this repo + workflow + environment, and
`pypa/gh-action-pypi-publish` also attaches **Sigstore provenance attestations** to each
release automatically — so releases are signed with nothing to manage.

The publish step runs inside the **`pypi` GitHub Environment**, which has **required
reviewers**. That's the human approval gate: the workflow pauses until a maintainer approves.

To change publishing behavior, the relevant settings live in:
- **PyPI** → project `scperteval` → *Publishing* → the Trusted Publisher entry (owner
  `Virtual-Cell-Research-Community`, repo `scPertEval`, workflow `release.yaml`, environment
  `pypi`).
- **GitHub** → *Settings* → *Environments* → `pypi` → required reviewers.

---

## Branch & tag protection

Configured under **Settings → Rules → Rulesets**:

- **Branch ruleset** targeting the default branch (`main`): require a PR, require the status
  checks `lint` / `package` / `Tests pass in all hatch environments`, require branches up to
  date, block force-pushes.
- **Tag ruleset** targeting `v*`: restrict creations/updates/deletions so only maintainers
  (bypass list) can create or move release tags. This is what stops an accidental tag from
  triggering a publish.

---

## Security tooling

- **Dependabot** (GitHub *Settings → Advanced Security*): alerts + malware alerts + security
  updates (grouped) are **on**; routine version-update PRs are **off** (deliberately, to keep
  noise down). No `.github/dependabot.yml` is needed — that file only configures the
  version-update PRs we've disabled; alerts and security updates are repo settings.
- **CodeQL** (default setup, via *Settings → Advanced Security*): scans our own code on
  push/PR + a schedule. Default setup is right for a pure-Python library — no build step to
  configure and the default query suite is solid. Switch to *advanced* (a committed workflow)
  only if we ever want the noisier `security-extended` query pack or custom paths.
- **Private vulnerability reporting** (recommended, *Settings → Advanced Security* →
  *Private vulnerability reporting*): gives researchers a private channel to report issues.
  Enabling this is why we don't need a separate `SECURITY.md` file.
- **Optional manual audit** before a release, if you want a belt-and-suspenders dependency
  check (Dependabot already covers this):

  ```bash
  uv export --no-dev --format requirements-txt | uvx pip-audit -r -
  ```

---

## Documentation (ReadTheDocs)

Docs live in this repo (`docs/`), and `docs/conf.py` reads the version from the installed
package metadata — so docs and package versions can never disagree.

- **`latest`** builds from every `main` push (development docs).
- **`stable`** builds from the newest release tag and is the **default version** visitors see.
  RTD creates and repoints `stable` automatically as you tag new releases.
- An **Automation Rule** (RTD → *Admin → Automation Rules*) activates each new `vX.Y.Z` tag:
  match *SemVer versions*, version type *Tag*, action *Activate version*.
- Set **default version = `stable`** in RTD → *Admin → Settings*. Note `stable` only appears
  in that dropdown **after the first tag exists**, so this is a one-time step to do right after
  the first release.

---

## One-time setup checklist (outside the repo)

These are the settings that make the above work; they live in web UIs, not in the repo. Do
them once:

- [ ] **PyPI**: create the project's Trusted Publisher (a *pending publisher* before the first
      release); enable 2FA on maintainer accounts.
- [ ] **GitHub → Environments**: create `pypi` with required reviewers.
- [ ] **GitHub → Rules → Rulesets**: branch protection on `main` + tag protection on `v*`.
- [ ] **GitHub → Advanced Security**: Dependabot (alerts/malware/security updates, grouped),
      CodeQL default setup, private vulnerability reporting, secret scanning + push protection.
- [ ] **GitHub → Actions → General**: default workflow permissions = read-only.
- [ ] **ReadTheDocs**: automation rule to activate tags; set default version to `stable`
      (after the first release).
- [ ] *(optional)* **Codecov**: activate the repo so coverage upload works.

---

## Troubleshooting

- **Published version came out as `0.1.dev…` instead of `0.1.0`** — the tag wasn't on a clean
  commit, or CI didn't fetch tags. Ensure you tagged a merged `main` commit and that the
  workflow checkout uses `fetch-depth: 0`.
- **`stable` isn't selectable as the default RTD version** — no tags exist yet; it appears
  only after the first release. Also confirm `stable` is *Active* on the RTD Versions page.
- **Release workflow didn't run** — it triggers on a *published* GitHub Release, not on a bare
  pushed tag. Create the Release, don't just `git push --tags`.
- **PyPI publish rejected** — check the Trusted Publisher entry on PyPI matches the repo,
  workflow filename (`release.yaml`), and environment (`pypi`) exactly.
