# Releasing conclave

This is the operator runbook for cutting a release of **conclave**.

Three names matter and they are deliberately different:

| Thing | Value |
|-------|-------|
| PyPI distribution name (what `pip install` uses) | `conclave-cli` |
| CLI command (what users type) | `conclave` |
| Import package (what you `import`) | `conclave` |
| GitHub repository | `DataScience-EngineeringExperts/conclave` |

Install is therefore `pip install conclave-cli`, but the command stays `conclave`
and the import stays `from conclave import Council`. The PyPI name `conclave` is an
**unrelated** package by another author (a blockchain client) — that is why the
distribution name is `conclave-cli`.

The publish + signing automation lives in
[`.github/workflows/release.yml`](.github/workflows/release.yml). That workflow is
triggered only when a GitHub *Release* is published. The publish job succeeds only when
PyPI trusts the exact organization repository and workflow identity below.

---

## 0. PyPI Trusted Publisher prerequisite

The workflow publishes through **OIDC Trusted Publishing**; there is no API token or
stored GitHub secret. On PyPI, open **conclave-cli → Manage → Publishing** and confirm
an active publisher with these exact values:

- **Owner:** `DataScience-EngineeringExperts`
- **Repository:** `conclave`
- **Workflow:** `release.yml`
- **Environment:** blank (the workflow has no GitHub environment)

If the repository was transferred from `ernestprovo23/conclave`, replace the old publisher
before releasing. Public provenance proves the identity used by past releases, not the current
private publisher configuration. A mismatch fails closed in the publish job; never fall back to
a token upload.

---

## 1. Cut a release

Do this on a clean checkout of `main` with all intended release changes merged.

1. **Update the changelog.** In [`CHANGELOG.md`](CHANGELOG.md), move the
   `## [Unreleased]` entries under a new `## [X.Y.Z] - <YYYY-MM-DD>` heading with
   today's date. Leave a fresh empty `## [Unreleased]` section above it.

2. **Bump the version in BOTH places.**
   - In [`pyproject.toml`](pyproject.toml), set `[project] version = "X.Y.Z"`.
   - In [`src/conclave/__init__.py`](src/conclave/__init__.py), set
   `__version__ = "X.Y.Z"`.
   (The distribution name `conclave-cli` is already set — do **not** change it.)

3. **Commit.**
   ```bash
   git add CHANGELOG.md pyproject.toml src/conclave/__init__.py
   git commit -m "release: vX.Y.Z"
   git push -u origin release/X.Y.Z
   ```

4. **Open and merge the release PR.** Wait for every required CI check, merge through
   branch protection, then fast-forward a clean local `main`. The tag must point to the
   merged release commit, not the pre-merge branch commit.

5. **Tag and push the tag.** (A tag alone does NOT publish anything — it only marks
   the commit. The Release in the next step is what triggers the workflow.)
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. **Create the GitHub Release.** This is the trigger.
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
   ```
   or use the GitHub UI: **Releases → Draft a new release → choose tag `vX.Y.Z` →
   Publish release**.

   Publishing the Release fires `release.yml`, which:
   - **build** — builds the sdist + wheel with `python -m build` and uploads them as
     workflow artifacts so publish + sign use the exact same bytes;
   - **pypi-publish** — publishes those artifacts to PyPI via OIDC Trusted
     Publishing (no token), with PEP 740 attestations attached. This **fails closed**
     if the Trusted Publisher from section 0 is not yet configured;
   - **sign** — signs the sdist + wheel with Sigstore keyless and attaches the
     `.sigstore` bundle(s) to the GitHub Release assets.

---

## 2. Post-release verification

1. **Install from PyPI** (give the CDN a minute):
   ```bash
   pip install conclave-cli
   conclave --help
   conclave providers   # the version is printed in this command's footer
   python -c "import conclave; print(conclave.__version__)"
   ```
   The install name is `conclave-cli`, the command is `conclave`, the import is
   `conclave`. The `python -c` line must print `X.Y.Z` (there is no `--version`
   flag; the running version is shown in the `conclave providers` footer).
   Remember to bump `__version__` in `src/conclave/__init__.py` to `X.Y.Z` in the
   release commit (step 1.2) alongside `pyproject.toml`.

2. **Verify the Sigstore bundle.** On the GitHub Release page, confirm there is a
   `.sigstore` (bundle) asset next to each `.tar.gz`/`.whl`. The `sign` job already
   self-verified against this workflow's own identity before attaching, but you can
   re-verify any artifact locally:
   ```bash
   pip install sigstore
   sigstore verify identity dist/conclave_cli-X.Y.Z-py3-none-any.whl \
     --bundle conclave_cli-X.Y.Z-py3-none-any.whl.sigstore \
     --cert-identity \
       "https://github.com/DataScience-EngineeringExperts/conclave/.github/workflows/release.yml@refs/tags/vX.Y.Z" \
     --cert-oidc-issuer "https://token.actions.githubusercontent.com"
   ```
   (Download the `.whl` and its `.sigstore` bundle from the Release assets first.)

3. **Confirm the PyPI page.** Visit <https://pypi.org/project/conclave-cli/> and check:
   - version `X.Y.Z` is listed;
   - the project URLs point at `DataScience-EngineeringExperts/conclave`;
   - "Publisher" shows the Trusted Publisher (OIDC), not a token upload;
   - PEP 740 attestations are present (the verified-publish badge).

---

## 3. Rollback / yank

PyPI uploads are **immutable** — you cannot overwrite a published version. If a
release is broken:

- **Yank** the bad version (keeps existing pins working, hides it from new
  installs): on <https://pypi.org/project/conclave-cli/> → **Manage → Releases →
  Options → Yank**. Yanking is reversible.
- **Ship a fix-forward patch release** following section 1 again. This is the
  preferred remedy — never try to re-upload the same version.
- **GitHub Release**: you may delete or edit the GitHub Release and its assets
  freely; that does not affect what is already on PyPI. Re-running the workflow
  against the same version will fail the PyPI publish (duplicate filename), which is
  the correct fail-closed behavior — bump the version instead.

---

## CI security gates (context for releasers)

- **pip-audit** runs in CI (the `audit` job in `.github/workflows/test.yml`) and is
  **fail-closed**: a known vulnerability in any resolved dependency fails CI.
  conclave's dependency surface is tiny (`httpx` plus a few well-maintained libs),
  so false-positive churn is low. If a transitive CVE with no available fix blocks an
  unrelated PR, suppress it narrowly with `pip-audit --ignore-vuln <GHSA/PYSEC id>`
  in the workflow step and leave a tracking note in the PR; remove the suppression
  once a fixed version is available.
- **requirements-dev.lock** is a hash-pinned lockfile of the full dev + runtime tree,
  generated with:
  ```bash
  uv pip compile --universal --generate-hashes --python-version 3.11 \
    --extra dev pyproject.toml -o requirements-dev.lock
  ```
  Regenerate it whenever you change dependencies in `pyproject.toml` so reproducible
  installs stay in sync.

---

## Why this design

- **No stored secret.** OIDC Trusted Publishing means GitHub never holds a PyPI
  token; PyPI trusts the workflow identity directly. Same trust model as the keyless
  Sigstore signing job.
- **Signed releases.** From v1.0.0 conclave signs its own release artifacts (the
  `sign` job) with Sigstore keyless, so consumers can verify the wheel they install
  came from this repo's release workflow. PEP 740 attestations on the PyPI upload add
  a second, PyPI-native provenance signal.
- **Explicit gesture.** A pushed tag does nothing; only *publishing a Release* ships.
  That keeps accidental tags from triggering a publish.
