# Releasing conclave

This is the operator runbook for cutting a release of **conclave**.

Three names matter and they are deliberately different:

| Thing | Value |
|-------|-------|
| PyPI distribution name (what `pip install` uses) | `conclave-cli` |
| CLI command (what users type) | `conclave` |
| Import package (what you `import`) | `conclave` |
| GitHub repository | `ernestprovo23/conclave` |

Install is therefore `pip install conclave-cli`, but the command stays `conclave`
and the import stays `from conclave import Council`. The PyPI name `conclave` is an
**unrelated** package by another author (a blockchain client) — that is why the
distribution name is `conclave-cli`.

The publish + signing automation lives in
[`.github/workflows/release.yml`](.github/workflows/release.yml). That workflow is
**inert until configured**: it only fires when a GitHub *Release* is published, and
the publish job only succeeds once the one-time PyPI Trusted Publisher below exists.

---

## 0. One-time PyPI setup (do this ONCE, before the first release)

The workflow publishes via **OIDC Trusted Publishing** — there is no API token and
no secret stored in GitHub. Instead, PyPI is told to trust releases that come from
this exact repo + workflow. Configure the publisher *before* the first release so
the very first upload is already OIDC-published.

### Recommended path — "pending publisher" (zero prior upload required)

1. Log in to <https://pypi.org> as the account that will own `conclave-cli`.
2. Go to **Account → Publishing** (<https://pypi.org/manage/account/publishing/>).
3. Under **Add a new pending publisher**, fill in **exactly**:
   - **PyPI Project Name**: `conclave-cli`
   - **Owner**: `ernestprovo23`
   - **Repository name**: `conclave`
   - **Workflow name**: `release.yml`
   - **Environment name**: *(leave blank — the workflow does not use a GitHub
     deployment environment; if you later add one, set it here and add an
     `environment:` block to the `pypi-publish` job)*
4. Save. PyPI now reserves the project name `conclave-cli` and will create it on the
   first successful OIDC upload from `release.yml`.

A "pending publisher" reserves the name and lets the FIRST release be OIDC-published
— no manual upload, no token ever. This is the clean path for conclave: there is no
prior token-publish history, so the supply chain is OIDC-only from release #1.

### Alternative path — manual first upload, then configure

If you would rather seed the project manually first:

1. Build locally: `python -m build` (produces `dist/*.tar.gz` + `dist/*.whl`).
2. `twine upload dist/*` with a temporary PyPI API token (creates `conclave-cli`).
3. Then go to **Manage project → Publishing** on the new `conclave-cli` project and
   add the Trusted Publisher with the same owner/repo/workflow values as above.
4. Revoke the temporary token.

> Prefer the pending-publisher path. It avoids ever minting a long-lived token and
> keeps the entire supply chain OIDC-only from release #1.

---

## 1. Cut a release

Do this on a clean checkout of `main` with all v1 PRs merged.

1. **Update the changelog.** In [`CHANGELOG.md`](CHANGELOG.md), move the
   `## [Unreleased]` entries under a new `## [1.0.0] - <YYYY-MM-DD>` heading with
   today's date. Leave a fresh empty `## [Unreleased]` section above it.

2. **Bump the version in BOTH places.**
   - In [`pyproject.toml`](pyproject.toml), set `[project] version = "1.0.0"`.
   - In [`src/conclave/__init__.py`](src/conclave/__init__.py), set
     `__version__ = "1.0.0"`.
   (The distribution name `conclave-cli` is already set — do **not** change it.)

3. **Commit.**
   ```bash
   git add CHANGELOG.md pyproject.toml src/conclave/__init__.py
   git commit -m "release: v1.0.0"
   git push origin main
   ```

4. **Tag and push the tag.** (A tag alone does NOT publish anything — it only marks
   the commit. The Release in the next step is what triggers the workflow.)
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

5. **Create the GitHub Release.** This is the trigger.
   ```bash
   gh release create v1.0.0 \
     --title "v1.0.0" \
     --notes-file <(awk '/## \[1.0.0\]/{f=1} /## \[0\./{if(f)exit} f' CHANGELOG.md)
   ```
   or use the GitHub UI: **Releases → Draft a new release → choose tag `v1.0.0` →
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
   `conclave`. The `python -c` line must print `1.0.0` (there is no `--version`
   flag; the running version is shown in the `conclave providers` footer).
   Remember to bump `__version__` in `src/conclave/__init__.py` to `1.0.0` in the
   release commit (step 1.2) alongside `pyproject.toml`.

2. **Verify the Sigstore bundle.** On the GitHub Release page, confirm there is a
   `.sigstore` (bundle) asset next to each `.tar.gz`/`.whl`. The `sign` job already
   self-verified against this workflow's own identity before attaching, but you can
   re-verify any artifact locally:
   ```bash
   pip install sigstore
   sigstore verify identity dist/conclave_cli-1.0.0-py3-none-any.whl \
     --bundle conclave_cli-1.0.0-py3-none-any.whl.sigstore \
     --cert-identity \
       "https://github.com/ernestprovo23/conclave/.github/workflows/release.yml@refs/tags/v1.0.0" \
     --cert-oidc-issuer "https://token.actions.githubusercontent.com"
   ```
   (Download the `.whl` and its `.sigstore` bundle from the Release assets first.)

3. **Confirm the PyPI page.** Visit <https://pypi.org/project/conclave-cli/> and check:
   - version `1.0.0` is listed;
   - the project URLs (homepage / repository) point at `ernestprovo23/conclave`;
   - "Publisher" shows the Trusted Publisher (OIDC), not a token upload;
   - PEP 740 attestations are present (the verified-publish badge).

---

## 3. Rollback / yank

PyPI uploads are **immutable** — you cannot overwrite a published version. If a
release is broken:

- **Yank** the bad version (keeps existing pins working, hides it from new
  installs): on <https://pypi.org/project/conclave-cli/> → **Manage → Releases →
  Options → Yank**. Yanking is reversible.
- **Ship a fix-forward release** (`1.0.1`) following section 1 again. This is the
  preferred remedy — never try to re-upload `1.0.0`.
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
