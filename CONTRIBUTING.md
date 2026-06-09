# Contributing to conclave

Thanks for helping improve conclave — a bring-your-own-keys multi-model council
for the CLI and as a Python library. Contributions are held to a clear bar:
tests green, the **BYO-keys** posture preserved, and the design specs in `docs/`
kept authoritative. This guide gets you set up and explains the rules that keep
the project trustworthy.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Never
report security vulnerabilities as public issues or PRs — follow
[`SECURITY.md`](SECURITY.md).

---

## Dev setup

conclave requires **Python ≥ 3.11**. Use a project-local virtual environment.

```bash
# from a clone of this repo
python -m venv .venv
.venv/bin/pip install -e ".[dev]"

# the CLI is then available as:
.venv/bin/conclave --help
```

`uv` works too if you prefer it (`uv venv .venv && uv pip install --python
.venv/bin/python -e ".[dev]"`).

conclave never needs real API keys to develop or test against — the suite mocks
the httpx transport, so it runs fully offline. You only need real keys (set as
environment variables) to make live calls.

---

## Running tests

```bash
# full suite
.venv/bin/python -m pytest -q

# a single file or test
.venv/bin/python -m pytest tests/test_council.py -q
.venv/bin/python -m pytest tests/test_modes.py::test_debate_dropout -q
```

The suite is **offline and key-free**: `tests/conftest.py` mocks the httpx
transport so no network call or credential is ever made. A change is not done
until `.venv/bin/python -m pytest -q` is fully green. CI runs the same suite on
Python 3.11 / 3.12 / 3.13, plus `ruff` (lint + format) and a coverage floor and
a `gitleaks` secret scan — all five checks must pass to merge.

---

## The BYO-keys contract (read before touching internals)

conclave's core promise is that a user's API keys are **never stored, logged, or
serialized** — they are read from the environment by **variable name**, at call
time, and used only to make the request. If your change touches any of the
following, it must preserve that contract:

- **Key resolution** (`registry.py`, `providers.py`) — keys are looked up by env
  var name at call time. Never attach a key value to an object, a log record, a
  `ModelAnswer`, or a `CouncilResult`. Key **presence** logic may be exposed;
  key **values** never.
- **Error capture** (`providers.py`, `adapters/base.py`) — provider error strings
  can contain credentials echoed back by the upstream API. They MUST pass through
  `redact()` before landing in `ModelAnswer.error` or any log. A change that adds
  a new error path must route it through `redact()`.
- **The `redact()` scrubber** (`adapters/base.py`) — it strips bearer tokens,
  `sk-`/`x-api-key` shapes, and known env-var values. Detection must **never
  weaken on a real secret** to reduce noise. Add coverage; do not drop it.
- **`CouncilResult` / `ModelAnswer` serialization** (`models.py`) — these are the
  stable downstream surface (the mcp-warden dev-time consumer keys on them). No
  field may carry a credential, and the shape must stay backward-compatible
  unless the change is intentional and version-bumped.

Practical rules:

- **No real secrets in the tree, ever.** Tests use obviously-synthetic,
  clearly-fake placeholders. The repo runs a `gitleaks` scan on every push and PR.
- **Adding a provider is one adapter registration.** Follow the existing pattern
  in `adapters/` — register one adapter per provider family. An OpenAI-compatible
  endpoint needs **no code**: it is configured under the `endpoints:` section of
  `~/.conclave/config.yml`. Don't add a bespoke code path for something the
  config seam already covers.
- **`call_model` never raises.** The whole provider highway is partial-failure
  resilient — a member that errors is captured as a failed `ModelAnswer`, not an
  exception that aborts the council. Preserve that.

---

## The PDD is the source of truth

`docs/PRODUCT_DESIGN_DOCUMENT.md` is the **canonical** product and architecture
spec; the 3-core docs (`README.md`, `SYSTEM_CONTEXT_DIAGRAM.md`,
`DOCUMENTATION_INDEX.md`) sit on top of it. When docs disagree, the PDD wins, and
your PR must reconcile them. A behavior or surface change updates the relevant doc
in the same PR.

---

## Proposing a feature

Modes, providers, and output options are welcome. The path that gets a change
merged:

1. **Open an issue first** describing the use case and the proposed surface
   (CLI flag / library API / config key). Roadmap items live as labeled issues.
2. **Check the PDD.** If your change shifts scope, non-goals, or the provider bar
   (§12), say how — the PDD decision history is authoritative.
3. **Implement** following existing patterns (the provider highway, the mode
   orchestration in `modes.py`, the `Council` API in `council.py`).
4. **Test** it: cover the happy path, partial failure, and — for anything in the
   key/error path — that no credential can leak.

---

## Pull request expectations

Before you open a PR, confirm:

- [ ] **Tests pass**: `.venv/bin/python -m pytest -q` is green.
- [ ] **BYO-keys preserved.** No key value is stored, logged, or serialized; new
      error paths route through `redact()`.
- [ ] **No secrets.** No real credentials anywhere in the tree; fixtures use
      obviously-fake placeholders. `gitleaks` is clean.
- [ ] **Lint clean.** `ruff check` and `ruff format --check` pass.
- [ ] **Docs in sync.** README/CLI reference updated if user-facing behavior or
      flags changed; the PDD and 3-core docs stay accurate.
- [ ] **Scoped commits** with a clear message describing the *why*.

Thank you for keeping the council honest.
