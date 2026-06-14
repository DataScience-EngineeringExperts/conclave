# Changelog

All notable changes to conclave are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-14

First stable release. conclave is feature-complete for its 1.0 scope: a
bring-your-own-keys multi-model council that fans a prompt to N foundation
models concurrently and merges their answers. This release integrates three
release-readiness workstreams — distribution/release engineering, key-leak
hardening + threat model, and synthesizer behavior documentation/versioning —
on top of v0.3.0.

### Added

- **Distribution name.** The package is now published to PyPI as `conclave-cli`
  (`pip install conclave-cli`); the import package, CLI command, and repo all
  stay `conclave`. The bare PyPI name `conclave` is an unrelated project.
- **Release engineering.** OIDC Trusted-Publisher release workflow
  (`.github/workflows/release.yml`) with Sigstore keyless signing and PEP 740
  attestations, inert until a GitHub Release fires and the publisher is
  configured; a hash-pinned dev + runtime lockfile (`requirements-dev.lock`)
  for reproducible installs/CI; and a `RELEASING.md` operator runbook.
- **Supply-chain CI.** A fail-closed `pip-audit` job added to the CI workflow.
- **Threat model.** `SECURITY.md` now carries a BYO-keys threat model and the
  key-handling guarantees consumers can rely on; `.gitleaks.toml` plus a
  dedicated `tests/test_keyleak_audit.py` regression suite guard against
  secret leakage.
- **Versioned synthesis prompt.** The synthesis prompt set is versioned via
  `conclave.prompts.SYNTHESIS_PROMPT_VERSION` and stamped onto every
  `CouncilResult.prompt_version`, so a downstream eval can detect a prompt
  change rather than silently absorb it.

### Changed

- **Key-leak: cause-chain fix.** The originating `httpx` exception is no longer
  attached to `TransportError.__cause__`, closing a path where a verbose
  traceback could surface a key-bearing transport exception.
- **Key-leak: transport-logging guard default-on.** `Council.__init__` now
  installs `conclave.transport.guard_transport_logging()` by default, dropping
  the httpx/httpcore `DEBUG` records that emit the auth header. Callers who
  genuinely need that DEBUG band opt out with
  `Council(..., allow_transport_debug_logging=True)`.
- **Synthesizer: observable degradation.** Synthesizer/judge degradation is
  confirmed (never silent) across synthesize, debate, and the adversarial-judge
  paths: an unkeyed or failed synthesizer surfaces on
  `CouncilResult.synthesis_error` (and `AdversarialResult.verdict_error`,
  mirrored to `synthesis_error`), with no path where synthesis is both absent
  and unexplained.
- **Synthesizer behavior documented.** README gains a "Synthesizer behavior"
  section covering selection precedence (`synthesizer=` arg → config →
  default), observable degradation, and the versioned prompt.

### Scope

- Feature-complete for 1.0: 4 council modes (synthesize / raw / debate /
  adversarial), 9 providers, streaming for synthesize/raw, an optional result
  cache, and debate convergence early-stop.

### Roadmap (post-1.0)

- `vote` mode (council issue #3) — a ranked/tallied decision mode — is
  documented as planned, not shipped.
- A stdio MCP server (council issue #8) is documented as planned; the earlier
  HTTP local-server-mode spike was evaluated and shelved.

## [0.3.0] - 2026-06-08

- Provider-highway refactor: LiteLLM removed in favor of an owned `httpx`
  transport + adapter registry across the (then) 5 providers.
- CI foundation: GitHub Actions matrix, ruff lint/format, coverage floor,
  gitleaks, and branch protection.
- Key-leak fix in `redact()` for custom OpenAI-compatible endpoints; CLI
  exit-code contract and httpx client lifecycle hardening; transport/CLI/logging
  test backfill; first public release with community files.

[1.0.0]: https://github.com/ernestprovo23/conclave/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/ernestprovo23/conclave/releases/tag/v0.3.0
