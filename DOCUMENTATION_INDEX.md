# conclave — Documentation Index

Master index for conclave's documentation. conclave follows the **3-Core Documentation
Rule**: every project maintains exactly three core docs — an overview (README), a system
context diagram, and this index linking everything together. The Product Design Document is
the canonical authority spec on top of those.

- **Repo:** `/Users/ernestprovo/dev/conclave/`
- **Version:** 0.1.0 · **License:** MIT
- **Last updated:** 2026-06-07

---

## Core documentation

| # | Doc | Path | Purpose |
|---|-----|------|---------|
| 1 | **README** (project overview) | [`README.md`](README.md) | What conclave is, install, BYO-keys, CLI + library quickstart, config, test. The fast on-ramp. |
| 2 | **System Context Diagram** | [`SYSTEM_CONTEXT_DIAGRAM.md`](SYSTEM_CONTEXT_DIAGRAM.md) | Mermaid system-context view: user/consumer → CLI/library → Council → provider highway (httpx transport + adapter registry) → 5 providers; config + env-var key inputs; custom OpenAI-compatible endpoints; mcp-warden as dev-time consumer. |
| 3 | **Documentation Index** | [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) | This file. Master map of all docs + source layout. |

## Authority spec

| Doc | Path | Purpose |
|-----|------|---------|
| **Product Design Document** | [`docs/PRODUCT_DESIGN_DOCUMENT.md`](docs/PRODUCT_DESIGN_DOCUMENT.md) | **Canonical** product spec and roadmap. Problem & vision, personas, BYO-keys/key-handling security, council modes (synthesize/raw/debate/adversarial=built; vote=roadmap), provider matrix, architecture, scope + non-goals, roadmap, the mcp-warden dev-time boundary, licensing & positioning, open questions. **When docs disagree, the PDD wins.** |

---

## Source layout

Package root: `src/conclave/` (installed as the `conclave` package; console script
`conclave = conclave.cli:app`).

| Module | Path | Responsibility |
|--------|------|----------------|
| Package API | [`src/conclave/__init__.py`](src/conclave/__init__.py) | Public exports: `Council`, `CouncilResult`, `ModelAnswer`, `TokenUsage`, `DebateRound`, `AdversarialResult`, `ConclaveConfig`, `load_config`, `__version__`. |
| Council | [`src/conclave/council.py`](src/conclave/council.py) | Primary importable entry point. Reusable primitives (`fan_out`, `synthesize_blocks`) + the public mode API (`ask`/`debate`/`adversarial`, async + sync). |
| Modes | [`src/conclave/modes.py`](src/conclave/modes.py) | `debate` (multi-round, anonymized peers, drop-out) and `adversarial` (propose → refute → verdict) orchestration, built on `Council.fan_out`/`synthesize_blocks`. |
| Prompts | [`src/conclave/prompts.py`](src/conclave/prompts.py) | Role/template strings for debate + adversarial and the anonymized peer-block builder. |
| Providers | [`src/conclave/providers.py`](src/conclave/providers.py) | Single async `call_model` path: resolves the adapter, reads the key by name at call time, calls transport, parses; latency/usage/redacted-error capture; never raises. |
| Transport | [`src/conclave/transport.py`](src/conclave/transport.py) | `post_json` — the single async httpx network boundary for the whole provider highway. |
| Adapter registry | [`src/conclave/adapters/__init__.py`](src/conclave/adapters/__init__.py) | `resolve_adapter(model_id, config)` — provider registry + **extension seam** (one registration per family; config-only for OpenAI-compatible endpoints). |
| Adapter base | [`src/conclave/adapters/base.py`](src/conclave/adapters/base.py) | `ProviderAdapter` protocol, `ProviderError`, and `redact()` (error-string secret scrubber). |
| OpenAI-compat adapter | [`src/conclave/adapters/openai_compat.py`](src/conclave/adapters/openai_compat.py) | `OpenAICompatAdapter` — openai/xai/perplexity + custom OpenAI-compatible endpoints. |
| Anthropic adapter | [`src/conclave/adapters/anthropic.py`](src/conclave/adapters/anthropic.py) | `AnthropicAdapter` — native `/v1/messages`, system-prompt hoist, required `max_tokens`. |
| Gemini adapter | [`src/conclave/adapters/gemini.py`](src/conclave/adapters/gemini.py) | `GeminiAdapter` — native `generateContent`, OpenAI-role mapping, `usageMetadata`. |
| Registry | [`src/conclave/registry.py`](src/conclave/registry.py) | Friendly-name → model-id defaults; provider → env-var mapping; key **presence** logic (never values). |
| Config | [`src/conclave/config.py`](src/conclave/config.py) | Loads/merges `~/.conclave/config.yml` over defaults; resolves model ids and named/CSV councils; parses the `endpoints:` section (custom OpenAI-compatible providers). |
| Models | [`src/conclave/models.py`](src/conclave/models.py) | Pydantic result contract: `TokenUsage`, `ModelAnswer`, `DebateRound`, `AdversarialResult`, `CouncilResult` (`mode`/`rounds`/`adversarial`). Stable downstream surface. |
| CLI | [`src/conclave/cli.py`](src/conclave/cli.py) | `conclave ask` (synthesize/raw/debate/adversarial; `--rounds`/`--proposer`) + `conclave providers`; rich panels and `--json`; never prints key values. |
| Logging | [`src/conclave/logging.py`](src/conclave/logging.py) | Logger factory; stderr; verbosity via `CONCLAVE_LOG_LEVEL` (default `WARNING`). |

## Tests

| File | Path | Covers |
|------|------|--------|
| Council tests | [`tests/test_council.py`](tests/test_council.py) | Fan-out, partial failure, synthesis behavior. |
| Modes tests | [`tests/test_modes.py`](tests/test_modes.py) | Debate multi-round flow, mid-round drop-out, peer anonymization; adversarial proposer/critic/verdict, proposal/critic failure paths, no-key judge, sync wrappers. |
| Registry/config tests | [`tests/test_registry_config.py`](tests/test_registry_config.py) | Name resolution, key-presence logic, config merge. |
| Fixtures | [`tests/conftest.py`](tests/conftest.py) | Shared fixtures; mocks the httpx transport so the suite needs no network and no API keys. |

Run: `pytest` (config in `pyproject.toml`, `asyncio_mode = "auto"`).

## Project files

| File | Path | Purpose |
|------|------|---------|
| Packaging | [`pyproject.toml`](pyproject.toml) | hatchling build, deps (httpx, pydantic, rich, typer, pyyaml — no LLM SDK), dev extras, console script, pytest config. License: MIT. |
| License | [`LICENSE`](LICENSE) | MIT License. Copyright (c) 2026 Ernest Provo. Matches the `pyproject.toml` license field. |

---

## Related projects

| Project | Relationship |
|---------|--------------|
| **mcp-warden** (sibling) | Imports conclave as a **DEV-TIME** dependency (design review, taxonomy labeling). **NOT** a runtime dependency — security findings need determinism; a stochastic council is the wrong tool for runtime adjudication. See PDD §10. |

## Documentation conventions

- Exactly **3 core docs** (README, System Context Diagram, this index); the PDD is the
  authority spec layered on top.
- Each doc stays **under 500 lines**; overflow is archived to dated/topic files and linked
  from here.
- Prefer **editing existing docs** over creating new ones. Check this index before adding
  any new document.
- The **PDD is canonical**; on any conflict, defer to `docs/PRODUCT_DESIGN_DOCUMENT.md`.
