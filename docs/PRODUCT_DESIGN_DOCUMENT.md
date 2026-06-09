# conclave — Product Design Document

> **Status:** v0.1.0 shipped; v0.2 (`debate` + `adversarial` modes) built; v0.3 dependency
> refactor landed — **LiteLLM removed**, replaced by conclave's own httpx-based provider
> highway (see §6). This is the **canonical authority document** for
> conclave's product scope, design, and roadmap. When this document and any other doc
> disagree, this document wins. Code is the source of truth for *current behavior*; this
> document marks anything not yet in code as **Roadmap**.

- **Repo:** `/Users/ernestprovo/dev/conclave/`
- **License:** MIT
- **Author:** Data Science & Engineering Experts, Inc. (DSE)
- **Last updated:** 2026-06-07

---

## 1. Problem & Vision

### Problem
Single foundation models are confidently wrong in ways that are hard to detect from inside
a single model. Different models have different training data, failure modes, and blind
spots. Today, if you want a "second opinion" you either:

- paste the same prompt into 3-5 web UIs by hand and eyeball the differences, or
- adopt a heavyweight multi-agent framework (LangGraph, AutoGen) and write orchestration
  graph code, accept a runtime, and learn its abstractions, or
- route everything through a hosted aggregator that takes a margin on your tokens and sees
  your prompts.

None of these is a lightweight, scriptable, **own-your-keys** primitive for "ask N models
the same thing and reconcile the answers."

### Vision
conclave is a small, sharp tool: **a council of foundation models you can call from one
CLI command or one Python import.** Fan a prompt out to several models concurrently — each
through *your own* API keys, no markup, no middleman — and aggregate the answers. The
v0.1 aggregation is a **synthesizer** that merges raw answers into one consolidated
response. v0.2 adds a small set of **council modes** — **debate** (multi-round) and
**adversarial** (propose → refute → verdict) — that turn a flat panel of opinions into a
structured deliberation; `vote` remains on the roadmap.

conclave's first real use was an **adversarial design review**: a council of Grok, Gemini,
Perplexity, and Claude critiquing a security-tool strategy and catching flaws a single
model missed. That origin is why the adversarial and debate modes are first-class — they
are now built, not a bolt-on. The product is opinionated about staying lightweight: a **library-first
primitive with structured results**, not an agent framework.

---

## 2. Target Users & Personas

| Persona | Who | What they want from conclave |
|---------|-----|------------------------------|
| **The skeptical engineer** | Senior dev / architect making a consequential technical call | A fast second/third opinion across models, with raw per-model answers visible so they can judge disagreement themselves. Uses the CLI ad hoc. |
| **The library integrator** | Developer building a tool that needs multi-model input at *design/eval time* | `from conclave import Council`, structured `CouncilResult` (latency, token usage, per-model errors), partial-failure resilience. The primary downstream example is **mcp-warden** (see §10). |
| **The researcher / evaluator** | Someone comparing model behavior on a prompt set | Deterministic structure around answers, JSON output (`--json`) for downstream analysis, per-model latency and token accounting. |
| **The cost-conscious power user** | Heavy LLM user who already pays each provider directly | BYO-keys with **no markup** and **no third party seeing the prompt**. conclave is a thin local orchestrator over the user's own accounts. |

Non-personas (explicitly *not* who we build for): teams wanting a hosted multi-agent SaaS,
or anyone needing a deterministic runtime adjudicator (see Non-Goals, §8, and the
mcp-warden boundary, §10).

---

## 3. BYO-keys Model & Key-Handling Security

conclave is **bring-your-own-keys** by design. This is both a positioning choice (no
markup, no middleman) and a security property.

**Key-handling invariants (enforced in code today):**

1. **Keys are referenced by env-var NAME only, never by value.** The provider registry
   (`registry.py`) maps each provider prefix to the env var(s) that satisfy it
   (e.g. `xai → ["XAI_API_KEY"]`, `gemini → ["GEMINI_API_KEY", "GOOGLE_API_KEY"]`). The
   functions `key_present()` and `key_source()` answer *"is a key set?"* and *"which
   variable name holds it?"* — they **never read, return, or log the value.**
2. **conclave never stores keys.** Config (`~/.conclave/config.yml`) references providers
   by friendly name and model id only. There is no field in `ConclaveConfig` that can hold
   a secret. The example config in `config.py` is keys-free by construction.
3. **The key value is read by name, at call time, and is transient in-process**
   (`providers.py`). `call_model` reads the relevant env var *by name* at call time, passes
   the value to the resolved adapter to build the auth header, and the httpx transport sends
   it. The value is **never stored on any object** (not config, not the registry, not
   `ModelAnswer`, not `CouncilResult`), **never logged, never serialized, and scrubbed from
   error strings** via `redact()` (`adapters/base.py`). It never transits a conclave data
   structure. (Honest framing: the value *is* read in-process to authenticate — conclave
   does not magically avoid touching it — but its lifetime is a single request and it leaves
   no trace on any persisted or returned object.)
4. **Secrets never reach serialized output.** `CouncilResult.model_dump()` (used by
   `--json`) contains prompts, answers, model ids, latency, token usage, and error strings
   — no key material. The `providers` CLI command shows a check/cross and the env-var
   *name*, never the value.
5. **Missing keys degrade gracefully, they don't crash.** A requested member whose key is
   absent is skipped with a warning and recorded in `CouncilResult.skipped`. Unknown
   providers (no static env-var mapping) are *not* pre-emptively skipped — the live call is
   attempted and any auth error is captured as a `ModelAnswer.error`.

**Residual considerations (worth a user's awareness):** error strings captured from a
provider could in principle echo a key fragment. As of the v0.3 refactor this path is
**hardened**: every provider/transport error is passed through `redact()` (`adapters/base.py`)
before it lands in `ModelAnswer.error`, scrubbing known key material from the string. The
residual risk is now limited to a provider emitting a secret in a shape `redact()` does not
recognize; the verbatim-passthrough gap that previously existed is closed. (Was §9 hardening
item 7 — now landed; see §9.)

---

## 4. Council Modes & Consensus Algorithms

A **council mode** is the algorithm used to turn N independent model calls into a single
useful output. v0.1 ships one true mode plus a pass-through.

| Mode | Status | What it does |
|------|--------|--------------|
| **synthesize** | **BUILT (v0.1)** | Fan out concurrently → collect each raw answer → a **synthesizer model** merges them into one consolidated answer, reconciling agreement, adjudicating disagreement, and flagging clearly-wrong answers. The synthesizer is instructed to rely only on the provided answers and not invent a model's position. |
| **raw** | **BUILT (v0.1)** | Fan out and return every member's raw answer with no synthesis. Not a deliberation mode — it is "synthesize off." Exposed as `--mode raw` / `ask(..., synthesize=False)`. |
| **debate** | **BUILT (v0.2)** | N rounds (`--rounds`, default 2). Round 1 is an independent fan-out; rounds 2..N show each member its peers' **anonymized** prior-round answers (`Model A/B/C`) and ask it to revise or defend. A member that errors in a round drops out of later rounds; the debate continues with survivors. The synthesizer consolidates the final round. Exposed as `--mode debate` / `Council.debate()` / `debate_sync()`. |
| **adversarial** | **BUILT (v0.2)** | Structured propose → refute → verdict. A `--proposer` (default: first member) answers; the remaining members are CRITICS explicitly prompted to refute it; the synthesizer acts as JUDGE, weighing proposal vs. critiques and issuing a verdict + strengthened answer. This is the mode conclave's origin story (the security design review) exercised by hand. Exposed as `--mode adversarial` / `Council.adversarial()` / `adversarial_sync()`. |
| **vote** | **ROADMAP (v0.3+)** | Structured majority. Each model answers a constrained question; conclave tallies a structured vote and reports the majority plus the split. |

### Synthesize algorithm (as built)
1. Resolve requested friendly names to model ids via config.
2. Partition members into *available* (key present) and *skipped* (no key).
3. Fan out the prompt to all available members concurrently (`asyncio.gather`,
   `return_exceptions=True` as a belt-and-suspenders guard).
4. Each call returns a structured `ModelAnswer` (answer **or** error — `call_model` never
   raises for provider failures), so partial results always survive.
5. If `synthesize=True` and at least one member succeeded, build a synthesis prompt that
   embeds the original prompt and each successful answer, and call the synthesizer model.
6. If the synthesizer has no key, or no member succeeded, set `synthesis_error` and return
   raw answers only. A run with zero available members returns an empty-answer result
   rather than raising.

**Consensus note:** synthesize is a *generative* reconciliation, not a deterministic vote.
It is inherently stochastic. This matters for the mcp-warden boundary in §10.

### Debate algorithm (as built)
1. Resolve + partition members (same as synthesize). Assign each member a stable
   position-based letter label (`Model A`, `Model B`, …).
2. **Round 1:** independent fan-out of the bare prompt (reuses `Council.fan_out`).
3. **Rounds 2..N:** each surviving member is shown its own prior answer (told which
   letter is "you") plus its peers' prior answers, **anonymized by letter, not brand**,
   and asked to revise or defend. Anonymization reduces brand-bias (a model deferring to
   or attacking another *by name*) while preserving the cross-pollination that makes
   debate useful. The answer *body* is passed verbatim; only the attribution is relabeled.
4. **Drop-out:** a member that errors in a round is removed from subsequent rounds; the
   debate continues with survivors. If every member fails a round, the debate ends there.
5. The synthesizer consolidates the final round's surviving answers (same call path as
   synthesize, with a debate-specific system prompt).

### Adversarial algorithm (as built)
1. Resolve + partition members. Pick the proposer: `--proposer` if given, else the first
   requested member; if that member has no key, fall back to the first available member.
2. **Propose:** the proposer answers the prompt (single-member `fan_out`).
3. **Refute:** every other available member is a CRITIC, explicitly prompted to find the
   strongest flaws in the proposal (not to agree). One critic failing never aborts the run.
   If the proposal itself failed, critics are skipped.
4. **Verdict:** the synthesizer acts as JUDGE — given the prompt, proposal, and critiques —
   accepting correct critiques, rejecting overstated ones, and issuing a verdict plus the
   strengthened final answer.

### Result-model extension (backward-compatible)
The deliberation modes extend `CouncilResult` **without breaking** synthesize/raw consumers:
- New `mode` field (`"synthesize" | "raw" | "debate" | "adversarial"`).
- New `rounds: list[DebateRound]` (debate) and `adversarial: AdversarialResult | None`.
- For **debate**, the final round is mirrored into the existing `answers`, and the
  consolidated answer into the existing `synthesis`. For **adversarial**, the proposal +
  critiques populate `answers` and the verdict mirrors into `synthesis`. Any existing
  consumer that reads `answers`/`synthesis`/`successful_answers` keeps working unchanged;
  new consumers read `rounds`/`adversarial` for the full structure. All fields are
  keys-free and serialize cleanly via `model_dump()` (`--json`).

---

## 5. Provider Support Matrix

Friendly names, default model ids, and the env var(s) that satisfy each. Defaults
live in `registry.DEFAULT_MODELS` / `registry.PROVIDER_ENV_VARS` and are overridable via
`~/.conclave/config.yml`.

| Provider | Friendly name | Default model id | Env var(s) (first present wins) | Status |
|----------|---------------|------------------|---------------------------------|--------|
| xAI | `grok` | `xai/grok-4.3` | `XAI_API_KEY` | BUILT |
| Google | `gemini` | `gemini/gemini-2.5-pro` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | BUILT |
| Anthropic | `claude` | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | BUILT |
| Perplexity | `perplexity` | `perplexity/sonar-pro` | `PERPLEXITY_API_KEY` | BUILT |
| OpenAI | `openai` | `openai/gpt-4.1` | `OPENAI_API_KEY` | BUILT |
| *(any provider known to an adapter)* | *raw id as name* | *passed through verbatim* | *adapter's provider env var* | SUPPORTED (untyped) |
| *(any OpenAI-compatible endpoint)* | *config `endpoints:` entry* | *your model id* | *the endpoint's `api_key_env`* | SUPPORTED (config-only) |

**Default synthesizer:** `claude`. **Default council** (when none is configured): all five
known providers. Because `resolve_model_id()` passes unknown names through verbatim, a user
can already add a council member by raw id whose prefix an adapter recognizes (e.g.
`openai/gpt-4o`) without a code change; it just won't have a static key-presence check
(treated as "attempt and catch"). A wholly new OpenAI-compatible vendor needs no code at all
— a `config.yml` `endpoints:` entry (base URL + `api_key_env`) makes it a first-class member
via `OpenAICompatAdapter` (see §6).

Expanding the *first-class* provider list (more friendly-name defaults) is Roadmap, §9.

---

## 6. Architecture

conclave is a thin, layered orchestrator over its **own provider highway** — an httpx
transport behind a per-provider adapter registry, with **no LLM-SDK dependency**. Each
module has one job; the data models are the stable contract between layers and downstream
consumers.

```
CLI (cli.py, typer+rich)   Library (from conclave import Council)
            \                         /
             v                       v
                 Council (council.py)
   fan_out · synthesize_blocks · skip-no-key · partial-results · synthesis
                 |                              |
   modes.py (debate · adversarial)      prompts.py (role templates)
                 |
              call_model (providers.py)
   resolve adapter · read key by name at call time · latency · usage · redacted error
                          |
              resolve_adapter (adapters/__init__.py)
        OpenAICompatAdapter   AnthropicAdapter   GeminiAdapter
         (openai·xai·         (/v1/messages)     (generateContent)
          perplexity·custom)         |                |
                          \          |               /
                           v         v              v
                        transport.post_json (single httpx async boundary)
                          |
        xai · gemini · anthropic · perplexity · openai · (custom OpenAI-compatible)
```

**Module responsibilities (ground truth):**

| Module | Responsibility |
|--------|----------------|
| `council.py` | `Council` — primary importable entry point. Resolves names, partitions members, and exposes two reusable primitives: `fan_out` (the single concurrent + partial-failure call loop) and `synthesize_blocks` (the single synthesizer/judge call path). Hosts the public mode API: `ask`/`ask_sync` (synthesize/raw), `debate`/`debate_sync`, `adversarial`/`adversarial_sync`. Sync wrappers guard against being called inside a running event loop. |
| `modes.py` | Deliberation orchestration: `run_debate` (multi-round, anonymized peers, drop-out) and `run_adversarial` (propose → refute → verdict). Built entirely on `Council.fan_out` + `synthesize_blocks` — no duplicated concurrency or synthesizer code. |
| `prompts.py` | Role/template strings for debate and adversarial (member, critic, judge, debate-final system prompts) and the anonymized peer-block builder. Separates *what each role is told* from *when to call whom*. |
| `providers.py` | `call_model` — the single async call path. Resolves the adapter for a model id, reads the key value *by name at call time*, calls the adapter+transport, parses the reply, and captures latency, token usage, and any (redacted) error into a `ModelAnswer`; never raises for provider-side failures. Signature and never-raises contract unchanged from v0.1/v0.2. |
| `transport.py` | The single async network boundary: `post_json` — one httpx call site for the whole highway. Nothing else in conclave touches the network. |
| `adapters/__init__.py` | `resolve_adapter(model_id, config)` — the provider registry and **extension seam**. Maps a model-id prefix (or a config `endpoints:` entry) to the adapter that serves it. Adding a provider family = one registration here; adding an OpenAI-compatible endpoint = config-only. |
| `adapters/base.py` | The `ProviderAdapter` protocol, `ProviderError`, and `redact()` — the secret-scrubber applied to every error string before it reaches `ModelAnswer.error`. |
| `adapters/openai_compat.py` | `OpenAICompatAdapter` — serves openai / xai / perplexity and any custom OpenAI-compatible endpoint. Per-provider full completions URL (note: Perplexity has no `/v1` segment). |
| `adapters/anthropic.py` | `AnthropicAdapter` — native `POST /v1/messages` (`x-api-key` + `anthropic-version`); system prompt hoisted to the top-level `system` field; `max_tokens` required (default 4096); parses `content[].text` and `input_tokens`/`output_tokens`. |
| `adapters/gemini.py` | `GeminiAdapter` — native `generateContent` (`x-goog-api-key`); OpenAI roles mapped (assistant→model), `systemInstruction` hoisted, `generationConfig.{temperature,maxOutputTokens}`; parses `usageMetadata`. |
| `registry.py` | Single source of truth for friendly-name → model-id defaults and provider → env-var mapping. Key *presence* logic only — never key values. |
| `config.py` | Loads/merges `~/.conclave/config.yml` over built-in defaults (`CONCLAVE_CONFIG` env var overrides path). Resolves model ids and named/CSV councils, and parses the `endpoints:` section (custom OpenAI-compatible providers). Keys-free by construction (endpoints carry a URL + key-env-var *name*, never a value). |
| `models.py` | Pydantic contract: `TokenUsage`, `ModelAnswer`, `CouncilResult` (+ `successful_answers`/`failed_answers`/`ok` helpers). The stable importable surface for downstream consumers. |
| `cli.py` | `conclave ask` and `conclave providers`. Rich panels for humans, `--json` for machines. Never prints key values. |
| `logging.py` | One logger factory, stderr, verbosity via `CONCLAVE_LOG_LEVEL` (default `WARNING`). |

**Key design properties:**
- **Library-first.** The CLI is a thin shell over the same `Council` any consumer imports.
- **Partial-failure resilience is structural,** not optional: failures become data
  (`ModelAnswer.error`), never exceptions that abort the run.
- **Structured results.** Every run yields per-model latency, token usage, and error
  capture — the differentiator vs. text-only multi-model tools.
- **Stable contract.** `models.py` field names are intentionally stable for downstream
  consumers (e.g. mcp-warden) to depend on.

**Provider highway & extension model (v0.3):** conclave owns its provider layer instead of
depending on an LLM SDK. `resolve_adapter` (`adapters/__init__.py`) maps each model id to a
`ProviderAdapter`; every adapter serializes its provider's request shape and hands it to the
**one** network call site, `transport.post_json` (`transport.py`). Three adapters cover the
five first-class providers: `OpenAICompatAdapter` (openai/xai/perplexity, OpenAI-style
`/chat/completions`), `AnthropicAdapter` (native `/v1/messages`, system-prompt hoist,
required `max_tokens`), and `GeminiAdapter` (native `generateContent`, role mapping,
`usageMetadata`). Extension is deliberately cheap: a **new provider family** is one
registration in `adapters/__init__.py`; a **new OpenAI-compatible endpoint** (local server,
gateway, another vendor) is **config-only** — a `~/.conclave/config.yml` `endpoints:` entry
giving a base URL and the env-var *name* of its key, served by `OpenAICompatAdapter` with no
code change. The key value is read by name at call time, passed to the adapter to build the
auth header, and never stored, logged, or serialized (see §3); error strings are scrubbed by
`redact()` (`adapters/base.py`).

**Stack:** Python 3.11+, `httpx` (async transport — the only network dependency), `asyncio`
(concurrency), Pydantic v2 (config + results), Typer + Rich (CLI), PyYAML (config). **No
LLM-SDK dependency.** Packaged with hatchling; console script `conclave = conclave.cli:app`.

---

## 7. Scope

**Shipped in v0.1:**
- `synthesize` and `raw` modes (fan-out, partial results, synthesizer merge).
- 5 first-class providers + pass-through for any model id an adapter recognizes.
- BYO-keys via env-var name only; graceful skip of missing-key members.
- Concurrent fan-out with per-call timeout and temperature.
- Structured `CouncilResult` with latency, token usage, per-model error capture.
- CLI (`ask`, `providers`) with human and `--json` output.
- Config file: named models, named councils, default synthesizer.
- Importable library API with sync and async entry points.
- Test suite that mocks the httpx transport (no network, no keys required).

**Added in v0.2:**
- `debate` mode — multi-round (`--rounds`), anonymized peers, per-member drop-out on
  failure, final synthesis. Per-round structure preserved in `CouncilResult.rounds`.
- `adversarial` mode — `--proposer` → critics refute → synthesizer judges. Structure in
  `CouncilResult.adversarial` (proposal / critiques / verdict).
- Backward-compatible `CouncilResult` extension (`mode`, `rounds`, `adversarial`); existing
  `answers`/`synthesis` consumers unaffected.
- Both modes exposed on the `Council` library API (async + sync) and the CLI, with rich
  per-round / proposal-critique-verdict rendering and `--json`.

**Added in v0.3 (dependency refactor):**
- **LiteLLM removed.** Replaced by conclave's own provider highway: an `httpx` async
  transport (`transport.py`) behind a per-provider adapter registry (`adapters/`). `httpx`
  is now the only network dependency; there is no LLM-SDK dependency (see §6).
- Three adapters cover the five first-class providers (`OpenAICompatAdapter` for
  openai/xai/perplexity, native `AnthropicAdapter`, native `GeminiAdapter`); `resolve_adapter`
  is the extension seam.
- **Custom OpenAI-compatible endpoints** via a config `endpoints:` section — config-only, no
  code change.
- **Key-leak hardening landed:** provider/transport error strings are scrubbed by `redact()`
  before reaching `ModelAnswer.error` (was Roadmap §9 item 7).
- `call_model`'s signature and never-raises contract are unchanged; the result contract
  (`CouncilResult`/`ModelAnswer`) is unchanged, so existing consumers are unaffected.

---

## 8. Non-Goals (v0.1, and some permanent)

- **Not a runtime adjudicator.** conclave is stochastic; it must not be used as a
  deterministic decision gate. (See mcp-warden boundary, §10.) This is a **permanent**
  non-goal for the synthesize/debate/adversarial modes.
- **Not an agent framework.** No tool-calling graphs, no long-running stateful agents, no
  orchestration DSL. We compete by being *small*. (Permanent.)
- **Not a key manager / secrets vault.** conclave reads env vars; it does not provision,
  rotate, store, or proxy keys. (Permanent.)
- **No hosted/proxied token path.** No conclave-operated endpoint that sees user prompts or
  takes a margin. BYO-keys, direct-to-provider, always. (Permanent.)
- **No persistence/caching of results** in v0.1 (caching is Roadmap, §9).
- **No streaming** in v0.1 (Roadmap, §9).
- **No server mode** in v0.1 (possible Roadmap, §9).
- **No `vote` mode** yet (Roadmap, §9 — flagged for build, not for removal). `debate` and
  `adversarial` shipped in v0.2.

---

## 9. Roadmap (v0.3+, NOT yet built)

Ordered roughly by strategic value to the origin use case and to mcp-warden.
(`adversarial` and `debate` shipped in v0.2 — see §4/§7.)

1. **`vote` mode** — structured majority with reported split. Needs a constrained
   answer schema so votes are comparable.
2. ~~**Debate convergence/stop criteria** — today debate runs a fixed `--rounds`; add optional
   early-stop when answers converge (and a configurable convergence signal).~~ **LANDED**
   (issue #4): opt-in early-stop via `converge_threshold` (config field, `Council.debate`
   param, and `--converge-threshold` / `--converge`/`--no-converge` CLI flags). The signal is
   round-over-round answer stability — per-member `difflib.SequenceMatcher` ratio averaged
   across members — deterministic, stdlib-only, offline-testable. Off by default (`None`),
   preserving fixed-rounds behavior exactly. Recorded on `CouncilResult.converged` +
   `convergence_score`; `converge_threshold` is part of the debate cache key so a converged
   run and a fixed run never collide. Kept in the list, struck through, for traceability.
3. **More first-class providers** — additional friendly-name defaults (e.g. more OpenAI,
   Anthropic, Google, and open-weights endpoints). New OpenAI-compatible vendors are already
   config-only via `endpoints:`; this item is about promoting common ones to typed defaults
   (and adding native adapters where a provider isn't OpenAI-compatible).
4. **Caching** — optional result cache keyed on (prompt, council, mode, model ids) to make
   repeated/eval runs cheap. Must remain off by default and never persist keys.
5. **Streaming** — stream member answers and/or the synthesis to the terminal/library.
6. **Local HTTP/server mode (under evaluation)** — a *local* server for convenience only;
   must not become a hosted token path or violate the no-middleman non-goal.
7. ~~**Key-leak hardening** — scrub/limit provider-originated error strings before they land
   in `ModelAnswer.error`.~~ **LANDED in v0.3** via `redact()` (`adapters/base.py`), applied
   to every provider/transport error before it reaches `ModelAnswer.error` (see §3). Kept in
   the list, struck through, to preserve roadmap traceability rather than deleting it.

**Roadmap discipline:** items are added and reprioritized freely; items are not *removed*
on the strength of a single data point — flag for discussion first. Completed items are
**marked done in place** (struck through with a "LANDED" note), not deleted.

---

## 10. Downstream Boundary: conclave ↔ mcp-warden

**mcp-warden** is a sibling project: an MCP-server security "integrity gateway." It will
**import conclave as a DEV-TIME dependency only** — for things like adversarial design
review of warden's own strategy and taxonomy/label brainstorming during development.

**mcp-warden will NOT use conclave as a RUNTIME dependency.** Security findings require
**determinism and reproducibility**; a stochastic council is the wrong tool for runtime
adjudication of security events. The same property that makes conclave valuable for design
review (diverse, generative, multi-model reconciliation) makes it unsuitable as a runtime
gate. This boundary is deliberate and load-bearing:

| | conclave (this project) | mcp-warden runtime |
|---|---|---|
| Nature | Stochastic, generative, multi-model | Deterministic, reproducible |
| Right use | Design review, eval, taxonomy labeling (dev time) | Runtime security adjudication |
| Dependency direction | — | imports conclave **at dev time only** |

If you find yourself wanting conclave inside mcp-warden's runtime decision path, that is a
design smell — re-read this section.

---

## 11. Licensing & Positioning

**License:** MIT (`pyproject.toml`). Permissive on purpose: conclave is meant to be a
small primitive others embed (starting with mcp-warden).

**Market reality (refreshed 2026-06-08).** Since this section was first written, the
"ask N models and reconcile" category got crowded, and two PyPI packages now occupy
conclave's original niche directly:

- **`llm-council-core`** (`llm-council.dev`) — library-first (`from llm_council import
  Council`), partial-results-on-timeout, direct-provider mode, anonymized peer ranking,
  structured verdicts, and a `doctor` health check reporting `key_source`. **Closest peer.**
- **`the-llm-council`** (PyPI) — library + CLI + agent skill, adversarial critique,
  graceful degradation (retry/fallback/skip), JSON-schema-validated structured output.

The honest consequence: **library-first + structured-result + partial-failure-resilient is
now table-stakes, not a moat.** Debate brand-anonymization (Model A/B/C) is likewise
**commodity** — shipped by `ai-council-mcp`, `cognition-wheel`, and `llm-council-core` —
so we no longer market it as distinctive.

**Positioning vs. prior art:**

| | conclave | `llm-council-core` (closest peer) | `llm-consortium` | LangGraph / AutoGen |
|---|---|---|---|---|
| Primary surface | **Library-first** (CLI is a thin shell) | Library + MCP + HTTP | CLI/plugin first | Framework/runtime first |
| Result shape | **Telemetry-grade typed contract** (`CouncilResult`: per-model latency, token usage, typed error capture — stable for downstream like mcp-warden) | Structured verdicts (content-oriented) | XML envelope / JSON dump (content-oriented) | Rich but framework-coupled |
| Failure model | **Partial-failure resilient by construction** (failures become data) | Partial results on timeout | Plugin-dependent | App must handle |
| Provider access | **Direct-to-provider only — no aggregator** | OpenRouter default; direct mode optional | via `llm` plugins | SDK/integration-dependent |
| Provider layer | **Owned, zero-LLM-SDK** — single httpx transport + adapter registry; OpenAI-compatible vendors config-only | provider SDKs / OpenRouter | `llm` per-provider plugins | SDK-dependent |
| Keys | **Name-only, never stored/logged/serialized; provider errors `redact()`-scrubbed** | BYOK incl. encrypted keychain storage | BYO via `llm` key store | BYO, app-managed |
| Modes | synthesize/raw/**debate**/**adversarial** now; vote planned | council + ranking | consortium (iterate-to-consensus) | arbitrary graphs you author |

**Where we are *now* distinct** (re-anchored on the parts competitors have not replicated):

1. **Owned, zero-LLM-SDK provider highway** — a single hand-owned httpx transport + adapter
   registry, no provider SDKs, no OpenRouter. Competitors lean on aggregators or vendor SDKs;
   none advertise an SDK-free transport they fully control. This is the most defensible claim.
2. **Direct-keys / no-middleman as a headline, not a footnote** — conclave never requires an
   aggregator and never proxies tokens. This is a sharper wedge than "library-first" against
   OpenRouter-locked tools (karpathy `llm-council`) and OpenRouter-default peers.
3. **Name-only key rigor + `redact()` scrubbing** — the value never transits any data
   structure, is never serialized, and is scrubbed from provider error strings. (Peers offer
   BYOK and even encrypted keychains — a different, valid posture; ours is minimal-surface.)
4. **A telemetry-grade `CouncilResult` contract** — per-model latency + token usage + typed
   error capture as a *stable downstream contract* (the mcp-warden dev-time dependency story),
   not just content structure from a synthesizer.

We are not trying to beat LangGraph/AutoGen at general agent orchestration — we are the
small, embeddable council primitive. Against the new direct peers (`llm-council-core`,
`the-llm-council`) we differentiate on the owned provider layer, the no-aggregator posture,
the key-handling rigor, and the stable result contract — **not** on library-first/structured/
resilient in general, which the category has caught up on.

---

## 12. Open Product Questions

Decisions are recorded in place (resolved 2026-06-08 except where noted). Resolved items
are kept here for traceability rather than deleted.

1. **Synthesizer-in-council policy.** ✅ **RESOLVED (2026-06-08): allow, document the
   self-reinforcement caveat, no code gate.** The default synthesizer (`claude`) may also be
   a council member in the same run; this mirrors the common "chairman" precedent and is
   low-stakes. The self-reinforcement risk is documented, not enforced.
2. **`vote` answer schema.** ⏸️ **DEFERRED** — does `vote` require a constrained/structured
   answer format (and therefore a prompt contract), or do we tally free-text answers post
   hoc? Pending decision. Note the hidden dependency: comparable votes need enforced
   structured-output support across all three adapters (none currently send
   `response_format`/`tool`/`responseSchema`) — that prerequisite must land before `vote`
   (#3) is scheduled.
3. **Per-member model/temperature overrides.** ✅ **RESOLVED (2026-06-08): yes, at the
   config level**, with the council-wide value as the default. Members may carry per-member
   `model`/`temperature` overrides in config; call-args overrides are out of scope for now.
4. **Server mode scope.** ✅ **RESOLVED (2026-06-08): localhost-bind only, no token-proxy
   path, explicit no-middleman guard.** If a local HTTP mode ships (#8), it binds to
   `127.0.0.1`, never becomes a hosted token path, and carries an explicit non-goal guard.
   (Peer `llm-council-core` shipped MCP + HTTP, so precedent exists — but the no-middleman
   non-goal §8 is load-bearing and overrides convenience.)
5. **First-class provider expansion criteria.** ✅ **RESOLVED (2026-06-08): promote a
   pass-through to a typed default when it is OpenAI-compatible (or a native adapter exists)
   AND has a stable public API AND shows common demand.** The long tail stays config-only via
   `endpoints:`.
