# conclave

A bring-your-own-keys **multi-model council** — a CLI and Python library that
fans a prompt out to several foundation models concurrently (each via *your own*
API keys) and merges their answers into one consolidated response.

Built on conclave's own **provider highway** — an `httpx` async transport behind a
per-provider adapter registry, so there is **no LLM-SDK dependency** — plus `asyncio` for
concurrent fan-out, `rich` for output, and `pydantic` for config.

It is **library-first** (the CLI is a thin shell over the same `Council` you import),
returns **structured results** (per-model latency, token usage, and error capture), and is
**partial-failure resilient** — one provider erroring never aborts the run. Keys are
**bring-your-own**, referenced by environment-variable *name* only — never stored or
logged. It ships four modes: **synthesize** (merge answers into one), **raw** (no merge),
**debate** (multi-round, members revise after seeing peers' anonymized answers), and
**adversarial** (propose → refute → verdict). conclave is intentionally lightweight — a
small council primitive, not an agent framework.

**The v1.1 wedge — the auditable council.** Every run also yields **a multi-model council
verdict you can act on — structured, scored for agreement, fully auditable**: a
`CouncilVerdict` exposing agreement, `conflicts`, `minority_reports`, and `provider_votes`;
a deterministic `consensus_score` (arithmetic over the model's clustering, *never* an
LLM-emitted number); and a redacted `ModelHarnessManifest` recording how the run executed and
which model produced the disagreement analysis. The verdict is **default-on**. (A `vote` mode
was once on the roadmap; it is **absorbed** by `provider_votes` — see below.)

See the canonical spec and design docs:

- [`docs/PRODUCT_DESIGN_DOCUMENT.md`](docs/PRODUCT_DESIGN_DOCUMENT.md) — canonical product
  spec, council modes, security model, roadmap, positioning (the authority doc).
- [`SYSTEM_CONTEXT_DIAGRAM.md`](SYSTEM_CONTEXT_DIAGRAM.md) — system context diagram.
- [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) — master index of all docs + source.

## Install

```bash
pip install conclave-cli
```

> **Name split (read this once).** The PyPI **distribution** name is
> `conclave-cli` — the name `conclave` on PyPI is an *unrelated* project (a
> blockchain client, not this one). Everything else stays `conclave`: the CLI
> command you type is `conclave`, the package you import is `conclave`, and the
> repo is `conclave`. So:
> `pip install conclave-cli` → run `conclave ...` / `from conclave import Council`.

From a source checkout (for development), install it editable instead:

```bash
# from the repo root
pip install -e .
# or with dev/test extras
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Bring your own keys

`conclave` never stores or hardcodes keys. It reads them from the environment
using each provider's standard variable name:

| Provider   | Friendly name | Default model id            | Env var(s)                       |
|------------|---------------|-----------------------------|----------------------------------|
| xAI        | `grok`        | `xai/grok-4.3`              | `XAI_API_KEY`                    |
| Google     | `gemini`      | `gemini/gemini-2.5-pro`     | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| Anthropic  | `claude`      | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY`            |
| Perplexity | `perplexity`  | `perplexity/sonar-pro`      | `PERPLEXITY_API_KEY`             |
| OpenAI     | `openai`      | `openai/gpt-4.1`            | `OPENAI_API_KEY`                 |
| Groq       | `groq`        | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY`                |
| DeepSeek   | `deepseek`    | `deepseek/deepseek-chat`    | `DEEPSEEK_API_KEY`               |
| Mistral    | `mistral`     | `mistral/mistral-large-latest` | `MISTRAL_API_KEY`             |
| Together   | `together`    | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |

Every first-class provider above is a **direct vendor key to a direct vendor
endpoint** — conclave never routes through an aggregator. Any other
OpenAI-compatible vendor (including aggregators/routers, which are deliberately
*not* first-class) remains usable config-only via an `endpoints:` entry.

Set whichever you have:

```bash
export XAI_API_KEY=...
export PERPLEXITY_API_KEY=...
```

Any requested provider whose key is absent is **skipped with a warning** — the
council runs with whoever is available. One provider erroring (network/auth)
never kills the run; you still get partial results plus a synthesis of the
survivors.

## Quickstart (CLI)

```bash
# Which providers have a key right now? (never prints key values)
conclave providers

# Fan out to a council and synthesize
conclave ask "Explain CRDTs in two sentences." \
  --council grok,gemini,claude,perplexity --mode synthesize

# Pick the synthesizer explicitly
conclave ask "Compare gRPC vs REST." -c grok,perplexity -s claude

# Raw answers only, no synthesis
conclave ask "Name three sorting algorithms." -c grok,perplexity --mode raw

# Debate: members revise over N rounds after seeing peers' anonymized answers
conclave ask "Is a service mesh worth it for 8 services?" \
  -c grok,gemini,claude --mode debate --rounds 3

# Debate with early-stop: stop before --rounds once answers stop changing
conclave ask "Is a service mesh worth it for 8 services?" \
  -c grok,gemini,claude --mode debate --rounds 5 --converge-threshold 0.95

# Adversarial: one model proposes, the rest refute, the synthesizer judges
conclave ask "Defend event sourcing for this ledger." \
  -c grok,gemini,perplexity --mode adversarial --proposer grok

# Machine-readable output (works for every mode; carries rounds/adversarial too)
conclave ask "..." -c grok,perplexity --mode debate --json
```

Mode flags at a glance: `--mode synthesize|raw|debate|adversarial`. `--rounds N`
(default 2) is the *maximum* round count for `debate`; `--converge-threshold FLOAT`
(or `--converge`/`--no-converge`) optionally stops a debate early once answers
stabilize round-over-round (off by default — `--rounds` runs in full). `--proposer
NAME` (default: first member) applies to `adversarial`. `--synthesizer/-s` overrides
the synthesizer *and* the adversarial judge.

`--council` accepts either a comma-separated list of friendly names or the name
of a council defined in your config (see below). The built-in `default` council
is all known providers.

Add `--stream` to render member (and synthesizer) tokens live as they arrive
(`synthesize`/`raw` modes only):

```bash
conclave ask "Explain CRDTs in two paragraphs." -c grok,gemini,claude --stream
```

Streaming and the non-streaming default produce the **same** final
`CouncilResult`; `--stream` only changes how output is rendered. It is ignored
with `--json` (which always emits the full structured payload), and on a cache
hit the cached text is rendered in one shot rather than as a fake token stream.

## Quickstart (library)

```python
from conclave import Council

council = Council(models=["grok", "perplexity"], synthesizer="claude")

# sync
result = council.ask_sync("What is the capital of France?")

# or async
# result = await council.ask("What is the capital of France?")

for answer in result.answers:
    print(answer.name, answer.latency_s, answer.error or answer.answer[:80])

print("SYNTHESIS:\n", result.synthesis)
```

## Auditable verdict

On a decision- or review-style prompt with at least two responding members, conclave
adjudicates the answers into a structured, agreement-scored, auditable **verdict**. It is
**default-on** — no flag needed.

From the CLI, a decision prompt renders a green `VERDICT` panel:

```text
$ conclave ask "Should we adopt a service mesh for 8 services?" -c grok,gemini,claude

╭─ VERDICT (decision) ─────────────────────────────────────────────────────────╮
│ A service mesh is not yet justified for 8 services.                            │
│                                                                                │
│ Start with library-level retries/timeouts and centralized metrics; revisit a   │
│ mesh past ~20 services or when mTLS/traffic-splitting become hard requirements. │
│                                                                                │
│ consensus: strong (0.75) — heuristic: position_cluster_ratio_v1                │
╰────────────────────────────────────────────────────────────────────────────────╯
Conflicts
  • operational cost: not-yet-worth-it (grok, claude) vs worth-it (gemini)
Minority reports
  • gemini: mesh pays off early if you need mTLS everywhere
```

The consensus line is framed as a **heuristic**, never authoritative: the number is plain
arithmetic over the model's clustering of the answers, never a value the model emitted.

The same structure is on the result object:

```python
from conclave import Council

council = Council(models=["grok", "gemini", "claude"])   # verdict is default-on
result = council.ask_sync("Should we adopt a service mesh for 8 services?")

if result.verdict is not None:
    print(result.verdict.verdict_type)        # "decision"
    print(result.verdict.headline)
    print(result.verdict.recommendation)
    print(result.consensus_label, result.consensus_score)   # e.g. "strong" 0.75

    for conflict in result.conflicts:
        print("conflict:", conflict.topic, conflict.position_labels)
    for position in result.verdict.positions:
        print(position.label, "backed by answers:", position.evidence_answer_ids)
    for vote in result.provider_votes:
        print(vote.provider, "->", vote.position_label)     # who voted for what
    for mr in result.minority_reports:
        print("minority:", mr.providers, mr.claim)
else:
    # open-ended / fewer-than-2 members / extraction failure: synthesis is still returned
    print("no verdict:", result.manifest.verdict_absent_reason)
    print(result.synthesis)

# the execution manifest rides on every result — first-class, secret-free
print(result.manifest.verdict_extraction.model_id)   # which model produced the clustering
print(result.manifest.secret_safety)                 # "verified_no_secrets"
```

To opt out (e.g. for cost-sensitive runs — the verdict adds one extra synthesizer call per
run), construct with `Council(extract_verdict=False)`; then `result.verdict` stays `None`.
`conclave ask ... --json` already carries the full structured `verdict` and `manifest`.

**How the consensus number stays honest.** The single LLM-assisted step is *clustering* the
members' stances; `consensus_score` is then deterministic arithmetic over that clustering
(largest cluster / members with a position — no text-similarity, never model-emitted). Every
cluster cites the member `answer_id`s backing it (`evidence_answer_ids`), and the manifest
records which model + prompt version did the clustering, so the score is reproducible and
traceable rather than a number to take on faith.

**The verdict is optional.** It is absent — `result.verdict is None`, with the synthesis and
member answers still returned — for an open-ended/creative prompt, for fewer than two
responding members, or if extraction fails schema validation; the exact reason is on
`result.manifest.verdict_absent_reason`.

### Streaming (synthesize/raw)

`Council.ask_stream` is an async generator that yields incremental `StreamEvent`s
as member (and synthesizer) tokens arrive, then a terminal `done` event carrying
the full `CouncilResult` — the same shape `ask()` returns, so downstream code is
unaffected:

```python
async for event in council.ask_stream("What is the capital of France?"):
    if event.type in ("member_delta", "synthesis_delta"):
        print(event.text, end="", flush=True)        # live token
    elif event.type == "done":
        result = event.result                          # full CouncilResult
```

Event types: `member_delta` / `member_done` (per member, interleaved),
`synthesis_delta` / `synthesis_done` (when synthesizing), and the final `done`.
A member that cannot stream, or any mid-stream failure, degrades gracefully —
partial text is preserved and the error lands on that member's `ModelAnswer`,
never raising (the same never-raises contract as `ask`). Streaming applies to
`synthesize`/`raw` only; `debate`/`adversarial` are not streamed.

### Debate and adversarial modes

```python
council = Council(models=["grok", "gemini", "claude"], synthesizer="claude")

# debate: multi-round, anonymized peers, partial-failure resilient
debate = council.debate_sync("Is P=NP likely false?", rounds=3)   # or: await council.debate(...)
for rnd in debate.rounds:
    print("round", rnd.round_number, [a.name for a in rnd.successful_answers])
print("FINAL:\n", debate.synthesis)

# debate with optional early-stop: stop before `rounds` once answers converge
quick = council.debate_sync("Is P=NP likely false?", rounds=5, converge_threshold=0.95)
print("ran", len(quick.rounds), "rounds; converged:", quick.converged, quick.convergence_score)

# adversarial: propose -> refute -> verdict
adv = council.adversarial_sync("Defend CRDTs for offline-first apps.", proposer="grok")
print("PROPOSAL by", adv.adversarial.proposer, "->", adv.adversarial.proposal.answer)
for crit in adv.adversarial.critiques:
    print("CRITIQUE", crit.name, ":", crit.error or crit.answer[:80])
print("VERDICT:\n", adv.adversarial.verdict)   # also mirrored to adv.synthesis
```

`CouncilResult` exposes `mode`, `answers` (per-model `ModelAnswer` with `model_id`,
`latency_s`, `usage`, `error`), `synthesis`, `synthesizer`, `skipped`, plus
`successful_answers` / `failed_answers` helpers. For `debate` it also carries
`rounds` (a list of `DebateRound`, each with per-member `answers`) plus
`converged`/`convergence_score` (set when an early-stop fired); for
`adversarial` it carries `adversarial` (an `AdversarialResult` with `proposer`,
`proposal`, `critiques`, `verdict`). For debate the final round is mirrored into
`answers` and the synthesis into `synthesis`; for adversarial the proposal +
critiques populate `answers` and the verdict mirrors into `synthesis` — so code
written against the v0.1 surface keeps working across every mode.

## Synthesizer behavior

The synthesizer is the single model that merges the council's answers (and is the
**judge** in `adversarial` mode and the final consolidator in `debate`). It is
chosen by this precedence, highest first:

1. the `synthesizer=` argument to `Council` (CLI: `--synthesizer/-s`);
2. the `synthesizer:` key in `~/.conclave/config.yml`;
3. the built-in default — **`claude`** (`anthropic/claude-sonnet-4-6`).

```bash
conclave ask "..." --council grok,gemini --synthesizer openai   # override per run
```

**Degradation is observable, never silent.** Synthesis is skipped — and the
reason is always surfaced on the result — in three cases:

| Situation | What happens |
|---|---|
| No usable member answers (all errored/skipped) | `synthesis = None`, `synthesis_error = "no successful member answers…"` |
| Synthesizer has no API key | `synthesis = None`, `synthesis_error = "…has no API key; returning raw answers only"`; member answers preserved |
| Synthesizer call fails | `synthesis = None`, `synthesis_error =` the provider error |

In every case the member answers are returned intact and a warning is logged, so
a caller can reliably detect a non-synthesis with
`result.synthesis is None and result.synthesis_error is not None`. There is **no
path** where concatenated or partial output is silently returned as if it were a
synthesis. In `adversarial` mode the same signal lands on
`adversarial.verdict_error` (mirrored to `synthesis_error`).

**The synthesis prompt is a versioned constant.** The synthesize-mode system
prompt is fixed in code (not built per call); the debate/judge prompts live in
`conclave.prompts`. The whole prompt set carries a version tag,
`conclave.prompts.SYNTHESIS_PROMPT_VERSION`, stamped onto **every**
`CouncilResult` as `result.prompt_version`. A downstream eval or regression suite
can compare it across runs to detect that the synthesis wording changed, instead
of silently attributing the shift to model drift. The test suite pins both the
prompt text and the version, so changing one without the other fails CI.

## Config (optional)

Create `~/.conclave/config.yml` to add models, define named councils, and set a
default synthesizer. It references providers by **name only** — never keys.

```yaml
models:
  grok: xai/grok-4.3
  claude: anthropic/claude-sonnet-4-6
councils:
  default: [grok, gemini, claude, perplexity]
  fast: [grok, perplexity]
synthesizer: claude
```

Then: `conclave ask "..." --council fast`.

## Result cache (optional, off by default)

Repeated or eval runs can be served from an on-disk cache instead of re-calling
the providers. It is **off by default** and **never persists API keys** — the
cache key is a SHA-256 over the normalized prompt, the ordered council members
(friendly name + resolved model id), the mode, the synthesizer/judge identity,
and the mode params (temperature, debate `rounds` + `converge_threshold`,
adversarial `proposer`). No key value or env-var name ever reaches the key or the
stored payload.

Enable it per run with `--cache` (or disable a config default with `--no-cache`):

```bash
conclave ask "..." --council fast --cache
```

or set a default in `~/.conclave/config.yml`:

```yaml
cache: true
```

A cache hit returns the prior `CouncilResult` with `cached: true` set and does
not touch the network. Entries live under `$XDG_CACHE_HOME/conclave` (else
`~/.cache/conclave`); a corrupt or unreadable entry is treated as a miss and
never crashes a run.

## Test

```bash
pytest
```

The suite mocks the httpx transport, so it needs no network and no API keys.

## Extending: custom OpenAI-compatible providers

conclave's provider layer is an adapter registry over a single `httpx` transport
(`resolve_adapter` in `src/conclave/adapters/`). The first-class providers are
adapters; adding a *new* provider family is one registration. Adding any
**OpenAI-compatible** endpoint (a local server, a gateway, another vendor's
`/chat/completions`, or an aggregator/router you choose to use) needs **no code** —
just an `endpoints:` entry in your config that names the base URL and the env-var
that holds its key:

```yaml
endpoints:
  myllm:
    base_url: https://my-gateway.internal/v1
    api_key_env: MYLLM_API_KEY
models:
  mymodel: myllm/some-model-id
```

The endpoint is referenced by **name only**; the key value is read from `MYLLM_API_KEY`
at call time and never stored in config or results.
