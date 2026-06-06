# conclave

A bring-your-own-keys **multi-model council** — a CLI and Python library that
fans a prompt out to several foundation models concurrently (each via *your own*
API keys) and merges their answers into one consolidated response.

Built on [LiteLLM](https://github.com/BerriAI/litellm) for provider abstraction,
`asyncio` for concurrent fan-out, `rich` for output, and `pydantic` for config.

## Install

```bash
# from the repo root
pip install -e .
# or with dev/test extras
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Bring your own keys

`conclave` never stores or hardcodes keys. It reads them from the environment
using LiteLLM's standard variable names:

| Provider   | Friendly name | Default model id            | Env var(s)                       |
|------------|---------------|-----------------------------|----------------------------------|
| xAI        | `grok`        | `xai/grok-4.3`              | `XAI_API_KEY`                    |
| Google     | `gemini`      | `gemini/gemini-2.5-pro`     | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| Anthropic  | `claude`      | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY`            |
| Perplexity | `perplexity`  | `perplexity/sonar-pro`      | `PERPLEXITY_API_KEY`             |
| OpenAI     | `openai`      | `openai/gpt-4.1`            | `OPENAI_API_KEY`                 |

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

# Machine-readable output
conclave ask "..." -c grok,perplexity --json
```

`--council` accepts either a comma-separated list of friendly names or the name
of a council defined in your config (see below). The built-in `default` council
is all known providers.

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

`CouncilResult` exposes `answers` (per-model `ModelAnswer` with `model_id`,
`latency_s`, `usage`, `error`), `synthesis`, `synthesizer`, `skipped`, plus
`successful_answers` / `failed_answers` helpers.

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

## Test

```bash
pytest
```

The suite mocks LiteLLM, so it needs no network and no API keys.
