# Adjudication Succession Implementation Plan (DSE-1512)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the judge/synthesizer/verdict-extractor fail over, in operator-declared order, to the next candidate on *infrastructure* failures only — so one vendor's outage no longer strips a council run of its adjudication — and record every attempt in the manifest.

**Architecture:** (1) Type the failure category at the raise site (transport + adapters) and thread it onto `ModelAnswer.failure_category` — no substring matching. (2) Replace the single `synthesizer` identity with an ordered `synthesizer_chain` (chain-of-one == today). (3) One new seam, `Council.adjudicate(role, …)`, walks the chain with a strict failover rule and returns the answer plus a per-attempt ledger; every adjudication role (synthesis, debate final, adversarial judge, verdict extraction, streaming synthesis) routes through it. (4) `ModelHarnessManifest.adjudication_succession` records the ledger with **no free text** (categories + HTTP status only) so the secret-safety stamp stays provably clean. (5) The full chain joins cache identity; a result adjudicated by a successor is never stored in the cache.

**Tech Stack:** Python 3.11+, pydantic v2, httpx (transport), typer (CLI), pytest + pytest-asyncio (offline, mocked seams — see `tests/conftest.py`).

**Execution locality:** edit + git on the laptop worktree `~/dev/worktrees/conclave-dse-1512`; run every test/lint on the builder via
`~/.claude/scripts/builder-run.sh conclave-dse-1512 '<cmd>'` (it rsyncs first). Never run pytest/ruff on the laptop.
Shorthand used below: `BR='~/.claude/scripts/builder-run.sh conclave-dse-1512'`.

---

## Ground rules (read before Task 1)

- **Additive only.** `ModelAnswer.error` text must be byte-identical to today for every existing test. New fields default to `None`/empty. Do not rename anything.
- **Chain of one == v1.3.0.** With no chain configured, every code path must behave exactly as it does now. The suite at `origin/main` is the regression oracle — run it after every task.
- **Failover fires on infrastructure failure only.** A valid-but-unwelcome or malformed answer is terminal. Never let adjudication shop for a result.
- **The manifest carries no free text for succession.** `AdjudicationAttempt` has categories and an HTTP status, never an error string. `scan_for_secret_material()` forbids the substrings `sk-`, `bearer`, `authorization`, `api_key`, `x-api-key` in the serialized manifest — a provider error body saying "Missing Authorization header" would un-verify the stamp. That is why the ticket's `redacted_reason` field is **replaced** by `failure_category` + `http_status`.
- **Do not touch** `redact()`, `scan_for_secret_material()`, `_receipt_error_category()`, or any credential path. Touching them re-classifies the PR as security-specific.
- **TDD.** Failing test → run → minimal code → run → commit. Commit after each task with the message given.
- **Do not modify the existing repair-retry behaviour in `extract_verdict`** (it still retries once on the same model even after an infra error). Out of scope; note it as a follow-up in the PR body.

Existing seams you will rely on:

| Seam | Where | Note |
|---|---|---|
| Member/synth call | `conclave.council.call_model` | patched by `patch_call_model` fixture |
| Verdict call | `conclave.verdict_synthesis.call_model` | autouse offline stub in conftest |
| Transport | `conclave.transport.post_json` | patched in `tests/test_providers.py` for end-to-end `call_model` |
| Key presence | `conclave.registry.key_present` | env-var name check; unknown providers return `True` |

---

### Task 1: Typed failure categories (models + transport + adapters)

**Files:**
- Modify: `src/conclave/models.py` (add near `TokenUsage`)
- Modify: `src/conclave/transport.py:94-140,184,192,281,306,310`
- Modify: `src/conclave/adapters/base.py:240-247`
- Modify: `src/conclave/adapters/__init__.py:96`
- Modify: `src/conclave/adapters/openai_compat.py:215-218`
- Modify: `src/conclave/adapters/anthropic.py:201-204`
- Modify: `src/conclave/adapters/gemini.py:344-345`
- Test: `tests/test_failure_category.py` (new)

**Step 1: Write the failing tests**

```python
# tests/test_failure_category.py
"""Typed failure categories are derived from status codes / exception types (DSE-1512)."""

from __future__ import annotations

import httpx
import pytest

from conclave import transport
from conclave.adapters import ProviderError, resolve_adapter
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter
from conclave.config import ConclaveConfig
from conclave.models import FAILOVER_CATEGORIES, categorize_http_status


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "auth"),
        (403, "auth"),
        (402, "quota"),
        (429, "quota"),
        (408, "timeout"),
        (500, "unavailable"),
        (502, "unavailable"),
        (503, "unavailable"),
        (529, "unavailable"),
        (400, "bad_request"),
        (404, "bad_request"),
        (422, "bad_request"),
    ],
)
def test_categorize_http_status(status, expected):
    assert categorize_http_status(status) == expected


def test_failover_set_is_infrastructure_only():
    assert FAILOVER_CATEGORIES == frozenset(
        {"unkeyed", "unresolved", "auth", "quota", "unavailable", "timeout", "transport"}
    )
    assert "bad_request" not in FAILOVER_CATEGORIES
    assert "malformed_response" not in FAILOVER_CATEGORIES
    assert "unexpected" not in FAILOVER_CATEGORIES


def test_provider_error_defaults_to_malformed_response():
    err = ProviderError("x: empty response")
    assert err.category == "malformed_response"
    assert err.http_status is None


def test_provider_error_carries_status_category():
    err = ProviderError("x: HTTP 429: slow down", category="quota", http_status=429)
    assert err.category == "quota"
    assert err.http_status == 429
    # message is still redacted on construction (existing contract)
    assert "sk-" not in str(ProviderError("leak sk-abc123def456ghi789", category="auth"))


def test_transport_error_category():
    assert (
        transport.TransportError("request timed out after 5s", category="timeout").category
        == "timeout"
    )
    assert transport.TransportError("network error: ConnectError").category == "transport"


@pytest.mark.parametrize(
    "adapter", [OpenAICompatAdapter("openai"), AnthropicAdapter(), GeminiAdapter()]
)
def test_adapters_type_non_2xx(adapter):
    with pytest.raises(ProviderError) as info:
        adapter.parse_response(401, {"error": {"message": "bad key"}})
    assert info.value.category == "auth"
    assert info.value.http_status == 401
    with pytest.raises(ProviderError) as info:
        adapter.parse_response(503, {"error": {"message": "down"}})
    assert info.value.category == "unavailable"


def test_adapter_malformed_is_not_failover():
    with pytest.raises(ProviderError) as info:
        OpenAICompatAdapter("openai").parse_response(200, {"choices": []})
    assert info.value.category == "malformed_response"
    assert info.value.category not in FAILOVER_CATEGORIES


def test_unresolved_provider_is_typed():
    with pytest.raises(ProviderError) as info:
        resolve_adapter("nope/model", ConclaveConfig())
    assert info.value.category == "unresolved"


async def test_post_json_timeout_is_typed(monkeypatch):
    class _Client:
        is_closed = False

        async def post(self, *a, **k):
            raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(transport, "_client", _Client())
    with pytest.raises(transport.TransportError) as info:
        await transport.post_json("https://x", {}, {}, 1.0)
    assert info.value.category == "timeout"


async def test_post_json_network_is_typed(monkeypatch):
    class _Client:
        is_closed = False

        async def post(self, *a, **k):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(transport, "_client", _Client())
    with pytest.raises(transport.TransportError) as info:
        await transport.post_json("https://x", {}, {}, 1.0)
    assert info.value.category == "transport"
```

Check the adapter constructor signatures before running (`grep -n "class .*Adapter\|def __init__" src/conclave/adapters/*.py`) and adjust the parametrize instantiation to match — the registry in `adapters/__init__.py` shows how each is constructed.

**Step 2: Run to verify it fails**

Run: `$BR '.venv/bin/python -m pytest -q tests/test_failure_category.py'`
Expected: FAIL — `ImportError: cannot import name 'FAILOVER_CATEGORIES'`

**Step 3: Implement**

`src/conclave/models.py` — add after `TokenUsage`:

```python
# DSE-1512 — typed failure categories. Derived at the RAISE SITE from the HTTP
# status or exception type, never by inspecting a rendered error string. The
# adjudication ladder (Council.adjudicate) fails over ONLY on the categories in
# FAILOVER_CATEGORIES: infrastructure failures where no model ever produced an
# answer. A model that answered (even malformed) is terminal for that role.
FailureCategory = Literal[
    "unkeyed",  # env var absent -- no call made
    "unresolved",  # unknown provider prefix -- no call made
    "auth",  # 401 / 403
    "quota",  # 402 / 429
    "unavailable",  # 5xx
    "timeout",  # 408 or transport deadline
    "transport",  # DNS / connection / other httpx network error
    "bad_request",  # other 4xx -- the request was wrong, not the vendor
    "malformed_response",  # 2xx with an unusable payload / empty content
    "unexpected",  # anything else -- never failed over
]

FAILOVER_CATEGORIES: frozenset[str] = frozenset(
    {"unkeyed", "unresolved", "auth", "quota", "unavailable", "timeout", "transport"}
)


def categorize_http_status(status: int) -> FailureCategory:
    """Map a non-2xx HTTP status to a :data:`FailureCategory` (pure, no I/O)."""
    if status in (401, 403):
        return "auth"
    if status in (402, 429):
        return "quota"
    if status == 408:
        return "timeout"
    if 500 <= status <= 599:
        return "unavailable"
    if 400 <= status <= 499:
        return "bad_request"
    return "malformed_response"
```

Add `from typing import Literal` if not already imported. Add two fields to `ModelAnswer` (after `warnings`), and document them in the class docstring:

```python
    failure_category: FailureCategory | None = None
    http_status: int | None = None
```

`src/conclave/transport.py`:

```python
from .models import FailureCategory, categorize_http_status  # add import


class TransportError(Exception):
    """...(keep docstring)..."""

    def __init__(self, message: str, *, category: FailureCategory = "transport") -> None:
        super().__init__(message)
        self.category: FailureCategory = category


def _raise_transport_error(message: str, category: FailureCategory = "transport") -> NoReturn:
    raise TransportError(message, category=category) from None
```

- Both `httpx.TimeoutException` sites: `_raise_transport_error(f"request timed out after {timeout:.0f}s", "timeout")`.
- Both `httpx.HTTPError` sites: unchanged call (default `"transport"`).
- `stream_sse` non-2xx: `raise TransportError(f"HTTP {response.status_code}: {detail}", category=categorize_http_status(response.status_code))`.

`src/conclave/adapters/base.py`:

```python
from ..models import FailureCategory, TokenUsage  # extend the existing import


class ProviderError(Exception):
    """...(keep docstring; add:) ``category``/``http_status`` are typed at the raise
    site (DSE-1512) so failover never depends on the message text."""

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory = "malformed_response",
        http_status: int | None = None,
    ) -> None:
        super().__init__(redact(message))
        self.category: FailureCategory = category
        self.http_status = http_status
```

Adapters — only the three buffered non-2xx raise sites change (stream-frame in-band errors carry `status 200` and stay `malformed_response` by design):

```python
# openai_compat.py parse_response
        if status < 200 or status >= 300:
            raise ProviderError(
                status_error(self.prefix, status, payload, secondary_keys=("type",)),
                category=categorize_http_status(status),
                http_status=status,
            )
```
Same shape in `anthropic.py` (secondary_keys `("type",)`) and `gemini.py` (secondary_keys `("status",)`). Import `categorize_http_status` from `..models` in each.

`src/conclave/adapters/__init__.py:96` — the unknown-provider raise gets `category="unresolved"`.

**Step 4: Run to verify it passes**

Run: `$BR '.venv/bin/python -m pytest -q tests/test_failure_category.py tests/test_adapters.py tests/test_transport.py tests/test_providers.py'`
Expected: PASS (all)

**Step 5: Full regression + commit**

Run: `$BR` (full suite) — Expected: same pass count as baseline, 0 failures.
Run: `$BR '.venv/bin/ruff check . && .venv/bin/ruff format --check .'`

```bash
git add src/conclave/models.py src/conclave/transport.py src/conclave/adapters tests/test_failure_category.py
git commit -m "feat(models): type provider failure categories at the raise site (DSE-1512)"
```

---

### Task 2: Thread the category onto `ModelAnswer` in `call_model` / `call_model_stream`

**Files:**
- Modify: `src/conclave/providers.py:169-260` (`call_model`), `:300-409` (`call_model_stream`)
- Test: `tests/test_providers.py` (append)

**Step 1: Failing tests** (append to `tests/test_providers.py`; follow the file's existing `post_json` patching style)

```python
async def test_call_model_types_unkeyed(monkeypatch, clear_keys):
    ans = await call_model("grok", "xai/grok-4.3", [{"role": "user", "content": "hi"}])
    assert ans.error and ans.failure_category == "unkeyed" and ans.http_status is None


async def test_call_model_types_http_status(monkeypatch, patch_transport):
    patch_transport(status=402, body={"error": {"message": "insufficient credit"}})
    ans = await call_model(
        "claude", "anthropic/claude-sonnet-4-6", [{"role": "user", "content": "hi"}]
    )
    assert ans.error and ans.failure_category == "quota" and ans.http_status == 402


async def test_call_model_types_timeout(monkeypatch, patch_transport_raise):
    patch_transport_raise(
        transport.TransportError("request timed out after 1s", category="timeout")
    )
    ans = await call_model(
        "claude", "anthropic/claude-sonnet-4-6", [{"role": "user", "content": "hi"}]
    )
    assert ans.failure_category == "timeout"


async def test_call_model_error_text_unchanged(monkeypatch, patch_transport):
    """Additive-only guarantee: the error STRING is byte-identical to before."""
    patch_transport(status=401, body={"error": {"message": "bad key"}})
    ans = await call_model(
        "claude", "anthropic/claude-sonnet-4-6", [{"role": "user", "content": "hi"}]
    )
    assert ans.error == "anthropic: HTTP 401: bad key"


async def test_call_model_unresolved_is_typed():
    ans = await call_model(
        "x", "nope/model", [{"role": "user", "content": "hi"}], config=ConclaveConfig()
    )
    assert ans.failure_category == "unresolved"
```

`clear_keys` exists in `tests/test_council.py`'s fixtures; if it is module-local, copy the pattern (monkeypatch.delenv each `*_API_KEY`) into a local fixture. Write `patch_transport`/`patch_transport_raise` as small local fixtures that `monkeypatch.setattr(transport, "post_json", fake)` — mirror the existing helpers in `tests/test_providers.py`. Set a dummy `ANTHROPIC_API_KEY` in those tests so the unkeyed branch is not taken.

**Step 2: Run** — `$BR '.venv/bin/python -m pytest -q tests/test_providers.py -k "types or unchanged"'` → FAIL (`failure_category` is `None`).

**Step 3: Implement** — in `call_model`, each early return / except branch adds the category (error strings untouched):

```python
    except ProviderError as exc:                       # unresolved adapter
        ... error=str(exc), failure_category=exc.category)
    if api_key is None:
        ... error=msg, failure_category="unkeyed")
    except (ProviderError, TransportError) as exc:
        ... error=message,
            failure_category=exc.category,
            http_status=getattr(exc, "http_status", None))
    except Exception as exc:  # noqa: BLE001
        ... error=message, failure_category="unexpected")
```

Apply the identical four-way mapping to every `yield ModelAnswer(... error=...)` in `call_model_stream`. The "empty response (no streamed content)" yield gets `failure_category="malformed_response"`.

**Step 4: Run** — same command → PASS. Then `$BR` full suite → 0 failures.

**Step 5: Commit**
```bash
git add src/conclave/providers.py tests/test_providers.py
git commit -m "feat(providers): carry typed failure_category/http_status on ModelAnswer (DSE-1512)"
```

---

### Task 3: `synthesizer_chain` config + parsing + `Council` resolution

**Files:**
- Modify: `src/conclave/config.py:57-95,190-236`
- Modify: `src/conclave/council.py:154-192`
- Modify: `config.example.yml`
- Test: `tests/test_registry_config.py` (append), `tests/test_council.py` (append)

**Step 1: Failing tests**

```python
# tests/test_registry_config.py (append)
from conclave.config import parse_synthesizer_chain, _load_config_uncached


def test_parse_synthesizer_chain_splits_and_dedupes():
    assert parse_synthesizer_chain("claude>grok > gemini>claude") == ["claude", "grok", "gemini"]
    assert parse_synthesizer_chain("claude") == ["claude"]
    assert parse_synthesizer_chain("  ") == []


def test_config_synthesizer_chain_from_yaml(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("synthesizer: claude\nsynthesizer_chain: [claude, grok]\n")
    cfg = _load_config_uncached(p)
    assert cfg.synthesizer == "claude"
    assert cfg.synthesizer_chain == ["claude", "grok"]


def test_config_synthesizer_chain_accepts_arrow_string(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("synthesizer_chain: 'claude>grok'\n")
    assert _load_config_uncached(p).synthesizer_chain == ["claude", "grok"]


def test_config_synthesizer_chain_bad_value_is_empty(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("synthesizer_chain: 42\n")
    assert _load_config_uncached(p).synthesizer_chain == []
```

```python
# tests/test_council.py (append)
def test_council_chain_defaults_to_single_synthesizer():
    c = Council(models=["grok"], config=ConclaveConfig(synthesizer="claude"))
    assert c.synthesizer_chain == ["claude"] and c.synthesizer == "claude"


def test_council_chain_from_constructor_string():
    c = Council(models=["grok"], synthesizer="claude>grok", config=ConclaveConfig())
    assert c.synthesizer_chain == ["claude", "grok"] and c.synthesizer == "claude"


def test_council_chain_from_constructor_list():
    c = Council(models=["grok"], synthesizer=["gemini", "grok"], config=ConclaveConfig())
    assert c.synthesizer_chain == ["gemini", "grok"] and c.synthesizer == "gemini"


def test_council_chain_from_config_overrides_scalar():
    cfg = ConclaveConfig(synthesizer="claude", synthesizer_chain=["grok", "gemini"])
    c = Council(models=["claude"], config=cfg)
    assert c.synthesizer_chain == ["grok", "gemini"] and c.synthesizer == "grok"


def test_council_constructor_arg_beats_config_chain():
    cfg = ConclaveConfig(synthesizer_chain=["grok", "gemini"])
    c = Council(models=["claude"], synthesizer="claude", config=cfg)
    assert c.synthesizer_chain == ["claude"]
```

**Step 2: Run** — `$BR '.venv/bin/python -m pytest -q tests/test_registry_config.py tests/test_council.py -k chain'` → FAIL.

**Step 3: Implement**

`config.py`:
```python
def parse_synthesizer_chain(spec: str) -> list[str]:
    """Split ``"a>b>c"`` into an ordered, de-duplicated candidate list."""
    seen: list[str] = []
    for part in spec.split(">"):
        name = part.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _coerce_chain(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return parse_synthesizer_chain(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return parse_synthesizer_chain(">".join(value))
    logger.warning("synthesizer_chain %r is not a list of names; ignoring", value)
    return []
```
Add `synthesizer_chain: list[str] = Field(default_factory=list)` to `ConclaveConfig` (docstring: ordered failover ladder; empty means "just `synthesizer`"). In `_load_config_uncached`: `synthesizer_chain=_coerce_chain(raw.get("synthesizer_chain"))`. Update the module docstring example.

`council.py` `__init__`: change the annotation to `synthesizer: str | Sequence[str] | None = None` and resolve:
```python
        self.synthesizer_chain = self._resolve_chain(synthesizer, self.config)
        # Back-compat: the primary candidate keeps the historic attribute.
        self.synthesizer = self.synthesizer_chain[0]

    @staticmethod
    def _resolve_chain(spec: str | Sequence[str] | None, config: ConclaveConfig) -> list[str]:
        """constructor arg → config.synthesizer_chain → [config.synthesizer]."""
        if isinstance(spec, str):
            chain = parse_synthesizer_chain(spec)
        elif spec is not None:
            chain = parse_synthesizer_chain(">".join(spec))
        else:
            chain = list(config.synthesizer_chain)
        return chain or [config.synthesizer]
```
Document the ladder in the class docstring `synthesizer:` entry. Add `synthesizer_chain: [claude, grok]` (commented) to `config.example.yml`.

**Step 4: Run** → PASS; `$BR` full → 0 failures.

**Step 5: Commit** — `git commit -m "feat(config): synthesizer_chain ordered failover ladder (DSE-1512)"`

---

### Task 4: Manifest ledger type + `Council.adjudicate` seam

**Files:**
- Modify: `src/conclave/manifest.py` (new `AdjudicationAttempt`, new manifest field)
- Modify: `src/conclave/council.py` (new `adjudicate`, `_record_adjudication`; `synthesize_blocks` becomes a wrapper)
- Test: `tests/test_adjudication.py` (new)

**Step 1: Failing tests**

```python
# tests/test_adjudication.py
"""Council.adjudicate walks the synthesizer chain with the infra-only failover rule."""

from __future__ import annotations

import pytest

from conclave.config import ConclaveConfig
from conclave.council import Council
from conclave.manifest import AdjudicationAttempt, ModelHarnessManifest
from conclave.models import CouncilResult, ModelAnswer
from conclave import council as council_mod

CFG = ConclaveConfig(models={"claude": "anthropic/c", "grok": "xai/g", "gemini": "gemini/m"})


def _fail(name, model_id, category, status=None):
    return ModelAnswer(
        name=name,
        model_id=model_id,
        error=f"{name} failed",
        failure_category=category,
        http_status=status,
    )


def _ok(name, model_id):
    return ModelAnswer(
        name=name, model_id=model_id, answer=f"{name} says yes", answer_id=f"{name}-1"
    )


def _install(monkeypatch, script: dict[str, ModelAnswer]):
    calls = []

    async def fake(name, model_id, messages, **kw):
        calls.append(name)
        return script[name]

    monkeypatch.setattr(council_mod, "call_model", fake)
    return calls


@pytest.fixture
def keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "dummy")


async def test_chain_of_one_success(monkeypatch, keys):
    calls = _install(monkeypatch, {"claude": _ok("claude", "anthropic/c")})
    c = Council(models=["grok"], synthesizer="claude", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer.ok and out.name == "claude" and calls == ["claude"]
    assert [a.outcome for a in out.attempts] == ["success"]


async def test_auth_failure_advances(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {"claude": _fail("claude", "anthropic/c", "auth", 401), "grok": _ok("grok", "xai/g")},
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer.ok and out.name == "grok" and out.model_id == "xai/g"
    assert calls == ["claude", "grok"]
    assert [(a.candidate, a.outcome, a.failure_category, a.http_status) for a in out.attempts] == [
        ("claude", "failed_over", "auth", 401),
        ("grok", "success", None, None),
    ]


async def test_bad_request_is_terminal(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {
            "claude": _fail("claude", "anthropic/c", "bad_request", 400),
            "grok": _ok("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert not out.answer.ok and out.name == "claude"
    assert calls == ["claude"]  # grok was never consulted
    assert [a.outcome for a in out.attempts] == ["terminal_failure"]


async def test_malformed_is_terminal(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {
            "claude": _fail("claude", "anthropic/c", "malformed_response"),
            "grok": _ok("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("judge", "sys", "user")
    assert calls == ["claude"] and out.attempts[0].outcome == "terminal_failure"


async def test_unkeyed_candidate_is_skipped_without_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    calls = _install(monkeypatch, {"grok": _ok("grok", "xai/g")})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert calls == ["grok"]
    assert [a.outcome for a in out.attempts] == ["skipped_unkeyed", "success"]


async def test_chain_exhausted(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {
            "claude": _fail("claude", "anthropic/c", "quota", 429),
            "grok": _fail("grok", "xai/g", "unavailable", 503),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert calls == ["claude", "grok"]
    assert not out.answer.ok and out.name == "grok"
    assert [a.outcome for a in out.attempts] == ["failed_over", "exhausted"]


async def test_all_unkeyed_returns_no_answer(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    calls = _install(monkeypatch, {})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer is None and calls == []
    assert [a.outcome for a in out.attempts] == ["skipped_unkeyed", "skipped_unkeyed"]


def test_record_adjudication_appends_ledger_and_receipts():
    c = Council(models=["grok"], synthesizer="claude", config=CFG)
    result = CouncilResult(
        prompt="p",
        manifest=ModelHarnessManifest(request_id="r", conclave_version="t", mode="synthesize"),
    )
    attempts = [
        AdjudicationAttempt(
            role="synthesis",
            candidate="claude",
            model_id="anthropic/c",
            attempt_index=1,
            outcome="failed_over",
            failure_category="auth",
            http_status=401,
        ),
        AdjudicationAttempt(
            role="synthesis", candidate="grok", model_id="xai/g", attempt_index=2, outcome="success"
        ),
    ]
    called = [_fail("claude", "anthropic/c", "auth", 401), _ok("grok", "xai/g")]
    c._record_adjudication(result, attempts, called, phase="synthesis")
    m = result.manifest
    assert m.adjudication_succession == attempts
    assert [(r.phase, r.attempt, r.outcome) for r in m.receipts] == [
        ("synthesis", 1, "failed"),
        ("synthesis", 2, "success"),
    ]
    assert m.secret_safety == "verified_no_secrets"


def test_ledger_has_no_free_text_fields():
    fields = set(AdjudicationAttempt.model_fields)
    assert fields == {
        "role",
        "candidate",
        "model_id",
        "attempt_index",
        "outcome",
        "failure_category",
        "http_status",
    }
```

**Step 2: Run** — `$BR '.venv/bin/python -m pytest -q tests/test_adjudication.py'` → FAIL (`ImportError: AdjudicationAttempt`).

**Step 3: Implement**

`manifest.py` — add before `ProviderExecutionReceipt`:
```python
AdjudicationRole = Literal["synthesis", "debate_final", "judge", "verdict_extraction"]
AdjudicationAttemptOutcome = Literal[
    "success",  # this candidate adjudicated
    "failed_over",  # infra failure; the next candidate was tried
    "exhausted",  # infra failure on the LAST candidate; nothing left to try
    "terminal_failure",  # the candidate answered unusably; failover refused by rule
    "skipped_unkeyed",  # no API key in the environment; no call made
]


class AdjudicationAttempt(BaseModel):
    """One step of the synthesizer/judge succession ladder (DSE-1512).

    Deliberately carries NO free text: only bounded categories and an HTTP
    status. A raw provider error string could contain words the secret-safety
    scan forbids (e.g. "authorization"), which would un-verify the manifest.
    """

    role: AdjudicationRole
    candidate: str
    model_id: str
    attempt_index: int = Field(ge=1)
    outcome: AdjudicationAttemptOutcome
    failure_category: str | None = None
    http_status: int | None = None
```
Add to `ModelHarnessManifest` (after `redacted_errors`, documented in the docstring):
```python
    adjudication_succession: list[AdjudicationAttempt] = Field(default_factory=list)
```

`council.py` — a small dataclass + the seam:
```python
@dataclass
class AdjudicationOutcome:
    """Return value of :meth:`Council.adjudicate`."""

    answer: (
        ModelAnswer | None
    )  # success, or the terminal/exhausted failure; None if no candidate could be called
    attempts: list[AdjudicationAttempt]
    called: list[ModelAnswer]  # every real call, in order (for receipts)

    @property
    def name(self) -> str | None:
        return self.answer.name if self.answer is not None else None

    @property
    def model_id(self) -> str | None:
        return self.answer.model_id if self.answer is not None else None
```
```python
async def adjudicate(
    self, role: AdjudicationRole, system_prompt: str, user_content: str
) -> AdjudicationOutcome:
    """Walk ``synthesizer_chain`` for one adjudication role (DSE-1512).

    Rule: a candidate is tried in declared order; an unkeyed candidate is
    skipped without a call; a call that fails with a category in
    :data:`FAILOVER_CATEGORIES` advances to the next candidate; ANY other
    failure is terminal for the role (a model that answered is never
    second-guessed by another vendor -- that would let adjudication shop for
    a result). No scoring, no health tracking: the order is the operator's.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    attempts: list[AdjudicationAttempt] = []
    called: list[ModelAnswer] = []
    last_failure: ModelAnswer | None = None
    chain = self.synthesizer_chain
    for index, candidate in enumerate(chain, start=1):
        model_id = self.config.resolve_model_id(candidate)
        if not key_present(model_id):
            attempts.append(
                AdjudicationAttempt(
                    role=role,
                    candidate=candidate,
                    model_id=model_id,
                    attempt_index=index,
                    outcome="skipped_unkeyed",
                    failure_category="unkeyed",
                )
            )
            continue
        answer = await call_model(
            candidate,
            model_id,
            messages,
            config=self.config,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        called.append(answer)
        if answer.ok:
            attempts.append(
                AdjudicationAttempt(
                    role=role,
                    candidate=candidate,
                    model_id=model_id,
                    attempt_index=index,
                    outcome="success",
                )
            )
            return AdjudicationOutcome(answer=answer, attempts=attempts, called=called)
        category = answer.failure_category
        if category in FAILOVER_CATEGORIES:
            is_last = index == len(chain)
            attempts.append(
                AdjudicationAttempt(
                    role=role,
                    candidate=candidate,
                    model_id=model_id,
                    attempt_index=index,
                    outcome="exhausted" if is_last else "failed_over",
                    failure_category=category,
                    http_status=answer.http_status,
                )
            )
            last_failure = answer
            if not is_last:
                logger.warning(
                    "%s: '%s' failed (%s); trying next candidate", role, candidate, category
                )
            continue
        attempts.append(
            AdjudicationAttempt(
                role=role,
                candidate=candidate,
                model_id=model_id,
                attempt_index=index,
                outcome="terminal_failure",
                failure_category=category,
                http_status=answer.http_status,
            )
        )
        return AdjudicationOutcome(answer=answer, attempts=attempts, called=called)
    return AdjudicationOutcome(answer=last_failure, attempts=attempts, called=called)


def _record_adjudication(
    self,
    result: CouncilResult,
    attempts: list[AdjudicationAttempt],
    called: list[ModelAnswer],
    *,
    phase: str,
    record_receipts: bool = True,
    protocol_version: str | None = None,
    prompt_version: str | None = SYNTHESIS_PROMPT_VERSION,
) -> None:
    """Append the succession ledger (always) and one receipt per real call."""
    if result.manifest is None:
        return
    result.manifest.adjudication_succession.extend(attempts)
    receipts = (
        [
            receipt_from_answer(
                answer,
                temperature=self.temperature,
                timeout=self.timeout,
                phase=phase,
                attempt=i,
                protocol_version=protocol_version,
                prompt_version=prompt_version,
            )
            for i, answer in enumerate(called, start=1)
        ]
        if record_receipts
        else []
    )
    if receipts:
        result.manifest.receipts.extend(receipts)
    self._recompute_manifest_accounting(result.manifest)  # re-stamps secret_safety
```
`synthesize_blocks` becomes a compatibility wrapper: `out = await self.adjudicate("synthesis", ...)`; return `out.answer` or a `ModelAnswer(name=self.synthesizer, model_id=resolve(self.synthesizer), error="no candidate in synthesizer chain has an API key")`. Import `FAILOVER_CATEGORIES` from `.models` and `AdjudicationAttempt, AdjudicationRole` from `.manifest`; `from dataclasses import dataclass`.

Attempt numbering for receipts: `attempt=i` where `i` counts *real calls*, matching how `extract_verdict` numbers its attempts.

**Step 4: Run** → PASS; `$BR` full → 0 failures (nothing calls `adjudicate` yet).

**Step 5: Commit** — `git commit -m "feat(council): adjudicate() succession seam + manifest ledger (DSE-1512)"`

---

### Task 5: Route synthesis, debate final, adversarial judge, and Elite through the seam

**Files:**
- Modify: `src/conclave/council.py` (`_synthesize`, `_ask_uncached`, `elite`)
- Modify: `src/conclave/modes.py` (`_debate_synthesize`, `run_debate`, `_adversarial_judge`, `run_adversarial`)
- Test: `tests/test_council.py`, `tests/test_modes.py`, `tests/test_manifest_all_modes.py` (append)

**Step 1: Failing tests** — one per mode; reuse the `_install`/`_fail`/`_ok` helpers by moving them into `tests/conftest.py` as `make_failed_answer(name, model_id, category, status=None)` / `make_ok_answer(name, model_id)` and a `patch_council_call_model(monkeypatch, script)` helper. Members need a script entry too (they go through the same council seam): give them `_ok`.

```python
async def test_synthesize_mode_fails_over_and_is_not_degraded(monkeypatch, keys):
    patch_council_call_model(monkeypatch, {"gemini": ok("gemini","gemini/m"), "claude": fail("claude","anthropic/c","quota",402), "grok": ok("grok","xai/g")})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.synthesis == "grok says yes" and r.synthesis_error is None and r.degraded is False
    assert (r.synthesizer, r.synthesizer_model_id) == ("grok", "xai/g")
    ledger = r.manifest.adjudication_succession
    assert [(a.role, a.candidate, a.outcome) for a in ledger] == [("synthesis","claude","failed_over"), ("synthesis","grok","success")]
    assert [(x.phase, x.attempt) for x in r.manifest.receipts if x.phase == "synthesis"] == [("synthesis",1), ("synthesis",2)]
    assert r.manifest.secret_safety == "verified_no_secrets"

async def test_synthesize_mode_exhausted_is_degraded(monkeypatch, keys):
    ... claude quota, grok unavailable → r.synthesis is None, r.synthesis_error == "grok failed", r.degraded is True,
        ledger outcomes == ["failed_over", "exhausted"]

async def test_synthesize_chain_of_one_message_unchanged(monkeypatch, clear_keys, patch_call_model):
    # exact string from the existing test_synthesizer_without_key_returns_raw must still hold
    ...

async def test_synthesize_chain_all_unkeyed_message(monkeypatch):
    # no keys for claude or grok, gemini keyed:
    assert r.synthesis_error == "synthesizer chain [claude, grok] has no API key for any candidate; returning raw answers only"
    assert [a.outcome for a in r.manifest.adjudication_succession] == ["skipped_unkeyed","skipped_unkeyed"]

async def test_debate_final_fails_over(monkeypatch, keys):
    # 2 members ok, chain claude(auth)>grok(ok), rounds=1 → r.synthesis from grok, ledger role == "debate_final"

async def test_adversarial_judge_fails_over(monkeypatch, keys):
    # proposer+critic ok, chain claude(503)>grok(ok) → r.adversarial.verdict from grok,
    # (r.adversarial.judge, r.adversarial.judge_model_id) == ("grok","xai/g"), r.degraded False, ledger role == "judge"

async def test_adversarial_judge_terminal_does_not_fail_over(monkeypatch, keys):
    # claude bad_request → adv.verdict_error set, grok never called, r.degraded True

async def test_elite_synthesis_fails_over(monkeypatch, keys):
    # 3 members ok through all phases (script ok for gemini/claude/grok as MEMBERS), chain "openai>grok"
    # with openai 429 → r.synthesis from grok, ledger role "synthesis", manifest has elite receipts + 2 synthesis receipts
```
Note for Elite: the synthesizer candidates must not collide with member names in the script dict unless you want the member to fail too — use distinct names (`openai`, `mistral`) for the chain and give them `models` entries in `CFG`.

**Step 2: Run** → FAIL.

**Step 3: Implement**

`Council._synthesize`:
```python
        usable = result.successful_answers
        if not usable:
            result.synthesis_error = "no successful member answers to synthesize"; ...; return None

        keyed = [c for c in self.synthesizer_chain if key_present(self.config.resolve_model_id(c))]
        result.synthesizer = self.synthesizer
        result.synthesizer_model_id = self.config.resolve_model_id(self.synthesizer)
        if not keyed:
            if len(self.synthesizer_chain) == 1:
                result.synthesis_error = (f"synthesizer '{self.synthesizer}' ({result.synthesizer_model_id}) has no API key; returning raw answers only")
            else:
                result.synthesis_error = (f"synthesizer chain [{', '.join(self.synthesizer_chain)}] has no API key for any candidate; returning raw answers only")
            logger.warning(result.synthesis_error)
            self._record_adjudication(result, self._skipped_attempts("synthesis"), [], phase="synthesis")
            return None

        ...build blocks/user_content exactly as today...
        outcome = await self.adjudicate("synthesis", _SYNTH_SYSTEM, user_content)
        result.synthesizer, result.synthesizer_model_id = outcome.name, outcome.model_id
        answer = outcome.answer
        if answer is not None and answer.ok:
            result.synthesis = answer.answer
        elif answer is not None:
            result.synthesis_error = answer.error
        self._record_adjudication(result, outcome.attempts, outcome.called, phase="synthesis",
                                  protocol_version=(ELITE_PROTOCOL_VERSION if result.mode == "elite" else None))
        return answer
```
`_skipped_attempts(role)` builds one `skipped_unkeyed` attempt per chain candidate. In `_ask_uncached` **delete** the manual `_append_manifest_receipts([...synthesis receipt...])` block — `_record_adjudication` now owns the synthesis receipts (otherwise they double). Same deletion in `elite()`, and move `self._ensure_manifest(result, "elite")` to **before** `self._synthesize(result)` so the ledger has a manifest to land on.

`modes.run_debate`: insert `council._ensure_manifest(result, "debate")` immediately before `await _debate_synthesize(council, result)`. `_debate_synthesize` mirrors `_synthesize` (keyed check with the chain-of-one message preserved: `"...has no API key; returning final-round answers only"`), then `outcome = await council.adjudicate("debate_final", prompts.DEBATE_FINAL_SYSTEM, user_content)` and `council._record_adjudication(result, outcome.attempts, outcome.called, phase="debate_final", prompt_version=None)`.

`modes.run_adversarial`: insert `council._ensure_manifest(result, "adversarial")` immediately before `await _adversarial_judge(council, prompt, adv)`. `_adversarial_judge(council, prompt, adv, result)` — add the `result` parameter so it can record; keep the proposal-failed and unkeyed short-circuits (chain-aware wording as above, `"...returning proposal and critiques only"`); then `outcome = await council.adjudicate("judge", prompts.JUDGE_SYSTEM, user_content)`; `adv.judge, adv.judge_model_id = outcome.name, outcome.model_id`; verdict / verdict_error from the answer; `council._record_adjudication(result, outcome.attempts, outcome.called, phase="judge", prompt_version=None)`.

`_degrade_to_synthesize` needs no change (`_synthesize` returns before consulting anyone when there are no usable answers).

**Step 4: Run** the new tests → PASS. Then `$BR` full. Expect `tests/test_manifest_all_modes.py` to flag that `debate`/`adversarial` manifests now carry a judge/synthesizer receipt they previously lacked. That is a **deliberate, documented improvement** (H0 principle: every real call gets a receipt). Update those assertions (count + phase) — do not weaken them — and add a line to the CHANGELOG entry in Task 9.

**Step 5: Commit** — `git commit -m "feat(council,modes): route all adjudication roles through the succession seam (DSE-1512)"`

---

### Task 6: Verdict extraction succession

**Files:**
- Modify: `src/conclave/verdict_synthesis.py` (`VerdictSynthesisResult`, tail of `extract_verdict`)
- Modify: `src/conclave/council.py` (`_apply_verdict`)
- Test: `tests/test_council_verdict.py` (append)

**Step 1: Failing tests** — drive the verdict seam (`conclave.verdict_synthesis.call_model`) with a handler that branches on `name`: primary returns an errored `ModelAnswer(failure_category="auth", http_status=401)` for both the initial and repair calls; successor returns valid extraction JSON (copy the fixture JSON `test_council_verdict.py` already uses).

```python
async def test_verdict_extraction_fails_over_to_successor(...):
    r = await Council(models=[...], synthesizer="claude>grok", config=CFG).ask("Should we X?")
    assert r.verdict is not None
    assert r.manifest.verdict_extraction.model_id == "xai/g"
    ledger = [a for a in r.manifest.adjudication_succession if a.role == "verdict_extraction"]
    assert [(a.candidate, a.outcome, a.failure_category) for a in ledger] == [("claude","failed_over","auth"), ("grok","success",None)]
    phases = [(x.phase, x.name) for x in r.manifest.receipts if x.phase.startswith("verdict")]
    assert phases == [("verdict_extraction","claude"), ("verdict_repair","claude"), ("verdict_extraction","grok")]

async def test_verdict_extraction_schema_failure_is_terminal(...):
    # primary returns prose twice (schema_invalid), successor would return valid JSON → successor NOT called,
    # r.verdict is None, verdict_absent_reason == "verdict extraction failed schema validation",
    # ledger == [("claude","terminal_failure","malformed_response")]
```

**Step 2: Run** → FAIL.

**Step 3: Implement**

`VerdictSynthesisResult` gains:
```python
    failure_category: str | None = None
    http_status: int | None = None
```
populated only on the `_REASON_EXTRACTION_FAILED` return: take them from the **last** attempt that errored (`retry` if the repair ran and `retry.error`, else `answer` if `answer.error`); when the last attempt *responded* but failed validation, set `failure_category="malformed_response"`.

`Council._apply_verdict` — replace the single call with a chain walk (verdict extraction receipts are already produced per attempt by `extract_verdict`; append them per candidate exactly as today):
```python
        chain = self.synthesizer_chain
        attempts: list[AdjudicationAttempt] = []
        vsr = None
        for index, candidate in enumerate(chain, start=1):
            model_id = self.config.resolve_model_id(candidate)
            if not key_present(model_id):
                attempts.append(skipped_unkeyed attempt); continue
            vsr = await extract_verdict_fn(result.prompt, result.answers, synthesizer_name=candidate,
                                           synthesizer_model_id=model_id, config=self.config,
                                           temperature=self.temperature, timeout=self.timeout,
                                           protocol_version=(ELITE_PROTOCOL_VERSION if result.mode == "elite" else None))
            if record_receipts:
                self._append_manifest_receipts(result, vsr.attempt_receipts)
            if vsr.verdict_absent_reason == _REASON_EXTRACTION_FAILED and vsr.failure_category in FAILOVER_CATEGORIES:
                is_last = index == len(chain)
                attempts.append(failed_over / exhausted attempt with vsr.failure_category, vsr.http_status)
                continue
            attempts.append(success attempt if vsr.verdict is not None or vsr.verdict_absent_reason in (N<2, open-ended)
                            else terminal_failure attempt with failure_category=vsr.failure_category)
            break
        if vsr is None:      # every candidate unkeyed -- today's behaviour was to call anyway and fail; keep verdict absent
            ...set result.manifest.verdict_absent_reason = "verdict extractor has no API key" ; record attempts; return
        ...hoist vsr fields exactly as today...
        if result.manifest is not None:
            result.manifest.adjudication_succession.extend(attempts)
            ...existing provenance writes + re-stamp...
```
Import `_REASON_EXTRACTION_FAILED` from `verdict_synthesis` (it is module-private today; export it as `REASON_EXTRACTION_FAILED` and keep the old name as an alias). Note the N<2 gate makes no call and must count as `success` for the ledger only when it is the *first* keyed candidate (it will always be — the gate is answer-driven, not model-driven); simplest: if `vsr.attempt_receipts` is empty, record the attempt as `success` and break.

**Step 4: Run** → PASS; `$BR` full → 0 failures.

**Step 5: Commit** — `git commit -m "feat(verdict): fail verdict extraction over on infra errors (DSE-1512)"`

---

### Task 7: Streaming parity

**Files:**
- Modify: `src/conclave/streaming.py:252-322`
- Test: `tests/test_streaming.py` (append)

**Step 1: Failing tests** — patch `conclave.streaming.call_model_stream` with an async generator that, for the primary, yields **only** a final errored `ModelAnswer(failure_category="quota", http_status=429)` (no deltas), and for the successor yields deltas then an ok answer.

```python
async def test_stream_synthesis_fails_over_before_first_delta(...):
    events = [e async for e in council.ask_stream("q")]
    deltas = [e.text for e in events if e.type == "synthesis_delta"]
    done = [e for e in events if e.type == "synthesis_done"]
    assert "".join(deltas) == "grok says yes" and len(done) == 1 and done[0].name == "grok"
    result = events[-1].result
    assert result.synthesis == "grok says yes" and result.degraded is False
    assert [(a.role, a.candidate, a.outcome) for a in result.manifest.adjudication_succession][:2] == [("synthesis","claude","failed_over"), ("synthesis","grok","success")]

async def test_stream_synthesis_does_not_fail_over_after_deltas(...):
    # primary yields one delta THEN an errored answer with category "unavailable" → terminal; successor never consulted;
    # result.synthesis_error set; ledger outcome "terminal_failure"
```

**Step 2: Run** → FAIL.

**Step 3: Implement** `_stream_synthesis`: keep the usable/keyed short-circuits (chain-aware, same messages as `_synthesize`); then loop the chain; per candidate stream; track `emitted = False`; set `emitted = True` on the first delta; on the final answer: ok → success/break; error with `failure_category in FAILOVER_CATEGORIES and not emitted` → `failed_over` (or `exhausted` on the last), continue; else → `terminal_failure`, break. Yield a single `synthesis_done` for the last consulted candidate. Set `result.synthesizer`/`synthesizer_model_id` to that candidate. Record via `council._record_adjudication(result, attempts, called, phase="synthesis", record_receipts=False)` — streaming keeps its documented "no synthesis receipt" contract; the ledger is not a receipt.

**Step 4: Run** → PASS; `$BR` full → 0 failures.

**Step 5: Commit** — `git commit -m "feat(streaming): synthesis succession before the first delta (DSE-1512)"`

---

### Task 8: Cache identity + no-store on succession

**Files:**
- Modify: `src/conclave/cache.py:59,148-296`
- Modify: `src/conclave/council.py` (`_cache_key`, `_cached_run`)
- Test: `tests/test_cache.py` (append)

**Step 1: Failing tests**

```python
def test_identity_includes_full_chain():
    a = make_key(prompt="p", mode="synthesize", members=[("g","xai/g")], synthesizer="claude", synthesizer_model_id="anthropic/c", synthesizer_chain=[("claude","anthropic/c")], temperature=0.7)
    b = make_key(..., synthesizer_chain=[("claude","anthropic/c"), ("grok","xai/g")])
    assert a != b

def test_cache_format_version_bumped():
    assert CACHE_FORMAT_VERSION == "4"

async def test_result_adjudicated_by_successor_is_not_stored(monkeypatch, tmp_path, keys):
    # cache on, claude auth-fails, grok succeeds → run twice; second run must call the providers again
    # (cache_mod.store was skipped); assert both results have cached is False and the counting fake saw 2x calls
```

**Step 2: Run** → FAIL.

**Step 3: Implement** — `build_identity`/`make_key` gain `synthesizer_chain: list[tuple[str, str]] | None = None` and write `"synthesizer_chain": [[n, m] ...]` (keep the existing `"synthesizer"` key). `CACHE_FORMAT_VERSION = "4"`. `Council._cache_key` passes `[(c, self.config.resolve_model_id(c)) for c in self.synthesizer_chain]` and includes every chain prefix in `used_prefixes`. In `_cached_run`, before `cache_mod.store`:
```python
        if result.manifest is not None and any(
            a.outcome == "failed_over" for a in result.manifest.adjudication_succession
        ):
            logger.info("not caching %s run: adjudicated by a successor after failover", mode)
            return result
```

**Step 4: Run** → PASS; `$BR` full → 0 failures.

**Step 5: Commit** — `git commit -m "feat(cache): chain in identity; never cache a successor-adjudicated run (DSE-1512)"`

---

### Task 9: CLI surface + docs + changelog

**Files:**
- Modify: `src/conclave/cli.py:497-500` (help), `:741` (`providers` footer)
- Modify: `README.md` (synthesizer section), `docs/PRODUCT_DESIGN_DOCUMENT.md` §4a (manifest table + a short "Adjudication succession" paragraph), `CHANGELOG.md` (Unreleased), `DOCUMENTATION_INDEX.md` (link this plan), `SYSTEM_CONTEXT_DIAGRAM.md` (only if it enumerates manifest fields)
- Test: `tests/test_cli.py` (append)

**Step 1: Failing tests**
```python
def test_cli_synthesizer_chain_parses_and_exits_zero_on_successor(...):
    # patch council seam: claude 402, grok ok; run `ask q -c gemini -s "claude>grok" --json`
    assert result.exit_code == 0 and payload["degraded"] is False and payload["synthesizer"] == "grok"
    assert payload["manifest"]["adjudication_succession"][0]["outcome"] == "failed_over"

def test_cli_chain_exhausted_exits_degraded(...):
    assert result.exit_code == cli._DEGRADED_EXIT_CODE
```

**Step 2: Run** → FAIL only if the CLI mangles the `>`; otherwise these pass immediately — that is fine, keep them as the contract.

**Step 3: Implement** — `--synthesizer` help: `"Synthesizer/judge model name, or an ordered failover ladder 'claude>grok>gemini' (DSE-1512): the next candidate is tried only on auth/quota/5xx/timeout/network failures."` `conclave providers` footer prints `synthesizer chain: a > b` when `cfg.synthesizer_chain` is non-empty. Docs:

- **CHANGELOG `[Unreleased]` → Added:** "Adjudication succession (DSE-1512)" — chain config/CLI, the infra-only rule, `manifest.adjudication_succession`, per-role application, cache no-store rule, `ModelAnswer.failure_category`/`http_status`; **Changed:** debate/adversarial manifests now carry the judge/synthesizer receipt(s) they previously omitted; cache format version 3 → 4 (old entries miss safely); **Not changed:** the verdict repair retry still runs on the same model after an infra error (follow-up).
- **README:** a "Synthesizer failover" subsection with the CLI form, the YAML form, the rule table (which categories advance), and one sentence on reading `manifest.adjudication_succession`.
- **PDD §4a:** add `adjudication_succession` to the manifest description; add the failover rule as a design decision (why content failures never fail over: reproducibility).
- **DOCUMENTATION_INDEX.md:** add `docs/plans/2026-09-03-adjudication-succession.md`.

**Step 4: Run** `$BR` full + ruff. **Step 5: Commit** — `git commit -m "docs(cli): synthesizer chain surface, changelog, PDD §4a (DSE-1512)"`

---

### Task 10: Ship

1. `git push -u origin feat/dse-1512-adjudication-succession` (from the laptop).
2. `gh pr create --title "feat: adjudication succession — judge/synthesizer failover on infrastructure errors (DSE-1512)" --body-file <body>` — body: summary, the rule table, the receipt-completeness change, the follow-up note, `Closes DSE-1512`, the required attribution footer.
3. Wait for `Test` (3.11/3.12/3.13), `ruff`, `pip-audit`, `Gitleaks` — all green.
4. `python3 ~/.claude/scripts/release_control.py classify` on the complete diff → must be `routine-non-security`.
5. `python3 ~/.claude/scripts/release_control.py merge --repo DataScience-EngineeringExperts/conclave --pr <n> --head-sha <40> --method squash`.
6. Linear DSE-1512 → Done with the merge SHA.
