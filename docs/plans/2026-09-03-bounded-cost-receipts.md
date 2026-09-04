# Bounded Cost Receipts + Pre-flight Spend Gate Implementation Plan (DSE-1514)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make a council run say — provably, in its own audit receipt — *"this cost no more than $X, priced against snapshot `<digest>` dated `<date>`"*, and let an operator refuse a run whose worst-case call plan would exceed a stated dollar cap **before the first provider call**.

**Architecture:** (1) The `Decimal`/`ROUND_CEILING` arithmetic currently quarantined in `conclave/evals/pricing.py` moves to a new shared `conclave/pricing.py`; `evals/` becomes a thin consumer with byte-identical behaviour. (2) A dated, hand-verified, checked-in price snapshot (`src/conclave/data/prices-YYYY-MM-DD.json`) is loaded once per process; `Council._price_manifest()` runs **last** — after every receipt is on the manifest — and stamps a per-receipt `cost_ceiling_usd` plus run-level ceiling/digest/date/unpriced fields under an **all-or-nothing** rule. (3) `max_output_tokens` becomes a real, threaded council setting (it exists only at the adapter layer today), because a ceiling on output is the precondition for a bound. (4) `Council.plan_calls()` enumerates the worst-case call plan per mode from the actual `modes.py` arithmetic; `Council._reserve_plan()` prices it pessimistically; over cap → `SpendCapExceeded` → CLI exit **4**, raised before any transport touch. (5) The snapshot digest joins cache identity so a cache hit can never serve a ceiling priced under a different snapshot.

**Tech Stack:** Python 3.11+, pydantic v2, `decimal` (`ROUND_CEILING`, `localcontext`), typer (CLI), hatchling (package data), pytest + pytest-asyncio (fully offline — see `tests/conftest.py`).

**Branch:** `feat/dse-1514-bounded-cost-receipts`, cut from `feat/dse-1512-adjudication-succession` @ `b2d7006` (NOT `main`). Rebase onto `main` after DSE-1512's PR #63 lands (squash).

**Execution locality:** edit + git on the laptop worktree `~/dev/worktrees/conclave-dse-1514`; run **every** test/lint on the builder via `~/.claude/scripts/builder-run.sh conclave-dse-1514 '<cmd>'` (it rsyncs first). Never run pytest/ruff/pip on the laptop.
Shorthand used throughout: `BR='~/.claude/scripts/builder-run.sh conclave-dse-1514'`.

**Baseline (verified 2026-09-03 on `util-prov01`):** `842 passed in 44.69s`. Every task below states the expected new total; the suite must never go backwards.

**Release classification (expected, not a defect):** `release_control.py classify` will almost certainly return **`security-specific`** for these diffs. The same vocabulary rule that hit DSE-1512 applies — the diff contains `provenance`, `token`, `validation`, `secret_safety`, and `key_present`. Nothing here changes authentication, authorization, secret handling, redaction, or network exposure. Merge therefore needs **one authenticated human receipt per PR** under `~/.claude/rules/release-control.md`. Plan for it; do not treat it as a finding.

---

## The resolved design question (Round 1's deliverable)

The ticket demands a decision on the input/output bound and an explicit refusal rule. Here it is, in three sentences:

1. **Input is bounded with zero model calls** by counting **UTF-8 bytes and treating each byte as at most one token** — the same attestation `conclave/evals/live.py` already ships — decomposed into the exact prompt bytes, the fixed system/template bytes, a provider framing allowance (`64 + 16 x message_count`, `+256` when a structured-output contract is attached), and, for any call whose input embeds a prior model's output, `sum(upstream output caps) x max_output_bytes_per_token` drawn from the snapshot entry.
2. **Output is bounded only by an explicit `max_output_tokens`** — there is no default, none is inferred from a provider's own limit, and none is invented.
3. **Refusal rule (all three exit `4`, before any provider call):** `--max-spend-usd` with no output cap refuses with `cannot bound spend: no output cap (set --max-output-tokens or config max_output_tokens)`; a plan containing any call whose model id is absent from the snapshot (or with no snapshot at all) refuses with `cannot bound spend: no priced rate for <model_id> in snapshot <digest> (<date>)`; and a fully bounded, fully priced plan whose reservation exceeds the cap refuses with the reserved total, the cap, and the call count.

---

## Where this plan departs from the pre-agreed design notes

The design notes at `dse-1514-design-notes.md` are binding except where the code makes them wrong. Three items are wrong, verified against `modes.py` and `council.py` on this branch:

| Note | What it said | What the code says | Resolution |
|---|---|---|---|
| #6, adversarial | `up to N proposer attempts + (N-1) critics + 1 judge + 2` | `run_adversarial` builds `tried = {failed proposers} | {winner}` and sets `critics = [m for m in members if m.name not in tried]`. A failed proposer is **never** also a critic. Proposer attempts `k` + critics `N-k` = exactly `N`. And `Council.adversarial` funnels through `_cached_run`, which calls `_ensure_manifest` only — **`_apply_verdict` is never called for `adversarial`**, so there is no `+2`. | Adversarial worst case is **`N + C`**. |
| #6, debate | `N*rounds + 1 + 2` | `Council.debate` also goes through `_cached_run`; `_apply_verdict` is never called for `debate` either. `_debate_synthesize` walks the chain, giving `C`. | Debate worst case is **`N*R + C`**. |
| #6, chain multiplier | "each adjudication role x `len(synthesizer_chain)` worst case" | `Council.adjudicate` **skips an unkeyed candidate without any call** (`if not key_present(model_id): ... continue`). An unkeyed candidate cannot cost money. | The multiplier is `C` = the number of **keyed** candidates in `synthesizer_chain`, not `len(synthesizer_chain)`. This is exact, not merely conservative. Note `registry.key_present` returns `True` for an unknown provider prefix, which errs toward counting the call — the safe direction. |

Two smaller clarifications, both recorded so the implementer does not have to re-derive them:

* **`_apply_verdict` does not pre-skip unkeyed candidates** (it deliberately lets `extract_verdict` run so the chain-of-one receipts stay byte-identical), but those calls return instantly from `call_model`'s no-key branch **with no network**, so they cost nothing and are excluded from the plan on the same `key_present` basis.
* **`unpriced_models` scope.** The ticket says "every model in the roster"; a member skipped for a missing key made no call and cannot be billed. This plan defines it as: **every distinct model id that produced a receipt, unioned with `manifest.model_ids` (= the members actually called), that has no snapshot entry.** Skipped-for-no-key members are in `providers_skipped`, never in `model_ids`, so they are correctly excluded. State this in the PDD.

---

## Worst-case call plan per mode (derived from `modes.py`, not from memory)

Let **N** = `len(Council._available_members()[0])` (keyed members). Let **C** = the number of `synthesizer_chain` candidates with `key_present(resolve_model_id(c))`. Let **R** = `rounds` (debate). Let **V** = 1 when `self.extract_verdict_enabled` else 0.

| Mode | Worst-case calls | Derivation |
|---|---|---|
| `raw` | `N` | `_ask_uncached` fan-out only; `synthesize=False` skips both `_synthesize` and `_apply_verdict`. |
| `synthesize` | `N + C + 2*C*V` | fan-out `N`; `_synthesize` walks up to `C` keyed candidates; `_apply_verdict` calls `extract_verdict` once per candidate and each one makes **1 initial + 1 repair** = `2C`. |
| `vote` | `N` | `run_vote` fans out once; no adjudication, no verdict. |
| `debate` | `N*R + C` | round 1 = `N`; rounds 2..R have at most `N` survivors each (drop-out only shrinks it) giving `N*R`; `_debate_synthesize` adds `C`. `converge_threshold` can only stop **early**. No verdict extraction. |
| `adversarial` | `N + C` | proposer attempts `k` (1..N) plus critics `N - k` equals `N` regardless of `k`; `_adversarial_judge` adds `C`. The all-proposers-fail path degrades to `_synthesize` over an empty `successful_answers`, which returns **before** any call, so still `N`. No verdict extraction. **Byte shape, corrected 2026-09-04 (Round 4 review, Fix A):** the count `N + C` was always right, but the ORIGINAL worked example in Task 10 gave every member call `upstream=0`, undercounting the input bound of every critic call. The byte-worst-case plan is 1 proposer call (`upstream=0`) + `N-1` critic calls (`upstream=1` each -- `_critic_messages_for` embeds the proposal's answer text in every critic call), plus one judge call per keyed chain candidate (`upstream=N` -- `judge_user` embeds the proposal and every critique). |
| `elite` | `3N + C + 2*C*V` | `run_elite` fans out three times (initial / critique / revision), each at most `N`, giving `3N`; then `_synthesize` adds `C` and `_apply_verdict` adds `2C`. With a chain of one and verdict on: `3N + 3` — the ticket's `3N + 2` shape **plus the repair retry**, which is exactly the call the ticket says must not be forgotten. |

Sanity check against the ticket: `synthesize` with `C=1, V=1` gives `N + 3`; `elite` with `C=1, V=1` gives `3N + 3`. Both match.

---

## Ground rules (read before Task 1)

- **`estimated_cost` stays `None` forever.** On the receipt and on the manifest. It is never assigned, never summed into, never renamed. A ceiling is a different claim from an estimate; the two must not be conflated in one field.
- **No float in rate math, anywhere.** `Decimal` in, `Decimal` out, `ROUND_CEILING`, quantized to `USD_MICROCENT` (`Decimal("0.000001")`). The CLI takes `--max-spend-usd` as a **`str`** and constructs `Decimal(value)` — never `float`, never typer float coercion. `Decimal(0.4)` is `0.40000000000000002220446` and would silently make the cap wrong.
- **All-or-nothing.** Any unpriced receipt or any unpriced model in the run means run-level `cost_ceiling_usd = None`, with `unpriced_models` / `unpriced_receipts` populated. Never a partial sum. A partial total reads exactly like a complete one — that is the failure mode the whole ticket exists to prevent.
- **Never invent a rate.** A model absent from the snapshot is unpriced. Never fall back to a "similar" model, a provider average, or a stale entry. A snapshot older than `PRICE_SNAPSHOT_MAX_AGE_DAYS = 90` emits a bounded warning string and keeps pricing at its stated rates — it never substitutes a rate and never silently suppresses the ceiling.
- **`pricing_warnings` carries fixed identifiers only.** The permitted values are exactly `"price_snapshot_stale"`, `"price_snapshot_unavailable"`, `"unpriced_models_present"`, `"unpriced_receipts_present"`, `"no_output_cap_configured"`. No interpolation, no provider text, no counts. `scan_for_secret_material()` must keep stamping VERIFIED.
- **Pricing never crashes a run.** `_price_manifest` catches `ValueError` / `OSError` / `json.JSONDecodeError` and degrades to "unpriced" with a bounded warning. A bad snapshot file is a missing ceiling, never a failed council run. (The spend **gate** is the opposite: it refuses rather than degrades.)
- **No spend flags means byte-identical behaviour to today,** except the additive manifest fields. With `max_output_tokens is None` nothing new reaches `call_model`; with no snapshot entry nothing new appears beyond `None`s. `origin/main`'s suite is the regression oracle.
- **Do not touch** `redact()`, `scan_for_secret_material()`, `verified_secret_safety()`, `_receipt_error_category()`, `_resolve_key()`, `key_present()`, `registry.py`, or any transport raise site. Reading them is fine; changing one re-classifies the PR and invalidates its approval.
- **Round 2 is a pure refactor.** No new behaviour, no new fields, no new tests beyond a delegation-identity test. `tests/evals/test_pricing.py` and every other eval test must pass **unchanged** — their assertions include `EXPECTED_PRICE_HASH = "sha256:46a29d0180e897fd8ba315781b9180121a4509226b8dfb8084164515e0efa53f"`, which is computed over `ModelPrice.schema_version` (`"conclave_eval_v1"`). If `hash_price_entries` or `ModelPrice`'s field set changes, that hash changes and the eval harness's frozen study designs break. Leave them alone.
- **TDD.** Failing test, run it, minimal implementation, run it, commit with the given message. Commit after every task.
- **Reading the code blocks in this plan.** `ruff format` (0.16) normalizes Python fenced blocks inside Markdown, including dedenting them to module level, and CI runs `ruff format --check .` over the repo — so a method shown here appears at column 0 even though it belongs inside a class. The prose immediately above each block says where it goes, and a leading `self` parameter confirms it is a method; re-indent by four spaces when you paste it. Blocks that are pure fragments (a parameter to insert into an existing signature, a dict entry) are fenced as `text` for the same reason: ruff would silently rewrite `foo: int | None = None,` into a tuple.

### Existing seams you will rely on

| Seam | Where | Note |
|---|---|---|
| Member / adjudicator call | `conclave.council.call_model` | patched by the `patch_call_model` fixture and by `install_council_script` |
| Verdict call | `conclave.verdict_synthesis.call_model` | autouse offline stub in `tests/conftest.py` |
| Streaming call | `conclave.streaming.call_model_stream` | patched in `tests/test_streaming.py` |
| Transport | `conclave.transport.post_json` | the only real network edge; a gate test asserts it is never reached |
| Key presence | `conclave.registry.key_present` | env-var **name** check; unknown prefixes return `True` |
| Manifest aggregates | `Council._recompute_manifest_accounting` | a `@staticmethod` with no snapshot access — this is *why* pricing is a separate late step |

---

## Shared-surface note (three tickets, three seams)

DSE-1512, DSE-1514 (this), and DSE-1517 all add fields to `ModelHarnessManifest`, all extend `cache.build_identity`, and all must survive the `secret_safety` re-stamp.

* **DSE-1512 lands first** and is already merged into this branch's base (`b2d7006`): it owns `adjudication_succession`, `synthesizer_chain` in identity, and `CACHE_FORMAT_VERSION = "4"`.
* **DSE-1514 (this ticket)** takes `CACHE_FORMAT_VERSION` from `"4"` to `"5"` and adds `price_snapshot_fingerprint` plus `generation.max_output_tokens` to `build_identity`. Rebase onto `main` after PR #63 squash-merges; the only expected conflicts are `CHANGELOG.md`'s `[Unreleased]` block and the `cache.py` version comment.
* **DSE-1517 lands after this** and must bump `"5"` to `"6"`. Leave the version comment in `cache.py` in the running-history style DSE-1512 established so 1517 can append one line.

---

# Round 2 — Extract the pricing primitives (PR #1, pure refactor)

**Scope:** move arithmetic to `src/conclave/pricing.py`; `evals/` delegates; zero behaviour change. Expected suite after Round 2: **849 passed**.

---

### Task 1: The shared arithmetic module

**Files:**
- Create: `src/conclave/pricing.py`
- Test: `tests/test_pricing_core.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_pricing_core.py
"""Shared Decimal/ROUND_CEILING price arithmetic (DSE-1514, Round 2)."""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

import pytest
from pydantic import ValidationError

from conclave.pricing import (
    USD_MICROCENT,
    PriceRates,
    reported_usage_cost,
    reserve_cost,
)

RATES = PriceRates(
    provider_id="fictional-provider-a",
    model_id="fictional-provider-a/fictional-model-a",
    input_ceiling_usd_per_million_tokens=Decimal("1.234567"),
    output_ceiling_usd_per_million_tokens=Decimal("4.567891"),
    max_output_bytes_per_token=4,
)


def test_rates_reject_float_input_and_output_ceilings():
    with pytest.raises(ValidationError):
        PriceRates(
            provider_id="p",
            model_id="p/m",
            input_ceiling_usd_per_million_tokens=1.23,
            output_ceiling_usd_per_million_tokens=Decimal("2"),
            max_output_bytes_per_token=4,
        )
    with pytest.raises(ValidationError):
        PriceRates(
            provider_id="p",
            model_id="p/m",
            input_ceiling_usd_per_million_tokens=Decimal("1"),
            output_ceiling_usd_per_million_tokens=2.5,
            max_output_bytes_per_token=4,
        )


def test_reserve_cost_covers_prompt_template_framing_and_upstream():
    amounts = reserve_cost(
        RATES,
        prompt_token_upper_bound=1_000,
        prompt_template_token_allowance=100,
        provider_framing_token_allowance=96,
        upstream_output_token_ceilings=(500, 500),
        upstream_output_bytes_per_token=4,
        max_output_tokens=2_000,
    )
    assert amounts.input_token_upper_bound == 1_000 + 100 + 96 + (1_000 * 4)
    assert amounts.output_token_upper_bound == 2_000
    assert amounts.reserved_cost_usd == (
        amounts.input_cost_upper_bound_usd + amounts.output_cost_upper_bound_usd
    ).quantize(USD_MICROCENT, rounding=ROUND_CEILING)
    assert isinstance(amounts.reserved_cost_usd, Decimal)


def test_reserve_cost_always_rounds_up():
    tiny = PriceRates(
        provider_id="p",
        model_id="p/m",
        input_ceiling_usd_per_million_tokens=Decimal("1"),
        output_ceiling_usd_per_million_tokens=Decimal("1"),
        max_output_bytes_per_token=4,
    )
    amounts = reserve_cost(
        tiny,
        prompt_token_upper_bound=1,
        prompt_template_token_allowance=0,
        provider_framing_token_allowance=0,
        upstream_output_token_ceilings=(),
        upstream_output_bytes_per_token=4,
        max_output_tokens=1,
    )
    # 2 tokens at $1/M is $0.000002 exactly; a sub-microcent residue must round UP.
    assert amounts.reserved_cost_usd == Decimal("0.000002")


def test_reported_usage_cost_charges_unattributed_at_the_higher_rate():
    cost = reported_usage_cost(
        RATES,
        prompt_tokens=1_000,
        completion_tokens=2_000,
        total_tokens=3_500,
    )
    expected = (
        Decimal(1_000) * RATES.input_ceiling_usd_per_million_tokens
        + Decimal(2_000) * RATES.output_ceiling_usd_per_million_tokens
        + Decimal(500) * RATES.output_ceiling_usd_per_million_tokens
    ) / Decimal(1_000_000)
    assert cost == expected.quantize(USD_MICROCENT, rounding=ROUND_CEILING)


def test_reported_usage_cost_rejects_impossible_totals():
    with pytest.raises(ValueError, match="smaller than its attributed usage"):
        reported_usage_cost(RATES, prompt_tokens=10, completion_tokens=10, total_tokens=5)


def test_token_bounds_reject_bools_and_negatives():
    with pytest.raises(TypeError):
        reserve_cost(
            RATES,
            prompt_token_upper_bound=True,
            prompt_template_token_allowance=0,
            provider_framing_token_allowance=0,
            upstream_output_token_ceilings=(),
            upstream_output_bytes_per_token=4,
            max_output_tokens=1,
        )
    with pytest.raises(ValueError, match="max_output_tokens must be positive"):
        reserve_cost(
            RATES,
            prompt_token_upper_bound=0,
            prompt_template_token_allowance=0,
            provider_framing_token_allowance=0,
            upstream_output_token_ceilings=(),
            upstream_output_bytes_per_token=4,
            max_output_tokens=0,
        )
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_core.py -q -p no:cacheprovider'
```
Expected: `ModuleNotFoundError: No module named 'conclave.pricing'` — 6 errors at collection.

**Step 3: Write the implementation**

```python
# src/conclave/pricing.py
"""Shared, float-free price arithmetic for ceilings and reservations (DSE-1514).

This module owns the ONE implementation of conclave's cost arithmetic. Both the
paid eval harness (:mod:`conclave.evals.pricing`, :mod:`conclave.evals.live`)
and the product run path (:mod:`conclave.council`) delegate to it, so a ceiling
computed for an audit receipt and a reservation computed for a paid eval cell
can never drift apart.

Three invariants hold everywhere in here and must survive every future edit:

* **No floats.** Rates are exact :class:`decimal.Decimal`; a ``float`` (or
  ``bool``) rate is rejected at validation time, not coerced.
* **Always up.** Every emitted amount is quantized to :data:`USD_MICROCENT`
  with :data:`decimal.ROUND_CEILING`. A number produced here is an upper bound,
  never a best guess, so it is safe to put inside an audit receipt.
* **Bytes bound tokens.** An input bound is computed from UTF-8 *bytes* with
  each byte treated as at most one token, and an upstream model's output is
  bounded by its output cap times ``max_output_bytes_per_token``. Both
  directions are deliberately pessimistic; this is the attestation the paid
  eval runner has shipped since the H1 live runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The quantum every emitted amount is rounded up to: one USD micro-cent.
USD_MICROCENT = Decimal("0.000001")
TOKENS_PER_MILLION = Decimal(1_000_000)


class PriceRates(BaseModel):
    """Pessimistic ceiling rates for one model, as exact decimals.

    This is the arithmetic input type shared by the eval harness and the product
    run path. ``model_id`` is the FULL provider-prefixed id the product uses
    (``"anthropic/claude-sonnet-4-6"``); the eval harness passes its own
    ``model_id`` through :meth:`conclave.evals.pricing.ModelPrice.as_rates`.

    Attributes:
        provider_id: Provider prefix (``"anthropic"``).
        model_id: Full model identifier. For the product snapshot this is the
            provider-prefixed id resolved by ``ConclaveConfig.resolve_model_id``.
        input_ceiling_usd_per_million_tokens: Exact upper-bound input rate.
        output_ceiling_usd_per_million_tokens: Exact upper-bound output rate.
        max_output_bytes_per_token: Attested upper bound on the UTF-8 byte
            length of one output token, used to bound a DOWNSTREAM call whose
            input embeds this model's output.
        source_url: Where the rate was read from. Required by the product
            snapshot loader; ``None`` for eval-derived rates, which are bound to
            a frozen study design instead. Deliberately NOT part of
            :meth:`PriceSnapshot.digest` -- correcting a citation must not
            invalidate every cache entry priced under identical rates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    input_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    output_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    max_output_bytes_per_token: int = Field(gt=0, strict=True)
    source_url: str | None = None

    @field_validator(
        "input_ceiling_usd_per_million_tokens",
        "output_ceiling_usd_per_million_tokens",
        mode="before",
    )
    @classmethod
    def require_exact_decimal_rate(cls, value: object) -> object:
        """Reject float/bool rates outright rather than coercing them.

        ``Decimal(0.4)`` is ``0.4000000000000000222``; a rate that has been
        through a float is no longer the number the vendor published, and a
        ceiling built on it is not falsifiable. Mirrors
        ``conclave.evals.pricing.ModelPrice`` byte for byte.
        """
        if isinstance(value, (bool, float)):
            raise ValueError("price rates must be exact decimal values, not floats")
        return value


@dataclass(frozen=True)
class ReservationAmounts:
    """The pessimistic token and USD bounds for one planned provider call."""

    input_token_upper_bound: int
    output_token_upper_bound: int
    input_cost_upper_bound_usd: Decimal
    output_cost_upper_bound_usd: Decimal
    reserved_cost_usd: Decimal


def validate_token_bound(name: str, value: int, *, positive: bool = False) -> None:
    """Reject a non-integer, boolean, or out-of-range token bound.

    ``bool`` is an ``int`` subclass in Python, so ``isinstance(True, int)`` is
    ``True``; a silently-accepted ``True`` would become the token count ``1``.

    Args:
        name: Parameter name, used verbatim in the raised message.
        value: The candidate bound.
        positive: When ``True`` the bound must be at least ``1`` rather than ``0``.

    Raises:
        TypeError: ``value`` is not an ``int`` (or is a ``bool``).
        ValueError: ``value`` is below the permitted minimum.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        comparator = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {comparator}")


def _precision_for(*pairs: tuple[int, Decimal]) -> int:
    """Return a decimal context precision that cannot lose a significant digit."""
    return max(64, *(len(str(count)) + len(rate.as_tuple().digits) + 20 for count, rate in pairs))


def reserve_cost(
    rates: PriceRates,
    *,
    prompt_token_upper_bound: int,
    prompt_template_token_allowance: int,
    provider_framing_token_allowance: int,
    upstream_output_token_ceilings: Sequence[int],
    upstream_output_bytes_per_token: int,
    max_output_tokens: int,
) -> ReservationAmounts:
    """Reserve the pessimistic USD cost of every token one call could consume.

    The input bound is the sum of four independently attested terms:

    #. ``prompt_token_upper_bound`` -- UTF-8 bytes of the exact known message
       content (plus any structured-output contract JSON), each byte counted as
       one token;
    #. ``prompt_template_token_allowance`` -- UTF-8 bytes of the fixed system
       and user template wording that will surround it;
    #. ``provider_framing_token_allowance`` -- per-request and per-message
       provider overhead;
    #. ``sum(upstream_output_token_ceilings) * upstream_output_bytes_per_token``
       -- the not-yet-produced output of every upstream call this call's input
       will embed, converted from its token cap to a byte bound.

    Args:
        rates: The model's exact ceiling rates.
        prompt_token_upper_bound: Bound on the known prompt content.
        prompt_template_token_allowance: Bound on fixed template wording.
        provider_framing_token_allowance: Bound on provider request framing.
        upstream_output_token_ceilings: One output cap per upstream call whose
            output is embedded in this call's input.
        upstream_output_bytes_per_token: Attested bytes-per-output-token used to
            convert those caps into an input bound.
        max_output_tokens: The hard output cap this call will be issued with.

    Returns:
        A :class:`ReservationAmounts` whose ``reserved_cost_usd`` is a true
        upper bound, quantized up to :data:`USD_MICROCENT`.

    Raises:
        TypeError: Any bound is not an integer.
        ValueError: Any bound is out of range.
    """
    validate_token_bound("prompt_token_upper_bound", prompt_token_upper_bound)
    validate_token_bound("prompt_template_token_allowance", prompt_template_token_allowance)
    validate_token_bound("provider_framing_token_allowance", provider_framing_token_allowance)
    validate_token_bound(
        "upstream_output_bytes_per_token",
        upstream_output_bytes_per_token,
        positive=True,
    )
    validate_token_bound("max_output_tokens", max_output_tokens, positive=True)
    upstream = tuple(upstream_output_token_ceilings)
    for index, ceiling in enumerate(upstream):
        validate_token_bound(f"upstream_output_token_ceilings[{index}]", ceiling)

    input_token_upper_bound = (
        prompt_token_upper_bound
        + prompt_template_token_allowance
        + provider_framing_token_allowance
        + (sum(upstream) * upstream_output_bytes_per_token)
    )
    precision = _precision_for(
        (input_token_upper_bound, rates.input_ceiling_usd_per_million_tokens),
        (max_output_tokens, rates.output_ceiling_usd_per_million_tokens),
    )
    with localcontext() as context:
        context.prec = precision
        input_cost = (
            Decimal(input_token_upper_bound)
            * rates.input_ceiling_usd_per_million_tokens
            / TOKENS_PER_MILLION
        )
        output_cost = (
            Decimal(max_output_tokens)
            * rates.output_ceiling_usd_per_million_tokens
            / TOKENS_PER_MILLION
        )
        reserved = (input_cost + output_cost).quantize(USD_MICROCENT, rounding=ROUND_CEILING)

    return ReservationAmounts(
        input_token_upper_bound=input_token_upper_bound,
        output_token_upper_bound=max_output_tokens,
        input_cost_upper_bound_usd=input_cost,
        output_cost_upper_bound_usd=output_cost,
        reserved_cost_usd=reserved,
    )


def reported_usage_cost(
    rates: PriceRates,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> Decimal:
    """Price a completed call from the usage the provider actually reported.

    Prompt tokens are charged at the input ceiling, completion tokens at the
    output ceiling, and any *unattributed* remainder
    (``total - prompt - completion``, e.g. provider-side reasoning or cache
    tokens) at the HIGHER of the two rates, because nothing in the reported
    usage says which side it belongs to.

    Args:
        rates: The model's exact ceiling rates.
        prompt_tokens: Reported prompt tokens.
        completion_tokens: Reported completion tokens.
        total_tokens: Reported total tokens.

    Returns:
        The cost ceiling in USD, quantized up to :data:`USD_MICROCENT`.

    Raises:
        ValueError: ``total_tokens`` is below the attributed sum, which would
            make the remainder negative and the result an under-bound.
    """
    attributed = prompt_tokens + completion_tokens
    if total_tokens < attributed:
        raise ValueError("provider total_tokens is smaller than its attributed usage")
    unattributed = total_tokens - attributed
    higher = max(
        rates.input_ceiling_usd_per_million_tokens,
        rates.output_ceiling_usd_per_million_tokens,
    )
    precision = _precision_for(
        (prompt_tokens, rates.input_ceiling_usd_per_million_tokens),
        (completion_tokens, rates.output_ceiling_usd_per_million_tokens),
        (total_tokens, higher),
    )
    with localcontext() as context:
        context.prec = precision
        cost = (
            Decimal(prompt_tokens) * rates.input_ceiling_usd_per_million_tokens
            + Decimal(completion_tokens) * rates.output_ceiling_usd_per_million_tokens
            + Decimal(unattributed) * higher
        ) / TOKENS_PER_MILLION
        return cost.quantize(USD_MICROCENT, rounding=ROUND_CEILING)
```

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_core.py -q -p no:cacheprovider'
```
Expected: `6 passed`.

**Step 5: Commit**

```bash
git add src/conclave/pricing.py tests/test_pricing_core.py
git commit -m "refactor(pricing): shared float-free ceiling arithmetic module (DSE-1514)"
```

---

### Task 2: `evals.pricing` delegates to the shared module

**Files:**
- Modify: `src/conclave/evals/pricing.py:16-17` (constants), `:22-45` (`ModelPrice`), `:180-259` (`_validate_token_bound`, `reserve_call_cost`)
- Test: `tests/test_pricing_core.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_pricing_core.py  (append)
def test_eval_reserve_call_cost_delegates_to_the_shared_arithmetic():
    from conclave.evals.pricing import ModelPrice, reserve_call_cost

    price = ModelPrice(
        provider_id="fictional-provider-a",
        model_id="fictional-model-a",
        model_revision="fixture-r1",
        input_ceiling_usd_per_million_tokens=Decimal("1.234567"),
        output_ceiling_usd_per_million_tokens=Decimal("4.567891"),
        max_output_bytes_per_token=4,
    )
    rates = price.as_rates()
    assert isinstance(rates, PriceRates)
    assert rates.model_id == "fictional-provider-a/fictional-model-a"
    assert rates.source_url is None

    kwargs = {
        "prompt_token_upper_bound": 900,
        "prompt_template_token_allowance": 12,
        "provider_framing_token_allowance": 96,
        "upstream_output_token_ceilings": (256,),
        "upstream_output_bytes_per_token": 4,
        "max_output_tokens": 1_024,
    }
    reservation = reserve_call_cost(price, **kwargs)
    amounts = reserve_cost(rates, **kwargs)

    assert reservation.reserved_cost_usd == amounts.reserved_cost_usd
    assert reservation.input_token_upper_bound == amounts.input_token_upper_bound
    assert reservation.input_cost_upper_bound_usd == amounts.input_cost_upper_bound_usd
    assert reservation.output_cost_upper_bound_usd == amounts.output_cost_upper_bound_usd
    # The eval CallReservation contract is unchanged: it still carries revision
    # identity and the eval schema_version that hash_price_entries depends on.
    assert reservation.model_revision == "fixture-r1"
    assert reservation.schema_version == "conclave_eval_v1"
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_core.py -q -p no:cacheprovider -k delegates'
```
Expected: `AttributeError: 'ModelPrice' object has no attribute 'as_rates'`.

**Step 3: Write the implementation**

In `src/conclave/evals/pricing.py`, replace the local constants and the arithmetic body with delegation. Keep `ModelPrice`, `PriceBook`, `CallReservation`, `hash_price_entries`, `validate_price_book`, and `load_price_book` **structurally identical** — only their internals change.

Replace lines 16-17 with:

```python
from ..pricing import (
    TOKENS_PER_MILLION,
    USD_MICROCENT,
    PriceRates,
    reserve_cost,
)
from ..pricing import validate_token_bound as _validate_token_bound
```

Keep `_PRICE_HASH_NAMESPACE` and `_ModelIdentity` exactly as they are. Delete the local `USD_MICROCENT` / `TOKENS_PER_MILLION` assignments and the whole local `_validate_token_bound` function (lines 180-187) — the re-exported names keep every existing import working. `TOKENS_PER_MILLION` and `USD_MICROCENT` must stay importable from this module because `evals/live.py` imports them from here.

Add to `ModelPrice`, after the `identity` property:

```python
    def as_rates(self) -> PriceRates:
        """Project this eval price entry onto the shared arithmetic type.

        The eval harness keys prices by ``(provider, model, revision)`` while the
        product keys them by the full provider-prefixed model id, so the
        projection joins provider and model with ``"/"``. The revision is not
        carried: it participates in eval identity and in
        :func:`hash_price_entries`, never in the arithmetic.
        """
        return PriceRates(
            provider_id=self.provider_id,
            model_id=f"{self.provider_id}/{self.model_id}",
            input_ceiling_usd_per_million_tokens=self.input_ceiling_usd_per_million_tokens,
            output_ceiling_usd_per_million_tokens=self.output_ceiling_usd_per_million_tokens,
            max_output_bytes_per_token=self.max_output_bytes_per_token,
        )
```

Replace the body of `reserve_call_cost`, keeping its signature identical:

```python
def reserve_call_cost(
    price: ModelPrice,
    *,
    prompt_token_upper_bound: int,
    prompt_template_token_allowance: int,
    provider_framing_token_allowance: int,
    upstream_output_token_ceilings: Sequence[int],
    upstream_output_bytes_per_token: int,
    max_output_tokens: int,
) -> CallReservation:
    """Reserve the pessimistic USD cost of all possible input and output tokens.

    Thin adapter over :func:`conclave.pricing.reserve_cost` (DSE-1514): the
    arithmetic, the validation, and the ROUND_CEILING behaviour are shared with
    the product run path, while the eval-only ``CallReservation`` contract --
    including ``model_revision`` and the frozen eval ``schema_version`` that
    :func:`hash_price_entries` depends on -- is assembled here unchanged.
    """
    amounts = reserve_cost(
        price.as_rates(),
        prompt_token_upper_bound=prompt_token_upper_bound,
        prompt_template_token_allowance=prompt_template_token_allowance,
        provider_framing_token_allowance=provider_framing_token_allowance,
        upstream_output_token_ceilings=upstream_output_token_ceilings,
        upstream_output_bytes_per_token=upstream_output_bytes_per_token,
        max_output_tokens=max_output_tokens,
    )
    return CallReservation(
        provider_id=price.provider_id,
        model_id=price.model_id,
        model_revision=price.model_revision,
        prompt_token_upper_bound=prompt_token_upper_bound,
        prompt_template_token_allowance=prompt_template_token_allowance,
        provider_framing_token_allowance=provider_framing_token_allowance,
        upstream_output_token_ceilings=tuple(upstream_output_token_ceilings),
        upstream_output_bytes_per_token=upstream_output_bytes_per_token,
        input_token_upper_bound=amounts.input_token_upper_bound,
        output_token_upper_bound=amounts.output_token_upper_bound,
        input_ceiling_usd_per_million_tokens=price.input_ceiling_usd_per_million_tokens,
        output_ceiling_usd_per_million_tokens=price.output_ceiling_usd_per_million_tokens,
        input_cost_upper_bound_usd=amounts.input_cost_upper_bound_usd,
        output_cost_upper_bound_usd=amounts.output_cost_upper_bound_usd,
        reserved_cost_usd=amounts.reserved_cost_usd,
    )
```

Drop the now-unused imports from the file header: `ROUND_CEILING` and `localcontext` are no longer referenced there (`Decimal` still is). `ruff check` will confirm.

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_core.py tests/evals -q -p no:cacheprovider'
```
Expected: `7 passed` from `tests/test_pricing_core.py` plus every existing eval test green, with **zero** changes to `tests/evals/`. If `tests/evals/test_pricing.py::test_price_book_hash_is_canonical_and_binds_exact_frozen_snapshot` fails, `hash_price_entries` or `ModelPrice`'s field set was disturbed: revert and retry — that hash is frozen study-design identity.

**Step 5: Commit**

```bash
git add src/conclave/evals/pricing.py tests/test_pricing_core.py
git commit -m "refactor(evals): reserve_call_cost delegates to conclave.pricing (DSE-1514)"
```

---

### Task 3: `evals.live._reported_usage_cost` delegates too

**Files:**
- Modify: `src/conclave/evals/live.py:645-690` (`_unattributed_usage_tokens`, `_reported_usage_cost`)
- Test: `tests/test_pricing_core.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_pricing_core.py  (append)
def test_live_reported_usage_cost_delegates_to_the_shared_arithmetic():
    from conclave.evals.live import _reported_usage_cost
    from conclave.evals.pricing import ModelPrice
    from conclave.models import TokenUsage

    price = ModelPrice(
        provider_id="fictional-provider-b",
        model_id="fictional-model-b",
        model_revision="fixture-r2",
        input_ceiling_usd_per_million_tokens=Decimal("2.000001"),
        output_ceiling_usd_per_million_tokens=Decimal("6.000001"),
        max_output_bytes_per_token=4,
    )
    usage = TokenUsage(prompt_tokens=137, completion_tokens=421, total_tokens=600)
    assert _reported_usage_cost(price, usage) == reported_usage_cost(
        price.as_rates(),
        prompt_tokens=137,
        completion_tokens=421,
        total_tokens=600,
    )
```

**Step 2: Run it**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_core.py -q -p no:cacheprovider -k live_reported'
```
Expected: it may already pass, because both implementations compute the same number today. That is fine — keep it as the contract that pins the two implementations together. It **must** still pass after Step 3.

**Step 3: Write the implementation**

In `src/conclave/evals/live.py`, delete `_unattributed_usage_tokens` and replace `_reported_usage_cost` with:

```python
def _reported_usage_cost(price: ModelPrice, usage: TokenUsage) -> Decimal:
    """Charge reported usage at ceiling rates via the shared arithmetic (DSE-1514).

    Thin adapter over :func:`conclave.pricing.reported_usage_cost`, which owns
    the unattributed-token rule (charged at the higher of the two rates) and the
    ROUND_CEILING quantization. Raises the same ``ValueError`` as before when a
    provider reports a total below its own attributed usage.
    """
    return reported_usage_cost(
        price.as_rates(),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )
```

Add `from ..pricing import reported_usage_cost` to `live.py`'s imports, and remove `ROUND_CEILING` / `localcontext` from its `decimal` import if nothing else in the file uses them (`ruff check` will tell you).

**Step 4: Run the full suite**

```
$BR '.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
$BR '.venv/bin/ruff check . && .venv/bin/ruff format --check .'
```
Expected: `849 passed` (842 baseline plus 7 new). ruff clean.

**Step 5: Commit**

```bash
git add src/conclave/evals/live.py tests/test_pricing_core.py
git commit -m "refactor(evals): _reported_usage_cost delegates to conclave.pricing (DSE-1514)"
```

**Ship Round 2 as its own PR.** Title: `refactor: extract shared price arithmetic into conclave.pricing (DSE-1514 round 2)`. The body must state: pure refactor, zero behaviour change, eval tests unchanged, the frozen `EXPECTED_PRICE_HASH` still matches.

---

# Round 3 — Ceilings on the receipt (PR #2)

**Scope:** the product price snapshot, the manifest fields, the all-or-nothing rule, staleness, and cache identity. No CLI, no gate, no output cap yet. Expected suite after Round 3: around **872 passed**.

---

### Task 4: The product price snapshot type + digest

**Files:**
- Modify: `src/conclave/pricing.py` (append)
- Test: `tests/test_pricing_snapshot.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_pricing_snapshot.py
"""The dated product price snapshot: shape, digest, lookup, staleness (DSE-1514)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from conclave.pricing import PRICE_SNAPSHOT_MAX_AGE_DAYS, PriceRates, PriceSnapshot


def _rates(model_id: str, *, source_url: str | None = "https://example.test/pricing") -> PriceRates:
    return PriceRates(
        provider_id=model_id.split("/", 1)[0],
        model_id=model_id,
        input_ceiling_usd_per_million_tokens=Decimal("3.00"),
        output_ceiling_usd_per_million_tokens=Decimal("15.00"),
        max_output_bytes_per_token=8,
        source_url=source_url,
    )


def _snapshot(**overrides) -> PriceSnapshot:
    payload = {
        "snapshot_id": "test-prices-2026-09-03",
        "captured_at": date(2026, 9, 3),
        "currency": "USD",
        "entries": (_rates("anthropic/claude-sonnet-4-6"), _rates("openai/gpt-4.1")),
    }
    payload.update(overrides)
    return PriceSnapshot(**payload)


def test_every_entry_must_cite_a_source_url():
    with pytest.raises(ValidationError, match="source_url"):
        _snapshot(entries=(_rates("openai/gpt-4.1", source_url=None),))


def test_duplicate_model_ids_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        _snapshot(entries=(_rates("openai/gpt-4.1"), _rates("openai/gpt-4.1")))


def test_rates_for_is_exact_and_never_fuzzy():
    snapshot = _snapshot()
    assert snapshot.rates_for("openai/gpt-4.1").model_id == "openai/gpt-4.1"
    assert snapshot.rates_for("openai/gpt-4.1-mini") is None
    assert snapshot.rates_for("anthropic/claude-opus-4") is None


def test_digest_is_order_independent_and_ignores_citations():
    forward = _snapshot()
    reversed_entries = _snapshot(entries=tuple(reversed(forward.entries)))
    recited = _snapshot(
        entries=tuple(
            entry.model_copy(update={"source_url": "https://elsewhere.test/prices"})
            for entry in forward.entries
        )
    )
    assert forward.digest() == reversed_entries.digest()
    assert forward.digest() == recited.digest()
    assert forward.digest().startswith("sha256:")


def test_digest_changes_when_any_rate_changes():
    forward = _snapshot()
    bumped = _snapshot(
        entries=(
            forward.entries[0].model_copy(
                update={"output_ceiling_usd_per_million_tokens": Decimal("15.000001")}
            ),
            forward.entries[1],
        )
    )
    assert forward.digest() != bumped.digest()


def test_staleness_is_reported_but_never_changes_a_rate():
    fresh = _snapshot(captured_at=date(2026, 9, 1))
    stale = _snapshot(captured_at=date(2026, 1, 1))
    assert PRICE_SNAPSHOT_MAX_AGE_DAYS == 90
    assert fresh.is_stale(as_of=date(2026, 9, 3)) is False
    assert stale.is_stale(as_of=date(2026, 9, 3)) is True
    entry = stale.rates_for("openai/gpt-4.1")
    assert entry.output_ceiling_usd_per_million_tokens == Decimal("15.00")
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_snapshot.py -q -p no:cacheprovider'
```
Expected: `ImportError: cannot import name 'PriceSnapshot' from 'conclave.pricing'`.

**Step 3: Write the implementation**

Append to `src/conclave/pricing.py`:

```python
# A snapshot older than this many days emits a bounded warning on the manifest.
# It is a STALENESS SIGNAL ONLY: an old snapshot still prices at exactly the
# rates it records. Substituting a rate because one looks old would replace a
# falsifiable claim with a guess, which is the failure this module exists to
# prevent.
PRICE_SNAPSHOT_MAX_AGE_DAYS = 90

# Digest namespace for the PRODUCT snapshot. Deliberately distinct from the eval
# harness's ``conclave_model_prices_v1`` so the two hash spaces can never collide
# and an eval price book can never be mistaken for a product snapshot.
_PRODUCT_PRICE_HASH_NAMESPACE = "conclave_product_prices_v1"


def _canonical_decimal(value: Decimal) -> str:
    """Render a Decimal in a normalized, representation-independent form.

    ``Decimal("3.00")`` and ``Decimal("3")`` are the same rate and must produce
    the same digest, so trailing fractional zeros are stripped and exponent
    notation is expanded.
    """
    sign, digits, exponent = value.as_tuple()
    digit_text = "".join(str(digit) for digit in digits)
    if exponent >= 0:
        integer = digit_text + ("0" * exponent)
        fraction = ""
    else:
        split_at = len(digit_text) + exponent
        if split_at > 0:
            integer = digit_text[:split_at]
            fraction = digit_text[split_at:]
        else:
            integer = "0"
            fraction = ("0" * -split_at) + digit_text
        fraction = fraction.rstrip("0")
    canonical = integer if not fraction else f"{integer}.{fraction}"
    return f"-{canonical}" if sign else canonical


class PriceSnapshot(BaseModel):
    """One dated, frozen, hand-verified set of product price ceilings.

    Snapshots are checked-in artifacts under ``src/conclave/data/``, never
    fetched at runtime (explicitly out of scope). Every entry cites the vendor
    page its rate was read from; a model whose published list price could not be
    verified is OMITTED, which makes it *unpriced* rather than guessed.

    Attributes:
        snapshot_id: Stable identifier, conventionally
            ``conclave-default-prices-<YYYY-MM-DD>``.
        captured_at: The date the rates were read from the vendor pages.
        currency: Always ``"USD"``.
        entries: The priced models, keyed by full provider-prefixed model id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str = Field(min_length=1)
    captured_at: date
    currency: Literal["USD"]
    entries: tuple[PriceRates, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> PriceSnapshot:
        """Require unique model ids and a citation on every entry."""
        model_ids = [entry.model_id for entry in self.entries]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("price snapshot model ids must be unique")
        uncited = sorted(entry.model_id for entry in self.entries if not entry.source_url)
        if uncited:
            raise ValueError(f"price snapshot entries must carry a source_url: {uncited}")
        return self

    def rates_for(self, model_id: str) -> PriceRates | None:
        """Return the EXACT entry for ``model_id``, or ``None``.

        Matching is exact and total: there is no prefix match, no provider
        fallback, and no nearest-neighbour rate. An absent model is unpriced,
        full stop.
        """
        for entry in self.entries:
            if entry.model_id == model_id:
                return entry
        return None

    def is_stale(self, *, as_of: date | None = None) -> bool:
        """Return whether this snapshot is older than the staleness threshold."""
        reference = as_of or date.today()
        return (reference - self.captured_at).days > PRICE_SNAPSHOT_MAX_AGE_DAYS

    def digest(self) -> str:
        """Return an order- and representation-independent digest of the RATES.

        Covers the namespace plus, per entry, ``provider_id``, ``model_id``, both
        ceiling rates in canonical decimal form, and
        ``max_output_bytes_per_token``. It deliberately EXCLUDES ``snapshot_id``,
        ``captured_at``, and ``source_url``: this digest joins cache identity, and
        re-dating or re-citing a snapshot whose rates are byte-identical must not
        invalidate entries whose ceiling would come out exactly the same.
        """
        ordered = sorted(self.entries, key=lambda entry: (entry.provider_id, entry.model_id))
        canonical = json.dumps(
            {
                "namespace": _PRODUCT_PRICE_HASH_NAMESPACE,
                "entries": [
                    {
                        "provider_id": entry.provider_id,
                        "model_id": entry.model_id,
                        "input_ceiling_usd_per_million_tokens": _canonical_decimal(
                            entry.input_ceiling_usd_per_million_tokens
                        ),
                        "output_ceiling_usd_per_million_tokens": _canonical_decimal(
                            entry.output_ceiling_usd_per_million_tokens
                        ),
                        "max_output_bytes_per_token": str(entry.max_output_bytes_per_token),
                    }
                    for entry in ordered
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"
```

Extend `src/conclave/pricing.py`'s imports: add `hashlib`, `json`, `from datetime import date`, `from typing import Literal`, and `model_validator` to the pydantic import.

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_snapshot.py -q -p no:cacheprovider'
```
Expected: `6 passed`.

**Step 5: Commit**

```bash
git add src/conclave/pricing.py tests/test_pricing_snapshot.py
git commit -m "feat(pricing): dated product price snapshot type with rate-only digest (DSE-1514)"
```

---

### Task 5: The snapshot data file + loader

> **Research task — this is the one place in the plan where you must go and look something up.** Do **not** copy a rate from this document, from memory, or from the eval fixture (those numbers are fictional and labelled as such). Every rate you write must come from the vendor's own published price page, read during this task.

**Files:**
- Create: `src/conclave/data/__init__.py`
- Create: `src/conclave/data/prices-<TODAY>.json`
- Modify: `src/conclave/pricing.py` (append the loader)
- Modify: `pyproject.toml` (only if Step 6's packaging check fails)
- Test: `tests/test_pricing_snapshot.py` (append)

**Step 1: Research the rates**

For each of the nine `conclave.registry.DEFAULT_MODELS` values —

```
xai/grok-4.3
gemini/gemini-2.5-pro
anthropic/claude-sonnet-4-6
perplexity/sonar-pro
openai/gpt-4.1
groq/llama-3.3-70b-versatile
deepseek/deepseek-chat
mistral/mistral-large-latest
together/meta-llama/Llama-3.3-70B-Instruct-Turbo
```

— find the vendor's **published list price per million input tokens and per million output tokens**, using `mcp__perplexity__perplexity_search` / `perplexity_ask` with `search_domain_filter` set to the vendor's own domain, or `WebFetch` on the vendor pricing page directly. Then:

* **Round every rate UP** to a clean decimal at or above the published number (a published `$2.50` is written `"2.50"`; a published `$0.27` is written `"0.27"`). Never round down. Never average a tiered or cached-input rate — take the **highest** tier a plain chat completion could be billed at.
* **Record the exact URL** you read it from in `source_url`.
* **OMIT any model whose list price you cannot verify from a first-party page.** An omitted model is unpriced, which is a correct and honest outcome. Do not guess, do not use a third-party aggregator, do not carry a rate over from a sibling model. If a price is tiered by context length or region, take the most expensive tier.
* `deepseek/deepseek-chat` and `together/meta-llama/...` are the likeliest omissions; that is fine and expected.
* Set `max_output_bytes_per_token` to **`8`** for every entry. This is a deliberate attestation, not a measurement: it is twice the eval harness's fixture value of 4, and it is used only to convert an upstream model's *output token cap* into a *downstream input byte bound*. Doubling it makes every downstream input bound twice as pessimistic — the safe direction for a ceiling — at negligible cost, because the dominant term of any reservation is `max_output_tokens` times the output rate. Record that reasoning in the file's `"_note"` key.

**Step 2: Write the data file**

```json
{
  "_note": "Hand-verified vendor list prices, rounded UP. Every entry cites the page it was read from. A model absent from this file is UNPRICED, never estimated. max_output_bytes_per_token is an attestation of 8 bytes/token (2x the eval harness fixture value of 4): it only ever converts an upstream output-token cap into a downstream input BYTE bound, so a larger value is strictly more pessimistic. Regenerate deliberately; never fetch at runtime.",
  "snapshot_id": "conclave-default-prices-<TODAY>",
  "captured_at": "<TODAY>",
  "currency": "USD",
  "entries": [
    {
      "provider_id": "openai",
      "model_id": "openai/gpt-4.1",
      "input_ceiling_usd_per_million_tokens": "<VERIFIED>",
      "output_ceiling_usd_per_million_tokens": "<VERIFIED>",
      "max_output_bytes_per_token": 8,
      "source_url": "<THE PAGE YOU READ>"
    }
  ]
}
```

All rates are JSON **strings**, never JSON numbers — a bare `2.5` would be parsed as a float by any consumer that forgets `parse_float=Decimal`, and the validator would reject it. Strings make the contract explicit.

**Step 3: Write the failing test**

```python
# tests/test_pricing_snapshot.py  (append)
def test_default_snapshot_loads_and_prices_the_verified_default_models():
    from conclave.pricing import load_default_price_snapshot
    from conclave.registry import DEFAULT_MODELS

    snapshot = load_default_price_snapshot()
    assert snapshot is not None
    assert snapshot.currency == "USD"
    assert snapshot.snapshot_id.startswith("conclave-default-prices-")

    priced = {entry.model_id for entry in snapshot.entries}
    # Every priced entry must be one of the shipped defaults -- the snapshot is
    # not a place to accumulate models the product does not resolve.
    assert priced <= set(DEFAULT_MODELS.values())
    # The four frontier defaults must be priced; anything unverifiable is omitted
    # deliberately and shows up as an unpriced model at runtime.
    assert {
        "openai/gpt-4.1",
        "anthropic/claude-sonnet-4-6",
        "xai/grok-4.3",
        "gemini/gemini-2.5-pro",
    } <= priced

    for entry in snapshot.entries:
        assert isinstance(entry.input_ceiling_usd_per_million_tokens, Decimal)
        assert isinstance(entry.output_ceiling_usd_per_million_tokens, Decimal)
        assert entry.source_url and entry.source_url.startswith("https://")
        assert entry.max_output_bytes_per_token == 8
        assert entry.provider_id == entry.model_id.split("/", 1)[0]


def test_default_snapshot_is_memoized_and_ships_inside_the_package():
    from pathlib import Path

    import conclave
    from conclave.pricing import load_default_price_snapshot

    assert load_default_price_snapshot() is load_default_price_snapshot()
    data_dir = Path(conclave.__file__).parent / "data"
    assert sorted(path.name for path in data_dir.glob("prices-*.json"))


def test_a_missing_snapshot_directory_degrades_to_none(monkeypatch, tmp_path):
    from conclave import pricing

    pricing.load_default_price_snapshot.cache_clear()
    monkeypatch.setattr(pricing, "_price_data_dir", lambda: tmp_path)
    try:
        assert pricing.load_default_price_snapshot() is None
    finally:
        pricing.load_default_price_snapshot.cache_clear()
```

**Step 4: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_snapshot.py -q -p no:cacheprovider'
```
Expected: `ImportError: cannot import name 'load_default_price_snapshot'`.

**Step 5: Write the loader**

Append to `src/conclave/pricing.py`:

```python
def _price_data_dir() -> Path:
    """Return the packaged directory holding the dated price snapshots."""
    return Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_default_price_snapshot() -> PriceSnapshot | None:
    """Load the newest packaged price snapshot, or ``None`` when unavailable.

    Picks the lexicographically-last ``prices-YYYY-MM-DD.json`` in
    ``conclave/data`` (ISO dates sort chronologically), parses it with
    ``parse_float=Decimal`` so a JSON number can never become a float rate, and
    validates it. Memoized for the life of the process: a snapshot is a frozen
    artifact, so re-reading it per run would be pure overhead.

    Returns ``None`` -- never raises -- when the directory is absent, empty,
    unreadable, or holds a file that fails validation, and logs a warning. A
    missing snapshot means a run carries NO ceiling; it must never mean a failed
    run. (The pre-flight spend gate treats the same condition as a refusal
    instead; that asymmetry is deliberate.)
    """
    try:
        candidates = sorted(_price_data_dir().glob("prices-*.json"))
    except OSError as exc:
        logger.warning("price snapshot directory unreadable: %s; pricing disabled", exc)
        return None
    if not candidates:
        logger.warning("no packaged price snapshot found; pricing disabled")
        return None
    path = candidates[-1]
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, parse_float=Decimal)
        payload.pop("_note", None)
        return PriceSnapshot.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.warning("price snapshot %s is unusable: %s; pricing disabled", path.name, exc)
        return None
```

Extend the imports: `from functools import lru_cache`, `from pathlib import Path`, `from pydantic import ValidationError`, and `from .logging import get_logger` with `logger = get_logger("pricing")` beside the constants.

Create `src/conclave/data/__init__.py`:

```python
"""Packaged, dated price snapshots. Data only -- no code, no runtime fetching."""
```

**Step 6: Run it and watch it pass, then prove it ships in the wheel**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_snapshot.py -q -p no:cacheprovider'
$BR '.venv/bin/python -m pip wheel --no-deps -w /tmp/dse1514wheel . >/dev/null && .venv/bin/python -c "import glob,zipfile; w=glob.glob(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(100)+chr(115)+chr(101)+chr(49)+chr(53)+chr(49)+chr(52)+chr(119)+chr(104)+chr(101)+chr(101)+chr(108)+chr(47)+chr(42)+chr(46)+chr(119)+chr(104)+chr(108))[0]; names=[n for n in zipfile.ZipFile(w).namelist() if chr(47)+chr(100)+chr(97)+chr(116)+chr(97)+chr(47) in n]; print(names); assert names"'
```
Expected: `9 passed`, and the wheel listing prints `['conclave/data/__init__.py', 'conclave/data/prices-<TODAY>.json']`.

Hatchling's wheel target includes every file under `packages = ["src/conclave"]`, data files included — unlike setuptools it needs no `package-data` declaration. **If the assertion fails**, add to `pyproject.toml` under `[tool.hatch.build.targets.wheel]`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/conclave/data" = "conclave/data"
```

and re-run the check.

**Step 7: Commit**

```bash
git add src/conclave/data src/conclave/pricing.py tests/test_pricing_snapshot.py pyproject.toml
git commit -m "feat(pricing): dated vendor-cited default price snapshot + packaged loader (DSE-1514)"
```

---

### Task 6: Manifest + receipt ceiling fields

**Files:**
- Modify: `src/conclave/manifest.py:11-14` (module docstring), `:143-191` (`ProviderExecutionReceipt`), `:193-280` (`ModelHarnessManifest`)
- Test: `tests/test_pricing_receipts.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_pricing_receipts.py
"""Ceiling fields on the manifest: additive, Decimal-only, all-or-nothing (DSE-1514)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from conclave.manifest import (
    ModelHarnessManifest,
    ProviderExecutionReceipt,
    scan_for_secret_material,
)


def _receipt(**overrides) -> ProviderExecutionReceipt:
    payload = {
        "name": "claude",
        "provider": "anthropic",
        "model_id": "anthropic/claude-sonnet-4-6",
    }
    payload.update(overrides)
    return ProviderExecutionReceipt(**payload)


def test_new_receipt_fields_default_to_none_and_estimated_cost_is_untouched():
    receipt = _receipt()
    assert receipt.cost_ceiling_usd is None
    assert receipt.cost_basis is None
    assert receipt.estimated_cost is None


def test_receipt_ceiling_is_decimal_and_rejects_a_float():
    receipt = _receipt(cost_ceiling_usd=Decimal("0.001234"), cost_basis="reported_usage")
    assert isinstance(receipt.cost_ceiling_usd, Decimal)
    assert receipt.cost_ceiling_usd == Decimal("0.001234")
    with pytest.raises(ValidationError):
        _receipt(cost_ceiling_usd=0.001234)


def test_receipt_cost_basis_is_a_closed_vocabulary():
    assert _receipt(cost_basis="reservation").cost_basis == "reservation"
    with pytest.raises(ValidationError):
        _receipt(cost_basis="guess")


def test_manifest_ceiling_fields_default_empty_and_estimated_cost_stays_none():
    manifest = ModelHarnessManifest(request_id="r", conclave_version="1.3.0", mode="synthesize")
    assert manifest.cost_ceiling_usd is None
    assert manifest.price_snapshot_digest is None
    assert manifest.priced_as_of is None
    assert manifest.unpriced_models == []
    assert manifest.unpriced_receipts == 0
    assert manifest.pricing_warnings == []
    assert manifest.estimated_cost is None
    assert manifest.pricing_snapshot_date is None


def test_populated_pricing_fields_keep_the_secret_scan_clean():
    manifest = ModelHarnessManifest(
        request_id="r",
        conclave_version="1.3.0",
        mode="elite",
        model_ids=["anthropic/claude-sonnet-4-6", "deepseek/deepseek-chat"],
        receipts=[_receipt(cost_ceiling_usd=Decimal("0.5"), cost_basis="reported_usage")],
        cost_ceiling_usd=Decimal("0.5"),
        price_snapshot_digest="sha256:" + "b" * 64,
        priced_as_of="2026-09-03",
        unpriced_models=["deepseek/deepseek-chat"],
        unpriced_receipts=0,
        pricing_warnings=["unpriced_models_present", "price_snapshot_stale"],
    )
    assert scan_for_secret_material(manifest) is True


def test_ceilings_round_trip_through_json_as_exact_decimals():
    manifest = ModelHarnessManifest(
        request_id="r",
        conclave_version="1.3.0",
        mode="synthesize",
        receipts=[_receipt(cost_ceiling_usd=Decimal("0.000001"), cost_basis="reservation")],
        cost_ceiling_usd=Decimal("0.000001"),
    )
    restored = ModelHarnessManifest.model_validate(manifest.model_dump(mode="json"))
    assert restored.cost_ceiling_usd == Decimal("0.000001")
    assert restored.receipts[0].cost_ceiling_usd == Decimal("0.000001")
    assert isinstance(restored.cost_ceiling_usd, Decimal)
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_receipts.py -q -p no:cacheprovider'
```
Expected: 6 failures — pydantic rejects the unknown `cost_ceiling_usd` / `cost_basis` / `pricing_warnings` kwargs.

**Step 3: Write the implementation**

In `src/conclave/manifest.py`, add `from decimal import Decimal` to the imports and define, above `ProviderExecutionReceipt`:

```python
# How a receipt's ``cost_ceiling_usd`` was derived. ``reported_usage`` means the
# provider told us its token counts and they were charged at ceiling rates;
# ``reservation`` means the call produced no usage (it failed, or the provider
# reported none) and the ceiling is the pessimistic pre-call reservation instead.
# There is no third basis: a call with neither usage nor a reservable output cap
# is UNPRICED (``None``/``None``), never estimated.
CostBasis = Literal["reported_usage", "reservation"]
```

Add to `ProviderExecutionReceipt`, immediately after `estimated_cost`:

```python
    cost_ceiling_usd: Decimal | None = None
    cost_basis: CostBasis | None = None
```

with these docstring entries:

```
        cost_ceiling_usd: A falsifiable UPPER BOUND on this call's cost in USD,
            or ``None`` when the call could not be bounded. Exact ``Decimal``,
            ROUND_CEILING, priced against the run's dated snapshot. Distinct
            from ``estimated_cost`` on purpose: an estimate is a guess and stays
            ``None`` forever; a ceiling is a checkable claim.
        cost_basis: Which rule produced ``cost_ceiling_usd`` -- see
            :data:`CostBasis`. ``None`` exactly when ``cost_ceiling_usd`` is.
```

Add to `ModelHarnessManifest`, immediately after `pricing_snapshot_date`:

```python
    # Bounded cost ceilings (DSE-1514). Additive; ``estimated_cost`` above is
    # untouched and stays ``None`` forever.
    cost_ceiling_usd: Decimal | None = None
    price_snapshot_digest: str | None = None
    priced_as_of: str | None = None
    unpriced_models: list[str] = Field(default_factory=list)
    unpriced_receipts: int = Field(default=0, ge=0)
    pricing_warnings: list[str] = Field(default_factory=list)
```

with these docstring entries:

```
        cost_ceiling_usd: The run's total cost CEILING in USD -- the sum of every
            receipt's ceiling -- populated only when the whole run is priceable.
            All-or-nothing: any unpriced model or unpriced receipt leaves this
            ``None`` with ``unpriced_models``/``unpriced_receipts`` naming why. A
            partial sum reads exactly like a complete one and is never emitted.
        price_snapshot_digest: Rate digest of the snapshot the ceiling was priced
            against. Accompanies every non-``None`` ceiling.
        priced_as_of: ISO date the snapshot's rates were captured.
        unpriced_models: Sorted model ids that ran (or were called) in this run
            and have no snapshot entry. Non-empty forces the run ceiling to
            ``None``. Drawn from the same vocabulary as ``model_ids``, so it adds
            no new secret-scan surface.
        unpriced_receipts: How many receipts could not be priced at all.
        pricing_warnings: Bounded, fixed identifiers only -- one of
            ``price_snapshot_stale``, ``price_snapshot_unavailable``,
            ``unpriced_models_present``, ``unpriced_receipts_present``,
            ``no_output_cap_configured``. NEVER provider text, never
            interpolated, so ``scan_for_secret_material`` stays provable.
```

Finally, amend the module docstring's cost paragraph (lines 11-14) to name the ceiling fields and restate that `estimated_cost` remains permanently `None`.

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_receipts.py tests/test_manifest.py tests/test_secret_safety_matrix.py -q -p no:cacheprovider'
```
Expected: `6 passed` from the new file, plus the existing manifest and secret-safety tests green.

**Step 5: Commit**

```bash
git add src/conclave/manifest.py tests/test_pricing_receipts.py
git commit -m "feat(manifest): additive cost-ceiling fields on receipts and the run manifest (DSE-1514)"
```

---

### Task 7: `Council._price_manifest` — the all-or-nothing pricing pass

**Files:**
- Modify: `src/conclave/council.py:104-113` (module constants), `:339-415` (`_cached_run`), and append `_price_manifest` beside `_recompute_manifest_accounting` (`:649`)
- Modify: `src/conclave/streaming.py:197` and `:270` (both `done` yields)
- Test: `tests/test_pricing_receipts.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_pricing_receipts.py  (append)
from datetime import date

from conclave.council import Council
from conclave.pricing import PriceRates, PriceSnapshot
from tests.conftest import make_response


def _snapshot(*model_ids: str, captured_at: date = date(2026, 9, 3)) -> PriceSnapshot:
    return PriceSnapshot(
        snapshot_id="test-prices",
        captured_at=captured_at,
        currency="USD",
        entries=tuple(
            PriceRates(
                provider_id=model_id.split("/", 1)[0],
                model_id=model_id,
                input_ceiling_usd_per_million_tokens=Decimal("3.00"),
                output_ceiling_usd_per_million_tokens=Decimal("15.00"),
                max_output_bytes_per_token=8,
                source_url="https://example.test/pricing",
            )
            for model_id in model_ids
        ),
    )


def _install_snapshot(monkeypatch, snapshot):
    import conclave.council as council_mod

    monkeypatch.setattr(council_mod, "load_default_price_snapshot", lambda: snapshot)


async def test_a_fully_priced_run_carries_a_ceiling_digest_and_date(
    monkeypatch, patch_call_model, keys
):
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", "anthropic/claude-sonnet-4-6"))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="claude", extract_verdict=False)
    result = await council.ask("q")

    manifest = result.manifest
    assert manifest.unpriced_models == []
    assert manifest.unpriced_receipts == 0
    assert isinstance(manifest.cost_ceiling_usd, Decimal)
    assert manifest.cost_ceiling_usd == sum(
        (receipt.cost_ceiling_usd for receipt in manifest.receipts), Decimal("0")
    )
    assert manifest.price_snapshot_digest.startswith("sha256:")
    assert manifest.priced_as_of == "2026-09-03"
    assert all(receipt.cost_basis == "reported_usage" for receipt in manifest.receipts)
    # The estimate slot is untouched, forever.
    assert manifest.estimated_cost is None
    assert all(receipt.estimated_cost is None for receipt in manifest.receipts)


async def test_one_unpriced_model_nulls_the_whole_run_ceiling(monkeypatch, patch_call_model, keys):
    # grok is priced, claude is NOT -> all-or-nothing.
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="claude", extract_verdict=False)
    result = await council.ask("q")

    manifest = result.manifest
    assert manifest.cost_ceiling_usd is None
    assert manifest.unpriced_models == ["anthropic/claude-sonnet-4-6"]
    assert manifest.unpriced_receipts == 1
    assert "unpriced_models_present" in manifest.pricing_warnings
    # The PRICED receipts still carry their own ceilings -- only the SUM is withheld.
    priced = [r for r in manifest.receipts if r.model_id == "xai/grok-4.3"]
    assert priced and all(r.cost_ceiling_usd is not None for r in priced)


async def test_a_stale_snapshot_warns_but_never_changes_a_rate(monkeypatch, patch_call_model, keys):
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", captured_at=date(2026, 9, 3)))
    fresh = (await council.ask("q", synthesize=False)).manifest

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", captured_at=date(2026, 1, 1)))
    stale = (await council.ask("q", synthesize=False)).manifest

    assert fresh.pricing_warnings == []
    assert "price_snapshot_stale" in stale.pricing_warnings
    assert stale.cost_ceiling_usd == fresh.cost_ceiling_usd


async def test_a_missing_snapshot_leaves_no_ceiling_and_never_raises(
    monkeypatch, patch_call_model, keys
):
    _install_snapshot(monkeypatch, None)
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)
    manifest = (await council.ask("q", synthesize=False)).manifest

    assert manifest.cost_ceiling_usd is None
    assert manifest.price_snapshot_digest is None
    assert manifest.pricing_warnings == ["price_snapshot_unavailable"]
    assert manifest.secret_safety == "verified_no_secrets"


async def test_a_failed_call_with_no_usage_is_unpriced_without_an_output_cap(
    monkeypatch, patch_call_model, keys
):
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    def handler(model_id, messages):
        raise RuntimeError("boom")

    patch_call_model(handler)
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)
    manifest = (await council.ask("q", synthesize=False)).manifest

    assert manifest.receipts[0].cost_ceiling_usd is None
    assert manifest.receipts[0].cost_basis is None
    assert manifest.unpriced_receipts == 1
    assert manifest.cost_ceiling_usd is None
    assert "unpriced_receipts_present" in manifest.pricing_warnings
    assert "no_output_cap_configured" in manifest.pricing_warnings


async def test_an_impossible_usage_total_degrades_to_unpriced(monkeypatch, keys):
    import conclave.council as council_mod
    from conclave.models import ModelAnswer, TokenUsage

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def bad_usage(name, model_id, messages, **kwargs):
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="ok",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=5),
        )

    monkeypatch.setattr(council_mod, "call_model", bad_usage)
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)
    manifest = (await council.ask("q", synthesize=False)).manifest

    assert manifest.receipts[0].cost_ceiling_usd is None
    assert manifest.cost_ceiling_usd is None
    assert manifest.unpriced_receipts == 1
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_receipts.py -q -p no:cacheprovider'
```
Expected: `AttributeError: module 'conclave.council' has no attribute 'load_default_price_snapshot'`.

**Step 3: Write the implementation**

In `src/conclave/council.py`, add `from decimal import Decimal` and:

```python
from .pricing import (
    PriceSnapshot,
    load_default_price_snapshot,
    reported_usage_cost,
    reserve_cost,
)
```

Importing `load_default_price_snapshot` as a module-level *name* is what makes `monkeypatch.setattr(council_mod, "load_default_price_snapshot", ...)` work — the same seam pattern `call_model` already uses.

Add beside `_SYNTH_SYSTEM`:

```python
# Fixed allowances used when a FAILED call must be priced from a reservation
# rather than from reported usage. A failed call carries no usage and no
# recorded message list, so its input is bounded by the raw prompt bytes plus
# these two constants: a template allowance covering any system/instruction
# wording the mode wrapped around the prompt, and the same per-request framing
# allowance the eval runner attests (64 + 16 per message, taken at 4 messages).
_PRICING_TEMPLATE_ALLOWANCE = 4096
_PRICING_FRAMING_ALLOWANCE = 64 + (16 * 4)
```

Add the method beside `_recompute_manifest_accounting`:

```python
    def _price_manifest(self, result: CouncilResult) -> None:
        """Stamp cost ceilings on the manifest -- the LAST step of a run (DSE-1514).

        Runs after ``_ensure_manifest`` and after every receipt is appended, so
        it sees the complete ledger. It is idempotent: it recomputes every field
        from the receipts each time, so re-pricing a cache hit is harmless.

        The rule, per receipt:

        * the model has no snapshot entry -> unpriced (``None``/``None``), and
          the model id joins ``unpriced_models``;
        * the provider reported usage -> ``reported_usage_cost`` at ceiling
          rates, basis ``"reported_usage"``;
        * no usage (the call failed, or the provider reported none) AND an
          output cap is configured -> the call's own pessimistic reservation,
          basis ``"reservation"``;
        * no usage and no output cap -> unpriced. Nothing is estimated.

        And at run level, ALL-OR-NOTHING: ``cost_ceiling_usd`` is the sum of
        every receipt ceiling only when ``unpriced_models`` is empty AND
        ``unpriced_receipts`` is zero. A run with no receipts at all (the
        memberless path) has a provable ceiling of ``Decimal("0")`` -- no call
        was made, so nothing was spent -- deliberately distinguished from
        ``None`` ("could not be bounded").

        Never raises: an unusable snapshot, an unreadable file, or a provider
        that reports an impossible token total degrades to "unpriced" with a
        bounded warning. A missing ceiling is an acceptable outcome; a failed
        council run because of pricing is not.
        """
        manifest = result.manifest
        if manifest is None:
            return

        snapshot: PriceSnapshot | None = load_default_price_snapshot()
        if snapshot is None:
            for receipt in manifest.receipts:
                receipt.cost_ceiling_usd = None
                receipt.cost_basis = None
            manifest.cost_ceiling_usd = None
            manifest.price_snapshot_digest = None
            manifest.priced_as_of = None
            manifest.unpriced_models = []
            manifest.unpriced_receipts = len(manifest.receipts)
            manifest.pricing_warnings = ["price_snapshot_unavailable"]
            manifest.secret_safety = verified_secret_safety(manifest)
            return

        cap = self.max_output_tokens
        unpriced_models: set[str] = {
            model_id for model_id in manifest.model_ids if snapshot.rates_for(model_id) is None
        }
        unpriced_receipts = 0
        for receipt in manifest.receipts:
            rates = snapshot.rates_for(receipt.model_id)
            if rates is None:
                unpriced_models.add(receipt.model_id)
                receipt.cost_ceiling_usd = None
                receipt.cost_basis = None
                unpriced_receipts += 1
                continue
            if receipt.usage is not None:
                try:
                    receipt.cost_ceiling_usd = reported_usage_cost(
                        rates,
                        prompt_tokens=receipt.usage.prompt_tokens,
                        completion_tokens=receipt.usage.completion_tokens,
                        total_tokens=receipt.usage.total_tokens,
                    )
                except ValueError:
                    # A provider reported a total below its own attributed usage.
                    # Bounding it would require inventing the missing tokens.
                    logger.warning(
                        "unusable reported usage for %s; leaving the call unpriced",
                        receipt.model_id,
                    )
                    receipt.cost_ceiling_usd = None
                    receipt.cost_basis = None
                    unpriced_receipts += 1
                    continue
                receipt.cost_basis = "reported_usage"
                continue
            if cap is None:
                receipt.cost_ceiling_usd = None
                receipt.cost_basis = None
                unpriced_receipts += 1
                continue
            receipt.cost_ceiling_usd = reserve_cost(
                rates,
                prompt_token_upper_bound=len(result.prompt.encode("utf-8")),
                prompt_template_token_allowance=_PRICING_TEMPLATE_ALLOWANCE,
                provider_framing_token_allowance=_PRICING_FRAMING_ALLOWANCE,
                upstream_output_token_ceilings=(),
                upstream_output_bytes_per_token=rates.max_output_bytes_per_token,
                max_output_tokens=cap,
            ).reserved_cost_usd
            receipt.cost_basis = "reservation"

        warnings: list[str] = []
        if unpriced_models:
            warnings.append("unpriced_models_present")
        if unpriced_receipts:
            warnings.append("unpriced_receipts_present")
            if cap is None:
                warnings.append("no_output_cap_configured")
        if snapshot.is_stale():
            warnings.append("price_snapshot_stale")

        manifest.price_snapshot_digest = snapshot.digest()
        manifest.priced_as_of = snapshot.captured_at.isoformat()
        manifest.unpriced_models = sorted(unpriced_models)
        manifest.unpriced_receipts = unpriced_receipts
        manifest.pricing_warnings = warnings
        manifest.cost_ceiling_usd = (
            sum((receipt.cost_ceiling_usd for receipt in manifest.receipts), Decimal("0"))
            if not unpriced_models and not unpriced_receipts
            else None
        )
        # Re-stamp: the ceiling fields were written after _build_manifest's scan.
        manifest.secret_safety = verified_secret_safety(manifest)
```

Wire the three call sites. In `_cached_run`, add `self._price_manifest(...)` immediately after **each** `_ensure_manifest` call — there are three — and in the live-miss branch it must come **before** the `primary_failed_over` check so a stored entry carries its ceiling:

```python
        if not self.cache_enabled:
            result = await run()
            self._ensure_manifest(result, mode)
            self._price_manifest(result)
            return result
```

```python
        hit = cache_mod.load(key)
        if hit is not None:
            logger.info("cache hit for %s run (%s)", mode, key[:12])
            self._ensure_manifest(hit, mode)
            self._price_manifest(hit)
            return hit

        result = await run()
        self._ensure_manifest(result, mode)
        self._price_manifest(result)
        if result.primary_failed_over:
            ...
```

In `src/conclave/streaming.py`, call `council._price_manifest(result)` immediately before **both** `yield StreamEvent(type="done", result=result)` statements (the memberless early return near line 197, and the terminal yield near line 270), with this comment:

```python
    # Pricing is the LAST step: it must see every receipt, including the verdict
    # extraction receipts _apply_verdict just appended. Mirrors Council._cached_run.
    council._price_manifest(result)
```

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_pricing_receipts.py -q -p no:cacheprovider'
$BR '.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
```
Expected: `12 passed` in the new file; full suite around `868 passed`, nothing red. If `test_manifest_all_modes.py` fails, a mode is not reaching `_price_manifest` — fix the wiring, not the test.

**Step 5: Commit**

```bash
git add src/conclave/council.py src/conclave/streaming.py tests/test_pricing_receipts.py
git commit -m "feat(council): price the manifest last, all-or-nothing, never estimating (DSE-1514)"
```

---

### Task 8: Snapshot digest joins cache identity

**Files:**
- Modify: `src/conclave/cache.py:60-64` (version), `:153-234` (`build_identity`), `:237-323` (`make_key`)
- Modify: `src/conclave/council.py:285-338` (`_cache_key`), `:200-240` (`__init__` placeholder attribute)
- Test: `tests/test_cache.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_cache.py  (append)
def test_cache_format_version_is_five_for_price_identity():
    from conclave import cache as cache_mod

    assert cache_mod.CACHE_FORMAT_VERSION == "5"


def test_identity_carries_the_price_snapshot_fingerprint_and_output_cap():
    from conclave.cache import build_identity

    base = {
        "prompt": "q",
        "mode": "synthesize",
        "members": [("grok", "xai/grok-4.3")],
        "synthesizer": "claude",
        "synthesizer_model_id": "anthropic/claude-sonnet-4-6",
        "temperature": 0.7,
    }
    plain = build_identity(**base)
    assert plain["price_snapshot_fingerprint"] is None
    assert plain["generation"]["max_output_tokens"] is None

    priced = build_identity(**base, price_snapshot_digest="sha256:" + "c" * 64)
    other = build_identity(**base, price_snapshot_digest="sha256:" + "d" * 64)
    assert priced["price_snapshot_fingerprint"] != plain["price_snapshot_fingerprint"]
    assert priced["price_snapshot_fingerprint"] != other["price_snapshot_fingerprint"]
    # The raw digest never enters the inspectable identity document -- it is
    # re-hashed, matching how source_bundle_digest is handled.
    assert "c" * 64 not in str(priced)

    capped = build_identity(**base, max_output_tokens=512)
    assert capped["generation"]["max_output_tokens"] == 512


def test_two_snapshots_never_share_a_cache_key():
    from conclave.cache import make_key

    base = {
        "prompt": "q",
        "mode": "synthesize",
        "members": [("grok", "xai/grok-4.3")],
        "synthesizer": "claude",
        "synthesizer_model_id": "anthropic/claude-sonnet-4-6",
        "temperature": 0.7,
    }
    assert make_key(**base, price_snapshot_digest="sha256:" + "c" * 64) != make_key(
        **base, price_snapshot_digest="sha256:" + "d" * 64
    )
    assert make_key(**base, max_output_tokens=512) != make_key(**base, max_output_tokens=1024)


async def test_a_cached_result_round_trips_its_decimal_ceiling(tmp_path, monkeypatch, keys):
    """A cache hit must return an exact Decimal ceiling, not a float or a string."""
    from decimal import Decimal

    import conclave.council as council_mod
    from conclave.council import Council
    from conclave.models import ModelAnswer, TokenUsage
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def ok(name, model_id, messages, **kwargs):
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="ok",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    monkeypatch.setattr(council_mod, "call_model", ok)
    council = Council(models=["grok"], synthesizer="grok", cache=True, extract_verdict=False)
    live = await council.ask("q", synthesize=False)
    hit = await council.ask("q", synthesize=False)

    assert hit.cached is True
    assert isinstance(hit.manifest.cost_ceiling_usd, Decimal)
    assert hit.manifest.cost_ceiling_usd == live.manifest.cost_ceiling_usd
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_cache.py -q -p no:cacheprovider -k "five or fingerprint or snapshots or round_trips"'
```
Expected: `assert '4' == '5'` and `TypeError: build_identity() got an unexpected keyword argument 'price_snapshot_digest'`.

**Step 3: Write the implementation**

In `src/conclave/cache.py`, bump and extend the version comment (append a line; do not rewrite DSE-1512's):

```python
# v4 (DSE-1512): identity now carries the full ordered synthesizer/judge chain,
# not just the primary candidate.
# v5 (DSE-1514): identity now carries the price-snapshot rate fingerprint and the
# max_output_tokens cap, so a re-priced or differently-capped run can never be
# served a stale ceiling (or a longer/shorter answer) from a prior entry.
CACHE_FORMAT_VERSION = "5"
```

Add two keyword-only parameters to **both** `build_identity` and `make_key`, placed just before `cache_format_version`:

```text
    price_snapshot_digest: str | None = None,
    max_output_tokens: int | None = None,
```

In `build_identity`, add `"max_output_tokens": max_output_tokens` to the `generation` dict and a new top-level key beside `source_bundle_fingerprint`:

```text
        # Re-hash the snapshot digest for the same reason the source bundle
        # digest is re-hashed: a malformed caller value must never appear in an
        # inspectable identity document, while still invalidating prior entries.
        "price_snapshot_fingerprint": (
            _digest(price_snapshot_digest) if price_snapshot_digest is not None else None
        ),
```

In `make_key`, forward both new arguments to `build_identity`. Extend both docstrings: `price_snapshot_digest` — "the rate digest of the price snapshot a run's ceilings are computed against (DSE-1514); two runs priced under different rates must not collide, because a hit would serve a ceiling that was never true of those rates". `max_output_tokens` — "the hard output cap; it changes the answers themselves, so it is part of generation identity".

In `src/conclave/council.py`'s `_cache_key`, resolve and forward both:

```python
        snapshot = load_default_price_snapshot()
        return cache_mod.make_key(
            prompt=prompt,
            mode=mode,
            members=members,
            synthesizer=self.synthesizer,
            synthesizer_model_id=synth_id,
            synthesizer_chain=chain_pairs,
            temperature=self.temperature,
            timeout=self.timeout,
            rounds=rounds,
            proposer=proposer,
            converge_threshold=converge_threshold,
            choices=choices,
            extract_verdict=self.extract_verdict_enabled,
            endpoint_urls={
                prefix: endpoint.completions_url
                for prefix, endpoint in self.config.endpoints.items()
                if prefix in used_prefixes
            },
            source_bundle_digest=self.source_bundle_digest,
            price_snapshot_digest=None if snapshot is None else snapshot.digest(),
            max_output_tokens=self.max_output_tokens,
            protocol_version=ELITE_PROTOCOL_VERSION,
            synthesis_prompt_version=SYNTHESIS_PROMPT_VERSION,
            elite_prompt_version=ELITE_PROMPT_VERSION,
        )
```

`self.max_output_tokens` does not exist until Task 9. For this task, add the attribute now with one line at the end of `Council.__init__`:

```python
        # Replaced by the real config/argument resolution in the output-cap task.
        self.max_output_tokens: int | None = None
```

That keeps every task independently green.

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_cache.py -q -p no:cacheprovider'
$BR '.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
```
Expected: `test_cache.py` fully green; full suite around `872 passed`.

**Step 5: Commit**

```bash
git add src/conclave/cache.py src/conclave/council.py tests/test_cache.py
git commit -m "feat(cache): price snapshot fingerprint + output cap in identity, format v5 (DSE-1514)"
```

**Ship Round 3 as PR #2.** Title: `feat: bounded cost ceilings on the council manifest (DSE-1514 round 3)`.

---

# Round 4 — The output cap and the pre-flight spend gate (PR #3)

**Scope:** thread `max_output_tokens` end to end, enumerate the call plan, reserve it, and refuse. Expected suite after Round 4: around **902 passed**.

---

### Task 9: `max_output_tokens` threaded end to end

**Files:**
- Modify: `src/conclave/config.py:88-94`, `:229-265` (loader)
- Modify: `src/conclave/council.py:200-240` (`__init__`), `:461-510` (`fan_out`), `:531-590` + `:593-648` (both manifest builders), `:1178-1275` (`adjudicate`), `:1335-1382` (`_record_adjudication`), `:1040-1060` (the `extract_verdict_fn` call)
- Modify: `src/conclave/manifest.py` (widen both `generation_settings` annotations)
- Modify: `src/conclave/providers.py:34-47` (`receipt_from_answer`)
- Modify: `src/conclave/verdict_synthesis.py:228-250`, `:546-560`, `:646-656`, `:684-692`
- Modify: `src/conclave/streaming.py:101-110`, `:368-375`
- Test: `tests/test_output_budget_plumbing.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_output_budget_plumbing.py  (append)
"""DSE-1514: the cap is a COUNCIL setting, not just an adapter parameter."""


def test_config_reads_and_sanitizes_max_output_tokens(tmp_path, monkeypatch):
    from conclave.config import clear_config_cache, load_config

    path = tmp_path / "config.yml"
    monkeypatch.setenv("CONCLAVE_CONFIG", str(path))

    path.write_text("max_output_tokens: 1024\n", encoding="utf-8")
    clear_config_cache()
    assert load_config().max_output_tokens == 1024

    path.write_text("max_output_tokens: not-a-number\n", encoding="utf-8")
    clear_config_cache()
    assert load_config().max_output_tokens is None

    path.write_text("max_output_tokens: 0\n", encoding="utf-8")
    clear_config_cache()
    assert load_config().max_output_tokens is None
    clear_config_cache()


async def test_the_cap_reaches_every_member_and_adjudication_call(monkeypatch, keys):
    import conclave.council as council_mod
    import conclave.verdict_synthesis as verdict_mod
    from conclave.council import Council
    from conclave.models import ModelAnswer

    seen: list[int | None] = []

    async def spy(name, model_id, messages, *, max_output_tokens=None, **kwargs):
        seen.append(max_output_tokens)
        return ModelAnswer(name=name, model_id=model_id, answer="ok")

    monkeypatch.setattr(council_mod, "call_model", spy)
    monkeypatch.setattr(verdict_mod, "call_model", spy)
    council = Council(models=["grok"], synthesizer="claude", max_output_tokens=777)
    await council.ask("q")

    # member fan-out + synthesis + verdict extraction + verdict repair
    assert len(seen) >= 4
    assert set(seen) == {777}


async def test_no_cap_configured_sends_nothing_new(monkeypatch, keys):
    import conclave.council as council_mod
    from conclave.council import Council
    from conclave.models import ModelAnswer

    seen: list[int | None] = []

    async def spy(name, model_id, messages, *, max_output_tokens=None, **kwargs):
        seen.append(max_output_tokens)
        return ModelAnswer(name=name, model_id=model_id, answer="ok")

    monkeypatch.setattr(council_mod, "call_model", spy)
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)
    await council.ask("q", synthesize=False)
    assert seen == [None]


async def test_the_cap_is_recorded_in_generation_settings_only_when_set(monkeypatch, keys):
    import conclave.council as council_mod
    from conclave.council import Council
    from conclave.models import ModelAnswer

    async def ok(name, model_id, messages, **kwargs):
        return ModelAnswer(name=name, model_id=model_id, answer="ok")

    monkeypatch.setattr(council_mod, "call_model", ok)

    capped = await Council(
        models=["grok"], synthesizer="grok", max_output_tokens=256, extract_verdict=False
    ).ask("q", synthesize=False)
    assert capped.manifest.generation_settings["max_output_tokens"] == 256
    assert capped.manifest.receipts[0].generation_settings["max_output_tokens"] == 256

    plain = await Council(models=["grok"], synthesizer="grok", extract_verdict=False).ask(
        "q", synthesize=False
    )
    assert "max_output_tokens" not in plain.manifest.generation_settings
    assert "max_output_tokens" not in plain.manifest.receipts[0].generation_settings


async def test_streaming_members_and_synthesis_receive_the_cap(monkeypatch, keys):
    import conclave.streaming as streaming_mod
    from conclave.council import Council
    from conclave.models import ModelAnswer

    seen: list[int | None] = []

    async def spy_stream(name, model_id, messages, *, max_output_tokens=None, **kwargs):
        seen.append(max_output_tokens)
        yield "tok"
        yield ModelAnswer(name=name, model_id=model_id, answer="tok")

    monkeypatch.setattr(streaming_mod, "call_model_stream", spy_stream)
    council = Council(
        models=["grok"], synthesizer="claude", max_output_tokens=333, extract_verdict=False
    )
    async for _event in council.ask_stream("q"):
        pass
    assert seen and set(seen) == {333}
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_output_budget_plumbing.py -q -p no:cacheprovider'
```
Expected: `TypeError: Council.__init__() got an unexpected keyword argument 'max_output_tokens'`.

**Step 3: Write the implementation**

**`config.py`** — add the field after `converge_threshold`:

```python
    max_output_tokens: int | None = None
```

with this docstring entry:

```
        max_output_tokens: opt-in hard ceiling on output tokens for EVERY call a
            council makes -- members, synthesis, judge, verdict extraction and
            its repair retry, and the streaming paths. ``None`` (the default)
            leaves each provider's own default in place, exactly as today. It is
            also the precondition for ``--max-spend-usd``: a run whose output is
            unbounded cannot have its spend bounded, so the gate refuses rather
            than inventing a number.
```

and a coercer mirroring `_coerce_threshold`:

```python
def _coerce_max_output_tokens(value: Any) -> int | None:
    """Coerce a config ``max_output_tokens`` value to a positive int, or ``None``.

    A non-integer, boolean, or non-positive value degrades to ``None`` (cap off)
    with a warning, matching this module's resilient-loading convention: a bad
    config field never crashes a run, it just disables the optional feature.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("max_output_tokens %r is not an integer; disabling the output cap", value)
        return None
    if value < 1:
        logger.warning("max_output_tokens %s is not positive; disabling the output cap", value)
        return None
    return value
```

Wire it in `_load_config_uncached`: `max_output_tokens = _coerce_max_output_tokens(raw.get("max_output_tokens"))`, then pass it into the `ConclaveConfig(...)` construction.

**`council.py`** — add the constructor parameter after `source_bundle_digest`:

```text
        max_output_tokens: int | None = None,
```

and replace the Task 8 placeholder with the real resolution:

```python
        # Explicit override wins; otherwise defer to config (off by default).
        # A cap is what makes a run's output -- and therefore its spend --
        # boundable at all; see Council.plan_calls and the --max-spend-usd gate.
        self.max_output_tokens = (
            self.config.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
```

Add the matching `Args:` docstring entry, and a helper beside `_available_members`:

```python
    def _generation_settings(self) -> dict[str, float | int]:
        """The generation settings actually used, for the manifest and receipts.

        ``max_output_tokens`` appears ONLY when a cap is configured, so an
        uncapped run's manifest is byte-identical to v1.3.0's.
        """
        settings: dict[str, float | int] = {
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if self.max_output_tokens is not None:
            settings["max_output_tokens"] = self.max_output_tokens
        return settings
```

Replace both manifest builders' `generation_settings={"temperature": ..., "timeout": ...}` with `generation_settings=self._generation_settings()`.

Pass `max_output_tokens=self.max_output_tokens` to:

* every `call_model(...)` in `fan_out` and in `adjudicate`;
* every `receipt_from_answer(...)` in `_build_manifest`, `_build_elite_manifest`, and `_record_adjudication`;
* the `extract_verdict_fn(...)` call in `_apply_verdict`.

**`manifest.py`** — widen both `generation_settings` annotations from `dict[str, float]` to `dict[str, float | int]` (on `ProviderExecutionReceipt` and `ModelHarnessManifest`). Pydantic's smart union keeps an `int` an `int`; existing float values are unaffected, so no existing test changes.

**`providers.py`** — add `max_output_tokens: int | None = None` to `receipt_from_answer`'s keyword-only parameters and build the receipt's settings the same way:

```python
    generation_settings: dict[str, float | int] = {"temperature": temperature, "timeout": timeout}
    if max_output_tokens is not None:
        generation_settings["max_output_tokens"] = max_output_tokens
```

**`verdict_synthesis.py`** — add `max_output_tokens: int | None = None` to `extract_verdict`'s and `_verdict_attempt_receipt`'s keyword-only parameters; pass it to **both** `model_caller(...)` calls (the initial extraction and the repair retry) and to both `_verdict_attempt_receipt(...)` calls. Document it: "the hard output ceiling for both the initial extraction call and its repair retry; ``None`` leaves the provider default in place."

**`streaming.py`** — pass `max_output_tokens=council.max_output_tokens` to `call_model_stream` in `_drive_member` and in `_stream_synthesis`.

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_output_budget_plumbing.py -q -p no:cacheprovider'
$BR '.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
```
Expected: `test_output_budget_plumbing.py` green with 5 new cases; full suite around `877 passed`.

**Step 5: Commit**

```bash
git add src/conclave/config.py src/conclave/council.py src/conclave/manifest.py \
        src/conclave/providers.py src/conclave/verdict_synthesis.py src/conclave/streaming.py \
        tests/test_output_budget_plumbing.py
git commit -m "feat(council): thread max_output_tokens through every call path (DSE-1514)"
```

---

### Task 10: `Council.plan_calls` — the worst-case call plan

**Files:**
- Modify: `src/conclave/council.py` (append `PlannedCall`, `CallPlan`, `_keyed_chain`, `plan_calls`)
- Modify: `src/conclave/verdict_synthesis.py` (two new module constants)
- Test: `tests/test_spend_plan.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_spend_plan.py
"""The worst-case call plan per mode, derived from modes.py arithmetic (DSE-1514)."""

from __future__ import annotations

import pytest

from conclave.council import Council

MEMBERS = ["grok", "gemini", "openai"]  # N = 3


def _council(**kwargs) -> Council:
    kwargs.setdefault("models", MEMBERS)
    kwargs.setdefault("synthesizer", "claude")
    kwargs.setdefault("max_output_tokens", 1_000)
    return Council(**kwargs)


@pytest.mark.parametrize(
    ("mode", "kwargs", "expected"),
    [
        ("raw", {}, 3),  # N
        ("synthesize", {}, 3 + 1 + 2),  # N + C + 2C
        ("vote", {"choices": ["a", "b"]}, 3),  # N
        ("debate", {"rounds": 3}, 3 * 3 + 1),  # N*R + C
        ("adversarial", {}, 3 + 1),  # N + C
        ("elite", {}, 3 * 3 + 1 + 2),  # 3N + C + 2C
    ],
)
def test_worst_case_call_counts_per_mode(keys, mode, kwargs, expected):
    plan = _council().plan_calls(mode, "q", **kwargs)
    assert len(plan.calls) == expected
    assert plan.mode == mode
    assert plan.member_count == 3
    assert plan.chain_count == 1


def test_a_longer_chain_multiplies_every_adjudication_role(keys):
    council = _council(synthesizer="claude>grok>gemini")  # C = 3
    assert len(council.plan_calls("synthesize", "q").calls) == 3 + 3 + 6
    assert len(council.plan_calls("elite", "q").calls) == 9 + 3 + 6
    assert len(council.plan_calls("adversarial", "q").calls) == 3 + 3
    assert council.plan_calls("synthesize", "q").chain_count == 3


def test_an_unkeyed_chain_candidate_is_not_planned(monkeypatch, keys):
    # mistral is unkeyed here -> it can never be called, so it can never cost
    # anything, so it is not in the plan.
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    council = _council(synthesizer="claude>mistral")
    plan = council.plan_calls("synthesize", "q")
    assert plan.chain_count == 1
    assert len(plan.calls) == 3 + 1 + 2


def test_verdict_extraction_off_removes_exactly_two_calls_per_candidate(keys):
    on = _council().plan_calls("synthesize", "q")
    off = _council(extract_verdict=False).plan_calls("synthesize", "q")
    assert len(on.calls) - len(off.calls) == 2


def test_every_planned_call_is_bounded_and_names_its_model(keys):
    for call in _council().plan_calls("elite", "q").calls:
        assert call.max_output_tokens == 1_000
        assert "/" in call.model_id
        assert call.prompt_token_upper_bound >= len(b"q")
        assert call.prompt_template_token_allowance >= 0
        assert call.provider_framing_token_allowance >= 64
        assert call.upstream_output_call_count >= 0


def test_downstream_phases_declare_their_upstream_dependencies(keys):
    by_phase: dict[str, list] = {}
    for call in _council().plan_calls("elite", "q").calls:
        by_phase.setdefault(call.phase, []).append(call)

    assert all(c.upstream_output_call_count == 0 for c in by_phase["initial"])
    assert all(c.upstream_output_call_count == 3 for c in by_phase["critique"])  # N initials
    assert all(c.upstream_output_call_count == 6 for c in by_phase["revision"])  # N + N
    assert by_phase["synthesis"][0].upstream_output_call_count == 3  # N revisions
    assert by_phase["verdict_extraction"][0].upstream_output_call_count == 3
    assert by_phase["verdict_repair"][0].upstream_output_call_count == 4  # + its own attempt


def test_an_unknown_mode_is_a_value_error(keys):
    with pytest.raises(ValueError, match="unknown mode"):
        _council().plan_calls("telepathy", "q")


def test_planning_without_an_output_cap_is_refused(keys):
    with pytest.raises(ValueError, match="cannot bound spend: no output cap"):
        Council(models=MEMBERS, synthesizer="claude").plan_calls("synthesize", "q")
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_spend_plan.py -q -p no:cacheprovider'
```
Expected: `AttributeError: 'Council' object has no attribute 'plan_calls'`.

**Step 3: Write the implementation**

First, in `src/conclave/verdict_synthesis.py`, add two module constants so the planner never rebuilds a JSON schema per run:

```python
# DSE-1514: byte sizes the pre-flight spend planner needs without making a call.
# The extraction schema and its system prompt are fixed, so their UTF-8 byte cost
# is a constant of this module rather than a per-run guess.
VERDICT_CONTRACT_BYTES = len(
    json.dumps(
        verdict_extraction_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
)
VERDICT_TEMPLATE_PROBE = _EXTRACTION_SYSTEM
```

Then append to `src/conclave/council.py` — the dataclasses near `AdjudicationOutcome`:

```python
@dataclass(frozen=True)
class PlannedCall:
    """One provider call a mode COULD make, bounded without making it.

    Every field is knowable before the first call: the resolved model, the
    output cap the call will be issued with, and a three-part input bound (exact
    prompt bytes, fixed template bytes, provider framing) plus a count of
    upstream calls whose not-yet-produced output this call's input will embed.

    Attributes:
        phase: The manifest phase this call would be recorded under.
        name: Friendly member / candidate name.
        model_id: Resolved provider-prefixed model id.
        prompt_token_upper_bound: UTF-8 bytes of the exact known content.
        prompt_template_token_allowance: UTF-8 bytes of the fixed system + user
            template wording that will surround it.
        provider_framing_token_allowance: ``64 + 16 * messages`` (+256 with a
            structured-output contract), mirroring the eval runner.
        upstream_output_call_count: How many upstream calls' outputs this call's
            input embeds. Multiplied by the output cap and the snapshot's
            ``max_output_bytes_per_token`` to bound them.
        max_output_tokens: The hard output cap this call would carry.
    """

    phase: str
    name: str
    model_id: str
    prompt_token_upper_bound: int
    prompt_template_token_allowance: int
    provider_framing_token_allowance: int
    upstream_output_call_count: int
    max_output_tokens: int


@dataclass(frozen=True)
class CallPlan:
    """The complete worst-case call plan for one run of one mode."""

    mode: str
    calls: tuple[PlannedCall, ...]
    member_count: int
    chain_count: int
```

the constants beside `_SYNTH_SYSTEM`:

```python
_VALID_PLAN_MODES = frozenset({"raw", "synthesize", "vote", "debate", "adversarial", "elite"})

_NO_OUTPUT_CAP_MESSAGE = (
    "cannot bound spend: no output cap (set --max-output-tokens or config max_output_tokens)"
)
```

and the methods beside `_available_members`:

```python
def _keyed_chain(self) -> list[tuple[str, str]]:
    """Resolve the synthesizer chain to the candidates that could be CALLED.

    :meth:`adjudicate` skips an unkeyed candidate without making a call, so
    an unkeyed candidate cannot cost anything and is excluded from the plan.
    ``registry.key_present`` returns ``True`` for an unknown provider prefix,
    which errs toward INCLUDING the call -- the safe direction for a ceiling.
    """
    pairs = [(name, self.config.resolve_model_id(name)) for name in self.synthesizer_chain]
    return [(name, model_id) for name, model_id in pairs if key_present(model_id)]


def plan_calls(
    self,
    mode: str,
    prompt: str,
    *,
    rounds: int = 2,
    proposer: str | None = None,
    choices: list[str] | None = None,
) -> CallPlan:
    """Enumerate every provider call this mode could make, worst case (DSE-1514).

    The counts are derived from :mod:`conclave.modes` and
    :meth:`_apply_verdict`, not from a remembered formula. With ``N`` keyed
    members, ``C`` keyed chain candidates, ``R`` debate rounds, and ``V`` = 1
    when verdict extraction is on:

    * ``raw`` -- ``N``: fan-out only.
    * ``synthesize`` -- ``N + C + 2CV``: fan-out, the chain, then
      extract+repair per candidate.
    * ``vote`` -- ``N``: fan-out only; no adjudication.
    * ``debate`` -- ``N*R + C``: every round at full membership (drop-out
      only shrinks it), then the final consolidation chain.
    * ``adversarial`` -- ``N + C``: ``k`` proposer attempts plus ``N - k``
      critics is exactly ``N`` for every ``k``; then the judge chain.
    * ``elite`` -- ``3N + C + 2CV``: three phases at full membership, then
      synthesis and verdict extraction.

    Convergence early-stop, member drop-out, and a proposer succeeding on the
    first try all make a real run CHEAPER than its plan. A plan is never an
    under-count, which is what makes it usable as a spend gate.

    Args:
        mode: One of ``raw``/``synthesize``/``vote``/``debate``/
            ``adversarial``/``elite``.
        prompt: The exact user prompt (bounded by its UTF-8 byte length).
        rounds: Debate rounds; ignored for other modes.
        proposer: Adversarial proposer. It does not change the COUNT (see
            above) and is accepted only for signature parity with the modes.
        choices: Vote choices, which enlarge the vote prompt template.

    Returns:
        The :class:`CallPlan`.

    Raises:
        ValueError: ``mode`` is not a known deliberation mode, or
            ``max_output_tokens`` is not configured (an unbounded output
            cannot be planned).
    """
    if mode not in _VALID_PLAN_MODES:
        raise ValueError(f"unknown mode for call planning: {mode}")
    cap = self.max_output_tokens
    if cap is None:
        raise ValueError(_NO_OUTPUT_CAP_MESSAGE)

    members, _skipped = self._available_members()
    chain = self._keyed_chain()
    n_members = len(members)
    prompt_bytes = len(prompt.encode("utf-8"))
    calls: list[PlannedCall] = []

    def member_calls(phase: str, *, template: str, upstream: int) -> None:
        for name, model_id in members:
            calls.append(
                PlannedCall(
                    phase=phase,
                    name=name,
                    model_id=model_id,
                    prompt_token_upper_bound=prompt_bytes,
                    prompt_template_token_allowance=len(template.encode("utf-8")),
                    provider_framing_token_allowance=64 + (16 * 2),
                    upstream_output_call_count=upstream,
                    max_output_tokens=cap,
                )
            )

    def chain_calls(phase: str, *, template: str, upstream: int, contract: bool) -> None:
        for name, model_id in chain:
            calls.append(
                PlannedCall(
                    phase=phase,
                    name=name,
                    model_id=model_id,
                    prompt_token_upper_bound=(
                        prompt_bytes + (_VERDICT_CONTRACT_BYTES if contract else 0)
                    ),
                    prompt_template_token_allowance=len(template.encode("utf-8")),
                    provider_framing_token_allowance=(64 + (16 * 3) + (256 if contract else 0)),
                    upstream_output_call_count=upstream,
                    max_output_tokens=cap,
                )
            )

    if mode in ("raw", "synthesize"):
        member_calls("member", template="", upstream=0)
    elif mode == "vote":
        member_calls(
            "member",
            template=prompts.VOTE_SYSTEM + prompts.vote_user("", choices or []),
            upstream=0,
        )
    elif mode == "debate":
        member_calls("round-1", template="", upstream=0)
        for round_no in range(2, max(1, rounds) + 1):
            member_calls(
                f"round-{round_no}",
                template=(
                    prompts.DEBATE_SYSTEM
                    + prompts.debate_round_user("", round_no, max(1, rounds), "")
                ),
                upstream=n_members,
            )
    elif mode == "adversarial":
        # CORRECTED 2026-09-04 (Round 4 review, Fix A): the worked example
        # originally shown here treated every member uniformly as
        # `member_calls("proposal", template="", upstream=0)`, which gave
        # the right TOTAL count (N + C) but the wrong per-call shape -- it
        # never gave a critic call an upstream dependency on the proposal,
        # so the byte bound for every critic call was silently too small.
        # `run_adversarial` embeds the proposal's answer text in EVERY
        # critic call (`_critic_messages_for`) and embeds the proposal AND
        # every critique in the judge call (`judge_user`). A real run's k
        # proposer attempts + (N-k) critics always equals N, for every k;
        # the BYTE-worst-case is k=1 (one proposer succeeds immediately),
        # which maximizes the number of upstream-embedding critic calls.
        # The corrected shape:
        if members:
            member_calls("proposal", targets=members[:1], template="", upstream=0)
            member_calls(
                "critique",
                targets=members[1:],
                template=prompts.CRITIC_SYSTEM + prompts.critic_user("", ""),
                upstream=1,
            )
    elif mode == "elite":
        member_calls("initial", template="", upstream=0)
        member_calls(
            "critique",
            template=prompts.ELITE_CRITIC_SYSTEM + prompts.elite_critic_user("", []),
            upstream=n_members,
        )
        member_calls("revision", template=prompts.ELITE_REVISION_SYSTEM, upstream=2 * n_members)

    if mode == "debate":
        chain_calls(
            "debate_final",
            template=(
                prompts.DEBATE_FINAL_SYSTEM + prompts.debate_final_user("", max(1, rounds), "")
            ),
            upstream=n_members,
            contract=False,
        )
    elif mode == "adversarial":
        chain_calls(
            "judge",
            template=prompts.JUDGE_SYSTEM + prompts.judge_user("", "", "", ""),
            upstream=n_members,
            contract=False,
        )
    elif mode in ("synthesize", "elite"):
        chain_calls("synthesis", template=_SYNTH_SYSTEM, upstream=n_members, contract=False)
        if self.extract_verdict_enabled:
            chain_calls(
                "verdict_extraction",
                template=_VERDICT_TEMPLATE_PROBE,
                upstream=n_members,
                contract=True,
            )
            chain_calls(
                "verdict_repair",
                template=_VERDICT_TEMPLATE_PROBE,
                upstream=n_members + 1,
                contract=True,
            )

    return CallPlan(
        mode=mode,
        calls=tuple(calls),
        member_count=n_members,
        chain_count=len(chain),
    )
```

`council.py` also needs `from . import prompts` at module level (it currently reaches prompts only via `modes.py`) and, at module level, the two verdict constants imported under private aliases:

```python
from .verdict_synthesis import VERDICT_CONTRACT_BYTES as _VERDICT_CONTRACT_BYTES
from .verdict_synthesis import VERDICT_TEMPLATE_PROBE as _VERDICT_TEMPLATE_PROBE
```

If that import creates a cycle (`verdict_synthesis` imports `manifest` and `models`, not `council`, so it should not), move it inside `plan_calls` as a deferred import and say so in the commit message.

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_spend_plan.py -q -p no:cacheprovider'
```
Expected: `13 passed` (8 test functions, one of them with 6 parametrized cases).

**Step 5: Commit**

```bash
git add src/conclave/council.py src/conclave/verdict_synthesis.py tests/test_spend_plan.py
git commit -m "feat(council): plan_calls enumerates the worst-case call plan per mode (DSE-1514)"
```

---

### Task 11: Reserve the plan and refuse

**Files:**
- Modify: `src/conclave/pricing.py` (append the exception types)
- Modify: `src/conclave/council.py` (`__init__`, `_reserve_plan`, `_enforce_spend_cap`, four wiring sites)
- Modify: `src/conclave/__init__.py` (exports)
- Test: `tests/test_spend_gate.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_spend_gate.py
"""The pre-flight spend gate refuses BEFORE the first provider call (DSE-1514)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from conclave.council import Council
from conclave.pricing import SpendCapExceeded, SpendRefused, SpendUnboundable
from tests.test_pricing_receipts import _install_snapshot, _snapshot


def test_a_spend_cap_without_an_output_cap_is_refused_at_construction(keys):
    with pytest.raises(SpendUnboundable) as excinfo:
        Council(models=["grok"], synthesizer="grok", max_spend_usd=Decimal("0.40"))
    assert str(excinfo.value) == (
        "cannot bound spend: no output cap (set --max-output-tokens or config max_output_tokens)"
    )
    assert isinstance(excinfo.value, SpendRefused)


async def test_an_over_budget_plan_refuses_before_any_provider_call(monkeypatch, keys):
    import conclave.council as council_mod

    _install_snapshot(
        monkeypatch,
        _snapshot("xai/grok-4.3", "gemini/gemini-2.5-pro", "anthropic/claude-sonnet-4-6"),
    )
    calls: list[str] = []

    async def tripwire(name, model_id, messages, **kwargs):
        calls.append(name)
        raise AssertionError("the gate must refuse before any provider call")

    monkeypatch.setattr(council_mod, "call_model", tripwire)
    council = Council(
        models=["grok", "gemini"],
        synthesizer="claude",
        max_output_tokens=100_000,
        max_spend_usd=Decimal("0.000001"),
    )
    with pytest.raises(SpendCapExceeded) as excinfo:
        await council.ask("q")

    error = excinfo.value
    assert error.cap == Decimal("0.000001")
    assert error.reserved > error.cap
    assert error.call_count == 2 + 1 + 2
    assert "reserved" in str(error) and "cap" in str(error) and "calls" in str(error)
    assert calls == []


async def test_an_under_budget_plan_runs_normally(monkeypatch, keys, patch_call_model):
    from tests.conftest import make_response

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
        extract_verdict=False,
    )
    result = await council.ask("q", synthesize=False)
    assert result.successful_answers
    assert result.manifest.cost_ceiling_usd is not None


async def test_an_unpriced_model_in_the_plan_refuses_rather_than_guessing(monkeypatch, keys):
    import conclave.council as council_mod

    # grok is priced; the claude synthesizer is not.
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def tripwire(name, model_id, messages, **kwargs):
        raise AssertionError("no call may happen")

    monkeypatch.setattr(council_mod, "call_model", tripwire)
    council = Council(
        models=["grok"],
        synthesizer="claude",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
    )
    with pytest.raises(SpendUnboundable, match="no priced rate for anthropic/claude-sonnet-4-6"):
        await council.ask("q")


async def test_a_missing_snapshot_refuses_the_gate(monkeypatch, keys):
    _install_snapshot(monkeypatch, None)
    council = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
    )
    with pytest.raises(SpendUnboundable, match="price snapshot unavailable"):
        await council.ask("q", synthesize=False)


async def test_a_cache_hit_is_never_gated(monkeypatch, tmp_path, keys, patch_call_model):
    from tests.conftest import make_response

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    patch_call_model(lambda model_id, messages: make_response("ok"))

    cheap = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
        cache=True,
        extract_verdict=False,
    )
    await cheap.ask("q", synthesize=False)

    # Same identity, an impossible cap: the hit costs nothing, so it is served.
    strict = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("0.000001"),
        cache=True,
        extract_verdict=False,
    )
    hit = await strict.ask("q", synthesize=False)
    assert hit.cached is True


async def test_the_gate_also_guards_the_streaming_path(monkeypatch, keys):
    import conclave.streaming as streaming_mod

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def tripwire(*args, **kwargs):
        raise AssertionError("no stream may start")
        yield  # pragma: no cover

    monkeypatch.setattr(streaming_mod, "call_model_stream", tripwire)
    council = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=100_000,
        max_spend_usd=Decimal("0.000001"),
        extract_verdict=False,
    )
    with pytest.raises(SpendCapExceeded):
        async for _event in council.ask_stream("q", synthesize=False):
            pass


def test_no_spend_flags_means_no_gate_at_all(keys):
    council = Council(models=["grok"], synthesizer="grok")
    assert council.max_spend_usd is None
    assert council.max_output_tokens is None
```

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_spend_gate.py -q -p no:cacheprovider'
```
Expected: `ImportError: cannot import name 'SpendRefused' from 'conclave.pricing'`.

**Step 3: Write the implementation**

Append to `src/conclave/pricing.py`:

```python
class SpendRefused(Exception):
    """Base class for a pre-flight spend refusal.

    Raised BEFORE any provider call. Every subclass maps to CLI exit code 4.
    Refusing is the honest outcome when a run cannot be bounded: the alternative
    is to invent a number, which is the exact failure this module exists to
    prevent.
    """


class SpendUnboundable(SpendRefused):
    """The call plan cannot be priced at all, so no cap can be enforced.

    Causes: no output cap configured, no price snapshot available, or a model in
    the plan with no snapshot entry. Never a fallback rate, never a guess.
    """


class SpendCapExceeded(SpendRefused):
    """The fully-priced worst-case plan reserves more than the stated cap.

    Attributes:
        reserved: The pessimistic total for the whole plan, in USD.
        cap: The operator's stated cap, in USD.
        call_count: How many provider calls the plan enumerated.
    """

    def __init__(self, reserved: Decimal, cap: Decimal, call_count: int) -> None:
        self.reserved = reserved
        self.cap = cap
        self.call_count = call_count
        super().__init__(
            f"refusing to run: reserved {reserved} USD for {call_count} calls "
            f"exceeds the cap of {cap} USD"
        )
```

In `src/conclave/council.py`, add the constructor parameter after `max_output_tokens`:

```text
        max_spend_usd: Decimal | None = None,
```

and, at the end of `__init__`:

```python
        # A spend cap without an output cap is not enforceable: output is the
        # unbounded term. Refuse at construction rather than at the first call,
        # so a library caller cannot get halfway into a run before finding out.
        self.max_spend_usd = max_spend_usd
        if max_spend_usd is not None and self.max_output_tokens is None:
            raise SpendUnboundable(_NO_OUTPUT_CAP_MESSAGE)
```

Add the two methods beside `plan_calls`:

```python
def _reserve_plan(self, plan: CallPlan) -> Decimal:
    """Price a :class:`CallPlan` pessimistically against the snapshot.

    Args:
        plan: The worst-case plan from :meth:`plan_calls`.

    Returns:
        The reserved total in USD -- an upper bound on what the run can cost.

    Raises:
        SpendUnboundable: No snapshot, or any planned call's model has no
            snapshot entry. Never falls back to a similar model's rate.
    """
    snapshot = load_default_price_snapshot()
    if snapshot is None:
        raise SpendUnboundable("cannot bound spend: price snapshot unavailable")
    total = Decimal("0")
    for call in plan.calls:
        rates = snapshot.rates_for(call.model_id)
        if rates is None:
            raise SpendUnboundable(
                f"cannot bound spend: no priced rate for {call.model_id} "
                f"in snapshot {snapshot.digest()} ({snapshot.captured_at.isoformat()})"
            )
        total += reserve_cost(
            rates,
            prompt_token_upper_bound=call.prompt_token_upper_bound,
            prompt_template_token_allowance=call.prompt_template_token_allowance,
            provider_framing_token_allowance=call.provider_framing_token_allowance,
            upstream_output_token_ceilings=(
                (call.max_output_tokens,) * call.upstream_output_call_count
            ),
            upstream_output_bytes_per_token=rates.max_output_bytes_per_token,
            max_output_tokens=call.max_output_tokens,
        ).reserved_cost_usd
    return total


def _enforce_spend_cap(
    self,
    mode: str,
    prompt: str,
    *,
    rounds: int | None = None,
    proposer: str | None = None,
    choices: list[str] | None = None,
) -> None:
    """Refuse an over-budget or unboundable run BEFORE any provider call.

    A no-op when ``max_spend_usd`` is unset, so a run with no spend flags is
    byte-identical to today. Deliberately NOT applied to a cache hit: a hit
    makes no provider call and therefore cannot exceed any cap.

    Raises:
        SpendUnboundable: The plan cannot be priced.
        SpendCapExceeded: The priced plan exceeds the cap.
    """
    if self.max_spend_usd is None:
        return
    plan = self.plan_calls(
        mode,
        prompt,
        rounds=2 if rounds is None else rounds,
        proposer=proposer,
        choices=choices,
    )
    reserved = self._reserve_plan(plan)
    if reserved > self.max_spend_usd:
        raise SpendCapExceeded(reserved, self.max_spend_usd, len(plan.calls))
    logger.info(
        "spend gate: reserved %s USD for %d calls, under the %s USD cap",
        reserved,
        len(plan.calls),
        self.max_spend_usd,
    )
```

Wire it at four sites. In `_cached_run`, immediately before each `await run()` — in the no-cache branch, and in the live-miss branch **after** the cache-hit early return:

```python
if not self.cache_enabled:
    self._enforce_spend_cap(mode, prompt, rounds=rounds, proposer=proposer, choices=choices)
    result = await run()
```

```python
        self._enforce_spend_cap(mode, prompt, rounds=rounds, proposer=proposer, choices=choices)
        result = await run()
```

In `ask_stream`, immediately before each `stream_ask(...)` — after the cache-hit early return, and in the no-cache branch.

Import `SpendCapExceeded`, `SpendRefused`, and `SpendUnboundable` in `council.py`'s `from .pricing import (...)` block, and re-export all three plus `PriceSnapshot` from `src/conclave/__init__.py`'s `__all__`.

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_spend_gate.py -q -p no:cacheprovider'
$BR '.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
```
Expected: `8 passed` in `test_spend_gate.py`; full suite around `898 passed`.

**Step 5: Commit**

```bash
git add src/conclave/pricing.py src/conclave/council.py src/conclave/__init__.py \
        tests/test_spend_gate.py
git commit -m "feat(council): pre-flight spend gate refuses before the first provider call (DSE-1514)"
```

---

### Task 12: CLI surface + exit code 4

**Files:**
- Modify: `src/conclave/cli.py:508` (add `_SPEND_REFUSED_EXIT_CODE`), `:539-631` (two new options), `:631-675` (the `ask` docstring exit-code list), `:704-712` (council construction + the run block)
- Test: `tests/test_cli.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_cli.py  (append)
"""DSE-1514: --max-output-tokens / --max-spend-usd and the exit-code-4 refusal."""


def test_spend_refusal_exit_code_is_four_and_distinct():
    from conclave import cli

    assert cli._SPEND_REFUSED_EXIT_CODE == 4
    assert cli._SPEND_REFUSED_EXIT_CODE != cli._DEGRADED_EXIT_CODE


def test_spend_cap_without_an_output_cap_exits_four(runner, keys):
    from conclave.cli import app

    result = runner.invoke(app, ["ask", "q", "--council", "grok", "--max-spend-usd", "0.40"])
    assert result.exit_code == 4
    assert "cannot bound spend: no output cap" in result.output + (result.stderr or "")


def test_an_over_budget_run_exits_four_and_names_reserved_cap_and_count(runner, monkeypatch, keys):
    import conclave.council as council_mod
    from conclave.cli import app
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", "anthropic/claude-sonnet-4-6"))

    async def tripwire(name, model_id, messages, **kwargs):
        raise AssertionError("the CLI gate must refuse before any provider call")

    monkeypatch.setattr(council_mod, "call_model", tripwire)
    result = runner.invoke(
        app,
        [
            "ask",
            "q",
            "--council",
            "grok",
            "--max-output-tokens",
            "100000",
            "--max-spend-usd",
            "0.000001",
        ],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 4
    assert "reserved" in combined and "0.000001" in combined and "calls" in combined


def test_an_unboundable_plan_exits_four_with_a_distinct_message(runner, monkeypatch, keys):
    from conclave.cli import app
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    # grok priced, the claude synthesizer is not.
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    result = runner.invoke(
        app,
        [
            "ask",
            "q",
            "--council",
            "grok",
            "--max-output-tokens",
            "512",
            "--max-spend-usd",
            "10.00",
        ],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 4
    assert "no priced rate for anthropic/claude-sonnet-4-6" in combined
    assert "reserved" not in combined


def test_a_non_numeric_spend_cap_is_a_usage_error(runner, keys):
    from conclave.cli import app

    result = runner.invoke(
        app, ["ask", "q", "--council", "grok", "--max-spend-usd", "cheap-please"]
    )
    assert result.exit_code == 2
    assert "--max-spend-usd" in result.output + (result.stderr or "")


def test_the_spend_cap_is_parsed_as_an_exact_decimal_never_a_float(runner, monkeypatch, keys):
    """0.4 through a float is 0.4000000000000000222; the cap must be exact."""
    from decimal import Decimal

    import conclave.cli as cli_mod
    from conclave.cli import app

    seen: dict[str, object] = {}
    real = cli_mod.Council

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "Council", spy)
    runner.invoke(
        app,
        [
            "ask",
            "q",
            "--council",
            "grok",
            "--max-output-tokens",
            "512",
            "--max-spend-usd",
            "0.4",
        ],
    )
    assert seen["max_spend_usd"] == Decimal("0.4")
    assert isinstance(seen["max_spend_usd"], Decimal)
    assert seen["max_output_tokens"] == 512


def test_the_json_payload_carries_the_ceiling_as_an_exact_string(
    runner, monkeypatch, keys, patch_call_model
):
    import json as json_mod

    from conclave.cli import app
    from tests.conftest import make_response
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    result = runner.invoke(app, ["ask", "q", "--council", "grok", "--mode", "raw", "--json"])
    manifest = json_mod.loads(result.output)["manifest"]
    assert manifest["estimated_cost"] is None
    assert isinstance(manifest["cost_ceiling_usd"], str)
    assert manifest["priced_as_of"] == "2026-09-03"
    assert manifest["price_snapshot_digest"].startswith("sha256:")
```

If `tests/test_cli.py` has no `runner` fixture, reuse whatever `CliRunner` construction the file already uses.

**Step 2: Run it and watch it fail**

```
$BR '.venv/bin/python -m pytest tests/test_cli.py -q -p no:cacheprovider -k "spend or ceiling or output_tokens"'
```
Expected: `AttributeError: module 'conclave.cli' has no attribute '_SPEND_REFUSED_EXIT_CODE'` and `No such option: --max-spend-usd`.

**Step 3: Write the implementation**

In `src/conclave/cli.py`, add `from decimal import Decimal, InvalidOperation` and `from .pricing import SpendCapExceeded, SpendRefused`, plus the exit code beside `_DEGRADED_EXIT_CODE`:

```python
# Distinct exit code for a pre-flight spend REFUSAL (DSE-1514): the run was
# never started because its worst-case cost could not be bounded, or was bounded
# and exceeded --max-spend-usd. Kept apart from 1 (nothing usable came back), 2
# (usage error), and 3 (degraded) because it is categorically different: nothing
# ran, nothing was spent, and retrying with a higher cap or an output cap is the
# fix. A caller doing `echo $?` adds one branch and reinterprets nothing.
_SPEND_REFUSED_EXIT_CODE = 4
```

Add the two options to `ask`, after `--stream`:

```text
    max_output_tokens: int | None = typer.Option(
        None,
        "--max-output-tokens",
        help=(
            "Hard ceiling on output tokens for every call this run makes "
            "(members, synthesizer, judge, verdict extraction and its repair). "
            "Defers to config `max_output_tokens` when unset. Required by "
            "--max-spend-usd: unbounded output cannot be bounded in dollars."
        ),
    ),
    max_spend_usd: str | None = typer.Option(
        None,
        "--max-spend-usd",
        help=(
            "Refuse the run BEFORE the first provider call if its worst-case "
            "call plan reserves more than this many USD, priced against the "
            "packaged dated snapshot at ceiling rates. Exits 4 on refusal. "
            "Requires --max-output-tokens (or config max_output_tokens)."
        ),
    ),
```

`max_spend_usd` is typed `str` on purpose: typer's `float` coercion would destroy the exactness the whole ticket rests on. Parse it right after the existing mode validation:

```python
spend_cap: Decimal | None = None
if max_spend_usd is not None:
    try:
        spend_cap = Decimal(max_spend_usd)
    except InvalidOperation:
        err_console.print(
            f"[red]--max-spend-usd must be an exact decimal amount, got '{max_spend_usd}'.[/red]"
        )
        raise typer.Exit(code=2) from None
    if spend_cap <= 0:
        err_console.print("[red]--max-spend-usd must be greater than zero.[/red]")
        raise typer.Exit(code=2)
```

Wrap the `Council(...)` construction:

```python
    try:
        c = Council(
            models=members,
            synthesizer=synthesizer,
            config=cfg,
            cache=cache,
            max_output_tokens=max_output_tokens,
            max_spend_usd=spend_cap,
        )
    except SpendRefused as refusal:
        err_console.print(f"[red]{refusal}[/red]")
        raise typer.Exit(code=_SPEND_REFUSED_EXIT_CODE) from None
```

and wrap every mode dispatch (streaming and buffered) so a refusal raised at run time lands on the same code:

```python
    try:
        ...  # the existing stream / debate / adversarial / vote / elite / ask dispatch
    except SpendCapExceeded as refusal:
        err_console.print(
            f"[red]{refusal}. Raise --max-spend-usd, lower --max-output-tokens, "
            f"or shrink the council.[/red]"
        )
        raise typer.Exit(code=_SPEND_REFUSED_EXIT_CODE) from None
    except SpendRefused as refusal:
        err_console.print(f"[red]{refusal}[/red]")
        raise typer.Exit(code=_SPEND_REFUSED_EXIT_CODE) from None
```

Extend the `ask` docstring's exit-code list:

```
    * 4 -- the run was REFUSED before any provider call (DSE-1514): either its
      worst-case call plan reserved more than ``--max-spend-usd`` (the message
      names the reserved total, the cap, and the call count), or the plan could
      not be bounded at all -- no output cap, no price snapshot, or a model with
      no snapshot entry. Nothing ran and nothing was spent. Distinct from 3
      (degraded): a degraded run happened and produced partial output; a refused
      run never started.
```

**Step 4: Run it and watch it pass**

```
$BR '.venv/bin/python -m pytest tests/test_cli.py -q -p no:cacheprovider'
$BR '.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
$BR '.venv/bin/ruff check . && .venv/bin/ruff format --check .'
```
Expected: `test_cli.py` green with 7 new cases; full suite around `902 passed`; ruff clean.

**Step 5: Commit**

```bash
git add src/conclave/cli.py tests/test_cli.py
git commit -m "feat(cli): --max-output-tokens, --max-spend-usd, refusal exit code 4 (DSE-1514)"
```

---

### Task 13: Cross-mode and secret-safety coverage

**Files:**
- Modify: `tests/test_manifest_all_modes.py` (append)
- Modify: `tests/test_secret_safety_matrix.py` (append)

**Step 1: Write the tests**

```python
# tests/test_manifest_all_modes.py  (append)
@pytest.mark.parametrize("mode", ["synthesize", "raw", "debate", "adversarial", "vote", "elite"])
async def test_every_mode_prices_its_manifest(monkeypatch, keys, patch_call_model, mode):
    """Pricing runs at the same chokepoint the manifest invariant runs at."""
    from decimal import Decimal

    from conclave.council import Council
    from tests.conftest import make_response
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    _install_snapshot(
        monkeypatch, _snapshot("xai/grok-4.3", "gemini/gemini-2.5-pro", "openai/gpt-4.1")
    )
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(
        models=["grok", "gemini", "openai"], synthesizer="openai", extract_verdict=False
    )

    if mode == "debate":
        result = await council.debate("q", rounds=2)
    elif mode == "adversarial":
        result = await council.adversarial("q")
    elif mode == "vote":
        result = await council.vote("q", choices=["a", "b"])
    elif mode == "elite":
        result = await council.elite("q")
    else:
        result = await council.ask("q", synthesize=(mode == "synthesize"))

    manifest = result.manifest
    assert manifest.price_snapshot_digest is not None
    assert manifest.priced_as_of is not None
    assert manifest.estimated_cost is None
    if manifest.receipts:
        assert manifest.cost_ceiling_usd == sum(
            (r.cost_ceiling_usd for r in manifest.receipts), Decimal("0")
        )


async def test_elite_prices_every_one_of_its_3n_plus_receipts(monkeypatch, keys, patch_call_model):
    from conclave.council import Council
    from tests.conftest import make_response
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    _install_snapshot(
        monkeypatch, _snapshot("xai/grok-4.3", "gemini/gemini-2.5-pro", "openai/gpt-4.1")
    )
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(
        models=["grok", "gemini", "openai"], synthesizer="openai", extract_verdict=False
    )
    manifest = (await council.elite("q")).manifest

    # 3 phases x 3 members + 1 synthesis = 10; every one of them priced.
    assert len(manifest.receipts) >= 3 * 3
    assert all(r.cost_ceiling_usd is not None for r in manifest.receipts)
    assert manifest.unpriced_receipts == 0
```

```python
# tests/test_secret_safety_matrix.py  (append)
def test_pricing_fields_never_un_verify_the_stamp():
    """A model id with an awkward substring must not break the stamp."""
    from decimal import Decimal

    from conclave.manifest import (
        SECRET_SAFETY_VERIFIED,
        ModelHarnessManifest,
        ProviderExecutionReceipt,
        verified_secret_safety,
    )

    manifest = ModelHarnessManifest(
        request_id="r",
        conclave_version="1.3.0",
        mode="synthesize",
        model_ids=["deepseek/deepseek-chat", "together/meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        receipts=[
            ProviderExecutionReceipt(
                name="deepseek",
                provider="deepseek",
                model_id="deepseek/deepseek-chat",
                cost_ceiling_usd=Decimal("0.000123"),
                cost_basis="reservation",
                generation_settings={
                    "temperature": 0.7,
                    "timeout": 120.0,
                    "max_output_tokens": 8,
                },
            )
        ],
        cost_ceiling_usd=Decimal("0.000123"),
        price_snapshot_digest="sha256:" + "e" * 64,
        priced_as_of="2026-09-03",
        unpriced_models=["together/meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        unpriced_receipts=0,
        pricing_warnings=[
            "unpriced_models_present",
            "price_snapshot_stale",
            "no_output_cap_configured",
        ],
    )
    assert verified_secret_safety(manifest) == SECRET_SAFETY_VERIFIED


def test_pricing_warnings_are_a_closed_vocabulary_in_the_code():
    """No pricing warning may be built by interpolation."""
    import re
    from pathlib import Path

    import conclave

    source = (Path(conclave.__file__).parent / "council.py").read_text(encoding="utf-8")
    appended = re.findall(r"warnings\.append\((.+?)\)", source)
    assert appended, "the pricing warning appends moved; update this guard"
    for expression in appended:
        assert expression.startswith('"') and expression.endswith('"'), (
            f"pricing warning must be a literal, got {expression}"
        )
```

**Step 2: Run**

```
$BR '.venv/bin/python -m pytest tests/test_manifest_all_modes.py tests/test_secret_safety_matrix.py -q -p no:cacheprovider'
```
Expected: green, 10 new cases (6 parametrized plus 4).

**Step 3: Commit**

```bash
git add tests/test_manifest_all_modes.py tests/test_secret_safety_matrix.py
git commit -m "test: cross-mode pricing coverage and secret-safety guard for ceilings (DSE-1514)"
```

---

### Task 14: Documentation

**Files:**
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `README.md` (new "Cost ceilings and spend gate" subsection)
- Modify: `docs/PRODUCT_DESIGN_DOCUMENT.md` §4a and §9
- Modify: `DOCUMENTATION_INDEX.md`
- Modify: `config.example.yml`
- Modify: `SYSTEM_CONTEXT_DIAGRAM.md` **only if** it enumerates manifest fields — check with `grep -n "estimated_cost" SYSTEM_CONTEXT_DIAGRAM.md` first

**Step 1: Write the content**

**`CHANGELOG.md` → `[Unreleased]` → `### Added`:**

> - **Bounded cost ceilings on the manifest (DSE-1514).** Every receipt now carries `cost_ceiling_usd` (exact `Decimal`, `ROUND_CEILING`) and `cost_basis` (`reported_usage` or `reservation`), and the manifest carries a run-level `cost_ceiling_usd`, `price_snapshot_digest`, `priced_as_of`, `unpriced_models`, `unpriced_receipts`, and a bounded `pricing_warnings` list. A ceiling is a falsifiable claim — *"this run cost no more than $X, priced against snapshot `<digest>` dated `<date>`"* — not an estimate. `estimated_cost` is untouched and stays `None`. **All-or-nothing:** one unpriced model or one unpriceable receipt leaves the run ceiling `None` rather than emitting a partial sum.
> - **Dated, vendor-cited price snapshot.** `src/conclave/data/prices-<date>.json` ships in the wheel. Every entry cites the vendor page its rate was read from; rates are rounded **up**; a model whose list price could not be verified is **omitted** (unpriced), never guessed. Nothing is fetched at runtime. A snapshot older than 90 days adds a `price_snapshot_stale` warning and still prices at exactly its recorded rates.
> - **`--max-output-tokens` / `max_output_tokens:`.** A hard output ceiling threaded to every call a council makes — members, synthesizer, judge, verdict extraction and its repair retry, and both streaming paths — and recorded in `generation_settings` when set.
> - **`--max-spend-usd` pre-flight spend gate.** Enumerates the worst-case call plan for the selected mode (`raw` `N`, `synthesize` `N+C+2C`, `vote` `N`, `debate` `N*R+C`, `adversarial` `N+C`, `elite` `3N+C+2C`, where `C` is the count of *keyed* synthesizer-chain candidates), reserves each call pessimistically at ceiling rates, and **refuses before the first provider call** with the reserved total, the cap, and the call count. New exit code **4**. A plan that cannot be bounded — no output cap, no snapshot, or an unpriced model — refuses with a distinct message rather than guessing.
>
> `### Changed`
> - Cache format version `4` → `5`: identity now carries the price-snapshot rate fingerprint and `max_output_tokens`. Old entries miss safely.
> - `generation_settings` on receipts and the manifest is now `dict[str, float | int]` so an integer token cap stays an integer.
>
> `### Not changed`
> - `estimated_cost` remains `None` everywhere, permanently. Ceilings and estimates are different claims and live in different fields.

**`README.md` — new subsection "Cost ceilings and spend gate":**

> A council run has always reported *tokens*. It now also reports *dollars* — as a **ceiling**, never an estimate.
>
> * An **estimate** is a guess. A wrong number inside an audit receipt is worse than no number, which is why `estimated_cost` is `None` and always will be.
> * A **ceiling** is a falsifiable claim: *"this run cost no more than $0.0412, priced against snapshot `sha256:...` dated 2026-09-03."* You can check it against your invoice.
>
> ```bash
> conclave ask "should we migrate?" --mode elite --max-output-tokens 4000 --max-spend-usd 0.40
> ```
>
> Before the first provider call, conclave enumerates the mode's worst-case call plan, prices every call at ceiling rates from a dated snapshot committed to this repo, and **refuses** (exit code `4`) if the total exceeds your cap — naming the reserved total, the cap, and the call count. It refuses just as firmly when it *cannot* bound the run: no `--max-output-tokens` (unbounded output cannot be bounded in dollars), no price snapshot, or a model absent from the snapshot. It never falls back to a similar model's rate.
>
> The prices are a hand-verified, dated file, not a live feed. A model whose published price could not be verified is simply absent — which makes it unpriced, and makes the whole run's ceiling `None` with `manifest.unpriced_models` naming it. That is the all-or-nothing rule: a partial sum reads exactly like a complete one.
>
> | Exit code | Meaning |
> |---|---|
> | `0` | clean run |
> | `1` | no usable answers |
> | `2` | usage/config error |
> | `3` | degraded — it ran, the judge step failed |
> | `4` | **refused — nothing ran, nothing was spent** |

**`docs/PRODUCT_DESIGN_DOCUMENT.md` §4a** — extend the `ModelHarnessManifest` field list with the six new fields, rewrite the "No invented pricing" bullet's second sentence to distinguish ceiling from estimate, and add:

> ### Cost ceilings, never estimates (v1.4)
>
> `manifest.py` has always said `estimated_cost` stays `None` because "a wrong number inside an audit receipt is worse than no number." That stands. What changed is that a *different* claim is now available: not an estimate but a **ceiling** — `cost_ceiling_usd`, computed with exact `Decimal` rates and `ROUND_CEILING` against a dated, content-digested, vendor-cited snapshot committed to the repo, and always accompanied by `price_snapshot_digest` and `priced_as_of` so it is checkable rather than trusted. The two live in separate fields on purpose; conflating them would let a guess inherit a ceiling's credibility.
>
> Three rules keep the ceiling honest. **All-or-nothing:** any model in the run absent from the snapshot, or any receipt that cannot be bounded, leaves the run-level ceiling `None` with `unpriced_models` / `unpriced_receipts` naming why — a partial sum is indistinguishable from a complete one and is the exact failure mode this design prevents. Scope note: `unpriced_models` covers the models that actually ran (`model_ids` plus every receipt's model), not members skipped for a missing key, which made no call and cannot be billed. **Never a substitute rate:** an absent model is unpriced; a similar model's rate is never borrowed, and a stale snapshot warns (`pricing_warnings`, bounded identifiers only) but still prices at the rates it actually records. **Priced last:** `Council._price_manifest` runs after `_ensure_manifest` and after the final receipt append, so it can never miss the synthesis or verdict-repair calls; the snapshot's rate digest joins cache identity so a hit can never serve a ceiling that was never true of those rates.
>
> The pre-flight `--max-spend-usd` gate is the same arithmetic run forward. `Council.plan_calls` enumerates the worst-case plan — `raw` `N`, `synthesize` `N + C + 2C`, `vote` `N`, `debate` `N*R + C`, `adversarial` `N + C`, `elite` `3N + C + 2C`, with `C` the number of **keyed** synthesizer-chain candidates and the verdict's repair retry always counted — bounds each call's input by UTF-8 bytes (plus the sum of upstream output caps times `max_output_bytes_per_token` for calls that embed a prior model's output), and refuses before the first call. Refusing is the designed outcome for an unbounded plan: inventing a number to get past the gate would defeat the gate.

**`docs/PRODUCT_DESIGN_DOCUMENT.md` §9** — after the H1 paragraph, add:

> H1's budget-matched ablations and H4's quality-per-dollar question now have a real denominator rather than a guess: every run carries a `cost_ceiling_usd` with its snapshot digest and capture date (§4a), so *"is Elite worth `3N + 2` calls?"* is answerable **at** the decision instead of after the invoice.

**`DOCUMENTATION_INDEX.md`** — add `docs/plans/2026-09-03-bounded-cost-receipts.md` beside the DSE-1512 plan entry.

**`config.example.yml`** — append:

```yaml
# Optional hard ceiling on OUTPUT tokens for every call a council makes -- members,
# synthesizer, judge, verdict extraction and its repair retry, and both streaming
# paths (OFF by default, unset). Override per invocation with --max-output-tokens.
# This is also the precondition for --max-spend-usd: a run whose output is
# unbounded cannot have its spend bounded, so the spend gate refuses (exit 4)
# rather than inventing a number.
# max_output_tokens: 4000
```

**Step 2: Verify the docs are ruff-clean**

`ruff format` normalizes Python code blocks inside Markdown, so a mis-formatted ```python block in a doc fails CI just like a source file. Run:

```
$BR '.venv/bin/ruff format --check README.md CHANGELOG.md docs DOCUMENTATION_INDEX.md'
$BR '.venv/bin/ruff check . && .venv/bin/ruff format --check .'
```
Expected: `N files already formatted`. If a block reformats, take ruff's version — 100 columns, double quotes, magic trailing commas.

**Step 3: Run the full suite one last time**

```
$BR '.venv/bin/python -m pytest -q --cov=conclave --cov-report=term-missing --cov-fail-under=75 -p no:cacheprovider 2>&1 | tail -6'
```
Expected: around `902 passed`, coverage at or above 75%.

**Step 4: Commit**

```bash
git add CHANGELOG.md README.md docs DOCUMENTATION_INDEX.md config.example.yml SYSTEM_CONTEXT_DIAGRAM.md
git commit -m "docs: cost ceilings, spend gate, exit code 4, PDD 4a/9 (DSE-1514)"
```

---

# Round 5 — Adversarial review (Opus, `qa-codebase-guardian`)

Read-only. Hunt specifically for:

1. **Any path emitting a number that is not a true ceiling.** Grep every assignment to `cost_ceiling_usd` and prove each one came from `reserve_cost` or `reported_usage_cost`.
2. **Partial sums.** Is there any branch where `cost_ceiling_usd` is non-`None` while `unpriced_models` or `unpriced_receipts` is non-empty? Construct one if you can.
3. **Float leakage.** `grep -rn "float(" src/conclave/pricing.py src/conclave/council.py`; check that `--max-spend-usd` is a `str` option; check that no snapshot rate is a bare JSON number; check that the `float | int` union on `generation_settings` does not coerce the cap.
4. **A gate that under-counts.** Re-derive every formula in the per-mode table against `modes.py` **on the merged code**, not against this document. Specifically: does the adversarial proposer loop ever exceed `N` attempts? Does a debate with `converge_threshold` ever run more than `rounds`? Does `_apply_verdict` ever call `extract_verdict` more than once per chain candidate? Does streaming synthesis ever exceed `C` calls?
5. **Ordering.** Is there any receipt appended *after* `_price_manifest` runs — on any of the six modes, the streaming path, or the Elite readiness path?
6. **Cache correctness.** Can a run priced under snapshot A be served a result priced under snapshot B?
7. **Secret safety.** Does any `pricing_warnings` entry contain interpolated text? Does `unpriced_models` ever contain anything but a model id?

# Round 6 — CI + merge (Sonnet, `ci-cd-pr-lifecycle-manager`)

1. Rebase onto `main` after DSE-1512's PR #63 squash-merges. Expected conflicts: `CHANGELOG.md` `[Unreleased]`, the `cache.py` version comment. Re-run the full suite after the rebase.
2. Push each round's branch and open its PR; wait for `Test` (3.11 / 3.12 / 3.13), `ruff`, `pip-audit`, and `Gitleaks` to go green on each.
3. `python3 ~/.claude/scripts/release_control.py classify` on each **complete** diff. Expect `security-specific` (see the header); obtain **one authenticated human receipt per PR** before merging.
4. `python3 ~/.claude/scripts/release_control.py merge --repo DataScience-EngineeringExperts/conclave --pr <n> --head-sha <40-char> --method squash`, in round order: 2, then 3, then 4.
5. Linear DSE-1514 → Done, with all three merge SHAs and the resolved answer to the open design question quoted in the closing comment.
