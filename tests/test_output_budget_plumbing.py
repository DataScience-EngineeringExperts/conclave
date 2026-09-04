"""Output-token ceilings are optional, provider-native, and end-to-end."""

from __future__ import annotations

from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter
from conclave.models import ModelAnswer
from conclave.providers import call_model, call_model_stream

MESSAGES = [{"role": "user", "content": "hi"}]


def test_adapter_caps_override_native_defaults_without_changing_default_bodies():
    openai = OpenAICompatAdapter("openai", "https://example.test/v1/chat", ("OPENAI_API_KEY",))
    anthropic = AnthropicAdapter()
    gemini = GeminiAdapter()

    _, _, openai_default = openai.build_request("openai/gpt-4.1", MESSAGES, 0.2, 30, "key")
    _, _, anthropic_default = anthropic.build_request(
        "anthropic/claude-sonnet-4-20250514", MESSAGES, 0.2, 30, "key"
    )
    _, _, gemini_default = gemini.build_request("gemini/gemini-2.5-pro", MESSAGES, 0.2, 30, "key")

    _, _, openai_capped = openai.build_request(
        "openai/gpt-4.1", MESSAGES, 0.2, 30, "key", max_output_tokens=321
    )
    _, _, anthropic_capped = anthropic.build_request(
        "anthropic/claude-sonnet-4-20250514",
        MESSAGES,
        0.2,
        30,
        "key",
        max_output_tokens=321,
    )
    _, _, gemini_capped = gemini.build_request(
        "gemini/gemini-2.5-pro", MESSAGES, 0.2, 30, "key", max_output_tokens=321
    )

    assert "max_tokens" not in openai_default
    assert anthropic_default["max_tokens"] == anthropic.max_tokens
    assert gemini_default["generationConfig"]["maxOutputTokens"] == gemini.max_output_tokens
    assert openai_capped["max_tokens"] == 321
    assert anthropic_capped["max_tokens"] == 321
    assert gemini_capped["generationConfig"]["maxOutputTokens"] == 321


def test_stream_requests_receive_the_same_optional_cap():
    adapters_and_models = (
        (
            OpenAICompatAdapter("openai", "https://example.test/v1/chat", ("OPENAI_API_KEY",)),
            "openai/gpt-4.1",
            lambda body: body["max_tokens"],
        ),
        (AnthropicAdapter(), "anthropic/claude-sonnet-4-20250514", lambda body: body["max_tokens"]),
        (
            GeminiAdapter(),
            "gemini/gemini-2.5-pro",
            lambda body: body["generationConfig"]["maxOutputTokens"],
        ),
    )
    for adapter, model_id, extract in adapters_and_models:
        _, _, body = adapter.stream_request(
            model_id, MESSAGES, 0.2, 30, "key", max_output_tokens=654
        )
        assert extract(body) == 654


async def test_call_model_threads_max_output_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")
    captured = {}

    async def fake_post_json(url, headers, json_body, timeout):
        captured["body"] = json_body
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("conclave.transport.post_json", fake_post_json)
    answer = await call_model(
        "openai",
        "openai/gpt-4.1",
        MESSAGES,
        max_output_tokens=777,
    )
    assert answer.ok
    assert captured["body"]["max_tokens"] == 777


async def test_call_model_stream_threads_max_output_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")
    captured = {}

    async def fake_stream_sse(url, headers, json_body, timeout):
        captured["body"] = json_body
        yield "", '{"choices":[{"delta":{"content":"ok"}}]}'
        yield "", "[DONE]"

    monkeypatch.setattr("conclave.transport.stream_sse", fake_stream_sse)
    items = [
        item
        async for item in call_model_stream(
            "openai",
            "openai/gpt-4.1",
            MESSAGES,
            max_output_tokens=888,
        )
    ]
    assert isinstance(items[-1], ModelAnswer)
    assert items[-1].ok
    assert captured["body"]["max_tokens"] == 888


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
    # Two members: verdict extraction's N<2-responder gate (CAC-05, DD-1) skips
    # the extraction call entirely for a single responder, which would hide the
    # cap from the verdict-extraction/repair sites this test means to cover.
    council = Council(models=["grok", "gemini"], synthesizer="claude", max_output_tokens=777)
    await council.ask("q")

    # member fan-out (x2) + synthesis + verdict extraction + verdict repair
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
