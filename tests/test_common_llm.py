import asyncio
import json

import httpx
import pytest
from bacmr_common.llm import LLMClient, LLMOutputError, normalize_gateway_base_url

SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": False,
}


class _GatewaySettings:
    model_gateway_url = "http://gw"
    model_gateway_key = "k"
    embedding_model = "embeddings"


def _settings() -> _GatewaySettings:
    return _GatewaySettings()


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


async def test_chat_json_parses_and_validates():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer": "ok"}'))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = await client.chat_json("internal-batch", "sys", "user", SCHEMA)
    assert out == {"answer": "ok"}
    assert calls[0]["model"] == "internal-batch"
    assert calls[0]["response_format"] == {"type": "json_object"}


async def test_chat_json_retries_then_raises():
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(200, json=_completion("not json at all"))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMOutputError):
        await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    assert n == 3  # blind try + two corrective retries


async def test_chat_json_retries_on_schema_violation():
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        # valid JSON, wrong shape -> corrective retries, then raise
        return httpx.Response(200, json=_completion('{"unexpected": 1}'))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMOutputError):
        await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    assert n == 3


async def test_chat_json_recovers_on_second_attempt():
    replies = iter(["oops", '{"answer": "second"}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(next(replies)))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    assert await client.chat_json("internal-batch", "sys", "u", SCHEMA) == {"answer": "second"}


async def test_chat_json_sends_images_as_data_urls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer": "img"}'))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    await client.chat_json("internal-vision", "sys", "look", SCHEMA, images=[b"\x89PNG123"])
    parts = seen["messages"][-1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_chat_json_sends_auth_header_and_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_completion('{"answer": "ok"}'))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    await client.chat_json("internal-batch", "sys", "user", SCHEMA)
    assert seen["url"] == "http://gw/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    await client.close()


async def test_chat_json_retry_feeds_validation_error_back():
    # observed live: blind retries repeat the same shape mistake — the retry
    # must show the model its rejected reply and the validator's complaint
    bodies: list[dict] = []
    replies = iter(['{"wrong": 1}', '{"answer": "fixed"}'])

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(next(replies)))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = await client.chat_json("internal-batch", "sys", "user", SCHEMA)
    await client.close()
    assert out == {"answer": "fixed"}
    first, second = bodies
    assert len(first["messages"]) == 2
    msgs = second["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["content"] == '{"wrong": 1}'
    assert msgs[-1]["role"] == "user" and "rejected by the JSON validator" in msgs[-1]["content"]


async def test_chat_json_null_content_is_retried_not_fatal():
    # safety-filter refusals return message.content = null — observed live
    # killing the figures stage with an uncaught TypeError
    replies = iter([None, '{"answer": "after-null"}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(next(replies)))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    await client.close()
    assert out == {"answer": "after-null"}


async def test_chat_json_retries_429_with_backoff(monkeypatch):
    # 429 twice, then success — all three attempts happen, sleeps are the
    # backoff schedule (patched to record instead of wait)
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        if n < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_completion('{"answer": "ok"}'))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    await client.close()
    assert out == {"answer": "ok"}
    assert n == 3
    assert len(sleeps) == 2 and sleeps[0] < sleeps[1]  # exponential


async def test_backoff_gives_up_after_max_attempts(monkeypatch):
    async def fake_sleep(s):
        pass

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(503, json={"error": "down"})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    await client.close()
    assert n == 5  # MAX_HTTP_ATTEMPTS


def _sse(*chunks: str, done: bool = True) -> bytes:
    body = "".join(f"data: {c}\n\n" for c in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


async def test_chat_stream_yields_content_deltas_in_order():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
                json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = [d async for d in client.chat_stream("student-chat", [{"role": "user", "content": "hi"}])]
    await client.close()
    assert out == ["Hel", "lo"]


async def test_chat_stream_skips_done_and_empty_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
                json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
                json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = [d async for d in client.chat_stream("student-chat", [{"role": "user", "content": "hi"}])]
    await client.close()
    assert out == ["ok"]


async def test_chat_stream_sends_stream_true_and_model():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_sse(), headers={"content-type": "text/event-stream"})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    [d async for d in client.chat_stream("student-chat", [{"role": "user", "content": "hi"}])]
    await client.close()
    assert seen["model"] == "student-chat"
    assert seen["stream"] is True


async def test_chat_stream_raises_on_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        [d async for d in client.chat_stream("student-chat", [{"role": "user", "content": "hi"}])]
    await client.close()


class _HangingStream:
    async def __aenter__(self):
        await asyncio.sleep(10)

    async def __aexit__(self, *exc):
        return False


async def test_chat_stream_times_out_via_asyncio_timeout(monkeypatch):
    client = LLMClient(_settings())
    monkeypatch.setattr(client._http, "stream", lambda *a, **k: _HangingStream())

    with pytest.raises(TimeoutError):
        async for _ in client.chat_stream(
            "student-chat", [{"role": "user", "content": "hi"}], timeout_s=0.01
        ):
            pass
    await client.close()


async def test_backoff_schedule_sleeps_per_schedule_then_raises(monkeypatch):
    # a custom schedule replaces the default exponential ladder: with an
    # always-429 transport, the client sleeps exactly the schedule's values
    # (not the default 1,2,4,8) then gives up once the schedule is exhausted
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("bacmr_common.llm.random.random", lambda: 0.0)  # strip jitter
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(429, json={"error": "rate limited"})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_json("internal-batch", "sys", "u", SCHEMA, backoff_schedule=(1.0, 2.0))
    await client.close()
    assert n == 3  # schedule (2 entries) + 1 final failing attempt
    assert sleeps == [1.0, 2.0]


async def test_backoff_schedule_none_keeps_default_ladder(monkeypatch):
    # byte-identical default: omitting backoff_schedule must behave exactly
    # like before this param existed — max_attempts stays MAX_HTTP_ATTEMPTS
    # and delays follow the 1,2,4,8s exponential ladder
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("bacmr_common.llm.random.random", lambda: 0.0)  # strip jitter
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(503, json={"error": "down"})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    await client.close()
    assert n == 5  # MAX_HTTP_ATTEMPTS, unchanged
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


async def test_embed_accepts_backoff_schedule(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("bacmr_common.llm.random.random", lambda: 0.0)
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        if n < 2:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = await client.embed(["hi"], dimensions=2, backoff_schedule=(3.0,))
    await client.close()
    assert out == [[1.0, 0.0]]
    assert sleeps == [3.0]


async def test_non_retryable_status_raises_immediately():
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(401, json={"error": "bad key"})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    await client.close()
    assert n == 1


async def test_usage_identity_is_isolated_between_concurrent_requests():
    from bacmr_common.usage import usage_context

    bodies = []

    async def handler(request):
        await asyncio.sleep(0)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer":"ok"}'))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))

    async def run(student):
        with usage_context(
            consumption="user", actor=student, operation="tutor", request_id=student + "-turn"
        ):
            await client.chat_json("internal-batch", "s", "q", SCHEMA)

    await asyncio.gather(run("alice"), run("bob"))
    await client.chat_json("ingest-text", "s", "q", SCHEMA)
    assert {b["user"] for b in bodies[:2]} == {"alice", "bob"}
    for b in bodies[:2]:
        assert "consumption:user" in b["metadata"]["tags"]
        assert b["metadata"]["request_id"] == b["user"] + "-turn"
    assert bodies[2]["user"].startswith("system:")
    assert "consumption:system" in bodies[2]["metadata"]["tags"]
    await client.close()


async def test_chat_json_records_provider_usage_into_the_active_tally():
    from bacmr_common.usage import usage_tally

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **_completion('{"answer": "ok"}'),
                "usage": {"prompt_tokens": 120, "completion_tokens": 8},
            },
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with usage_tally() as tally:
        await client.chat_json("internal-batch", "sys", "user", SCHEMA)
    await client.close()
    assert (tally.prompt_tokens, tally.completion_tokens) == (120, 8)
    assert tally.unmeasured_calls == 0


async def test_every_http_attempt_is_counted_not_just_the_successful_one():
    """A corrective retry is a second call the provider really served. Counting
    only the last one is how a budget silently undercounts by a multiple."""
    from bacmr_common.usage import usage_tally

    replies = iter(["oops", '{"answer": "second"}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **_completion(next(replies)),
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with usage_tally() as tally:
        await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    await client.close()
    assert tally.measured_calls == 2
    assert tally.measured_tokens == 110


async def test_reply_without_usage_is_recorded_as_an_estimate_never_as_zero():
    from bacmr_common.usage import usage_tally

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion('{"answer": "ok"}'))

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with usage_tally() as tally:
        await client.chat_json("internal-batch", "system prompt here", "user text", SCHEMA)
    await client.close()
    assert tally.measured_calls == 0
    assert tally.unmeasured_calls == 1
    assert tally.estimated_tokens > 0  # a labelled guess, not a silent zero


async def test_embeddings_usage_lands_in_the_tally_too():
    from bacmr_common.usage import usage_tally

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 42, "total_tokens": 42},
            },
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with usage_tally() as tally:
        await client.embed(["hello"])
    await client.close()
    assert tally.measured_tokens == 42


async def test_stream_records_usage_from_the_trailing_usage_only_frame():
    from bacmr_common.usage import usage_tally

    def handler(request):
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":900,"completion_tokens":60}}\n\n'
            "data: [DONE]\n\n",
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with usage_tally() as tally:
        assert [x async for x in client.chat_stream("student-chat", [])] == ["hello"]
    await client.close()
    assert (tally.prompt_tokens, tally.completion_tokens) == (900, 60)
    assert tally.unmeasured_calls == 0


async def test_stream_without_a_usage_frame_still_records_a_labelled_estimate():
    from bacmr_common.usage import usage_tally

    def handler(request):
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with usage_tally() as tally:
        assert [
            x
            async for x in client.chat_stream(
                "student-chat", [{"role": "user", "content": "a" * 400}]
            )
        ] == ["hello"]
    await client.close()
    assert tally.unmeasured_calls == 1
    assert tally.estimated_tokens >= 100  # the prompt counts, not just the reply


async def test_abandoned_stream_still_records_what_it_burned():
    """A client disconnect mid-reply does not refund the prompt. Recording
    nothing here would be exactly the fabricated zero AGENTS.md forbids."""
    from bacmr_common.usage import usage_tally

    def handler(request):
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":900,"completion_tokens":60}}\n\n'
            "data: [DONE]\n\n",
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    with usage_tally() as tally:
        stream = client.chat_stream("student-chat", [{"role": "user", "content": "q" * 80}])
        async for _first in stream:
            break  # walk away before the usage frame ever arrives
        await stream.aclose()
    await client.close()
    # No provider number reached us, so: unmeasured + a labelled estimate.
    assert tally.unmeasured_calls == 1
    assert tally.estimated_tokens > 0


async def test_stream_requests_usage_and_handles_usage_only_final_frame():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1}}\n\n'
            "data: [DONE]\n\n",
        )

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    assert [x async for x in client.chat_stream("student-chat", [])] == ["hello"]
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert "metadata" in seen[0]
    await client.close()


async def test_retry_attempts_and_embeddings_have_accounting_metadata(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", no_sleep)
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    await client.embed(["text"])
    assert [b["metadata"]["attempt"] for b in bodies] == [1, 2]
    assert bodies[0]["metadata"]["call_id"] == bodies[1]["metadata"]["call_id"]
    assert "consumption:system" in bodies[1]["metadata"]["tags"]
    await client.close()


async def test_service_tag_prefers_settings_usage_service_over_module_fallback():
    # PR #43 review Minor-2: the module-package fallback produced
    # `service:test_common_llm`-style tags; an explicit `usage_service` on the
    # settings object (now set on both real Settings classes) must win.
    class _NamedSettings(_GatewaySettings):
        usage_service = "knowledge"

    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer": "ok"}'))

    client = LLMClient(_NamedSettings(), transport=httpx.MockTransport(handler))
    await client.chat_json("internal-batch", "sys", "u", SCHEMA)
    await client.close()
    assert "service:knowledge" in bodies[0]["metadata"]["tags"]
    assert bodies[0]["metadata"]["service"] == "knowledge"


def test_accounting_fields_use_litellm_persisted_metadata_container():
    from bacmr_common.usage import gateway_body, usage_context

    with usage_context(
        consumption="system",
        actor="system:ingestion",
        job_id="job-1",
        document_id="book-1",
        stage="figures",
    ):
        body = gateway_body({}, service="knowledge", call_id="call-1", attempt=2)
    saved = body["metadata"]["spend_logs_metadata"]
    assert saved["job_id"] == "job-1"
    assert saved["document_id"] == "book-1"
    assert saved["stage"] == "figures"
    assert saved["call_id"] == "call-1"
    assert saved["attempt"] == 2


# --- model-gateway base URL normalization -----------------------------------
#
# Found 2026-10-04 on the `bacmr-v2-next` stack: its rendered env carried
# MODEL_GATEWAY_URL=http://litellm:4000/v1 while all five live containers
# carried http://litellm:4000. Every request method here spells the path in
# full (`/v1/chat/completions`), and httpx appends a relative path to the base
# URL's path rather than replacing it, so the `/v1` form produced
# `http://litellm:4000/v1/v1/chat/completions`. LiteLLM answers that 404, and
# 404 is not in the transient set, so the backoff ladder does not retry -- the
# first model call of the first ingestion stage simply fails.


def test_normalize_strips_a_trailing_v1():
    assert normalize_gateway_base_url("http://litellm:4000/v1") == "http://litellm:4000"
    assert normalize_gateway_base_url("http://litellm:4000/v1/") == "http://litellm:4000"


def test_normalize_leaves_a_bare_origin_alone():
    assert normalize_gateway_base_url("http://litellm:4000") == "http://litellm:4000"
    assert normalize_gateway_base_url("http://litellm:4000/") == "http://litellm:4000"


def test_normalize_does_not_eat_a_v1_that_is_part_of_a_longer_path():
    # A reverse-proxy prefix must survive; only a trailing `/v1` SEGMENT goes.
    assert normalize_gateway_base_url("https://x/gw/v1") == "https://x/gw"
    assert normalize_gateway_base_url("https://x/v1beta") == "https://x/v1beta"


@pytest.mark.parametrize("configured", ["http://litellm:4000", "http://litellm:4000/v1"])
def test_both_spellings_reach_the_same_completions_url(configured):
    """The property that actually matters: one request URL, either way in."""

    class _S:
        model_gateway_url = configured
        model_gateway_key = "sk-test"
        usage_service = "test"

    client = LLMClient(_S())
    # `build_request` applies the same base-URL merge a real send would, with no
    # socket involved -- the double-/v1 bug was purely in that merge.
    assert (
        str(client._http.build_request("POST", "/v1/chat/completions").url)
        == "http://litellm:4000/v1/chat/completions"
    )
    assert (
        str(client._http.build_request("POST", "/v1/embeddings").url)
        == "http://litellm:4000/v1/embeddings"
    )
