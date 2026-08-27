import asyncio
import json

import httpx
import pytest
from bacmr_common.llm import LLMClient, LLMOutputError

SCHEMA = {"type": "object", "required": ["answer"],
          "properties": {"answer": {"type": "string"}}, "additionalProperties": False}


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
        return httpx.Response(200, content=_sse(
            json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
            json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
        ), headers={"content-type": "text/event-stream"})

    client = LLMClient(_settings(), transport=httpx.MockTransport(handler))
    out = [d async for d in client.chat_stream("student-chat", [{"role": "user", "content": "hi"}])]
    await client.close()
    assert out == ["Hel", "lo"]


async def test_chat_stream_skips_done_and_empty_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(
            json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
            json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ), headers={"content-type": "text/event-stream"})

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
                "student-chat", [{"role": "user", "content": "hi"}], timeout_s=0.01):
            pass
    await client.close()


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
