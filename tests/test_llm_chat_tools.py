"""LLMClient.chat_tools: OpenAI-format tool calling (BacMR/py-common#16)."""

import json

import httpx
import pytest

from bacmr_common.accounting import CallTags
from bacmr_common.llm import LLMClient, ToolReply
from bacmr_common.usage import usage_tally

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search the textbooks",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
MESSAGES = [{"role": "user", "content": "find the lesson on vectors"}]


class _GatewaySettings:
    model_gateway_url = "http://gw"
    model_gateway_key = "k"
    embedding_model = "embeddings"


def _call(call_id: str, name: str, arguments, **extra) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        **extra,
    }


def _reply(content=None, tool_calls=None, finish_reason="tool_calls", usage=None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    out: dict = {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
    if usage is not None:
        out["usage"] = usage
    return out


def _client(handler) -> LLMClient:
    return LLMClient(_GatewaySettings(), transport=httpx.MockTransport(handler))


async def test_single_tool_call_is_parsed_and_body_is_openai_shaped():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(
            (request.url.path, request.headers["authorization"], json.loads(request.content))
        )
        return httpx.Response(
            200, json=_reply(tool_calls=[_call("c1", "search", '{"query": "vecteurs"}')])
        )

    client = _client(handler)
    out = await client.chat_tools("tutor-agent", MESSAGES, TOOLS)
    await client.close()

    assert isinstance(out, ToolReply)
    assert out.content == ""  # null content coerced
    assert out.finish_reason == "tool_calls"
    assert len(out.tool_calls) == 1
    call = out.tool_calls[0]
    assert (call.id, call.name, call.arguments, call.arguments_error) == (
        "c1",
        "search",
        {"query": "vecteurs"},
        None,
    )
    path, auth, body = bodies[0]
    assert path == "/v1/chat/completions"
    assert auth == "Bearer k"
    assert body["model"] == "tutor-agent"
    assert body["messages"] == MESSAGES
    assert body["tools"] == TOOLS
    assert body["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in body  # provider default; some reject the key
    assert "max_tokens" not in body
    assert "response_format" not in body


async def test_parallel_tool_calls_keep_order():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_reply(
                tool_calls=[
                    _call("a", "search", '{"query": "x"}'),
                    _call("b", "show_figure", '{"figure_id": "f-1"}'),
                    _call("c", "search", '{"query": "y"}'),
                ]
            ),
        )

    client = _client(handler)
    out = await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    assert [(c.id, c.name, c.arguments) for c in out.tool_calls] == [
        ("a", "search", {"query": "x"}),
        ("b", "show_figure", {"figure_id": "f-1"}),
        ("c", "search", {"query": "y"}),
    ]


async def test_no_tool_reply_returns_content_and_empty_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply(content="DONE", finish_reason="stop"))

    client = _client(handler)
    out = await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    assert out.content == "DONE"
    assert out.tool_calls == []
    assert out.finish_reason == "stop"


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ('{"query": "vec', "not valid JSON"),  # truncated by max_tokens
        ('["vecteurs"]', "JSON object"),
        ("42", "JSON object"),
    ],
)
async def test_malformed_arguments_do_not_raise(raw, fragment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_reply(
                tool_calls=[_call("bad", "search", raw), _call("ok", "search", '{"query": "z"}')]
            ),
        )

    client = _client(handler)
    out = await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    bad, ok = out.tool_calls
    assert bad.arguments == {}
    assert bad.raw_arguments == raw
    assert fragment in (bad.arguments_error or "")
    assert ok.arguments == {"query": "z"} and ok.arguments_error is None


async def test_empty_and_object_arguments_are_accepted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_reply(
                tool_calls=[_call("e", "list_chapters", ""), _call("o", "search", {"query": "q"})]
            ),
        )

    client = _client(handler)
    out = await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    empty, obj = out.tool_calls
    assert (empty.arguments, empty.arguments_error) == ({}, None)
    assert (obj.arguments, obj.arguments_error) == ({"query": "q"}, None)


async def test_tool_choice_max_tokens_parallel_and_tags_reach_the_body():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(content="ok", finish_reason="stop"))

    client = _client(handler)
    tags = CallTags("tutor-api", "chat", "req-1", plan_tier="free", user_id="u-1")
    await client.chat_tools(
        "m",
        MESSAGES,
        TOOLS,
        tool_choice={"name": "search"},
        max_tokens=256,
        parallel_tool_calls=False,
        tags=tags,
    )
    await client.chat_tools("m", MESSAGES, TOOLS, tool_choice="required")
    await client.close()
    first, second = bodies
    assert first["tool_choice"] == {"type": "function", "function": {"name": "search"}}
    assert first["max_tokens"] == 256
    assert first["parallel_tool_calls"] is False
    assert set(tags.apply_to_body({})["metadata"]["tags"]) <= set(first["metadata"]["tags"])
    assert second["tool_choice"] == "required"


async def test_empty_tools_omits_tools_and_tool_choice():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(content="answer", finish_reason="stop"))

    client = _client(handler)
    await client.chat_tools("m", MESSAGES, [])
    await client.close()
    assert "tools" not in bodies[0] and "tool_choice" not in bodies[0]


async def test_bad_tool_choice_raises_before_any_call():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    client = _client(handler)
    with pytest.raises(ValueError):
        await client.chat_tools("m", MESSAGES, TOOLS, tool_choice="search")
    await client.close()


async def test_assistant_message_echoes_provider_extras():
    call = _call(
        "c1", "search", '{"query": "x"}', provider_specific_fields={"thought_signature": "sig"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply(tool_calls=[call]))

    client = _client(handler)
    out = await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    echoed = out.assistant_message()
    assert echoed["role"] == "assistant"
    assert echoed["content"] == ""  # null dropped, then defaulted
    assert echoed["tool_calls"][0]["provider_specific_fields"] == {"thought_signature": "sig"}


async def test_usage_lands_in_the_active_tally_like_chat_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_reply(
                tool_calls=[_call("c1", "search", '{"query": "x"}')],
                usage={"prompt_tokens": 300, "completion_tokens": 20},
            ),
        )

    client = _client(handler)
    with usage_tally() as tally:
        out = await client.chat_tools("m", MESSAGES, TOOLS)
        await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    assert out.usage == {"prompt_tokens": 300, "completion_tokens": 20}
    assert (tally.prompt_tokens, tally.completion_tokens) == (600, 40)
    assert tally.measured_calls == 2 and tally.unmeasured_calls == 0


async def test_tool_only_reply_without_usage_is_estimated_from_the_arguments():
    args = json.dumps({"query": "q" * 400})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply(tool_calls=[_call("c1", "search", args)]))

    client = _client(handler)
    with usage_tally() as tally:
        await client.chat_tools("m", [{"role": "user", "content": "x"}], TOOLS)
    await client.close()
    assert tally.unmeasured_calls == 1
    assert tally.estimated_tokens >= 100  # the 400-char arguments count, not just "x"


async def test_retries_429_and_5xx_with_the_chat_json_ladder(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)
    statuses = iter([429, 503])
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        status = next(statuses, 200)
        if status != 200:
            return httpx.Response(status, json={"error": "transient"})
        return httpx.Response(200, json=_reply(content="DONE", finish_reason="stop"))

    client = _client(handler)
    with usage_tally() as tally:
        out = await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    assert out.content == "DONE"
    assert n == 3
    assert len(sleeps) == 2 and sleeps[0] < sleeps[1]  # same exponential ladder
    assert tally.measured_calls + tally.unmeasured_calls == 1  # only the served call


async def test_gives_up_after_max_attempts_and_honours_a_smaller_cap(monkeypatch):
    async def fake_sleep(s):
        pass

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(502, json={"error": "down"})

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_tools("m", MESSAGES, TOOLS)
    assert n == 5  # MAX_HTTP_ATTEMPTS, as chat_json
    n = 0
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_tools("m", MESSAGES, TOOLS, max_attempts=2)
    assert n == 2
    await client.close()


async def test_non_retryable_status_raises_immediately():
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(400, json={"error": "bad tools schema"})

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_tools("m", MESSAGES, TOOLS)
    await client.close()
    assert n == 1
