"""Client timeouts + no blind re-send of read-timed-out paid calls (BacMR/docs#237)."""

import logging
from types import SimpleNamespace

import httpx
import pytest

from bacmr_common.llm import DEFAULT_TIMEOUT_S, LLMClient

SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}


def _settings(**extra) -> SimpleNamespace:
    return SimpleNamespace(
        model_gateway_url="http://gw", model_gateway_key="k", embedding_model="emb", **extra
    )


def _ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/embeddings":
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "ok"}'}}]})


@pytest.fixture
def no_sleep(monkeypatch):
    async def fake_sleep(_s):
        return None

    monkeypatch.setattr("bacmr_common.llm.asyncio.sleep", fake_sleep)


def _recording(seen: list, failures: list | None = None):
    failures = failures if failures is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if failures:
            f = failures.pop(0)
            if isinstance(f, Exception):
                raise f
            return httpx.Response(f, json={"error": "x"})
        return _ok(request)

    return handler


def _read_timeout(request: httpx.Request) -> float:
    return request.extensions["timeout"]["read"]


async def test_default_timeout_is_long_enough_for_slow_writers():
    seen: list = []
    client = LLMClient(_settings(), transport=httpx.MockTransport(_recording(seen)))
    await client.chat_json("content-writer", "s", "u", SCHEMA)
    assert DEFAULT_TIMEOUT_S >= 300
    assert _read_timeout(seen[0]) == DEFAULT_TIMEOUT_S
    assert seen[0].extensions["timeout"]["connect"] == 10.0  # dead gateway still fails fast


async def test_timeout_precedence_call_over_alias_over_client():
    seen: list = []
    client = LLMClient(
        _settings(llm_timeout_s=200, llm_alias_timeouts={"claude-sonnet": 900, "x": 1}),
        transport=httpx.MockTransport(_recording(seen)),
        alias_timeouts={"x": 50},  # keyword wins over settings
    )
    await client.chat_json("other", "s", "u", SCHEMA)
    await client.chat_json("claude-sonnet", "s", "u", SCHEMA)
    await client.chat_json("x", "s", "u", SCHEMA)
    await client.chat_json("claude-sonnet", "s", "u", SCHEMA, timeout_s=30)
    await client.chat_tools("claude-sonnet", [{"role": "user", "content": "u"}], [], timeout_s=40)
    await client.embed(["t"], timeout_s=5)
    assert [_read_timeout(r) for r in seen] == [200, 900, 50, 30, 40, 5]


async def test_constructor_timeout_beats_settings():
    client = LLMClient(_settings(llm_timeout_s=200), timeout_s=45)
    assert client.timeout_for("any") == 45


async def test_chat_read_timeout_is_not_resent_and_is_logged(no_sleep, caplog):
    seen: list = []
    client = LLMClient(
        _settings(),
        transport=httpx.MockTransport(_recording(seen, [httpx.ReadTimeout("slow")])),
    )
    with (
        caplog.at_level(logging.WARNING, logger="bacmr_common.llm"),
        pytest.raises(httpx.ReadTimeout),
    ):
        await client.chat_json("claude-sonnet", "s", "u", SCHEMA)
    assert len(seen) == 1  # one billed call, not five
    rec = next(r for r in caplog.records if getattr(r, "event", None) == "llm.read_timeout")
    assert rec.fields["retry"] is False and rec.fields["model"] == "claude-sonnet"


async def test_direct_post_path_used_by_content_worker_is_covered(no_sleep):
    seen: list = []
    client = LLMClient(
        _settings(),
        transport=httpx.MockTransport(_recording(seen, [httpx.ReadTimeout("slow")])),
    )
    with pytest.raises(httpx.ReadTimeout):
        await client._post_with_backoff(
            "/v1/chat/completions", {"model": "content-writer", "messages": []}, max_attempts=5
        )
    assert len(seen) == 1


async def test_read_timeout_retry_opt_in_retries_at_most_that_many(no_sleep, caplog):
    seen: list = []
    fails = [httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")]
    client = LLMClient(_settings(), transport=httpx.MockTransport(_recording(seen, fails)))
    with (
        caplog.at_level(logging.WARNING, logger="bacmr_common.llm"),
        pytest.raises(httpx.ReadTimeout),
    ):
        await client._post_with_backoff(
            "/v1/chat/completions", {"model": "m"}, read_timeout_retries=1
        )
    assert len(seen) == 2
    flags = [
        r.fields["retry"] for r in caplog.records if getattr(r, "event", "") == "llm.read_timeout"
    ]
    assert flags == [True, False]


async def test_read_timeout_retry_once_then_success(no_sleep):
    seen: list = []
    client = LLMClient(
        _settings(),
        transport=httpx.MockTransport(_recording(seen, [httpx.ReadTimeout("slow")])),
    )
    resp = await client._post_with_backoff(
        "/v1/chat/completions", {"model": "m"}, read_timeout_retries=1
    )
    assert resp.status_code == 200 and len(seen) == 2


async def test_embeddings_keep_retrying_read_timeouts(no_sleep):
    seen: list = []
    client = LLMClient(
        _settings(),
        transport=httpx.MockTransport(_recording(seen, [httpx.ReadTimeout("slow")] * 2)),
    )
    out = await client.embed(["t"])
    assert out == [[1.0, 0.0]] and len(seen) == 3


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("slow dial"),
        httpx.PoolTimeout("pool"),
        429,
        503,
    ],
)
async def test_connect_errors_and_transient_statuses_keep_the_ladder(no_sleep, failure):
    seen: list = []
    client = LLMClient(
        _settings(), transport=httpx.MockTransport(_recording(seen, [failure, failure]))
    )
    out = await client.chat_json("claude-sonnet", "s", "u", SCHEMA)
    assert out == {"answer": "ok"} and len(seen) == 3


async def test_connect_errors_still_exhaust_the_full_ladder(no_sleep):
    seen: list = []
    client = LLMClient(
        _settings(),
        transport=httpx.MockTransport(_recording(seen, [httpx.ConnectError("x")] * 9)),
    )
    with pytest.raises(httpx.ConnectError):
        await client.chat_json("m", "s", "u", SCHEMA)
    assert len(seen) == 5


async def test_subclass_with_old_post_signature_still_works_without_timeout():
    """knowledge's LabLLM and tutor test doubles override `_post_with_backoff`
    with the pre-0.9 signature; an unset timeout must not pass the new kwarg."""
    seen: list = []

    class Old(LLMClient):
        async def _post_with_backoff(self, path, json, max_attempts=5, backoff_schedule=None):
            seen.append(path)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"answer": "ok"}'}}],
                    "data": [{"index": 0, "embedding": [1.0]}],
                },
            )

    client = Old(_settings())
    assert await client.chat_json("m", "s", "u", SCHEMA) == {"answer": "ok"}
    await client.chat_tools("m", [{"role": "user", "content": "u"}], [])
    assert await client.embed(["t"]) == [[1.0]]
    assert seen == ["/v1/chat/completions", "/v1/chat/completions", "/v1/embeddings"]
