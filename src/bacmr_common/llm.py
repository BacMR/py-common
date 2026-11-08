"""Thin OpenAI-compatible client for the internal model gateway.

Owns a single httpx.AsyncClient so every stage shares connection pooling.
`chat_json` forces JSON output, validates it against a caller-supplied JSON Schema,
and retries once before giving up — models fail this way often enough to be worth
one cheap retry, but not worth an unbounded loop.

`LLMClient.__init__` takes a duck-typed settings object rather than a concrete
`Settings` class from any one service — anything exposing `.model_gateway_url`
and `.model_gateway_key` (and `.embedding_model` for `embed`) works. This lets
both the knowledge service and the tutor service share this client without
either importing the other's settings module.
"""

import asyncio
import base64
import json
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from jsonschema import Draft202012Validator, ValidationError

from bacmr_common.accounting import CallTags
from bacmr_common.usage import estimate_tokens, gateway_body, new_call_id, record_usage

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_HTTP_ATTEMPTS = 5
_BACKOFF_BASE = 1.0  # 1, 2, 4, 8 seconds (+ up to 25% jitter)

# Client timeouts (BacMR/py-common#23, BacMR/docs#237). The read timeout
# bounds how long a NON-streamed completion may take to answer, and LiteLLM
# sends nothing until the model is done, so it is effectively the whole
# generation. It used to be a hard 120 s: Sonnet lesson calls run 127-156 s,
# timed out client-side, and the transport retry re-sent them, every attempt
# billed (5x in the 2026-11-08 bake-off). 600 s fits the slowest measured
# writer with margin; anything user-facing should pass a shorter one.
DEFAULT_TIMEOUT_S = 600.0
CONNECT_TIMEOUT_S = 10.0
CHAT_PATH = "/v1/chat/completions"

_log = logging.getLogger("bacmr_common.llm")


def _timeout_kw(timeout_s: float | None) -> dict:
    """`timeout_s` for `_post_with_backoff` only when set: subclasses and test
    doubles in dependent repos override `_post_with_backoff` with the older
    signature, and an unset timeout must not break them."""
    return {} if timeout_s is None else {"timeout_s": timeout_s}


def _timeout(seconds: float) -> httpx.Timeout:
    """`seconds` for read/write/pool; connect stays short so a dead gateway
    still fails (and retries) fast."""
    return httpx.Timeout(seconds, connect=min(CONNECT_TIMEOUT_S, seconds))


def _body_chars(body: dict) -> int:
    """Characters this request sends the model — the input half of the
    last-resort estimate used only when a reply carries no `usage` block.
    Base64 image parts are deliberately NOT counted: an image's token cost is
    a provider-specific tiling function of its pixels, and its base64 length
    would inflate the guess by orders of magnitude. A vision call without
    provider usage is therefore an under-estimate, which is exactly why the
    tally records it as an ESTIMATE from an UNMEASURED call rather than a
    number anyone should trust."""
    chars = 0
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(
                len(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    payload_input = body.get("input")
    if isinstance(payload_input, str):
        chars += len(payload_input)
    elif isinstance(payload_input, list):
        chars += sum(len(t) for t in payload_input if isinstance(t, str))
    return chars


def _reply_chars(payload: dict) -> int:
    """Characters the model generated: the text content plus, for a
    tool-calling reply, each call's name and raw `arguments` (a reply that is
    only tool calls has null content but still burned completion tokens)."""
    if not isinstance(payload, dict):
        return 0
    chars = 0
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        chars += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            fn = (call.get("function") or {}) if isinstance(call, dict) else {}
            args = fn.get("arguments")
            chars += len(fn.get("name") or "")
            chars += len(args) if isinstance(args, str) else len(json.dumps(args or {}))
    return chars


def _record_call_usage(request_body: dict, resp: httpx.Response) -> None:
    """Fold ONE completed gateway call into the caller's `usage_tally` (a
    no-op when there is none). Called per HTTP attempt, not per logical call:
    a corrective JSON retry or a post-429 retry is a second call the provider
    really did serve, and dropping it is how "200k tokens" ends up meaning
    millions."""
    try:
        payload = resp.json()
    except ValueError:  # non-JSON body — nothing to measure, nothing to fake
        payload = {}
    usage = payload.get("usage") if isinstance(payload, dict) else None
    record_usage(usage, estimate=estimate_tokens(_body_chars(request_body) + _reply_chars(payload)))


class LLMOutputError(RuntimeError):
    """The model returned unparseable or schema-invalid JSON twice."""


@dataclass(frozen=True)
class ToolCall:
    """One function call the model asked for.

    `arguments` is always a dict. When the model's `arguments` string is not a
    JSON object (truncated by `max_tokens`, a stray trailing comma, a bare
    list), `arguments` is `{}`, `arguments_error` says why and `raw_arguments`
    keeps the original text, so the agent loop can feed the error back as the
    tool result instead of the whole turn dying on a JSONDecodeError.
    """

    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    arguments_error: str | None = None


@dataclass(frozen=True)
class ToolReply:
    """A parsed tool-enabled completion.

    `message` is the assistant message exactly as the provider returned it;
    `assistant_message()` is the form to append to `messages` before the tool
    results (provider extras such as Gemini thought signatures ride on the
    tool-call entries and the next call fails without them).
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None
    finish_reason: str | None = None
    message: dict = field(default_factory=dict)

    def assistant_message(self) -> dict:
        out = {k: v for k, v in self.message.items() if v is not None}
        out["role"] = "assistant"
        out.setdefault("content", "")
        return out


ToolChoice = Literal["auto", "required", "none"] | dict


def _tool_choice_body(tool_choice: ToolChoice) -> str | dict:
    """`"auto"`/`"required"`/`"none"` pass through; `{"name": "search"}` becomes
    the OpenAI forced-function shape; a full OpenAI dict passes through."""
    if isinstance(tool_choice, str):
        if tool_choice not in ("auto", "required", "none"):
            raise ValueError(
                f"tool_choice must be 'auto', 'required', 'none' or {{'name': ...}}, "
                f"got {tool_choice!r}"
            )
        return tool_choice
    if isinstance(tool_choice, dict):
        if "type" in tool_choice:
            return tool_choice
        if isinstance(tool_choice.get("name"), str):
            return {"type": "function", "function": {"name": tool_choice["name"]}}
    raise ValueError(f"unsupported tool_choice: {tool_choice!r}")


def _parse_tool_call(index: int, call: Any) -> ToolCall:
    call = call if isinstance(call, dict) else {}
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    raw = fn.get("arguments")
    call_id = call.get("id") or f"call_{index}"
    if isinstance(raw, dict):  # some providers hand back an object already
        return ToolCall(id=call_id, name=name, arguments=raw, raw_arguments=json.dumps(raw))
    raw_text = raw if isinstance(raw, str) else ""
    if not raw_text.strip():  # a no-argument tool is legitimately sent as ""
        return ToolCall(id=call_id, name=name, arguments={}, raw_arguments=raw_text)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            raw_arguments=raw_text,
            arguments_error=f"arguments are not valid JSON: {exc}",
        )
    if not isinstance(parsed, dict):
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            raw_arguments=raw_text,
            arguments_error=f"arguments must be a JSON object, got {type(parsed).__name__}",
        )
    return ToolCall(id=call_id, name=name, arguments=parsed, raw_arguments=raw_text)


def normalize_gateway_base_url(url: str) -> str:
    """Strip a trailing `/v1` (and trailing slashes) off a model-gateway base URL.

    Every request method below spells the OpenAI-compatible path in full
    (`/v1/chat/completions`, `/v1/embeddings`), so this base URL must be the
    ORIGIN only. httpx does not treat that as a detail you can get away with:
    `AsyncClient(base_url="http://litellm:4000/v1")` turns a request for
    `/v1/chat/completions` into `http://litellm:4000/v1/v1/chat/completions`,
    which LiteLLM answers 404 — and a 404 is not in the transient set, so the
    backoff ladder does not retry it and every ingestion stage that needs a
    model fails on its first call.

    Normalizing rather than rejecting, because `…:4000/v1` is the form every
    OpenAI SDK asks for and the form an operator will therefore write. Measured
    2026-10-04: one rendered env carried `MODEL_GATEWAY_URL=http://litellm:4000/v1`
    while the live containers had `http://litellm:4000`, i.e. the two spellings
    were already in production side by side with only one of them working. Accepting both makes the key
    un-mis-settable instead of leaving a 404 waiting for the next
    `refresh-env.sh` run on the live stack.
    """
    trimmed = url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed


class LLMClient:
    def __init__(
        self,
        settings: Any,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        timeout_s: float | None = None,
        alias_timeouts: dict[str, float] | None = None,
    ):
        # `settings`: any object with `.model_gateway_url` / `.model_gateway_key`
        # (and `.embedding_model` for `embed`) — duck-typed, no service import.
        # Optional, also duck-typed: `.llm_timeout_s` (client default) and
        # `.llm_alias_timeouts` ({alias: seconds}); the keyword arguments win.
        self._settings = settings
        self._service = getattr(
            settings, "usage_service", settings.__class__.__module__.split(".")[0]
        )
        self.timeout_s = float(
            timeout_s or getattr(settings, "llm_timeout_s", None) or DEFAULT_TIMEOUT_S
        )
        self.alias_timeouts: dict[str, float] = {
            **(getattr(settings, "llm_alias_timeouts", None) or {}),
            **(alias_timeouts or {}),
        }
        self._http = httpx.AsyncClient(
            base_url=normalize_gateway_base_url(settings.model_gateway_url),
            transport=transport,
            timeout=_timeout(self.timeout_s),
            headers={"Authorization": f"Bearer {settings.model_gateway_key}"},
        )

    def timeout_for(self, model: str | None, timeout_s: float | None = None) -> float:
        """Seconds a call to `model` may take: the per-call value, else the
        alias override, else the client default."""
        if timeout_s:
            return float(timeout_s)
        if model and model in self.alias_timeouts:
            return float(self.alias_timeouts[model])
        return self.timeout_s

    async def _post_with_backoff(
        self,
        path: str,
        json: dict,
        max_attempts: int = MAX_HTTP_ATTEMPTS,
        backoff_schedule: tuple[float, ...] | None = None,
        *,
        timeout_s: float | None = None,
        read_timeout_retries: int | None = None,
    ) -> httpx.Response:
        """POST with exponential backoff on transient failures.

        A single 429 used to abort a whole stage run (pilot follow-up #1);
        now transient statuses and transport errors retry with jitter, and
        only a non-transient status or exhausted attempts surface.

        `max_attempts` defaults to the full ladder for batch/ingestion callers;
        user-facing paths (e.g. the rerank fallback) pass a smaller cap so a
        dead upstream can't ride the full multi-second backoff.

        `backoff_schedule`, when given, REPLACES the default 1,2,4,8s
        exponential ladder with explicit per-attempt delays (still jittered)
        and forces `max_attempts` to `len(backoff_schedule) + 1` regardless of
        the `max_attempts` argument — the wait-not-fallback policy (Task 3):
        subscription-only ingestion callers pass a long wait ladder (minutes,
        not seconds) so a 429-heavy bulk run WAITS for the subscription rate
        limit window to reset rather than falling through to a metered model.
        `None` (the default) leaves every existing caller's behavior
        byte-identical to before this parameter existed.

        `timeout_s` overrides the read timeout for this call (else the alias
        override for `json["model"]`, else the client default, see
        `timeout_for`).

        Read timeouts are NOT transient like the rest: the request reached the
        gateway and the provider is most likely still generating (and will
        bill) the reply we stopped waiting for, so a re-send pays twice for
        one result. `read_timeout_retries` caps how many re-sends a
        `httpx.ReadTimeout` gets: `None` means 0 for chat completions (paid,
        non-idempotent generation) and the full ladder for anything else
        (embeddings: cheap, deterministic). Every read timeout is logged
        (`llm.read_timeout`). Connect errors, pool/connect timeouts, other
        transport errors and 429/5xx keep the ladder as before.
        """
        if backoff_schedule:
            max_attempts = len(backoff_schedule) + 1
        if read_timeout_retries is None:
            read_timeout_retries = 0 if path == CHAT_PATH else max_attempts
        model = json.get("model") if isinstance(json, dict) else None
        seconds = self.timeout_for(model, timeout_s)
        read_timeouts = 0
        last_exc: Exception | None = None
        call_id = new_call_id()
        for attempt in range(max_attempts):
            try:
                resp = await self._http.post(
                    path,
                    json=gateway_body(
                        json, service=self._service, call_id=call_id, attempt=attempt + 1
                    ),
                    timeout=_timeout(seconds),
                )
            except httpx.ReadTimeout as exc:
                last_exc = exc
                read_timeouts += 1
                retry = read_timeouts <= read_timeout_retries and attempt < max_attempts - 1
                _log.warning(
                    "LLM call to %s timed out after %.0f s waiting for the reply "
                    "(attempt %d, %s); the provider may still bill it",
                    model or path,
                    seconds,
                    attempt + 1,
                    "re-sending" if retry else "not re-sending",
                    extra={
                        "event": "llm.read_timeout",
                        "fields": {
                            "model": model,
                            "path": path,
                            "call_id": call_id,
                            "attempt": attempt + 1,
                            "timeout_s": seconds,
                            "retry": retry,
                        },
                    },
                )
                if not retry:
                    raise
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()
                    _record_call_usage(json, resp)
                    return resp
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            if attempt < max_attempts - 1:
                delay = (
                    backoff_schedule[attempt] if backoff_schedule else _BACKOFF_BASE * (2**attempt)
                )
                await asyncio.sleep(delay * (1 + random.random() * 0.25))  # noqa: S311 — retry jitter, not crypto
        if last_exc is None:
            # Only reachable if max_attempts <= 0 (a caller passing 0, or a
            # backoff_schedule that somehow computed one): the loop never ran,
            # so no request was made. `raise None` would die with a bare
            # TypeError that says nothing about the real mistake.
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        raise last_exc

    async def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict,
        images: list[bytes] | None = None,
        backoff_schedule: tuple[float, ...] | None = None,
        tags: CallTags | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        content: str | list = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()},
                }
                for img in images
            ]
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        body: dict = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        if tags is not None:
            body = tags.apply_to_body(body)
        last_err: Exception | None = None
        last_raw: str | None = None
        for attempt in range(3):  # blind try, then two corrective retries
            attempt_body = body
            if attempt and last_raw is not None:
                # feed the failure back — a blind retry usually repeats the same
                # shape mistake (observed live: three figure-stage deaths on
                # three different missing fields), a corrected one rarely does
                attempt_body = {
                    **body,
                    "messages": messages
                    + [
                        {"role": "assistant", "content": last_raw},
                        {
                            "role": "user",
                            "content": "That reply was rejected by the JSON validator: "
                            f"{str(last_err)[:500]}\n"
                            "Reply again with corrected JSON only.",
                        },
                    ],
                }
            resp = await self._post_with_backoff(
                CHAT_PATH,
                attempt_body,
                backoff_schedule=backoff_schedule,
                **_timeout_kw(timeout_s),
            )
            # content can legitimately be null (safety-filter refusals do this) —
            # coerce to "" so it fails as a normal JSONDecodeError and gets the
            # corrective-retry treatment instead of an uncaught TypeError
            raw = resp.json()["choices"][0]["message"].get("content") or ""
            try:
                out = json.loads(raw)
                Draft202012Validator(schema).validate(out)
                return out
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err, last_raw = exc, raw
        raise LLMOutputError(f"model {model} returned invalid JSON three times: {last_err}")

    async def chat_tools(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: ToolChoice = "auto",
        max_tokens: int | None = None,
        parallel_tool_calls: bool = True,
        tags: CallTags | None = None,
        *,
        max_attempts: int = MAX_HTTP_ATTEMPTS,
        backoff_schedule: tuple[float, ...] | None = None,
        timeout_s: float | None = None,
    ) -> ToolReply:
        """One non-streamed, tool-enabled chat completion (OpenAI format; LiteLLM
        normalises every provider to it).

        Transport, virtual key, accounting tags, retry/backoff and per-attempt
        usage folding into the active `usage_tally` are the same path as
        `chat_json` (`_post_with_backoff`). There is no corrective retry: a
        malformed `arguments` string is returned as a `ToolCall` with
        `arguments_error` set, never raised. `max_attempts` defaults to the full
        ladder like `chat_json`; user-facing agent loops pass a smaller cap.

        `tools` and `tool_choice` are omitted from the body when `tools` is
        empty (providers reject `tools: []`). `parallel_tool_calls` is sent only
        when False: True is the provider default, and some providers reject the
        parameter outright.
        """
        body: dict = {"model": model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = _tool_choice_body(tool_choice)
            if not parallel_tool_calls:
                body["parallel_tool_calls"] = False
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tags is not None:
            body = tags.apply_to_body(body)
        resp = await self._post_with_backoff(
            CHAT_PATH,
            body,
            max_attempts=max_attempts,
            backoff_schedule=backoff_schedule,
            **_timeout_kw(timeout_s),
        )
        payload = resp.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ToolReply(
            content=message.get("content") or "",
            tool_calls=[
                _parse_tool_call(i, call) for i, call in enumerate(message.get("tool_calls") or [])
            ],
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            finish_reason=choice.get("finish_reason"),
            message=message,
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        timeout_s: float = 120.0,
        tags: CallTags | None = None,
    ) -> AsyncIterator[str]:
        """Streamed chat completion for the student-facing model alias.

        Unlike `chat_json`, this is a SINGLE attempt: `_post_with_backoff`'s
        retry ladder doesn't apply to a stream — retrying after tokens have
        already reached the student would duplicate/garble what they've
        seen. The whole call (connect through last chunk) is bounded by
        `timeout_s` via `asyncio.timeout`, not just the connect phase, since
        student-chat is user-facing and must not hang the turn indefinitely.
        `timeout_s` is caller-supplied rather than read off `self._settings`
        because this client is shared/duck-typed across services (spec:
        extracted in Task 2) and only the tutor service's Settings carries
        `chat_timeout_s`.
        """
        stream_body: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tags is not None:
            stream_body = tags.apply_to_body(stream_body)
        body = gateway_body(
            stream_body,
            service=self._service,
            call_id=new_call_id(),
            attempt=1,
        )
        # Accounting for a STREAM is not the same problem as for a single
        # reply: the `usage` block arrives (if at all) in a trailing frame
        # after the last token, so it is only knowable once the iteration is
        # over — including when it ends badly. The `finally` is the point:
        # a timeout, a client disconnect or a mid-stream provider error still
        # burned the prompt plus whatever was generated, and recording nothing
        # there would be the fabricated zero AGENTS.md forbids.
        usage: dict | None = None
        streamed_chars = 0
        try:
            async with asyncio.timeout(timeout_s):
                async with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if not data or data == "[DONE]":
                            continue
                        chunk = json.loads(data)
                        # Which frame carries `usage` is provider-dependent:
                        # the usage-only final frame (empty `choices`) is the
                        # documented shape, but some put it on the last
                        # content frame. Take the last one seen from ANY frame
                        # rather than betting on a frame shape.
                        if isinstance(chunk.get("usage"), dict):
                            usage = chunk["usage"]
                        if not chunk.get("choices"):
                            continue  # final usage-only frame has an empty choices list
                        delta = chunk["choices"][0]["delta"].get("content")
                        if delta:
                            streamed_chars += len(delta)
                            yield delta
        finally:
            record_usage(usage, estimate=estimate_tokens(_body_chars(body) + streamed_chars))

    async def embed(
        self,
        texts: list[str],
        dimensions: int = 1536,
        model: str | None = None,
        backoff_schedule: tuple[float, ...] | None = None,
        tags: CallTags | None = None,
        *,
        timeout_s: float | None = None,
    ) -> list[list[float]]:
        import math

        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            req = {
                "model": model or self._settings.embedding_model,
                "input": batch,
                "dimensions": dimensions,
            }
            resp = await self._post_with_backoff(
                "/v1/embeddings",
                tags.apply_to_body(req) if tags is not None else req,
                backoff_schedule=backoff_schedule,
                **_timeout_kw(timeout_s),
            )
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            for item in data:
                v = item["embedding"]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / norm for x in v])
        return out

    async def close(self) -> None:
        await self._http.aclose()


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_HTTP_ATTEMPTS",
    "RETRYABLE_STATUS",
    "LLMClient",
    "LLMOutputError",
    "ToolCall",
    "ToolChoice",
    "ToolReply",
]
