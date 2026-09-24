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
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx
from jsonschema import Draft202012Validator, ValidationError

from bacmr_common.usage import estimate_tokens, gateway_body, new_call_id, record_usage

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_HTTP_ATTEMPTS = 5
_BACKOFF_BASE = 1.0   # 1, 2, 4, 8 seconds (+ up to 25% jitter)


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
            chars += sum(len(part.get("text", "")) for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    payload_input = body.get("input")
    if isinstance(payload_input, str):
        chars += len(payload_input)
    elif isinstance(payload_input, list):
        chars += sum(len(t) for t in payload_input if isinstance(t, str))
    return chars


def _reply_chars(payload: dict) -> int:
    if not isinstance(payload, dict):
        return 0
    return sum(len((choice.get("message") or {}).get("content") or "")
               for choice in payload.get("choices") or [] if isinstance(choice, dict))


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
    record_usage(usage, estimate=estimate_tokens(
        _body_chars(request_body) + _reply_chars(payload)))


class LLMOutputError(RuntimeError):
    """The model returned unparseable or schema-invalid JSON twice."""


class LLMClient:
    def __init__(self, settings: Any,
                 transport: httpx.AsyncBaseTransport | None = None):
        # `settings`: any object with `.model_gateway_url` / `.model_gateway_key`
        # (and `.embedding_model` for `embed`) — duck-typed, no service import.
        self._settings = settings
        self._service = getattr(settings, "usage_service", settings.__class__.__module__.split(".")[0])
        self._http = httpx.AsyncClient(
            base_url=settings.model_gateway_url, transport=transport, timeout=120,
            headers={"Authorization": f"Bearer {settings.model_gateway_key}"})

    async def _post_with_backoff(self, path: str, json: dict,
                                 max_attempts: int = MAX_HTTP_ATTEMPTS,
                                 backoff_schedule: tuple[float, ...] | None = None,
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
        """
        if backoff_schedule:
            max_attempts = len(backoff_schedule) + 1
        last_exc: Exception | None = None
        call_id = new_call_id()
        for attempt in range(max_attempts):
            try:
                resp = await self._http.post(path, json=gateway_body(
                    json, service=self._service, call_id=call_id, attempt=attempt + 1))
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()
                    _record_call_usage(json, resp)
                    return resp
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp)
            if attempt < max_attempts - 1:
                delay = (backoff_schedule[attempt] if backoff_schedule
                        else _BACKOFF_BASE * (2 ** attempt))
                await asyncio.sleep(delay * (1 + random.random() * 0.25))
        if last_exc is None:
            # Only reachable if max_attempts <= 0 (a caller passing 0, or a
            # backoff_schedule that somehow computed one): the loop never ran,
            # so no request was made. `raise None` would die with a bare
            # TypeError that says nothing about the real mistake.
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        raise last_exc

    async def chat_json(self, model: str, system: str, user: str, schema: dict,
                        images: list[bytes] | None = None,
                        backoff_schedule: tuple[float, ...] | None = None) -> dict:
        content: str | list = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(img).decode()}}
                for img in images]
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": content}]
        body: dict = {"model": model, "response_format": {"type": "json_object"},
                      "messages": messages}
        last_err: Exception | None = None
        last_raw: str | None = None
        for attempt in range(3):  # blind try, then two corrective retries
            attempt_body = body
            if attempt and last_raw is not None:
                # feed the failure back — a blind retry usually repeats the same
                # shape mistake (observed live: three figure-stage deaths on
                # three different missing fields), a corrected one rarely does
                attempt_body = {**body, "messages": messages + [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user", "content":
                        "That reply was rejected by the JSON validator: "
                        f"{str(last_err)[:500]}\n"
                        "Reply again with corrected JSON only."}]}
            resp = await self._post_with_backoff("/v1/chat/completions", attempt_body,
                                                 backoff_schedule=backoff_schedule)
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
        raise LLMOutputError(
            f"model {model} returned invalid JSON three times: {last_err}")

    async def chat_stream(self, model: str, messages: list[dict],
                          timeout_s: float = 120.0) -> AsyncIterator[str]:
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
        body = gateway_body({"model": model, "messages": messages, "stream": True,
                             "stream_options": {"include_usage": True}},
                            service=self._service, call_id=new_call_id(), attempt=1)
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
                async with self._http.stream(
                        "POST", "/v1/chat/completions", json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
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
            record_usage(usage, estimate=estimate_tokens(
                _body_chars(body) + streamed_chars))

    async def embed(self, texts: list[str], dimensions: int = 1536,
                    model: str | None = None,
                    backoff_schedule: tuple[float, ...] | None = None) -> list[list[float]]:
        import math

        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            resp = await self._post_with_backoff("/v1/embeddings", {
                "model": model or self._settings.embedding_model,
                "input": batch, "dimensions": dimensions},
                backoff_schedule=backoff_schedule)
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            for item in data:
                v = item["embedding"]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / norm for x in v])
        return out

    async def close(self) -> None:
        await self._http.aclose()


__all__ = ["MAX_HTTP_ATTEMPTS", "RETRYABLE_STATUS", "LLMClient", "LLMOutputError"]
