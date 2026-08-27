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

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_HTTP_ATTEMPTS = 5
_BACKOFF_BASE = 1.0   # 1, 2, 4, 8 seconds (+ up to 25% jitter)


class LLMOutputError(RuntimeError):
    """The model returned unparseable or schema-invalid JSON twice."""


class LLMClient:
    def __init__(self, settings: Any,
                 transport: httpx.AsyncBaseTransport | None = None):
        # `settings`: any object with `.model_gateway_url` / `.model_gateway_key`
        # (and `.embedding_model` for `embed`) — duck-typed, no service import.
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.model_gateway_url, transport=transport, timeout=120,
            headers={"Authorization": f"Bearer {settings.model_gateway_key}"})

    async def _post_with_backoff(self, path: str, json: dict,
                                 max_attempts: int = MAX_HTTP_ATTEMPTS) -> httpx.Response:
        """POST with exponential backoff on transient failures.

        A single 429 used to abort a whole stage run (pilot follow-up #1);
        now transient statuses and transport errors retry with jitter, and
        only a non-transient status or exhausted attempts surface.

        `max_attempts` defaults to the full ladder for batch/ingestion callers;
        user-facing paths (e.g. the rerank fallback) pass a smaller cap so a
        dead upstream can't ride the full multi-second backoff.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = await self._http.post(path, json=json)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()
                    return resp
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp)
            if attempt < max_attempts - 1:
                delay = _BACKOFF_BASE * (2 ** attempt)
                await asyncio.sleep(delay * (1 + random.random() * 0.25))
        raise last_exc

    async def chat_json(self, model: str, system: str, user: str, schema: dict,
                        images: list[bytes] | None = None) -> dict:
        content: str | list = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(img).decode()}}
                for img in images]
        body = {"model": model, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content}]}
        last_err: Exception | None = None
        last_raw: str | None = None
        for attempt in range(3):  # blind try, then two corrective retries
            attempt_body = body
            if attempt and last_raw is not None:
                # feed the failure back — a blind retry usually repeats the same
                # shape mistake (observed live: three figure-stage deaths on
                # three different missing fields), a corrected one rarely does
                attempt_body = {**body, "messages": body["messages"] + [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user", "content":
                        "That reply was rejected by the JSON validator: "
                        f"{str(last_err)[:500]}\n"
                        "Reply again with corrected JSON only."}]}
            resp = await self._post_with_backoff("/v1/chat/completions", attempt_body)
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
        body = {"model": model, "messages": messages, "stream": True}
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
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta

    async def embed(self, texts: list[str], dimensions: int = 1536,
                    model: str | None = None) -> list[list[float]]:
        import math

        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            resp = await self._post_with_backoff("/v1/embeddings", {
                "model": model or self._settings.embedding_model,
                "input": batch, "dimensions": dimensions})
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            for item in data:
                v = item["embedding"]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / norm for x in v])
        return out

    async def close(self) -> None:
        await self._http.aclose()


__all__ = ["MAX_HTTP_ATTEMPTS", "RETRYABLE_STATUS", "LLMClient", "LLMOutputError"]
