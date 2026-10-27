# py-common — shared Python library

**Owns:** the code `knowledge` and `tutor` (and any future Python service) must agree on:

- `settings` base (pydantic-settings conventions, Infisical-rendered env)
- Postgres engine/session helpers (asyncpg/SQLAlchemy), transactional outbox helper
- LiteLLM client wrapper (alias routing, spend tags, retries)
- logging / request-id / tracing middleware
- event envelope + RabbitMQ publisher/consumer helpers for `bacmr.events`

**Does not own:** any business logic, any database schema (the vendored `bacmr_contracts` JSON schemas are message contracts, not DB schemas).

**Versioning:** semantic version git tags (`v0.1.0`…); services pin `bacmr-py-common @ git+https://github.com/BacMR/py-common@vX.Y.Z`. A change here is released by tag **before** the service PR that needs it.

**Provenance:** extracted from `archive-backend/libs/py-common`.

**Status (2026-10-22):** extracted with history from `archive-backend` (`libs/py-common` + `contracts/py`); version `0.3.0` added the explicit `CallTags.task` Spend bucket (BacMR/docs#55); version `0.4.0` adds `bacmr_common.language`, the one Arabic-script rule shared by knowledge retrieval and the tutor (BacMR/docs#9). Tag `v0.4.0` is cut on `dev` after that PR merges. Version `0.4.1` adds `bacmr_common.amqp.client_properties` and a `connection_name` parameter on `EventConsumer` / `OutboxPoller`, so every broker connection carries a name the gateway status page can attribute (BacMR/docs#83). Version `0.5.0` adds the `news-classify` purpose and the `news-classification` Spend task for the education news monitor (BacMR/docs#111). Version `0.6.0` adds `bacmr_common.logging`, the one log contract for every Python service (BacMR/docs#118); `EventConsumer` binds and carries `x-request-id`/`x-*-id` headers and logs `event.consumed`/`event.failed`. CI: ruff + pytest (integration tests are opt-in, see below).

## Usage

Packages (distribution `bacmr-py-common`, Python >= 3.12):

| Module | Purpose |
|---|---|
| `bacmr_common.llm` | LiteLLM client wrapper |
| `bacmr_common.outbox` | transactional outbox `emit` + DDL |
| `bacmr_common.poller` | outbox poller publishing to RabbitMQ |
| `bacmr_common.consumer` | idempotent event consumer (`processed` ledger) |
| `bacmr_common.accounting` | `CallTags`: `user` + `metadata.tags` labels for every LiteLLM call |
| `bacmr_common.usage`, `.subjects`, `._sql` | usage tally, subject-profile table, SQL identifier guard |
| `bacmr_common.language` | `is_arabic` / `arabic_share`: the one Arabic-script rule (≥ 30 % of letters, five Arabic blocks) used by retrieval and the tutor |
| `bacmr_common.logging` | JSON log contract, request-id context, redaction, ASGI `RequestIdMiddleware`, HTTP/AMQP propagation helpers (BacMR/docs#118) |
| `bacmr_contracts` | event envelope + JSON schemas (`make_envelope`, `validate_event`) |

```bash
uv sync
uv run ruff check .
uv run pytest -q          # unit tests; `-m integration` needs the isolated bacmr-it stack (see tests/_env.py)
```

Install in a service: `bacmr-py-common @ git+ssh://git@bacmr-py-common/BacMR/py-common.git@v0.1.0`.

### Accounting labels

Every LLM call through LiteLLM carries labels (ruling R7, BacMR/docs#23). Per-service virtual keys say *which service* spent; per-call tags say *for whom and why*, so `LiteLLM_SpendLogs` can be sliced by service, plan tier and purpose. The gateway is the only wallet charger: these labels are accounting, never billing.

```python
from bacmr_common.accounting import CallTags

tags = CallTags(
    "tutor-api",
    "chat",
    request_id,
    plan_tier="premium",
    user_id=profile_id,
    task="student-question",
)
await llm.chat_json(model, system, user, schema, tags=tags)  # also chat_stream / embed
await llm.embed(
    texts, tags=CallTags.system("knowledge-worker", "embed", task="embedding")
)  # system work: uuid4 id, no user
```

Request fields: `user` = student profile id (omitted for system work); `metadata.tags = [service, plan_tier, request_id, purpose]` (exact order; `request_id` is the gateway-minted `X-Request-Id`). LiteLLM records them as `end_user` / `request_tags`. Allowed values: service in `gateway|tutor-api|knowledge-api|knowledge-worker|scraper`; plan_tier in `free|standard|premium|system`; purpose in `chat|understanding|sufficiency|summary|retrieve-rescue|ingest-text|ingest-vision|ingest-enrich|embed|rerank|moderation`; anything else raises `ValueError`. Optional `task=` (since v0.3.0, BacMR/docs#55) names the admin Spend bucket and is appended as a fifth tag `task:<label>`, with label in `student-question|query-understanding|sufficiency-check|conversation-summary|retrieval-rerank|embedding|ocr-vision-page|text-extraction|figure-description|chunk-enrichment|bridge-questions|translation|moderation|evals|quiz-generation`; the first four tags never move, and calls without `task` still emit exactly four. `tags` is optional on the client; `bacmr_common.usage` still appends its own `consumption:`/`service:`/`operation:` tags after these four, and an explicit `user` in the body (i.e. `user_id`) overrides the usage-context actor, so only authenticated boundaries (gateway/tutor-api after verifying the session) may set `user_id`. When `tags` is passed it replaces any caller-supplied `metadata.tags`; other metadata keys are kept.

### Logging (v0.6.0, BacMR/docs#118)

Every service writes one JSON object per line on stdout; Grafana Alloy ships them to Loki with only `service`, `level`, `env` as labels (ids stay in the body).

```python
from bacmr_common.logging import RequestIdMiddleware, httpx_event_hooks, setup_logging

setup_logging("tutor-api")  # root + uvicorn loggers -> JSON
app.add_middleware(RequestIdMiddleware)  # accept/mint X-Request-Id, echo it, log http.request
client = httpx.AsyncClient(event_hooks=httpx_event_hooks())  # forwards X-Request-Id
log.info("turn done", extra={"event": "chat.turn", "duration_ms": 812, "status": "ok"})
```

Fields: `time` (UTC ISO, ms, `Z`), `level` (`debug|info|warn|error|fatal`), `service`, `env` (`APP_ENV`, default `dev`), `version` (`SERVICE_VERSION`/`GIT_SHA`, else `unknown`), `event` (default `log`), `msg`, `logger`, the context ids `request_id`, `user_id`, `conversation_id`, `document_id`, `page`, `job_id`, then `duration_ms`, `status`, `error{type,message,stack}` and any flat extras. Extras come from `extra={...}` or the legacy `extra={"fields": {...}}`.

- Context: `log_context(**ids)` / `bind`/`unbind`; `request_id` falls back to the `usage_context` one, and `usage.gateway_body` falls back to the log one, so LiteLLM `metadata.request_id`/`litellm_trace_id` always match the log lines.
- Propagation: `outgoing_headers()` / `httpx_event_hooks()` (HTTP `X-Request-Id`), `amqp_headers()` (`x-request-id`, `x-job-id`, ...) for publishers, `bind_from_amqp_headers(message.headers)` for consumers (`EventConsumer` does it for you and keeps the id headers on retry/DLQ copies). The outbox poller publishes later, outside any request, so outbox events carry no request id.
- Inbound ids must match `^[A-Za-z0-9._:-]{1,128}$`, else a uuid4 is minted.
- Redaction (`redact`, `redact_value`): bearer/basic credentials, JWTs, `sk-` keys, `password=`/`token:`-style pairs, URL userinfo, emails (`[email]`), phone numbers (`[phone]`, +222 and international); dict keys that look like secrets are replaced wholesale; `*_id` fields are never touched. Never log message/answer content at info.
- Optional env: `LOG_LEVEL` (`INFO`), `LOG_FORMAT` (`json`|`text`), `APP_ENV`, `SERVICE_VERSION`. uvicorn access lines are off by default (the middleware's `http.request` replaces them; `setup_logging(..., access_log=True)` keeps them, as `http.access`). structlog users: `structlog_processor(service)` before `JSONRenderer()`.

### Arabic detection

`is_arabic(text)` is True when at least `ARABIC_SHARE_THRESHOLD` (0.30) of the text's letters (`str.isalpha`) are Arabic script (U+0600–06FF, 0750–077F, 08A0–08FF, FB50–FDFF, FE70–FEFF). Digits, symbols and tashkeel do not count; `arabic_share` returns `None` for text with no letters. Knowledge's `detect_query_langs` and the tutor's `detect_language` both call it, so a question is never retrieved as Arabic and answered in French. The threshold was picked from the evals question banks (BacMR/docs#9): every real question at 20–30 % is an English/French question quoting an Arabic name. Change it only with a bench run.

### Extraction notes

- `bacmr_contracts` (monorepo `contracts/py`) is vendored here because `outbox` imports it and it has no other consumer yet; its history is carried too.
- Tests that need the sibling services (`bacmr_knowledge`, `bacmr_tutor`) skip when those are not installed; the cross-suite endpoint-guard scan in `tests/test_env_guard.py` is empty here.
