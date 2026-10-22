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

**Status (2026-10-22):** extracted with history from `archive-backend` (`libs/py-common` + `contracts/py`); version `0.3.0` adds the explicit `CallTags.task` Spend bucket (BacMR/docs#55). Tag `v0.3.0` is cut on `dev` after that PR merges. CI: ruff + pytest (integration tests are opt-in, see below).

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
    "tutor-api", "chat", request_id, plan_tier="premium", user_id=profile_id, task="student-question"
)
await llm.chat_json(model, system, user, schema, tags=tags)  # also chat_stream / embed
await llm.embed(
    texts, tags=CallTags.system("knowledge-worker", "embed", task="embedding")
)  # system work: uuid4 id, no user
```

Request fields: `user` = student profile id (omitted for system work); `metadata.tags = [service, plan_tier, request_id, purpose]` (exact order; `request_id` is the gateway-minted `X-Request-Id`). LiteLLM records them as `end_user` / `request_tags`. Allowed values: service in `gateway|tutor-api|knowledge-api|knowledge-worker|scraper`; plan_tier in `free|standard|premium|system`; purpose in `chat|understanding|sufficiency|summary|retrieve-rescue|ingest-text|ingest-vision|ingest-enrich|embed|rerank|moderation`; anything else raises `ValueError`. Optional `task=` (since v0.3.0, BacMR/docs#55) names the admin Spend bucket and is appended as a fifth tag `task:<label>`, with label in `student-question|query-understanding|sufficiency-check|conversation-summary|retrieval-rerank|embedding|ocr-vision-page|text-extraction|figure-description|chunk-enrichment|bridge-questions|translation|moderation|evals|quiz-generation`; the first four tags never move, and calls without `task` still emit exactly four. `tags` is optional on the client; `bacmr_common.usage` still appends its own `consumption:`/`service:`/`operation:` tags after these four, and an explicit `user` in the body (i.e. `user_id`) overrides the usage-context actor, so only authenticated boundaries (gateway/tutor-api after verifying the session) may set `user_id`. When `tags` is passed it replaces any caller-supplied `metadata.tags`; other metadata keys are kept.

### Extraction notes

- `bacmr_contracts` (monorepo `contracts/py`) is vendored here because `outbox` imports it and it has no other consumer yet; its history is carried too.
- Tests that need the sibling services (`bacmr_knowledge`, `bacmr_tutor`) skip when those are not installed; the cross-suite endpoint-guard scan in `tests/test_env_guard.py` is empty here.
