# py-common — shared Python library

**Owns:** the code `knowledge` and `tutor` (and any future Python service) must agree on:

- `settings` base (pydantic-settings conventions, Infisical-rendered env)
- Postgres engine/session helpers (asyncpg/SQLAlchemy), transactional outbox helper
- LiteLLM client wrapper (alias routing, spend tags, retries)
- logging / request-id / tracing middleware
- event envelope + RabbitMQ publisher/consumer helpers for `bacmr.events`

**Does not own:** any business logic, any schema.

**Versioning:** semantic version git tags (`v0.1.0`…); services pin `bacmr-py-common @ git+https://github.com/BacMR/py-common@vX.Y.Z`. A change here is released by tag **before** the service PR that needs it.

**Provenance:** extracted from `archive-backend/libs/py-common`.

**Status (2026-10-20):** scaffold; extraction PR pending.
