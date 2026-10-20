"""Transactional outbox (spec §3): events written in the caller's transaction,
published by OutboxPoller. The outbox table is also the permanent event archive."""

import json
import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from bacmr_common._sql import validate_schema_ident
from bacmr_contracts import make_envelope


def outbox_ddl(schema: str) -> str:
    schema = validate_schema_ident(schema)
    return f"""
    CREATE TABLE IF NOT EXISTS {schema}.outbox (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        event_id UUID NOT NULL UNIQUE,
        routing_key TEXT NOT NULL,
        payload JSONB NOT NULL,
        occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        published_at TIMESTAMPTZ
    )"""


async def emit(conn: AsyncConnection, schema: str, routing_key: str, payload: dict) -> uuid.UUID:
    schema = validate_schema_ident(schema)
    envelope = make_envelope(routing_key, payload)
    await conn.execute(
        # `schema` is validated above (validate_schema_ident) as a bare SQL identifier;
        # every runtime value (:eid, :rk, :payload) binds as a query parameter. Same
        # audited pattern as retrieval.py's nosemgrep sites.
        sa.text(  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
            f"INSERT INTO {schema}.outbox (event_id, routing_key, payload) "  # noqa: S608
            "VALUES (:eid, :rk, CAST(:payload AS JSONB))"
        ),
        {"eid": envelope["event_id"], "rk": routing_key, "payload": json.dumps(payload)},
    )
    return uuid.UUID(envelope["event_id"])
