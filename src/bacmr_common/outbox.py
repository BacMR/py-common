"""Transactional outbox (spec §3): events written in the caller's transaction,
published by OutboxPoller. The outbox table is also the permanent event archive."""
import json
import uuid

import sqlalchemy as sa
from bacmr_contracts import make_envelope
from sqlalchemy.ext.asyncio import AsyncConnection


def outbox_ddl(schema: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {schema}.outbox (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        event_id UUID NOT NULL UNIQUE,
        routing_key TEXT NOT NULL,
        payload JSONB NOT NULL,
        occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        published_at TIMESTAMPTZ
    )"""


async def emit(
    conn: AsyncConnection, schema: str, routing_key: str, payload: dict
) -> uuid.UUID:
    envelope = make_envelope(routing_key, payload)
    await conn.execute(
        sa.text(
            f"INSERT INTO {schema}.outbox (event_id, routing_key, payload) "
            "VALUES (:eid, :rk, CAST(:payload AS JSONB))"
        ),
        {"eid": envelope["event_id"], "rk": routing_key, "payload": json.dumps(payload)},
    )
    return uuid.UUID(envelope["event_id"])
