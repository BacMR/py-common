"""Publishes outbox rows to RabbitMQ. SKIP LOCKED makes concurrent pollers safe;
rows are only stamped published_at after broker confirm, so delivery is at-least-once."""

import asyncio
import json

import aio_pika
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from bacmr_common._sql import validate_schema_ident


class OutboxPoller:
    def __init__(
        self,
        engine: AsyncEngine,
        schema: str,
        amqp_url: str,
        exchange: str = "bacmr.events",
        batch: int = 100,
    ):
        self._engine = engine
        self._schema = validate_schema_ident(schema)
        self._amqp_url = amqp_url
        self._exchange_name = exchange
        self._batch = batch
        self._conn: aio_pika.abc.AbstractRobustConnection | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    async def _ensure_channel(self) -> aio_pika.abc.AbstractExchange:
        if self._exchange is None:
            self._conn = await aio_pika.connect_robust(self._amqp_url)
            ch = await self._conn.channel(publisher_confirms=True)
            self._exchange = await ch.declare_exchange(
                self._exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
            )
        return self._exchange

    async def run_once(self) -> int:
        ex = await self._ensure_channel()
        published = 0
        # `self._schema` is validated in __init__ (validate_schema_ident) as a bare SQL
        # identifier; `:n`/`:id` bind as query parameters. Same audited pattern as
        # retrieval.py's nosemgrep sites.
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        sa.text(  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                            "SELECT id, event_id, routing_key, payload, occurred_at "  # noqa: S608
                            f"FROM {self._schema}.outbox WHERE published_at IS NULL "
                            "ORDER BY id LIMIT :n FOR UPDATE SKIP LOCKED"
                        ),
                        {"n": self._batch},
                    )
                )
                .mappings()
                .all()
            )
            for r in rows:
                envelope = {
                    "event_id": str(r["event_id"]),
                    "routing_key": r["routing_key"],
                    "occurred_at": r["occurred_at"].isoformat(),
                    "version": 1,
                    "payload": r["payload"],
                }
                await ex.publish(
                    aio_pika.Message(
                        body=json.dumps(envelope).encode(),
                        message_id=str(r["event_id"]),
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=r["routing_key"],
                )
                await conn.execute(
                    sa.text(  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                        f"UPDATE {self._schema}.outbox SET published_at = now() "  # noqa: S608
                        "WHERE id = :id"
                    ),
                    {"id": r["id"]},
                )
                published += 1
        return published

    async def run_forever(self, interval: float = 1.0) -> None:
        while True:
            n = await self.run_once()
            if n == 0:
                await asyncio.sleep(interval)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._exchange = None
