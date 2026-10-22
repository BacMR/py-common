"""Idempotent event consumer (spec §3): processed_events dedupe, bounded retries
via a per-queue retry queue (per-message TTL dead-lettering back), then DLQ."""

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable

import aio_pika
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from bacmr_common._sql import validate_schema_ident
from bacmr_common.amqp import client_properties


def processed_ddl(schema: str) -> str:
    schema = validate_schema_ident(schema)
    return f"""
    CREATE TABLE IF NOT EXISTS {schema}.processed_events (
        event_id UUID PRIMARY KEY,
        processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )"""


class EventConsumer:
    def __init__(
        self,
        engine: AsyncEngine,
        schema: str,
        amqp_url: str,
        queue: str,
        routing_keys: list[str],
        handler: Callable[[dict], Awaitable[None]],
        exchange: str = "bacmr.events",
        max_attempts: int = 3,
        retry_delay_ms: int = 10_000,
        connection_name: str | None = None,
    ):
        self._engine = engine
        self._schema = validate_schema_ident(schema)
        self._amqp_url = amqp_url
        self._queue_name = queue
        self._routing_keys = routing_keys
        self._handler = handler
        self._exchange_name = exchange
        self._max_attempts = max_attempts
        self._retry_delay_ms = retry_delay_ms
        self._connection_name = connection_name
        self._conn: aio_pika.abc.AbstractRobustConnection | None = None

    async def start(self) -> None:
        self._conn = await aio_pika.connect_robust(
            self._amqp_url, **client_properties(self._connection_name)
        )
        ch = await self._conn.channel()
        await ch.set_qos(prefetch_count=10)
        ex = await ch.declare_exchange(
            self._exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
        )
        self._dlq = await ch.declare_queue(
            f"{self._queue_name}.dlq", durable=True, arguments={"x-queue-type": "quorum"}
        )
        # retry queue dead-letters expired messages back into the main queue. Stays classic
        # (not quorum): it relies on per-message TTL (`expiration=`) + a DLX to reinject
        # expired messages, a pattern quorum queues don't support the same way — quorum
        # queues have no per-message TTL dead-lettering guarantee equivalent to classic
        # queues' expired-message-at-head semantics used here.
        self._retry_q = await ch.declare_queue(
            f"{self._queue_name}.retry",
            durable=True,
            arguments={"x-dead-letter-exchange": "", "x-dead-letter-routing-key": self._queue_name},
        )
        # Quorum queues default to delivery_limit=20 (broker policy) with no configured
        # dead-letter route: once a message exhausts its 20 redeliveries RabbitMQ drops
        # it silently. A fast nack-requeue loop (e.g. during a dedupe-store outage) can
        # burn all 20 in milliseconds. Routing delivery-limit-exceeded messages to the
        # DLQ here is the safety net that turns "silently dropped" into "inspectable".
        self._main_q = await ch.declare_queue(
            self._queue_name,
            durable=True,
            arguments={
                "x-queue-type": "quorum",
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": f"{self._queue_name}.dlq",
            },
        )
        for rk in self._routing_keys:
            await self._main_q.bind(ex, rk)
        self._ch = ch
        await self._main_q.consume(self._on_message)

    async def _already_processed(self, event_id: str) -> bool:
        # `self._schema` is validated in __init__ (validate_schema_ident) as a bare SQL
        # identifier; `:eid` binds as a query parameter. Same audited pattern as
        # retrieval.py's nosemgrep sites.
        async with self._engine.connect() as conn:
            row = await conn.execute(
                sa.text(  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                    f"SELECT 1 FROM {self._schema}.processed_events WHERE event_id = :eid"  # noqa: S608
                ),
                {"eid": event_id},
            )
            return row.first() is not None

    async def _mark_processed(self, event_id: str) -> None:
        # after successful handling only — a crash mid-handler leaves the message
        # unacked and unmarked, so redelivery reprocesses it (at-least-once).
        # `self._schema` is validated in __init__; `:eid` binds as a query parameter.
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.text(  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                    f"INSERT INTO {self._schema}.processed_events (event_id) "  # noqa: S608
                    "VALUES (:eid) ON CONFLICT DO NOTHING"
                ),
                {"eid": event_id},
            )

    async def _on_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        try:
            envelope = json.loads(message.body)
            event_id = envelope["event_id"]
            # a non-UUID event_id (wrong type, malformed string) would otherwise pass
            # this guard and blow up inside _already_processed's asyncpg UUID cast,
            # where the blanket infra-error except misclassifies it as a DB outage and
            # nack-requeues it forever. Reject it here as malformed instead.
            uuid.UUID(str(event_id))
        except (
            json.JSONDecodeError,
            KeyError,
            UnicodeDecodeError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            # poison message: not our JSON envelope shape at all, no amount of retrying
            # will fix it. Dead-letter immediately so it doesn't wedge the queue forever.
            await self._ch.default_exchange.publish(
                aio_pika.Message(
                    body=message.body,
                    headers={"x-dead-reason": "malformed"},
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=f"{self._queue_name}.dlq",
            )
            await message.ack()
            return

        try:
            already_processed = await self._already_processed(event_id)
        except Exception:  # noqa: BLE001 - dedupe store unreachable; not the message's fault
            # infrastructure error (e.g. DB down), not a poison message: requeue and let
            # redelivery retry the dedupe check once the store is back. Throttle the
            # redelivery so a sustained outage doesn't burn the quorum queue's
            # delivery-limit (20) in milliseconds; the DLX declared on the main queue
            # is the safety net if it's exceeded anyway.
            await asyncio.sleep(1.0)
            await message.nack(requeue=True)
            return
        if already_processed:
            await message.ack()
            return
        try:
            await self._handler(envelope)
        except Exception:  # noqa: BLE001 - intentional catch-all for retry routing
            # aio-pika types a header value as the whole AMQP field union. Ours
            # is always the int we published, but a broker/plugin round trip can
            # hand back a string or bytes, and int() on a table/timestamp/None
            # would raise INSIDE the retry-routing handler -- killing the
            # consumer on the one path that exists to keep it alive.
            raw_attempts = message.headers.get("x-attempts", 0)
            attempts = (
                int(raw_attempts) if isinstance(raw_attempts, int | float | str | bytes) else 0
            ) + 1
            body = message.body
            if attempts >= self._max_attempts:
                await self._ch.default_exchange.publish(
                    aio_pika.Message(
                        body=body,
                        headers={"x-attempts": attempts},
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=f"{self._queue_name}.dlq",
                )
            else:
                # exponential backoff: 1x, 2x, 4x, ... retry_delay_ms. Note the retry queue
                # is a single shared classic queue using per-message TTL (`expiration=`) to
                # dead-letter back to the main queue — RabbitMQ only expires messages from
                # the head, so once delays differ across messages, an expired message stuck
                # behind an unexpired one waits for the one ahead of it (head-of-line
                # blocking). Fine at this volume; revisit if retry latency matters more.
                await self._ch.default_exchange.publish(
                    aio_pika.Message(
                        body=body,
                        headers={"x-attempts": attempts},
                        expiration=self._retry_delay_ms / 1000 * 2 ** (attempts - 1),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=f"{self._queue_name}.retry",
                )
            await message.ack()
            return
        await self._mark_processed(event_id)
        await message.ack()

    async def stop(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
