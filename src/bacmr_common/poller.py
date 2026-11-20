"""Publishes outbox rows to RabbitMQ. SKIP LOCKED makes concurrent pollers safe;
rows are only stamped published_at after broker confirm, so delivery is at-least-once.

Every broker await is bounded (BacMR/knowledge#532): a broker drop can hold the
batch's row locks for at most `batch_timeout` (plus one `publish_timeout`), and
Postgres ends a transaction left idle longer than that even if this process wedges.
"""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable

import aio_pika
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from bacmr_common._sql import validate_schema_ident
from bacmr_common.amqp import client_properties

_log = logging.getLogger("bacmr_common.poller")
_sleep = asyncio.sleep  # module-level so tests can observe the backoff

# Server-side backstops, SET LOCAL so they end with the transaction.
_TX_TIMEOUTS_SQL = sa.text(
    "SELECT set_config('idle_in_transaction_session_timeout', :idle, true), "
    "set_config('statement_timeout', :stmt, true)"
)


class OutboxPoller:
    def __init__(
        self,
        engine: AsyncEngine,
        schema: str,
        amqp_url: str,
        exchange: str = "bacmr.events",
        batch: int = 100,
        connection_name: str | None = None,
        *,
        connect_timeout: float = 30.0,
        channel_timeout: float = 30.0,
        publish_timeout: float = 10.0,
        batch_timeout: float = 30.0,
        max_backoff: float = 60.0,
        on_heartbeat: Callable[[], None] | None = None,
    ):
        self._engine = engine
        self._schema = validate_schema_ident(schema)
        self._amqp_url = amqp_url
        self._exchange_name = exchange
        self._batch = batch
        self._connection_name = connection_name
        self._connect_timeout = connect_timeout
        self._channel_timeout = channel_timeout
        self._publish_timeout = publish_timeout
        self._batch_timeout = batch_timeout
        self._max_backoff = max_backoff
        self._on_heartbeat = on_heartbeat
        self._conn: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None
        # time.monotonic() of the last loop iteration / last successful batch.
        self.last_heartbeat: float | None = None
        self.last_success: float | None = None

    async def _ensure_channel(self) -> aio_pika.abc.AbstractExchange:
        if self._exchange is not None:
            return self._exchange
        if self._conn is None or self._conn.is_closed:
            # aio-pika reuses `timeout` for every reconnect attempt of the robust
            # connection; the outer bound is a backstop for the handshake itself.
            async with asyncio.timeout(self._connect_timeout * 2):
                self._conn = await aio_pika.connect_robust(
                    self._amqp_url,
                    timeout=self._connect_timeout,
                    **client_properties(self._connection_name),
                )
        channel = self._conn.channel(publisher_confirms=True)
        self._channel = channel
        try:
            async with asyncio.timeout(self._channel_timeout):
                await channel
                self._exchange = await channel.declare_exchange(
                    self._exchange_name,
                    aio_pika.ExchangeType.TOPIC,
                    durable=True,
                    timeout=self._channel_timeout,
                )
        except BaseException:
            await self._drop_channel()
            raise
        return self._exchange

    async def _drop_channel(self) -> None:
        channel, self._channel, self._exchange = self._channel, None, None
        if channel is None:
            return
        with contextlib.suppress(Exception):
            async with asyncio.timeout(self._channel_timeout):
                await channel.close()

    def _envelope_message(self, r) -> aio_pika.Message:
        envelope = {
            "event_id": str(r["event_id"]),
            "routing_key": r["routing_key"],
            "occurred_at": r["occurred_at"].isoformat(),
            "version": 1,
            "payload": r["payload"],
        }
        return aio_pika.Message(
            body=json.dumps(envelope).encode(),
            message_id=str(r["event_id"]),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

    async def _publish(self, ex: aio_pika.abc.AbstractExchange, r) -> None:
        timeout = self._publish_timeout
        async with asyncio.timeout(timeout):
            await ex.publish(
                self._envelope_message(r), routing_key=r["routing_key"], timeout=timeout
            )

    async def _claim(self, conn: AsyncConnection):
        # `self._schema` is validated in __init__ (validate_schema_ident) as a bare SQL
        # identifier; `:n`/`:id` bind as query parameters. Same audited pattern as
        # retrieval.py's nosemgrep sites.
        await conn.execute(
            _TX_TIMEOUTS_SQL,
            {
                "idle": str(int((self._batch_timeout + 30) * 1000)),
                "stmt": str(int(self._batch_timeout * 1000)),
            },
        )
        result = await conn.execute(
            sa.text(  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                "SELECT id, event_id, routing_key, payload, occurred_at "  # noqa: S608
                f"FROM {self._schema}.outbox WHERE published_at IS NULL "
                "ORDER BY id LIMIT :n FOR UPDATE SKIP LOCKED"
            ),
            {"n": self._batch},
        )
        return result.mappings().all()

    async def _stamp(self, conn: AsyncConnection, row_id: int) -> None:
        await conn.execute(
            sa.text(  # nosemgrep: python.sqlalchemy.security.audit.avoid-sqlalchemy-text.avoid-sqlalchemy-text
                f"UPDATE {self._schema}.outbox SET published_at = now() "  # noqa: S608
                "WHERE id = :id"
            ),
            {"id": row_id},
        )

    async def run_once(self) -> int:
        """Publish one batch. On a publish failure or timeout the rows confirmed so
        far are committed, the rest are released for the next call, and the error
        is re-raised once the transaction has ended."""
        ex = await self._ensure_channel()
        published = 0
        failure: BaseException | None = None
        async with self._engine.begin() as conn:
            rows = await self._claim(conn)
            deadline = time.monotonic() + self._batch_timeout
            for r in rows:
                if time.monotonic() >= deadline:
                    break  # the rest stay unpublished, unlocked at commit
                try:
                    await self._publish(ex, r)
                except Exception as exc:  # TimeoutError, channel/connection errors
                    failure = exc
                    break
                await self._stamp(conn, r["id"])
                published += 1
        if failure is not None:
            await self._drop_channel()
            raise failure
        return published

    def _beat(self) -> None:
        self.last_heartbeat = time.monotonic()
        if self._on_heartbeat is None:
            return
        try:
            self._on_heartbeat()
        except Exception as exc:
            _log.warning(
                "outbox heartbeat callback failed",
                extra={"event": "outbox.heartbeat_error", "error": exc},
            )

    async def run_forever(self, interval: float = 1.0) -> None:
        """Poll until cancelled. A failed iteration is logged as `outbox.poll_error`
        and retried after a backoff (interval doubling up to `max_backoff`); it never
        ends the loop. Each iteration beats `last_heartbeat` / `on_heartbeat`."""
        backoff = interval
        while True:
            try:
                n = await self.run_once()
            except Exception as exc:
                _log.warning(
                    "outbox poll failed; retrying",
                    extra={
                        "event": "outbox.poll_error",
                        "schema": self._schema,
                        "backoff_s": backoff,
                        "error": exc,
                    },
                )
                self._beat()
                await _sleep(backoff)
                backoff = min(backoff * 2, max(self._max_backoff, interval))
                continue
            backoff = interval
            self.last_success = time.monotonic()
            self._beat()
            if n == 0:
                await _sleep(interval)

    async def close(self) -> None:
        await self._drop_channel()
        if self._conn is not None:
            conn, self._conn = self._conn, None
            with contextlib.suppress(Exception):
                async with asyncio.timeout(self._connect_timeout):
                    await conn.close()
