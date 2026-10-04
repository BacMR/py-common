"""End-to-end skeleton proof: outbox emit -> poller -> rabbitmq -> consumer -> dedupe."""

import asyncio
import uuid

import _env
import pytest
import sqlalchemy as sa
from bacmr_common.consumer import EventConsumer, processed_ddl
from bacmr_common.outbox import emit, outbox_ddl
from bacmr_common.poller import OutboxPoller
from sqlalchemy.ext.asyncio import create_async_engine

# Endpoints come from _env, which defaults to the isolated bacmr-it stack and
# refuses the production brokers / the host Postgres engine outright. These
# constants used to be hardcoded at 127.0.0.1:5673 -- the LIVE v2 broker.
# The DSN names its ROLE here: only `knowledge_svc` may create and TRUNCATE in
# `content` (`tutor_svc` is denied -- deploy/tests/test_db_isolation.py pins that),
# so TEST_DATABASE_URL carries the server and each suite names the role it needs.
DSN = _env.role_dsn("knowledge_svc", "knowledge_dev")
AMQP = _env.AMQP_URL
pytestmark = pytest.mark.integration


async def test_full_round_trip():
    eng = create_async_engine(DSN)
    async with eng.begin() as conn:
        await conn.execute(sa.text(outbox_ddl("content")))
        await conn.execute(sa.text(processed_ddl("content")))
        await conn.execute(sa.text("TRUNCATE content.outbox, content.processed_events"))

    received: list[dict] = []

    async def handler(env: dict) -> None:
        received.append(env)

    qname = f"smoke.{uuid.uuid4().hex[:8]}"
    consumer = EventConsumer(eng, "content", AMQP, qname, ["content.document.ingested"], handler)
    await consumer.start()

    payload = {
        "document_id": str(uuid.uuid4()),
        "subject_id": "math",
        "language": "en",
        "chunk_count": 7,
        "source": "curriculum",
    }
    async with eng.begin() as conn:
        eid = await emit(conn, "content", "content.document.ingested", payload)

    poller = OutboxPoller(eng, "content", AMQP)
    await poller.run_once()

    deadline = asyncio.get_event_loop().time() + 10
    while not received and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.2)

    assert [e["event_id"] for e in received] == [str(eid)]
    # the consumer stamps processed_events in a separate transaction AFTER the
    # handler returns — poll for the row like we polled for delivery, or a
    # loaded host loses the commit race and reads 0
    stamp_deadline = asyncio.get_event_loop().time() + 10
    n = 0
    while n == 0 and asyncio.get_event_loop().time() < stamp_deadline:
        async with eng.connect() as conn:
            n = (
                await conn.execute(
                    sa.text("SELECT count(*) FROM content.processed_events WHERE event_id = :e"),
                    {"e": str(eid)},
                )
            ).scalar()
        if n == 0:
            await asyncio.sleep(0.2)
    assert n == 1

    await consumer.stop()
    await poller.close()
    await eng.dispose()
