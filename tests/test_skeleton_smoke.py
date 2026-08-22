"""End-to-end skeleton proof: outbox emit -> poller -> rabbitmq -> consumer -> dedupe."""
import asyncio
import uuid

import pytest
import sqlalchemy as sa
from bacmr_common.consumer import EventConsumer, processed_ddl
from bacmr_common.outbox import emit, outbox_ddl
from bacmr_common.poller import OutboxPoller
from sqlalchemy.ext.asyncio import create_async_engine

DSN = "postgresql+asyncpg://knowledge_svc:knowledge_dev@127.0.0.1:5433/bacmr"
AMQP = "amqp://bacmr:bacmr_dev@127.0.0.1:5673/"
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
    consumer = EventConsumer(eng, "content", AMQP, qname,
                             ["content.document.ingested"], handler)
    await consumer.start()

    payload = {"document_id": str(uuid.uuid4()), "subject_id": "math",
               "language": "en", "chunk_count": 7, "source": "curriculum"}
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
            n = (await conn.execute(sa.text(
                "SELECT count(*) FROM content.processed_events WHERE event_id = :e"
            ), {"e": str(eid)})).scalar()
        if n == 0:
            await asyncio.sleep(0.2)
    assert n == 1

    await consumer.stop()
    await poller.close()
    await eng.dispose()
