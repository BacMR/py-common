import asyncio
import json
import uuid

import _env
import aio_pika
import pytest
import sqlalchemy as sa
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


@pytest.fixture
async def engine():
    eng = create_async_engine(DSN)
    async with eng.begin() as conn:
        await conn.execute(sa.text(outbox_ddl("content")))
        await conn.execute(sa.text("TRUNCATE content.outbox"))
    yield eng
    await eng.dispose()


async def test_poller_publishes_and_stamps(engine):
    conn_mq = await aio_pika.connect_robust(AMQP)
    ch = await conn_mq.channel()
    ex = await ch.declare_exchange("bacmr.events", aio_pika.ExchangeType.TOPIC, durable=True)
    q = await ch.declare_queue("test.poller", exclusive=True)
    await q.bind(ex, "content.#")

    payload = {
        "document_id": str(uuid.uuid4()),
        "subject_id": "svt",
        "language": "ar",
        "chunk_count": 3,
        "source": "curriculum",
    }
    async with engine.begin() as c:
        eid = await emit(c, "content", "content.document.ingested", payload)

    poller = OutboxPoller(engine, "content", AMQP)
    n = await poller.run_once()
    assert n == 1

    msg = await asyncio.wait_for(q.get(timeout=5), timeout=10)
    body = json.loads(msg.body)
    assert body["event_id"] == str(eid)
    assert body["payload"] == payload
    await msg.ack()

    async with engine.connect() as c:
        left = (
            await c.execute(
                sa.text("SELECT count(*) FROM content.outbox WHERE published_at IS NULL")
            )
        ).scalar()
    assert left == 0

    assert await poller.run_once() == 0  # idempotent: nothing left
    await poller.close()
    await conn_mq.close()
