import asyncio
import uuid

import _env
import aio_pika
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


@pytest.fixture
async def engine():
    eng = create_async_engine(DSN)
    async with eng.begin() as conn:
        await conn.execute(sa.text(outbox_ddl("content")))
        await conn.execute(sa.text(processed_ddl("content")))
        await conn.execute(sa.text("TRUNCATE content.outbox, content.processed_events"))
    yield eng
    await eng.dispose()


def _payload():
    return {
        "document_id": str(uuid.uuid4()),
        "subject_id": "svt",
        "language": "fr",
        "chunk_count": 1,
        "source": "curriculum",
    }


async def _emit_and_publish(engine, payload):
    async with engine.begin() as c:
        await emit(c, "content", "content.document.ingested", payload)
    poller = OutboxPoller(engine, "content", AMQP)
    await poller.run_once()
    await poller.close()


async def test_delivers_once_and_dedupes(engine):
    seen: list[dict] = []

    async def handler(env: dict) -> None:
        seen.append(env)

    qname = f"test.consumer.{uuid.uuid4().hex[:8]}"
    consumer = EventConsumer(engine, "content", AMQP, qname, ["content.document.ingested"], handler)
    await consumer.start()
    payload = _payload()
    await _emit_and_publish(engine, payload)
    await asyncio.sleep(2)
    assert len(seen) == 1

    # simulate broker redelivery of the same envelope
    conn = await aio_pika.connect_robust(AMQP)
    ch = await conn.channel()
    import json

    await ch.default_exchange.publish(
        aio_pika.Message(body=json.dumps(seen[0]).encode()), routing_key=qname
    )
    await asyncio.sleep(2)
    assert len(seen) == 1  # deduped
    await consumer.stop()
    await conn.close()


async def test_failures_retry_then_dlq(engine):
    attempts: list[int] = []

    async def bad_handler(env: dict) -> None:
        attempts.append(1)
        raise ValueError("boom")

    qname = f"test.consumer.{uuid.uuid4().hex[:8]}"
    consumer = EventConsumer(
        engine,
        "content",
        AMQP,
        qname,
        ["content.document.ingested"],
        bad_handler,
        max_attempts=3,
        retry_delay_ms=200,
    )
    await consumer.start()
    await _emit_and_publish(engine, _payload())
    await asyncio.sleep(5)
    assert len(attempts) == 3

    conn = await aio_pika.connect_robust(AMQP)
    ch = await conn.channel()
    dlq = await ch.declare_queue(f"{qname}.dlq", durable=True, passive=True)
    msg = await dlq.get(timeout=5)
    assert msg is not None
    await msg.ack()
    await consumer.stop()
    await conn.close()


async def test_malformed_body_goes_to_dlq_without_wedging(engine):
    seen: list[dict] = []

    async def handler(env: dict) -> None:
        seen.append(env)

    qname = f"test.consumer.{uuid.uuid4().hex[:8]}"
    consumer = EventConsumer(engine, "content", AMQP, qname, ["content.document.ingested"], handler)
    await consumer.start()

    conn = await aio_pika.connect_robust(AMQP)
    ch = await conn.channel()
    await ch.default_exchange.publish(aio_pika.Message(body=b"not json at all"), routing_key=qname)
    await asyncio.sleep(2)

    dlq = await ch.declare_queue(f"{qname}.dlq", durable=True, passive=True)
    msg = await dlq.get(timeout=5)
    assert msg is not None
    assert msg.body == b"not json at all"
    assert msg.headers.get("x-dead-reason") == "malformed"
    await msg.ack()

    # prove the consumer isn't wedged: a subsequently emitted valid event is still consumed
    await _emit_and_publish(engine, _payload())
    await asyncio.sleep(2)
    assert len(seen) == 1

    await consumer.stop()
    await conn.close()


async def test_non_object_json_body_goes_to_dlq_without_wedging(engine):
    # b"null" parses fine as JSON but isn't a dict, so envelope["event_id"] raises
    # TypeError rather than the KeyError/JSONDecodeError the malformed-body guard
    # originally caught -- that TypeError must not escape _on_message and leave the
    # message permanently unacked.
    seen: list[dict] = []

    async def handler(env: dict) -> None:
        seen.append(env)

    qname = f"test.consumer.{uuid.uuid4().hex[:8]}"
    consumer = EventConsumer(engine, "content", AMQP, qname, ["content.document.ingested"], handler)
    await consumer.start()

    conn = await aio_pika.connect_robust(AMQP)
    ch = await conn.channel()
    await ch.default_exchange.publish(aio_pika.Message(body=b"null"), routing_key=qname)
    await asyncio.sleep(2)

    dlq = await ch.declare_queue(f"{qname}.dlq", durable=True, passive=True)
    msg = await dlq.get(timeout=5)
    assert msg is not None
    assert msg.body == b"null"
    assert msg.headers.get("x-dead-reason") == "malformed"
    await msg.ack()

    # prove the consumer isn't wedged: a subsequently emitted valid event is still consumed
    await _emit_and_publish(engine, _payload())
    await asyncio.sleep(2)
    assert len(seen) == 1

    await consumer.stop()
    await conn.close()


async def test_non_uuid_event_id_goes_to_dlq_without_wedging(engine):
    # A well-formed JSON object with a non-UUID event_id (e.g. "oops") passes the
    # malformed-body guard, but must not reach _already_processed and trip asyncpg's
    # UUID cast (DataError), which the infra-error except would otherwise misclassify
    # as an infra outage and nack-requeue forever. It must be dead-lettered as malformed.
    import json

    seen: list[dict] = []

    async def handler(env: dict) -> None:
        seen.append(env)

    qname = f"test.consumer.{uuid.uuid4().hex[:8]}"
    consumer = EventConsumer(engine, "content", AMQP, qname, ["content.document.ingested"], handler)
    await consumer.start()

    conn = await aio_pika.connect_robust(AMQP)
    ch = await conn.channel()
    body = json.dumps({"event_id": "oops"}).encode()
    await ch.default_exchange.publish(aio_pika.Message(body=body), routing_key=qname)
    await asyncio.sleep(2)

    dlq = await ch.declare_queue(f"{qname}.dlq", durable=True, passive=True)
    msg = await dlq.get(timeout=5)
    assert msg is not None
    assert msg.body == body
    assert msg.headers.get("x-dead-reason") == "malformed"
    await msg.ack()

    # prove the consumer isn't wedged: a subsequently emitted valid event is still consumed
    await _emit_and_publish(engine, _payload())
    await asyncio.sleep(2)
    assert len(seen) == 1

    await consumer.stop()
    await conn.close()
