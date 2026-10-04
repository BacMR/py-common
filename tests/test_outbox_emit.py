import uuid

import _env
import pytest
import sqlalchemy as sa
from bacmr_common.outbox import emit, outbox_ddl
from sqlalchemy.ext.asyncio import create_async_engine

# Same resolver as the other integration modules here: itest defaults, and the
# production engine refused outright. This TRUNCATEs content.outbox.
# The DSN names its ROLE here: only `knowledge_svc` may create and TRUNCATE in
# `content` (`tutor_svc` is denied -- deploy/tests/test_db_isolation.py pins that),
# so TEST_DATABASE_URL carries the server and each suite names the role it needs.
DSN = _env.role_dsn("knowledge_svc", "knowledge_dev")
pytestmark = pytest.mark.integration


@pytest.fixture
async def engine():
    eng = create_async_engine(DSN)
    async with eng.begin() as conn:
        await conn.execute(sa.text(outbox_ddl("content")))
        await conn.execute(sa.text("TRUNCATE content.outbox"))
    yield eng
    await eng.dispose()


PAYLOAD = {
    "document_id": str(uuid.uuid4()),
    "subject_id": "svt",
    "language": "fr",
    "chunk_count": 12,
    "source": "curriculum",
}


async def test_emit_writes_row_in_same_transaction(engine):
    async with engine.begin() as conn:
        eid = await emit(conn, "content", "content.document.ingested", PAYLOAD)
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                sa.text("SELECT event_id, routing_key, published_at FROM content.outbox")
            )
        ).one()
    assert row.event_id == eid
    assert row.routing_key == "content.document.ingested"
    assert row.published_at is None


async def test_emit_rolls_back_with_transaction(engine):
    with pytest.raises(RuntimeError):
        async with engine.begin() as conn:
            await emit(conn, "content", "content.document.ingested", PAYLOAD)
            raise RuntimeError("business logic failed")
    async with engine.connect() as conn:
        n = (await conn.execute(sa.text("SELECT count(*) FROM content.outbox"))).scalar()
    assert n == 0


async def test_emit_rejects_invalid_payload(engine):
    import jsonschema

    async with engine.begin() as conn:
        with pytest.raises(jsonschema.ValidationError):
            await emit(conn, "content", "content.document.ingested", {"nope": 1})
