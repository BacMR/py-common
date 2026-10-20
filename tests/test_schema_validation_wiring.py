"""Schema-identifier validation is wired into every constructor/DDL entry point that
interpolates it into SQL (see `bacmr_common._sql.validate_schema_ident`). These are
pure/construction-time checks -- no DB or broker connection is made -- so unlike their
siblings in test_consumer.py/test_poller.py/test_outbox_emit.py they don't need the
compose stack and aren't marked `integration`."""

import pytest

from bacmr_common.consumer import EventConsumer, processed_ddl
from bacmr_common.outbox import outbox_ddl
from bacmr_common.poller import OutboxPoller

BAD_SCHEMA = "content; DROP TABLE outbox;--"


def test_outbox_ddl_rejects_unsafe_schema():
    with pytest.raises(ValueError, match="not a valid SQL schema identifier"):
        outbox_ddl(BAD_SCHEMA)


def test_processed_ddl_rejects_unsafe_schema():
    with pytest.raises(ValueError, match="not a valid SQL schema identifier"):
        processed_ddl(BAD_SCHEMA)


def test_outbox_poller_rejects_unsafe_schema():
    with pytest.raises(ValueError, match="not a valid SQL schema identifier"):
        OutboxPoller(engine=None, schema=BAD_SCHEMA, amqp_url="amqp://unused/")


def test_event_consumer_rejects_unsafe_schema():
    async def _handler(_event: dict) -> None:
        pass

    with pytest.raises(ValueError, match="not a valid SQL schema identifier"):
        EventConsumer(
            engine=None,
            schema=BAD_SCHEMA,
            amqp_url="amqp://unused/",
            queue="q",
            routing_keys=["rk"],
            handler=_handler,
        )
