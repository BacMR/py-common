"""Broker connection naming (BacMR/docs#83): unit level, no broker needed."""

import contextlib

from sqlalchemy.ext.asyncio import create_async_engine

from bacmr_common import consumer as consumer_mod
from bacmr_common import poller as poller_mod
from bacmr_common.amqp import client_properties


def test_client_properties_names_the_connection():
    assert client_properties("knowledge-worker") == {
        "client_properties": {"connection_name": "knowledge-worker"}
    }


def test_client_properties_empty_without_a_name():
    assert client_properties(None) == {}
    assert client_properties("") == {}


class _Boom(Exception):
    pass


def _capture(monkeypatch, module):
    seen: dict = {}

    async def fake_connect_robust(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        raise _Boom  # stop right after the dial: no broker in unit tests

    monkeypatch.setattr(module.aio_pika, "connect_robust", fake_connect_robust)
    return seen


async def _dial(coro):
    with contextlib.suppress(_Boom):
        await coro


async def test_poller_dials_with_its_connection_name(monkeypatch):
    seen = _capture(monkeypatch, poller_mod)
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:1/x")
    p = poller_mod.OutboxPoller(engine, "content", "amqp://x/", connection_name="knowledge-outbox")
    await _dial(p._ensure_channel())
    assert seen["kwargs"] == {"client_properties": {"connection_name": "knowledge-outbox"}}
    await engine.dispose()


async def test_poller_without_a_name_keeps_library_defaults(monkeypatch):
    seen = _capture(monkeypatch, poller_mod)
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:1/x")
    await _dial(poller_mod.OutboxPoller(engine, "content", "amqp://x/")._ensure_channel())
    assert seen["kwargs"] == {}
    await engine.dispose()


async def test_consumer_dials_with_its_connection_name(monkeypatch):
    seen = _capture(monkeypatch, consumer_mod)
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:1/x")

    async def handler(_: dict) -> None:
        return None

    c = consumer_mod.EventConsumer(
        engine, "content", "amqp://x/", "q", ["k"], handler, connection_name="knowledge-worker"
    )
    await _dial(c.start())
    assert seen["url"] == "amqp://x/"
    assert seen["kwargs"] == {"client_properties": {"connection_name": "knowledge-worker"}}
    await engine.dispose()
