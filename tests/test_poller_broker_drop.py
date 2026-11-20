"""OutboxPoller under a broker drop (BacMR/knowledge#532), with fakes only.

A RabbitMQ restart must not hold outbox row locks or a DB transaction past a
bound, and `run_forever` must keep polling (and beating) through it.
"""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

import aio_pika
import pytest

from bacmr_common import poller as poller_mod
from bacmr_common.poller import OutboxPoller

# --- fake Postgres: row locks live as long as the transaction that took them ---


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)


class FakeDB:
    def __init__(self, n_rows: int = 0):
        self.rows: dict[int, dict] = {}
        self.locked: set[int] = set()
        self.open_tx = 0
        self.settings: list[dict] = []
        self.fail_commit = False
        for _ in range(n_rows):
            self.add_row()

    def add_row(self) -> int:
        rid = len(self.rows) + 1
        self.rows[rid] = {
            "id": rid,
            "event_id": uuid.uuid4(),
            "routing_key": "content.document.ingested",
            "payload": {"n": rid},
            "occurred_at": datetime(2026, 11, 20, tzinfo=UTC),
            "published_at": None,
        }
        return rid

    def unpublished(self) -> list[int]:
        return [rid for rid, r in self.rows.items() if r["published_at"] is None]

    def begin(self):
        return _Tx(self)


class _Tx:
    def __init__(self, db: FakeDB):
        self._db = db
        self._locks: set[int] = set()
        self._stamped: set[int] = set()

    async def __aenter__(self):
        self._db.open_tx += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is None and self._db.fail_commit:
            self._db.locked -= self._locks
            self._db.open_tx -= 1
            raise ConnectionResetError("connection lost on COMMIT")
        if exc_type is None:
            for rid in self._stamped:
                self._db.rows[rid]["published_at"] = datetime.now(UTC)
        self._db.locked -= self._locks
        self._db.open_tx -= 1
        return False

    async def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        params = params or {}
        if "set_config" in sql:
            self._db.settings.append(dict(params))
            return _Result([])
        if sql.startswith("SELECT"):
            assert "FOR UPDATE SKIP LOCKED" in sql
            free = [
                rid
                for rid in self._db.unpublished()
                if rid not in self._db.locked and rid not in self._stamped
            ][: params["n"]]
            self._locks |= set(free)
            self._db.locked |= set(free)
            return _Result([dict(self._db.rows[rid]) for rid in free])
        if sql.startswith("UPDATE"):
            assert params["id"] in self._locks
            self._stamped.add(params["id"])
            return _Result([])
        raise AssertionError(f"unexpected SQL: {sql}")


# --- fake RabbitMQ: publish / channel open / connect can hang or raise ---


class FakeBroker:
    def __init__(self):
        self.publish_mode = "ok"  # ok | hang | raise | slow
        self.publish_delay = 0.0
        self.channel_mode = "ok"  # ok | hang
        self.connect_failures = 0
        self.connect_calls: list[dict] = []
        self.channels_opened = 0
        self.channels_closed = 0
        self.published: list[dict] = []
        self.fail_after: int | None = None  # raise once this many messages went out

    async def connect_robust(self, url, **kwargs):
        self.connect_calls.append(kwargs)
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise ConnectionError("broker restarting")
        return _Conn(self)


async def _forever():
    await asyncio.Event().wait()


class _Conn:
    def __init__(self, broker: FakeBroker):
        self._broker = broker
        self.is_closed = False

    def channel(self, publisher_confirms: bool = True):
        assert publisher_confirms is True
        return _Channel(self._broker)

    async def close(self):
        self.is_closed = True


class _Channel:
    def __init__(self, broker: FakeBroker):
        self._broker = broker
        self.is_closed = False

    def __await__(self):
        return self._open().__await__()

    async def _open(self):
        if self._broker.channel_mode == "hang":
            await _forever()
        self._broker.channels_opened += 1
        return self

    async def declare_exchange(self, name, type_, durable, timeout=None):
        assert timeout is not None
        return _Exchange(self._broker, self)

    async def close(self):
        self.is_closed = True
        self._broker.channels_closed += 1


class _Exchange:
    def __init__(self, broker: FakeBroker, channel: _Channel):
        self._broker = broker
        self._channel = channel

    async def publish(self, message, routing_key, timeout=None):
        b = self._broker
        if self._channel.is_closed:
            raise aio_pika.exceptions.ChannelInvalidStateError("channel closed")
        if b.fail_after is not None and len(b.published) >= b.fail_after:
            raise aio_pika.exceptions.ChannelInvalidStateError("No active transport in channel")
        if b.publish_mode == "hang":
            await _forever()
        if b.publish_mode == "raise":
            raise aio_pika.exceptions.ChannelInvalidStateError("No active transport in channel")
        if b.publish_mode == "slow":
            await asyncio.sleep(b.publish_delay)
        b.published.append({"routing_key": routing_key, "body": json.loads(message.body)})


@pytest.fixture
def broker(monkeypatch):
    b = FakeBroker()
    monkeypatch.setattr(poller_mod.aio_pika, "connect_robust", b.connect_robust)
    return b


def _poller(db, **kw):
    kw.setdefault("publish_timeout", 0.05)
    kw.setdefault("batch_timeout", 0.5)
    kw.setdefault("channel_timeout", 0.05)
    return OutboxPoller(db, "content", "amqp://fake/", **kw)


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- run_once: no lock or transaction outlives a broker drop ---


async def test_hanging_publish_releases_row_locks_within_bound(broker):
    db = FakeDB(3)
    broker.publish_mode = "hang"
    poller = _poller(db)

    task = asyncio.create_task(poller.run_once())
    await asyncio.sleep(0.5)

    assert db.open_tx == 0, "transaction still open while the broker hangs"
    assert db.locked == set(), "outbox rows still locked while the broker hangs"
    assert task.done()
    with pytest.raises(TimeoutError):
        task.result()
    assert db.unpublished() == [1, 2, 3]


async def test_publish_error_mid_batch_keeps_confirmed_rows_and_retries_the_rest(broker):
    db = FakeDB(3)
    broker.fail_after = 1
    poller = _poller(db)

    with pytest.raises(aio_pika.exceptions.ChannelInvalidStateError):
        await poller.run_once()

    assert db.open_tx == 0
    assert db.locked == set()
    assert db.unpublished() == [2, 3]
    assert broker.channels_closed == 1

    broker.fail_after = None
    assert await poller.run_once() == 2
    assert db.unpublished() == []
    assert broker.channels_opened == 2, "a fresh channel replaces the broken one"
    assert [m["body"]["payload"]["n"] for m in broker.published] == [1, 2, 3]


async def test_broken_channel_is_dropped_even_when_the_commit_fails(broker):
    db = FakeDB(2)
    broker.fail_after = 1
    db.fail_commit = True
    poller = _poller(db)

    with pytest.raises(ConnectionResetError):
        await poller.run_once()

    assert broker.channels_closed == 1
    assert db.unpublished() == [1, 2]  # row 1 goes out again: at-least-once


async def test_batch_deadline_bounds_how_long_locks_are_held(broker):
    db = FakeDB(10)
    broker.publish_mode = "slow"
    broker.publish_delay = 0.03
    poller = _poller(db, publish_timeout=1.0, batch_timeout=0.1)

    first = await poller.run_once()

    assert 0 < first < 10
    assert db.open_tx == 0
    assert db.locked == set()
    total = first
    while total < 10:
        total += await poller.run_once()
    assert db.unpublished() == []


async def test_transaction_carries_server_side_timeouts(broker):
    db = FakeDB(1)
    poller = _poller(db, publish_timeout=10.0, batch_timeout=30.0)

    await poller.run_once()

    (cfg,) = db.settings
    assert int(cfg["idle"]) == 60_000
    assert int(cfg["stmt"]) == 30_000


async def test_connect_is_bounded_and_named(broker):
    db = FakeDB(0)
    poller = OutboxPoller(db, "content", "amqp://fake/", connection_name="knowledge-outbox")

    await poller.run_once()

    assert broker.connect_calls == [
        {"timeout": 30.0, "client_properties": {"connection_name": "knowledge-outbox"}}
    ]


async def test_hanging_channel_open_is_bounded_then_recovers(broker):
    db = FakeDB(1)
    broker.channel_mode = "hang"
    poller = _poller(db)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(poller.run_once(), timeout=1.0)
    assert db.open_tx == 0

    broker.channel_mode = "ok"
    assert await poller.run_once() == 1
    assert len(broker.connect_calls) == 1, "the robust connection is reused"


async def test_failed_connect_is_retried_on_the_next_call(broker):
    db = FakeDB(1)
    broker.connect_failures = 1
    poller = _poller(db)

    with pytest.raises(ConnectionError):
        await poller.run_once()
    assert await poller.run_once() == 1
    assert len(broker.connect_calls) == 2


async def test_close_is_idempotent_and_reconnects_after(broker):
    db = FakeDB(1)
    poller = _poller(db)
    await poller.run_once()

    await poller.close()
    await poller.close()

    db.add_row()
    assert await poller.run_once() == 1
    assert len(broker.connect_calls) == 2


# --- run_forever: logs, backs off, keeps beating, never exits on an error ---


async def test_run_forever_survives_a_broker_restart_and_keeps_beating(broker, caplog):
    db = FakeDB(2)
    broker.connect_failures = 2
    beats: list[float] = []
    poller = _poller(db, max_backoff=0.02, on_heartbeat=lambda: beats.append(1.0))

    with caplog.at_level(logging.WARNING, logger="bacmr_common.poller"):
        task = asyncio.create_task(poller.run_forever(interval=0.01))
        for _ in range(200):
            if not db.unpublished():
                break
            await asyncio.sleep(0.01)
        assert not task.done(), "run_forever exited"
        await _cancel(task)

    assert db.unpublished() == []
    errors = [r for r in caplog.records if getattr(r, "event", None) == "outbox.poll_error"]
    assert len(errors) == 2
    assert errors[0].backoff_s <= errors[1].backoff_s
    assert len(beats) >= 3
    assert poller.last_heartbeat is not None
    assert poller.last_success is not None


async def test_run_forever_keeps_polling_while_publish_hangs(broker):
    db = FakeDB(1)
    broker.publish_mode = "hang"
    poller = _poller(db, max_backoff=0.02)

    task = asyncio.create_task(poller.run_forever(interval=0.01))
    released = 0
    for _ in range(60):  # ~0.3 s: each bounded attempt locks, then lets go
        await asyncio.sleep(0.005)
        released += db.locked == set() and db.open_tx == 0
    beat_while_down = poller.last_heartbeat
    assert not task.done()
    assert released > 0, "the row lock was never released while the broker hung"
    assert poller.last_success is None

    broker.publish_mode = "ok"
    for _ in range(100):
        if not db.unpublished():
            break
        await asyncio.sleep(0.01)
    await _cancel(task)

    assert beat_while_down is not None
    assert db.unpublished() == []
    assert poller.last_success is not None


async def test_backoff_doubles_up_to_the_cap_and_resets_after_success(broker, monkeypatch):
    db = FakeDB(0)
    broker.connect_failures = 4
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 6:
            raise asyncio.CancelledError

    monkeypatch.setattr(poller_mod, "_sleep", fake_sleep)
    poller = _poller(db, max_backoff=4.0)

    with pytest.raises(asyncio.CancelledError):
        await poller.run_forever(interval=1.0)

    assert sleeps == [1.0, 2.0, 4.0, 4.0, 1.0, 1.0]


async def test_a_raising_heartbeat_does_not_stop_the_loop(broker, caplog):
    db = FakeDB(1)

    def bad_beat():
        raise RuntimeError("watchdog file unwritable")

    poller = _poller(db, on_heartbeat=bad_beat)
    with caplog.at_level(logging.WARNING, logger="bacmr_common.poller"):
        task = asyncio.create_task(poller.run_forever(interval=0.01))
        await asyncio.sleep(0.1)
        assert not task.done()
        await _cancel(task)

    assert db.unpublished() == []
    assert any(getattr(r, "event", None) == "outbox.heartbeat_error" for r in caplog.records)
