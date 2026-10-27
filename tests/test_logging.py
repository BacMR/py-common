import asyncio
import io
import json
import logging

import pytest

from bacmr_common import logging as blog
from bacmr_common.usage import current_usage_field, gateway_body, usage_context


@pytest.fixture
def capture():
    buf = io.StringIO()
    handler = blog.setup_logging("tutor-api", env="dev", version="abc123", stream=buf)
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines() if line]
    logging.getLogger().removeHandler(handler)


def test_contract_fields(capture):
    log = logging.getLogger("bacmr_tutor.x")
    with blog.log_context(request_id="rid-1", conversation_id="c1", user_id="u1"):
        log.info("hello %s", "world", extra={"event": "chat.turn", "duration_ms": 12.5})
    (line,) = capture()
    assert line["time"].endswith("Z") and "T" in line["time"]
    assert line["level"] == "info"
    assert (line["service"], line["env"], line["version"]) == ("tutor-api", "dev", "abc123")
    assert line["event"] == "chat.turn"
    assert line["msg"] == "hello world"
    assert line["request_id"] == "rid-1"
    assert line["conversation_id"] == "c1" and line["user_id"] == "u1"
    assert line["duration_ms"] == 12.5


def test_legacy_fields_extra_and_default_event(capture):
    log = logging.getLogger("bacmr_knowledge.api")
    log.warning("w", extra={"fields": {"event": "retrieve", "hits": 3}})
    log.info("plain")
    a, b = capture()
    assert a["level"] == "warn" and a["event"] == "retrieve" and a["hits"] == 3
    assert b["event"] == "log"


def test_error_object(capture):
    log = logging.getLogger("x")
    try:
        raise ValueError("boom user@example.com")
    except ValueError:
        log.exception("failed", extra={"event": "job.failed", "status": "failed"})
    (line,) = capture()
    assert line["level"] == "error"
    assert line["error"]["type"] == "ValueError"
    assert "[email]" in line["error"]["message"]
    assert "Traceback" in line["error"]["stack"]


def test_request_id_falls_back_to_usage_context(capture):
    with usage_context(request_id="from-usage"):
        logging.getLogger("x").info("m")
    assert capture()[0]["request_id"] == "from-usage"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Authorization: Bearer abcdefghijklmnop", "[redacted]"),
        ("password=hunter22 next", "password=[redacted] next"),
        ('{"api_key": "sk-abcdefghijklmnopqrst"}', "[redacted]"),
        ("mail ali@bacmr.com now", "mail [email] now"),
        ("call +222 36 12 34 56", "call [phone]"),
        ("call 36123456 please", "call [phone] please"),
        ("jwt eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM.abcdef", "jwt [redacted]"),
        ("amqp://bacmr:pw@mq.invalid:1/", "[redacted]@mq.invalid:1/"),
    ],
)
def test_redact(raw, expected):
    assert expected in blog.redact(raw)
    assert "hunter22" not in blog.redact(raw)


def test_redact_keeps_ids_and_numbers():
    rid = "12345678-1234-4234-9234-123456789012"
    assert blog.redact_value(rid, "request_id") == rid
    assert blog.redact("took 12 ms, page 34, 2026-10-27T12:00:00Z") == (
        "took 12 ms, page 34, 2026-10-27T12:00:00Z"
    )
    assert blog.redact_value({"token": "x", "nested": {"secret": 1}}) == {
        "token": "[redacted]",
        "nested": {"secret": "[redacted]"},
    }


def test_valid_request_id():
    assert blog.valid_request_id("abc-1.2:3_x") == "abc-1.2:3_x"
    assert blog.valid_request_id(b"abc") == "abc"
    assert blog.valid_request_id("a b") is None
    assert blog.valid_request_id("x" * 129) is None
    assert blog.valid_request_id("evil\n{") is None
    assert blog.valid_request_id(None) is None


def test_propagation_helpers():
    assert blog.outgoing_headers() == {}
    with blog.log_context(request_id="r1", job_id="j1", page=3):
        assert blog.outgoing_headers() == {"X-Request-Id": "r1"}
        assert blog.amqp_headers() == {"x-request-id": "r1", "x-job-id": "j1"}

        class Req:
            headers: dict = {}

        req = Req()
        req.headers = {}
        asyncio.run(blog.httpx_event_hooks()["request"][0](req))
        assert req.headers["X-Request-Id"] == "r1"


def test_bind_from_amqp_headers_sets_usage_and_log():
    with blog.bind_from_amqp_headers({"x-request-id": "r9", "x-job-id": "bad id"}):
        assert blog.current_request_id() == "r9"
        assert current_usage_field("request_id") == "r9"
        assert "job_id" not in blog.current_context()
    assert blog.current_request_id() is None


def test_gateway_body_uses_log_context_request_id():
    with blog.log_context(request_id="r-log"):
        body = gateway_body({"model": "m"}, service="tutor-api", call_id="c", attempt=1)
    assert body["metadata"]["request_id"] == "r-log"
    assert body["litellm_trace_id"] == "r-log"


def test_uvicorn_access_record(capture):
    rec = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "f",
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:1", "GET", "/x?token=abc", "1.1", 200),
        None,
    )
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").handle(rec)
    (line,) = capture()
    assert line["event"] == "http.access" and line["path"] == "/x" and line["status"] == 200


def test_text_format():
    buf = io.StringIO()
    h = blog.setup_logging("knowledge-api", fmt="text", stream=buf)
    try:
        with blog.log_context(request_id="r"):
            logging.getLogger("x").info("hi", extra={"event": "e"})
    finally:
        logging.getLogger().removeHandler(h)
    assert "knowledge-api e hi request_id=r" in buf.getvalue()


def test_setup_is_idempotent():
    h1 = blog.setup_logging("a", stream=io.StringIO())
    h2 = blog.setup_logging("a", stream=io.StringIO())
    marked = [h for h in logging.getLogger().handlers if getattr(h, "_bacmr_log_handler", False)]
    assert marked == [h2] and h1 is not h2
    logging.getLogger().removeHandler(h2)


def test_structlog_processor():
    proc = blog.structlog_processor("scraper", env="dev", version="v")
    with blog.log_context(request_id="r"):
        out = proc(None, "warning", {"event": "hello a@b.co", "event_name": "scrape.page", "k": 1})
    assert out["level"] == "warn" and out["event"] == "scrape.page"
    assert out["msg"] == "hello [email]" and out["request_id"] == "r" and out["k"] == 1


async def _app(scope, receive, send):
    from bacmr_common.usage import current_usage_field

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"x-seen", (current_usage_field("request_id") or "").encode())],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


def _run(mw, headers):
    sent = []

    async def send(m):
        sent.append(m)

    async def receive():
        return {"type": "http.request"}

    scope = {"type": "http", "method": "GET", "path": "/v1/x", "headers": headers}
    asyncio.run(mw(scope, receive, send))
    return dict(sent[0]["headers"])


def test_middleware_accepts_and_echoes(capture):
    hdrs = _run(blog.RequestIdMiddleware(_app), [(b"x-request-id", b"abc-123")])
    assert hdrs[b"x-request-id"] == b"abc-123"
    assert hdrs[b"x-seen"] == b"abc-123"
    line = capture()[-1]
    assert line["event"] == "http.request" and line["status"] == 200
    assert line["request_id"] == "abc-123" and "duration_ms" in line


def test_middleware_mints_on_bad_id(capture):
    hdrs = _run(blog.RequestIdMiddleware(_app), [(b"x-request-id", b"bad id\n")])
    rid = hdrs[b"x-request-id"].decode()
    assert rid != "bad id\n" and len(rid) == 36
