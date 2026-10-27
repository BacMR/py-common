"""One log contract for every Python service (BacMR/docs#118).

Every line on stdout is one JSON object with the contract fields::

    time level service env version event msg request_id user_id
    conversation_id document_id page job_id duration_ms status error{type,message,stack}

* `setup_logging("tutor-api")` installs the JSON formatter on the root logger
  and routes uvicorn's loggers through it (idempotent, safe to call twice).
* `log_context(...)` / `bind(...)` put ids in a ContextVar; every line logged
  inside the block carries them. `request_id` also falls back to the
  accounting context (`bacmr_common.usage`), so ids bound there still show.
* `RequestIdMiddleware` (pure ASGI, works with FastAPI/Starlette) accepts or
  mints `X-Request-Id`, binds it, echoes it, and logs one `http.request` line.
* `outgoing_headers()`, `httpx_event_hooks()`, `amqp_headers()` and
  `bind_from_amqp_headers()` propagate the id over HTTP and RabbitMQ.
* Redaction: secrets (bearer/JWT/api keys/`password=`), emails and phone
  numbers are masked in `msg` and in every string field; dict keys that look
  like secrets are replaced wholesale.

Call sites log extra fields either way::

    log.info("retrieved", extra={"event": "retrieve", "duration_ms": 12})
    log.info("retrieved", extra={"fields": {"event": "retrieve"}})  # legacy form

Labels in Loki are only service/level/env; ids stay in the body.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

REQUEST_ID_HEADER = "X-Request-Id"
AMQP_REQUEST_ID_HEADER = "x-request-id"

# Ids bound through the context and copied onto every line.
CONTEXT_FIELDS = (
    "request_id",
    "user_id",
    "conversation_id",
    "document_id",
    "page",
    "job_id",
)
CONTRACT_FIELDS = (
    "time",
    "level",
    "service",
    "env",
    "version",
    "event",
    "msg",
    *CONTEXT_FIELDS,
    "duration_ms",
    "status",
    "error",
)
_ID_FIELDS = frozenset(
    {"request_id", "user_id", "conversation_id", "document_id", "job_id", "event_id"}
)
_LEVELS = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "fatal",
}
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_MARK = "_bacmr_log_handler"
# Attributes every LogRecord has; anything else came in through `extra=`.
_RECORD_ATTRS = frozenset(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {
    "message",
    "asctime",
    "taskName",
    "fields",
    "color_message",
}

_ctx: ContextVar[dict[str, Any] | None] = ContextVar("bacmr_log_ctx", default=None)


# --------------------------------------------------------------------------- context


def new_request_id() -> str:
    return str(uuid.uuid4())


def valid_request_id(value: Any) -> str | None:
    """The inbound id if it is a safe token, else None (caller mints a new one)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _REQUEST_ID_RE.match(value) else None


def bind(**fields: Any) -> Token:
    """Add ids to the ambient log context; returns a token for `unbind`."""
    values = dict(_ctx.get() or {})
    values.update({k: v for k, v in fields.items() if k in CONTEXT_FIELDS and v is not None})
    return _ctx.set(values)


def unbind(token: Token) -> None:
    try:
        _ctx.reset(token)
    except ValueError:  # token from another Context (async generators): just clear
        _ctx.set(None)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    token = bind(**fields)
    try:
        yield
    finally:
        unbind(token)


def current_context() -> dict[str, Any]:
    out = dict(_ctx.get() or {})
    if "request_id" not in out:
        rid = _usage_request_id()
        if rid:
            out["request_id"] = rid
    return out


def current_request_id() -> str | None:
    return current_context().get("request_id")


def _usage_request_id() -> str | None:
    try:
        from bacmr_common.usage import current_usage_field

        return current_usage_field("request_id")
    except Exception:  # noqa: BLE001 - a log line must never raise
        return None


# ----------------------------------------------------------------------- propagation


def outgoing_headers() -> dict[str, str]:
    """Headers for an outgoing internal HTTP call (`X-Request-Id`)."""
    rid = current_request_id()
    return {REQUEST_ID_HEADER: rid} if rid else {}


def _sync_hook(request: Any) -> None:
    rid = current_request_id()
    if rid and REQUEST_ID_HEADER not in request.headers:
        request.headers[REQUEST_ID_HEADER] = rid


async def _async_hook(request: Any) -> None:
    _sync_hook(request)


def httpx_event_hooks(*, asynchronous: bool = True) -> dict[str, list]:
    """`httpx.AsyncClient(event_hooks=httpx_event_hooks())` stamps X-Request-Id."""
    return {"request": [_async_hook if asynchronous else _sync_hook]}


def amqp_headers() -> dict[str, str]:
    """AMQP message headers carrying the ambient ids (`x-request-id`, `x-job-id`, ...)."""
    out: dict[str, str] = {}
    for key, value in current_context().items():
        if value is not None and key != "page":
            out["x-" + key.replace("_", "-")] = str(value)
    return out


@contextmanager
def bind_from_amqp_headers(headers: Mapping[str, Any] | None) -> Iterator[None]:
    """Bind the ids an inbound message carries for the duration of its handling."""
    fields: dict[str, Any] = {}
    for key in CONTEXT_FIELDS:
        if key == "page":
            continue
        raw = (headers or {}).get("x-" + key.replace("_", "-"))
        value = valid_request_id(raw)  # same safe-token rule for every id
        if value:
            fields[key] = value
    with log_context(**fields), _usage_scope(fields.get("request_id")):
        yield


@contextmanager
def _usage_scope(request_id: str | None) -> Iterator[None]:
    if not request_id:
        yield
        return
    from bacmr_common.usage import current_usage_field, usage_context

    if current_usage_field("request_id"):
        yield
        return
    with usage_context(request_id=request_id):
        yield


# ------------------------------------------------------------------------- redaction

_SECRET_KEY_RE = re.compile(
    r"(pass(word|wd)?|secret|token|api[_-]?key|authorization|cookie|credential|private[_-]?key)",
    re.IGNORECASE,
)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # key=value / key: value / "key": "value" with a secret-looking key
    (
        re.compile(
            r"""(?ix)
            (["']?\b[\w.-]*(?:pass(?:word|wd)?|secret|token|api[_-]?key|authorization|cookie)
            [\w.-]*["']?\s*[:=]\s*["']?)
            (?:bearer\s+|basic\s+)?
            [^\s"',;&}]+
            """
        ),
        r"\1[redacted]",
    ),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 [redacted]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"), "[redacted]"),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}"), "[redacted]"),
    (re.compile(r"(?i)\b(?:amqps?|postgres(?:ql)?|redis|mongodb)://[^\s/@]+@"), "[redacted]@"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    # Mauritanian numbers (+222 / 00222 / bare 8 digits starting 2/3/4) and
    # any other international +NNNNNNNN form.
    (re.compile(r"(?<![\w.-])(?:\+|00)222[ .-]?[234]\d(?:[ .-]?\d{2}){3}(?![\w-])"), "[phone]"),
    (re.compile(r"(?<![\w.-])\+\d[\d .-]{7,16}\d(?![\w-])"), "[phone]"),
    (re.compile(r"(?<![\w.:/-])[234]\d(?:[ .]?\d{2}){3}(?![\w.:/-])"), "[phone]"),
)


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def redact_value(value: Any, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY_RE.search(key) and value is not None:
        return "[redacted]"
    if key in _ID_FIELDS:
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(k): redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_value(v) for v in value]
    return value


# ------------------------------------------------------------------------- formatter


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def error_fields(exc: BaseException | None = None, exc_info: Any = None) -> dict[str, str]:
    if exc is None and exc_info:
        exc = exc_info[1]
    if exc is None:
        return {}
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "type": type(exc).__name__,
        "message": redact(str(exc))[:2000],
        "stack": redact(stack)[-8000:],
    }


def _uvicorn_access(record: logging.LogRecord) -> dict[str, Any] | None:
    if record.name != "uvicorn.access" or not isinstance(record.args, tuple):
        return None
    if len(record.args) != 5:
        return None
    _client, method, path, _http, status = record.args
    return {
        "event": "http.access",
        "method": method,
        "path": str(path).split("?", 1)[0],
        "status": status,
    }


class JsonFormatter(logging.Formatter):
    """Renders a LogRecord as one contract JSON line."""

    def __init__(self, service: str, env: str = "dev", version: str = "unknown"):
        super().__init__()
        self.service = service
        self.env = env
        self.version = version

    def to_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        legacy = getattr(record, "fields", None)
        if isinstance(legacy, Mapping):
            extras.update(legacy)
        for key, value in record.__dict__.items():
            if key not in _RECORD_ATTRS and not key.startswith("_"):
                extras[key] = value
        access = _uvicorn_access(record)
        if access:
            extras = {**access, **extras}
            msg = f"{access['method']} {access['path']} {access['status']}"
        else:
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001 - bad %-args must not lose the line
                msg = str(record.msg)

        out: dict[str, Any] = {
            "time": _iso(record.created),
            "level": _LEVELS.get(record.levelno, record.levelname.lower()),
            "service": self.service,
            "env": self.env,
            "version": self.version,
            "event": extras.pop("event", None) or "log",
            "msg": redact(msg),
            "logger": record.name,
        }
        try:
            ctx = current_context()
        except Exception:  # noqa: BLE001
            ctx = {}
        for key in CONTEXT_FIELDS:
            if key in ctx:
                out[key] = ctx[key]
        err = extras.pop("error", None)
        if isinstance(err, BaseException):
            out["error"] = error_fields(err)
        elif isinstance(err, Mapping):
            out["error"] = redact_value(dict(err))
        elif err is not None:
            out["error"] = {"type": "Error", "message": redact(str(err))}
        if record.exc_info and "error" not in out:
            out["error"] = error_fields(exc_info=record.exc_info)
        for key, value in extras.items():
            if key in ("time", "level", "service", "env", "version", "msg", "logger"):
                key = "field_" + key
            out[key] = redact_value(value, key)
        return out

    def format(self, record: logging.LogRecord) -> str:
        try:
            return json.dumps(self.to_dict(record), ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001 - never lose a line to formatting
            return json.dumps(
                {
                    "time": _iso(record.created),
                    "level": "error",
                    "service": self.service,
                    "env": self.env,
                    "version": self.version,
                    "event": "log.format_failed",
                    "msg": redact(str(record.msg))[:2000],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )


class TextFormatter(JsonFormatter):
    """Human-readable variant for local runs (`LOG_FORMAT=text`)."""

    def format(self, record: logging.LogRecord) -> str:
        d = self.to_dict(record)
        ids = " ".join(f"{k}={d[k]}" for k in CONTEXT_FIELDS if k in d)
        line = f"{d['time']} {d['level']:<5} {d['service']} {d['event']} {d['msg']} {ids}"
        if "error" in d and d["error"].get("stack"):
            line += "\n" + d["error"]["stack"]
        return line.rstrip()


# ----------------------------------------------------------------------------- setup

UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _env_version() -> str:
    for key in ("SERVICE_VERSION", "GIT_SHA", "BUILD_SHA"):
        value = os.environ.get(key, "").strip()
        if value:
            return value[:40]
    return "unknown"


def make_formatter(
    service: str, *, env: str | None = None, version: str | None = None, fmt: str | None = None
) -> JsonFormatter:
    env = env or os.environ.get("APP_ENV") or "dev"
    version = version or _env_version()
    fmt = (fmt or os.environ.get("LOG_FORMAT") or "json").lower()
    cls = TextFormatter if fmt == "text" else JsonFormatter
    return cls(service, env, version)


def setup_logging(
    service: str,
    *,
    env: str | None = None,
    version: str | None = None,
    level: str | int | None = None,
    fmt: str | None = None,
    access_log: bool = False,
    stream: Any = None,
) -> logging.Handler:
    """Install the contract formatter on the root logger (idempotent).

    uvicorn's own loggers lose their handlers and propagate to root, so their
    lines become JSON too. `access_log=False` (default) silences uvicorn's
    access lines because `RequestIdMiddleware` logs a richer `http.request`.
    """
    level = level or os.environ.get("LOG_LEVEL") or "INFO"
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _MARK, False):
            root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(make_formatter(service, env=env, version=version, fmt=fmt))
    setattr(handler, _MARK, True)
    root.addHandler(handler)
    root.setLevel(level)
    for name in UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.INFO if access_log else logging.WARNING)
    # Chatty client libraries: their request lines would duplicate ours.
    for name in ("httpx", "httpcore", "aiormq", "aio_pika"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    return handler


def uvicorn_log_config(service: str, **kwargs: Any) -> dict[str, Any]:
    """`uvicorn.run(..., log_config=uvicorn_log_config("tutor-api"))` equivalent:
    no uvicorn handlers, everything propagates to the root handler installed
    by `setup_logging` (call that first)."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {name: {"handlers": [], "propagate": True} for name in UVICORN_LOGGERS},
    }


def structlog_processor(service: str, *, env: str | None = None, version: str | None = None):
    """A structlog processor adding the contract fields + context ids + redaction.

    Put it before `structlog.processors.JSONRenderer()`; structlog's `event`
    (the message) is renamed `msg`, and an explicit `event_name=` becomes `event`.
    """
    fmt = make_formatter(service, env=env, version=version, fmt="json")

    def processor(_logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        msg = event_dict.pop("event", "")
        out: dict[str, Any] = {
            "time": _iso(time.time()),
            "level": {"warning": "warn", "critical": "fatal", "exception": "error"}.get(
                method, method
            ),
            "service": fmt.service,
            "env": fmt.env,
            "version": fmt.version,
            "event": event_dict.pop("event_name", None) or "log",
            "msg": redact(str(msg)),
        }
        out.update({k: v for k, v in current_context().items() if k in CONTEXT_FIELDS})
        for key, value in event_dict.items():
            out[key] = redact_value(value, key)
        return out

    return processor


# ------------------------------------------------------------------------ middleware

_QUIET_PATHS = ("/health", "/healthz", "/ready", "/live", "/metrics")


class RequestIdMiddleware:
    """Pure-ASGI middleware: accept/mint X-Request-Id, bind it (log + usage
    context), echo it on the response, log one `http.request` line."""

    def __init__(self, app: Any, *, logger_name: str = "bacmr.http") -> None:
        self.app = app
        self.log = logging.getLogger(logger_name)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        inbound = None
        for name, value in scope.get("headers") or []:
            if name.lower() == b"x-request-id":
                inbound = valid_request_id(value)
                break
        rid = inbound or new_request_id()
        status: dict[str, int] = {"code": 500}

        async def send_wrapper(message: dict) -> None:
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 500))
                headers = [
                    (k, v)
                    for k, v in (message.get("headers") or [])
                    if k.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", rid.encode()))
                message = {**message, "headers": headers}
            await send(message)

        from bacmr_common.usage import usage_context

        started = time.perf_counter()
        path = scope.get("path", "")
        token = bind(request_id=rid)
        try:
            with usage_context(request_id=rid):
                await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            self.log.error(
                "unhandled error",
                extra={
                    "event": "http.request",
                    "method": scope.get("method"),
                    "path": path,
                    "status": 500,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": exc,
                },
            )
            raise
        else:
            code = status["code"]
            lvl = logging.DEBUG if path.startswith(_QUIET_PATHS) and code < 400 else logging.INFO
            if code >= 500:
                lvl = logging.ERROR
            self.log.log(
                lvl,
                "%s %s %s",
                scope.get("method"),
                path,
                code,
                extra={
                    "event": "http.request",
                    "method": scope.get("method"),
                    "path": path,
                    "status": code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        finally:
            unbind(token)
