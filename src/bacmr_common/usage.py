"""Request-local accounting context; shared clients never store student identity.

Only authenticated service boundaries may populate actor fields. Metadata travels
through LiteLLM, while our internal headers propagate across tutor -> knowledge.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

_context: ContextVar[dict | None] = ContextVar("llm_usage", default=None)
_FIELDS = ("consumption", "actor", "operation", "request_id", "session_id", "job_id",
           "document_id", "stage")


@contextmanager
def usage_context(**fields):
    values = dict(_context.get() or {})
    values.update({k: str(v) for k, v in fields.items() if k in _FIELDS and v is not None})
    token = _context.set(values)
    try:
        yield
    finally:
        _context.reset(token)


def gateway_body(body: dict, *, service: str, call_id: str, attempt: int) -> dict:
    ctx = dict(_context.get() or {})
    consumption = ctx.get("consumption", "system")
    actor = ctx.get("actor", "system:" + service)
    operation = ctx.get("operation", body.get("model", "unknown"))
    metadata = {**body.get("metadata", {}), **ctx,
                "consumption": consumption, "service": service, "operation": operation,
                "call_id": call_id, "attempt": attempt}
    metadata["tags"] = list(dict.fromkeys([
        *metadata.get("tags", []), "consumption:" + consumption, "service:" + service,
        "operation:" + operation,
        *(["stage:" + ctx["stage"]] if "stage" in ctx else [])]))
    # Only this native container preserves arbitrary fields in SpendLogs.
    metadata["spend_logs_metadata"] = {
        **body.get("metadata", {}).get("spend_logs_metadata", {}),
        **{k: v for k, v in metadata.items() if k not in ("tags", "spend_logs_metadata")}}
    # LiteLLM's native trace/session field groups all calls for one tutor turn/job.
    trace_id = ctx.get("request_id") or ctx.get("job_id") or call_id
    return {**body, "user": actor, "metadata": metadata, "litellm_trace_id": trace_id}


def propagation_headers() -> dict[str, str]:
    return {"X-Bacmr-Usage-" + k.replace("_", "-"): str(v)
            for k, v in (_context.get() or {}).items() if k in _FIELDS}


def propagated_context(headers) -> dict[str, str]:
    # Called only behind the internal bearer-auth dependency. Bound sizes so an
    # authenticated malformed request cannot balloon per-call logging metadata.
    return {k: headers.get("X-Bacmr-Usage-" + k.replace("_", "-"))[:256]
            for k in _FIELDS if headers.get("X-Bacmr-Usage-" + k.replace("_", "-"))}


def new_call_id() -> str:
    return str(uuid4())
