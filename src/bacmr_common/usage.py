"""Request-local accounting context; shared clients never store student identity.

Only authenticated service boundaries may populate actor fields. Metadata travels
through LiteLLM, while our internal headers propagate across tutor -> knowledge.

Two independent request-local things live here:

* `usage_context` — WHO the tokens are for (identity/attribution metadata sent
  to LiteLLM on every call).
* `usage_tally` / `TurnUsage` — HOW MANY tokens a unit of work actually burned,
  summed over EVERY model call it makes. LiteLLM's spend ledger stays the
  system of record for cost; this is the same-process total a service needs
  *during* the request (e.g. the tutor's per-student daily budget), which it
  cannot get by asking the ledger mid-turn.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from uuid import uuid4

_context: ContextVar[dict | None] = ContextVar("llm_usage", default=None)
_FIELDS = (
    "consumption",
    "actor",
    "operation",
    "request_id",
    "session_id",
    "job_id",
    "document_id",
    "stage",
)

# One shared definition of the char-count fallback so "4 chars ~ 1 token" is
# not re-invented (with a different constant) in every caller that needs it.
_CHARS_PER_TOKEN = 4


@contextmanager
def usage_context(**fields):
    values = dict(_context.get() or {})
    values.update({k: str(v) for k, v in fields.items() if k in _FIELDS and v is not None})
    token = _context.set(values)
    try:
        yield
    finally:
        _context.reset(token)


def current_usage_field(name: str) -> str | None:
    """Read one field out of the ambient accounting context.

    Lets a stage stamp its own DB rows (e.g. content.enrich_skips.run_id) with
    the SAME run identifier that reaches LiteLLM_SpendLogs as
    metadata->spend_logs_metadata->>'job_id', so ledger spend and what the run
    actually produced can be joined without guessing at time windows.
    """
    return (_context.get() or {}).get(name)


def gateway_body(body: dict, *, service: str, call_id: str, attempt: int) -> dict:
    ctx = dict(_context.get() or {})
    consumption = ctx.get("consumption", "system")
    actor = ctx.get("actor", "system:" + service)
    operation = ctx.get("operation", body.get("model", "unknown"))
    metadata = {
        **body.get("metadata", {}),
        **ctx,
        "consumption": consumption,
        "service": service,
        "operation": operation,
        "call_id": call_id,
        "attempt": attempt,
    }
    metadata["tags"] = list(
        dict.fromkeys(
            [
                *metadata.get("tags", []),
                "consumption:" + consumption,
                "service:" + service,
                "operation:" + operation,
                *(["stage:" + ctx["stage"]] if "stage" in ctx else []),
            ]
        )
    )
    # Only this native container preserves arbitrary fields in SpendLogs.
    metadata["spend_logs_metadata"] = {
        **body.get("metadata", {}).get("spend_logs_metadata", {}),
        **{k: v for k, v in metadata.items() if k not in ("tags", "spend_logs_metadata")},
    }
    # LiteLLM's native trace/session field groups all calls for one tutor turn/job.
    trace_id = ctx.get("request_id") or ctx.get("job_id") or call_id
    return {
        **body,
        "user": body.get("user", actor),
        "metadata": metadata,
        "litellm_trace_id": trace_id,
    }


def propagation_headers() -> dict[str, str]:
    return {
        "X-Bacmr-Usage-" + k.replace("_", "-"): str(v)
        for k, v in (_context.get() or {}).items()
        if k in _FIELDS
    }


def propagated_context(headers) -> dict[str, str]:
    # Called only behind the internal bearer-auth dependency. Bound sizes so an
    # authenticated malformed request cannot balloon per-call logging metadata.
    return {
        k: headers.get("X-Bacmr-Usage-" + k.replace("_", "-"))[:256]
        for k in _FIELDS
        if headers.get("X-Bacmr-Usage-" + k.replace("_", "-"))
    }


def new_call_id() -> str:
    return str(uuid4())


def estimate_tokens(text_or_chars) -> int:
    """The LAST-RESORT stand-in for a provider count: ~4 characters per token.

    Accepts a string or an already-counted character total. Whatever consumes
    this must label the result an ESTIMATE wherever it is stored — see
    AGENTS.md "Mandatory LLM accounting": a missing provider counter is
    *unknown*, and an estimate may never be passed off as a measurement.
    """
    chars = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars)
    return max(chars, 0) // _CHARS_PER_TOKEN


@dataclass
class TurnUsage:
    """Token usage accumulated across every model call of ONE unit of work.

    The four counters are deliberately never collapsed into a single number:

    * `prompt_tokens`/`completion_tokens` — PROVIDER-REPORTED, from a reply's
      `usage` block. The only real measurements here.
    * `estimated_tokens` — the char-heuristic stand-in, and ONLY for calls whose
      reply carried no usage block.
    * `unmeasured_calls` — how many calls that was. `> 0` means `total_tokens`
      is a LOWER BOUND with a guess mixed in, not a measurement.

    Keeping them apart is what lets a consumer store the total AND still say
    where it came from (AGENTS.md: missing counters are unknown, never a
    fabricated zero). A `TurnUsage` is mutated in place by `record`, so a
    holder can read the running total at any point; `since()` gives the delta
    contributed by one stage (e.g. just the streamed reply).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    measured_calls: int = 0
    estimated_tokens: int = 0
    unmeasured_calls: int = 0

    @property
    def measured_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.measured_tokens + self.estimated_tokens

    def record(self, usage: dict | None, *, estimate: int = 0) -> None:
        """Fold one model call in. `usage` is the provider's block verbatim.

        A block carrying at least one integer counter counts as MEASURED, and a
        counter that is absent *within* such a block is genuinely zero rather
        than unknown — an embeddings reply has no completion side at all. No
        block (streamed reply with usage suppressed, a provider that omits it,
        a call that died mid-flight) counts as UNMEASURED and contributes only
        the caller's labelled estimate.
        """
        prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(prompt, int) or isinstance(completion, int):
            self.prompt_tokens += prompt if isinstance(prompt, int) else 0
            self.completion_tokens += completion if isinstance(completion, int) else 0
            self.measured_calls += 1
            return
        self.unmeasured_calls += 1
        self.estimated_tokens += max(int(estimate), 0)

    def note_unmeasured_call(self) -> None:
        """Flag a model call this process CANNOT see (e.g. the embedding and
        rerank calls another service makes while serving our retrieve request).
        No estimate is invented for it — the honest record is "there were N
        calls whose tokens we do not know"."""
        self.unmeasured_calls += 1

    def snapshot(self) -> "TurnUsage":
        return replace(self)

    def since(self, earlier: "TurnUsage") -> "TurnUsage":
        """Delta against an earlier `snapshot()` — what one stage added."""
        return TurnUsage(
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=self.completion_tokens - earlier.completion_tokens,
            measured_calls=self.measured_calls - earlier.measured_calls,
            estimated_tokens=self.estimated_tokens - earlier.estimated_tokens,
            unmeasured_calls=self.unmeasured_calls - earlier.unmeasured_calls,
        )


_tally: ContextVar[TurnUsage | None] = ContextVar("llm_usage_tally", default=None)


@contextmanager
def usage_tally(tally: TurnUsage | None = None):
    """Collect the token usage of every model call made inside this block.

    A ContextVar rather than a parameter for the same reason `usage_context`
    is one: the calls happen three or four layers below the code that needs
    the total (tutor turn -> understanding/sufficiency -> LLMClient), and
    threading an accumulator through every shared, duck-typed client signature
    would make every service pay for one service's budget feature. Nested
    blocks are independent tallies; an outer one does NOT also see the inner
    calls, so a caller always knows exactly which calls it is charging for.
    """
    tally = tally if tally is not None else TurnUsage()
    token = _tally.set(tally)
    try:
        yield tally
    finally:
        push_back(token)


def push_tally(tally: TurnUsage):
    """Non-`with` form of `usage_tally`, for async generators: the set and the
    reset happen in two different resumptions of the same generator, which a
    `with` spanning a `yield` cannot express safely. Pair with `push_back`."""
    return _tally.set(tally)


def push_back(token) -> None:
    try:
        _tally.reset(token)
    except ValueError:
        # The token was created in a different Context — possible when the
        # producer is an async generator whose first and last resumption came
        # from different tasks. Dropping the reset leaks nothing durable (the
        # tally is per-Context and dies with it) and must never take a request
        # down over bookkeeping.
        _tally.set(None)


def record_usage(usage: dict | None, *, estimate: int = 0) -> None:
    """Fold one model call into the active tally, if any. A no-op outside a
    `usage_tally` block, so shared clients can call it unconditionally."""
    tally = _tally.get()
    if tally is not None:
        tally.record(usage, estimate=estimate)


def active_tally() -> TurnUsage | None:
    return _tally.get()
