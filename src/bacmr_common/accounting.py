"""Per-call accounting labels for every LiteLLM request (ruling R7, BacMR/docs#23).

LiteLLM reads `user` and `metadata.tags` from the OpenAI-compatible request body
and records them in `LiteLLM_SpendLogs` (`end_user`, `request_tags`). Together
with per-service virtual keys this slices spend by service, plan and purpose.
`task` (BacMR/docs#55) is the explicit Spend-page bucket, emitted as a fifth
`task:<label>` tag so the positional four stay unchanged.
"""

import uuid
from dataclasses import dataclass

SERVICES = frozenset({"gateway", "tutor-api", "knowledge-api", "knowledge-worker", "scraper"})
PLAN_TIERS = frozenset({"free", "standard", "premium", "system"})
PURPOSES = frozenset(
    {
        "chat",
        "understanding",
        "sufficiency",
        "summary",
        "retrieve-rescue",
        "ingest-text",
        "ingest-vision",
        "ingest-enrich",
        "embed",
        "rerank",
        "moderation",
    }
)

# Spend-page cost buckets (BacMR/docs#55). Orthogonal to `purpose`: e.g. purpose
# `ingest-vision` covers both `ocr-vision-page` and `figure-description`, and
# `chat` covers `student-question`, `evals` and `quiz-generation`.
TASKS = frozenset(
    {
        "student-question",
        "query-understanding",
        "sufficiency-check",
        "conversation-summary",
        "retrieval-rerank",
        "embedding",
        "ocr-vision-page",
        "text-extraction",
        "figure-description",
        "chunk-enrichment",
        "bridge-questions",
        "translation",
        "moderation",
        "evals",
        "quiz-generation",
    }
)
TASK_TAG_PREFIX = "task:"


@dataclass(frozen=True)
class CallTags:
    service: str
    purpose: str
    request_id: str
    plan_tier: str = "system"
    user_id: str | None = None
    task: str | None = None

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("service", self.service, SERVICES),
            ("plan_tier", self.plan_tier, PLAN_TIERS),
            ("purpose", self.purpose, PURPOSES),
        ):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; expected one of {sorted(allowed)}")
        if self.task is not None and self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; expected one of {sorted(TASKS)}")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")

    @classmethod
    def system(cls, service: str, purpose: str, *, task: str | None = None) -> "CallTags":
        """System/ingestion work: no student, fresh uuid4 request id."""
        return cls(service, purpose, str(uuid.uuid4()), task=task)

    def as_request_fields(self) -> dict:
        tags = [self.service, self.plan_tier, self.request_id, self.purpose]
        if self.task is not None:
            tags.append(TASK_TAG_PREFIX + self.task)
        fields: dict = {"metadata": {"tags": tags}}
        if self.user_id is not None:
            fields = {"user": self.user_id, **fields}
        return fields

    def apply_to_body(self, body: dict) -> dict:
        """Return a copy of `body` carrying these labels. Only `metadata.tags`
        is replaced; every other caller-provided metadata key is preserved."""
        fields = self.as_request_fields()
        out = {**body, **{k: v for k, v in fields.items() if k != "metadata"}}
        out["metadata"] = {**(body.get("metadata") or {}), **fields["metadata"]}
        return out


__all__ = ["PLAN_TIERS", "PURPOSES", "SERVICES", "TASKS", "TASK_TAG_PREFIX", "CallTags"]
