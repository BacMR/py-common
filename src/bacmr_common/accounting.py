"""Per-call accounting labels for every LiteLLM request (ruling R7, BacMR/docs#23).

LiteLLM reads `user` and `metadata.tags` from the OpenAI-compatible request body
and records them in `LiteLLM_SpendLogs` (`end_user`, `request_tags`). Together
with per-service virtual keys this slices spend by service, plan and purpose.
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


@dataclass(frozen=True)
class CallTags:
    service: str
    purpose: str
    request_id: str
    plan_tier: str = "system"
    user_id: str | None = None

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("service", self.service, SERVICES),
            ("plan_tier", self.plan_tier, PLAN_TIERS),
            ("purpose", self.purpose, PURPOSES),
        ):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; expected one of {sorted(allowed)}")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")

    @classmethod
    def system(cls, service: str, purpose: str) -> "CallTags":
        """System/ingestion work: no student, fresh uuid4 request id."""
        return cls(service, purpose, str(uuid.uuid4()))

    def as_request_fields(self) -> dict:
        fields: dict = {
            "metadata": {"tags": [self.service, self.plan_tier, self.request_id, self.purpose]}
        }
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


__all__ = ["PLAN_TIERS", "PURPOSES", "SERVICES", "CallTags"]
