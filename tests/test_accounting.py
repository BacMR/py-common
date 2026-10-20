import uuid

import pytest

from bacmr_common.accounting import PLAN_TIERS, PURPOSES, SERVICES, CallTags


def test_field_shape_exact_order():
    t = CallTags("tutor-api", "chat", "req-1", plan_tier="premium", user_id="p-9")
    assert t.as_request_fields() == {
        "user": "p-9",
        "metadata": {"tags": ["tutor-api", "premium", "req-1", "chat"]},
    }


def test_user_omitted_when_none():
    fields = CallTags("scraper", "embed", "r").as_request_fields()
    assert "user" not in fields
    assert fields["metadata"]["tags"] == ["scraper", "system", "r", "embed"]


def test_system_mints_uuid4():
    a = CallTags.system("knowledge-worker", "ingest-text")
    b = CallTags.system("knowledge-worker", "ingest-text")
    assert a.plan_tier == "system" and a.user_id is None
    assert uuid.UUID(a.request_id).version == 4
    assert a.request_id != b.request_id


def test_frozen():
    with pytest.raises(AttributeError):
        CallTags.system("gateway", "chat").service = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        {"service": "nope"},
        {"purpose": "nope"},
        {"plan_tier": "gold"},
        {"request_id": ""},
    ],
)
def test_validation(kw):
    base = {"service": "gateway", "purpose": "chat", "request_id": "r"}
    with pytest.raises(ValueError):
        CallTags(**{**base, **kw})


def test_enums():
    assert "knowledge-api" in SERVICES and "system" in PLAN_TIERS and "rerank" in PURPOSES
    assert len(SERVICES) == 5 and len(PLAN_TIERS) == 4 and len(PURPOSES) == 11
