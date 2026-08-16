import uuid

import jsonschema
import pytest
from bacmr_contracts.validate import make_envelope, validate_event

GOOD_MESSAGE = {
    "session_id": str(uuid.uuid4()),
    "message_id": str(uuid.uuid4()),
    "student_id": "stu_1",
    "role": "student",
    "language": "ar",
    "chapter_id": None,
}


def test_valid_payload_passes():
    validate_event("tutor.message.created", GOOD_MESSAGE)


def test_wrong_enum_fails():
    bad = {**GOOD_MESSAGE, "language": "de"}
    with pytest.raises(jsonschema.ValidationError):
        validate_event("tutor.message.created", bad)


def test_bad_uuid_format_fails():
    bad = {**GOOD_MESSAGE, "session_id": "not-a-uuid"}
    with pytest.raises(jsonschema.ValidationError):
        validate_event("tutor.message.created", bad)


def test_extra_field_fails():
    bad = {**GOOD_MESSAGE, "surprise": 1}
    with pytest.raises(jsonschema.ValidationError):
        validate_event("tutor.message.created", bad)


def test_unknown_routing_key_fails():
    with pytest.raises(KeyError):
        validate_event("tutor.message.deleted", GOOD_MESSAGE)


def test_envelope_shape():
    env = make_envelope("tutor.message.created", GOOD_MESSAGE)
    assert set(env) == {"event_id", "routing_key", "occurred_at", "version", "payload"}
    assert env["version"] == 1
    uuid.UUID(env["event_id"])


def test_every_schema_file_is_loadable():
    from bacmr_contracts.validate import SCHEMAS
    assert len(SCHEMAS) >= 8
