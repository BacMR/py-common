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


def test_document_registered_schema_roundtrip():
    from bacmr_contracts.validate import make_envelope

    env = make_envelope(
        "content.document.registered",
        {
            "document_id": "0e2a2b1c-9a75-4c2e-9d9a-1a2b3c4d5e6f",
            "subject_id": "physique-7c",
            "language": "fr",
            "checksum_sha256": "a" * 64,
        },
    )
    assert env["routing_key"] == "content.document.registered"


def _registered(language: str) -> dict:
    return {
        "document_id": str(uuid.uuid4()),
        "subject_id": "pulaar-1af",
        "language": language,
        "checksum_sha256": "0" * 64,
    }


@pytest.mark.parametrize("language", ["ar", "fr", "en", "ff", "wo", "snk"])
def test_document_registered_accepts_curriculum_languages(language):
    """knowledge#212: the 1AF-2AF national-language readers (Pulaar, Wolof, Soninke)."""
    validate_event("content.document.registered", _registered(language))


def test_document_registered_still_refuses_unknown_language():
    with pytest.raises(jsonschema.ValidationError):
        validate_event("content.document.registered", _registered("de"))


@pytest.mark.parametrize("language", ["ff", "wo", "snk"])
def test_document_ingested_accepts_national_languages(language):
    validate_event(
        "content.document.ingested",
        {
            "document_id": str(uuid.uuid4()),
            "subject_id": "pulaar-1af",
            "language": language,
            "chunk_count": 3,
            "source": "curriculum",
        },
    )
