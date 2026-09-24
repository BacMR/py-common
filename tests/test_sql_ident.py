import pytest
from bacmr_common._sql import validate_schema_ident


@pytest.mark.parametrize("schema", ["content", "tutor", "_private", "a1", "A_B2"])
def test_accepts_valid_identifiers(schema):
    assert validate_schema_ident(schema) == schema


@pytest.mark.parametrize(
    "schema",
    [
        "content; DROP TABLE outbox;--",
        "content.outbox",
        "content outbox",
        "1content",
        "",
        "content'",
    ],
)
def test_rejects_unsafe_identifiers(schema):
    with pytest.raises(ValueError, match="not a valid SQL schema identifier"):
        validate_schema_ident(schema)
