"""Identifier validation for schema-qualified SQL built via string interpolation.

`sqlalchemy.text()` binds *values*, never identifiers (schema/table/column names)
-- there is no parameterized-query mechanism for those, so callers that need a
schema-qualified table name are stuck building it into the SQL string. The
schema names actually used (`"tutor"`, `"content"`) are hardcoded literals at
every call site today, never end-user input, but this validates the identifier
shape regardless so a future caller can't turn this into a real injection
vector by passing through unsanitized input.
"""

import re

_VALID_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_schema_ident(schema: str) -> str:
    """Return `schema` unchanged if it's a safe-to-interpolate SQL identifier,
    otherwise raise ValueError."""
    if not _VALID_IDENT.match(schema):
        raise ValueError(f"not a valid SQL schema identifier: {schema!r}")
    return schema
