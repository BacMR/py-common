"""Event contract validation. Schemas are the source of truth, shipped as package data
under bacmr_contracts/schemas so they survive a non-editable (wheel) install."""

import datetime as dt
import json
import uuid
from importlib import resources

from jsonschema import Draft202012Validator, FormatChecker

_SCHEMA_DIR = resources.files("bacmr_contracts") / "schemas"


def _load() -> dict[str, dict]:
    out = {}
    for f in sorted(_SCHEMA_DIR.iterdir(), key=lambda p: p.name):
        if f.name.endswith(".v1.json"):
            key = f.name.removesuffix(".v1.json")
            out[key] = json.loads(f.read_text())
    return out


SCHEMAS = _load()
if not SCHEMAS:
    raise RuntimeError("bacmr_contracts: no schemas found — broken installation")


def _validate(instance: dict, schema: dict) -> None:
    # Draft202012Validator + FormatChecker: plain jsonschema.validate() ignores `format`
    # (e.g. uuid, date-time) under draft 2020-12 unless a format checker is passed in.
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)


def validate_event(routing_key: str, payload: dict) -> None:
    _validate(payload, SCHEMAS[routing_key])


def make_envelope(routing_key: str, payload: dict) -> dict:
    validate_event(routing_key, payload)
    env = {
        "event_id": str(uuid.uuid4()),
        "routing_key": routing_key,
        "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
        "version": 1,
        "payload": payload,
    }
    _validate(env, SCHEMAS["envelope"])
    return env
