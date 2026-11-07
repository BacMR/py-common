"""Ready-content contracts: content block schema v1 and pack manifests (BacMR/py-common#19).

- `ContentItem` / `parse_item`: the typed-block body of every generated item.
- `PackManifest` / `PackDelta` / `diff_manifests`: versioned offline packs.
- `content_hash` / `compressed_size`: hashes and size budgets.
- `json_schemas()`: the exported JSON Schemas (`schemas/*.json`, shipped as package data);
  `fixtures()`: one AR and one FR item per kind, the shared renderer test set.

Regenerate the committed JSON Schemas after a model change:
`uv run python -m bacmr_common.content.export`.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from .models import (
    BLOCK_TYPES,
    KIND_AUDIENCES,
    SCHEMA_VERSION,
    Block,
    BoardZone,
    Callout,
    ContentItem,
    Definition,
    Example,
    Exercise,
    Figure,
    Formula,
    Heading,
    Paragraph,
    Source,
    SourceRef,
    Table,
    dir_for,
    latex_errors,
    markdown_errors,
    parse_item,
)
from .pack import (
    PACK_SCHEMA_VERSION,
    DeltaItem,
    PackChapter,
    PackDelta,
    PackFigure,
    PackItem,
    PackManifest,
    diff_manifests,
)
from .size import canonical_json, compressed_size, content_hash, raw_size

SCHEMA_FILES = {
    "content-v1.json": ContentItem,
    "pack-manifest-v1.json": PackManifest,
    "pack-delta-v1.json": PackDelta,
}

_PKG = resources.files("bacmr_common.content")


def build_json_schema(model: type) -> dict[str, Any]:
    """JSON Schema (draft 2020-12) for a model, as committed under `schemas/`."""
    schema = model.model_json_schema(mode="validation")
    name = next(k for k, v in SCHEMA_FILES.items() if v is model)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://schemas.bacmr.com/{name}",
        **schema,
    }


def json_schemas() -> dict[str, dict[str, Any]]:
    """The committed JSON Schemas, by file name."""
    return {name: json.loads((_PKG / "schemas" / name).read_text()) for name in SCHEMA_FILES}


def fixtures() -> dict[str, dict[str, Any]]:
    """The shared fixtures (`fixtures/<kind>.<lang>.json`), by file stem."""
    out = {}
    for f in sorted((_PKG / "fixtures").iterdir(), key=lambda p: p.name):
        if f.name.endswith(".json"):
            out[f.name.removesuffix(".json")] = json.loads(f.read_text())
    return out


__all__ = [
    "BLOCK_TYPES",
    "KIND_AUDIENCES",
    "PACK_SCHEMA_VERSION",
    "SCHEMA_FILES",
    "SCHEMA_VERSION",
    "Block",
    "BoardZone",
    "Callout",
    "ContentItem",
    "Definition",
    "DeltaItem",
    "Example",
    "Exercise",
    "Figure",
    "Formula",
    "Heading",
    "PackChapter",
    "PackDelta",
    "PackFigure",
    "PackItem",
    "PackManifest",
    "Paragraph",
    "Source",
    "SourceRef",
    "Table",
    "build_json_schema",
    "canonical_json",
    "compressed_size",
    "content_hash",
    "diff_manifests",
    "dir_for",
    "fixtures",
    "json_schemas",
    "latex_errors",
    "markdown_errors",
    "parse_item",
    "raw_size",
]
