"""Canonical bytes, hashes and compressed sizes of content items (size budgets, plan §4.4).

Canonical JSON = sorted keys, no whitespace, UTF-8 (not ASCII-escaped, so Arabic counts as
it is stored), `None` fields dropped. The same bytes feed `content_hash` (the
`content_hash` column, the manifest and content-addressed bundle URLs) and the sizes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel

Codec = Literal["gzip", "brotli"]


def canonical_json(item: BaseModel | dict[str, Any]) -> bytes:
    data = item.model_dump(mode="json", exclude_none=True) if isinstance(item, BaseModel) else item
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def content_hash(item: BaseModel | dict[str, Any]) -> str:
    """`sha256:<hex>` of the canonical JSON."""
    return "sha256:" + hashlib.sha256(canonical_json(item)).hexdigest()


def raw_size(item: BaseModel | dict[str, Any]) -> int:
    return len(canonical_json(item))


def compressed_size(item: BaseModel | dict[str, Any], codec: Codec = "gzip") -> int:
    """Compressed bytes of the canonical JSON (gzip level 9 by default, deterministic).

    `brotli` (what the gateway prefers on the wire) needs the optional `brotli` package.
    """
    raw = canonical_json(item)
    if codec == "gzip":
        return len(gzip.compress(raw, compresslevel=9, mtime=0))
    if codec == "brotli":
        try:
            import brotli  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("codec='brotli' needs the 'brotli' package installed") from exc
        return len(brotli.compress(raw, quality=11))
    raise ValueError(f"unknown codec {codec!r}")
