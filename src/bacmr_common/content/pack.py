"""Pack manifest v1 and deltas (plan 2026-11-07 §4.4, docs#230).

A pack is one curriculum (class × subject × language) × one audience. Its manifest lists
the chapters in order, each with the published item versions, hashes and sizes, and its
figures. `pack_version` is monotonic per (curriculum, audience, lang): every publish that
changes the pack bumps it by one. The gateway serves the manifest with ETag = pack_version
and `delta?since=` from `diff_manifests`.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Audience, Kind, Lang

PACK_SCHEMA_VERSION = 1

Hash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PackItem(_Strict):
    item_id: uuid.UUID
    kind: Kind
    version: Annotated[int, Field(ge=1)]
    schema_version: Annotated[int, Field(ge=1)] = 1
    content_hash: Hash
    bytes: Annotated[int, Field(ge=0)]
    """Compressed size (gzip) of the item body, for the download size shown first."""


class PackFigure(_Strict):
    figure_id: uuid.UUID
    content_hash: Hash
    bytes: Annotated[int, Field(ge=0)]


class PackChapter(_Strict):
    chapter_id: Annotated[str, Field(min_length=1, max_length=128)]
    position: Annotated[int, Field(ge=0)]
    title: Annotated[str, Field(min_length=1, max_length=500)]
    bundle_hash: Hash
    """Hash of the chapter bundle (the download unit, content-addressed URL `v=`)."""
    items: list[PackItem] = []
    figures: list[PackFigure] = []

    @model_validator(mode="after")
    def _one_per_kind(self) -> PackChapter:
        kinds = [i.kind for i in self.items]
        if len(kinds) != len(set(kinds)):
            raise ValueError(f"chapter {self.chapter_id}: one published item per kind")
        return self


class PackManifest(_Strict):
    model_config = ConfigDict(extra="forbid", frozen=True, title="BacMR content pack manifest v1")

    schema_version: Literal[1]
    curriculum_id: uuid.UUID
    audience: Audience
    lang: Lang
    pack_version: Annotated[int, Field(ge=1)]
    built_at: dt.datetime
    chapters: list[PackChapter]

    @model_validator(mode="after")
    def _unique(self) -> PackManifest:
        ids = [c.chapter_id for c in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("chapter_id repeated in manifest")
        pos = [c.position for c in self.chapters]
        if len(pos) != len(set(pos)):
            raise ValueError("chapter position repeated in manifest")
        items = [i.item_id for c in self.chapters for i in c.items]
        if len(items) != len(set(items)):
            raise ValueError("item_id repeated in manifest")
        return self

    def total_bytes(self, *, figures: bool = False) -> int:
        """Download size of the whole pack (text only unless `figures`)."""
        n = sum(i.bytes for c in self.chapters for i in c.items)
        if figures:
            n += sum(f.bytes for c in self.chapters for f in c.figures)
        return n


class DeltaItem(PackItem):
    chapter_id: Annotated[str, Field(min_length=1, max_length=128)]


class PackDelta(_Strict):
    """What changed between two versions of the same pack."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="BacMR content pack delta v1")

    schema_version: Literal[1]
    curriculum_id: uuid.UUID
    audience: Audience
    lang: Lang
    since: Annotated[int, Field(ge=0)]
    pack_version: Annotated[int, Field(ge=1)]
    upserted: list[DeltaItem] = []
    """Items added or with a new version / hash."""
    removed: list[uuid.UUID] = []
    chapters_changed: list[str] = []
    """Chapters whose order, title, bundle or figures changed (refetch their entry)."""
    chapters_removed: list[str] = []

    @model_validator(mode="after")
    def _order(self) -> PackDelta:
        if self.pack_version < self.since:
            raise ValueError("pack_version must be >= since")
        return self


def diff_manifests(old: PackManifest | None, new: PackManifest) -> PackDelta:
    """Delta from `old` (None = empty pack, since=0) to `new`; same pack identity required."""
    if old is not None:
        if (old.curriculum_id, old.audience, old.lang) != (
            new.curriculum_id,
            new.audience,
            new.lang,
        ):
            raise ValueError("manifests belong to different packs")
        if new.pack_version < old.pack_version:
            raise ValueError("pack_version went backwards")
    old_items = {
        i.item_id: (c.chapter_id, i) for c in (old.chapters if old else []) for i in c.items
    }
    new_items = {i.item_id: (c.chapter_id, i) for c in new.chapters for i in c.items}
    upserted = [
        DeltaItem(chapter_id=ch, **item.model_dump())
        for iid, (ch, item) in new_items.items()
        if iid not in old_items or old_items[iid] != (ch, item)
    ]
    removed = [iid for iid in old_items if iid not in new_items]

    def chapter_key(c: PackChapter) -> tuple:
        return (c.position, c.title, c.bundle_hash, c.figures)

    old_ch = {c.chapter_id: c for c in (old.chapters if old else [])}
    new_ch = {c.chapter_id: c for c in new.chapters}
    changed = [
        cid
        for cid, c in new_ch.items()
        if cid not in old_ch or chapter_key(old_ch[cid]) != chapter_key(c)
    ]
    return PackDelta(
        schema_version=1,
        curriculum_id=new.curriculum_id,
        audience=new.audience,
        lang=new.lang,
        since=old.pack_version if old else 0,
        pack_version=new.pack_version,
        upserted=upserted,
        removed=removed,
        chapters_changed=changed,
        chapters_removed=[cid for cid in old_ch if cid not in new_ch],
    )
