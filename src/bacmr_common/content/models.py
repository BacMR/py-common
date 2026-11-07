"""Content block schema v1: the body of every generated content item (BacMR/py-common#19).

One schema for the writer and quality gate (knowledge), the gateway (validation), the
Flutter and React renderers and the PDF templates (plan 2026-11-07 §4.1, docs#230).

An item is an envelope (`kind`, `lang`, `audience`, `schema_version`, `blocks`, `sources`)
around an ordered list of typed blocks, discriminated by `type`. Every block carries
`lang` and `dir`; `dir` must agree with `lang` (`ar` is `rtl`, the others `ltr`), so an
Arabic paragraph inside a French item, or a French term inside an Arabic one, is marked
explicitly. LaTeX (`formula.latex` and `$...$` inside Markdown text) is always laid out
left to right by the renderers whatever the block's `dir`.

Rules that JSON Schema cannot express (balanced LaTeX braces, `\\begin`/`\\end` pairs,
`$` pairing, table widths, kind/block rules, the audience rule) are enforced here; the
exported JSON Schema (`schemas/content-v1.json`) carries the structure, enums, the
`lang`/`dir` pairing, forbidden LaTeX commands and the kind/audience matrix.

Changes inside v1 are additive only (new optional field, widened enum, new block type is
NOT additive for old renderers and needs `schema_version` 2).
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

Lang = Literal["ar", "fr", "en"]
Dir = Literal["rtl", "ltr"]
Kind = Literal[
    "lesson", "teacher_notes", "board", "exercises", "test_bank", "recap", "parent_brief"
]
Audience = Literal["student", "teacher", "parent"]
Difficulty = Literal["easy", "medium", "hard"]
ExerciseType = Literal["mcq", "multi", "true_false", "short", "numeric", "open"]
CalloutVariant = Literal["warning", "common_mistake", "tip", "note", "objective", "prerequisite"]
SourceKind = Literal["textbook", "teacher_guide", "exam", "materials", "other"]

RTL_LANGS: frozenset[str] = frozenset({"ar"})

# Which audiences each kind may be written for (plan §4.1 table).
KIND_AUDIENCES: dict[str, frozenset[str]] = {
    "lesson": frozenset({"student", "teacher"}),
    "teacher_notes": frozenset({"teacher"}),
    "board": frozenset({"teacher"}),
    "exercises": frozenset({"student", "teacher"}),
    "test_bank": frozenset({"student", "teacher"}),
    "recap": frozenset({"student", "teacher", "parent"}),
    "parent_brief": frozenset({"parent"}),
}


def dir_for(lang: str) -> str:
    """The text direction a block in `lang` must declare."""
    return "rtl" if lang in RTL_LANGS else "ltr"


# --- LaTeX and Markdown checks --------------------------------------------------------

# Commands that read files, define macros or emit raw output: never valid in content,
# whatever renderer (KaTeX, MathJax, server-side TeX for the PDF) consumes it.
FORBIDDEN_LATEX = (
    "input",
    "include",
    "write",
    "immediate",
    "openout",
    "openin",
    "read",
    "def",
    "gdef",
    "edef",
    "xdef",
    "let",
    "newcommand",
    "renewcommand",
    "providecommand",
    "DeclareMathOperator",
    "catcode",
    "csname",
    "url",
    "href",
    "includegraphics",
    "special",
)
_FORBIDDEN_RE = re.compile(r"\\(?:" + "|".join(FORBIDDEN_LATEX) + r")(?![A-Za-z])")
# The same rule as a JSON Schema `pattern` (ECMA-262 compatible).
FORBIDDEN_LATEX_PATTERN = r"^(?![\s\S]*\\(?:" + "|".join(FORBIDDEN_LATEX) + r")(?![A-Za-z]))"
_ENV_RE = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")


def latex_errors(latex: str) -> list[str]:
    """Why `latex` is not a valid formula body; empty list when it is.

    Structural checks only (no TeX engine): non-empty, no `$` delimiters (the block is the
    delimiter), no forbidden command, balanced `{}` (ignoring `\\{`/`\\}`), matching
    `\\begin{env}`/`\\end{env}` and `\\left`/`\\right` pairs.
    """
    errors: list[str] = []
    if not latex.strip():
        return ["empty formula"]
    if "$" in latex.replace("\\$", ""):
        errors.append("formula must not contain $ delimiters")
    if m := _FORBIDDEN_RE.search(latex):
        errors.append(f"forbidden LaTeX command {m.group(0)}")
    depth = 0
    i = 0
    while i < len(latex):
        ch = latex[i]
        if ch == "\\":
            i += 2  # skip the escaped char (\{ \} \\ ...)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append("unbalanced braces: unexpected }")
                break
        i += 1
    if depth > 0:
        errors.append("unbalanced braces: missing }")
    stack: list[str] = []
    for kind, env in _ENV_RE.findall(latex):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            errors.append(f"unmatched \\end{{{env}}}")
            break
    else:
        if stack:
            errors.append(f"unclosed \\begin{{{stack[-1]}}}")
    lefts = len(re.findall(r"\\left(?![A-Za-z])", latex))
    rights = len(re.findall(r"\\right(?![A-Za-z])", latex))
    if lefts != rights:
        errors.append("unbalanced \\left / \\right")
    return errors


_HTML_RE = re.compile(r"<\s*/?\s*[A-Za-z!]")
_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def markdown_errors(text: str) -> list[str]:
    """Why `text` is outside the Markdown subset; empty list when it is inside.

    Allowed: paragraphs, `**bold**`, `*italic*`, `-`/`1.` lists, `` `code` ``, links-free
    text, and inline math `$...$` (each segment must pass `latex_errors`). Not allowed:
    raw HTML, images (use a `figure` block), `#` headings (use a `heading` block), display
    math `$$` (use a `formula` block).
    """
    errors: list[str] = []
    if _HTML_RE.search(text):
        errors.append("raw HTML is not allowed")
    if "![" in text:
        errors.append("images are not allowed (use a figure block)")
    if any(line.lstrip().startswith("#") for line in text.splitlines()):
        errors.append("Markdown headings are not allowed (use a heading block)")
    if "$$" in text:
        errors.append("display math $$ is not allowed (use a formula block)")
        return errors
    parts = _UNESCAPED_DOLLAR.split(text)
    if len(parts) % 2 == 0:
        errors.append("unpaired $ in inline math")
        return errors
    for segment in parts[1::2]:
        errors.extend(f"inline math: {e}" for e in latex_errors(segment))
    return errors


def _check_md(value: str, field: str) -> None:
    if errs := markdown_errors(value):
        raise ValueError(f"{field}: " + "; ".join(errs))


# --- Blocks ---------------------------------------------------------------------------

_LANG_DIR_RULE = {
    "if": {"properties": {"lang": {"const": "ar"}}},
    "then": {"properties": {"dir": {"const": "rtl"}}},
    "else": {"properties": {"dir": {"const": "ltr"}}},
}

Text = Annotated[str, Field(min_length=1, max_length=20_000)]
Line = Annotated[str, Field(min_length=1, max_length=2_000)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Block(_Strict):
    """Fields every block carries."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra={"allOf": [_LANG_DIR_RULE]}
    )

    lang: Lang
    dir: Dir
    id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")] | None = (
        None
    )
    """Optional anchor, stable across versions (deep links, refine targets, attempts)."""

    @model_validator(mode="after")
    def _dir_matches_lang(self) -> _Block:
        if self.dir != dir_for(self.lang):
            raise ValueError(f"dir must be {dir_for(self.lang)!r} for lang {self.lang!r}")
        return self


class Heading(_Block):
    type: Literal["heading"]
    level: Annotated[int, Field(ge=1, le=3)]
    text: Line


class Paragraph(_Block):
    """Markdown subset (see `markdown_errors`), inline math as `$...$`."""

    type: Literal["paragraph"]
    text: Text

    @model_validator(mode="after")
    def _md(self) -> Paragraph:
        _check_md(self.text, "text")
        return self


class Definition(_Block):
    type: Literal["definition"]
    term: Line
    text: Text

    @model_validator(mode="after")
    def _md(self) -> Definition:
        _check_md(self.term, "term")
        _check_md(self.text, "text")
        return self


class Formula(_Block):
    """One display formula. `latex` has no `$` delimiters and is always rendered LTR;
    `lang`/`dir` apply to the caption and to the block's placement in the page."""

    type: Literal["formula"]
    latex: Annotated[
        str,
        # Pydantic's Rust regex has no look-ahead: the pattern is for JSON Schema consumers,
        # `latex_errors` enforces the same rule here.
        Field(
            min_length=1, max_length=4_000, json_schema_extra={"pattern": FORBIDDEN_LATEX_PATTERN}
        ),
    ]
    caption: Line | None = None

    @model_validator(mode="after")
    def _latex(self) -> Formula:
        if errs := latex_errors(self.latex):
            raise ValueError("latex: " + "; ".join(errs))
        if self.caption:
            _check_md(self.caption, "caption")
        return self


class Example(_Block):
    type: Literal["example"]
    title: Line | None = None
    statement: Text
    solution_steps: Annotated[list[Text], Field(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def _md(self) -> Example:
        _check_md(self.statement, "statement")
        for i, s in enumerate(self.solution_steps):
            _check_md(s, f"solution_steps[{i}]")
        return self


class Figure(_Block):
    """A figure crop knowledge already stores (`content.figures`), by id."""

    type: Literal["figure"]
    figure_id: uuid.UUID
    caption: Line | None = None
    alt: Line | None = None


class Table(_Block):
    type: Literal["table"]
    caption: Line | None = None
    header: Annotated[list[Line], Field(min_length=1, max_length=12)]
    rows: Annotated[list[list[str]], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def _shape(self) -> Table:
        width = len(self.header)
        for i, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"rows[{i}] has {len(row)} cells, header has {width}")
            for j, cell in enumerate(row):
                if cell:
                    _check_md(cell, f"rows[{i}][{j}]")
        return self


class Callout(_Block):
    type: Literal["callout"]
    variant: CalloutVariant
    title: Line | None = None
    text: Text

    @model_validator(mode="after")
    def _md(self) -> Callout:
        _check_md(self.text, "text")
        return self


class BoardZone(_Block):
    """One zone of a whiteboard plan: what the teacher writes there, in order."""

    type: Literal["board_zone"]
    zone: Literal["left", "center", "right", "top", "bottom", "full"]
    title: Line
    items: Annotated[list[Text], Field(min_length=1, max_length=40)]
    minutes: Annotated[int, Field(ge=1, le=240)] | None = None

    @model_validator(mode="after")
    def _md(self) -> BoardZone:
        for i, s in enumerate(self.items):
            _check_md(s, f"items[{i}]")
        return self


class Exercise(_Block):
    """A practice or test item. `correct_choices` are 0-based indexes into `choices`.

    For `true_false`, choices are the two localized labels and `correct_choices` has one
    index. `answer` is the final answer in words or a value; `solution_steps` the worked
    solution. In student test banks the gateway hides `answer`, `correct_choices` and
    `solution_steps` until the attempt is submitted (plan §4.1)."""

    type: Literal["exercise"]
    id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")]
    exercise_type: ExerciseType
    stem: Text
    choices: Annotated[list[Line], Field(min_length=2, max_length=8)] | None = None
    correct_choices: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)] | None = None
    answer: Text
    solution_steps: Annotated[list[Text], Field(min_length=1, max_length=40)]
    difficulty: Difficulty
    concept_ids: Annotated[list[Annotated[str, Field(min_length=1)]], Field(max_length=20)] = []
    points: Annotated[float, Field(gt=0, le=100)]
    figure_ids: list[uuid.UUID] = []

    @model_validator(mode="after")
    def _choices(self) -> Exercise:
        choice_types = {"mcq", "multi", "true_false"}
        if self.exercise_type in choice_types:
            if not self.choices or not self.correct_choices:
                raise ValueError(f"{self.exercise_type} needs choices and correct_choices")
            if self.exercise_type == "true_false" and len(self.choices) != 2:
                raise ValueError("true_false needs exactly 2 choices")
            if self.exercise_type in {"mcq", "true_false"} and len(self.correct_choices) != 1:
                raise ValueError(f"{self.exercise_type} needs exactly one correct choice")
            if any(c >= len(self.choices) for c in self.correct_choices):
                raise ValueError("correct_choices index out of range")
            if len(set(self.correct_choices)) != len(self.correct_choices):
                raise ValueError("correct_choices has duplicates")
        elif self.choices is not None or self.correct_choices is not None:
            raise ValueError(f"{self.exercise_type} takes no choices")
        _check_md(self.stem, "stem")
        _check_md(self.answer, "answer")
        for i, c in enumerate(self.choices or []):
            _check_md(c, f"choices[{i}]")
        for i, s in enumerate(self.solution_steps):
            _check_md(s, f"solution_steps[{i}]")
        return self


class SourceRef(_Strict):
    """Where content came from: a document and its pages (chunk/figure ids optional)."""

    document_id: uuid.UUID
    source_kind: SourceKind
    pages: Annotated[list[Annotated[int, Field(ge=1)]], Field(max_length=500)] = []
    chunk_ids: Annotated[list[str], Field(max_length=500)] = []
    figure_ids: Annotated[list[uuid.UUID], Field(max_length=200)] = []


class Source(_Block):
    """An inline citation shown to the reader ("Manuel 7AS, p. 42")."""

    type: Literal["source"]
    document_id: uuid.UUID
    source_kind: SourceKind
    pages: Annotated[list[Annotated[int, Field(ge=1)]], Field(max_length=500)] = []
    label: Line


Block = Annotated[
    Heading
    | Paragraph
    | Definition
    | Formula
    | Example
    | Figure
    | Table
    | Callout
    | BoardZone
    | Exercise
    | Source,
    Field(discriminator="type"),
]

BLOCK_TYPES: tuple[str, ...] = (
    "heading",
    "paragraph",
    "definition",
    "formula",
    "example",
    "figure",
    "table",
    "callout",
    "board_zone",
    "exercise",
    "source",
)


# --- Item envelope --------------------------------------------------------------------


def _kind_audience_schema(schema: dict[str, Any]) -> None:
    schema["allOf"] = [
        _LANG_DIR_RULE,
        *(
            {
                "if": {"properties": {"kind": {"const": kind}}},
                "then": {"properties": {"audience": {"enum": sorted(auds)}}},
            }
            for kind, auds in KIND_AUDIENCES.items()
        ),
    ]


class ContentItem(_Strict):
    """One generated item: chapter × kind × lang × audience (one version of it)."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        title="BacMR content item v1",
        json_schema_extra=_kind_audience_schema,
    )

    schema_version: Literal[1]
    kind: Kind
    lang: Lang
    dir: Dir
    audience: Audience
    title: Line
    chapter_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    blocks: Annotated[list[Block], Field(min_length=1, max_length=2_000)]
    sources: Annotated[list[SourceRef], Field(max_length=200)] = []

    @model_validator(mode="after")
    def _rules(self) -> ContentItem:
        if self.dir != dir_for(self.lang):
            raise ValueError(f"dir must be {dir_for(self.lang)!r} for lang {self.lang!r}")
        if self.audience not in KIND_AUDIENCES[self.kind]:
            raise ValueError(f"kind {self.kind!r} is not written for audience {self.audience!r}")
        types = [b.type for b in self.blocks]
        if self.kind in {"exercises", "test_bank"} and "exercise" not in types:
            raise ValueError(f"{self.kind} needs at least one exercise block")
        if self.kind == "board" and "board_zone" not in types:
            raise ValueError("board needs at least one board_zone block")
        if self.kind != "board" and "board_zone" in types:
            raise ValueError("board_zone blocks belong to board items only")
        ids = [b.id for b in self.blocks if b.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("block ids must be unique within an item")
        # Audience rule (plan §4.1, docs#228): student items never draw on a teacher guide.
        if self.audience != "teacher":
            guide = [s for s in self.sources if s.source_kind == "teacher_guide"]
            guide += [
                b for b in self.blocks if b.type == "source" and b.source_kind == "teacher_guide"
            ]
            if guide:
                raise ValueError(f"{self.audience} items must not cite a teacher_guide source")
        return self


def parse_item(data: dict[str, Any] | str | bytes) -> ContentItem:
    """Validate a dict or a JSON document into a `ContentItem` (raises `ValidationError`)."""
    if isinstance(data, str | bytes):
        return ContentItem.model_validate_json(data)
    return ContentItem.model_validate(data)
