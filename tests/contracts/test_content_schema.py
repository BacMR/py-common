"""Content block schema v1 + pack manifests (BacMR/py-common#19).

Every rule is checked twice where JSON Schema can express it: through the pydantic models
(what knowledge and the gateway run) and through the exported JSON Schema (what the Flutter
and React renderers and non-Python consumers run)."""

import copy
import datetime as dt
import json
import uuid

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from bacmr_common.content import (
    BLOCK_TYPES,
    KIND_AUDIENCES,
    SCHEMA_FILES,
    ContentItem,
    PackChapter,
    PackItem,
    PackManifest,
    build_json_schema,
    compressed_size,
    content_hash,
    diff_manifests,
    fixtures,
    json_schemas,
    latex_errors,
    markdown_errors,
    parse_item,
    raw_size,
)

FIXTURES = fixtures()
SCHEMAS = json_schemas()
ITEM_VALIDATOR = Draft202012Validator(SCHEMAS["content-v1.json"], format_checker=FormatChecker())


def schema_ok(item: dict) -> bool:
    return ITEM_VALIDATOR.is_valid(item)


def lesson(lang: str = "ar") -> dict:
    return copy.deepcopy(FIXTURES[f"lesson.{lang}"])


def block_of(item: dict, type_: str) -> dict:
    return next(b for b in item["blocks"] if b["type"] == type_)


# --- schema files and fixtures --------------------------------------------------------


def test_committed_schemas_match_models():
    for name, model in SCHEMA_FILES.items():
        assert SCHEMAS[name] == json.loads(json.dumps(build_json_schema(model))), (
            f"{name} drifted: run `uv run python -m bacmr_common.content.export`"
        )
        Draft202012Validator.check_schema(SCHEMAS[name])


def test_one_ar_and_one_fr_fixture_per_kind():
    expected = {f"{k}.{lang}" for k in KIND_AUDIENCES for lang in ("ar", "fr")}
    assert set(FIXTURES) == expected


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_validates_in_models_and_schema(name):
    data = FIXTURES[name]
    item = parse_item(data)
    assert item.kind == name.split(".")[0]
    assert item.lang == name.split(".")[1]
    errors = list(ITEM_VALIDATOR.iter_errors(data))
    assert not errors, [e.message for e in errors]
    # round trip: the model dump is the fixture (no field dropped or renamed)
    assert parse_item(item.model_dump(mode="json")) == item
    assert parse_item(json.dumps(data)) == item


def test_fixtures_cover_every_block_type():
    seen = {b["type"] for f in FIXTURES.values() for b in f["blocks"]}
    assert seen == set(BLOCK_TYPES)


def test_arabic_fixtures_mix_arabic_and_latex():
    item = FIXTURES["lesson.ar"]
    para = block_of(item, "paragraph")
    assert para["dir"] == "rtl" and "$" in para["text"]
    assert any(b["lang"] == "fr" and b["dir"] == "ltr" for b in item["blocks"])


# --- unknown blocks, invalid formulas -------------------------------------------------


def test_unknown_block_type_rejected():
    item = lesson()
    item["blocks"].append({"type": "video", "lang": "ar", "dir": "rtl", "url": "x"})
    with pytest.raises(ValidationError):
        parse_item(item)
    assert not schema_ok(item)


def test_unknown_field_rejected():
    item = lesson()
    item["blocks"][0]["color"] = "red"
    with pytest.raises(ValidationError):
        parse_item(item)
    assert not schema_ok(item)


@pytest.mark.parametrize(
    "latex",
    [
        "",
        "\\frac{1}{2",
        "x^{2}}",
        "$x^2$",
        "\\begin{cases} x \\end{matrix}",
        "\\begin{pmatrix} 1 & 2",
        "\\left( x",
        "\\input{/etc/passwd}",
        "\\def\\x{1}",
        "\\href{http://x}{y}",
    ],
)
def test_invalid_formula_rejected(latex):
    item = lesson()
    block_of(item, "formula")["latex"] = latex
    with pytest.raises(ValidationError):
        parse_item(item)
    assert latex == "" or latex_errors(latex)


@pytest.mark.parametrize("latex", ["", "\\input{a}", "\\newcommand{\\a}{b}", "\\write18{rm}"])
def test_forbidden_formula_rejected_by_json_schema_too(latex):
    item = lesson()
    block_of(item, "formula")["latex"] = latex
    assert not schema_ok(item)


@pytest.mark.parametrize(
    "latex",
    [
        "\\{x \\mid x>0\\}",
        "\\left\\{ x \\right.",
        "\\text{prix: } 5\\$",
        "\\inputvalue",  # not \input
        "\\sqrt[3]{x}\\leq\\frac{a}{b}",
    ],
)
def test_valid_formula_edge_cases(latex):
    assert latex_errors(latex) == []


@pytest.mark.parametrize(
    "text",
    [
        "<script>alert(1)</script>",
        "![x](http://img)",
        "# titre",
        "$$x^2$$",
        "prix $x^2",
        "$\\frac{1}{2$ ici",
        "$\\input{a}$",
    ],
)
def test_markdown_subset_violations_rejected(text):
    assert markdown_errors(text)
    item = lesson("fr")
    block_of(item, "paragraph")["text"] = text
    with pytest.raises(ValidationError):
        parse_item(item)


def test_markdown_subset_allows_lists_emphasis_and_escaped_dollar():
    assert markdown_errors("- **gras** et *italique*\n1. coût 5 \\$ et $x^2$") == []


# --- RTL / LTR mixing -----------------------------------------------------------------


@pytest.mark.parametrize("field", ["dir", "lang"])
def test_text_block_without_dir_or_lang_rejected(field):
    item = lesson("ar")
    del block_of(item, "paragraph")[field]
    with pytest.raises(ValidationError):
        parse_item(item)
    assert not schema_ok(item)


@pytest.mark.parametrize("type_", ["heading", "paragraph", "definition", "formula", "exercise"])
def test_arabic_block_marked_ltr_rejected(type_):
    item = copy.deepcopy(FIXTURES["exercises.ar"] if type_ == "exercise" else FIXTURES["lesson.ar"])
    block_of(item, type_)["dir"] = "ltr"
    with pytest.raises(ValidationError, match="dir must be 'rtl'"):
        parse_item(item)
    assert not schema_ok(item)


def test_french_block_marked_rtl_rejected():
    item = lesson("fr")
    block_of(item, "paragraph")["dir"] = "rtl"
    with pytest.raises(ValidationError, match="dir must be 'ltr'"):
        parse_item(item)
    assert not schema_ok(item)


def test_item_dir_must_match_item_lang():
    item = lesson("ar")
    item["dir"] = "ltr"
    with pytest.raises(ValidationError):
        parse_item(item)
    assert not schema_ok(item)


def test_french_term_inside_arabic_item_is_valid_when_marked():
    item = lesson("ar")
    item["blocks"].insert(
        1, {"type": "paragraph", "lang": "fr", "dir": "ltr", "text": "Le *discriminant* $\\Delta$"}
    )
    parse_item(item)
    assert schema_ok(item)


def test_arabic_paragraph_with_latex_is_rtl_and_valid():
    item = lesson("fr")
    item["blocks"].append(
        {
            "type": "paragraph",
            "lang": "ar",
            "dir": "rtl",
            "text": "إذا كان $\\Delta<0$ فلا يوجد حل في $\\mathbb{R}$.",
        }
    )
    parse_item(item)
    assert schema_ok(item)


# --- envelope rules -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "audience"),
    [("board", "student"), ("teacher_notes", "student"), ("parent_brief", "teacher")],
)
def test_kind_audience_matrix(kind, audience):
    item = copy.deepcopy(FIXTURES[f"{kind}.fr"])
    item["audience"] = audience
    item["sources"] = [s for s in item["sources"] if s["source_kind"] != "teacher_guide"]
    item["blocks"] = [
        b
        for b in item["blocks"]
        if not (b["type"] == "source" and b["source_kind"] == "teacher_guide")
    ]
    with pytest.raises(ValidationError, match="not written for audience"):
        parse_item(item)
    assert not schema_ok(item)


def test_student_item_cannot_cite_teacher_guide():
    item = lesson("ar")
    assert item["audience"] == "student"
    item["sources"].append(
        {"document_id": str(uuid.uuid4()), "source_kind": "teacher_guide", "pages": [3]}
    )
    with pytest.raises(ValidationError, match="teacher_guide"):
        parse_item(item)


def test_exercises_need_an_exercise_and_board_needs_a_zone():
    ex = copy.deepcopy(FIXTURES["exercises.fr"])
    ex["blocks"] = [b for b in ex["blocks"] if b["type"] != "exercise"]
    with pytest.raises(ValidationError, match="exercise block"):
        parse_item(ex)
    board = copy.deepcopy(FIXTURES["board.fr"])
    board["blocks"] = [b for b in board["blocks"] if b["type"] != "board_zone"]
    with pytest.raises(ValidationError, match="board_zone"):
        parse_item(board)
    lsn = lesson("fr")
    lsn["blocks"].append(FIXTURES["board.fr"]["blocks"][1])
    with pytest.raises(ValidationError, match="board items only"):
        parse_item(lsn)


def test_schema_version_is_pinned():
    item = lesson()
    item["schema_version"] = 2
    with pytest.raises(ValidationError):
        parse_item(item)
    assert not schema_ok(item)


def test_duplicate_block_ids_rejected():
    item = copy.deepcopy(FIXTURES["exercises.ar"])
    exs = [b for b in item["blocks"] if b["type"] == "exercise"]
    exs[1]["id"] = exs[0]["id"]
    with pytest.raises(ValidationError, match="unique"):
        parse_item(item)


@pytest.mark.parametrize(
    "patch",
    [
        {"correct_choices": [5]},
        {"correct_choices": [0, 1]},
        {"choices": None},
        {"exercise_type": "short"},
        {"points": 0},
        {"solution_steps": []},
        {"difficulty": "extreme"},
    ],
)
def test_bad_exercise_rejected(patch):
    item = copy.deepcopy(FIXTURES["exercises.fr"])
    mcq = next(b for b in item["blocks"] if b.get("exercise_type") == "mcq")
    mcq.update(patch)
    if patch.get("choices", 1) is None:
        del mcq["choices"]
    with pytest.raises(ValidationError):
        parse_item(item)


def test_table_rows_must_match_header():
    item = lesson("fr")
    block_of(item, "table")["rows"].append(["only one"])
    with pytest.raises(ValidationError, match="cells"):
        parse_item(item)


def test_models_are_immutable():
    item = parse_item(FIXTURES["recap.fr"])
    with pytest.raises(ValidationError):
        item.title = "x"
    assert isinstance(item, ContentItem)


# --- size helper ----------------------------------------------------------------------


def test_size_and_hash_are_deterministic_and_stable_across_representations():
    data = FIXTURES["lesson.ar"]
    item = parse_item(data)
    assert content_hash(item) == content_hash(parse_item(json.dumps(data)))
    assert content_hash(item).startswith("sha256:") and len(content_hash(item)) == 71
    assert compressed_size(item) == compressed_size(item)
    assert 0 < compressed_size(item) < raw_size(item)
    # Arabic is measured as UTF-8, not as \\u escapes
    assert raw_size(item) == len(
        json.dumps(
            item.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )


def test_hash_changes_with_content():
    a = parse_item(FIXTURES["recap.fr"])
    data = copy.deepcopy(FIXTURES["recap.fr"])
    data["title"] = "Autre"
    assert content_hash(a) != content_hash(parse_item(data))


def test_fixture_chapter_fits_text_budget():
    # plan §4.4: a chapter bundle (one lang, one audience) is <= 60 KB compressed
    for lang in ("ar", "fr"):
        total = sum(compressed_size(parse_item(v)) for k, v in FIXTURES.items() if k.endswith(lang))
        assert total < 60_000


# --- packs ----------------------------------------------------------------------------

H = "sha256:" + "0" * 64
H2 = "sha256:" + "1" * 64


def manifest(version: int, items: list[PackItem], **kw) -> PackManifest:
    return PackManifest(
        schema_version=1,
        curriculum_id=kw.get("curriculum_id", uuid.UUID(int=1)),
        audience="student",
        lang="ar",
        pack_version=version,
        built_at=dt.datetime(2026, 11, 7, tzinfo=dt.UTC),
        chapters=[
            PackChapter(chapter_id="ch-1", position=0, title="الفصل 1", bundle_hash=H, items=items)
        ],
    )


def pitem(n: int, version: int = 1, kind: str = "lesson", h: str = H) -> PackItem:
    return PackItem(item_id=uuid.UUID(int=n), kind=kind, version=version, content_hash=h, bytes=900)


def test_manifest_validates_against_its_json_schema():
    m = manifest(3, [pitem(1), pitem(2, kind="exercises")])
    v = Draft202012Validator(SCHEMAS["pack-manifest-v1.json"], format_checker=FormatChecker())
    assert v.is_valid(m.model_dump(mode="json"))
    assert m.total_bytes() == 1800


def test_manifest_rejects_two_items_of_one_kind_in_a_chapter():
    with pytest.raises(ValidationError, match="one published item per kind"):
        manifest(1, [pitem(1), pitem(2)])


def test_manifest_rejects_bad_hash_and_version_zero():
    with pytest.raises(ValidationError):
        pitem(1, h="md5:abc")
    with pytest.raises(ValidationError):
        manifest(0, [pitem(1)])


def test_delta_between_versions():
    old = manifest(1, [pitem(1), pitem(2, kind="exercises")])
    new = manifest(2, [pitem(1, version=2, h=H2), pitem(3, kind="recap")])
    d = diff_manifests(old, new)
    assert (d.since, d.pack_version) == (1, 2)
    assert {i.item_id.int for i in d.upserted} == {1, 3}
    assert [r.int for r in d.removed] == [2]
    assert d.chapters_changed == [] and d.chapters_removed == []
    v = Draft202012Validator(SCHEMAS["pack-delta-v1.json"], format_checker=FormatChecker())
    assert v.is_valid(d.model_dump(mode="json"))


def test_delta_from_nothing_and_guards():
    new = manifest(1, [pitem(1)])
    d = diff_manifests(None, new)
    assert d.since == 0 and len(d.upserted) == 1 and d.chapters_changed == ["ch-1"]
    assert diff_manifests(new, new).upserted == []
    with pytest.raises(ValueError, match="backwards"):
        diff_manifests(manifest(2, [pitem(1)]), new)
    with pytest.raises(ValueError, match="different packs"):
        diff_manifests(manifest(1, [pitem(1)], curriculum_id=uuid.UUID(int=9)), new)


# --- review fixes: links and HTML-emitting LaTeX (py-common#20) -----------------------


@pytest.mark.parametrize(
    "text",
    [
        "[x](javascript:alert(1))",
        "voir [ici] (http://x)",
        "[a]: javascript:alert(1)",
        "texte\n  [ref]: http://evil",
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "voir https://evil.example",
        "écrire à mailto:a@b.c",
        "www.evil.example",
        "انظر http://evil",
    ],
)
def test_links_and_urls_rejected(text):
    assert markdown_errors(text)
    item = lesson("ar")
    block_of(item, "paragraph")["text"] = text
    with pytest.raises(ValidationError):
        parse_item(item)


@pytest.mark.parametrize(
    "text",
    [
        "Remarque : $x$ est réel.",
        "Devoir: exercice 3 p. 45.",
        "[a, b] est un intervalle et $[0,1]$ aussi.",
        "les ensembles $[a;b]$ (fermés)",
        "datation: 1960",
    ],
)
def test_ordinary_text_with_colons_and_brackets_allowed(text):
    assert markdown_errors(text) == []


HTML_LATEX = [
    "\\class{x}{y}",
    "\\style{color:red}{x}",
    "\\cssId{a}{x}",
    "\\require{action}",
    "\\unicode{x41}",
    "\\htmlClass{a}{x}",
    "\\htmlId{a}{x}",
    "\\htmlStyle{color:red}{x}",
    "\\htmlData{foo=bar}{x}",
    "\\newenvironment{a}{b}{c}",
    "\\renewenvironment{a}{b}{c}",
    "\\DeclareRobustCommand{\\a}{b}",
    "\\expandafter\\a\\b",
    "^^5cinput{/etc/passwd}",
    "x^^2",
]


@pytest.mark.parametrize("latex", HTML_LATEX)
def test_html_and_expansion_latex_rejected_by_models_and_schema(latex):
    assert latex_errors(latex)
    item = lesson()
    block_of(item, "formula")["latex"] = latex
    with pytest.raises(ValidationError):
        parse_item(item)
    assert not schema_ok(item)


@pytest.mark.parametrize("latex", ["\\htmlClass{a}{x}", "^^5cinput{a}"])
def test_html_latex_rejected_in_inline_math(latex):
    assert markdown_errors(f"soit ${latex}$")


@pytest.mark.parametrize("latex", ["\\displaystyle x^2", "\\textstyle\\frac{a}{b}", "x^{2^3}"])
def test_similar_safe_latex_allowed(latex):
    assert latex_errors(latex) == []
    item = lesson()
    block_of(item, "formula")["latex"] = latex
    parse_item(item)
    assert schema_ok(item)


# --- multi-part exercises (BacMR/knowledge#331) ---------------------------------------


def multipart(lang: str = "fr") -> tuple[dict, dict]:
    item = copy.deepcopy(FIXTURES[f"exercises.{lang}"])
    return item, next(b for b in item["blocks"] if b.get("parts"))


@pytest.mark.parametrize("lang", ["fr", "ar"])
def test_fixtures_carry_a_multipart_exercise(lang):
    item, block = multipart(lang)

    parsed = parse_item(item)
    ex = next(b for b in parsed.blocks if b.type == "exercise" and b.parts)
    assert len(ex.parts) >= 3
    assert any(p.depends_on for p in ex.parts)
    assert schema_ok(item)


def test_parts_points_are_the_bareme_of_the_exercise():
    item, block = multipart()
    block["points"] = 6

    with pytest.raises(ValidationError, match="add up to 5"):
        parse_item(item)


@pytest.mark.parametrize(
    ("index", "depends_on", "match"),
    [(0, ["1"], "not an earlier part"), (1, ["3"], "not an earlier part"), (2, ["9"], "earlier")],
)
def test_a_part_depends_only_on_earlier_parts(index, depends_on, match):
    item, block = multipart()
    block["parts"][index]["depends_on"] = depends_on

    with pytest.raises(ValidationError, match=match):
        parse_item(item)


def test_part_labels_are_unique():
    item, block = multipart()
    block["parts"][1]["label"] = "1"

    with pytest.raises(ValidationError, match="duplicate label"):
        parse_item(item)


@pytest.mark.parametrize("etype", ["short", "numeric"])
def test_only_open_exercises_have_parts(etype):
    item, block = multipart()
    block["exercise_type"] = etype

    with pytest.raises(ValidationError, match="exercise_type 'open'"):
        parse_item(item)


@pytest.mark.parametrize(
    "patch",
    [{"prompt": "voir [ici](https://x.y)"}, {"skill": "guess"}, {"label": "1 a"}, {"points": 0}],
)
def test_bad_part_rejected(patch):
    item, block = multipart()
    block["parts"][0].update(patch)

    with pytest.raises(ValidationError):
        parse_item(item)


def test_one_part_is_not_a_multipart_exercise():
    item, block = multipart()
    block["parts"] = block["parts"][:1]
    block["points"] = block["parts"][0]["points"]

    with pytest.raises(ValidationError):
        parse_item(item)
    assert not schema_ok(item)


def test_exercises_without_parts_are_unchanged():
    item = copy.deepcopy(FIXTURES["exercises.fr"])
    item["blocks"] = [b for b in item["blocks"] if not b.get("parts")]

    assert parse_item(item)
    assert schema_ok(item)
