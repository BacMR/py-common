"""Method kinds (method_card, model_answer, guided_practice) and their blocks."""

import copy

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from bacmr_common.content import fixtures, json_schemas, parse_item

FIXTURES = fixtures()
VALIDATOR = Draft202012Validator(json_schemas()["content-v1.json"], format_checker=FormatChecker())
KINDS = ("method_card", "model_answer", "guided_practice")


def fx(kind: str, lang: str = "ar") -> dict:
    return copy.deepcopy(FIXTURES[f"{kind}.{lang}"])


def blocks(item: dict, type_: str) -> list[dict]:
    return [b for b in item["blocks"] if b["type"] == type_]


def reject(item: dict, match: str, *, schema_rejects: bool = False) -> None:
    with pytest.raises(ValidationError, match=match):
        parse_item(item)
    if schema_rejects:
        assert not VALIDATOR.is_valid(item)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("lang", ["ar", "fr"])
def test_valid_examples(kind, lang):
    item = fx(kind, lang)
    parsed = parse_item(item)
    assert parsed.task_key and parsed.dir == ("rtl" if lang == "ar" else "ltr")
    assert VALIDATOR.is_valid(item)


@pytest.mark.parametrize("kind", KINDS)
def test_task_key_required_for_method_kinds(kind):
    item = fx(kind)
    del item["task_key"]
    reject(item, "task_key", schema_rejects=True)


@pytest.mark.parametrize("bad", ["A.b", "x", "-ab", "a b", "a" * 81, "ar_bac"])
def test_task_key_pattern(bad):
    item = fx("method_card")
    item["task_key"] = bad
    for b in blocks(item, "rubric"):
        b["task_key"] = "ar.bac-lit.analytical-essay"
    reject(item, "task_key", schema_rejects=True)


@pytest.mark.parametrize("kind", ["lesson", "exercises", "board"])
def test_task_key_must_be_null_for_old_kinds(kind):
    item = copy.deepcopy(FIXTURES[f"{kind}.fr"])
    item["task_key"] = "fr.bac-lit.resume"
    reject(item, "task_key", schema_rejects=True)


@pytest.mark.parametrize("block_type", ["method_step", "rubric", "answer_segment"])
def test_old_kinds_reject_method_blocks(block_type):
    donor = {
        "method_step": "method_card",
        "rubric": "method_card",
        "answer_segment": "model_answer",
    }
    block = blocks(fx(donor[block_type], "fr"), block_type)[0]
    lesson = copy.deepcopy(FIXTURES["lesson.fr"])
    lesson["blocks"].append(block)
    reject(lesson, "method kinds only")


def test_chapter_id_rules():
    mc = fx("method_card")
    mc["chapter_id"] = "ch-1"
    reject(mc, "chapter_id", schema_rejects=True)
    gp = fx("guided_practice")
    del gp["chapter_id"]
    reject(gp, "chapter_id", schema_rejects=True)
    ma = fx("model_answer")
    assert "chapter_id" not in ma
    parse_item(ma)
    ma["chapter_id"] = "ch-1"
    parse_item(ma)
    assert VALIDATOR.is_valid(ma)


@pytest.mark.parametrize("kind", ["method_card", "model_answer", "guided_practice"])
def test_audience_student_and_teacher_only(kind):
    item = fx(kind)
    item["audience"] = "parent"
    reject(item, "not written for audience", schema_rejects=True)
    item["audience"] = "teacher"
    parse_item(item)


def test_method_card_needs_two_steps():
    item = fx("method_card")
    item["blocks"].remove(blocks(item, "method_step")[1])
    for c in blocks(item, "rubric")[0]["criteria"]:
        c["step"] = None
    reject(item, "at least 2 method_step")


def test_method_steps_unique_and_contiguous():
    item = fx("method_card")
    blocks(item, "method_step")[1]["n"] = 3
    reject(item, "contiguous")
    item = fx("method_card")
    blocks(item, "method_step")[1]["n"] = 1
    reject(item, "contiguous")


def test_method_step_limits():
    item = fx("method_card")
    blocks(item, "method_step")[0]["phrases"] = ["x"] * 13
    reject(item, "phrases", schema_rejects=True)
    item = fx("method_card")
    blocks(item, "method_step")[0]["n"] = 0
    reject(item, "greater than or equal to 1", schema_rejects=True)
    item = fx("method_card")
    blocks(item, "method_step")[0]["points"] = 0
    reject(item, "points", schema_rejects=True)


def test_markdown_subset_enforced_in_new_blocks():
    item = fx("method_card")
    blocks(item, "method_step")[0]["do"] = "see [x](http://a.b)"
    reject(item, "links are not allowed")


def test_model_answer_needs_segment():
    item = fx("model_answer")
    item["blocks"] = [b for b in item["blocks"] if b["type"] != "answer_segment"]
    reject(item, "answer_segment")


def test_model_answer_needs_exam_source():
    item = fx("model_answer")
    item["sources"][0]["source_kind"] = "textbook"
    for b in blocks(item, "source"):
        b["source_kind"] = "textbook"
    reject(item, "source of kind exam")
    # either location is enough
    item["sources"][0]["source_kind"] = "exam"
    parse_item(item)


def test_model_answer_exam_source_block_alone_is_enough():
    item = fx("model_answer")
    item["sources"] = []
    parse_item(item)


def test_guided_practice_needs_open_exercise_and_one_rubric():
    item = fx("guided_practice")
    blocks(item, "exercise")[0].update(exercise_type="short", answer="x")
    reject(item, "open exercise")
    item = fx("guided_practice")
    item["blocks"].append(copy.deepcopy(blocks(item, "rubric")[0]))
    reject(item, "exactly one rubric")
    item = fx("guided_practice")
    item["blocks"] = [b for b in item["blocks"] if b["type"] != "rubric"]
    reject(item, "rubric_ref must name")


def test_rubric_task_key_must_match_item():
    item = fx("guided_practice")
    blocks(item, "rubric")[0]["task_key"] = "ar.bac-lit.other"
    reject(item, "rubric task_key")


def test_rubric_sum_must_equal_total():
    item = fx("guided_practice")
    blocks(item, "rubric")[0]["total"] = 7
    reject(item, "sum to total")
    blocks(item, "rubric")[0]["total"] = 6.0000000001
    parse_item(item)


def test_rubric_criterion_ids_unique():
    item = fx("guided_practice")
    cs = blocks(item, "rubric")[0]["criteria"]
    cs[1]["id"] = cs[0]["id"]
    reject(item, "unique")


def test_rubric_level_cannot_exceed_criterion():
    item = fx("guided_practice")
    blocks(item, "rubric")[0]["criteria"][0]["levels"][0]["points"] = 3
    reject(item, "exceeds criterion points")


def test_rubric_criteria_bounds():
    item = fx("guided_practice")
    blocks(item, "rubric")[0]["criteria"] = []
    reject(item, "criteria", schema_rejects=True)


def test_segment_step_must_reference_existing_method_step():
    item = fx("model_answer")
    assert blocks(item, "method_step")
    blocks(item, "answer_segment")[0]["step"] = 9
    reject(item, "does not match any method_step")


def test_criterion_step_must_reference_existing_method_step():
    item = fx("method_card")
    blocks(item, "rubric")[0]["criteria"][0]["step"] = 7
    reject(item, "does not match any method_step")


def test_step_refs_unchecked_without_method_steps():
    item = fx("guided_practice")
    assert not blocks(item, "method_step")
    blocks(item, "rubric")[0]["criteria"][0]["step"] = 9
    parse_item(item)


def test_teacher_guide_rule_unchanged():
    item = fx("method_card")
    item["sources"].append(
        {"document_id": "6f1c2a7e-3b4d-4e5f-8a9b-0c1d2e3f4a5b", "source_kind": "teacher_guide"}
    )
    reject(item, "teacher_guide")
    item["audience"] = "teacher"
    parse_item(item)


def test_open_exercise_needs_rubric_ref_in_guided_practice():
    item = fx("guided_practice")
    assert blocks(item, "exercise")[0]["rubric_ref"] == "rubric-1"
    del blocks(item, "exercise")[0]["rubric_ref"]
    reject(item, "need a rubric_ref")


def test_rubric_ref_must_name_a_rubric_in_the_item():
    item = fx("guided_practice")
    blocks(item, "exercise")[0]["rubric_ref"] = "nope"
    reject(item, "rubric_ref must name")
    lesson = copy.deepcopy(FIXTURES["exercises.fr"])
    blocks(lesson, "exercise")[0]["rubric_ref"] = "rubric-1"
    reject(lesson, "rubric_ref must name")
    del blocks(lesson, "exercise")[0]["rubric_ref"]
    parse_item(lesson)


def test_method_card_needs_exam_or_textbook_source():
    item = fx("method_card")
    item["sources"][0]["source_kind"] = "other"
    for b in blocks(item, "source"):
        b["source_kind"] = "other"
    reject(item, "exam or textbook")
    item["sources"][0]["source_kind"] = "textbook"
    parse_item(item)
