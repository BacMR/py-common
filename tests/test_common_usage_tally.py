"""Unit tests for the per-turn token tally (`bacmr_common.usage`).

The rule under test throughout is AGENTS.md's: a provider-reported count is
preserved as a measurement, a missing counter is UNKNOWN — never a fabricated
zero — and a char estimate is never allowed to look like a measurement.
"""
import asyncio

from bacmr_common.usage import (
    TurnUsage,
    active_tally,
    estimate_tokens,
    record_usage,
    usage_tally,
)


def test_provider_counts_are_preserved_verbatim():
    tally = TurnUsage()
    tally.record({"prompt_tokens": 900, "completion_tokens": 60})
    assert (tally.prompt_tokens, tally.completion_tokens) == (900, 60)
    assert tally.measured_tokens == 960
    assert tally.measured_calls == 1
    assert (tally.estimated_tokens, tally.unmeasured_calls) == (0, 0)


def test_absent_usage_block_is_an_estimate_and_a_flag_not_a_zero():
    tally = TurnUsage()
    tally.record(None, estimate=25)
    assert tally.measured_tokens == 0
    assert tally.measured_calls == 0
    assert tally.estimated_tokens == 25
    assert tally.unmeasured_calls == 1  # the "this number is a guess" marker
    assert tally.total_tokens == 25


def test_embeddings_style_usage_counts_as_measured():
    # An embeddings reply has no completion side at all; a counter that is
    # absent INSIDE a present usage block is genuinely zero, not unknown.
    tally = TurnUsage()
    tally.record({"prompt_tokens": 512, "total_tokens": 512})
    assert (tally.measured_calls, tally.unmeasured_calls) == (1, 0)
    assert tally.measured_tokens == 512


def test_malformed_usage_block_degrades_to_unmeasured():
    tally = TurnUsage()
    tally.record({"prompt_tokens": "lots"}, estimate=10)
    assert tally.measured_calls == 0
    assert (tally.unmeasured_calls, tally.estimated_tokens) == (1, 10)


def test_note_unmeasured_call_invents_no_tokens():
    # For a call made in ANOTHER process (knowledge's embedding/rerank spend):
    # we know it happened, we cannot know its size, so we say exactly that.
    tally = TurnUsage()
    tally.note_unmeasured_call()
    assert tally.unmeasured_calls == 1
    assert tally.total_tokens == 0
    assert tally.estimated_tokens == 0


def test_since_gives_one_stage_s_contribution():
    tally = TurnUsage()
    tally.record({"prompt_tokens": 100, "completion_tokens": 20})
    before = tally.snapshot()
    tally.record({"prompt_tokens": 900, "completion_tokens": 60})
    stage = tally.since(before)
    assert (stage.prompt_tokens, stage.completion_tokens) == (900, 60)
    assert stage.measured_calls == 1
    assert tally.measured_tokens == 1080  # the snapshot is a copy, not an alias


def test_estimate_tokens_accepts_text_or_a_char_count():
    assert estimate_tokens("12345678") == 2
    assert estimate_tokens(8) == 2
    assert estimate_tokens(-5) == 0


def test_record_usage_outside_a_tally_is_a_no_op():
    # Shared clients call this unconditionally; a service that never opens a
    # tally must not pay for the feature, and must not crash.
    assert active_tally() is None
    record_usage({"prompt_tokens": 1, "completion_tokens": 1})


def test_usage_tally_nests_without_leaking_into_the_outer_block():
    with usage_tally() as outer:
        record_usage({"prompt_tokens": 10, "completion_tokens": 0})
        with usage_tally() as inner:
            record_usage({"prompt_tokens": 5, "completion_tokens": 0})
        assert inner.prompt_tokens == 5
        # An outer block charging for calls it did not make is how one
        # student's turn ends up paying for another's.
        assert outer.prompt_tokens == 10
        record_usage({"prompt_tokens": 1, "completion_tokens": 0})
    assert outer.prompt_tokens == 11
    assert active_tally() is None


async def test_tallies_are_isolated_between_concurrent_tasks():
    async def one(tokens):
        with usage_tally() as tally:
            await asyncio.sleep(0)
            record_usage({"prompt_tokens": tokens, "completion_tokens": 0})
            await asyncio.sleep(0)
            return tally.prompt_tokens

    assert await asyncio.gather(one(7), one(9)) == [7, 9]
