"""The shared subject table (plan §2's consolidation of §3.2's and §3.4's twins).

Two things are defended here:

1. **The table itself** — stems, not whole ids; the STEM/humanities split §2
   names; `sciences-naturelles` deliberately absent.
2. **That it really is the single source.** The cross-service assertions live
   in THIS file rather than one copy per service on purpose: a test duplicated
   into both services to prove a table is not duplicated would be the same
   mistake one level up. Each service keeps only what is genuinely its own —
   its default — pinned in its own suite.

The predecessor of this file was a knowledge-side test that read the tutor's
`subjects.py` off disk and `exec`'d it to compare the two dicts. That check was
load-bearing precisely because the duplication existed; it is deleted with the
duplication.
"""

import pytest

from bacmr_common import subjects
from bacmr_common.subjects import GENERIC, HUMANITIES, STEM, profile_for_subject

# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject_id",
    ["mathematiques-7c", "physique-7c", "chimie", "maths", "math", "physique-chimie-3as"],
)
def test_stem_subjects_map_to_the_stem_profile(subject_id):
    assert profile_for_subject(subject_id, default=GENERIC) == STEM


@pytest.mark.parametrize(
    "subject_id",
    [
        "arabe-7cd",
        "arabe-2as",
        "philosophie-7a",
        "francais",
        "anglais",
        "histoire-geographie-4as",
        "education-islamique-7b",
    ],
)
def test_humanities_subjects_map_to_the_humanities_profile(subject_id):
    assert profile_for_subject(subject_id, default=GENERIC) == HUMANITIES


def test_lookup_matches_subject_id_stems_not_whole_ids():
    """Live ids carry a level suffix (`mathematiques-7c`, `arabe-7cd`). Matching
    whole ids would drop a subject into the default the first time a new level
    is ingested, silently and only for that level."""
    assert profile_for_subject("mathematiques-9c", default=GENERIC) == STEM
    assert profile_for_subject("MATHEMATIQUES-7C", default=GENERIC) == STEM
    assert profile_for_subject("  arabe-2as  ", default=GENERIC) == HUMANITIES
    # multi-word stems survive the segment walk
    assert profile_for_subject("histoire-geographie-7a", default=GENERIC) == HUMANITIES


def test_sciences_naturelles_is_unmapped_on_purpose():
    """§2: it "goes wherever its eval says", and that eval has not run. An
    unmapped subject must fall to the caller's default, not be guessed into a
    profile because its name sounds scientific."""
    assert "sciences-naturelles" not in subjects.SUBJECT_PROFILES
    assert profile_for_subject("sciences-naturelles-7d", default=GENERIC) == GENERIC


@pytest.mark.parametrize("subject_id", [None, "", "   ", "informatique-7c", "unknown"])
def test_unknown_and_missing_subjects_return_the_callers_default(subject_id):
    """Never raises, never returns None: every caller is on a path where an
    unrecognised subject must still produce a well-formed result."""
    assert profile_for_subject(subject_id, default=GENERIC) == GENERIC
    assert profile_for_subject(subject_id, default=HUMANITIES) == HUMANITIES


def test_the_default_must_be_supplied_by_the_caller():
    """There is no module-level default, deliberately: the three consumers
    disagree about what an unknown subject should get, and an optional
    parameter would have handed one of them the other's failure mode."""
    with pytest.raises(TypeError):
        profile_for_subject("unknown")  # type: ignore[call-arg]


def test_a_known_subject_ignores_the_default():
    assert profile_for_subject("physique-7c", default=HUMANITIES) == STEM


# --------------------------------------------------------------------------
# single source: both services read THIS table
# --------------------------------------------------------------------------


def test_both_services_resolve_through_the_shared_table(monkeypatch):
    """The functional proof, not a structural one: a key added HERE must be
    visible to both services' lookups. Asserting that each module imports the
    right name would pass just as happily against a re-pasted copy."""
    pytest.importorskip("bacmr_knowledge")  # sibling services; not in this repo
    pytest.importorskip("bacmr_tutor")
    from bacmr_knowledge import enrich
    from bacmr_tutor.prompts import subjects as tutor_subjects

    monkeypatch.setitem(subjects.SUBJECT_PROFILES, "astronomie", STEM)
    # and through the stem walk, so both services get the level-suffix
    # behaviour from the shared lookup too
    assert enrich.profile_for_subject("astronomie-7c") == STEM
    assert tutor_subjects.profile_for_subject("astronomie-7c") == STEM


def test_retrieval_resolves_through_the_shared_table_too(monkeypatch):
    """§2's own consumer: the retrieval profile lookup is the third caller,
    and the reason the table was consolidated at all."""
    pytest.importorskip("bacmr_knowledge")  # sibling service; not in this repo
    from bacmr_knowledge import retrieval

    monkeypatch.setitem(subjects.SUBJECT_PROFILES, "astronomie", STEM)
    assert retrieval.retrieval_profile_for_subject("astronomie-7c").name == STEM


def test_the_three_callers_defaults_are_each_their_own():
    """The difference is the point, not drift to be tidied away.

    * tutor -> HUMANITIES: its extra rule only tightens grounding, while the
      STEM rule emits LaTeX that reaches prose as visible `$$`. One answer.
    * enrichment -> GENERIC: byte-identical to the pre-§3.2 prompt. A wrong
      guess is baked into `chunks.summary` for a whole book.
    * retrieval -> GENERIC: equal weights, i.e. today's behaviour, for every
      request whose subject nobody has evaluated.
    """
    pytest.importorskip("bacmr_knowledge")  # sibling services; not in this repo
    pytest.importorskip("bacmr_tutor")
    from bacmr_knowledge import enrich, retrieval
    from bacmr_tutor.prompts import subjects as tutor_subjects

    assert tutor_subjects.DEFAULT_PROFILE == HUMANITIES
    assert enrich.DEFAULT_PROFILE == GENERIC
    assert retrieval.DEFAULT_RETRIEVAL_PROFILE.name == GENERIC

    assert tutor_subjects.profile_for_subject("sciences-naturelles-7d") == HUMANITIES
    assert enrich.profile_for_subject("sciences-naturelles-7d") == GENERIC
    assert retrieval.retrieval_profile_for_subject("sciences-naturelles-7d").name == GENERIC
