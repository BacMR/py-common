"""The subject -> profile table, shared by every stage that varies by subject.

THIS IS THE SINGLE SOURCE. It began as two copies -- one in
`bacmr_tutor.prompts.subjects` (§3.4, the answer prompt) and a deliberate
duplicate in `bacmr_knowledge.enrich` (§3.2, the enrichment prompt), each
carrying a comment saying the right home was a shared lib and that §2 was the
moment to build it. §2 needs the same table a third time (retrieval profiles),
and a table that is copied three times is a table that drifts. The duplicate
was policed by a test that read the tutor's file off disk and `exec`'d it,
which is the shape of a problem, not a solution.

`libs/py-common` is that home and it is not a new precedent: `LLMClient` was
moved here for exactly this reason -- the two services must not import each
other, so anything both need lives in the lib both already depend on.

WHAT THIS MODULE DOES NOT DECIDE: the default. Callers pass their own, because
the three consumers are not equally safe to be wrong in:

* the tutor defaults to HUMANITIES -- its extra rule only tightens grounding,
  while the STEM rule emits LaTeX that reaches prose as visible `$$` markup.
  The blast radius is one live answer.
* enrichment defaults to GENERIC -- byte-identical to the pre-§3.2 prompt.
  A wrong guess there is baked into `chunks.summary` for a whole book and is
  only undone by a re-enrichment.
* retrieval defaults to GENERIC too, which `bacmr_knowledge.retrieval` maps to
  its equal-weight default profile.

So `profile_for_subject` takes `default` as a REQUIRED keyword argument. An
optional one would have had to pick a winner between those three, and the
loser would have inherited the other's failure mode silently.

Keys are subject-id *stems*, not whole ids. Live ids carry a level suffix
(`mathematiques-7c`, `physique-7c`, `philosophie-7a`, `arabe-7cd`,
`arabe-2as`), so matching whole ids would silently drop a subject into the
default the first time a new level is ingested. The lookup drops trailing
"-" segments until a stem matches, which also keeps multi-word stems
(`histoire-geographie`) working.

`sciences-naturelles` is deliberately absent: §2 says it "goes wherever its
eval says", and that eval has not run.
"""

STEM = "stem"
HUMANITIES = "humanities"
# The third profile is "we have not profiled this subject". It is a real
# member rather than a `None`, because enrichment and retrieval both need
# something to key a table on -- see their own DEFAULT_PROFILE constants.
GENERIC = "generic"

SUBJECT_PROFILES: dict[str, str] = {
    # STEM (§2: "math, physics, chemistry")
    "mathematiques": STEM,
    "maths": STEM,
    "math": STEM,
    "physique": STEM,
    "physique-chimie": STEM,
    "chimie": STEM,
    # Humanities (§2: "Arabic, French, English, philosophy, history,
    # geography, Islamic education")
    "arabe": HUMANITIES,
    "francais": HUMANITIES,
    "anglais": HUMANITIES,
    "philosophie": HUMANITIES,
    "histoire": HUMANITIES,
    "geographie": HUMANITIES,
    "histoire-geographie": HUMANITIES,
    "education-islamique": HUMANITIES,
}


def profile_for_subject(subject_id: str | None, *, default: str) -> str:
    """Profile for `subject_id`, or `default` when it is unknown.

    Never raises and never returns `None`: every caller is on a path where an
    unrecognised subject must still produce a well-formed result (a prompt, an
    enrichment call, a search) rather than a failure.
    """
    parts = (subject_id or "").strip().lower().split("-")
    while parts:
        stem = "-".join(parts)
        if stem in SUBJECT_PROFILES:
            return SUBJECT_PROFILES[stem]
        parts.pop()
    return default


__all__ = ["GENERIC", "HUMANITIES", "STEM", "SUBJECT_PROFILES", "profile_for_subject"]
