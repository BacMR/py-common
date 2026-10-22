"""The one "is this text written in Arabic?" rule (BacMR/docs#9).

Retrieval (`bacmr_knowledge.retrieval.detect_query_langs`) and the tutor
(`bacmr_tutor.understanding.detect_language`) both ask it of the student's
message, and until v0.4.0 they answered it differently (20 % over five Unicode
blocks vs 30 % over U+0600-06FF only), so a question in between was treated as
Arabic by retrieval and answered in French/English by the tutor.

It is a script test, not a language detector: the share of *letters*
(`str.isalpha`) that are Arabic script. Digits, punctuation, spaces and math
symbols do not count either way, so "2 + 3 = 5" has no letters and no answer
(`arabic_share` returns None).

Threshold 0.30, measured on the evals question banks (2026-10-22): every real
question in [0.20, 0.30) is an English/French question quoting an Arabic title
or name (`bench/items/v1/retrieval.jsonl` 3/110, `bench/golden/accepted.jsonl`
4/162) — the student writes in English/French, so it must not be called Arabic.
Arabic questions carrying a Latin formula sit well above it (an Arabic sentence
outweighs the single-letter variables of a formula).
"""

import re

# Arabic, Arabic Supplement, Arabic Extended-A, and the two presentation-form
# blocks (text pasted from PDFs often arrives in presentation forms).
ARABIC_LETTER = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")

# At or above this share of letters, the text is Arabic.
ARABIC_SHARE_THRESHOLD = 0.30


def arabic_share(text: str) -> float | None:
    """Share of the letters in `text` that are Arabic script; None when it has no letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    return sum(1 for ch in letters if ARABIC_LETTER.match(ch)) / len(letters)


def is_arabic(text: str) -> bool:
    """True when `text` is written in Arabic script (share >= ARABIC_SHARE_THRESHOLD)."""
    share = arabic_share(text)
    return share is not None and share >= ARABIC_SHARE_THRESHOLD


__all__ = ["ARABIC_LETTER", "ARABIC_SHARE_THRESHOLD", "arabic_share", "is_arabic"]
