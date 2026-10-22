import pytest

from bacmr_common.language import ARABIC_SHARE_THRESHOLD, arabic_share, is_arabic


def test_threshold_is_thirty_percent():
    assert ARABIC_SHARE_THRESHOLD == 0.30


@pytest.mark.parametrize(
    "text",
    [
        "ما هي عناصر القصة القصيرة؟",
        # Arabic with Latin math: the formula's variables do not outweigh the sentence.
        "أوجد قيمة x إذا كان 2x + 3 = 7",
        "احسب مشتقة الدالة f(x) = x^2 + 3x",
        "ادرس تغيرات الدالة f المعرفة بـ f(x) = ln(x) - x",
        # Arabic with a French term in parentheses.
        "ما هو قانون نيوتن الثاني (deuxième loi de Newton)؟",
        # Presentation forms (PDF copy-paste): U+FE8D.. / U+FEFB.
        "ﺍﻟﺴﻼﻡ ﻋﻠﻴﻜﻢ",
        # Arabic Supplement letter (U+0750) only.
        "\u0750\u0751\u0752",
        # Tashkeel and Arabic-Indic digits do not count as letters; the rest is Arabic.
        "اُكْتُبْ ٣ أمثلة",
    ],
)
def test_arabic(text):
    assert is_arabic(text)


@pytest.mark.parametrize(
    "text",
    [
        "Qu'est-ce qu'une fonction dérivable ?",
        "What is the derivative of x^2?",
        "Résoudre l'équation 2x + 3 = 7",
        # Latin-transliterated Arabic is Latin script.
        "chou ma3na al-istiqra2 fi al-riyadiyat",
        "ma hiya anwa3 al-jumla?",
        # French question quoting an Arabic word: real bench item fr-ar-001 (share 0.11).
        "Selon le premier vers du poème, quel est le bienfait de la coopération (التعاون) ?",
        # English question quoting an Arabic name: bench item en-ar-r2-000 (share 0.25),
        # in the old 20-30 % gap between retrieval and the tutor.
        "What roles and qualities are attributed to المرابط محمد الأمين بن أحمد زيدان "
        "after his return from studying with well-known scholars?",
    ],
)
def test_not_arabic(text):
    assert not is_arabic(text)


@pytest.mark.parametrize("text", ["", "   ", "2 + 3 = 5", "١٢٣ + 45", "?!", "√(16) = 4"])
def test_no_letters(text):
    assert arabic_share(text) is None
    assert not is_arabic(text)


def test_boundary_is_inclusive():
    # 3 Arabic letters of 10 = exactly 0.30.
    assert arabic_share("ابت abcdefg") == pytest.approx(0.30)
    assert is_arabic("ابت abcdefg")
    # 2 of 10 = 0.20: Arabic by the old retrieval rule, not any more.
    assert not is_arabic("اب abcdefgh")


def test_share_counts_letters_only():
    assert arabic_share("سؤال 1+1") == 1.0
    assert arabic_share("ab سؤ") == 0.5
