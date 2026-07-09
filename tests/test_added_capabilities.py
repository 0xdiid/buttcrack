"""Tests for the seven capabilities added from the block-transposition generalization work:

1. unit-k (block) transposition           -> ciphers.columnar._encode_units/_decode_units
2. repeat-excised digraph diagnostic        -> analysis.repeat_adjusted_stats
3. family/sibling baseline comparison       -> analysis.family_baseline
4. corpus-derived key candidates            -> keysources.keys_from_corpus
5. composed-key build + decompose           -> keysources.compose_key/decompose_key
6. Latin n-gram scoring                      -> scoring (lang="latin")
7. length-scaled reveal-period cap           -> transsub.reliable_period_cap/reveal_score
"""

from __future__ import annotations

import random

import pytest

from buttcrack import analysis, keysources, transsub
from buttcrack.ciphers.columnar import _decode_units, _encode_letters, _encode_units
from buttcrack.scoring import LANGUAGES, get_scorer, ngram_table_available
from buttcrack.text import only_letters

ENGLISH = (
    "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHEEARLYMORNINGSUNROSESLOWLYOVERTHE"
    "QUIETVILLAGEANDTHEPEOPLEWENTABOUTTHEIRWORKWITHASTEADYANDFAMILIARRHYTHM"
)
# A long, *non-repeating* English passage (distinct sentences, not a triplication) so the
# repeat-excision diagnostic sees genuine diffuse digraph structure that survives excision.
REAL_ENGLISH = only_letters(
    "the analysis routines need a healthy stretch of perfectly ordinary english prose so "
    "that the frequency statistics and quadgram fitness scores can lock onto the underlying "
    "message and recover the original text without any prior knowledge of the secret key "
    "that was chosen to encipher it in the beginning of this rather long and varied sample "
    "which mentions many different words about weather mountains rivers harbours and markets "
    "so that no single trigram dominates the way a repeated block would in a flat cipher"
)


# 1 ----------------------------------------------------------------- unit-k transposition
@pytest.mark.parametrize("unit", [1, 2, 3, 5])
@pytest.mark.parametrize("n", [40, 93, 279, 277])
def test_unit_transposition_roundtrip(unit, n):
    rng = random.Random(n * 10 + unit)
    s = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))
    for width in (3, 7, 9):
        order = list(range(width))
        rng.shuffle(order)
        ct = _encode_units(s, order, unit)
        assert len(ct) == n
        assert _decode_units(ct, order, unit) == s  # round-trips, incl. ragged tail


def test_unit1_equals_letter_columnar():
    order = [2, 0, 1]
    assert _encode_units(ENGLISH, order, 1) == _encode_letters(ENGLISH, order)


def test_unit3_preserves_trigrams():
    # A trigram-granular transposition keeps 3-letter blocks intact; the multiset of
    # length-3 blocks is unchanged by the column shuffle.
    s = ENGLISH[: 3 * (len(ENGLISH) // 3)]
    ct = _encode_units(s, [2, 0, 1], 3)

    def blocks(t):
        return sorted(t[i : i + 3] for i in range(0, len(t), 3))

    assert blocks(ct) == blocks(s)


# 2 -------------------------------------------------------- repeat-excised diagnostic
def test_repeat_adjusted_flags_artifact_vs_real():
    # Random-ish text whose only digraph structure is two thrice-repeated trigrams:
    rng = random.Random(7)
    base = [rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(279)]
    for pos in (30, 90, 200):
        base[pos : pos + 3] = list("YMR")
    for pos in (60, 150, 240):
        base[pos : pos + 3] = list("HTE")
    artifact = analysis.repeat_adjusted_stats("".join(base))
    assert artifact["digraph_ratio_full"] > artifact["digraph_ratio_excised"]
    assert "repeats only" in artifact["verdict"]

    real = analysis.repeat_adjusted_stats(REAL_ENGLISH)
    assert real["excised_z_vs_shuffle"] > artifact["excised_z_vs_shuffle"]
    assert "real" in real["verdict"]


# 3 -------------------------------------------------------------- family baseline
def test_family_baseline_normal_and_outlier():
    rng = random.Random(3)
    # A family spanning flat-random up to English IoC, so the band brackets a flat target.
    sibs = {
        f"S{i}": "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(280))
        for i in range(4)
    }
    sibs["S_eng"] = REAL_ENGLISH
    target = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(279))
    res = analysis.family_baseline(target, sibs)
    assert res["ioc_family_normal"] is True  # flat target is within the family IoC band
    assert "FAMILY-NORMAL" in res["verdict"]
    # a degenerate single-letter target (IoC ~ 1.0) is a clear outlier above the band
    out = analysis.family_baseline("A" * 200, sibs)
    assert out["ioc_family_normal"] is False


# 4 ---------------------------------------------------------- corpus key sources
def test_keys_from_corpus_kinds_and_cleanliness():
    corpus = {
        "logA": "Two crates of maps arrived. The archivist filed each one.",
        "logB": "The workshop is filled with old tools.",
    }
    keys = keysources.keys_from_corpus(corpus, window_lengths=(6,))
    kinds = {k["kind"] for k in keys}
    assert {"full", "acrostic-word", "word", "window:6"} <= kinds
    assert any(k.startswith("reverse:") for k in kinds)
    assert any(k.startswith("atbash:") for k in kinds)
    for k in keys:  # every candidate is uppercase letters only, with provenance
        assert k["value"] == only_letters(k["value"]) and k["value"].isupper()
        assert k["source"] in corpus


# 5 ------------------------------------------------------- composed-key build/decompose
def test_compose_decompose_roundtrip():
    key = keysources.compose_key("WATERMELON", "LAVENDER")  # word-pair canon, lcm(10,8)=40
    assert len(key) == 40
    pairs = keysources.decompose_key(key, ["WATERMELON", "LAVENDER", "MAPLE", "MEADOW"])
    recovered = {(p["word_a"], p["word_b"]) for p in pairs}
    assert ("WATERMELON", "LAVENDER") in recovered  # the self-validating recovery


def test_compose_matches_known_composed_key():
    # A fixed composed-key vector: QuagmireKRYPTOS(WATERMELON, LAVENDER).
    assert (
        keysources.compose_key("WATERMELON", "LAVENDER")
        == "HHKVQYVMVKNMWUFNYXRTJLIFMZAYXPBBUMWPTRJQ"
    )


# 6 ----------------------------------------------------------------- Latin scoring
def test_latin_scorer_available_and_discriminates():
    assert "latin" in LANGUAGES
    assert ngram_table_available("quadgrams", "latin")
    lat = get_scorer("quadgrams", "latin")
    latin_txt = "GALLIAESTOMNISDIVISAINPARTESTRESQUARUMUNAMINCOLUNTBELGAE"
    # the Latin scorer rates real Latin far above random letters
    rnd = "QXZWKVBFJGPYMCDLNHRSTUAEIO" * 3
    assert lat.score(latin_txt) / len(latin_txt) > lat.score(rnd) / len(rnd)


# 7 ------------------------------------------------------- reveal-period cap scaling
def test_reliable_period_cap_scales_with_length():
    assert transsub.reliable_period_cap(279) == 17  # period 36 genuinely unreliable here
    assert transsub.reliable_period_cap(900) > 18  # long text no longer blinded at 18
    # reveal_score honours the scaled cap (never examines an unreliable high period)
    long_text = ENGLISH * 8
    _, period = transsub.reveal_score(long_text)
    assert period <= transsub.reliable_period_cap(len(long_text))
