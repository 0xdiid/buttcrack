"""Tests for the statistical per-type cipher identifier (``cipher_id``).

The key validation builds a *labelled* set by encoding one fixed English
paragraph with several buttcrack ciphers, then asserts ``identify_types`` puts
the TRUE family in the top-2 for the clear family-level cases. Fine-grained type
accuracy is a bonus; family routing is the contract.
"""

from __future__ import annotations

import pytest

from buttcrack import registry
from buttcrack.cipher_id import compute_features, identify_types

# A long, ordinary English paragraph — enough letters for stable statistics.
PLAINTEXT = (
    "we hold these truths to be self evident that all men are created equal "
    "that they are endowed by their creator with certain unalienable rights "
    "that among these are life liberty and the pursuit of happiness that to "
    "secure these rights governments are instituted among men deriving their "
    "just powers from the consent of the governed and that whenever any form "
    "of government becomes destructive of these ends it is the right of the "
    "people to alter or to abolish it and to institute new government"
)


def _encode(name: str, key: str) -> str:
    return registry.get(name).encode(PLAINTEXT, key)


# (label, cipher-name, key, expected family). Each is a CLEAR family case.
LABELLED_CASES = [
    ("plaintext-mono", None, None, "monoalphabetic"),
    ("caesar", "caesar", "3", "monoalphabetic"),
    ("substitution", "substitution", "QWERTYUIOPASDFGHJKLZXCVBNM", "monoalphabetic"),
    ("vigenere-5", "vigenere", "LEMON", "polyalphabetic"),
    ("vigenere-7", "vigenere", "RAINBOW", "polyalphabetic"),
    ("vigenere-3", "vigenere", "CAT", "polyalphabetic"),
    ("columnar-zebra", "columnar", "ZEBRA", "transposition"),
    ("columnar-cipher", "columnar", "CIPHER", "transposition"),
    ("railfence", "railfence", "4", "transposition"),
    ("playfair", "playfair", "MONARCHY", "playfair"),
    ("bifid", "bifid", "KEYWORD/7", "bifid"),
    ("nihilist-numeric", "nihilist-substitution", "KEYWORD/EXTRA", "numeric"),
]


def _ciphertext(name: str | None, key: str | None) -> str:
    if name is None:
        return PLAINTEXT
    assert key is not None
    return _encode(name, key)


@pytest.mark.parametrize(
    ("label", "name", "key", "family"),
    LABELLED_CASES,
    ids=[c[0] for c in LABELLED_CASES],
)
def test_true_family_in_top_two(label: str, name: str | None, key: str | None, family: str) -> None:
    """The true cipher family must appear in the top-2 ranked types."""
    ct = _ciphertext(name, key)
    result = identify_types(ct)
    ranked = result["ranked"]
    assert ranked, f"{label}: no ranked types returned"
    top2 = [entry["type"] for entry in ranked[:2]]
    assert family in top2, (
        f"{label}: expected family {family!r} in top-2, got "
        f"{[(e['type'], e['score']) for e in ranked[:3]]}"
    )


def test_monoalphabetic_beats_polyalphabetic_and_vice_versa() -> None:
    """The mono/poly distinction (driven by IoC) must be unambiguous, not just top-2."""
    mono = identify_types(_encode("caesar", "5"))["ranked"][0]["type"]
    poly = identify_types(_encode("vigenere", "LEMON"))["ranked"][0]["type"]
    assert mono == "monoalphabetic"
    assert poly == "polyalphabetic"


def test_transposition_outranks_monoalphabetic() -> None:
    """Transposition keeps English IoC but breaks digraphs -> must rank #1."""
    result = identify_types(_encode("columnar", "ZEBRA"))
    assert result["ranked"][0]["type"] == "transposition"


def test_playfair_detected_via_even_digraphic_signature() -> None:
    """Playfair's boundary-aligned digraph IoC (EDI) must surface it as #1."""
    result = identify_types(_encode("playfair", "MONARCHY"))
    assert result["ranked"][0]["type"] == "playfair"
    features = result["features"]
    # EDI should clearly exceed DIC for Playfair (its defining tell).
    assert features["even_digraphic_ioc"] > features["digraphic_ioc"]


def test_numeric_routes_from_digits() -> None:
    """Digit-dominated ciphertext routes to the numeric family with high score."""
    ct = _encode("nihilist-substitution", "KEYWORD/EXTRA")
    result = identify_types(ct)
    assert result["ranked"][0]["type"] == "numeric"
    assert result["features"]["digit_fraction"] > 0.5


def test_polyalphabetic_period_annotation() -> None:
    """A detected polyalphabetic case should report a plausible key period."""
    result = identify_types(_encode("vigenere", "LEMON"))  # period 5
    poly = next(e for e in result["ranked"] if e["type"] == "polyalphabetic")
    assert poly.get("period") == result["features"]["best_period"]
    # The smallest detected period for a period-5 key should be a divisor of 5.
    assert result["features"]["best_period"] in (5,)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    ["", "      ", "HELLO", "ab cd", "12 34"],
    ids=["empty", "spaces", "five-letters", "tiny", "few-digits"],
)
def test_too_short_is_undetermined(text: str) -> None:
    result = identify_types(text)
    assert result["ranked"][0]["type"] == "undetermined"


def test_digit_only_long_is_numeric_even_without_letters() -> None:
    """Long digit-only input routes to numeric despite having no letters."""
    digits = " ".join(str(n) for n in range(11, 70))
    result = identify_types(digits)
    assert result["ranked"][0]["type"] == "numeric"


def test_features_shape_and_keys() -> None:
    """The feature panel must expose the documented statistics."""
    features = identify_types(_encode("vigenere", "LEMON"))["features"]
    for key in (
        "ioc",
        "ioc_x1000",
        "max_period_ioc",
        "best_period",
        "max_kappa",
        "digraphic_ioc",
        "even_digraphic_ioc",
        "log_digraph",
        "repeat3_root",
        "odd_repeat_pct",
        "digit_fraction",
        "alphabet_size",
        "vowel_ratio",
        "doubled_fraction",
        "even_length",
    ):
        assert key in features, f"missing feature {key!r}"


def test_ranked_entries_have_reasons_and_sorted() -> None:
    """Every ranked entry carries reasons and the list is sorted best-first."""
    result = identify_types(_encode("playfair", "MONARCHY"))
    ranked = result["ranked"]
    scores = [entry["score"] for entry in ranked]
    assert scores == sorted(scores, reverse=True)
    for entry in ranked:
        assert entry["reasons"], f"{entry['type']} has no reasons"
        assert isinstance(entry["reasons"], list)


def test_ioc_distinguishes_english_from_polyalphabetic() -> None:
    """Sanity: monoalphabetic IoC is English-like; polyalphabetic is flattened."""
    mono_ioc = compute_features(_encode("caesar", "3"))["ioc"]
    poly_ioc = compute_features(_encode("vigenere", "LEMON"))["ioc"]
    assert mono_ioc > 0.060
    assert poly_ioc < 0.050


# --------------------------------------------------------------------------- #
# Length-aware / small-sample coset significance
# --------------------------------------------------------------------------- #
def test_period_significance_flags_genuine_period_at_length() -> None:
    """A real period-7 Vigenere on a long message: high z AND not small-sample."""
    from buttcrack.cipher_id import period_significance
    from buttcrack.text import only_letters

    # period-7 key over a long paragraph -> ~40+ letters/coset.
    ct = _encode("vigenere", "RAINBOW")  # PLAINTEXT is > 400 letters
    assert len(only_letters(ct)) >= 300
    mean_ioc, letters_per_coset, z, small_sample = period_significance(ct, periods=[7])[7]
    assert z > 3.0  # coset IoC sits far above its shuffle null -> a real period
    assert letters_per_coset >= 8
    assert small_sample is False  # plenty of letters/coset -> trustworthy


def test_period_significance_flags_small_sample_artifact() -> None:
    """A short random string whose coset IoC bumps at a thin-coset period is flagged UNRELIABLE.

    This is the trap that derailed a 150-hour effort: a "period" whose elevation is pure
    small-sample noise (few letters per coset), which evaporates as the message lengthens.
    """
    import random

    from buttcrack.cipher_id import period_significance

    rng = random.Random(1)
    noise = "".join(chr(65 + rng.randrange(26)) for _ in range(150))  # n=150
    # period 24 -> ~6 letters/coset: even random text can bump the coset IoC well above the floor.
    _mean, letters_per_coset, _z, small_sample = period_significance(noise, periods=[24])[24]
    assert letters_per_coset < 8
    assert small_sample is True  # flagged regardless of how elevated the coset IoC looks


def test_features_expose_small_sample_fields() -> None:
    """compute_features gains the coset sample-size fields without dropping the old ones."""
    features = compute_features(_encode("vigenere", "RAINBOW"))
    assert "best_period_letters_per_coset" in features
    assert "best_period_small_sample" in features
    # A period-7 key over a long paragraph is not small-sample.
    if features["best_period"] >= 2:
        assert features["best_period_small_sample"] is False
