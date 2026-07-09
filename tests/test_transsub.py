"""Tests for the transposition-OVER-substitution solver (mirror of `layered`)."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.columnar import _decode_letters, _encode_letters, _read_order
from buttcrack.ciphers.quagmire3 import QuagmireIII
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters
from buttcrack.transsub import (
    crack_columnar_reveal_enum,
    crack_double_columnar_keywords,
    crack_transposition_over_sub,
    reveal_score,
    reveal_spectrum,
)

_PLAIN = (
    "THEOLDLIGHTHOUSEKEEPERCLIMBSTHENARROWSTAIRSEACHEVENINGTOTRIMTHELAMPAND"
    "POLISHTHEGREATGLASSLENSHEHASCOUNTEDTHESTEPSFORTHIRTYYEARSANDKNOWSEVERY"
    "WORNSTONEBYHEARTFROMTHEGALLERYHEWATCHESTHEFISHINGBOATSRETURNBEFOREDARK"
)[:240]


def _encode_trans_outer(indicator: str, keyword: str) -> str:
    """plaintext -> Quagmire III (inner) -> single columnar (outer)."""
    inner = QuagmireIII().encode(_PLAIN, f"KRYPTOS/{indicator}")
    return _encode_letters(inner, _read_order(keyword))


def test_reveal_discriminator_is_mapping_independent():
    # Undoing the *correct* outer columnar must re-expose the inner period (IoC jumps to
    # ~English), while the raw ciphertext stays flat — and this holds without knowing the
    # substitution alphabet/key.
    keyword = "WORKABLE"
    ct = _encode_trans_outer("MEADOW", keyword)
    undone = _decode_letters(ct, _read_order(keyword))
    assert reveal_score(undone)[0] > 0.062
    assert reveal_score(ct)[0] < 0.055
    assert reveal_score(undone)[0] - reveal_score(ct)[0] > 0.015


def test_single_columnar_over_quagmire_recovers_plaintext():
    keyword = "WORKABLE"
    ct = _encode_trans_outer("MEADOW", keyword)
    res = crack_transposition_over_sub(
        ct,
        get_scorer(),
        layers=1,
        widths=[8],
        keywords=["RANDOMLY", keyword, "THEFENCE", "PORTABLE"],
    )
    assert res["recovered"] is True
    assert res["word_coverage"] >= 0.45
    assert _PLAIN[:48] in res["plaintext"]


def test_no_false_positive_on_unstructured_text():
    # A keyword that is not the true outer order must not be reported as a confident solve.
    ct = _encode_trans_outer("MEADOW", "WORKABLE")
    res = crack_transposition_over_sub(
        ct, get_scorer(), layers=1, widths=[8], keywords=["MONDAYBC"]
    )
    assert res["recovered"] is False


# A word-dense plaintext whose length (226) is NOT a multiple of width 6 -> an INCOMPLETE
# columnar with ragged column lengths, the regime where the reveal-IoC discriminator is
# sharpest and where a keyword sweep over a *non-dictionary* order would miss the answer.
_ENUM_PLAIN = only_letters(
    "THEUNANIMOUSDECLARATIONOFTHETHIRTEENUNITEDSTATESOFAMERICAWHENINTHE"
    "COURSEOFHUMANEVENTSITBECOMESNECESSARYFORONEPEOPLETODISSOLVETHE"
    "POLITICALBANDSWHICHHAVECONNECTEDTHEMWITHANOTHERANDTOASSUMEAMONG"
    "THEPOWERSOFTHEEARTHTHESEPARATEANDEQUALSTATIONWHICHTHELAWSOFNATURE"
)[:226]

# A non-dictionary read-order at width 6 (no English keyword sorts to this permutation).
_ENUM_ORDER = [4, 1, 5, 0, 3, 2]


@pytest.mark.slow
def test_enum_recovers_nondictionary_incomplete_columnar():
    # period-8 KRYPTOS Quagmire (inner) -> incomplete width-6 columnar in a NON-dictionary
    # order (outer). The keyword sweep cannot reach this order; the full w! enumeration,
    # ranked by the mapping-independent reveal-IoC, must recover it to clean English.
    assert len(_ENUM_PLAIN) % 6 != 0  # genuinely incomplete
    inner = QuagmireIII().encode(_ENUM_PLAIN, "KRYPTOS/PORTABLE")  # 8-letter indicator
    ct = _encode_letters(inner, _ENUM_ORDER)

    res = crack_columnar_reveal_enum(ct, get_scorer(), widths=[6], top_orders=8, null_samples=16)
    assert res["recovered"] is True
    assert res["word_coverage"] >= 0.6
    assert res["structure"]["columnar_order"] == _ENUM_ORDER
    assert _ENUM_PLAIN[:48] in res["plaintext"]
    # The recovered reveal must beat the shuffled-search null (not a selection-bias fluke).
    assert res["reveal_null"]["verdict"] == "beats null"
    assert res["reveal_null"]["beats_null_max"] is True


@pytest.mark.slow
def test_enum_null_gate_rejects_structureless_text():
    # Random letters carry no transposition-over-substitution structure: the best-of-
    # enumeration reveal sits inside the shuffled-search band, so the null gate must veto
    # the report even before the substitution fails to solve to English.
    rng = random.Random(11)
    junk = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(160))
    res = crack_columnar_reveal_enum(junk, get_scorer(), widths=[5], top_orders=2, null_samples=12)
    assert res["recovered"] is False
    assert res["reveal_null"]["verdict"] == "within null (overfit)"


@pytest.mark.slow
def test_keyword_sweep_solve_reports_beat_null():
    # The cheap keyword-sweep path also calibrates its reported reveal against a shuffled
    # replay of the SAME orders; a genuine solve must clear that null.
    keyword = "WORKABLE"
    ct = _encode_trans_outer("MEADOW", keyword)
    res = crack_transposition_over_sub(
        ct,
        get_scorer(),
        layers=1,
        widths=[8],
        keywords=["RANDOMLY", keyword, "THEFENCE", "PORTABLE"],
        null_samples=12,
    )
    assert res["recovered"] is True
    assert "reveal_null" in res
    assert res["reveal_null"]["verdict"] == "beats null"


# --- double-columnar keyword-pair sweep ----------------------------------- #

_DBL_PLAIN = only_letters(
    "THEOLDLIGHTHOUSEKEEPERCLIMBSTHENARROWSTAIRSEACHEVENINGTOTRIMTHELAMPAND"
    "POLISHTHEGREATGLASSLENSHEHASCOUNTEDTHESTEPSFORTHIRTYYEARSANDKNOWSEVERY"
    "WORNSTONEBYHEARTFROMTHEGALLERYHEWATCHESTHEFISHINGBOATSRETURNBEFOREDARK"
)


def _encode_double(indicator: str, kw1: str, kw2: str) -> str:
    """plaintext -> Quagmire III (inner) -> columnar kw1 -> columnar kw2 (outer)."""
    inner = QuagmireIII().encode(_DBL_PLAIN, f"KRYPTOS/{indicator}")
    return _encode_letters(_encode_letters(inner, _read_order(kw1)), _read_order(kw2))


def test_double_columnar_keyword_pair_recovers_synthetic():
    # period-13 Vigenere/KRYPTOS inner, then TELESCOPE then HURRICANE columnar (double-columnar
    # shape).
    # The directed pair sweep over a small wordlist containing the true keywords must undo
    # both layers (in the correct decrypt order) and solve the exposed substitution.
    ct = _encode_double("TELESCOPESAYS", "TELESCOPE", "HURRICANE")
    wordlist = ["TELESCOPE", "HURRICANE", "RANDOMERS", "WORKPLACE", "SCRAMBLED"]
    res = crack_double_columnar_keywords(
        ct,
        get_scorer(),
        lengths=[9],
        wordlist=wordlist,
        period_band=range(11, 16),
        null_samples=12,
    )
    assert res["recovered"] is True
    assert res["word_coverage"] >= 0.45
    # Decrypt undoes the OUTER columnar (HURRICANE) first, then the inner (TELESCOPE).
    assert res["structure"]["columnar_keywords"] == ["HURRICANE", "TELESCOPE"]
    assert res["structure"]["period"] == 13
    assert res["structure"]["convention"] == "vigenere"
    assert _DBL_PLAIN[:48] in res["plaintext"]
    assert res["reveal_null"]["verdict"] == "beats null"
    assert res["reveal_null"]["beats_null_max"] is True


def test_double_columnar_keyword_pair_rejects_wrong_wordlist():
    # No keyword pair in the list undoes the true transposition: nothing should be reported
    # as recovered (the reveal pre-filter never fires / the null gate vetoes).
    ct = _encode_double("TELESCOPESAYS", "TELESCOPE", "HURRICANE")
    res = crack_double_columnar_keywords(
        ct,
        get_scorer(),
        lengths=[9],
        wordlist=["RANDOMERS", "WORKPLACE", "SCRAMBLED"],
        period_band=range(11, 16),
        null_samples=12,
    )
    assert res["recovered"] is False


def test_reveal_spectrum_flags_real_layering():
    # reveal_spectrum (the butt-diagnose feed): the width whose enumeration re-exposes the
    # inner period must rank top and beat the search-aware null.
    keyword = "WORKABLE"
    ct = _encode_trans_outer("MEADOW", keyword)
    spec = reveal_spectrum(ct, widths=[8], periods=range(3, 12), null_samples=12)
    assert spec["widths"], "at least one width reported"
    top = spec["widths"][0]
    assert top["width"] == 8
    assert top["best_reveal"] > spec["raw_reveal"]
    assert top["verdict"] == "beats null"


def test_reveal_spectrum_no_false_layering_on_random():
    # Structureless text: no width should claim a beats-null reveal.
    rng = random.Random(5)
    junk = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(120))
    spec = reveal_spectrum(junk, widths=[5, 6], periods=range(3, 10), null_samples=10)
    assert all(row["verdict"] == "within null (overfit)" for row in spec["widths"])
