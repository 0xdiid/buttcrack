"""Tests for the block-transposition-over-periodic-substitution generalizations added to the toolkit:
#1 confidence gate (validate.solve_confidence + solver result fields)
#2 transsub.sweep_known_alphabet (calibrated known-alphabet order decider)
#4 unit=3 block transposition threaded through transsub
#5 Beaufort fixed-alphabet solve
#6 analysis.block_transposition_signal (block-of-b fingerprint)
"""

import random

import numpy as np

from buttcrack import transsub
from buttcrack.analysis import block_transposition_signal
from buttcrack.ciphers.columnar import _encode_units, _read_order
from buttcrack.quagmire_solver import KRYPTOS_ALPHABET, _encrypt, solve_fixed_alphabet
from buttcrack.scoring import resolve_scorer
from buttcrack.text import only_letters
from buttcrack.validate import solve_confidence

ENGLISH = only_letters(
    "THEHARBORMASTERKEEPSALOGOFEVERYVESSELTHATPASSESTHEBREAKWATERANDNOTESTHEWEAT"
    "HERINAMARGINWITHBLUEINKHISDAUGHTERPAINTSSMALLPORTRAITSOFTHECAPTAINSWHILETHE"
    "YWAITFORTHETIDEANDSELLSTHEMFROMABASKETNEARTHECUSTOMSHOUSEONMARKETDAYSSHE"
)


# ----- #1 confidence gate -----
def test_solve_confidence_passes_english_fails_garbage():
    good = solve_confidence(ENGLISH[:240], 240)
    assert good["recovered"] is True
    assert good["word_coverage"] > 0.4
    garbage = solve_confidence("QXZJ" * 60, 240)
    assert garbage["recovered"] is False


def test_fixed_alphabet_result_carries_recovered_flag():
    ct = _encrypt(ENGLISH[:180], KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, [3, 17, 0, 9, 22, 5, 14, 11])
    r = solve_fixed_alphabet(ct, "KRYPTOS", kind="quagmire3", periods=range(2, 14))
    assert r["recovered"] is True and r["plaintext"] == ENGLISH[:180]


# ----- #5 Beaufort -----
def test_beaufort_fixed_alphabet_roundtrip():
    pt = ENGLISH[:150]
    shifts = [4, 18, 2, 0, 11, 7]
    ct = _encrypt(pt, KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, shifts, beaufort=True)
    r = solve_fixed_alphabet(ct, "KRYPTOS", kind="beaufort", periods=range(2, 12))
    assert r["plaintext"] == pt and r["recovered"] is True


# ----- #6 block-transposition fingerprint -----
def test_block_signal_detects_hill3_and_not_plaintext():
    A = ord("A")
    pt = (ENGLISH * 2)[:270]
    P = np.array([ord(c) - A for c in pt]).reshape(-1, 3)
    K = np.array([[6, 24, 1], [13, 16, 10], [20, 17, 15]])  # invertible mod 26
    hill = "".join(chr(A + int(x)) for row in (P @ K.T) % 26 for x in row)
    assert block_transposition_signal(hill)["best_block"] == 3
    # plain English and a shuffle of it must NOT be flagged as a block cipher
    assert block_transposition_signal(pt)["best_block"] is None
    sh = list(pt)
    random.Random(1).shuffle(sh)
    assert block_transposition_signal("".join(sh))["best_block"] is None


# ----- #4 unit=3 block columnar threaded through transsub -----
def _make_block_columnar_over_sub(keyword, period, seed=7):
    """English -> Quag3(KRYPTOS, period) -> trigraph (unit=3) columnar by `keyword`."""
    width = len(keyword)
    n_blocks = (len(ENGLISH * 3) // (3 * width)) * width  # blocks divisible by width
    pt = (ENGLISH * 3)[: 3 * n_blocks]
    rng = random.Random(seed)
    shifts = [rng.randrange(26) for _ in range(period)]
    S = _encrypt(pt, KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, shifts)
    ct = _encode_units(S, _read_order(keyword), unit=3)
    return pt, ct


def test_transsub_unit3_single_columnar_recovers():
    pt, ct = _make_block_columnar_over_sub("NEEDLEWORK", period=11)
    scorer = resolve_scorer("quadgrams")
    r = transsub.crack_transposition_over_sub(
        ct, scorer, alphabet="KRYPTOS", layers=1, widths=[10],
        keywords=["NEEDLEWORK"], unit=3,
    )
    assert r["recovered"] is True
    assert only_letters(r["plaintext"]) == pt


def test_transsub_unit_rejected_for_blind_double():
    import pytest
    with pytest.raises(ValueError):
        transsub.crack_transposition_over_sub(
            ENGLISH, resolve_scorer("quadgrams"), layers=2, unit=3,
        )


# ----- #2 sweep_known_alphabet decider -----
def test_sweep_ranks_true_order_above_null():
    pt, ct = _make_block_columnar_over_sub("NEEDLEWORK", period=11, seed=3)
    true_order = _read_order("NEEDLEWORK")
    rng = random.Random(5)
    decoys = []
    for _ in range(6):
        o = list(range(10))
        rng.shuffle(o)
        decoys.append(o)
    orders = decoys[:3] + [true_order] + decoys[3:]
    res = transsub.sweep_known_alphabet(
        ct, orders, alphabet="KRYPTOS", unit=3, periods=range(6, 16), null_samples=12,
    )
    best = res["candidates"][0]
    assert best["order"] == list(true_order)
    assert best["recovered"] is True
    assert best["word_coverage"] > (res["null"]["max"] or 0.0)


def test_unit3_double_columnar_keywords_recovered_gate():
    """Regression for the unit=3 null fix: a genuine trigraph DOUBLE columnar over a sub must
    gate recovered=True (the search-aware null must undo at the SAME unit, not unit=1)."""
    from buttcrack.ciphers.columnar import _encode_units, _read_order
    kw1, kw2 = "HURRICANE", "TELESCOPE"  # both width 9
    width = 9
    n_blocks = (len(ENGLISH * 3) // (3 * width)) * width
    pt = (ENGLISH * 3)[: 3 * n_blocks]
    shifts = [3, 17, 0, 9, 22, 5, 14, 11, 1, 19, 6]  # period 11
    S = _encrypt(pt, KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, shifts)
    # encrypt applies inner o2 then outer o1; the cracker undoes o1 then o2
    ct = _encode_units(_encode_units(S, _read_order(kw2), unit=3), _read_order(kw1), unit=3)
    r = transsub.crack_double_columnar_keywords(
        ct, resolve_scorer("quadgrams"), lengths=[9], wordlist=[kw1, kw2],
        alphabet="KRYPTOS", period_band=range(9, 14), null_samples=10, unit=3,
    )
    assert r["recovered"] is True
    assert only_letters(r["plaintext"]) == pt
