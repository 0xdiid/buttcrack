"""Fast fixed-alphabet periodic solve (quagmire_solver.solve_fixed_alphabet).

When the keyed alphabet is *known* (the common Kryptos-family case), we recover only
the per-period shifts by per-column chi-square — orders of magnitude faster than the
blind annealing in solve(). These assert exact recovery at periods with enough
letters/column, the period sweep, and that it accepts a keyword or a full alphabet.
"""

import time

from buttcrack.quagmire_solver import (
    KRYPTOS_ALPHABET,
    _encrypt,
    _keyed_alphabet,
    solve_fixed_alphabet,
)
from buttcrack.text import only_letters

PLAIN = only_letters(
    "BETWEENSUBTLESHADINGANDTHEABSENCEOFLIGHTLIESTHENUANCEOFIQLUSIONITWASTOTALLY"
    "INVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHSMAGNETICFIELDXTHEINFORMATIONWAS"
    "GATHEREDANDTRANSMITTEDUNDERGRUUNDTOANUNKNOWNLOCATION"
)


def test_recovers_quagmire3_known_alphabet_exactly():
    """Quagmire III, KRYPTOS keyed alphabet, period 8 (~25 letters/column)."""
    shifts = [3, 17, 0, 9, 22, 5, 14, 11]
    ct = _encrypt(PLAIN, KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, shifts)
    r = solve_fixed_alphabet(ct, KRYPTOS_ALPHABET, kind="quagmire3", periods=range(2, 16))
    assert r["period"] == 8
    assert r["plaintext"] == PLAIN
    assert r["shifts"] == shifts


def test_accepts_keyword_or_full_alphabet():
    """`alphabet` may be a keyword (expanded) or a full 26-letter alphabet."""
    shifts = [1, 7, 19, 4, 12, 25]
    ct = _encrypt(PLAIN, KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, shifts)
    by_keyword = solve_fixed_alphabet(ct, "KRYPTOS", periods=[6])
    by_alphabet = solve_fixed_alphabet(ct, KRYPTOS_ALPHABET, periods=[6])
    assert by_keyword["plaintext"] == PLAIN
    assert by_alphabet["plaintext"] == by_keyword["plaintext"]
    assert _keyed_alphabet("KRYPTOS") == KRYPTOS_ALPHABET


def test_vigenere_kind_uses_straight_alphabet():
    """For `vigenere`, alphabets are straight A-Z regardless of the keyword given."""
    shifts = [4, 18, 2, 0, 11]
    straight = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ct = _encrypt(PLAIN, straight, straight, shifts)
    r = solve_fixed_alphabet(ct, "ANYTHING", kind="vigenere", periods=range(2, 12))
    assert r["period"] == 5
    assert r["plaintext"] == PLAIN


def test_fast_and_sweeps_periods():
    """The whole point: sweeping a wide period band is cheap (no annealing)."""
    shifts = [2, 9, 14, 3, 20, 7, 1]
    ct = _encrypt(PLAIN, KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, shifts)
    t0 = time.time()
    r = solve_fixed_alphabet(ct, KRYPTOS_ALPHABET, kind="quagmire3", periods=range(2, 31))
    assert r["period"] == 7
    assert r["plaintext"] == PLAIN
    assert time.time() - t0 < 5.0
