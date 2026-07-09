"""Tests for decoupled recovery of a periodic substitution over a Playfair inner.

CT = periodic_Vigenere/Quagmire( playfair(PT) ). The Playfair sibling of
``sub_fractionation``: given a candidate inner square it recovers the outer key by a
constrained coordinate descent (drop-letter admissibility + a no-doubled-digraph penalty),
and a driver ranks candidate squares by the recovered decode's objective.
"""

from __future__ import annotations

import random

from buttcrack.sub_playfair import (
    crack_sub_over_playfair,
    encrypt_sub_over_playfair,
    playfair_decode,
    playfair_encode,
    recover_outer_key_over_playfair,
    resolve_grid,
    resolve_square,
    sub_decode,
    sub_encode,
)
from buttcrack.text import only_letters

ENGLISH = only_letters(
    "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGEACHTREEFORFR"
    "OSTHEPRUNESTHEDEADBRANCHESWITHASMALLCURVEDKNIFEANDTIESTHEYOUNGSHOOTSTOCED"
).replace("J", "I")


def _plant(pt, square_kw, shifts, *, drop="J", alpha="KRYPTOS"):
    return encrypt_sub_over_playfair(
        pt, square_kw, outer_alphabet=alpha, outer_shifts=shifts, drop_letter=drop
    )


def test_sub_encode_decode_roundtrip():
    alpha = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
    shifts = [3, 17, 0, 9, 22, 5, 14]
    inter = ENGLISH[:120]
    assert sub_decode(sub_encode(inter, alpha, shifts), alpha, shifts) == inter


def test_playfair_inner_roundtrip():
    sq = resolve_square("BUTTERFLY", "J")
    enc = playfair_encode(ENGLISH, sq)
    dec = playfair_decode(enc, sq)
    # Playfair round-trips up to its X-padding; the recovered text contains the message.
    assert "EARLYINTHEMORNING" in dec.replace("X", "")


def test_recover_outer_key_given_correct_square():
    """Given the correct inner square, the outer Quagmire-7 key + decode are recovered."""
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "BUTTERFLY", shifts)
    sq = resolve_square("BUTTERFLY", "J")
    _rshifts, pt, _score = recover_outer_key_over_playfair(
        ct, sq, outer_alphabet="KRYPTOS", outer_period=7, drop_letter="J"
    )
    assert "EARLYINTHEMORNING" in pt.replace("X", "")


def test_playfair_inner_roundtrip_non_j_drop():
    """Drop-letter aware: a Q-dropped square (keeping J) round-trips without folding J->I."""
    sq = resolve_square("WATERMELON", "Q")
    pt = "ATTACKATDAWNJUMPSOVERALAZYFOG"  # contains J, which must be preserved
    dec = playfair_decode(playfair_encode(pt, sq), sq)
    assert "ATTACKATDAWNJUMPS" in dec.replace("X", "")


def test_crack_drop_sweep_does_not_crash():
    """crack with drop_letter='sweep' runs for every drop (regression: used to KeyError)."""
    rng = random.Random(3)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "BUTTERFLY", shifts, drop="Q")
    res = crack_sub_over_playfair(
        ct, outer_period=7, squares=["BUTTERFLY", "MEADOW"], drop_letter="sweep", top=3
    )
    assert res  # completed across all 26 drop letters without raising


def test_crack_ranks_correct_square_far_above_plateau():
    """The full driver puts the correct square on top, clear of the wrong-square plateau."""
    rng = random.Random(2)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "BUTTERFLY", shifts)
    squares = ["RANDOMXYZ", "MEADOW", "BUTTERFLY", "NEEDLE", "SILVER"]
    res = crack_sub_over_playfair(
        ct, outer_period=7, squares=squares, objective="fitness", drop_letter="J", top=5
    )
    assert res, "expected candidates"
    best_sq, _key, best_pt, best_score = res[0]
    assert best_sq == resolve_square("BUTTERFLY", "J")
    assert "EARLYINTHEMORNING" in best_pt.replace("X", "")
    # decisive separation: correct square scores well above the runner-up
    assert best_score - res[1][3] > 1.0


# --------------------------------------------------------------------------- #
# 26-cell rectangular grids (2x13): all 26 letters, no drop letter
# --------------------------------------------------------------------------- #
def test_resolve_grid_2x13_is_26_letters():
    grid = resolve_grid("BUTTERFLY", (2, 13))
    assert len(grid) == 26 and set(grid) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # 5x5 shape still yields the classic 25-cell square
    assert resolve_grid("BUTTERFLY", (5, 5)) == resolve_square("BUTTERFLY", "J")


def test_playfair_inner_roundtrip_2x13_preserves_j():
    grid = resolve_grid("NEEDLE", (2, 13))
    pt = "MAJORINJURYJEOPARDIZESTHEJOURNEYAHEAD"  # J is a first-class letter here
    dec = playfair_decode(playfair_encode(pt, grid, (2, 13)), grid, (2, 13))
    assert "MAIORIN" not in dec  # J was NOT folded to I
    assert "MAJORIN" in dec.replace("X", "")


def test_recover_outer_key_given_correct_2x13_grid():
    """Given the correct 26-cell grid, the outer Quagmire-7 key + decode are recovered."""
    rng = random.Random(11)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = encrypt_sub_over_playfair(
        ENGLISH, "DANGERLESS", outer_alphabet="KRYPTOS", outer_shifts=shifts, shape=(2, 13)
    )
    grid = resolve_grid("DANGERLESS", (2, 13))
    _rshifts, pt, _score = recover_outer_key_over_playfair(
        ct, grid, outer_alphabet="KRYPTOS", outer_period=7, shape=(2, 13)
    )
    assert "EARLYINTHEMORNING" in pt.replace("X", "")


def test_crack_2x13_ranks_correct_grid_above_plateau():
    """The driver, sweeping a 2x13 shape, puts the correct grid on top, clear of the plateau."""
    rng = random.Random(12)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = encrypt_sub_over_playfair(
        ENGLISH, "DANGERLESS", outer_alphabet="KRYPTOS", outer_shifts=shifts, shape=(2, 13)
    )
    squares = ["RANDOMXYZ", "MEADOW", "DANGERLESS", "NEEDLE", "SILVER"]
    res = crack_sub_over_playfair(
        ct, outer_period=7, squares=squares, shapes=(2, 13), objective="fitness", top=5
    )
    assert res
    best_grid, _key, best_pt, best_score = res[0]
    assert best_grid == resolve_grid("DANGERLESS", (2, 13))
    assert "EARLYINTHEMORNING" in best_pt.replace("X", "")
    assert best_score - res[1][3] > 1.0
