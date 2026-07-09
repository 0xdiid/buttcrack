"""Tests for decoupled recovery of a periodic substitution over a Seriated-Playfair inner.

CT = periodic_Vigenere/Quagmire( seriated_playfair(PT) ). The seriated sibling of
``sub_playfair``: the digraphs are vertical pairs of a period-N block, so they live at
non-adjacent positions ``(blockstart+j, blockstart+width+j)``. Given a candidate inner square
the outer key is recovered by a constrained coordinate descent (drop-letter admissibility + a
no-doubled-digraph penalty over the *vertical* pairing), and a driver ranks candidate squares.
"""

from __future__ import annotations

import random

from buttcrack.sub_seriated_playfair import (
    crack_sub_over_seriated_playfair,
    encrypt_sub_over_seriated_playfair,
    recover_outer_key_over_seriated_playfair,
    resolve_square,
    seriated_digraph_pairs,
    seriated_playfair_decode,
    seriated_playfair_encode,
    sub_decode,
    sub_encode,
)
from buttcrack.text import only_letters

ENGLISH = only_letters(
    "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGEACHTREEFORFR"
    "OSTHEPRUNESTHEDEADBRANCHESWITHASMALLCURVEDKNIFEANDTIESTHEYOUNGSHOOTSTOCED"
).replace("J", "I")


def _plant(pt, square_kw, shifts, *, period=7, drop="J", alpha="KRYPTOS"):
    return encrypt_sub_over_seriated_playfair(
        pt, square_kw, outer_alphabet=alpha, inner_period=period,
        outer_shifts=shifts, drop_letter=drop,
    )


def test_sub_encode_decode_roundtrip():
    alpha = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
    shifts = [3, 17, 0, 9, 22, 5, 14]
    inter = ENGLISH[:120]
    assert sub_decode(sub_encode(inter, alpha, shifts), alpha, shifts) == inter


def test_seriated_inner_roundtrip():
    sq = resolve_square("BUTTERFLY", "J")
    enc = seriated_playfair_encode(ENGLISH, sq, 7)
    dec = seriated_playfair_decode(enc, sq, 7)
    assert len(enc) % 2 == 0
    assert "EARLYINTHEMORNING" in dec.replace("X", "")


def test_digraph_pairs_geometry():
    # 152 letters, period 7: ten full width-7 blocks + a final width-6 block.
    pairs = seriated_digraph_pairs(152, 7)
    assert len(pairs) == 76
    # full-block digraphs pair a position with the one 7 further on (same outer coset)
    assert (0, 7) in pairs and (6, 13) in pairs
    # final width-6 block starts at 140: top 140..145 over bottom 146..151
    assert (140, 146) in pairs and (145, 151) in pairs


def test_recover_outer_key_given_correct_square():
    """Given the correct inner square, the outer Quagmire-7 key + decode are recovered."""
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "BUTTERFLY", shifts)
    sq = resolve_square("BUTTERFLY", "J")
    _rshifts, pt, _score = recover_outer_key_over_seriated_playfair(
        ct, sq, outer_alphabet="KRYPTOS", inner_period=7, outer_period=7, drop_letter="J"
    )
    assert "EARLYINTHEMORNING" in pt.replace("X", "")


def test_crack_ranks_correct_square_far_above_plateau():
    """The full driver puts the correct square on top, clear of the wrong-square plateau."""
    rng = random.Random(2)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "BUTTERFLY", shifts)
    squares = ["RANDOMXYZ", "MEADOW", "BUTTERFLY", "NEEDLE", "SILVER"]
    res = crack_sub_over_seriated_playfair(
        ct, inner_period=7, outer_period=7, squares=squares,
        objective="fitness", drop_letter="J", top=5,
    )
    assert res, "expected candidates"
    best_sq, _key, best_pt, best_score = res[0]
    assert best_sq == resolve_square("BUTTERFLY", "J")
    assert "EARLYINTHEMORNING" in best_pt.replace("X", "")
    assert best_score - res[1][3] > 1.0
