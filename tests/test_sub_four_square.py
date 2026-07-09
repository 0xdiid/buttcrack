"""Tests for decoupled recovery of a periodic substitution over a Four-square inner.

CT = periodic_Vigenere/Quagmire( four_square(PT) ). The two-keyed-square sibling of
``sub_playfair``: Four-square emits from two 25-cell cipher squares (drop-letter free), so the
outer key is recovered per candidate square PAIR by a drop-letter-pruned coordinate descent
(there is no no-doubled-digraph rule for Four-square).
"""

from __future__ import annotations

import random

from buttcrack.sub_four_square import (
    crack_sub_over_four_square,
    encrypt_sub_over_four_square,
    four_square_decode,
    four_square_encode,
    fs_alphabet,
    fs_grid,
    recover_outer_key_over_four_square,
    sub_decode,
    sub_encode,
)
from buttcrack.text import only_letters

ENGLISH = only_letters(
    "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGEACHTREEFORFR"
    "OSTHEPRUNESTHEDEADBRANCHESWITHASMALLCURVEDKNIFEANDTIESTHEYOUNGSHOOTSTOCED"
)


def test_sub_encode_decode_roundtrip():
    alpha = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
    shifts = [3, 17, 0, 9, 22, 5, 14]
    inter = ENGLISH[:120]
    assert sub_decode(sub_encode(inter, alpha, shifts), alpha, shifts) == inter


def test_four_square_inner_roundtrip():
    a25 = fs_alphabet("Q")
    tg, bg = fs_grid("WATERMELON", "Q"), fs_grid("LAVENDER", "Q")
    enc = four_square_encode(ENGLISH, tg, bg, a25)
    dec = four_square_decode(enc, tg, bg, a25)
    assert len(enc) % 2 == 0
    assert dec[:40] == ENGLISH[:40]


def test_recover_outer_key_given_correct_pair():
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = encrypt_sub_over_four_square(ENGLISH, "WATERMELON", "LAVENDER", outer_shifts=shifts)
    tg, bg = fs_grid("WATERMELON", "Q"), fs_grid("LAVENDER", "Q")
    _sh, pt, _sc = recover_outer_key_over_four_square(ct, tg, bg, outer_period=7)
    assert "EARLYINTHEMORNING" in pt


def test_crack_ranks_correct_pair_above_plateau():
    rng = random.Random(2)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = encrypt_sub_over_four_square(ENGLISH, "WATERMELON", "LAVENDER", outer_shifts=shifts)
    words = ["RANDOMKEY", "MEADOW", "WATERMELON", "LAVENDER", "SILVER"]
    res = crack_sub_over_four_square(
        ct, outer_period=7, tr_squares=words, bl_squares=words, objective="fitness", top=5
    )
    assert res
    tg, bg, _key, pt, score = res[0]
    assert tg == fs_grid("WATERMELON", "Q") and bg == fs_grid("LAVENDER", "Q")
    assert "EARLYINTHEMORNING" in pt
    assert score - res[1][4] > 1.0
