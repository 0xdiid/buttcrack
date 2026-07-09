"""Tests for decoupled recovery of a periodic substitution over a Two-square inner.

CT = periodic_Vigenere/Quagmire( two_square(PT) ). The double-Playfair sibling of
``sub_four_square``: two keyed Q-dropped squares, reciprocal with a transparency rule, so the
outer key is recovered per candidate (pair, layout) by a drop-letter-pruned coordinate descent.
"""

from __future__ import annotations

import random

from buttcrack.sub_two_square import (
    crack_sub_over_two_square,
    encrypt_sub_over_two_square,
    fs_grid,
    recover_outer_key_over_two_square,
    sub_decode,
    sub_encode,
    two_square_transform,
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


def test_two_square_is_involution():
    tg, bg = fs_grid("WATERMELON", "Q"), fs_grid("LAVENDER", "Q")
    msg = ENGLISH.replace("Q", "")  # Q is dropped from the square alphabet
    for vertical in (True, False):
        enc = two_square_transform(msg, tg, bg, vertical=vertical)
        dec = two_square_transform(enc, tg, bg, vertical=vertical)
        assert dec == msg[: len(dec)]  # reciprocal map recovers the message


def test_recover_outer_key_given_correct_pair():
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = encrypt_sub_over_two_square(ENGLISH, "WATERMELON", "LAVENDER", outer_shifts=shifts)
    tg, bg = fs_grid("WATERMELON", "Q"), fs_grid("LAVENDER", "Q")
    _sh, pt, _sc = recover_outer_key_over_two_square(ct, tg, bg, outer_period=7, vertical=True)
    assert "EARLYINTHEMORNING" in pt


def test_crack_ranks_correct_pair_above_plateau():
    rng = random.Random(2)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = encrypt_sub_over_two_square(ENGLISH, "WATERMELON", "LAVENDER", outer_shifts=shifts)
    words = ["RANDOMKEY", "MEADOW", "WATERMELON", "LAVENDER", "SILVER"]
    res = crack_sub_over_two_square(
        ct, outer_period=7, top_squares=words, bot_squares=words, objective="fitness", top=5
    )
    assert res
    tg, bg, vertical, _key, pt, score = res[0]
    assert tg == fs_grid("WATERMELON", "Q") and bg == fs_grid("LAVENDER", "Q")
    assert vertical is True
    assert "EARLYINTHEMORNING" in pt
    # Two-square's transparency rule leaves wrong pairs partly readable, so the margin is
    # smaller than four-square's; the correct pair still ranks first, clear of the runner-up.
    assert score > res[1][5]
