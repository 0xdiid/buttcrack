"""Crib-anchored crack of a monoalphabetic substitution over a columnar transposition.

The keyed layered class blind search can't touch (no gradient). A crib makes it
tractable: untranspose(ct) == sub(plaintext), so for the right column order the
un-transposed stream contains sub(crib); brute-rank column orders by the crib's
partial-decryption quadgram, then solve the residual substitution.
"""

import random
import string

import pytest

from buttcrack import engine
from buttcrack.ciphers.columnar import _encode_letters
from buttcrack.text import only_letters

A = string.ascii_uppercase
PLAIN = only_letters(
    "IN THE BEGINNING THE UNIVERSE WAS CREATED THIS HAS MADE A LOT OF PEOPLE VERY ANGRY AND "
    "BEEN WIDELY REGARDED AS A BAD MOVE MANY RACES BELIEVE THAT IT WAS CREATED BY SOME SORT "
    "OF GOD ALTHOUGH THE PEOPLE OF VILTVODLE SIX FIRMLY BELIEVE THE ENTIRE UNIVERSE WAS "
    "SNEEZED OUT OF THE NOSE OF A BEING CALLED THE GREAT GREEN ARKLESEIZURE SPACE IS BIG"
)


def _make_ct(width: int, sub_seed: int, order_seed: int) -> tuple[str, list[int]]:
    sub = {p: c for p, c in zip(A, random.Random(sub_seed).sample(list(A), 26), strict=True)}
    order = list(range(width))
    random.Random(order_seed).shuffle(order)
    ct = "".join(sub[c] for c in _encode_letters(PLAIN, order))
    return ct, order


@pytest.mark.slow
def test_crib_recovers_keyed_sub_over_columnar():
    ct, _ = _make_ct(width=6, sub_seed=9, order_seed=3)
    hit = engine._layered_crib_crack(
        ct, "UNIVERSE", seed=1, timeout=40, lang="english", max_width=7
    )
    assert hit is not None, "crib crack returned nothing"
    cand, note = hit
    assert only_letters(cand.plaintext) == PLAIN
    assert "UNIVERSE" in cand.plaintext
    assert cand.cipher == "substitution+columnar"
    assert cand.meta["width"] == 6


@pytest.mark.slow
def test_transpose_outer_layer_order_also_recovers():
    # ct = transpose(sub(pt)) — the other layer order; untranspose(ct) == sub(pt) too.
    sub = {p: c for p, c in zip(A, random.Random(9).sample(list(A), 26), strict=True)}
    order = list(range(6))
    random.Random(3).shuffle(order)
    ct = _encode_letters("".join(sub[c] for c in PLAIN), order)
    hit = engine._layered_crib_crack(
        ct, "UNIVERSE", seed=1, timeout=40, lang="english", max_width=7
    )
    assert hit is not None
    assert only_letters(hit[0].plaintext) == PLAIN


@pytest.mark.slow
def test_wrong_crib_yields_no_false_positive():
    ct, _ = _make_ct(width=6, sub_seed=9, order_seed=3)
    # ZEBRAFISH is not in the plaintext: the crack must not invent a "solution".
    hit = engine._layered_crib_crack(
        ct, "ZEBRAFISH", seed=1, timeout=20, lang="english", max_width=6
    )
    assert hit is None


def test_short_input_and_short_crib_are_skipped():
    ct, _ = _make_ct(width=6, sub_seed=9, order_seed=3)
    short_crib = engine._layered_crib_crack(ct, "ABC", seed=1, timeout=5, lang="english")
    short_text = engine._layered_crib_crack(ct[:120], "UNIVERSE", seed=1, timeout=5, lang="english")
    assert short_crib is None
    assert short_text is None
