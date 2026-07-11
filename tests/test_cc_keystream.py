"""Tests for the self-generating keystream ciphers (linear-recurrence + LCG)."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.keystream import (
    LcgKeystream,
    LinearRecurrenceKeystream,
    lcg_stream,
    linrec_stream,
)
from buttcrack.scoring import get_scorer

PLAIN = (
    "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHEEARLYMORNINGSUNROSESLOWLYOVERTHE"
    "QUIETVILLAGE"
)


def test_linrec_stream_is_lagged_fibonacci():
    # coeffs 1,1 is chain addition: k[i] = k[i-1] + k[i-2] mod 26.
    assert linrec_stream([1, 1], [1, 1], 7) == [1, 1, 2, 3, 5, 8, 13]


def test_lcg_stream_recurrence():
    assert lcg_stream(3, 1, 0, 5) == [0, 1, 4, 13, 40 % 26]


@pytest.mark.parametrize("combiner", ["vigenere", "beaufort", "variant"])
def test_linrec_roundtrip(combiner):
    c = LinearRecurrenceKeystream()
    key = f"2,1,3/5,9,4/{combiner}"
    assert c.decode(c.encode(PLAIN, key), key) == PLAIN


@pytest.mark.parametrize("combiner", ["vigenere", "beaufort", "variant"])
def test_lcg_roundtrip(combiner):
    c = LcgKeystream()
    key = f"7,3,11/{combiner}"
    assert c.decode(c.encode(PLAIN, key), key) == PLAIN


def test_linrec_letter_seed_form():
    # Seeds/coeffs may be given as letters (A=0..Z=25); "1,1/HE" == seed H,E.
    c = LinearRecurrenceKeystream()
    key_digits = "1,1/7,4"
    key_letters = "B,B/HE"  # coeffs B,B = 1,1 ; seed H,E = 7,4
    assert c.encode(PLAIN, key_digits) == c.encode(PLAIN, key_letters)


def test_linrec_blind_crack_order2():
    scorer = get_scorer()
    c = LinearRecurrenceKeystream()
    key = "3,1/8,5/vigenere"  # order-2 recurrence, fully blind-searchable
    ct = c.encode(PLAIN, key)
    out = c.crack(ct, scorer, top=5, rng=random.Random(0))
    assert out, "expected candidates"
    assert out[0].plaintext.upper().replace(" ", "") == PLAIN


def test_linrec_crack_fixed_coeffs_order3():
    scorer = get_scorer()
    c = LinearRecurrenceKeystream()
    key = "1,1,1/4,9,2/beaufort"
    ct = c.encode(PLAIN, key)
    out = c.crack(ct, scorer, top=5, rng=random.Random(0), coeffs="1,1,1", combiner="beaufort")
    assert out
    assert out[0].plaintext.upper().replace(" ", "") == PLAIN


def test_lcg_blind_crack():
    scorer = get_scorer()
    c = LcgKeystream()
    key = "5,7,3/vigenere"
    ct = c.encode(PLAIN, key)
    out = c.crack(ct, scorer, top=5, rng=random.Random(0))
    assert out
    assert out[0].plaintext.upper().replace(" ", "") == PLAIN


def test_registered_in_registry():
    from buttcrack import registry

    assert registry.get("keystream").name == "keystream"
    assert registry.get("lfsr").name == "keystream"
    assert registry.get("lcg").name == "lcg"
