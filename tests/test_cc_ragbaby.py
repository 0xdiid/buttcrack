"""Tests for the Ragbaby cipher."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.ragbaby import Ragbaby, build_alphabet
from buttcrack.scoring import get_scorer

_MERGE = {"J": "I", "X": "W"}


def _prepared(text: str) -> str:
    """The plaintext as Ragbaby sees it: J->I, X->W merge, layout preserved."""
    return "".join(_MERGE.get(ch.upper(), ch.upper()) if ch.isalpha() else ch for ch in text)


def test_build_alphabet_keyword():
    # Spec worked example: FRANKLIN -> drop repeats, append A-Z, merge J->I, X->W.
    assert build_alphabet("FRANKLIN") == "FRANKLIBCDEGHMOPQSTUVWYZ"


def test_build_alphabet_explicit():
    # A key that spells out the finished 24-letter alphabet is used verbatim.
    assert build_alphabet("Keyed alphabet ALPHBETCDFGIKMNOQRSUVWYZ") == "ALPHBETCDFGIKMNOQRSUVWYZ"


def test_vector():
    """Published worked example.

    Source: Wikipedia 'Ragbaby cipher' / dCode.fr canonical example.
    Keyed alphabet ALPHBETCDFGIKMNOQRSUVWYZ. Word 1 "RAG" shifts 1,2,3;
    word 2 "BABY" shifts 2,3,4,5:
        R+1=S, A+2=P, G+3=M, B+2=T, A+3=H, B+4=D, Y+5=H  ->  "SPM THDH".
    """
    cipher = Ragbaby()
    key = "Keyed alphabet ALPHBETCDFGIKMNOQRSUVWYZ"
    ct = cipher.encode("RAG BABY", key)
    assert ct == "SPM THDH"
    # Letters-only, uppercased form matches the published ciphertext exactly.
    assert "".join(c for c in ct if c.isalpha()).upper() == "SPMTHDH"


def test_round_trip():
    cipher = Ragbaby()
    key = "FRANKLIN"
    # Avoid J and X so the merge is a no-op and recovery is exact.
    msg = "ATTACK AT DAWN QUICKLY BEFORE NOON ARRIVES OVER THE HILLTOP"
    prepared = _prepared(msg)
    ct = cipher.encode(msg, key)
    assert cipher.decode(ct, key) == prepared


def test_round_trip_other_keys():
    cipher = Ragbaby()
    msg = "MEET ME BY THE OLD CLOCK TOWER AT MIDNIGHT BRING THE DOCUMENTS"
    for key in ("CRYPTOGRAM", "ZEBRAS", "Keyed alphabet ALPHBETCDFGIKMNOQRSUVWYZ"):
        prepared = _prepared(msg)
        assert cipher.decode(cipher.encode(msg, key), key) == prepared


def test_encode_not_reciprocal():
    # Encrypt != decrypt (count forward vs backward).
    cipher = Ragbaby()
    key = "FRANKLIN"
    msg = "THE QUICK BROWN FOWL"
    assert cipher.encode(msg, key) != cipher.decode(msg, key)


def test_long_word_wraps():
    # A word longer than 24 letters exercises the position-number wrap (>24 -> 1).
    cipher = Ragbaby()
    key = "FRANKLIN"
    msg = "ABRACADABRAABRACADABRAABRACADABRA SHORT"
    prepared = _prepared(msg)
    assert cipher.decode(cipher.encode(msg, key), key) == prepared


@pytest.mark.slow
def test_crack_recovers():
    """The keyless crack recovers plaintext and the keyed alphabet on a long sample.

    The per-position shift schedule makes the score surface rugged, so the
    hill-climb needs a long sample (~720 letters) and many random restarts.
    """
    cipher = Ragbaby()
    scorer = get_scorer()
    key = "FRANKLIN"
    # Uppercase prepared stream with no J/X (which would merge and be lost), so
    # the crack can recover it character-for-character.
    plaintext = (
        "THE ANALYSIS ROUTINES NEED A HEALTHY STRETCH OF PERFECTLY ORDINARY ENGLISH "
        "PROSE SO THAT THE FREQUENCY STATISTICS AND QUADGRAM FITNESS SCORES CAN LOCK "
        "ONTO THE UNDERLYING MESSAGE AND RECOVER THE ORIGINAL CONTENT WITHOUT PRIOR "
        "KNOWLEDGE OF THE SECRET KEY THAT WAS CHOSEN TO ENCIPHER IT IN THE BEGINNING "
        "AND WE ADD STILL MORE WORDS HERE TO GIVE THE CLIMBER A FAIR AMOUNT OF SIGNAL "
        "TO WORK WITH AS IT SEARCHES OVER THE KEYED ALPHABET PERMUTATIONS ONE BY ONE"
    ) * 2
    ct = cipher.encode(plaintext, key)
    results = cipher.crack(ct, scorer, top=3, rng=random.Random(7), timeout=120)
    assert results, "crack returned no candidates"
    assert results[0].plaintext == plaintext
    assert results[0].key == "FRANKLIBCDEGHMOPQSTUVWYZ"
