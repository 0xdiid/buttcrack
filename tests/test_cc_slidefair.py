"""Tests for the Slidefair cipher.

Vector source: ACA cipher description PDF "SLIDEFAIR"
(cryptogram.org/downloads/aca.info/ciphers/Slidefair.pdf), verified
computationally with C1=(P2-k), C2=(P1+k) and the vertical-pair shift-right rule.
"""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.slidefair import Slidefair
from buttcrack.scoring import get_scorer

# The ACA Slidefair example: plaintext digraphs
# TH ES LI DE FA IR CA NB EU SE DW IT HV IG EN ER EV AR IA NT OR BE AU FO RT
# under keyword DIGRAPH (Vigenere table) give the ciphertext below.
VECTOR_PLAINTEXT = "THESLIDEFAIRCANBEUSEDWITHVIGENEREVARIANTORBEAUFORT"
VECTOR_KEY = "DIGRAPH"
VECTOR_CIPHERTEXT = "EWKMCRNUAFCXTJYQMMYYFUTIGWZPKHJMPKBSAIECKVCFMIILCI"

# Small per-pair examples from the ACA table-of-examples (key letter B):
# pt 'ca' -> ZD (Vigenere), BB (Variant), BZ (Beaufort)
# pt 'de' -> EF (Vigenere), FC (Variant), XY (Beaufort)


def test_vector_vigenere():
    sf = Slidefair()
    assert sf.encode(VECTOR_PLAINTEXT, VECTOR_KEY) == VECTOR_CIPHERTEXT


def test_vector_pairs_all_tables():
    sf = Slidefair()
    assert sf.encode("CA", "B/VIGENERE") == "ZD"
    assert sf.encode("DE", "B/VIGENERE") == "EF"
    assert sf.encode("CA", "B/VARIANT") == "BB"
    assert sf.encode("DE", "B/VARIANT") == "FC"
    assert sf.encode("CA", "B/BEAUFORT") == "BZ"
    assert sf.encode("DE", "B/BEAUFORT") == "XY"


def test_vector_decode_roundtrips():
    sf = Slidefair()
    assert sf.decode(VECTOR_CIPHERTEXT, VECTOR_KEY) == VECTOR_PLAINTEXT


@pytest.mark.parametrize("table", ["VIGENERE", "VARIANT", "BEAUFORT"])
def test_roundtrip(table):
    sf = Slidefair()
    msg = "WEAREDISCOVEREDFLEEATONCEANDTAKETHEGOLDWITHYOU"  # even length
    key = f"SECRET/{table}"
    assert sf.decode(sf.encode(msg, key), key) == msg


def test_roundtrip_odd_padding():
    sf = Slidefair()
    # Odd-length plaintext pads with X; decode recovers the padded stream.
    msg = "ATTACKATDAWNXYZ"  # 15 letters -> padded to 16 on encode
    key = "KEYWORD"
    ct = sf.encode(msg, key)
    assert sf.decode(ct, key) == msg + "X"


@pytest.mark.slow
def test_crack_recovers_vigenere():
    sf = Slidefair()
    scorer = get_scorer()
    plaintext = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOM"
        "ITWASTHEAGEOFFOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCH"
        "OFINCREDULITYITWASTHESEASONOFLIGHTITWASTHESEASONOFDARKNESS"
    )
    key = "MARTIN/VIGENERE"
    ct = sf.encode(plaintext, key)
    rng = random.Random(12345)
    results = sf.crack(ct, scorer, top=5, rng=rng, timeout=60.0, key_length=6, tables=("VIGENERE",))
    assert results, "crack returned no candidates"
    best_plain = "".join(c for c in results[0].plaintext.upper() if "A" <= c <= "Z")
    expected = plaintext + "X" if len(plaintext) % 2 else plaintext
    assert best_plain == expected
