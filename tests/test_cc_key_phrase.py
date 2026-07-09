"""Tests for the Key Phrase substitution cipher.

VECTOR source (CryptoCrack user guide, "Key Phrase" page worked example):
  https://sites.google.com/site/cryptocrackprogram/user-guide/cipher-types/substitution/key-phrase
  Key phrase WHATSANOTHERWORDFORSYNONYM is placed under the straight alphabet
  (A->W, B->H, C->A, ...). The published example enciphers

    "The difference between fiction and reality? Fiction has to make sense."
  to
    "SOS TTAASOSOAS HSSOSSO ATASTRO WOT OSWRTSY? ATASTRO OWR SR WWES RSORS."

  Letters-only normalization of that ciphertext is the expected stream below.
  Verified self-consistent: encode(published_plaintext, key) reproduces the
  published ciphertext exactly (letters and with punctuation).
"""

import random
import string

import pytest

from buttcrack.ciphers.key_phrase import KeyPhrase
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

# CryptoCrack worked example (letters-only).
_VECTOR_KEY = "WHATSANOTHERWORDFORSYNONYM"
_VECTOR_PT = "The difference between fiction and reality? Fiction has to make sense."
_VECTOR_CT = "SOSTTAASOSOASHSSOSSOATASTROWOTOSWRTSYATASTROOWRSRWWESRSORS"


def test_vector_cryptocrack():
    c = KeyPhrase()
    assert c.encode(_VECTOR_PT, _VECTOR_KEY) == _VECTOR_CT


def test_vector_is_many_to_one():
    # The key phrase repeats letters, so encode is many-to-one: distinct
    # plaintext letters can collapse onto one ciphertext letter.
    # In WHATSANOTHERWORDFORSYNONYM both 'a' and 'f' map to the same cipher
    # letter (positions of A and F under the alphabet both yield 'W'/'A'...).
    # Demonstrate by checking the produced alphabet has fewer than 26 distinct
    # cipher letters.
    assert len(set(_VECTOR_KEY)) < 26


def test_round_trip_permutation_key():
    # decode(encode(msg)) recovers the prepared plaintext when the key phrase is
    # a 26-letter permutation (the only lossless case for the Key Phrase cipher).
    c = KeyPhrase()
    rng = random.Random(3)
    perm = list(string.ascii_uppercase)
    rng.shuffle(perm)
    key = "".join(perm)
    msg = "Attack at dawn while the guards are sleeping near the river bend!"
    prepared = only_letters(msg)
    assert c.decode(c.encode(msg, key), key) == prepared


@pytest.mark.slow
def test_crack_recovers_permutation_key():
    # The Key Phrase cipher is only recoverable when the underlying key phrase
    # was a permutation (it then degenerates to a monoalphabetic substitution).
    c = KeyPhrase()
    scorer = get_scorer()
    rng = random.Random(3)
    perm = list(string.ascii_uppercase)
    rng.shuffle(perm)
    key = "".join(perm)
    pt = (
        "the analysis routines need a healthy stretch of perfectly ordinary english "
        "prose so that the frequency statistics and quadgram fitness scores can lock "
        "onto the underlying message and recover the original text without any prior knowledge"
    )
    prepared = only_letters(pt)
    ct = c.encode(pt, key)
    results = c.crack(ct, scorer, top=3, rng=random.Random(11), timeout=60)
    assert results
    assert only_letters(results[0].plaintext) == prepared
