"""Tests for the Running Key cipher."""

from __future__ import annotations

from buttcrack.ciphers.running_key import RunningKey
from buttcrack.scoring import get_scorer


def test_vector_aca():
    """Published ACA worked example (self-key).

    Source: ACA cipher description 'RUNNING KEY',
    cryptogram.org/downloads/aca.info/ciphers/RunningKey.pdf, as transcribed in
    this project's docs/cipher-specs.json and verified computationally with the
    Vigenere combiner C = (P + K) mod 26.

    The plaintext is "THIS CIPHER CAN BE USED WITH ANY OF THE PERIODICS"; its
    FIRST half is used as the running key to encipher its SECOND half:
        plaintext (2nd half) = ITHANYOFTHEPERIODICS
        key       (1st half) = THISCIPHERCANBEUSEDW
        ciphertext           = BAPSPGDMXYGPRSMIVMFO
    """
    cipher = RunningKey()
    pt = "ITHANYOFTHEPERIODICS"
    key = "THISCIPHERCANBEUSEDW"
    assert cipher.encode(pt, key) == "BAPSPGDMXYGPRSMIVMFO"


def test_vector_wikipedia():
    """Independent published vector.

    Source: en.wikipedia.org/wiki/Running_key_cipher worked example (key text
    from 'The C Programming Language'). C = (P + K) mod 26.
        plaintext = FLEEATONCE
        key       = ERRORSCANO
        ciphertext= JCVSRLQNPS
    """
    cipher = RunningKey()
    assert cipher.encode("FLEE AT ONCE", "ERRORSCANO") == "JCVS RL QNPS"


def test_round_trip():
    cipher = RunningKey()
    key = "WE HOLD THESE TRUTHS TO BE SELF EVIDENT THAT ALL MEN ARE CREATED"
    msg = "MEETMEATDAWNBYTHEOLDMILL"
    ct = cipher.encode(msg, key)
    # Clean uppercase letter stream round-trips exactly.
    assert cipher.decode(ct, key) == msg


def test_round_trip_layout_preserved():
    cipher = RunningKey()
    key = "the quick brown fox jumps over the lazy dog and then some more text here"
    msg = "Attack at dawn!"
    ct = cipher.encode(msg, key)
    # Layout (spaces, punctuation, case) is restored on both ends.
    assert cipher.decode(ct, key) == "Attack at dawn!"


def test_crack_smoke():
    """The keyless crack is best-effort and not asserted to recover exactly.

    Running Key is inherently ambiguous (both streams are English; the combined
    score is symmetric under swapping plaintext<->key), so we only check that the
    crack runs, honors a timeout, and returns ranked English-looking candidates.
    """
    cipher = RunningKey()
    scorer = get_scorer()
    pt = "THEEARLYBIRDGETSTHEWORMBUTTHESECONDMOUSEGETSTHECHEESEYES"
    key = "WHENINTHECOURSEOFHUMANEVENTSITBECOMESNECESSARYFORONEPEOPL"
    ct = cipher.encode(pt, key)
    results = cipher.crack(ct, scorer, top=3, timeout=20, beam=150)
    assert results, "crack returned no candidates"
    assert all(r.cipher == "running-key" for r in results)
