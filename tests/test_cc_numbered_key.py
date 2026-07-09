"""Tests for the ACA Numbered Key cipher.

Authoritative vector: the worked example on the ACA cipher sheet
NumberedKey.pdf (cryptogram.org/downloads/aca.info/ciphers/NumberedKey.pdf,
mirrored at the CryptoCrack user guide). Key phrase "I like ciphers." rotated by
offset 18 yields the numbered key

    00=m 01=n 02=o 03=q 04=t 05=u 06=v 07=w 08=x 09=y 10=z 11=i 12=l 13=i
    14=k 15=e 16=c 17=i 18=p 19=h 20=e 21=r 22=s 23=a 24=b 25=d 26=f 27=g 28=j

and the sheet's ciphertext

    04 19 20 21 02 23 25 04 02 22 05 16 16 15 22 22 11 22 23 12 07 23
    09 22 05 01 25 20 21 16 02 01 22 04 21 05 16 04 17 02 01

deciphers to "THE ROAD TO SUCCESS IS ALWAYS UNDER CONSTRUCTION". We verified this
vector is self-consistent (every number decodes uniquely; the multi-homophone
letters e and i use varied homophones exactly as the sheet shows).
"""

from __future__ import annotations

import random

from buttcrack.ciphers.numbered_key import NumberedKey

KEY = "I like ciphers./18"

# Authoritative ACA ciphertext -> plaintext vector (NumberedKey.pdf worked example).
ACA_CT = (
    "04 19 20 21 02 23 25 04 02 22 05 16 16 15 22 22 11 22 23 12 07 23 "
    "09 22 05 01 25 20 21 16 02 01 22 04 21 05 16 04 17 02 01"
)
ACA_PT = "THEROADTOSUCCESSISALWAYSUNDERCONSTRUCTION"


def test_vector_aca_decode():
    """Decode the published ACA ciphertext to the published plaintext (exact)."""
    assert NumberedKey().decode(ACA_CT, KEY) == ACA_PT


def test_vector_numbered_table():
    """The rotated/numbered key reproduces the ACA sheet's table exactly."""
    cipher = NumberedKey()
    # Each published number, decoded singly, must be the sheet's letter.
    table = {
        0: "M",
        1: "N",
        2: "O",
        3: "Q",
        4: "T",
        5: "U",
        6: "V",
        7: "W",
        8: "X",
        9: "Y",
        10: "Z",
        11: "I",
        12: "L",
        13: "I",
        14: "K",
        15: "E",
        16: "C",
        17: "I",
        18: "P",
        19: "H",
        20: "E",
        21: "R",
        22: "S",
        23: "A",
        24: "B",
        25: "D",
        26: "F",
        27: "G",
        28: "J",
    }
    for num, letter in table.items():
        assert cipher.decode(f"{num:02d}", KEY) == letter


def test_encode_deterministic_roundtrip():
    """Deterministic encode (lowest homophone) round-trips the prepared plaintext."""
    cipher = NumberedKey()
    msg = "THE ROAD TO SUCCESS IS ALWAYS UNDER CONSTRUCTION"
    prepared = "".join(c for c in msg.upper() if c.isalpha())
    ct = cipher.encode(msg, KEY)
    assert cipher.decode(ct, KEY) == prepared


def test_roundtrip_random_keys():
    rng = random.Random(7)
    cipher = NumberedKey()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for _ in range(20):
        phrase = "".join(rng.choice(alphabet) for _ in range(rng.randint(8, 20)))
        offset = rng.randint(0, 30)
        key = f"{phrase}/{offset}"
        msg = "".join(rng.choice(alphabet) for _ in range(rng.randint(20, 60)))
        # randomized encode must still decode back to the prepared plaintext
        ct = cipher.encode(msg, key, rng=random.Random(rng.random()))
        assert cipher.decode(ct, key) == msg


def test_crack_returns_empty():
    # Keyless cracking is not attempted (homophonic + deceptive n-gram landscape).
    cipher = NumberedKey()
    ct = cipher.encode("THE ROAD TO SUCCESS", "I like ciphers./18")
    assert cipher.crack(ct, scorer=_DummyScorer(), timeout=1.0) == []


class _DummyScorer:
    def score(self, text: str) -> float:  # pragma: no cover - crack returns early
        return 0.0

    def confidence(self, text: str) -> float:  # pragma: no cover
        return 0.0
