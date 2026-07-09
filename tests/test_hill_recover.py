"""Blind 3x3 Hill recovery by row decomposition (buttcrack.ciphers._hill_recover).

These pin the capability the old 2x2-only crack declared out of reach: recovering a 3x3
Hill (and its affine / keyed-alphabet generalisation) from ciphertext alone. The headline
regression is a hard synthetic instance — a 3x3 Hill (keyword SIGNATURE) over the KRYPTOS
keyed alphabet with a period-2 additive (VELVET) — recovered from its first 153 letters,
the length at which a matrix brute is hopeless.
"""

from __future__ import annotations

import pytest

from buttcrack.ciphers.hill import Hill
from buttcrack.ciphers import _hill_recover as hr
from buttcrack.scoring import get_scorer

KRY = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Synthetic affine-Hill vector: 3x3 Hill (SIGNATURE, KRYPTOS index) applied to
# plaintext + period-2 additive (VELVET), emitted via the KRYPTOS alphabet.
AFFINE_CT = (
    "ERUJOVKDBMFWPURZAMKHPPAUEJVWCYTNNBHYVDENJYGEKAZXDWBGBWWIYIXOVBTEGICQFVSGBV"
    "BOMGHOHBRGXPCBLEMIUFGCTJVJYQWIYGGYGBAEPVDJBBUIXFELOYLIBXRBCOJVWRERUNFRPTJI"
    "NKGEKGFUYYNGBWKBUEUKRWVDPNFPQPWQRRFEFXVQZLEVZLGJMILIBOJGQFSJJSJBDRIXZHBYJE"
    "MWAUCAHCPNAHFKVNKAKHSTHM"
)
AFFINE_START = "THESTATIONMASTERLIGHTSTHESIGNA"


def _prep(msg: str) -> str:
    p = "".join(c for c in msg.upper() if "A" <= c <= "Z")
    if len(p) % 3:
        p += "X" * (3 - len(p) % 3)
    return p


def test_projective_row_count():
    """The covector quotient has the expected number of representatives."""
    assert len(hr._ROWS) == 1471


def test_recover_pure_hill_std():
    """A synthetic 3x3 Hill over the standard alphabet is recovered exactly."""
    pt = _prep(
        "MEET ME AT DAWN BY THE OLD OAK TREE ON THE NORTH SIDE OF THE RIVER WHERE THE "
        "WATER IS SHALLOW AND BRING THE MAPS AND THE LANTERNS SO WE CAN FIND THE DOOR"
    )
    ct = Hill().encode(pt, "6,24,1,13,16,10,20,17,15")
    recs = hr.recover(ct, get_scorer(), alphabet=STD, q_values=(1,), top=3)
    assert recs and recs[0].plaintext.startswith(pt[:60])
    assert recs[0].is_plain_hill


def test_recover_keyed_affine_full():
    """Keyed alphabet + period-2 additive recovered from the full ciphertext."""
    recs = hr.recover(AFFINE_CT, get_scorer(), alphabet=KRY, q_values=(1, 2), top=2)
    assert recs and recs[0].plaintext.startswith(AFFINE_START)
    assert recs[0].q == 2
    # exact decryption matrix (inverse of the SIGNATURE matrix in KRYPTOS index)
    assert [list(r) for r in recs[0].decrypt_matrix] == [[23, 18, 23], [3, 11, 9], [22, 19, 5]]


def test_recover_keyed_affine_from_153_letters():
    """The hard case: recovery from only 153 letters (pair_brute rescues an outlier row)."""
    recs = hr.recover(AFFINE_CT[:153], get_scorer(), alphabet=KRY, q_values=(2,), top=2, pair_brute=True)
    assert recs and recs[0].plaintext.startswith(AFFINE_START)


def test_recover_rejects_non_english():
    """Random letters yield no confidently-English recovery (no false positive)."""
    import random

    rng = random.Random(0)
    junk = "".join(rng.choice(STD) for _ in range(156))
    scorer = get_scorer()
    recs = hr.recover(junk, scorer, alphabet=STD, q_values=(1,), top=1)
    # best recovery must stay under the 0.5 confidence the cracker treats as "solved"
    assert not recs or scorer.confidence(recs[0].plaintext) < 0.45


def test_crack3_roundtrips_via_hill_key():
    """Hill.crack recovers a 3x3 and the reported key round-trips through decode."""
    pt = _prep(
        "IN THE MORNING THE LIBRARIAN OPENED EVERY WINDOW AND LET THE COOL AIR MOVE "
        "THROUGH THE READING ROOM WHERE STUDENTS HAD GATHERED TO STUDY THEIR LESSONS "
        "BEFORE THE EXAMS BEGAN AND THE OLD CLOCK ABOVE THE DOOR MARKED EACH HOUR"
    )
    key = "GYBNQKURP"  # 9-letter keyword -> invertible 3x3
    ct = Hill().encode(pt, key)
    res = Hill().crack(ct, get_scorer(), top=3, timeout=30)
    assert res and res[0].confidence > 0.5
    assert res[0].meta.get("n") == 3
    assert Hill().decode(ct, res[0].key).startswith(pt[:40])


@pytest.mark.slow
def test_crack_keyed_affine_via_opts():
    """Hill.crack recovers the affine instance when told alphabet and additive period."""
    res = Hill().crack(AFFINE_CT, get_scorer(), top=2, alphabet="KRYPTOS", q_values=(1, 2), timeout=60)
    assert res and res[0].plaintext.replace(" ", "").startswith(AFFINE_START)
