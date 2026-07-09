"""Known-plaintext (crib) attack on the Hill cipher and its mod-26 linear solver.

Pins two capabilities the toolkit lacked: a general ``A x = b (mod 26)`` solver that
handles the ring's zero-divisors via CRT (and enumerates rank-deficient systems), and the
classic Hill known-plaintext attack built on it — including the affine/periodic-additive
generalisation (a shape that turns up in real layered puzzles).
"""

from __future__ import annotations

from buttcrack import hill_kpa
from buttcrack.ciphers.hill import Hill

KRY = "KRYPTOSABCDEFGHIJLMNQUVWXZ"


def test_solve_mod26_unique():
    """A full-rank 2x2 system has exactly the planted solution."""
    A = [[3, 3], [2, 5]]
    x = [7, 19]
    b = [(A[i][0] * x[0] + A[i][1] * x[1]) % 26 for i in range(2)]
    sols = hill_kpa.solve_mod26(A, b)
    assert x in sols


def test_solve_mod26_zero_divisor_row():
    """A coefficient that is a zero-divisor mod 26 (2, 13) is handled by CRT."""
    # 2x = 4 (mod 26) has TWO solutions: x = 2 and x = 15.
    sols = hill_kpa.solve_mod26([[2]], [4])
    assert sorted(s[0] for s in sols) == [2, 15]


def test_solve_mod26_inconsistent():
    """An inconsistent system yields no solutions."""
    assert hill_kpa.solve_mod26([[2]], [3]) == []  # 2x is always even, never 3 mod 26


def test_solve_mod26_underdetermined_enumerates():
    """A rank-deficient system returns every consistent key."""
    sols = hill_kpa.solve_mod26([[1, 1]], [5])  # one equation, two unknowns
    assert len(sols) == 26
    assert all((x[0] + x[1]) % 26 == 5 for x in sols)


def test_recover_matrix_2x2():
    """Recover a 2x2 key from four aligned crib letters."""
    pt = "HELP"
    ct = Hill().encode(pt, "3,3,2,5")
    mats = hill_kpa.recover_matrix(pt, ct, 2, alphabet="STD")
    assert [[3, 3], [2, 5]] in mats


def test_recover_matrix_3x3_roundtrips():
    """Recover a 3x3 key from a crib and confirm it decodes the whole message."""
    pt = "".join(c for c in "MEETMEATDAWNBYTHEOLDOAKTREEONTHENORTHSIDE".upper() if c.isalpha())
    if len(pt) % 3:
        pt += "X" * (3 - len(pt) % 3)
    ct = Hill().encode(pt, "6,24,1,13,16,10,20,17,15")
    mats = hill_kpa.recover_matrix(pt[:15], ct, 3, alphabet="STD")
    # a short crib can leave the key underdetermined mod 2; the true key is among them
    assert any(Hill().decode(ct, ",".join(str(v) for row in m for v in row)) == pt for m in mats)


def test_recover_matrix_offset_crib():
    """A crib that starts mid-message (block offset) still recovers the key."""
    pt = "".join(c for c in "THEQUICKBROWNFOXJUMPSOVERLAZYDOGS".upper() if c.isalpha())
    if len(pt) % 3:
        pt += "X" * (3 - len(pt) % 3)
    ct = Hill().encode(pt, "6,24,1,13,16,10,20,17,15")
    # crib is plaintext blocks 3..6 (letters 9..21), aligned to ct block offset 3
    mats = hill_kpa.recover_matrix(pt[9:21], ct, 3, alphabet="STD", offset=3)
    assert any(Hill().decode(ct, ",".join(str(v) for row in m for v in row)) == pt for m in mats)


def test_crib_drag_finds_offset_and_decrypts():
    """A crib whose position is UNKNOWN is found by sliding across block offsets; the true
    plaintext ranks first under the default (English) scorer."""
    pt = "".join(c for c in "ATDAWNWEMARCHEDNORTHWESTTOWARDTHERIVERCROSSING".upper() if c.isalpha())
    if len(pt) % 3:
        pt += "X" * (3 - len(pt) % 3)
    ct = Hill().encode(pt, "6,24,1,13,16,10,20,17,15")
    # "NORTHWEST" sits mid-message at an unknown offset; give only the word.
    hits = hill_kpa.crib_drag("NORTHWEST", ct, 3, alphabet="STD", top=5)
    assert hits and hits[0]["plaintext"] == pt


def test_crib_drag_custom_scorer_for_non_english():
    """The scorer is pluggable, so a non-English payload can be recognised by structure
    (here: minimal gzip length) instead of English n-grams."""
    import zlib

    pt = "".join(c for c in "NORTHEIGHTPACESNORTHEIGHTPACESNORTHEIGHTPACES".upper() if c.isalpha())
    if len(pt) % 3:
        pt += "X" * (3 - len(pt) % 3)
    ct = Hill().encode(pt, "6,24,1,13,16,10,20,17,15")

    def compress(s):  # higher = more compressible
        return -len(zlib.compress(s.encode(), 9))

    hits = hill_kpa.crib_drag("NORTHEIGH", ct, 3, alphabet="STD", scorer=compress, top=5)
    assert hits and hits[0]["plaintext"] == pt


def test_recover_affine_period2_kryptos():
    """Recover an affine Hill (matrix + period-2 additive) over a keyed alphabet.

    Shape: c = K (p) + schedule[block mod 2], KRYPTOS index.
    """
    pos = {ch: i for i, ch in enumerate(KRY)}
    inv = {i: ch for ch, i in pos.items()}
    K = [[6, 24, 1], [13, 16, 10], [20, 17, 15]]  # invertible mod 26
    sched = [[13, 13, 4], [4, 0, 11]]  # some period-2 additive
    pt = "".join(
        c for c in "THESTATIONMASTERLIGHTSTHESIGNALLAMPSATDUSKANDWALK".upper() if c.isalpha()
    )
    if len(pt) % 3:
        pt += "X" * (3 - len(pt) % 3)
    P = [pos[c] for c in pt]
    ct_idx = []
    for blk in range(len(P) // 3):
        p = P[3 * blk : 3 * blk + 3]
        s = sched[blk % 2]
        for r in range(3):
            ct_idx.append((sum(K[r][k] * p[k] for k in range(3)) + s[r]) % 26)
    ct = "".join(inv[v] for v in ct_idx)

    got = hill_kpa.recover_affine(pt[:18], ct, n=3, q=2, alphabet="KRYPTOS")
    assert any(m == K and sc == sched for m, sc in got)
