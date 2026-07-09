"""Tests for the Hill matrix helpers added in Task 4:

* exact integer modular inverse for any n (2x2/3x3 fast paths + general adjugate),
  with an ``M . M^-1 == I (mod 26)`` self-check (no float ``round(det*inv())`` path);
* keyword->matrix builders for wide Hills: circulant, companion, kronecker, and the
  plain row-by-row ``matrix_from_word``.
"""

from __future__ import annotations

import pytest

from buttcrack.ciphers.hill import (
    Hill,
    _parse_key,
    circulant_matrix,
    companion_matrix,
    determinant_mod26,
    inverse_mod26,
    is_invertible_mod26,
    kronecker_matrix,
    matmul_mod26,
    matrix_from_word,
    matrix_to_key,
    verify_inverse,
)

STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _prep(msg: str, n: int) -> str:
    p = "".join(ch for ch in msg.upper() if "A" <= ch <= "Z")
    if len(p) % n:
        p += "X" * (n - len(p) % n)
    return p


# --- exact modular inverse -------------------------------------------------- #
def test_inverse_is_exact_2x2_3x3():
    """The published 2x2/3x3 keys invert exactly: M . M^-1 == I mod 26."""
    for key, n in (("3,3,2,5", 2), ("6,24,1,13,16,10,20,17,15", 3)):
        m, _ = _parse_key(key)
        inv = inverse_mod26(m)
        assert matmul_mod26(m, inv) == _identity(n)
        assert verify_inverse(m) is True


def test_inverse_general_4x4_and_5x5():
    """The general adjugate inverse is exact for wider matrices too."""
    kron = kronecker_matrix("DDCF", "DDCF", STD)  # 4x4
    assert len(kron) == 4 and len(kron[0]) == 4
    assert verify_inverse(kron) is True
    circ = circulant_matrix("DELTA", STD)  # 5x5
    assert len(circ) == 5
    assert matmul_mod26(circ, inverse_mod26(circ)) == _identity(5)


def test_singular_matrix_rejected():
    """A determinant sharing a factor with 26 is not invertible."""
    m = [[2, 4], [6, 8]]  # det = -8 = 18 mod 26, gcd(18,26)=2
    assert is_invertible_mod26(m) is False
    with pytest.raises(ValueError):
        inverse_mod26(m)


# --- keyword -> matrix builders --------------------------------------------- #
def test_matrix_from_word_matches_parser():
    """matrix_from_word reproduces the row-by-row matrix the Hill parser builds."""
    m, n = _parse_key("GYBNQKURP")
    assert n == 3
    assert matrix_from_word("GYBNQKURP", STD) == m


def test_circulant_structure_and_roundtrip():
    """circulant rows are cyclic shifts; the derived wide Hill round-trips."""
    m = circulant_matrix("DELTA", STD)
    v = [STD.index(c) for c in "DELTA"]
    # Row i, column j holds v[(j - i) mod n].
    assert m[0] == v
    assert m[1] == [v[(j - 1) % 5] for j in range(5)]
    assert is_invertible_mod26(m)
    key = matrix_to_key(m)
    msg = _prep("MEET ME BY THE OLD OAK TREE AT DAWN NORTH", 5)
    assert Hill().decode(Hill().encode(msg, key), key) == msg


def test_companion_determinant_and_roundtrip():
    """companion det = (-1)^n * a0; an invertible pick round-trips through the cipher."""
    word = "DELTA"  # a0 = D = 3 (coprime to 26) -> invertible
    m = companion_matrix(word, STD)
    n = len(word)
    a0 = STD.index("D")
    assert determinant_mod26(m) == (((-1) ** n) * a0) % 26
    assert is_invertible_mod26(m)
    key = matrix_to_key(m)
    msg = _prep("SEND THE MAPS AND LANTERNS TO THE HARBOUR", n)
    assert Hill().decode(Hill().encode(msg, key), key) == msg


def test_kronecker_from_two_small_keys_roundtrips():
    """A 4x4 Hill built as M(word_a) (x) M(word_b) round-trips end to end."""
    m = kronecker_matrix("DDCF", "HILL", STD)  # 2x2 (x) 2x2 -> 4x4
    assert len(m) == 4 and is_invertible_mod26(m)
    key = matrix_to_key(m)
    msg = _prep("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG NOW", 4)
    assert Hill().decode(Hill().encode(msg, key), key) == msg
