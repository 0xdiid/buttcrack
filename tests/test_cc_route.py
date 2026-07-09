"""Tests for the Route transposition cipher."""

from __future__ import annotations

import pytest

from buttcrack.ciphers.route import Route
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

# Published vector: CryptoCrack User Guide, Route Transposition page.
# Plaintext "WE ARE DISCOVERED", width 4, write-in by rows, read-out a spiral
# inwards counter-clockwise from the top-right corner -> "RAEWE CREDX ESIDO V".
VECTOR_PLAINTEXT = "WE ARE DISCOVERED"
VECTOR_KEY = "width 4; write rows; read spiral-ccw-tr"
VECTOR_CIPHERTEXT = "RAEWECREDXESIDOV"


def test_vector():
    cipher = Route()
    out = cipher.encode(VECTOR_PLAINTEXT, VECTOR_KEY)
    assert out == VECTOR_CIPHERTEXT


def test_vector_bare_key_forms():
    # bare integer width + bare route name applied to the read-out
    cipher = Route()
    assert cipher.encode(VECTOR_PLAINTEXT, "4; spiral-ccw-tr") == VECTOR_CIPHERTEXT


def test_roundtrip():
    cipher = Route()
    # width 7 divides the 28-letter message exactly, so no padding is needed and
    # decode recovers the letters-only plaintext exactly.
    msg = "Defend the east wall of the castle!"
    key = "width 7; write rows; read spiral-cw-tl"
    enc = cipher.encode(msg, key)
    dec = cipher.decode(enc, key)
    assert dec == only_letters(msg).upper()


def test_roundtrip_many_routes():
    cipher = Route()
    msg = "Defend the east wall of the castle"  # 28 letters
    expected = only_letters(msg).upper()
    routes = ("rows", "cols", "serp-rows-bl", "cols-tr", "spiral-cw-br", "spiral-ccw-bl")
    for width in (4, 7, 14):
        for read in routes:
            key = f"width {width}; write rows; read {read}"
            assert cipher.decode(cipher.encode(msg, key), key) == expected


@pytest.mark.slow
def test_crack_recovers_plaintext():
    cipher = Route()
    scorer = get_scorer()
    plain = (
        "the quick brown fox jumps over the lazy dog and then the dog runs "
        "across the field while the farmer watches from his porch enjoying "
        "the warm afternoon sun as birds sing in the tall green trees nearby"
    )
    expected = only_letters(plain).upper()
    # width 3 divides the 162-letter message exactly (no padding artefacts)
    key = "width 3; write rows; read spiral-ccw-tr"
    ciphertext = cipher.encode(plain, key)

    candidates = cipher.crack(ciphertext, scorer, top=5)
    assert candidates, "crack returned no candidates"
    assert candidates[0].plaintext == expected


# --- diagonal routes (added alongside the row/col/spiral families) ---

def test_diagonal_route_names_registered():
    """Both diagonal families over four corners, plain and serpentine, are present."""
    from buttcrack.ciphers.route import ROUTE_NAMES

    diag = [n for n in ROUTE_NAMES if "diag" in n]
    assert len(diag) == 16
    for corner in ("tl", "tr", "bl", "br"):
        for label in ("diag", "maindiag"):
            assert f"{label}-{corner}" in ROUTE_NAMES
            assert f"serp-{label}-{corner}" in ROUTE_NAMES


@pytest.mark.parametrize("dims", [(3, 3), (9, 17), (17, 9), (4, 7), (5, 6), (1, 8), (8, 1)])
def test_diagonal_routes_are_permutations(dims):
    """Every diagonal route visits each grid cell exactly once (invertibility)."""
    from buttcrack.ciphers.route import ROUTE_NAMES, _make_route

    rows, cols = dims
    full = sorted((r, c) for r in range(rows) for c in range(cols))
    for name in ROUTE_NAMES:
        if "diag" in name:
            assert sorted(_make_route(rows, cols, name)) == full, name


@pytest.mark.parametrize(
    "key",
    [
        "width 9; write rows; read serp-maindiag-bl",
        "width 6; write diag-tl; read maindiag-br",
        "width 9; write serp-diag-tr; read diag-bl",
    ],
)
def test_diagonal_route_roundtrip(key):
    """Write/read via diagonal routes round-trips on an evenly-divided grid."""
    cipher = Route()
    msg = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGSABCDEFGHIJKLMNOPQR"  # 54 letters
    assert only_letters(cipher.decode(cipher.encode(msg, key), key)) == only_letters(msg)


def test_diagonal_distinct_from_row_reading():
    """A diagonal read-out actually permutes (isn't accidentally the row order)."""
    cipher = Route()
    msg = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 52
    assert cipher.encode(msg, "width 4; read diag-tl") != only_letters(msg)
