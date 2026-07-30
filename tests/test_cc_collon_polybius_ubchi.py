"""Polybius, Collon and Ubchi — the three classical types the registry was missing.

Every encode is checked against a published vector before any crack is trusted.
"""

import random

import pytest

from buttcrack import registry
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PT = (
    "the analysis routines need a healthy stretch of perfectly ordinary english prose so "
    "that the frequency statistics and quadgram fitness scores can lock onto the underlying "
    "message and recover the original text without any prior knowledge of the secret key "
    "that was chosen to encipher it in the beginning of this particular exercise today"
)
TARGET = only_letters(PT).upper().replace("J", "I")


def _solved(cipher_name, key, *, timeout=180, prefix=60):
    cipher = registry.get(cipher_name)
    ct = cipher.encode(PT, key)
    cands = cipher.crack(ct, get_scorer(), top=3, rng=random.Random(7), timeout=timeout)
    assert cands, f"{cipher_name} returned no candidates"
    return cands[0]


# -- Polybius ------------------------------------------------------------------


def test_polybius_published_vector():
    """Wikipedia/dCode: on the ZEBRAS square, D is at row 2 column 3."""
    assert registry.get("polybius").encode("D", "ZEBRAS") == "23"


def test_polybius_standard_square():
    assert registry.get("polybius").encode("A", "") == "11"
    assert registry.get("polybius").encode("Z", "") == "55"


@pytest.mark.parametrize("key", ["KRYPTOS", "", "ZEBRAS"])
def test_polybius_roundtrip(key):
    p = registry.get("polybius")
    assert p.decode(p.encode(PT, key), key) == TARGET


def test_polybius_merges_j_into_i():
    p = registry.get("polybius")
    assert p.encode("JAM", "") == p.encode("IAM", "")


def test_polybius_decode_ignores_separators():
    p = registry.get("polybius")
    assert p.decode("11 12 13", "") == p.decode("111213", "")


def test_polybius_crack_recovers_plaintext_and_square():
    best = _solved("polybius", "KRYPTOS")
    assert only_letters(best.plaintext).upper()[:60] == TARGET[:60]
    # Cells whose coordinate pair never occurred are honestly reported as unknown.
    assert best.meta["cells_recovered"] >= 20
    assert best.key.startswith("KRYPTOS")


def test_polybius_crack_declines_short_input():
    assert registry.get("polybius").crack("11 12 13", get_scorer()) == []


# -- Collon --------------------------------------------------------------------


def test_collon_published_vector_dc():
    """dCode: standard square, N=2, 'DC' -> 'AAYX' (rows AA, then columns YX)."""
    assert registry.get("collon").encode("DC", "/2") == "AAYX"


def test_collon_published_vector_decode():
    """dCode: N=3, 'AKKXZV' splits to rows AKK / columns XZV -> 'CKF'."""
    assert registry.get("collon").decode("AKKXZV", "/3") == "CKF"


@pytest.mark.parametrize("key", ["KRYPTOS/5", "/3", "ZEBRAS/7", "PALIMPSEST/2", "KRYPTOS/1"])
def test_collon_roundtrip(key):
    c = registry.get("collon")
    assert c.decode(c.encode(PT, key), key) == TARGET


def test_collon_ciphertext_uses_at_most_ten_letters():
    """The structural fact the cracker exploits: five row labels, five column labels."""
    ct = registry.get("collon").encode(PT, "KRYPTOS/5")
    assert len(set(ct)) <= 10


def test_collon_doubles_the_length():
    assert len(registry.get("collon").encode(PT, "KRYPTOS/5")) == 2 * len(TARGET)


@pytest.mark.parametrize("key", ["KRYPTOS/5", "/3", "ZEBRAS/7", "PALIMPSEST/2"])
def test_collon_crack_recovers_plaintext(key):
    best = _solved("collon", key)
    assert only_letters(best.plaintext).upper()[:60] == TARGET[:60]


def test_collon_recovered_key_round_trips():
    """The square is only recovered up to a row/column permutation — which is free,
    because a label travels with its row, so every consistent square decodes alike."""
    c = registry.get("collon")
    ct = c.encode(PT, "KRYPTOS/5")
    best = _solved("collon", "KRYPTOS/5")
    assert only_letters(c.decode(ct, best.key)).upper()[:60] == TARGET[:60]


def test_collon_rejects_key_without_group_size():
    with pytest.raises(ValueError):
        registry.get("collon").encode(PT, "KRYPTOS")


def test_collon_crack_declines_short_input():
    assert registry.get("collon").crack("AAYX", get_scorer()) == []


# -- Ubchi ---------------------------------------------------------------------


def test_ubchi_published_vector():
    """dCode: 'SECRET' under key 'UBER' with one null -> 'TECXRES'."""
    assert registry.get("ubchi").encode("SECRET", "UBER/1") == "TECXRES"


def test_ubchi_published_vector_decode():
    assert registry.get("ubchi").decode("TECXRES", "UBER/1") == "SECRET"


@pytest.mark.parametrize("key", ["UBER/2", "ZEBRAS", "UBER/0", "1,2,3,0/1"])
def test_ubchi_roundtrip(key):
    u = registry.get("ubchi")
    assert u.decode(u.encode(PT, key), key) == TARGET


def test_ubchi_defaults_to_zero_nulls():
    u = registry.get("ubchi")
    assert u.encode(PT, "UBER") == u.encode(PT, "UBER/0")


def test_ubchi_is_the_columnar_applied_twice():
    """The definition, asserted directly: same key, both passes."""
    columnar = registry.get("columnar")
    once = columnar.encode(TARGET, "UBER")
    assert registry.get("ubchi").encode(PT, "UBER/0") == columnar.encode(once, "UBER")


def test_ubchi_differs_from_a_single_columnar():
    assert registry.get("ubchi").encode(PT, "UBER/0") != registry.get("columnar").encode(PT, "UBER")


def test_ubchi_crack_recovers_order_and_null_count():
    best = _solved("ubchi", "UBER/2")
    assert only_letters(best.plaintext).upper()[:60] == TARGET[:60]
    # UBER sorts to B,E,R,U -> read order 1,2,3,0.
    assert best.meta["order"] == [1, 2, 3, 0]
    assert best.meta["nulls"] == 2


def test_ubchi_crack_declines_short_input():
    assert registry.get("ubchi").crack("ABC", get_scorer()) == []
