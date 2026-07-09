"""Tests for the crib module: additive crib-drag plus the product / keyed / autokey solvers.

Each new solver is validated end-to-end: encode a synthetic with the EXACT structure
the solver inverts (plus a known crib), then confirm the solver recovers the original
readable English plaintext from the crib alone.
"""

from __future__ import annotations

from math import gcd

from buttcrack.ciphers.quagmire3 import keyed_alphabet
from buttcrack.crib import (
    autokey_crib_unzip,
    crib_drag,
    keyed_alphabet_crib_drag,
    product_crib_solve,
    product_crib_sweep,
)

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# A chunk of readable English with no spaces (segments into long dictionary words).
_PT = (
    "THEOLDLIGHTHOUSEKEEPERCLIMBSTHENARROWSTAIRSEACHEVENINGTOTRIMTHELAMPANDPOLISH"
    "THEGREATGLASSLENSHEHASCOUNTEDTHESTEPSFORTHIRTYYEARSANDKNOWSEVERYWORNSTONEBYH"
    "EARTFROMTHEGALLER"
)


def _idx(letters: str, alpha: str) -> list[int]:
    pos = {c: i for i, c in enumerate(alpha)}
    return [pos[c] for c in letters]


# --------------------------------------------------------------------------- #
# product_crib_solve / product_crib_sweep
# --------------------------------------------------------------------------- #


def _product_encode(pt: str, a: list[int], b: list[int], alpha: str) -> str:
    """c_idx[i] = pt_idx[i] + a[i%p] + b[i%q] in keyed-alphabet index space."""
    p, q = len(a), len(b)
    pidx = _idx(pt, alpha)
    return "".join(alpha[(pidx[i] + a[i % p] + b[i % q]) % 26] for i in range(len(pt)))


def test_product_crib_solve_recovers_full_plaintext():
    alpha = keyed_alphabet("KRYPTOS")
    p, q = 3, 4  # coprime
    assert gcd(p, q) == 1
    a = [5, 17, 2]
    b = [9, 0, 14, 22]
    ct = _product_encode(_PT, a, b, alpha)

    # crib of length >= p+q-1 == 6 placed at its true position connects the graph.
    pos = 10
    crib = _PT[pos : pos + 7]
    res = product_crib_solve(ct, crib, pos, p, q, alphabet="KRYPTOS")

    assert res["contradiction"] is False
    assert res["determined"] == len(_PT)  # whole graph connected -> everything decoded
    assert res["plaintext"] == _PT
    assert res["word_coverage"] > 0.45


def test_product_crib_solve_partial_when_crib_too_short():
    """A crib shorter than p+q-1 only partially connects the graph (no contradiction)."""
    alpha = keyed_alphabet("KRYPTOS")
    p, q = 3, 5
    a = [1, 2, 3]
    b = [4, 5, 6, 7, 8]
    ct = _product_encode(_PT, a, b, alpha)
    res = product_crib_solve(ct, _PT[0:3], 0, p, q, alphabet="KRYPTOS")
    assert res["contradiction"] is False
    assert 0 < res["determined"] <= len(_PT)


def test_product_crib_solve_wrong_position_or_crib_contradicts_or_garbles():
    """A wrong placement either contradicts or fails the readability gate."""
    alpha = keyed_alphabet("KRYPTOS")
    p, q = 3, 4
    a = [5, 17, 2]
    b = [9, 0, 14, 22]
    ct = _product_encode(_PT, a, b, alpha)
    # correct crib text but wrong position
    crib = _PT[10:17]
    res = product_crib_solve(ct, crib, 0, p, q, alphabet="KRYPTOS")
    # Either a hard contradiction, or it "solves" to non-English (low coverage).
    assert res["contradiction"] or res["word_coverage"] < 0.4 or res["plaintext"] != _PT


def test_product_crib_sweep_finds_it_without_known_position():
    alpha = keyed_alphabet("KRYPTOS")
    p, q = 3, 4
    a = [5, 17, 2]
    b = [9, 0, 14, 22]
    ct = _product_encode(_PT, a, b, alpha)
    crib = _PT[10:17]
    hits = product_crib_sweep(
        ct, [crib], p_range=range(2, 6), q_range=range(2, 6), alphabet="KRYPTOS"
    )
    assert hits, "sweep should find at least one readable propagation"
    assert hits[0]["plaintext"] == _PT
    assert hits[0]["p"] == p and hits[0]["q"] == q


# --------------------------------------------------------------------------- #
# keyed_alphabet_crib_drag
# --------------------------------------------------------------------------- #


def _keyed_vig_encode(pt: str, key: str, alpha: str) -> str:
    """Quagmire/Vigenere over a keyed alphabet: c = p + k (index space)."""
    pos = {c: i for i, c in enumerate(alpha)}
    return "".join(alpha[(pos[pt[i]] + pos[key[i % len(key)]]) % 26] for i in range(len(pt)))


def test_keyed_alphabet_crib_drag_surfaces_keyword():
    alpha = keyed_alphabet("KRYPTOS")
    keyword = "MEADOW"
    ct = _keyed_vig_encode(_PT, keyword, alpha)
    # Place the crib at a position that is a multiple of the period so the implied
    # key fragment equals the repeating keyword exactly.
    pos = 12  # 12 % 6 == 0
    crib = _PT[pos : pos + len(keyword)]
    out = keyed_alphabet_crib_drag(ct, crib, alphabet="KRYPTOS")
    assert "vigenere" in out
    placements = out["vigenere"]
    # The true placement should report the keyword as the implied key fragment.
    frags = {pl["position"]: pl["key_fragment"] for pl in placements}
    assert pos in frags
    assert frags[pos] == keyword


def test_keyed_alphabet_crib_drag_beaufort_convention_present():
    alpha = keyed_alphabet("KRYPTOS")
    ct = _keyed_vig_encode(_PT, "MEADOW", alpha)
    out = keyed_alphabet_crib_drag(
        ct, _PT[12:18], alphabet="KRYPTOS", conventions=("vigenere", "beaufort", "variant")
    )
    assert set(out) == {"vigenere", "beaufort", "variant"}
    for places in out.values():
        assert places  # each convention yields ranked placements


# --------------------------------------------------------------------------- #
# autokey_crib_unzip
# --------------------------------------------------------------------------- #


def _autokey_vig_encode(pt: str, primer: str, alpha: str) -> str:
    """Plaintext-autokey vigenere: c = pt + key, key[i] = primer[i] (i<L) else pt[i-L]."""
    pos = {c: i for i, c in enumerate(alpha)}
    n = len(pt)
    L = len(primer)
    out = []
    for i in range(n):
        kc = primer[i] if i < L else pt[i - L]
        out.append(alpha[(pos[pt[i]] + pos[kc]) % 26])
    return "".join(out)


def test_autokey_crib_unzip_recovers_plaintext_kryptos():
    alpha = keyed_alphabet("KRYPTOS")
    primer = "FORGING"  # L = 7
    ct = _autokey_vig_encode(_PT, primer, alpha)
    # crib spanning >= L consecutive positions seeds all L chains.
    pos = 20
    crib = _PT[pos : pos + 10]
    hits = autokey_crib_unzip(
        ct, [crib], alphabets=("KRYPTOS",), conventions=("vigenere",), max_primer=12
    )
    assert hits, "autokey unzip should recover at least one readable plaintext"
    best = hits[0]
    # the unzipped plaintext matches the original from position L onward (primer unknown).
    assert best["plaintext"][len(primer) :] == _PT[len(primer) :]
    assert best["primer_len"] == len(primer)
    assert best["word_coverage"] > 0.45


def test_autokey_crib_unzip_std_alphabet():
    alpha = _STD
    primer = "SPRING"  # L = 6
    ct = _autokey_vig_encode(_PT, primer, alpha)
    pos = 15
    crib = _PT[pos : pos + 9]
    hits = autokey_crib_unzip(
        ct, [crib], alphabets=("STD",), conventions=("vigenere",), max_primer=10
    )
    assert hits
    assert hits[0]["plaintext"][len(primer) :] == _PT[len(primer) :]
    assert hits[0]["alphabet"] == "STD"


def test_autokey_crib_unzip_rejects_unrelated_crib():
    alpha = keyed_alphabet("KRYPTOS")
    ct = _autokey_vig_encode(_PT, "FORGING", alpha)
    # A crib that does not actually appear: unzips to garbage -> filtered by coverage.
    hits = autokey_crib_unzip(
        ct, ["ZZZZZZZZZZ"], alphabets=("KRYPTOS",), conventions=("vigenere",), max_primer=12
    )
    assert all(h["plaintext"][7:] != _PT[7:] for h in hits)


# --------------------------------------------------------------------------- #
# regression: the original additive crib_drag still works
# --------------------------------------------------------------------------- #


def test_crib_drag_still_ranks_running_key():
    # plain Vigenere with a running English key: implied key at the true offset is English.
    key = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGTHEQUICKBROWN"
    pt = _PT[: len(key)]
    pos = {c: i for i, c in enumerate(_STD)}
    ct = "".join(_STD[(pos[pt[i]] + pos[key[i]]) % 26] for i in range(len(pt)))
    out = crib_drag(ct, pt[:8])
    assert "vigenere" in out
    # best placement is offset 0, implied key fragment is the running key prefix.
    assert out["vigenere"][0]["position"] == 0
    assert out["vigenere"][0]["key_fragment"] == key[:8]
