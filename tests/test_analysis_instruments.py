"""Plant-gated tests for harmonic corroboration, lag-difference scan and the
separable-pad annihilator.

Every assertion here is a known-answer case: the instrument must fire on a plant
of the exact construction it claims to detect and stay quiet on a matched control.
"""

from __future__ import annotations

import random

import pytest

from buttcrack.analysis import (
    harmonic_corroboration,
    lag_difference_scan,
    separable_pad_annihilator,
)
from buttcrack.validate import _FILLER

ENGLISH = (_FILLER * 8)[:1200]


def _rand_text(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))


def _vigenere(pt: str, key: list[int]) -> str:
    return "".join(chr((ord(c) - 65 + key[i % len(key)]) % 26 + 65) for i, c in enumerate(pt))


# ------------------------------------------------------ harmonic corroboration


def test_harmonic_corroborates_a_true_period_7():
    rng = random.Random(1)
    key = [rng.randrange(26) for _ in range(7)]
    ct = _vigenere(ENGLISH[:350], key)
    res = harmonic_corroboration(ct, 7, samples=200)
    assert res["corroborated"], res
    assert res["z_base"] > 3 and res["z_double"] > 3


def test_harmonic_rejects_a_period_claim_on_random_text():
    rng = random.Random(2)
    res = harmonic_corroboration(_rand_text(350, rng), 7, samples=200)
    assert not res["corroborated"], res


def test_harmonic_rejects_the_wrong_period_on_a_true_cipher():
    """Period 5 claimed on a true period-7 cipher: 10 is not forced, pair fails."""
    rng = random.Random(3)
    key = [rng.randrange(26) for _ in range(7)]
    ct = _vigenere(ENGLISH[:350], key)
    res = harmonic_corroboration(ct, 5, samples=200)
    assert not res["corroborated"], res


def test_harmonic_validates_range():
    with pytest.raises(ValueError):
        harmonic_corroboration(ENGLISH[:100], 30)


# ------------------------------------------------------ lag-difference scan


def test_lag_difference_finds_a_cycled_key():
    rng = random.Random(4)
    key = [rng.randrange(26) for _ in range(11)]
    ct = _vigenere(ENGLISH[:600], key)
    res = lag_difference_scan(ct, max_lag=30)
    top = res["lags"]
    # the key period (or a multiple) must lead the scan with a strong z
    assert top[0]["lag"] in (11, 22), top[:3]
    assert top[0]["z"] > 4
    assert res["scan_p"] < 0.05


def test_lag_difference_reads_a_ciphertext_autokey_tail():
    lag = 9
    pt = ENGLISH[:400]
    ct: list[int] = []
    for i, c in enumerate(pt):
        prev = ct[i - lag] if i >= lag else 7  # constant primer
        ct.append((ord(c) - 65 + prev) % 26)
    text = "".join(chr(x + 65) for x in ct)
    res = lag_difference_scan(text, max_lag=30)
    top = res["lags"]
    assert top[0]["lag"] == lag
    # the difference stream IS the plaintext past the primer
    assert top[0]["head"] == pt[lag : lag + 24]
    assert res["scan_p"] < 0.05


def test_lag_difference_quiet_on_random_text():
    # median scan_p over a few random texts must be unremarkable (a single fixed
    # text can legitimately land in the tail — that is what a p-value means)
    ps = []
    for seed in (100, 101, 102, 103, 104):
        rng = random.Random(seed)
        res = lag_difference_scan(_rand_text(600, rng), max_lag=30, samples=100)
        ps.append(res["scan_p"])
    assert sorted(ps)[len(ps) // 2] > 0.05, ps


# ------------------------------------------------------ separable-pad annihilator


def _separable_ct(n: int, p: int, q: int, rng: random.Random) -> str:
    a = [rng.randrange(26) for _ in range(p)]
    b = [rng.randrange(26) for _ in range(q)]
    return "".join(
        chr((ord(c) - 65 + a[i % p] + b[i % q]) % 26 + 65) for i, c in enumerate(ENGLISH[:n])
    )


def test_annihilator_fires_on_a_true_separable_pad():
    rng = random.Random(6)
    ct = _separable_ct(1200, 7, 4, rng)
    res = separable_pad_annihilator(ct, 7, 4, samples=200)
    assert res["z"] > 4, res


def test_annihilator_quiet_on_a_non_separable_long_key():
    rng = random.Random(7)
    key = [rng.randrange(26) for _ in range(28)]
    ct = _vigenere(ENGLISH[:1200], key)
    res = separable_pad_annihilator(ct, 7, 4, samples=200)
    assert res["z"] < 3, res


def test_annihilator_quiet_on_random_text():
    rng = random.Random(8)
    res = separable_pad_annihilator(_rand_text(1200, rng), 7, 4, samples=200)
    assert res["z"] < 3, res


def test_annihilator_validates_args():
    with pytest.raises(ValueError):
        separable_pad_annihilator(ENGLISH[:1200], 7, 7)
    with pytest.raises(ValueError):
        separable_pad_annihilator(ENGLISH[:45], 7, 4)
