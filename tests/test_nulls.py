"""Null-model harness: the right null must be invariant to the structure it preserves."""

from __future__ import annotations

import random

from buttcrack import nulls
from buttcrack.text import only_letters


def _coset_ioc(seq: str, p: int) -> float:
    """Mean over the p cosets of (IoC * 26) — elevated when a period-p key is present."""
    letters = only_letters(seq)
    total, used = 0.0, 0
    for r in range(p):
        col = letters[r::p]
        n = len(col)
        if n < 2:
            continue
        counts = [0] * 26
        for ch in col:
            counts[ord(ch) - 65] += 1
        total += sum(c * (c - 1) for c in counts) / (n * (n - 1)) * 26
        used += 1
    return total / used if used else 0.0


_PLAIN = (
    "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHEEARLYMORNINGSUNROSESLOWLYOVERTHE"
    "QUIETVILLAGEANDTHEPEOPLEWENTABOUTTHEIRWORKWITHASTEADYANDFAMILIARRHYTHMTHAT"
) * 2


def _vigenere(plain: str, key: str) -> str:
    ks = [ord(c) - 65 for c in key]
    return "".join(
        chr((ord(c) - 65 + ks[i % len(ks)]) % 26 + 65) for i, c in enumerate(only_letters(plain))
    )


PERIOD = 5
CT = _vigenere(_PLAIN, "LEMON")  # period-5 polyalphabetic


def test_within_coset_null_leaves_coset_ioc_invariant():
    # The honest null preserves each coset's multiset, so coset-IC at the period is
    # EXACTLY invariant -> it cannot manufacture a "signal". p must be ~1 (not significant).
    res = nulls.null_test(
        CT,
        lambda s: _coset_ioc(s, PERIOD),
        nulls.within_coset(PERIOD),
        trials=200,
        rng=random.Random(0),
    )
    assert res.null_sd == 0.0
    assert res.obs == res.null_mean  # invariant
    assert not res.significant
    assert res.p_value == 1.0


def test_permutation_null_detects_real_period():
    # Against a whole-message shuffle (which destroys the period), the elevated
    # coset-IC IS a real signal -> highly significant.
    res = nulls.null_test(
        CT,
        lambda s: _coset_ioc(s, PERIOD),
        nulls.permutation,
        trials=300,
        rng=random.Random(1),
    )
    assert res.obs > res.null_mean
    assert res.z > 3
    assert res.significant


def test_block_shuffle_preserves_mod_block_cosets():
    rng = random.Random(2)
    null = nulls.block_shuffle(PERIOD)
    before = _coset_ioc(CT, PERIOD)
    after = _coset_ioc("".join(null(CT, rng)), PERIOD)
    assert after == before  # mod-block coset multisets are invariant under block shuffle


def test_permutation_preserves_multiset():
    rng = random.Random(3)
    out = "".join(nulls.permutation(CT, rng))
    assert sorted(out) == sorted(CT)
    assert out != CT


def test_scan_test_finds_true_period():
    periods = list(range(2, 13))
    res = nulls.scan_test(
        CT,
        lambda s: [_coset_ioc(s, p) for p in periods],
        nulls.permutation,
        periods,
        trials=300,
        rng=random.Random(4),
        family=True,
    )
    assert res.argmax in (PERIOD, 2 * PERIOD)  # period or its harmonic
    assert res.scan_max_p < 0.05  # look-elsewhere-corrected, still significant
    assert res.family_max_p is not None and res.family_max_p < 0.05


def test_scan_test_random_text_not_significant():
    rng = random.Random(5)
    rand = "".join(chr(65 + rng.randrange(26)) for _ in range(len(CT)))
    periods = list(range(2, 13))
    res = nulls.scan_test(
        rand,
        lambda s: [_coset_ioc(s, p) for p in periods],
        nulls.permutation,
        periods,
        trials=300,
        rng=random.Random(6),
    )
    assert res.scan_max_p > 0.05  # no real period -> not significant after correction


def test_honest_max_over_search_null():
    # A statistic that is itself a max over a search: the null maxes the same way,
    # so a lucky single high tooth doesn't read as significant on its own.
    periods = list(range(2, 13))
    stat = lambda s: max(_coset_ioc(s, p) for p in periods)  # noqa: E731
    res = nulls.null_test(CT, stat, nulls.permutation, trials=200, rng=random.Random(7))
    assert res.obs > res.null_mean  # the real period survives the max-vs-max comparison


def test_add_one_smoothing_never_zero():
    res = nulls.null_test(
        CT,
        lambda s: _coset_ioc(s, PERIOD),
        nulls.permutation,
        trials=50,
        rng=random.Random(8),
    )
    assert res.p_value >= 1 / (res.trials + 1)
