"""Tests for coset-preserving windings and triangular read orders."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from buttcrack.windings import (
    coset_preserving_perm,
    coset_preserving_shuffle,
    faro_perm,
    fold_perm,
    is_coset_preserving,
    triangle_orders,
)


def _apply(seq: list[int], order: list[int]) -> list[int]:
    """Read ``seq`` through ``order``: ``out[i] = seq[order[i]]``."""
    return [seq[i] for i in order]


def _invert(order: list[int]) -> list[int]:
    inv = [0] * len(order)
    for i, o in enumerate(order):
        inv[o] = i
    return inv


# --- triangular read orders --------------------------------------------------


def test_triangle_orders_full_permutations():
    orders = triangle_orders(153)  # T(17)
    assert {"by-row", "by-column", "by-diagonal"} <= set(orders)
    for name, order in orders.items():
        assert sorted(order) == list(range(153)), f"{name} is not a permutation of range(153)"


def test_triangle_orders_roundtrip_identity():
    n = 153
    src = list(range(n))
    for name, order in triangle_orders(n).items():
        applied = _apply(src, order)
        restored = _apply(applied, _invert(order))
        assert restored == src, f"apply-then-invert not identity for {name}"


def test_triangle_orders_rejects_non_triangular():
    with pytest.raises(ValueError):
        triangle_orders(153 - 1)  # 152 is not triangular


# --- coset-preserving permutations -------------------------------------------


def test_coset_preserving_perm_fixes_residue():
    rng = random.Random(20240709)
    for _ in range(8):
        perm = coset_preserving_perm(153, 7, rng=rng)
        assert sorted(perm) == list(range(153))  # a genuine permutation
        assert is_coset_preserving(perm, 7)
        assert all(perm[i] % 7 == i % 7 for i in range(153))


def test_fold_and_faro_preserve_cosets():
    for builder in (fold_perm, faro_perm):
        perm = builder(153, 7)
        assert sorted(perm) == list(range(153))
        assert is_coset_preserving(perm, 7)
    # they are non-trivial windings, not the identity
    assert fold_perm(153, 7) != list(range(153))
    assert faro_perm(153, 7) != list(range(153))


def test_is_coset_preserving_rejects_cross_coset_swap():
    perm = list(range(153))
    perm[0], perm[1] = perm[1], perm[0]  # swaps residues 0 and 1 mod 7
    assert not is_coset_preserving(perm, 7)


# --- the honest within-coset null --------------------------------------------


def test_coset_preserving_shuffle_preserves_coset_multisets():
    mod, n = 7, 154  # 22 letters per coset
    gen = np.random.default_rng(0)
    # Build a string whose each mod-7 coset has a peaked letter distribution:
    # coset r is ~80% its own peak letter, the rest uniform noise.
    peak = "ABCDEFG"
    chars: list[str] = [""] * n
    for r in range(mod):
        for i in range(r, n, mod):
            if gen.random() < 0.8:
                chars[i] = peak[r]
            else:
                chars[i] = chr(ord("A") + int(gen.integers(0, 26)))
    text = "".join(chars)

    rng = random.Random(99)
    out = coset_preserving_shuffle(text, mod, rng=rng)
    assert isinstance(out, list)

    # Each coset's exact letter multiset is preserved.
    for r in range(mod):
        before = Counter(text[i] for i in range(r, n, mod))
        after = Counter(out[i] for i in range(r, n, mod))
        assert before == after, f"coset {r} multiset changed"

    # ...while the within-coset order actually changed for some coset.
    changed = any(
        [out[i] for i in range(r, n, mod)] != [text[i] for i in range(r, n, mod)]
        for r in range(mod)
    )
    assert changed
    # The overall multiset is preserved too (union of the cosets).
    assert Counter(out) == Counter(text)
