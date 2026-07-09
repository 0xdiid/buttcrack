"""Coset-preserving *windings* and triangular read orders.

A *winding* is a transposition that rewinds a message's letters along a
geometric thread (a coil, a fold, a shuffle) rather than the keyed columns of a
classical columnar. The functions here build such reorderings as explicit
position-permutations of ``range(n)`` (a *read order*: ``out[i] = src[order[i]]``).

The pivotal notion is a **coset-preserving** permutation: one that moves every
index only *within* its residue class mod ``mod``. A "coset" is the subsequence
of positions ``r, r+mod, r+2*mod, ...`` for a fixed residue ``r``. Because such
a permutation merely shuffles each coset internally, every coset keeps its exact
letter multiset, so any per-coset statistic is invariant under it -- in
particular the coset index of coincidence (coset-IC), the fingerprint left by a
period-``mod`` polyalphabetic key.

That invariance is what fixes the *correct null hypothesis*. When a ciphertext
shows an elevated coset-IC at period ``mod``, the honest test of "does the
within-coset ORDER carry a message?" must permute letters only within each coset
(:func:`coset_preserving_shuffle`), preserving that fingerprint. A plain
whole-message letter shuffle is the *wrong* null: it destroys the coset-IC and
thereby manufactures false positives, because any structured arrangement will
trivially beat a null that has thrown the signal away.

Contents:

* :func:`triangle_orders`      -- read orders over a triangular grid ``T_k`` (``n = k(k+1)/2``).
* :func:`coset_preserving_perm` -- a random coset-preserving permutation (affine within each coset).
* :func:`fold_perm`, :func:`faro_perm` -- deterministic coset-preserving fold(bight)/faro windings.
* :func:`coset_preserving_shuffle` -- the honest within-coset null.
* :func:`is_coset_preserving`  -- the residue-fixing invariant check.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


# --- triangular read orders --------------------------------------------------


def _triangle_side(n: int) -> int:
    """Return ``k`` for a triangular number ``n = k(k+1)/2``; raise otherwise.

    ``n`` is triangular iff ``8*n + 1`` is a perfect square; ``k`` is then the
    number of rows of the triangle (row ``r`` holding ``r + 1`` cells).
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    root = math.isqrt(8 * n + 1)
    if root * root != 8 * n + 1:
        raise ValueError(f"n={n} is not a triangular number k(k+1)/2")
    return (root - 1) // 2


def triangle_orders(n: int) -> dict[str, list[int]]:
    """Named read orders over the triangular grid ``T_k`` with ``n = k(k+1)/2`` cells.

    The letters are laid left-aligned into a triangle whose row ``r`` holds
    ``r + 1`` cells; ``by-row`` is the natural row-major order and the flat index
    of cell ``(r, j)`` is ``r*(r+1)//2 + j``. Every value is a full permutation
    of ``range(n)`` usable as a read order (``out[i] = src[order[i]]``):

    * ``by-row``          -- rows top-to-bottom, each left-to-right (the identity).
    * ``by-column``       -- columns left-to-right, each top-to-bottom.
    * ``by-diagonal``     -- diagonals parallel to the hypotenuse (constant ``r - j``).
    * ``by-row-reversed`` -- rows bottom-to-top (a coil read from the wide end).
    * ``boustrophedon``   -- rows top-to-bottom, alternating scan direction.

    Raise ``ValueError`` if ``n`` is not a triangular number.
    """
    k = _triangle_side(n)
    # rowspans[r] holds the flat indices of the cells of row r, left-to-right.
    rowspans: list[list[int]] = []
    idx = 0
    for r in range(k):
        rowspans.append(list(range(idx, idx + r + 1)))
        idx += r + 1

    by_row = [c for span in rowspans for c in span]
    by_column = [rowspans[r][j] for j in range(k) for r in range(j, k)]
    by_diagonal = [rowspans[r][r - d] for d in range(k) for r in range(d, k)]
    by_row_reversed = [c for span in reversed(rowspans) for c in span]
    boustrophedon = [
        c for i, span in enumerate(rowspans) for c in (span if i % 2 == 0 else span[::-1])
    ]
    return {
        "by-row": by_row,
        "by-column": by_column,
        "by-diagonal": by_diagonal,
        "by-row-reversed": by_row_reversed,
        "boustrophedon": boustrophedon,
    }


# --- coset-preserving permutations -------------------------------------------


def _validate(n: int, mod: int) -> None:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if mod < 1:
        raise ValueError(f"mod must be >= 1, got {mod}")


def _units_mod(length: int) -> list[int]:
    """Multipliers ``a`` (``1 <= a < length``) coprime to ``length`` (``[1]`` for length 1)."""
    if length <= 1:
        return [1]
    return [a for a in range(1, length) if math.gcd(a, length) == 1]


def coset_preserving_perm(n: int, mod: int, *, rng: random.Random) -> list[int]:
    """A random permutation of ``range(n)`` that fixes every index's residue mod ``mod``.

    Within each residue class -- the coset ``r, r+mod, ...`` of length ``L`` -- the
    positions are permuted by a random affine map ``t -> (a*t + b) mod L`` with
    ``a`` drawn from the units mod ``L`` (so the map is a bijection) and ``b``
    uniform in ``range(L)``. Different cosets get independent ``(a, b)``; cosets of
    different length (e.g. 22 and 21 when ``n = 153``, ``mod = 7``) each get an ``a``
    valid for their own length. The result satisfies :func:`is_coset_preserving`.
    """
    _validate(n, mod)
    perm = list(range(n))
    for r in range(mod):
        seq = list(range(r, n, mod))
        length = len(seq)
        if length == 0:
            continue
        a = rng.choice(_units_mod(length))
        b = rng.randrange(length)
        for t, src in enumerate(seq):
            perm[src] = seq[(a * t + b) % length]
    return perm


def _fold_order(length: int) -> list[int]:
    """Fold(bight) order of ``range(length)``: ``0, L-1, 1, L-2, ...`` (fold in half, read across).

    The thread is doubled back on itself -- the first move of every knot -- and the
    two strands are read alternately from the fold's open end.
    """
    order: list[int] = []
    for t in range(length // 2):
        order += [t, length - 1 - t]
    if length % 2:
        order.append(length // 2)
    return order


def _faro_order(length: int) -> list[int]:
    """Out-faro (out-shuffle) order of ``range(length)``: interleave the front and back halves.

    ``0, h, 1, h+1, ...`` with ``h = ceil(length/2)``; the front half's first
    element stays first, as in a magician's out-shuffle.
    """
    half = (length + 1) // 2
    front = list(range(half))
    back = list(range(half, length))
    order: list[int] = []
    for i in range(half):
        order.append(front[i])
        if i < len(back):
            order.append(back[i])
    return order


def _within_coset_perm(n: int, mod: int, order_fn) -> list[int]:
    """Build a coset-preserving permutation by applying ``order_fn`` inside each coset."""
    _validate(n, mod)
    perm = list(range(n))
    for r in range(mod):
        seq = list(range(r, n, mod))
        order = order_fn(len(seq))
        for t, src in enumerate(seq):
            perm[src] = seq[order[t]]
    return perm


def fold_perm(n: int, mod: int) -> list[int]:
    """Deterministic coset-preserving *fold(bight)* winding of ``range(n)`` at period ``mod``.

    Each coset is read in :func:`_fold_order` (folded in half and read across). The
    result fixes every index's residue mod ``mod`` (see :func:`is_coset_preserving`).
    """
    return _within_coset_perm(n, mod, _fold_order)


def faro_perm(n: int, mod: int) -> list[int]:
    """Deterministic coset-preserving *faro* winding of ``range(n)`` at period ``mod``.

    Each coset is read in :func:`_faro_order` (front and back halves interleaved,
    an out-shuffle). The result fixes every index's residue mod ``mod``.
    """
    return _within_coset_perm(n, mod, _faro_order)


# --- the honest null and the invariant check ---------------------------------


def coset_preserving_shuffle(seq: Sequence[T], mod: int, *, rng: random.Random) -> list[T]:
    """Randomly permute ``seq`` *within* each residue class mod ``mod`` (the honest null).

    Each coset ``r, r+mod, ...`` is shuffled independently, so every coset's exact
    element multiset is preserved while its internal order is randomized. This is
    the correct null for a coset-IC-elevated ciphertext: it keeps the per-coset
    fingerprint that a plain whole-message shuffle would destroy. Accepts a ``str``
    or a ``list`` (any sequence) and returns a new ``list``.
    """
    _validate(len(seq), mod)
    out = list(seq)
    for r in range(mod):
        idxs = list(range(r, len(out), mod))
        letters = [out[i] for i in idxs]
        rng.shuffle(letters)
        for i, ch in zip(idxs, letters, strict=True):
            out[i] = ch
    return out


def is_coset_preserving(perm: list[int], mod: int) -> bool:
    """True iff ``perm[i] % mod == i % mod`` for every ``i`` (each index keeps its coset)."""
    if mod < 1:
        raise ValueError(f"mod must be >= 1, got {mod}")
    return all(value % mod == i % mod for i, value in enumerate(perm))
