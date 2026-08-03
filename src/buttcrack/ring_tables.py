"""N-gram tables indexed by a CIPHER RING's positions, so the commonest silent bug is impossible.

THE BUG THIS EXISTS TO PREVENT
------------------------------
Classical work on a keyed alphabet keeps text as positions in a *ring* — for the Kryptos family,
``KRYPTOSABCDEFGHIJLMNQUVWXZ`` — while every n-gram table is keyed by **letter identity** (A=0 …
Z=25). Indexing an A–Z table with ring positions is a type error that Python cannot see: both are
``int`` in ``0..25``, so it runs, returns plausible numbers, and scores a *permuted* text.

It is the single most repeatable mistake in this codebase. In one session it appeared three times
in three unrelated places:

* a Held-Karp columnar solver, where it ranked the TRUE column order **below uniform**
  (−3.36/bigram against −2.83 for uniform — impossible for real English, which is what exposed it);
* a batched superposition solver;
* a multiset-fit channel comparing ring-indexed counts to A–Z letter profiles.

Each was caught only by a gate failing in a way that made no sense. A comment does not prevent it;
the two index spaces have to stop being interchangeable at the API.

THE FIX
-------
Fold the ring into the table. :func:`ring_ngram_table` returns a table whose axes are **ring
positions**, so indexing it with ring positions is correct *by construction* and there is no
A–Z-indexed object left lying around to reach for by mistake::

    tab = ring_ngram_table("bigrams", KRYPTOS)      # 26x26, axes are RING positions
    score = tab[r_i][r_j]                            # r_* are ring positions — correct

The identity ring ``ABCDEFGHIJKLMNOPQRSTUVWXYZ`` gives back the ordinary A–Z table, so callers
that are not working in a keyed alphabet lose nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .scoring import get_scorer

__all__ = [
    "ALPHABET_RING",
    "KRYPTOS_RING",
    "RingFlatTable",
    "ring_flat_table",
    "ring_letter_map",
    "ring_ngram_table",
    "ring_score",
    "ring_to_letters",
]

KRYPTOS_RING = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
ALPHABET_RING = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# keys vary in arity: nested tables key on (name, ring, lang), flat ones prepend "flat"
_CACHE: dict[tuple[str, ...], object] = {}


def _validate(ring: str) -> None:
    if len(ring) != 26 or len(set(ring)) != 26 or not ring.isalpha():
        raise ValueError(f"ring must be a permutation of the 26 letters, got {ring!r}")


def ring_letter_map(ring: str) -> list[int]:
    """``map[ring_position] = A–Z index``. The conversion this module exists to encapsulate."""
    _validate(ring)
    return [ord(ring[i].upper()) - 65 for i in range(26)]


def ring_to_letters(indices: Sequence[int], ring: str) -> str:
    """Render ring positions as text. Use before anything that expects letters."""
    _validate(ring)
    return "".join(ring[i] for i in indices)


def ring_ngram_table(name: str, ring: str, *, lang: str = "english") -> Any:
    """An n-gram log-probability table whose axes are RING positions, not A–Z indices.

    ``name`` is any table :func:`buttcrack.scoring.get_scorer` accepts (``"bigrams"``,
    ``"trigrams"``, ``"quadgrams"``, …). The result is nested ``n`` levels deep, each of length 26,
    indexed by ring position; unseen n-grams carry the scorer's floor.

    Cost is ``26^n`` cells, so this is intended for n ≤ 3 as a nested list. For quadgrams prefer a
    flat numpy table built the same way (see ``superposition._quad_table``) — the *principle* is
    what matters: build the ring into the table once, rather than converting at every call site
    and eventually forgetting.
    """
    _validate(ring)
    key = (name, ring, lang)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    sc = get_scorer(name, lang)
    n = sc.n
    if n > 3:
        raise ValueError(
            f"{name!r} has n={n}; a nested list would be 26^{n} cells. Use "
            f"ring_flat_table({name!r}, ring) instead — same ring-folding principle, "
            f"workable memory, and it handles every order."
        )
    r2a = ring_letter_map(ring)
    a2r = [0] * 26
    for r, a in enumerate(r2a):
        a2r[a] = r

    def build(depth: int) -> Any:
        if depth == 0:
            return sc.floor
        return [build(depth - 1) for _ in range(26)]

    table = build(n)
    for gram, lp in sc.log_probs.items():
        if len(gram) != n or not gram.isalpha():
            continue
        cur = table
        idx = [a2r[ord(ch.upper()) - 65] for ch in gram]
        for i in idx[:-1]:
            cur = cur[i]
        cur[idx[-1]] = float(lp)
    _CACHE[key] = table
    return table


class RingFlatTable:
    """A flat, ring-indexed n-gram table that works at **every** order, including 5 and 6.

    :func:`ring_ngram_table` stops at n=3 because a nested list of 26^n cells stops being
    reasonable. That left the high orders to be hand-rolled at each call site — and hand-rolling is
    exactly where the ring/A–Z confusion documented at the top of this module keeps reappearing. It
    reappeared *again* while building a joint running-key solver: the streams were ring positions,
    the hand-built quadgram table was A–Z indexed, and the true English pair scored 500 log units
    **below** garbage. That is what this class exists to make impossible.

    Storage adapts to the order: dense ``numpy`` below ~12M cells (n ≤ 5), a sparse dict above it
    (26^6 dense would be 1.2 GB). Either way ``table[idx]`` takes a **ring-position index** built by
    :meth:`index`, and unseen n-grams return :attr:`floor`.

        tab = ring_flat_table("hexagrams", KRYPTOS_RING)
        tab[tab.index(positions)]        # positions are RING positions — correct by construction
    """

    _DENSE_MAX = 26**5

    def __init__(self, name: str, ring: str, *, lang: str = "english") -> None:
        _validate(ring)
        sc = get_scorer(name, lang)
        self.n = int(sc.n)
        self.floor = float(sc.floor)
        self.ring = ring
        r2a = ring_letter_map(ring)
        a2r = [0] * 26
        for r, a in enumerate(r2a):
            a2r[a] = r

        size = 26**self.n
        self.dense = size <= self._DENSE_MAX
        table: Any
        if self.dense:
            import numpy as np

            table = np.full(size, self.floor, dtype=np.float32)
        else:
            table = {}
        for gram, lp in sc.log_probs.items():
            if len(gram) != self.n or not gram.isalpha():
                continue
            idx = 0
            for ch in gram.upper():
                idx = idx * 26 + a2r[ord(ch) - 65]
            table[idx] = float(lp)
        self._t = table

    def index(self, positions: Sequence[int]) -> int:
        """Pack ring positions into a table index, most-significant first."""
        idx = 0
        for r in positions:
            idx = idx * 26 + int(r)
        return idx

    def __getitem__(self, idx: int) -> float:
        if self.dense:
            return float(self._t[idx])
        return float(self._t.get(idx, self.floor))

    def score(self, positions: Sequence[int]) -> float:
        """Total log-probability over sliding windows of ring-indexed ``positions``."""
        n = self.n
        if len(positions) < n:
            return math.nan
        return sum(self[self.index(positions[i : i + n])] for i in range(len(positions) - n + 1))


def ring_flat_table(name: str, ring: str, *, lang: str = "english") -> RingFlatTable:
    """Cached :class:`RingFlatTable` — the ring-safe way to score at orders 4, 5 and 6."""
    key = ("flat", name, ring, lang)
    cached = _CACHE.get(key)
    if cached is not None:
        assert isinstance(cached, RingFlatTable)
        return cached
    built = RingFlatTable(name, ring, lang=lang)
    _CACHE[key] = built
    return built


def ring_score(indices: Sequence[int], table: Any, n: int) -> float:
    """Mean log-probability of ring-indexed ``indices`` under a ring-indexed ``table``.

    Both arguments are in the same index space by construction, which is the whole point.
    """
    if len(indices) < n:
        return float("nan")
    total = 0.0
    count = 0
    for i in range(len(indices) - n + 1):
        cur: Any = table
        for j in range(n):
            cur = cur[indices[i + j]]
        total += float(cur)
        count += 1
    return total / count if count else math.nan
