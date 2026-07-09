"""Generic simulated-annealing search over a discrete key space.

Most classical-cipher cracks are the same shape: search a discrete state — a
substitution alphabet, a column permutation, a keyed alphabet + cycleword — to
maximise an n-gram fitness. This is the shared, restart-able simulated-annealing
engine (the AZdecrypt-class search the crackers otherwise reimplement ad hoc):
a caller supplies only ``init`` (a fresh random state per restart), ``neighbour``
(a small mutation), and ``score`` (higher = better).

Metropolis acceptance with geometric cooling lets the search climb out of the
local optima that trap plain greedy hill-climbing (the failure mode seen on long
keyed alphabets and layered ciphers).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TypeVar

S = TypeVar("S")
T = TypeVar("T")


def anneal(
    init: Callable[[], S],
    neighbour: Callable[[S, object], S],
    score: Callable[[S], float],
    *,
    rng,
    restarts: int = 8,
    iters_per_temp: int = 200,
    temp0: float = 8.0,
    cooling: float = 0.92,
    min_temp: float = 0.05,
    deadline: float | None = None,
) -> tuple[S, float]:
    """Maximise ``score`` over states reachable by ``neighbour`` moves.

    ``init()`` returns a fresh (usually random) start per restart; ``neighbour``
    returns a *near* state (a copy with one small change); ``score`` is the fitness
    (higher better). Returns ``(best_state, best_score)``. At least one start is
    always evaluated (even with an already-expired ``deadline``), so the returned
    state is never ``None``.
    """
    best_state = init()
    best_score = score(best_state)
    restart = 0
    while restart < restarts:
        restart += 1
        cur, cur_score = (best_state, best_score) if restart == 1 else _fresh(init, score)
        if cur_score > best_score:
            best_state, best_score = cur, cur_score
        temp = temp0
        while temp > min_temp:
            if deadline is not None and time.monotonic() > deadline:
                return best_state, best_score
            for _ in range(iters_per_temp):
                cand = neighbour(cur, rng)
                cand_score = score(cand)
                delta = cand_score - cur_score
                if delta > 0 or rng.random() < math.exp(delta / temp):
                    cur, cur_score = cand, cand_score
                    if cur_score > best_score:
                        best_state, best_score = cur, cur_score
            temp *= cooling
    return best_state, best_score


def _fresh(init: Callable[[], S], score: Callable[[S], float]) -> tuple[S, float]:
    state = init()
    return state, score(state)


def swap_neighbour(state: list[T], rng) -> list[T]:
    """A copy of ``state`` (a list) with two random positions swapped.

    The standard move for permutation/alphabet search (substitution keys, keyed
    alphabets, column orders).
    """
    out = list(state)
    i, j = rng.randrange(len(out)), rng.randrange(len(out))
    out[i], out[j] = out[j], out[i]
    return out


def shuffled(items: list[T], rng) -> list[T]:
    """A fresh shuffled copy of ``items`` — a convenient ``init`` for restarts."""
    out = list(items)
    rng.shuffle(out)
    return out
