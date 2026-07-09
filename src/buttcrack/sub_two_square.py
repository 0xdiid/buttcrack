"""Decoupled recovery of a periodic substitution laid OVER a Two-square (double Playfair) inner.

The construction this attacks is ``CT = outer( two_square_inner( PT ) )`` where

* ``two_square_inner`` is a classic Two-square / double-Playfair cipher over two keyed 5x5
  squares (see :mod:`buttcrack.ciphers.two_square`) built on the 25-letter Q-dropped alphabet
  (I and J both kept), in the VERTICAL or HORIZONTAL layout; and
* ``outer`` is a periodic substitution — a Vigenere / Quagmire-III shift of period
  ``outer_period`` over a keyed ``outer_alphabet`` (default the KRYPTOS keyed alphabet).

This is the double-Playfair sibling of :mod:`buttcrack.sub_four_square` (four-square) and
:mod:`buttcrack.sub_playfair` (single-square Playfair). Like four-square it uses a **pair** of
keyed squares — the setter's paired-thematic-word signature — and its output is drop-letter free
(both squares drop the same letter), so the outer period-``p`` shift is pruned per-coset by the
drop-letter rule (the sole structural lever; two-square, being reciprocal with a transparency
rule, imposes no no-doubled-digraph constraint). The search is over the square PAIR (top, bottom)
and the layout; for each, the outer key is recovered by the drop-letter-pruned descent.

Public API
----------
``sub_encode`` / ``sub_decode``                  periodic shift over a keyed alphabet (re-exported)
``two_square_transform``                          the reciprocal two-square map given two grids
``encrypt_sub_over_two_square``                   plant CT = outer(two_square(PT))
``recover_outer_key_over_two_square``             key-given-structure recovery (one pair/layout)
``crack_sub_over_two_square``                     driver over candidate square pairs + layouts/periods
"""

from __future__ import annotations

import random as _random
import time
from collections.abc import Callable, Iterable, Sequence
from itertools import product

from .scoring import index_of_coincidence  # noqa: F401  (handy for callers)
from .sub_four_square import _admissible_shifts, fs_alphabet, fs_grid
from .sub_fractionation import make_objective, resolve_alphabet, sub_decode, sub_encode
from .text import only_letters

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

__all__ = [
    "fs_alphabet",
    "fs_grid",
    "two_square_transform",
    "sub_encode",
    "sub_decode",
    "encrypt_sub_over_two_square",
    "recover_outer_key_over_two_square",
    "crack_sub_over_two_square",
]


# --------------------------------------------------------------------------- #
# Two-square primitive (reciprocal: the same map encodes and decodes)
# --------------------------------------------------------------------------- #
def two_square_transform(text: str, top_grid: str, bot_grid: str, *, vertical: bool = True) -> str:
    """Apply the reciprocal Two-square map to ``text`` over two 25-cell grids.

    Digraph ``(a, b)``: ``a`` is located in ``top_grid``, ``b`` in ``bot_grid``. VERTICAL — if
    they share a column the pair passes through unchanged (transparency), else take the opposite
    rectangle corners (top-square letter first). HORIZONTAL — same but the transparency is a
    shared row. Odd length: the trailing letter is dropped (callers pad even). Encoding and
    decoding are identical (the map is an involution).
    """
    tp = {c: i for i, c in enumerate(top_grid)}
    bp = {c: i for i, c in enumerate(bot_grid)}
    # keep only letters present in the (Q-dropped) squares; residuals already satisfy this,
    # but plaintext passed to the encode helper may contain the dropped letter.
    grid_set = set(top_grid)
    s = "".join(c for c in only_letters(text) if c in grid_set)
    n = len(s) - (len(s) % 2)
    out: list[str] = []
    for i in range(0, n, 2):
        a, b = s[i], s[i + 1]
        ra, ca = divmod(tp[a], 5)
        rb, cb = divmod(bp[b], 5)
        share = (ca == cb) if vertical else (ra == rb)
        if share:
            out.append(a)
            out.append(b)
        else:
            out.append(top_grid[ra * 5 + cb])
            out.append(bot_grid[rb * 5 + ca])
    return "".join(out)


def encrypt_sub_over_two_square(
    pt: str,
    top_keyword: str,
    bot_keyword: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_shifts: Sequence[int],
    drop_letter: str = "Q",
    vertical: bool = True,
) -> str:
    """Plant ``CT = periodic_sub( two_square( PT ) )`` — the structure this module attacks."""
    alpha = resolve_alphabet(outer_alphabet)
    a25 = fs_alphabet(drop_letter)
    prepared = "".join(c for c in pt.upper() if c in set(a25))
    inner = two_square_transform(prepared, fs_grid(top_keyword, drop_letter),
                                 fs_grid(bot_keyword, drop_letter), vertical=vertical)
    return sub_encode(inner, alpha, outer_shifts)


# --------------------------------------------------------------------------- #
# Key-given-structure recovery (one candidate pair + layout)
# --------------------------------------------------------------------------- #
def recover_outer_key_over_two_square(
    ciphertext: str,
    top_grid: str,
    bot_grid: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_period: int,
    drop_letter: str = "Q",
    vertical: bool = True,
    objective: str = "fitness",
    languages: Sequence[str] = ("english",),
    objective_fn: Callable[[str], float] | None = None,
    restarts: int = 6,
    brute_cap: int = 200000,
    exhaustive: bool | None = None,
    rng=None,
    max_passes: int = 10,
) -> tuple[list[int], str, float]:
    """Recover the outer period-``p`` shift key GIVEN a Two-square pair and layout.

    Prunes each column's shifts by the drop-letter rule, then finds the best combination by the
    objective score of the FULL de-two-squared decode — exhaustively when the admissible product
    is small, else by constrained coordinate descent with restarts. Returns
    ``(shifts, plaintext, score)``.
    """
    rng = rng or _random.Random(0)
    A = resolve_alphabet(outer_alphabet)
    aidx = {ch: i for i, ch in enumerate(A)}
    letters = only_letters(ciphertext)
    if len(letters) % 2:
        letters = letters[:-1]
    n = len(letters)
    p = int(outer_period)
    if n < p or p < 1:
        return ([0] * max(p, 1), "", float("-inf"))
    c_idx = [aidx[ch] for ch in letters]
    drop = str(drop_letter).upper()[:1]
    drop_pos = aidx[drop]

    obj = objective_fn or make_objective(objective, languages=languages)
    allowed = _admissible_shifts(c_idx, n, p, drop_pos)

    def decode_for(shifts: Sequence[int]) -> str:
        residual = "".join(A[(c_idx[i] - shifts[i % p]) % 26] for i in range(n))
        return two_square_transform(residual, top_grid, bot_grid, vertical=vertical)

    def score(shifts: Sequence[int]) -> tuple[float, str]:
        pt = decode_for(shifts)
        return obj(pt), pt

    prod = 1
    for a in allowed:
        prod *= len(a)
        if prod > brute_cap:
            break

    if exhaustive or (exhaustive is None and prod <= brute_cap):
        best: tuple[float, list[int], str] | None = None
        for combo in product(*allowed):
            sc, pt = score(list(combo))
            if best is None or sc > best[0]:
                best = (sc, list(combo), pt)
        assert best is not None
        return best[1], best[2], best[0]

    def descend(init: list[int]) -> tuple[list[int], str, float]:
        shifts = list(init)
        best_sc, best_pt = score(shifts)
        improved, passes = True, 0
        while improved and passes < max_passes:
            improved, passes = False, passes + 1
            for j in range(p):
                loc_s, loc_sc, loc_pt = shifts[j], best_sc, best_pt
                for s in allowed[j]:
                    if s == shifts[j]:
                        continue
                    shifts[j] = s
                    sc, pt = score(shifts)
                    if sc > loc_sc:
                        loc_sc, loc_s, loc_pt = sc, s, pt
                shifts[j] = loc_s
                if loc_sc > best_sc + 1e-12:
                    best_sc, best_pt, improved = loc_sc, loc_pt, True
        return shifts, best_pt, best_sc

    overall = descend([col[0] for col in allowed])
    for _ in range(max(0, restarts - 1)):
        init = [col[rng.randrange(len(col))] for col in allowed]
        cand = descend(init)
        if cand[2] > overall[2]:
            overall = cand
    return overall


# --------------------------------------------------------------------------- #
# Driver: scan candidate square pairs + layouts, recover the outer key, rank
# --------------------------------------------------------------------------- #
def _kw_iter(squares, drop_letter: str) -> list[tuple[str, str]]:
    if isinstance(squares, str) and squares.lower() in ("dictionary", "dict", "keywords"):
        from .ciphers._quagmire_solver import BUILTIN_KEYWORDS

        words: Iterable[str] = BUILTIN_KEYWORDS
        return [(w, fs_grid(w, drop_letter)) for w in words]
    return [(str(item), fs_grid(str(item), drop_letter)) for item in squares]


def _drop_list(drop_letter) -> list[str]:
    if drop_letter is None:
        return ["Q"]
    if isinstance(drop_letter, str):
        s = drop_letter.strip()
        if s.lower() in ("sweep", "all", "*"):
            return list(_STD)
        letters = [c for c in s.upper() if "A" <= c <= "Z"]
        return letters or ["Q"]
    return [str(x).upper()[:1] for x in drop_letter]


def _shifts_to_keystr(shifts: Sequence[int]) -> str:
    return "".join(chr(65 + (s % 26)) for s in shifts)


def crack_sub_over_two_square(
    ciphertext: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_period: int | Iterable[int] = 7,
    top_squares="dictionary",
    bot_squares=None,
    layouts: Sequence[bool] = (True, False),
    objective: str = "fitness",
    drop_letter: str | None = None,
    languages: Sequence[str] = ("english",),
    top: int = 5,
    timeout: float | None = None,
    rng=None,
    brute_cap: int = 200000,
) -> list[tuple[str, str, bool, str, str, float]]:
    """Crack ``CT = periodic_sub( two_square( PT ) )`` by scanning candidate square PAIRS + layouts.

    ``top_squares`` and ``bot_squares`` are keyword iterables (``bot_squares=None`` reuses
    ``top_squares``); every ordered pair and each layout in ``layouts`` (True=vertical) is tried.
    For each, the outer key is recovered by :func:`recover_outer_key_over_two_square` and
    hypotheses are ranked by the decode's ``objective``. Returns up to ``top`` tuples
    ``(top_grid, bot_grid, vertical, key, plaintext, score)`` best-first.
    """
    drops = _drop_list(drop_letter or "Q")
    letters = only_letters(ciphertext)
    obj = make_objective(objective, languages=languages)
    periods = [outer_period] if isinstance(outer_period, int) else [int(x) for x in outer_period]
    deadline = (time.monotonic() + timeout) if timeout else None

    results: list[tuple[float, str, str, bool, str, str]] = []
    for drop in drops:
        if deadline and time.monotonic() > deadline:
            break
        top_list = _kw_iter(top_squares, drop)
        bot_list = top_list if bot_squares is None else _kw_iter(bot_squares, drop)
        for _tk, tg in top_list:
            if deadline and time.monotonic() > deadline:
                break
            for _bk, bg in bot_list:
                if deadline and time.monotonic() > deadline:
                    break
                for vertical in layouts:
                    for op in periods:
                        shifts, pt, score = recover_outer_key_over_two_square(
                            letters, tg, bg, outer_alphabet=outer_alphabet, outer_period=op,
                            drop_letter=drop, vertical=vertical, objective_fn=obj, rng=rng,
                            brute_cap=brute_cap)
                        if not pt:
                            continue
                        results.append((score, tg, bg, vertical, _shifts_to_keystr(shifts), pt))

    results.sort(key=lambda r: r[0], reverse=True)
    seen: set[str] = set()
    out: list[tuple[str, str, bool, str, str, float]] = []
    for score, tg, bg, vertical, keystr, pt in results:
        if pt in seen:
            continue
        seen.add(pt)
        out.append((tg, bg, vertical, keystr, pt, score))
        if len(out) >= top:
            break
    return out
