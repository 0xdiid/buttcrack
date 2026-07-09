"""Decoupled recovery of a periodic substitution laid OVER a Four-square (or Two-square) inner.

The construction this attacks is ``CT = outer( four_square_inner( PT ) )`` where

* ``four_square_inner`` is a classic Four-square (two keyed cipher squares + two straight
  plaintext squares; see :mod:`buttcrack.ciphers.four_square`) over a 25-letter alphabet that
  DROPS one letter (canonical Four-square drops ``Q`` and keeps both I and J); and
* ``outer`` is a periodic substitution — a Vigenere / Quagmire-III shift of period
  ``outer_period`` over a keyed ``outer_alphabet`` (default the KRYPTOS keyed alphabet).

This is the *two-keyed-square* sibling of :mod:`buttcrack.sub_playfair`. It matters for
puzzle families whose setters favour **paired thematic keywords**, because Four-square /
Two-square are the digraphic-class flatteners that use a *pair* of keyed squares —
matching that signature exactly.

Why the decoupling works. Four-square emits its first digraph letter from the top-right cipher
square and its second from the bottom-left cipher square; both are 25-cell squares that drop the
same letter, so the residual (the Four-square ciphertext, recovered by stripping the outer
substitution) never contains the drop letter. For a period-``p`` outer shift each key position
acts on its own coset, so a candidate shift ``s`` for column ``j`` is admissible only if
stripping it leaves that coset free of the drop letter — pruning each position to a small
candidate set (the same lever as :mod:`sub_playfair`). Unlike Playfair there is **no**
no-doubled-digraph rule (Four-square can emit equal pairs), so drop-letter admissibility is the
sole structural lever; controls show it is enough to separate the correct square *pair*.

Because the plaintext squares are the straight alphabet, the search is over the square PAIR
(top-right, bottom-left); the driver scans a candidate pair set and, for each pair, recovers the
outer key by the drop-letter-pruned descent, ranking by the decode's objective.

Public API
----------
``sub_encode`` / ``sub_decode``                      periodic shift over a keyed alphabet (re-exported)
``four_square_encode`` / ``four_square_decode``       Four-square inner given two 25-cell grids
``encrypt_sub_over_four_square``                      plant CT = outer(four_square(PT))
``recover_outer_key_over_four_square``                key-given-structure recovery (one pair)
``crack_sub_over_four_square``                        driver over candidate square pairs + periods/drops
"""

from __future__ import annotations

import random as _random
import time
from collections.abc import Callable, Iterable, Sequence
from itertools import product

from .ciphers.squares import PolybiusSquare
from .scoring import index_of_coincidence  # noqa: F401  (handy for callers)
from .sub_fractionation import make_objective, resolve_alphabet, sub_decode, sub_encode
from .text import only_letters

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"

__all__ = [
    "fs_alphabet",
    "fs_grid",
    "four_square_encode",
    "four_square_decode",
    "sub_encode",
    "sub_decode",
    "encrypt_sub_over_four_square",
    "recover_outer_key_over_four_square",
    "crack_sub_over_four_square",
]


# --------------------------------------------------------------------------- #
# Four-square primitives
# --------------------------------------------------------------------------- #
def fs_alphabet(drop_letter: str = "Q") -> str:
    """The 25-letter straight alphabet for a Four-square that omits ``drop_letter``.

    Canonical Four-square drops ``Q`` (keeping both I and J); pass ``"J"`` for the KRYPTOS-style
    I/J merge convention used elsewhere in the series.
    """
    d = str(drop_letter).upper()[:1]
    return "".join(c for c in _STD if c != d)


def fs_grid(keyword: str, drop_letter: str = "Q") -> str:
    """Return the 25-letter row-by-row cipher square for a keyword (or full permutation)."""
    return "".join(PolybiusSquare(keyword, size=5, alphabet=fs_alphabet(drop_letter)).grid)


def four_square_encode(pt: str, tr_grid: str, bl_grid: str, alphabet: str) -> str:
    """Four-square encode over two cipher grids and a straight ``alphabet`` (plaintext squares).

    Keeps only ``alphabet`` letters and pads to even length with the last alphabet letter's
    successor is *not* assumed; a lone final letter is padded with ``X`` when present in the
    alphabet else the drop-complement's first letter.
    """
    pos = {c: i for i, c in enumerate(alphabet)}
    s = "".join(c for c in pt.upper() if c in pos)
    if len(s) % 2:
        s += "X" if "X" in pos else alphabet[-1]
    out: list[str] = []
    for i in range(0, len(s), 2):
        ra, ca = divmod(pos[s[i]], 5)
        rb, cb = divmod(pos[s[i + 1]], 5)
        out.append(tr_grid[ra * 5 + cb])
        out.append(bl_grid[rb * 5 + ca])
    return "".join(out)


def four_square_decode(residual: str, tr_grid: str, bl_grid: str, alphabet: str) -> str:
    """De-Four-square ``residual`` over two cipher grids and a straight ``alphabet``."""
    tr_pos = {c: i for i, c in enumerate(tr_grid)}
    bl_pos = {c: i for i, c in enumerate(bl_grid)}
    out: list[str] = []
    n = len(residual) - (len(residual) % 2)
    for i in range(0, n, 2):
        i1 = tr_pos[residual[i]]
        i2 = bl_pos[residual[i + 1]]
        r1, col1 = divmod(i1, 5)
        r2, col2 = divmod(i2, 5)
        out.append(alphabet[r1 * 5 + col2])
        out.append(alphabet[r2 * 5 + col1])
    return "".join(out)


def encrypt_sub_over_four_square(
    pt: str,
    tr_keyword: str,
    bl_keyword: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_shifts: Sequence[int],
    drop_letter: str = "Q",
) -> str:
    """Plant ``CT = periodic_sub( four_square( PT ) )`` — the structure this module attacks."""
    alpha = resolve_alphabet(outer_alphabet)
    a25 = fs_alphabet(drop_letter)
    inner = four_square_encode(pt, fs_grid(tr_keyword, drop_letter),
                               fs_grid(bl_keyword, drop_letter), a25)
    return sub_encode(inner, alpha, outer_shifts)


# --------------------------------------------------------------------------- #
# Key-given-structure recovery (one candidate square pair)
# --------------------------------------------------------------------------- #
def _admissible_shifts(c_idx: list[int], n: int, p: int, drop_pos: int) -> list[list[int]]:
    """Per-column shifts leaving the residual coset free of the drop letter (sole lever)."""
    allowed: list[list[int]] = []
    for j in range(p):
        col_vals = {c_idx[i] for i in range(j, n, p)}
        banned = {(v - drop_pos) % 26 for v in col_vals}
        allowed.append([s for s in range(26) if s not in banned] or list(range(26)))
    return allowed


def recover_outer_key_over_four_square(
    ciphertext: str,
    tr_grid: str,
    bl_grid: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_period: int,
    drop_letter: str = "Q",
    objective: str = "fitness",
    languages: Sequence[str] = ("english",),
    objective_fn: Callable[[str], float] | None = None,
    restarts: int = 6,
    brute_cap: int = 200000,
    exhaustive: bool | None = None,
    rng=None,
    max_passes: int = 10,
) -> tuple[list[int], str, float]:
    """Recover the outer period-``p`` shift key GIVEN a candidate Four-square pair.

    Prunes each column's shifts by the drop-letter rule, then finds the best combination by the
    objective score of the FULL de-Four-squared decode — exhaustively when the admissible
    product is small, else by constrained coordinate descent with restarts. Returns
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
    a25 = fs_alphabet(drop)

    obj = objective_fn or make_objective(objective, languages=languages)
    allowed = _admissible_shifts(c_idx, n, p, drop_pos)

    def decode_for(shifts: Sequence[int]) -> str:
        residual = "".join(A[(c_idx[i] - shifts[i % p]) % 26] for i in range(n))
        return four_square_decode(residual, tr_grid, bl_grid, a25)

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
# Driver: scan candidate square pairs, recover the outer key under each, rank
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


def crack_sub_over_four_square(
    ciphertext: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_period: int | Iterable[int] = 7,
    tr_squares="dictionary",
    bl_squares=None,
    objective: str = "fitness",
    drop_letter: str | None = None,
    languages: Sequence[str] = ("english",),
    top: int = 5,
    timeout: float | None = None,
    rng=None,
    brute_cap: int = 200000,
) -> list[tuple[str, str, str, str, float]]:
    """Crack ``CT = periodic_sub( four_square( PT ) )`` by scanning candidate square PAIRS.

    ``tr_squares`` and ``bl_squares`` are keyword iterables (``bl_squares=None`` reuses
    ``tr_squares``); every ordered pair is tried. For each pair and ``outer_period`` the outer
    key is recovered by :func:`recover_outer_key_over_four_square` and hypotheses are ranked by
    the decode's ``objective``. Returns up to ``top`` tuples
    ``(tr_grid, bl_grid, key, plaintext, score)`` best-first.
    """
    drops = _drop_list(drop_letter or "Q")
    letters = only_letters(ciphertext)
    obj = make_objective(objective, languages=languages)
    periods = [outer_period] if isinstance(outer_period, int) else [int(x) for x in outer_period]
    deadline = (time.monotonic() + timeout) if timeout else None

    results: list[tuple[float, str, str, str, str]] = []
    for drop in drops:
        if deadline and time.monotonic() > deadline:
            break
        tr_list = _kw_iter(tr_squares, drop)
        bl_list = tr_list if bl_squares is None else _kw_iter(bl_squares, drop)
        for _tk, tg in tr_list:
            if deadline and time.monotonic() > deadline:
                break
            for _bk, bg in bl_list:
                if deadline and time.monotonic() > deadline:
                    break
                for op in periods:
                    shifts, pt, score = recover_outer_key_over_four_square(
                        letters, tg, bg, outer_alphabet=outer_alphabet, outer_period=op,
                        drop_letter=drop, objective_fn=obj, rng=rng, brute_cap=brute_cap)
                    if not pt:
                        continue
                    results.append((score, tg, bg, _shifts_to_keystr(shifts), pt))

    results.sort(key=lambda r: r[0], reverse=True)
    seen: set[str] = set()
    out: list[tuple[str, str, str, str, float]] = []
    for score, tg, bg, keystr, pt in results:
        if pt in seen:
            continue
        seen.add(pt)
        out.append((tg, bg, keystr, pt, score))
        if len(out) >= top:
            break
    return out
