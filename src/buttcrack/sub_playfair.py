"""Decoupled recovery of a periodic substitution laid OVER a Playfair inner.

The construction this attacks is ``CT = outer( playfair_inner( PT ) )`` where

* ``playfair_inner`` is a classic 5x5 Playfair over a keyed square that DROPS one
  letter (default ``J``, merged into ``I``); and
* ``outer`` is a periodic substitution — a Vigenere / Quagmire-III shift of period
  ``p`` over a keyed ``outer_alphabet`` (default the KRYPTOS keyed alphabet).

This is the Playfair sibling of :mod:`buttcrack.sub_fractionation` (which attacks a
*bifid* inner). A joint blind search over ``key x square`` is hopeless (two coupled
isolated optima); the decoupling that makes it tractable is a pair of STRUCTURAL
CONSTRAINTS on the intermediate stream (the residual left after stripping the outer
substitution — i.e. the Playfair ciphertext):

1. **Drop-letter-free.** A 25-cell Playfair square can never emit its dropped letter, so
   the residual contains no ``J`` (or whatever letter the square omits). For a period-``p``
   outer shift each key position acts on its own coset, so a candidate shift ``s`` for
   column ``j`` is admissible only if stripping it leaves that coset free of the drop
   letter — pruning each position to a small candidate set (same lever as ``sub_fractionation``).
2. **No doubled digraph.** Playfair maps every plaintext digraph ``(a, b)`` with ``a != b``
   to a ciphertext digraph whose two letters also differ (same-row / same-column / rectangle
   rules all preserve distinctness). So the residual never has an equal adjacent pair at an
   even/odd digraph boundary. For each digraph ``(i, i+1)`` in columns ``(j1, j2)`` this bans
   exactly one value of the relative shift ``s_j2 - s_j1`` — a strong cross-column constraint
   Playfair enjoys that bifid does not.

Within the admissible shifts the outer key is recovered by constrained coordinate descent
(or exhaustive scan when the admissible product is small), scoring the FULL de-Playfaired
decode with a pluggable objective and a penalty for any residual doubled digraph. The
discriminator is sharp: with the CORRECT square the recovered decode reads as language
(quadgram fitness well clear of the wrong-square plateau); with any wrong square it plateaus.

Public API
----------
``sub_encode`` / ``sub_decode``                  periodic shift over a keyed alphabet (re-exported)
``playfair_encode`` / ``playfair_decode``         Playfair inner given a 25-cell square string
``encrypt_sub_over_playfair``                     plant CT = outer(playfair(PT))
``recover_outer_key_over_playfair``               key-given-structure recovery (one square)
``crack_sub_over_playfair``                        driver over candidate squares + periods/drops
"""

from __future__ import annotations

import random as _random
import time
from collections.abc import Callable, Iterable, Sequence
from itertools import product

from .ciphers.rectangular_playfair import _transform as _rect_transform
from .ciphers.rectangular_playfair import grid_letters as _rect_grid_letters
from .ciphers.squares import PolybiusSquare
from .scoring import index_of_coincidence  # noqa: F401  (handy for callers)
from .sub_fractionation import (
    make_objective,
    resolve_alphabet,
    sub_decode,
    sub_encode,
)
from .text import only_letters

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"

__all__ = [
    "square_alphabet5",
    "resolve_square",
    "resolve_grid",
    "playfair_encode",
    "playfair_decode",
    "sub_encode",
    "sub_decode",
    "encrypt_sub_over_playfair",
    "recover_outer_key_over_playfair",
    "crack_sub_over_playfair",
]


# --------------------------------------------------------------------------- #
# Square / Playfair primitives
# --------------------------------------------------------------------------- #
def square_alphabet5(drop_letter: str = "J") -> str:
    """The 25-letter alphabet for a 5x5 square that omits ``drop_letter``."""
    d = str(drop_letter).upper()[:1]
    return "".join(c for c in _STD if c != d)


def resolve_square(item: str, drop_letter: str = "J") -> str:
    """Return the 25-letter row-by-row square string for a keyword or full permutation."""
    sq = PolybiusSquare(item, size=5, alphabet=square_alphabet5(drop_letter))
    return "".join(sq.grid)


def resolve_grid(item: str, shape: tuple[int, int] = (5, 5), drop_letter: str = "J") -> str:
    """Return the ``rows*cols``-cell grid string for a keyword/permutation and ``shape``.

    A 25-cell grid drops ``drop_letter`` (the classic square); a 26-cell grid (e.g. ``2x13``)
    keeps all 26 letters, so no letter is merged and the intermediate can contain any letter —
    the regime a 5x5 square cannot reach (it never emits its dropped letter).
    """
    rows, cols = shape
    cells = rows * cols
    if cells == 25:
        return resolve_square(item, drop_letter)
    if cells == 26:
        return _rect_grid_letters(item, rows, cols)
    raise ValueError(f"sub-over-Playfair grid must be 25 or 26 cells; {rows}x{cols}={cells}")


def _grid_meta(grid: str) -> tuple[str | None, str | None, str]:
    """Return ``(dropped, merge_target, filler)`` for a grid.

    For a 25-cell grid: ``dropped`` is the absent letter, ``merge_target`` the present neighbour
    it folds onto (I for the classic J-drop), ``filler`` the double-splitter. For a 26-cell grid
    no letter is dropped, so ``dropped``/``merge_target`` are ``None``.
    """
    present = set(grid)
    if len(present) == 26:
        return None, None, ("X" if "X" in present else "Q")
    dropped = next((c for c in _STD if c not in present), "J")
    i = _STD.index(dropped)
    merge_target = next((_STD[(i + d) % 26] for d in (-1, 1, -2, 2) if _STD[(i + d) % 26] in present),
                        next(iter(present)))
    filler = "X" if "X" in present else next(c for c in "QZ" + _STD if c in present)
    return dropped, merge_target, filler


#: kept for backward compatibility (25-cell only); prefer :func:`_grid_meta`.
def _square_meta(square25: str) -> tuple[str, str, str]:
    dropped, merge, filler = _grid_meta(square25)
    return dropped or "J", merge or "I", filler


def _pairs_drop_aware(letters: str, grid: str) -> list[tuple[str, str]]:
    """Even-length digraph split; folds the grid's dropped letter (if any) onto its merge-mate."""
    dropped, merge_target, _ = _grid_meta(grid)
    s = letters if dropped is None else letters.replace(dropped, merge_target)
    if len(s) % 2:
        s = s[:-1]
    return [(s[i], s[i + 1]) for i in range(0, len(s), 2)]


def _prepare_drop_aware(letters: str, grid: str) -> list[tuple[str, str]]:
    """Encode-prepare (fold drop if any, split doubles with a present filler, pad tail)."""
    dropped, merge_target, filler = _grid_meta(grid)
    s = letters if dropped is None else letters.replace(dropped, merge_target)
    fallback = merge_target or (next(c for c in _STD if c in set(grid) and c != filler))
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        a = s[i]
        b = s[i + 1] if i + 1 < len(s) else ""
        if b == "" or a == b:
            pairs.append((a, filler if a != filler else fallback))
            i += 1
        else:
            pairs.append((a, b))
            i += 2
    return pairs


def playfair_decode(residual: str, grid: str, shape: tuple[int, int] = (5, 5)) -> str:
    """De-Playfair ``residual`` over ``grid`` of ``shape`` — inverse direction.

    Drop-letter aware (25-cell) and drop-free (26-cell). Residuals from the pruned recovery never
    contain a 25-cell grid's dropped letter, so no folding occurs.
    """
    rows, cols = shape
    return _rect_transform(_pairs_drop_aware(residual, grid), grid, rows, cols, -1)


def playfair_encode(pt: str, grid: str, shape: tuple[int, int] = (5, 5)) -> str:
    """Playfair-encode ``pt`` (splits doubles, pads a lone final letter) over ``grid`` of ``shape``.

    Drop-letter aware: a 25-cell grid folds its dropped letter onto its mate before enciphering;
    a 26-cell grid uses every letter directly.
    """
    rows, cols = shape
    return _rect_transform(_prepare_drop_aware(only_letters(pt), grid), grid, rows, cols, +1)


def encrypt_sub_over_playfair(
    pt: str,
    square: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_shifts: Sequence[int],
    drop_letter: str = "J",
    shape: tuple[int, int] = (5, 5),
) -> str:
    """Plant ``CT = periodic_sub( playfair( PT ) )`` — the exact structure this module attacks."""
    alpha = resolve_alphabet(outer_alphabet)
    intermediate = playfair_encode(pt, resolve_grid(square, shape, drop_letter), shape)
    return sub_encode(intermediate, alpha, outer_shifts)


# --------------------------------------------------------------------------- #
# Key-given-structure recovery (one candidate square)
# --------------------------------------------------------------------------- #
def _admissible_shifts(c_idx: list[int], n: int, p: int, drop_pos: int | None) -> list[list[int]]:
    """Per-column shifts leaving the residual coset free of the drop letter (constraint 1).

    For a 26-cell grid there is no dropped letter (``drop_pos is None``), so every shift is
    admissible and this constraint is void — the no-doubled-digraph rule (constraint 2) still bites.
    """
    if drop_pos is None:
        return [list(range(26)) for _ in range(p)]
    allowed: list[list[int]] = []
    for j in range(p):
        col_vals = {c_idx[i] for i in range(j, n, p)}
        banned = {(v - drop_pos) % 26 for v in col_vals}
        allowed.append([s for s in range(26) if s not in banned] or list(range(26)))
    return allowed


def _double_banned(c_idx: list[int], n: int, p: int) -> dict[tuple[int, int], set[int]]:
    """Banned relative shifts ``s_j2 - s_j1`` from the no-doubled-digraph rule (constraint 2)."""
    banned: dict[tuple[int, int], set[int]] = {}
    m = n - (n % 2)
    for k in range(0, m, 2):
        j1, j2 = k % p, (k + 1) % p
        banned.setdefault((j1, j2), set()).add((c_idx[k + 1] - c_idx[k]) % 26)
    return banned


def recover_outer_key_over_playfair(
    ciphertext: str,
    grid: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_period: int,
    drop_letter: str = "J",
    shape: tuple[int, int] = (5, 5),
    objective: str = "fitness",
    languages: Sequence[str] = ("english",),
    objective_fn: Callable[[str], float] | None = None,
    restarts: int = 8,
    brute_cap: int = 200000,
    exhaustive: bool | None = None,
    double_penalty: float = 5.0,
    use_double: bool = True,
    rng=None,
    max_passes: int = 10,
) -> tuple[list[int], str, float]:
    """Recover the outer period-``p`` shift key GIVEN a candidate inner Playfair ``grid``.

    Prunes each column's shifts by the drop-letter rule (void for a 26-cell grid), penalises
    residual doubled digraphs, and finds the best combination by objective score of the FULL
    de-Playfaired decode — exhaustively when the admissible product is small, else by constrained
    coordinate descent with random restarts. Returns ``(shifts, plaintext, score)``.
    """
    rng = rng or _random.Random(0)
    A = resolve_alphabet(outer_alphabet)
    aidx = {ch: i for i, ch in enumerate(A)}
    letters = only_letters(ciphertext)
    n = len(letters)
    p = int(outer_period)
    if n < p or p < 1:
        return ([0] * max(p, 1), "", float("-inf"))
    c_idx = [aidx[ch] for ch in letters]
    dropped = _grid_meta(grid)[0]  # the grid's own dropped letter (None for a 26-cell grid)
    drop_pos = None if dropped is None else aidx[dropped]

    obj = objective_fn or make_objective(objective, languages=languages)
    allowed = _admissible_shifts(c_idx, n, p, drop_pos)
    dbanned = _double_banned(c_idx, n, p) if use_double else {}

    def decode_for(shifts: Sequence[int]) -> str:
        residual = "".join(A[(c_idx[i] - shifts[i % p]) % 26] for i in range(n))
        return playfair_decode(residual, grid, shape)

    def penalty(shifts: Sequence[int]) -> float:
        if not dbanned:
            return 0.0
        v = sum(
            1
            for (j1, j2), diffs in dbanned.items()
            if ((shifts[j2] - shifts[j1]) % 26) in diffs
        )
        return double_penalty * v

    def score(shifts: Sequence[int]) -> tuple[float, str]:
        pt = decode_for(shifts)
        return obj(pt) - penalty(shifts), pt

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
# Driver: scan candidate squares, recover the outer key under each, rank
# --------------------------------------------------------------------------- #
def _grid_iter(squares, shape: tuple[int, int], drop_letter: str) -> list[tuple[str, str]]:
    if isinstance(squares, str) and squares.lower() in ("dictionary", "dict", "keywords"):
        from .ciphers._quagmire_solver import BUILTIN_KEYWORDS

        words: Iterable[str] = BUILTIN_KEYWORDS
        return [(w, resolve_grid(w, shape, drop_letter)) for w in words]
    return [(str(item), resolve_grid(str(item), shape, drop_letter)) for item in squares]


def _drop_list(drop_letter) -> list[str]:
    if drop_letter is None:
        return ["J"]
    if isinstance(drop_letter, str):
        s = drop_letter.strip()
        if s.lower() in ("sweep", "all", "*"):
            return [c for c in _STD]
        letters = [c for c in s.upper() if "A" <= c <= "Z"]
        return letters or ["J"]
    return [str(x).upper()[:1] for x in drop_letter]


def _shape_list(shapes) -> list[tuple[int, int]]:
    """Normalise ``shapes`` (a single ``(r, c)`` or an iterable of them) to a list of tuples."""
    if shapes is None:
        return [(5, 5)]
    if isinstance(shapes, tuple) and len(shapes) == 2 and all(isinstance(x, int) for x in shapes):
        return [shapes]
    return [(int(r), int(c)) for r, c in shapes]


def crack_sub_over_playfair(
    ciphertext: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_period: int | Iterable[int] | None = 7,
    squares="dictionary",
    shapes=None,
    objective: str = "fitness",
    drop_letter: str | None = None,
    languages: Sequence[str] = ("english",),
    top: int = 5,
    timeout: float | None = None,
    use_double: bool = True,
    rng=None,
) -> list[tuple[str, str, str, float]]:
    """Crack ``CT = periodic_sub( playfair( PT ) )`` by scanning candidate inner grids.

    For each ``shape`` (``(rows, cols)``; default 5x5), ``drop_letter``, grid
    (keyword/permutation or the built-in ``"dictionary"`` set) and ``outer_period``, recover the
    outer key by :func:`recover_outer_key_over_playfair` and rank hypotheses by the decode's
    ``objective`` score. Returns up to ``top`` tuples ``(grid, key, plaintext, score)`` best-first.

    * ``shapes`` — a single ``(r, c)`` or an iterable of them; 25-cell shapes are classic
      drop-letter squares, 26-cell shapes (e.g. ``(2, 13)``) keep all 26 letters. For a 26-cell
      shape the drop-letter sweep is skipped (there is no dropped letter).
    * ``outer_period`` — an int, an iterable, or ``None`` (sweep 2..12).
    * ``drop_letter`` — defaults to ``"J"``; a letter, letter string, or ``"sweep"`` (25-cell only).
    """
    letters = only_letters(ciphertext)
    obj = make_objective(objective, languages=languages)

    if outer_period is None:
        periods = list(range(2, 13))
    elif isinstance(outer_period, int):
        periods = [outer_period]
    else:
        periods = [int(x) for x in outer_period]

    deadline = (time.monotonic() + timeout) if timeout else None
    results: list[tuple[float, str, str, str]] = []
    for shape in _shape_list(shapes):
        # a 26-cell grid has no dropped letter, so the drop sweep collapses to one pass
        drops = ["J"] if shape[0] * shape[1] == 26 else _drop_list(drop_letter)
        for drop in drops:
            if deadline and time.monotonic() > deadline:
                break
            for _label, grid in _grid_iter(squares, shape, drop):
                if deadline and time.monotonic() > deadline:
                    break
                for p in periods:
                    shifts, pt, score = recover_outer_key_over_playfair(
                        letters,
                        grid,
                        outer_alphabet=outer_alphabet,
                        outer_period=p,
                        drop_letter=drop,
                        shape=shape,
                        objective_fn=obj,
                        use_double=use_double,
                        rng=rng,
                    )
                    if pt:
                        key = "".join(chr(65 + (s % 26)) for s in shifts)
                        results.append((score, grid, key, pt))

    results.sort(key=lambda r: r[0], reverse=True)
    seen: set[str] = set()
    out: list[tuple[str, str, str, float]] = []
    for score, grid, key, pt in results:
        if pt in seen:
            continue
        seen.add(pt)
        out.append((grid, key, pt, score))
        if len(out) >= top:
            break
    return out
