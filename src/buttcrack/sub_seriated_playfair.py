"""Decoupled recovery of a periodic substitution laid OVER a Seriated Playfair inner.

The construction this attacks is ``CT = outer( seriated_playfair_inner( PT ) )`` where

* ``seriated_playfair_inner`` is a classic Seriated Playfair (ACA type; see
  :mod:`buttcrack.ciphers.seriated_playfair`) over a 5x5 keyed square that DROPS one
  letter (default ``J``, merged into ``I``) and seriates at a fixed ``inner_period``; and
* ``outer`` is a periodic substitution — a Vigenere / Quagmire-III shift of period
  ``outer_period`` over a keyed ``outer_alphabet`` (default the KRYPTOS keyed alphabet).

This is the seriated sibling of :mod:`buttcrack.sub_playfair` (regular Playfair inner) and
:mod:`buttcrack.sub_fractionation` (bifid inner). It matters for layered digraphic puzzles
because Seriated Playfair is the one digraphic-class inner that is BOTH odd-length capable
(a short final block) AND natively period-``N`` — the regular-Playfair cracker is
even-length-only and cannot even represent such a ciphertext.

Why the decoupling works (same two structural constraints, adapted to seriation):

1. **Drop-letter-free.** A 25-cell square can never emit its dropped letter, so the residual
   (the Seriated-Playfair ciphertext, recovered by stripping the outer substitution) contains
   no ``J`` (or whatever letter the square omits). For a period-``p`` outer shift each key
   position acts on its own coset, so a candidate shift ``s`` for column ``j`` is admissible
   only if stripping it leaves that coset free of the drop letter — pruning each position to a
   small candidate set (the same lever as :mod:`sub_playfair`/:mod:`sub_fractionation`).
2. **No doubled digraph.** Seriated Playfair enciphers *vertical* pairs of a two-row block, so
   its digraphs live at ciphertext positions ``(blockstart + j, blockstart + width + j)`` — NOT
   at adjacent indices. Playfair maps a distinct-letter digraph to a distinct-letter digraph
   (and X-insertion prevents equal input pairs), so the residual never has an equal pair at any
   such vertical boundary. For a digraph whose two positions fall in outer-cosets ``(ja, jb)``
   this bans exactly one value of the relative shift ``s_jb - s_ja``. NOTE: for the full
   blocks the two positions share a coset (``width`` is a multiple of the outer period in the
   aligned case), so the ban is on relative shift 0 — a shift-independent *validity* test
   rather than a pruner; the short final block, where ``width`` is not a multiple of the
   period, contributes genuine cross-coset bans.

Within the admissible shifts the outer key is recovered by constrained coordinate descent (or
an exhaustive scan when the admissible product is small), scoring the FULL de-seriated decode
with a pluggable objective and a penalty for any residual doubled digraph. As with the
siblings the discriminator is sharp: with the CORRECT square the recovered decode reads as
language (or, for a route/list payload under the payload-agnostic objectives, structures);
with any wrong square it plateaus in the noise.

Public API
----------
``sub_encode`` / ``sub_decode``                      periodic shift over a keyed alphabet (re-exported)
``seriated_playfair_encode`` / ``_decode``            Seriated-Playfair inner given a 25-cell square
``encrypt_sub_over_seriated_playfair``                plant CT = outer(seriated_playfair(PT))
``recover_outer_key_over_seriated_playfair``          key-given-structure recovery (one square)
``crack_sub_over_seriated_playfair``                  driver over candidate squares + periods/drops
"""

from __future__ import annotations

import random as _random
import time
from collections.abc import Callable, Iterable, Sequence
from itertools import product

from .ciphers.seriated_playfair import _decode_to_prepared, _encode_prepared, _seriate
from .ciphers.squares import PolybiusSquare
from .scoring import index_of_coincidence  # noqa: F401  (handy for callers)
from .sub_fractionation import make_objective, resolve_alphabet, sub_decode, sub_encode
from .sub_playfair import resolve_square, square_alphabet5
from .text import only_letters

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"

__all__ = [
    "square_alphabet5",
    "resolve_square",
    "seriated_playfair_encode",
    "seriated_playfair_decode",
    "sub_encode",
    "sub_decode",
    "encrypt_sub_over_seriated_playfair",
    "recover_outer_key_over_seriated_playfair",
    "crack_sub_over_seriated_playfair",
    "seriated_digraph_pairs",
]


# --------------------------------------------------------------------------- #
# Seriated-Playfair primitives (thin wrappers over the cipher module)
# --------------------------------------------------------------------------- #
def seriated_playfair_encode(pt: str, square25: str, period: int) -> str:
    """Seriated-Playfair encode ``pt`` over a 25-cell ``square25`` at seriation ``period``.

    Mirrors :class:`buttcrack.ciphers.seriated_playfair.SeriatedPlayfair.encode`: seriate
    (inserting ``X`` nulls to break vertical doubles, padding a short final block), then
    encipher each vertical pair and read off two rows per block. Output length is even and may
    exceed ``len(pt)`` when nulls are inserted.
    """
    prepared = _seriate(only_letters(pt), period)
    return _encode_prepared(prepared, square25, period)


def seriated_playfair_decode(residual: str, square25: str, period: int) -> str:
    """De-Seriated-Playfair ``residual`` over ``square25`` at ``period`` (nulls retained).

    Returns the seriated plaintext stream in natural reading order (top row then bottom row of
    each block are consecutive plaintext), with inserted/padding ``X`` nulls left in place. A
    trailing odd letter (if ``residual`` is odd length) is ignored, matching the cipher.
    """
    r = only_letters(residual)
    if len(r) % 2:
        r = r[:-1]
    return _decode_to_prepared(r, square25, period)


def encrypt_sub_over_seriated_playfair(
    pt: str,
    square: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    inner_period: int = 7,
    outer_shifts: Sequence[int],
    drop_letter: str = "J",
) -> str:
    """Plant ``CT = periodic_sub( seriated_playfair( PT ) )`` — the structure this attacks."""
    alpha = resolve_alphabet(outer_alphabet)
    inner = seriated_playfair_encode(pt, resolve_square(square, drop_letter), inner_period)
    return sub_encode(inner, alpha, outer_shifts)


# --------------------------------------------------------------------------- #
# Digraph geometry & structural constraints
# --------------------------------------------------------------------------- #
def seriated_digraph_pairs(n: int, period: int) -> list[tuple[int, int]]:
    """The ``(top_pos, bottom_pos)`` index pairs of every vertical digraph.

    Replicates the block/width walk of the cipher's take-off so the pairing matches decode
    exactly: full blocks have ``width == period`` (top ``period`` then bottom ``period``); a
    short final block has ``width = (remaining)//2``.
    """
    m = n - (n % 2)
    pairs: list[tuple[int, int]] = []
    i = 0
    while i < m:
        width = period if (m - i) >= 2 * period else (m - i) // 2
        for j in range(width):
            pairs.append((i + j, i + width + j))
        i += 2 * width
    return pairs


def _admissible_shifts(c_idx: list[int], n: int, p: int, drop_pos: int) -> list[list[int]]:
    """Per-column shifts leaving the residual coset free of the drop letter (constraint 1)."""
    allowed: list[list[int]] = []
    for j in range(p):
        col_vals = {c_idx[i] for i in range(j, n, p)}
        banned = {(v - drop_pos) % 26 for v in col_vals}
        allowed.append([s for s in range(26) if s not in banned] or list(range(26)))
    return allowed


def _double_banned(
    c_idx: list[int], n: int, p: int, inner_period: int
) -> dict[tuple[int, int], set[int]]:
    """Banned relative shifts ``s_jb - s_ja`` from the no-doubled-digraph rule (constraint 2).

    For each vertical digraph at positions ``(a, b)`` in outer-cosets ``(ja, jb)`` the residual
    letters must differ: ``(c_a - s_ja) != (c_b - s_jb) (mod 26)`` ⇒ ban
    ``s_jb - s_ja == (c_b - c_a) mod 26``. When ``ja == jb`` (full aligned blocks) this bans
    the value 0 iff ``c_a == c_b`` — a shift-independent validity flag stored under ``(j, j)``.
    """
    banned: dict[tuple[int, int], set[int]] = {}
    for a, b in seriated_digraph_pairs(n, inner_period):
        ja, jb = a % p, b % p
        banned.setdefault((ja, jb), set()).add((c_idx[b] - c_idx[a]) % 26)
    return banned


# --------------------------------------------------------------------------- #
# Key-given-structure recovery (one candidate square)
# --------------------------------------------------------------------------- #
def recover_outer_key_over_seriated_playfair(
    ciphertext: str,
    square25: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    outer_period: int,
    inner_period: int = 7,
    drop_letter: str = "J",
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
    """Recover the outer period-``p`` shift key GIVEN a candidate inner Seriated-Playfair square.

    Prunes each column's shifts by the drop-letter rule, penalises residual doubled digraphs,
    and finds the best combination by objective score of the FULL de-seriated decode —
    exhaustively when the admissible product is small, else by constrained coordinate descent
    with random restarts. Returns ``(shifts, plaintext, score)``.
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
    dbanned = _double_banned(c_idx, n, p, int(inner_period)) if use_double else {}

    def decode_for(shifts: Sequence[int]) -> str:
        residual = "".join(A[(c_idx[i] - shifts[i % p]) % 26] for i in range(n))
        return _decode_to_prepared(residual, square25, int(inner_period))

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
def _square_iter(squares, drop_letter: str) -> list[tuple[str, str]]:
    if isinstance(squares, str) and squares.lower() in ("dictionary", "dict", "keywords"):
        from .ciphers._quagmire_solver import BUILTIN_KEYWORDS

        words: Iterable[str] = BUILTIN_KEYWORDS
        return [(w, resolve_square(w, drop_letter)) for w in words]
    return [(str(item), resolve_square(str(item), drop_letter)) for item in squares]


def _drop_list(drop_letter) -> list[str]:
    if drop_letter is None:
        return ["J"]
    if isinstance(drop_letter, str):
        s = drop_letter.strip()
        if s.lower() in ("sweep", "all", "*"):
            return list(_STD)
        letters = [c for c in s.upper() if "A" <= c <= "Z"]
        return letters or ["J"]
    return [str(x).upper()[:1] for x in drop_letter]


def _shifts_to_keystr(shifts: Sequence[int]) -> str:
    return "".join(chr(65 + (s % 26)) for s in shifts)


def crack_sub_over_seriated_playfair(
    ciphertext: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    inner_period: int | Iterable[int] = 7,
    outer_period: int | Iterable[int] | None = 7,
    squares="dictionary",
    objective: str = "fitness",
    drop_letter: str | None = None,
    languages: Sequence[str] = ("english",),
    top: int = 5,
    timeout: float | None = None,
    rng=None,
    brute_cap: int = 200000,
) -> list[tuple[str, str, str, float]]:
    """Crack ``CT = periodic_sub( seriated_playfair( PT ) )`` by scanning candidate squares.

    For every candidate ``drop_letter``, square, ``inner_period`` and ``outer_period``, recover
    the outer key by :func:`recover_outer_key_over_seriated_playfair`, then rank all hypotheses
    by the recovered decode's ``objective`` score. Returns up to ``top`` tuples
    ``(square25, key, plaintext, score)`` best-first. ``inner_period``/``outer_period`` may be
    an int or an iterable of ints (``outer_period=None`` sweeps 2..12).
    """
    drops = _drop_list(drop_letter)
    letters = only_letters(ciphertext)
    obj = make_objective(objective, languages=languages)

    def _as_periods(v, default_sweep):
        if v is None:
            return list(default_sweep)
        if isinstance(v, int):
            return [v]
        return [int(x) for x in v]

    outer_periods = _as_periods(outer_period, range(2, 13))
    inner_periods = _as_periods(inner_period, range(3, 13))
    deadline = (time.monotonic() + timeout) if timeout else None

    results: list[tuple[float, str, str, str]] = []
    for drop in drops:
        if deadline and time.monotonic() > deadline:
            break
        for _label, sq25 in _square_iter(squares, drop):
            if deadline and time.monotonic() > deadline:
                break
            for ip in inner_periods:
                for op in outer_periods:
                    if deadline and time.monotonic() > deadline:
                        break
                    shifts, pt, score = recover_outer_key_over_seriated_playfair(
                        letters,
                        sq25,
                        outer_alphabet=outer_alphabet,
                        inner_period=ip,
                        outer_period=op,
                        drop_letter=drop,
                        objective_fn=obj,
                        rng=rng,
                        brute_cap=brute_cap,
                    )
                    if not pt:
                        continue
                    results.append((score, sq25, _shifts_to_keystr(shifts), pt))

    results.sort(key=lambda r: r[0], reverse=True)
    seen: set[str] = set()
    out: list[tuple[str, str, str, float]] = []
    for score, sq25, keystr, pt in results:
        if pt in seen:
            continue
        seen.add(pt)
        out.append((sq25, keystr, pt, score))
        if len(out) >= top:
            break
    return out
