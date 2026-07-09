"""Rectangular Playfair: digraphic substitution over an ``R x C`` keyed grid.

Generalises the classic 5x5 :class:`~buttcrack.ciphers.playfair.Playfair` to any
rectangle whose cell count is 25 or 26. Two regimes:

* ``R*C == 25`` — drops one letter (``J``->``I`` by default), exactly like standard
  Playfair; the 5x5 case is byte-for-byte identical to :class:`Playfair`.
* ``R*C == 26`` — uses ALL 26 letters, so no letter is merged and the ciphertext can
  contain every letter (e.g. a ``2x13`` grid). This is the variant to reach for when a
  message's histogram shows all 26 letters yet the map is digraphic (a 5x5 square can
  never emit its dropped letter, so a full-alphabet output rules the 5x5 square out).

The three Playfair rules generalise to the grid:

* same row     -> shift the column coordinate by ``+/-1 (mod C)``,
* same column  -> shift the row coordinate by ``+/-1 (mod R)``,
* rectangle    -> swap the two column coordinates.

The shape rides on the key as an optional ``"/RxC"`` suffix, e.g. ``"NEEDLE/2x13"``;
a bare keyword defaults to :data:`DEFAULT_SHAPE`. Cracking hill-climbs the grid (swap
letters / rows / columns) against the quadgram score, like Playfair — so, like Playfair,
it needs a few hundred letters to converge; short digraphic messages want the decoupled
``sub_playfair`` recovery instead.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .playfair import _prepare as _prepare5
from .squares import PolybiusSquare

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_SHAPE = (2, 13)


def parse_key(key: str) -> tuple[str, int, int]:
    """Split ``"KEYWORD/RxC"`` into ``(keyword, rows, cols)``; bare keyword -> default shape."""
    keyword, _, shape = str(key).partition("/")
    if not shape:
        return keyword, DEFAULT_SHAPE[0], DEFAULT_SHAPE[1]
    r, _, c = shape.lower().partition("x")
    return keyword, int(r), int(c)


def grid_letters(keyword: str, rows: int, cols: int, drop: str = "J") -> str:
    """The ``rows*cols``-cell grid string (row-major) built from ``keyword``.

    25 cells reuse :class:`PolybiusSquare` (so a 5x5 grid matches :class:`Playfair`
    exactly); 26 cells use the full alphabet with no drop.
    """
    n = rows * cols
    if n == 25:
        return "".join(PolybiusSquare(keyword, size=5).grid)
    if n == 26:
        kw: list[str] = []
        for ch in only_letters(keyword):
            if ch not in kw:
                kw.append(ch)
        return "".join(kw) + "".join(c for c in _STD if c not in kw)
    raise ValueError(f"rectangular Playfair needs a 25- or 26-cell grid; {rows}x{cols}={n}")


def _prepare(letters: str, cells: int) -> list[tuple[str, str]]:
    """Encode-prepare: split doubles and pad a lone final letter (drop J->I only for 25 cells)."""
    if cells == 25:
        return _prepare5(letters)  # identical to Playfair for the classic square
    s = letters  # 26 cells: no letter is merged
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        a = s[i]
        b = s[i + 1] if i + 1 < len(s) else ""
        if b == "" or a == b:
            pairs.append((a, "X" if a != "X" else "Q"))
            i += 1
        else:
            pairs.append((a, b))
            i += 2
    return pairs


def _pairs_from_text(letters: str, cells: int) -> list[tuple[str, str]]:
    s = letters.replace("J", "I") if cells == 25 else letters
    if len(s) % 2:
        s = s[:-1]
    return [(s[i], s[i + 1]) for i in range(0, len(s), 2)]


def _transform(pairs: list[tuple[str, str]], grid: str, rows: int, cols: int, direction: int) -> str:
    pos = {c: i for i, c in enumerate(grid)}
    out: list[str] = []
    for a, b in pairs:
        ra, ca = divmod(pos[a], cols)
        rb, cb = divmod(pos[b], cols)
        if ra == rb:
            out.append(grid[ra * cols + (ca + direction) % cols])
            out.append(grid[rb * cols + (cb + direction) % cols])
        elif ca == cb:
            out.append(grid[((ra + direction) % rows) * cols + ca])
            out.append(grid[((rb + direction) % rows) * cols + cb])
        else:
            out.append(grid[ra * cols + cb])
            out.append(grid[rb * cols + ca])
    return "".join(out)


def _mutate(grid: list[str], rows: int, cols: int, rng: random.Random) -> list[str]:
    new = grid[:]
    r = rng.random()
    if r < 0.8:  # swap two cells
        i, j = rng.randrange(len(new)), rng.randrange(len(new))
        new[i], new[j] = new[j], new[i]
    elif r < 0.9 and rows > 1:  # swap two rows
        a, b = rng.randrange(rows), rng.randrange(rows)
        for c in range(cols):
            new[a * cols + c], new[b * cols + c] = new[b * cols + c], new[a * cols + c]
    elif cols > 1:  # swap two columns
        a, b = rng.randrange(cols), rng.randrange(cols)
        for rr in range(rows):
            new[rr * cols + a], new[rr * cols + b] = new[rr * cols + b], new[rr * cols + a]
    return new


class RectangularPlayfair(Cipher):
    name = "rectangular_playfair"
    aliases = ("rectplayfair", "playfair_rect")
    description = "Digraphic substitution over an RxC keyed grid (25- or 26-cell; default 2x13)."
    key_format = "keyword with optional /RxC shape suffix (e.g. NEEDLE/2x13); 25 or 26 cells"
    key_example = "NEEDLE/2x13"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        keyword, rows, cols = parse_key(key)
        grid = grid_letters(keyword, rows, cols)
        return _transform(_prepare(only_letters(text), rows * cols), grid, rows, cols, +1)

    def decode(self, text: str, key: str) -> str:
        keyword, rows, cols = parse_key(key)
        grid = grid_letters(keyword, rows, cols)
        return _transform(_pairs_from_text(only_letters(text), rows * cols), grid, rows, cols, -1)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 20:
            return []
        rows = int(opts.get("rows", DEFAULT_SHAPE[0]))
        cols = int(opts.get("cols", DEFAULT_SHAPE[1]))
        cells = rows * cols
        base = list(_STD) if cells == 26 else list("ABCDEFGHIKLMNOPQRSTUVWXYZ")
        pairs = _pairs_from_text(letters, cells)
        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 3))
        temp0 = float(opts.get("temp", 12.0))
        step = float(opts.get("temp_step", 0.3))
        iters = int(opts.get("iters", 3000))
        deadline = (time.monotonic() + timeout) if timeout else None

        def score_of(grid_list: list[str]) -> float:
            return scorer.score(_transform(pairs, "".join(grid_list), rows, cols, -1))

        best_grid = base[:]
        best_score = float("-inf")
        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = base[:]
            rng.shuffle(parent)
            cur = score_of(parent)
            temp = temp0
            while temp > 0:
                if deadline and time.monotonic() > deadline:
                    break
                for _ in range(iters):
                    child = _mutate(parent, rows, cols, rng)
                    s = score_of(child)
                    delta = s - cur
                    if delta > 0 or rng.random() < math.exp(delta / temp):
                        parent, cur = child, s
                        if s > best_score:
                            best_grid, best_score = child[:], s
                temp -= step

        plain = _transform(pairs, "".join(best_grid), rows, cols, -1)
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=f"{''.join(best_grid)}/{rows}x{cols}",
                score=best_score,
                confidence=scorer.confidence(plain),
                meta={"grid": "".join(best_grid), "rows": rows, "cols": cols},
            )
        ]
