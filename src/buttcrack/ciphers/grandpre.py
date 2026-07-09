"""Grandpre cipher: a homophonic substitution onto a stream of two-digit codes.

The key is a SQUARE GRID of words. Each of the ``N`` rows is an ``N``-letter
word, and the ``N`` letters of the FIRST COLUMN also spell a word; collectively
every letter of the alphabet must appear somewhere in the grid (so common
letters land in several cells). Rows and columns are numbered ``1..N`` (for the
10x10 form column/row 10 is written as the digit ``0``).

ENCRYPT (one-to-many / homophonic): each plaintext letter is replaced by the
two-digit code ``(row)(column)`` of SOME cell that holds it; when a letter
appears in several cells one is chosen (here pseudo-randomly when an ``rng`` is
supplied, else the first cell found, deterministically). The output is a stream
of two-digit numbers.

DECRYPT (deterministic): read the digits in pairs ``(row, column)`` and emit the
letter sitting at that grid cell.

KEY FORMAT
    The ``N`` row words, separated by ``/`` or whitespace, e.g.::

        "ADJACENT/NAZARENE/AGGRIEVE/REQUITED/CHATEAUX/HALFBACK/IMPUNITY/CROSSBOW"

    ``N`` may be 8, 9, or 10 (the grid must be square: every row word must have
    length ``N`` and there must be ``N`` rows). The grid itself IS the key;
    there is no separate numeric key.

Encryption is not the inverse of decryption (encryption is one-to-many), so the
cipher is not reciprocal.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher


def _split_rows(key: str) -> list[str]:
    """Parse the row words from the key (``/``- or whitespace-separated)."""
    raw = str(key).upper().replace("/", " ").replace(",", " ")
    rows = ["".join(ch for ch in word if "A" <= ch <= "Z") for word in raw.split()]
    return [r for r in rows if r]


def _label_to_index(digit: int, size: int) -> int:
    """Map a label digit (1..9, or 0 meaning 10 on a 10x10 grid) to 0-based."""
    if digit == 0:
        return size - 1  # only valid when size == 10 (label "0" == 10)
    return digit - 1


def _index_to_label(idx: int, size: int) -> str:
    """Map a 0-based row/col index to its label digit string ('1'..'9' or '0')."""
    label = idx + 1
    return "0" if label == 10 else str(label)


class _Grid:
    """A parsed Grandpre grid: encode (letter->codes) and decode (code->letter)."""

    def __init__(self, key: str):
        rows = _split_rows(key)
        if not rows:
            raise ValueError("grandpre key must contain at least one row word")
        size = len(rows)
        if size not in (8, 9, 10):
            raise ValueError(f"grandpre grid must be 8x8, 9x9 or 10x10, got {size} rows")
        for word in rows:
            if len(word) != size:
                raise ValueError(f"grandpre row {word!r} has length {len(word)}, expected {size}")
        self.size = size
        self.rows = rows
        self.codes: dict[str, list[str]] = {}
        for r, word in enumerate(rows):
            for c, ch in enumerate(word):
                code = _index_to_label(r, size) + _index_to_label(c, size)
                self.codes.setdefault(ch, []).append(code)

    def decode_code(self, code: str) -> str:
        r = _label_to_index(int(code[0]), self.size)
        c = _label_to_index(int(code[1]), self.size)
        return self.rows[r][c]


class Grandpre(Cipher):
    name = "grandpre"
    aliases = ("grandpré",)
    description = "Homophonic substitution onto two-digit codes from a grid of words."
    key_format = "N row words ('/'-separated), each N letters, N in {8,9,10}; all A-Z covered"
    key_example = "ADJACENT/NAZARENE/AGGRIEVE/REQUITED/CHATEAUX/HALFBACK/IMPUNITY/CROSSBOW"
    complexity = 4

    def encode(self, text: str, key: str, *, rng: random.Random | None = None) -> str:
        grid = _Grid(key)
        out: list[str] = []
        for ch in only_letters(text):
            options = grid.codes.get(ch)
            if not options:
                raise ValueError(f"letter {ch!r} is not present in the grandpre grid")
            code = rng.choice(options) if rng is not None else options[0]
            out.append(code)
        return " ".join(out)

    def decode(self, text: str, key: str) -> str:
        grid = _Grid(key)
        digits = "".join(c for c in str(text) if c.isdigit())
        out: list[str] = []
        for i in range(0, len(digits) - 1, 2):
            out.append(grid.decode_code(digits[i : i + 2]))
        return "".join(out)

    def crack(
        self,
        text: str,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Not implemented as a keyless solver.

        Recovering a Grandpre grid is a crossword-reconstruction problem: each
        row and the first column must spell real words while every code observed
        constrains which letter sits at a given cell. Solving it requires a large
        dictionary of N-letter words plus a crib, which this package does not
        carry, so we return no candidates rather than a misleading guess.
        """
        _ = (text, scorer, top, rng, timeout, opts)
        deadline = None if timeout is None else time.monotonic() + timeout
        _ = deadline
        return []
