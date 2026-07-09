"""Checkerboard cipher (ACA): digraphic substitution over a labelled 5x5 square.

A keyed 5x5 Polybius square (J merged into I, 25 letters, filled row-by-row) is
labelled on two axes: a 5-letter ROW keyword runs down the left side and a
5-letter COLUMN keyword runs across the top. Each plaintext letter is located in
the square and replaced by the DIGRAPH ``(row-label)(column-label)`` — the row
letter first. A plaintext of length L therefore becomes a ciphertext of length
2L drawn from only the ten axis letters, with row-letters in even positions and
column-letters in odd positions.

Decryption takes the ciphertext two letters at a time: the first letter selects
a row (its index in the row keyword), the second selects a column (its index in
the column keyword), and the plaintext is the square cell at that (row, col).

This is *not* the straddling checkerboard (no variable-length codes); it is the
ACA "Checkerboard" digraphic substitution.

KEY FORMAT
----------
A single ``--key`` of three slash-separated keywords::

    SQUAREKEY/ROWKEY/COLKEY

  * ``SQUAREKEY`` — keyword (or full 25-letter alphabet) for the 5x5 square,
    filled row-by-row: deduplicated keyword letters first, then the remaining
    A-Z letters in order, J->I. To reproduce a published square that uses a
    spiral/other fill, pass the square's letters read row-by-row as the keyword.
  * ``ROWKEY``  — a 5-letter word of distinct letters labelling the rows.
  * ``COLKEY``  — a 5-letter word of distinct letters labelling the columns.

Example: ``BACKUP/BRAIN/WAVES`` builds the square keyed by BACKUP, labels its
rows B,R,A,I,N and its columns W,A,V,E,S.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .squares import ALPHABET_5, PolybiusSquare


def _parse_key(key: str) -> tuple[str, str, str]:
    """Split ``SQUAREKEY/ROWKEY/COLKEY`` and validate the two 5-letter axes."""
    parts = str(key).split("/")
    if len(parts) != 3:
        raise ValueError("checkerboard key must be 'SQUAREKEY/ROWKEY/COLKEY'")
    sq_kw, row_kw, col_kw = (p.strip() for p in parts)
    row = only_letters(row_kw).replace("J", "I")
    col = only_letters(col_kw).replace("J", "I")
    for axis_name, axis in (("row", row), ("column", col)):
        if len(axis) != 5 or len(set(axis)) != 5:
            raise ValueError(f"checkerboard {axis_name} keyword must have 5 distinct letters")
    return sq_kw, row, col


class Checkerboard(Cipher):
    """ACA Checkerboard digraphic substitution over a labelled 5x5 keyed square.

    KEY FORMAT: ``SQUAREKEY/ROWKEY/COLKEY`` (slash-separated). The square is the
    standard row-by-row keyed 5x5 (J->I); ``ROWKEY``/``COLKEY`` are 5-letter
    words of distinct letters labelling the rows (left) and columns (top). Each
    plaintext letter -> digraph (row-label)(col-label); ciphertext is twice the
    plaintext length over only the ten axis letters. Not reciprocal.
    """

    name = "checkerboard"
    aliases = ("aca_checkerboard",)
    description = (
        "Digraphic substitution over a labelled 5x5 keyed square "
        "(J->I); key 'SQUAREKEY/ROWKEY/COLKEY'."
    )
    key_format = "squarekeyword/rowkey/colkey (rowkey and colkey are 5 distinct letters each)"
    key_example = "BACKUP/BRAIN/WAVES"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        sq_kw, row, col = _parse_key(key)
        sq = PolybiusSquare(sq_kw)
        out: list[str] = []
        for ch in sq.prepare(text):
            r, c = sq.rc(ch)
            out.append(row[r])
            out.append(col[c])
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        sq_kw, row, col = _parse_key(key)
        sq = PolybiusSquare(sq_kw)
        row_idx = {ch: i for i, ch in enumerate(row)}
        col_idx = {ch: i for i, ch in enumerate(col)}
        letters = only_letters(text).replace("J", "I")
        if len(letters) % 2:
            letters = letters[:-1]
        out: list[str] = []
        for i in range(0, len(letters), 2):
            a, b = letters[i], letters[i + 1]
            if a not in row_idx or b not in col_idx:
                raise ValueError(f"ciphertext digraph {a}{b} uses non-axis letters")
            out.append(sq.at(row_idx[a], col_idx[b]))
        return "".join(out)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        """Keyless best-effort: recover axis alphabets by parity, anneal the square.

        The ciphertext draws row-letters from even positions and column-letters
        from odd positions, so the two 5-letter axis alphabets fall out directly
        from position parity. What remains is a monoalphabetic substitution: each
        ciphertext digraph denotes one of the 25 square cells, and we must place
        the 25 plaintext letters into those cells. We simulated-anneal that 25-cell
        arrangement against the quadgram score of the decrypt, restarting from
        random squares to escape local optima.
        """
        letters = only_letters(text).replace("J", "I")
        if len(letters) % 2:
            letters = letters[:-1]
        if len(letters) < 60:
            return []
        # Axis alphabets by parity: 5 distinct row-letters (even), 5 col (odd).
        row_set = sorted({letters[i] for i in range(0, len(letters), 2)})
        col_set = sorted({letters[i] for i in range(1, len(letters), 2)})
        if len(row_set) > 5 or len(col_set) > 5:
            return []  # not a clean single-word-per-axis checkerboard
        row_alpha = "".join(row_set)
        col_alpha = "".join(col_set)
        row_idx = {ch: i for i, ch in enumerate(row_alpha)}
        col_idx = {ch: i for i, ch in enumerate(col_alpha)}

        # Pre-compute each digraph's flat cell index (row*5 + col) once.
        cells: list[int] = []
        for i in range(0, len(letters), 2):
            cells.append(row_idx[letters[i]] * 5 + col_idx[letters[i + 1]])

        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None
        restarts = int(opts.get("restarts", 4))
        temp0 = float(opts.get("temp", 10.0))
        step = float(opts.get("temp_step", 0.5))
        iters = int(opts.get("iters", 2000))
        base = list(ALPHABET_5)

        def decrypt_with(square: list[str]) -> str:
            return "".join(square[cell] for cell in cells)

        best_sq = base[:]
        best_score = float("-inf")
        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = base[:]
            rng.shuffle(parent)
            cur = scorer.score(decrypt_with(parent))
            temp = temp0
            while temp > 0:
                if deadline and time.monotonic() > deadline:
                    break
                for _ in range(iters):
                    i, j = rng.randrange(25), rng.randrange(25)
                    child = parent[:]
                    child[i], child[j] = child[j], child[i]
                    s = scorer.score(decrypt_with(child))
                    delta = s - cur
                    if delta > 0 or rng.random() < math.exp(delta / temp):
                        parent, cur = child, s
                        if s > best_score:
                            best_sq, best_score = child[:], s
                temp -= step

        sq_str = "".join(best_sq)
        plain = decrypt_with(best_sq)
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=f"{sq_str}/{row_alpha}/{col_alpha}",
                score=best_score,
                confidence=scorer.confidence(plain),
                meta={"square": sq_str, "row": row_alpha, "col": col_alpha},
            )
        ]
