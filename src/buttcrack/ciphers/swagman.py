"""Swagman cipher.

A periodic transposition (ACA member BUNYIP, *The Cryptogram* Sep-Oct 1977)
built on an ``n x n`` **key square** that is a Latin square of the digits
``1..n`` -- each digit appears exactly once in every row and every column
(a sudoku-like constraint).

Algorithm (ACA *The ACA and You*, ``Swagman.pdf``):

* Pick a square size ``n`` (ACA: 4-8). Fill an ``n x n`` grid with ``1..n`` so
  no digit repeats in any row or column.
* Write the plaintext horizontally into a rectangle of exactly ``n`` rows,
  padding with nulls so the rectangle is complete (width
  ``W = ceil(L / n)`` columns).
* Process the rectangle in consecutive ``n``-column blocks, all blocks sharing
  the **same** key square. Within each block column ``c`` (local index
  ``c % n``), reorder the column's ``n`` letters vertically by the key square:
  ciphertext row ``r`` receives the plaintext letter from the row that holds
  key-digit ``r + 1`` in that key-square column.
* Read the ciphertext off **vertically**, column by column.

Encrypt and decrypt differ (the per-column permutation is inverted on decode).

KEY FORMAT
----------
The ``n`` rows of the Latin square, each a permutation of ``1..n``. Accept
either one block of ``n*n`` digits (``"3214515324245315341241253"``) or the
rows separated by ``/``, ``,``, whitespace or newlines
(``"32145/15324/24531/53412/41253"``). ``n`` is inferred from the digit count
(a perfect square) and the square is validated as a Latin square.

Reference: American Cryptogram Association, *The ACA and You*, ``Swagman.pdf``
(cryptogram.org/downloads/aca.info/ciphers/Swagman.pdf); CryptoCrack user
guide, Swagman page.
"""

from __future__ import annotations

import math
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

PAD = "X"


def _parse_key(key: str) -> list[list[int]]:
    """Parse the key into an ``n x n`` Latin square of ``1..n``.

    Accepts either a flat block of ``n*n`` digits or the rows separated by
    ``/ , ;`` whitespace or newlines.
    """
    s = str(key).strip()
    if not s:
        raise ValueError("swagman key is empty")

    sep = any(ch in s for ch in "/,;\n\t ")
    if sep:
        parts = [p for p in s.replace(",", " ").replace("/", " ").replace(";", " ").split() if p]
        if not all(p.isdigit() for p in parts):
            raise ValueError("swagman key rows must be digit strings")
        rows = [[int(ch) for ch in p] for p in parts]
        n = len(rows)
        if any(len(r) != n for r in rows):
            raise ValueError(f"swagman key must be a square ({n} rows of {n} digits)")
    else:
        if not s.isdigit():
            raise ValueError("swagman key must be digits")
        n = math.isqrt(len(s))
        if n * n != len(s):
            raise ValueError(f"swagman key has {len(s)} digits, not a perfect square")
        rows = [[int(ch) for ch in s[i * n : (i + 1) * n]] for i in range(n)]

    _validate_latin(rows, n)
    return rows


def _validate_latin(rows: list[list[int]], n: int) -> None:
    want = set(range(1, n + 1))
    for r, row in enumerate(rows):
        if set(row) != want:
            raise ValueError(f"swagman key row {r} is not a permutation of 1..{n}")
    for c in range(n):
        if {rows[r][c] for r in range(n)} != want:
            raise ValueError(f"swagman key column {c} is not a permutation of 1..{n}")


def _col_perms(square: list[list[int]]) -> list[list[int]]:
    """For each key column, the source plaintext row for each ciphertext row.

    ``perm[c][r]`` is the plaintext row holding key-digit ``r + 1`` in key
    column ``c`` (so ciphertext row ``r`` <- plaintext row ``perm[c][r]``).
    """
    n = len(square)
    perms: list[list[int]] = []
    for c in range(n):
        col = [square[r][c] for r in range(n)]
        src = [0] * n
        for r in range(n):
            src[r] = col.index(r + 1)
        perms.append(src)
    return perms


def _encode_letters(letters: str, square: list[list[int]]) -> str:
    n = len(square)
    width = max(1, math.ceil(len(letters) / n)) if letters else 0
    padded = letters.ljust(n * width, PAD)
    grid = [list(padded[r * width : (r + 1) * width]) for r in range(n)]
    perms = _col_perms(square)
    ct_cols: list[str] = []
    for c in range(width):
        src = perms[c % n]
        ct_cols.append("".join(grid[src[r]][c] for r in range(n)))
    return "".join(ct_cols)


def _decode_with_perms(cipher: str, perms: list[list[int]], n: int) -> str:
    """Decode given per-column source permutations (``perms[c % n][r]``)."""
    width = len(cipher) // n
    grid = [["" for _ in range(width)] for _ in range(n)]
    for c in range(width):
        src = perms[c % n]
        col = cipher[c * n : (c + 1) * n]
        for r in range(n):
            grid[src[r]][c] = col[r]
    return "".join("".join(grid[r]) for r in range(n))


def _decode_letters(cipher: str, square: list[list[int]]) -> str:
    n = len(square)
    if len(cipher) % n != 0:
        raise ValueError(f"ciphertext length {len(cipher)} is not a multiple of {n}")
    return _decode_with_perms(cipher, _col_perms(square), n)


def _square_from_perms(perms: list[list[int]], n: int) -> list[list[int]] | None:
    """Reconstruct the key square from per-column source perms, or ``None``.

    ``square[perms[c][r]][c] = r + 1``. Returns the square only if it is a valid
    Latin square (every row/column a permutation of ``1..n``).
    """
    square = [[0] * n for _ in range(n)]
    for c in range(n):
        src = perms[c]
        if sorted(src) != list(range(n)):
            return None
        for r in range(n):
            square[src[r]][c] = r + 1
    want = set(range(1, n + 1))
    for row in square:
        if set(row) != want:
            return None
    for c in range(n):
        if {square[r][c] for r in range(n)} != want:
            return None
    return square


def _square_key_str(square: list[list[int]]) -> str:
    return "/".join("".join(str(d) for d in row) for row in square)


def _perms_key_str(perms: list[list[int]], n: int) -> str:
    sq = _square_from_perms(perms, n)
    if sq is not None:
        return _square_key_str(sq)
    # Fall back to reporting the per-column source order (not a Latin square).
    return "cols:" + "|".join(",".join(str(x) for x in p) for p in perms)


class Swagman(Cipher):
    name = "swagman"
    description = "Periodic transposition over an n x n Latin-square key (digits 1..n)."
    key_format = "n x n Latin square of 1..n, rows separated by / (or one flat digit block)"
    key_example = "32145/15324/24531/53412/41253"
    complexity = 4

    # Transposition only reorders letters, so word spacing cannot be preserved;
    # encode/decode operate on a clean uppercase letter stream.
    def encode(self, text: str, key: str) -> str:
        return _encode_letters(only_letters(text), _parse_key(key))

    def decode(self, text: str, key: str) -> str:
        return _decode_letters(only_letters(text), _parse_key(key))

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
        """Best-effort keyless crack via random-restart hill-climbing.

        The same key square applies to every block, so the whole message is
        decoded by ``n`` per-column source permutations (one per key-square
        column). For each tractable size ``n`` (a divisor of the ciphertext
        length, restricted to the ACA range 4-8) we hill-climb those ``n``
        permutations independently with pairwise swaps and random restarts,
        scoring with the n-gram fitness. The recovered square is reported when
        the winning permutations happen to form a Latin square (they normally
        do); otherwise the per-column order is reported. Honors ``timeout``.
        """
        import random

        letters = only_letters(text)
        if len(letters) < 8:
            return []
        rng = rng or random.Random(0x5A6A1A4)

        sizes = (
            [int(opts["n"])] if opts.get("n") else [n for n in range(4, 9) if len(letters) % n == 0]
        )
        restarts = int(opts.get("restarts", 80))
        deadline = (time.monotonic() + timeout) if timeout else None

        best: dict[str, Candidate] = {}

        def consider(perms: list[list[int]], n: int) -> float:
            plain = _decode_with_perms(letters, perms, n)
            sc = scorer.score(plain)
            human = _perms_key_str(perms, n)
            if human not in best:
                best[human] = Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=human,
                    score=sc,
                    confidence=scorer.confidence(plain),
                    meta={"n": n},
                )
            return sc

        for n in sizes:
            if n > len(letters):
                continue
            for _ in range(restarts):
                if deadline and time.monotonic() > deadline:
                    break
                perms = [list(range(n)) for _ in range(n)]
                for p in perms:
                    rng.shuffle(p)
                cur = consider(perms, n)
                improved = True
                while improved:
                    if deadline and time.monotonic() > deadline:
                        break
                    improved = False
                    for col in range(n):
                        for i in range(n):
                            for j in range(i + 1, n):
                                perms[col][i], perms[col][j] = perms[col][j], perms[col][i]
                                sc = consider(perms, n)
                                if sc > cur:
                                    cur, improved = sc, True
                                else:
                                    perms[col][i], perms[col][j] = perms[col][j], perms[col][i]

        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ranked[:top]
