"""Collon cipher — seriated row/column coordinates written as LETTERS, not digits.

The only genuinely missing member of the 5x5 grid family. Like Bifid it takes each
letter's row and column in a keyed square and writes all the rows of a group before
all the columns; unlike Bifid it never re-reads the coordinate stream as new pairs.
The coordinates are emitted as letters — the letter that begins the row and the
letter that ends the column — so the ciphertext is twice the plaintext length and
looks like ordinary text rather than digits.

ALGORITHM
---------
Build a keyed mixed 5x5 square (J merged into I). Split the plaintext into groups of
``N``. For each group, emit the row label of every letter, then the column label of
every letter.

Labels are taken from the square itself: the row label of row ``r`` is the letter at
``(r, 0)``, the column label of column ``c`` is the letter at ``(4, c)``. On the
standard square, ``D`` (row 0, column 3) has row label ``A`` and column label ``Y``,
so ``DC`` at N=2 encrypts to ``AAYX``.

Decryption splits each ``2N``-character block into ``N`` row labels and ``N`` column
labels, pairs them positionally, and reads the intersection. Any letter identifies
its row or column, not just the canonical label.

KEY FORMAT
----------
``KEYWORD/N`` — square keyword and group size, e.g. ``KRYPTOS/5``. An empty keyword
gives the standard A-Z (J->I) square.

Reference: dCode "Collon Cipher".
"""

from __future__ import annotations

import time


from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .squares import ALPHABET_5, PolybiusSquare


def _parse_key(key: str) -> tuple[str, int]:
    s = str(key).strip()
    head, sep, tail = s.rpartition("/")
    if not sep or not tail.strip().isdigit():
        raise ValueError("collon key must be 'KEYWORD/N' (square keyword and group size)")
    n = int(tail)
    if n < 1:
        raise ValueError("collon group size N must be >= 1")
    return head.strip(), n


def _labels(square: PolybiusSquare) -> tuple[list[str], list[str]]:
    """(row labels, column labels) — first letter of each row, last of each column."""
    rows = [square.at(r, 0) for r in range(5)]
    cols = [square.at(4, c) for c in range(5)]
    return rows, cols


def _encode_letters(letters: str, square: PolybiusSquare, n: int) -> str:
    rows, cols = _labels(square)
    out: list[str] = []
    for start in range(0, len(letters), n):
        group = letters[start : start + n]
        coords = [square.rc(ch) for ch in group]
        out.extend(rows[r] for r, _ in coords)
        out.extend(cols[c] for _, c in coords)
    return "".join(out)


def _decode_letters(cipher: str, square: PolybiusSquare, n: int) -> str:
    out: list[str] = []
    pos = 0
    while pos < len(cipher):
        # A short final group is emitted as 2m characters for its m letters, so the
        # block size follows from what is left rather than being fixed at 2N.
        m = min(n, (len(cipher) - pos) // 2)
        if m < 1:
            break
        block = cipher[pos : pos + 2 * m]
        pos += 2 * m
        for row_label, col_label in zip(block[:m], block[m:], strict=True):
            row = square.rc(row_label)[0]
            col = square.rc(col_label)[1]
            out.append(square.at(row, col))
    return "".join(out)


class Collon(Cipher):
    name = "collon"
    aliases = ("collon-cipher",)
    description = "Seriated 5x5 row/column coordinates written as letters; ciphertext is 2x."
    key_format = "square keyword and group size, 'KEYWORD/N'"
    key_example = "KRYPTOS/5"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        keyword, n = _parse_key(key)
        square = PolybiusSquare(keyword)
        return _encode_letters(square.prepare(text), square, n)

    def decode(self, text: str, key: str) -> str:
        keyword, n = _parse_key(key)
        square = PolybiusSquare(keyword)
        return _decode_letters(square.prepare(text), square, n)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Solve outright — the ciphertext only ever uses TEN distinct letters.

        Annealing the 25-cell square is the obvious attack and it is the wrong one: it
        ignores the constraint that gives the cipher away. Only five letters can appear
        in a row position (the five row labels) and five in a column position, so the
        ciphertext alphabet is at most 10 and each (row label, column label) pair maps
        to exactly one plaintext letter. That is a monoalphabetic substitution over at
        most 25 pair-symbols, which the substitution solver settles directly.

        The group size falls out of the same constraint: at the true ``N`` the row half
        and column half of each block each draw on only five letters, and at a wrong
        ``N`` the halves mix. Every ``N`` that survives that filter is solved and ranked.
        """
        from .substitution import Substitution

        letters = PolybiusSquare("").prepare(only_letters(text))
        if len(letters) < 80:
            return []
        deadline = (time.monotonic() + timeout) if timeout else None
        max_n = min(int(opts.get("max_group", 12)), len(letters) // 8)
        groups = opts.get("groups") or list(range(1, max_n + 1))

        candidates: list[Candidate] = []
        for n in groups:
            if deadline and time.monotonic() > deadline:
                break
            pairs = _split_blocks(letters, n)
            if pairs is None:
                continue
            row_set = {a for a, _ in pairs}
            col_set = {b for _, b in pairs}
            # Five rows and five columns is the whole grid; more means this N slices
            # across the row/column boundary and is not the group size.
            if len(row_set) > 5 or len(col_set) > 5:
                continue

            row_labels = sorted(row_set)
            col_labels = sorted(col_set)
            index: dict[tuple[str, str], int] = {}
            order: list[tuple[str, str]] = []
            for p in pairs:
                if p not in index:
                    index[p] = len(order)
                    order.append(p)
            proxy = "".join(ALPHABET_5[index[p]] for p in pairs)

            for cand in Substitution().crack(proxy, scorer, top=2, rng=rng, timeout=timeout):
                plain = only_letters(cand.plaintext)
                if len(plain) != len(pairs):
                    continue
                square = _rebuild_square(pairs, plain, row_labels, col_labels)
                candidates.append(
                    Candidate(
                        plaintext=plain,
                        cipher=self.name,
                        key=f"{square}/{n}" if square else f"?/{n}",
                        score=cand.score,
                        confidence=cand.confidence,
                        meta={
                            "group_size": n,
                            "square": square,
                            "row_labels": "".join(row_labels),
                            "column_labels": "".join(col_labels),
                            "method": "monoalphabetic solve over label pairs",
                        },
                    )
                )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]


def _split_blocks(letters: str, n: int) -> list[tuple[str, str]] | None:
    """Pair each block's row half with its column half; None if the text won't split."""
    pairs: list[tuple[str, str]] = []
    pos = 0
    while pos < len(letters):
        m = min(n, (len(letters) - pos) // 2)
        if m < 1:
            break
        block = letters[pos : pos + 2 * m]
        pos += 2 * m
        pairs.extend(zip(block[:m], block[m:], strict=True))
    return pairs or None


def _rebuild_square(
    pairs: list[tuple[str, str]], plain: str, row_labels: list[str], col_labels: list[str]
) -> str:
    """Reconstruct a 25-letter square that reproduces this decrypt.

    The labels identify a row or column but say nothing about its *position*, so the
    square is only recovered up to a permutation of rows and of columns. That costs
    nothing: ``decode`` finds a row by looking up its label, and a label travels with
    its row, so every consistent ordering decodes identically. Any one of them is
    therefore a working key. Cells for pairs the ciphertext never used are unknown and
    filled with the alphabet's unused letters so the key stays a valid square.
    """
    cell: dict[tuple[str, str], str] = {}
    for (row_label, col_label), letter in zip(pairs, plain, strict=True):
        cell[(row_label, col_label)] = letter

    grid: list[str] = []
    for row_label in row_labels:
        for col_label in col_labels:
            grid.append(cell.get((row_label, col_label), ""))
    if len(row_labels) < 5 or len(col_labels) < 5:
        return ""
    spare = [c for c in ALPHABET_5 if c not in set(grid)]
    filled = [c if c else spare.pop(0) for c in grid]
    return "".join(filled) if len(set(filled)) == 25 else ""
