"""Myszkowski transposition cipher.

A columnar transposition whose keyword is chosen to contain *repeated* letters.
Unlike ordinary columnar transposition (where tied keyword letters are broken
left-to-right), repeated letters here all receive the **same** column number.
Columns with a unique number are read straight down; columns sharing a number
are read together row by row. The key is the keyword and the period is its
length.
"""

from __future__ import annotations

import time
from itertools import product

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

_PAD = "X"


def _column_numbers(key: str) -> list[int]:
    """Column numbers for a keyword: identical letters share a number.

    Numbers are the 1-based alphabetical rank of the *distinct* letters, so the
    keyword ``TOMATO`` (letters A, M, O, T) maps to ``[4, 3, 2, 1, 4, 3]``.
    """
    letters = only_letters(key)
    if not letters:
        raise ValueError("myszkowski key must contain at least one letter")
    rank = {ch: i + 1 for i, ch in enumerate(sorted(set(letters)))}
    return [rank[ch] for ch in letters]


def _groups(nums: list[int]) -> list[list[int]]:
    """Column indices grouped by ascending column number (read order)."""
    return [[c for c, n in enumerate(nums) if n == num] for num in sorted(set(nums))]


def _encode_letters(letters: str, nums: list[int]) -> str:
    width = len(nums)
    rem = len(letters) % width
    if rem:
        letters = letters + _PAD * (width - rem)
    rows = [letters[i : i + width] for i in range(0, len(letters), width)]
    out: list[str] = []
    for cols in _groups(nums):
        if len(cols) == 1:
            c = cols[0]
            out.extend(row[c] for row in rows)
        else:
            for row in rows:
                out.extend(row[c] for c in cols)
    return "".join(out)


def _decode_letters(cipher: str, nums: list[int], *, strip_pad: bool = True) -> str:
    width = len(nums)
    nrows = len(cipher) // width  # complete-rectangle assumption
    grid: list[list[str]] = [[""] * width for _ in range(nrows)]
    idx = 0
    for cols in _groups(nums):
        if len(cols) == 1:
            c = cols[0]
            for r in range(nrows):
                grid[r][c] = cipher[idx]
                idx += 1
        else:
            for r in range(nrows):
                for c in cols:
                    grid[r][c] = cipher[idx]
                    idx += 1
    plain = "".join(grid[r][c] for r in range(nrows) for c in range(width))
    if strip_pad:
        # Encode pads the final row with at most width-1 trailing X's.
        stripped = plain.rstrip(_PAD)
        if 0 <= len(plain) - len(stripped) < width:
            plain = stripped
    return plain


# All distinct number-partitions of `width` columns into ordered groups, used by
# `crack` when the keyword (hence the repeated-letter pattern) is unknown. A
# partition assigns each column a number 1..width such that the set of numbers
# is exactly {1..k} for some k and the read order follows the numbers.
def _partitions(width: int):
    """Yield every distinct column-number assignment for ``width`` columns.

    Each yielded value is a list ``nums`` of length ``width`` whose distinct
    values are exactly ``1..k``; ties (shared numbers) model repeated keyword
    letters. The *numeric* values matter because they set the read order
    (ascending number), so assignments are deduplicated by the read-order group
    structure rather than by relabelling. Restricted to ``width <= 8`` to stay
    tractable.
    """
    seen: set[tuple] = set()
    for combo in product(range(1, width + 1), repeat=width):
        distinct = sorted(set(combo))
        if distinct != list(range(1, len(distinct) + 1)):
            continue
        groups = tuple(tuple(c for c in range(width) if combo[c] == num) for num in distinct)
        if groups in seen:
            continue
        seen.add(groups)
        yield list(combo)


class Myszkowski(Cipher):
    name = "myszkowski"
    aliases = ("mysz",)
    description = "Columnar transposition with a repeated-letter keyword read row-by-row."
    key_format = "keyword (letters) with at least one repeated letter"
    key_example = "TOMATO"
    complexity = 4

    # Transposition only reorders letters, so it cannot preserve word spacing;
    # encode/decode operate on a clean uppercase letter stream (no reflow, which
    # would leak the plaintext's word lengths into the ciphertext).
    def encode(self, text: str, key: str) -> str:
        return _encode_letters(only_letters(text), _column_numbers(key))

    def decode(self, text: str, key: str) -> str:
        return _decode_letters(only_letters(text), _column_numbers(key))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 4:
            return []
        max_width = int(opts.get("max_width", 7))
        max_width = min(max_width, 8, len(letters))
        widths = [int(opts["width"])] if opts.get("width") else range(2, max_width + 1)
        deadline = (time.monotonic() + timeout) if timeout else None

        candidates: list[Candidate] = []
        seen: set[str] = set()
        truncated_at = None
        for width in widths:
            if width < 2 or width > len(letters):
                continue
            # The true ciphertext is padded to a whole number of rows, so the
            # real period divides its length; widths that don't are skipped to
            # avoid decoding a ragged grid.
            if len(letters) % width != 0:
                continue
            for nums in _partitions(width):
                if deadline and time.monotonic() > deadline:
                    truncated_at = width
                    break
                tag = ",".join(map(str, nums))
                if tag in seen:
                    continue
                seen.add(tag)
                plain = _decode_letters(letters, nums, strip_pad=False)
                candidates.append(
                    Candidate(
                        plaintext=plain,
                        cipher=self.name,
                        key=tag,
                        score=scorer.score(plain),
                        confidence=scorer.confidence(plain),
                        meta={"width": width, "column_numbers": nums},
                    )
                )
            if truncated_at is not None:
                break
        candidates.sort(key=lambda c: c.score, reverse=True)
        out = candidates[:top]
        if truncated_at is not None and out:
            out[-1].meta["timeout_truncated_at_width"] = truncated_at
        return out
