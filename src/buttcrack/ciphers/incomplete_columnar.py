"""Incomplete columnar transposition.

Like complete columnar but with NO null padding: the message length need not be
a multiple of the keyword length, so the final grid row is partially filled and
the trailing (rightmost-by-position) columns are one letter shorter.

Key may be a keyword (columns ordered by the alphabetical rank of its letters,
ties broken left-to-right) or an explicit 0-based read order such as ``1,2,0``.
``crack`` reports the numeric read order, which feeds straight back into
``decode``.
"""

from __future__ import annotations

import math
import time
from itertools import permutations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher


def _read_order(key: str) -> list[int]:
    """Return columns in ascending-rank (read) order as physical positions.

    ``order[k]`` is the physical column read k-th. A keyword is ranked
    alphabetically (ties left-to-right); a numeric key is taken verbatim as the
    read order.
    """
    s = str(key).strip()
    if s and all(ch.isdigit() or ch in ", " for ch in s):
        order = [int(x) for x in s.replace(",", " ").split()]
    else:
        letters = only_letters(s)
        if not letters:
            raise ValueError("incomplete-columnar key must be a keyword or numeric read order")
        order = [idx for _, idx in sorted((ch, i) for i, ch in enumerate(letters))]
    if sorted(order) != list(range(len(order))):
        raise ValueError(
            f"incomplete-columnar read order must be a permutation of 0..{len(order) - 1}"
        )
    return order


def _column_heights(n: int, width: int) -> list[int]:
    """Heights of the physical columns 0..width-1 for an incomplete grid.

    With ``rows = ceil(n / width)`` and ``r = n mod width``, the first ``r``
    physical columns are LONG (height ``rows``) and the rest are SHORT (height
    ``rows - 1``). When ``r == 0`` the grid is complete and all columns are
    equal height.
    """
    rows = math.ceil(n / width) if n else 0
    r = n % width
    if r == 0:
        return [rows] * width
    return [rows if c < r else rows - 1 for c in range(width)]


def _encode_letters(letters: str, order: list[int]) -> str:
    width = len(order)
    columns: list[list[str]] = [[] for _ in range(width)]
    # Fill row by row across the physical columns; the short final row simply
    # runs out of letters before reaching the rightmost columns.
    for i, ch in enumerate(letters):
        columns[i % width].append(ch)
    return "".join("".join(columns[c]) for c in order)


def _decode_letters(cipher: str, order: list[int]) -> str:
    width = len(order)
    n = len(cipher)
    heights = _column_heights(n, width)
    # Peel a block off the ciphertext for each column, walking columns in
    # ascending rank order; each block's length is its physical column height.
    columns: list[str] = [""] * width
    idx = 0
    for c in order:
        columns[c] = cipher[idx : idx + heights[c]]
        idx += heights[c]
    # Read the grid back row by row, walking physical columns left-to-right and
    # skipping any column already exhausted (trailing cells of the short row).
    pos = [0] * width
    out: list[str] = []
    while len(out) < n:
        for c in range(width):
            if pos[c] < heights[c]:
                out.append(columns[c][pos[c]])
                pos[c] += 1
    return "".join(out)


class IncompleteColumnar(Cipher):
    name = "incomplete-columnar"
    aliases = ("incomplete-coltrans", "incolumnar")
    description = "Incomplete columnar transposition (no padding); key is a keyword or read order."
    key_format = "keyword (letters) or 0-based numeric read order, e.g. 1,2,0"
    key_example = "REALITY"
    complexity = 4

    # Transposition cannot preserve word spacing; operate on a clean uppercase
    # letter stream (no reflow, which would leak plaintext word lengths).
    def encode(self, text: str, key: str) -> str:
        return _encode_letters(only_letters(text), _read_order(key))

    def decode(self, text: str, key: str) -> str:
        return _decode_letters(only_letters(text), _read_order(key))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 4:
            return []
        max_width = int(opts.get("max_width", 7))
        widths = [int(opts["width"])] if opts.get("width") else range(2, max_width + 1)
        deadline = (time.monotonic() + timeout) if timeout else None

        candidates: list[Candidate] = []
        truncated_at = None
        for width in widths:
            if width < 2 or width > len(letters):
                continue
            for perm in permutations(range(width)):
                if deadline and time.monotonic() > deadline:
                    truncated_at = width
                    break
                order = list(perm)
                plain = _decode_letters(letters, order)
                candidates.append(
                    Candidate(
                        plaintext=plain,
                        cipher=self.name,
                        key=",".join(map(str, order)),
                        score=scorer.score(plain),
                        confidence=scorer.confidence(plain),
                        meta={"width": width},
                    )
                )
            if truncated_at is not None:
                break
        candidates.sort(key=lambda c: c.score, reverse=True)
        out = candidates[:top]
        if truncated_at is not None and out:
            out[-1].meta["timeout_truncated_at_width"] = truncated_at
        return out
