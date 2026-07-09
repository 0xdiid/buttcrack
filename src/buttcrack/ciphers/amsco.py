"""AMSCO incomplete columnar transposition (A.M. Scott).

The grid is filled left-to-right, row by row, with chunks whose size alternates
1, 2, 1, 2, ... *continuously* across the whole fill (the alternation does not
reset each row, so successive rows offset). Columns are then read top-to-bottom
in ascending key-rank order and concatenated.

Key format
----------
``"<order>"`` or ``"<order>:<start>"`` where:

* ``<order>`` is a column permutation given either as digit ranks (e.g. ``31452``
  or ``2,1,3``, 1-based) or as a keyword reduced to the alphabetical rank of its
  letters (as in :mod:`buttcrack.ciphers.columnar`).
* ``<start>`` is the cutting-sequence flag, ``1`` (cells take 1,2,1,2,...) or
  ``2`` (2,1,2,1,...). Defaults to ``1`` when omitted.

``crack`` reports the key as ``"<comma-order>:<start>"``, which feeds straight
back into ``decode``.
"""

from __future__ import annotations

import time
from itertools import permutations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher


def _parse_key(key: str) -> tuple[list[int], int]:
    """Return ``(read_order, start)`` from a key string.

    ``read_order`` is the list of 0-based column indices in the order their
    columns are read out (ascending key rank). ``start`` is 1 or 2.
    """
    s = str(key).strip()
    start = 1
    if ":" in s:
        s, _, flag = s.partition(":")
        s = s.strip()
        flag = flag.strip()
        if flag not in ("1", "2"):
            raise ValueError("amsco cutting-sequence flag must be 1 or 2")
        start = int(flag)

    if "," in s or " " in s:
        parts = s.replace(",", " ").split()
    elif s.isdigit():
        # Compact form like "31452" -> one digit per column.
        parts = list(s)
    else:
        parts = []
    if parts and all(part.isdigit() for part in parts):
        ranks = [int(part) for part in parts]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError(f"amsco numeric key must be a permutation of 1..{len(ranks)}")
        # column index sorted by its rank value
        read_order = [idx for _, idx in sorted((rank, i) for i, rank in enumerate(ranks))]
    else:
        letters = only_letters(s)
        if not letters:
            raise ValueError("amsco key must be a keyword or numeric column order")
        read_order = [idx for _, idx in sorted((ch, i) for i, ch in enumerate(letters))]
    return read_order, start


def _row_sizes(n: int, width: int, start: int) -> list[list[int]]:
    """Chunk size of every cell, grouped by row, for ``n`` letters.

    The chunk size alternates 1/2 continuously across the whole fill; the final
    chunk may be short if the plaintext runs out mid-cell.
    """
    rows: list[list[int]] = []
    row: list[int] = []
    pos = 0
    size = start
    while pos < n:
        take = min(size, n - pos)
        pos += take
        row.append(take)
        size = 2 if size == 1 else 1
        if len(row) == width:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def _column_totals(rows: list[list[int]], width: int) -> list[int]:
    totals = [0] * width
    for row in rows:
        for ci, sz in enumerate(row):
            totals[ci] += sz
    return totals


def _encode_letters(letters: str, read_order: list[int], start: int) -> str:
    width = len(read_order)
    rows = _row_sizes(len(letters), width, start)
    columns: list[list[str]] = [[] for _ in range(width)]
    pos = 0
    for row in rows:
        for ci, sz in enumerate(row):
            columns[ci].append(letters[pos : pos + sz])
            pos += sz
    return "".join("".join(columns[c]) for c in read_order)


def _decode_letters(cipher: str, read_order: list[int], start: int) -> str:
    width = len(read_order)
    n = len(cipher)
    rows = _row_sizes(n, width, start)
    totals = _column_totals(rows, width)
    # Peel the ciphertext into columns following the read order.
    column_text: list[str] = [""] * width
    idx = 0
    for c in read_order:
        column_text[c] = cipher[idx : idx + totals[c]]
        idx += totals[c]
    # Refill the grid row by row, consuming each column's letters in order.
    col_pos = [0] * width
    out: list[str] = []
    for row in rows:
        for ci, sz in enumerate(row):
            out.append(column_text[ci][col_pos[ci] : col_pos[ci] + sz])
            col_pos[ci] += sz
    return "".join(out)


class Amsco(Cipher):
    name = "amsco"
    aliases = ("amscott",)
    description = "AMSCO incomplete columnar transposition with alternating 1/2-letter chunks."
    key_format = "order/keyword, optional :start flag (1 or 2), e.g. 31452:1 or KEY:2"
    key_example = "31452:1"
    complexity = 4

    # Transposition only reorders letters, so it cannot preserve word spacing;
    # encode/decode operate on a clean uppercase letter stream (no reflow, which
    # would leak the plaintext's word lengths into the ciphertext).
    def encode(self, text: str, key: str) -> str:
        read_order, start = _parse_key(key)
        return _encode_letters(only_letters(text), read_order, start)

    def decode(self, text: str, key: str) -> str:
        read_order, start = _parse_key(key)
        return _decode_letters(only_letters(text), read_order, start)

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
            stop = False
            for start in (1, 2):
                for perm in permutations(range(width)):
                    if deadline and time.monotonic() > deadline:
                        truncated_at = width
                        stop = True
                        break
                    read_order = list(perm)
                    plain = _decode_letters(letters, read_order, start)
                    ranks = [0] * width
                    for rank, col in enumerate(read_order, start=1):
                        ranks[col] = rank
                    key_str = ",".join(map(str, ranks)) + f":{start}"
                    candidates.append(
                        Candidate(
                            plaintext=plain,
                            cipher=self.name,
                            key=key_str,
                            score=scorer.score(plain),
                            confidence=scorer.confidence(plain),
                            meta={"width": width, "start": start},
                        )
                    )
                if stop:
                    break
            if truncated_at is not None:
                break
        candidates.sort(key=lambda c: c.score, reverse=True)
        out = candidates[:top]
        if truncated_at is not None and out:
            out[-1].meta["timeout_truncated_at_width"] = truncated_at
        return out
