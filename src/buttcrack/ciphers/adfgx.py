"""ADFGX cipher: 5x5 Polybius fractionation followed by a columnar transposition.

A two-stage WWI field cipher (Nebel, 1918). Stage 1 replaces each plaintext
letter with the (row, col) label pair of its cell in a keyed 5x5 square whose
rows/cols are labelled A,D,F,G,X (J merges into I). Stage 2 writes the doubled
A/D/F/G/X stream row-by-row under a transposition keyword and reads the columns
out in the keyword's alphabetical order.

KEY FORMAT (one --key string, "/" separator):  SQUAREKEY/COLKEY
  * SQUAREKEY builds the 5x5 keyed alphabet (keyword, or a full 25-letter mixed
    alphabet to reproduce an arbitrary historical square; J/I merged).
  * COLKEY is the columnar transposition keyword (any length >= 1), or an
    explicit 0-based read order such as "1,0,3,4,2".

Example: "BTALPDHOZKQFVSNGICUXMREWY/CARGO" reproduces the Wikipedia worked example.

Output is a clean uppercase letter stream over {A,D,F,G,X} (no reflow).
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from . import _fractionation as frac
from .base import Cipher
from .squares import PolybiusSquare

LABELS = "ADFGX"
#: the 25 possible fractionation digraphs over the row/col labels
_ALL_DIGRAPHS = [a + b for a in LABELS for b in LABELS]
#: plaintext symbols a 5x5 square holds (25 letters; J merges into I)
_TARGET_SYMBOLS = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
#: below this many fractionation symbols, blind two-phase recovery is unreliable
BLIND_MIN_SYMBOLS = 200


def _split_key(key: str) -> tuple[str, str]:
    if "/" not in key:
        raise ValueError("ADFGX key must be 'SQUAREKEY/COLKEY' (slash-separated)")
    square_key, _, col_key = key.partition("/")
    if not square_key.strip() or not col_key.strip():
        raise ValueError("ADFGX key must be 'SQUAREKEY/COLKEY' (both parts required)")
    return square_key, col_key


def _read_order(col_key: str) -> list[int]:
    """Column read order: numeric permutation, or keyword's alphabetical rank."""
    s = str(col_key).strip()
    if s and all(ch.isdigit() or ch in ", " for ch in s):
        order = [int(x) for x in s.replace(",", " ").split()]
    else:
        letters = only_letters(s)
        if not letters:
            raise ValueError("columnar key must be a keyword or numeric read order")
        order = [idx for _, idx in sorted((ch, i) for i, ch in enumerate(letters))]
    if sorted(order) != list(range(len(order))):
        raise ValueError(f"columnar read order must be a permutation of 0..{len(order) - 1}")
    return order


def _column_lengths(n: int, width: int) -> list[int]:
    full_rows, extra = divmod(n, width)
    return [full_rows + (1 if c < extra else 0) for c in range(width)]


def _transpose_encode(stream: str, order: list[int]) -> str:
    width = len(order)
    columns: list[list[str]] = [[] for _ in range(width)]
    for i, ch in enumerate(stream):
        columns[i % width].append(ch)
    return "".join("".join(columns[c]) for c in order)


def _transpose_decode(cipher: str, order: list[int]) -> str:
    width = len(order)
    n = len(cipher)
    lengths = _column_lengths(n, width)
    columns: list[str] = [""] * width
    idx = 0
    for c in order:
        columns[c] = cipher[idx : idx + lengths[c]]
        idx += lengths[c]
    pos = [0] * width
    out = []
    for i in range(n):
        c = i % width
        out.append(columns[c][pos[c]])
        pos[c] += 1
    return "".join(out)


def _fractionate(letters: str, square: PolybiusSquare) -> str:
    out = []
    for ch in square.prepare(letters):
        r, c = square.rc(ch)
        out.append(LABELS[r])
        out.append(LABELS[c])
    return "".join(out)


def _defractionate(stream: str, square: PolybiusSquare) -> str:
    out = []
    for i in range(0, len(stream) - 1, 2):
        a, b = stream[i], stream[i + 1]
        if a not in LABELS or b not in LABELS:
            continue
        out.append(square.at(LABELS.index(a), LABELS.index(b)))
    return "".join(out)


class ADFGX(Cipher):
    """ADFGX: 5x5 keyed-square fractionation + columnar transposition.

    KEY FORMAT (one --key string): ``SQUAREKEY/COLKEY``
      * ``SQUAREKEY`` -- keyword (or full 25-letter mixed alphabet) for the 5x5
        Polybius square; J merges into I.
      * ``COLKEY`` -- columnar transposition keyword, or a numeric read order
        like ``1,0,3,4,2``.

    Encode/decode operate on a letters-only uppercase stream; output is over
    {A,D,F,G,X}. Decode recovers the prepared plaintext (J appears as I).
    """

    name = "adfgx"
    aliases = ("adfg",)
    description = "5x5 Polybius fractionation (A/D/F/G/X) plus columnar transposition."
    key_format = "squarekey/colkey (5x5 keyword and columnar transposition keyword)"
    key_example = "KEYWORD/CARGO"
    complexity = 6
    ciphertext_alphabet = "ADFGX"

    def encode(self, text: str, key: str) -> str:
        square_key, col_key = _split_key(key)
        square = PolybiusSquare(square_key, size=5)
        order = _read_order(col_key)
        stream = _fractionate(text, square)
        return _transpose_encode(stream, order)

    def decode(self, text: str, key: str) -> str:
        square_key, col_key = _split_key(key)
        square = PolybiusSquare(square_key, size=5)
        order = _read_order(col_key)
        stream = _transpose_decode(only_letters(text), order)
        return _defractionate(stream, square)

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
        """Best-effort blind two-phase recovery (long messages).

        ADFGX is fractionation then transposition, so the layers peel in order:
        recover the columnar transposition by the **digraph index-of-coincidence**
        (mapping-independent — it spikes only when the columns are un-transposed so
        that consecutive pairs reconstitute the original digraphs), then solve the
        resulting digraph stream as a simple substitution (the 5x5 square). See
        :mod:`buttcrack.ciphers._fractionation`. Reliable only on long messages
        (>= ~200 A/D/F/G/X symbols) and not guaranteed.
        """
        # ADFGX ciphertext is written ENTIRELY in A/D/F/G/X; reject input that
        # isn't, so we don't "solve" the handful of matching letters of an
        # unrelated message (shared guard via ``ciphertext_alphabet``).
        if not self.ciphertext_alphabet_ok(text):
            return []
        stream = "".join(ch for ch in only_letters(text) if ch in LABELS)
        deadline = (time.monotonic() + timeout) if timeout else None
        return frac.two_phase_crack(
            text,
            stream,
            scorer=scorer,
            rng=rng or random.Random(),
            deadline=deadline,
            name=self.name,
            target_symbols=_TARGET_SYMBOLS,
            all_digraphs=_ALL_DIGRAPHS,
            min_symbols=BLIND_MIN_SYMBOLS,
            opts=opts,
        )
