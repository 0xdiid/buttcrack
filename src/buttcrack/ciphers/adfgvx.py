"""ADFGVX cipher: 6x6 fractionation (A-Z + 0-9) followed by columnar transposition.

Two-stage WWI field cipher (Fritz Nebel, 1918):

  STAGE 1 - fractionation.  A 6x6 keyed square holds all 36 characters
  (26 letters + the digits 0-9; there is NO I/J merge, unlike the 5x5 ADFGX).
  Its rows and columns are labelled A, D, F, G, V, X. Each plaintext character
  is replaced by its (rowLabel, colLabel) pair, doubling the stream length.

  STAGE 2 - columnar transposition.  The A/D/F/G/V/X stream is written row by
  row under a transposition keyword (one column per keyword letter), the columns
  are ranked by the alphabetical order of the keyword's letters, and the columns
  are read out in that ranked order and concatenated. The final row may be
  partial, so columns are unequal: the leftmost (by original position) columns
  hold the extra cells.

KEY FORMAT: a single ``--key`` string ``SQUARE/COLKEY`` (separator ``/``).
  * ``SQUARE``  - either a keyword/phrase used to fill the 6x6 square over
    A-Z0-9 (remaining cells in alphabet-then-digits order, via PolybiusSquare),
    OR an explicit 36-character grid (a permutation of A-Z0-9, read row by row)
    when you need an exact published square.
  * ``COLKEY``  - the columnar transposition keyword (or an explicit numeric
    read order such as ``4,5,2,0,1,3,6``).

Example (Wikipedia worked example): key
``"NA1C3H8TB2OME5WRPD4F6G7I9J0KLQSUVXYZ/PRIVACY"`` encodes ``attackat1200am``
to ``DGDDDAGDDGAFADDFDADVDVFAADVX``.

Cracking a keyless ADFGVX message is hard; see ``crack`` for the (best-effort)
behaviour.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from . import _fractionation as frac
from .base import Cipher
from .squares import ALPHABET_6, PolybiusSquare

LABELS = "ADFGVX"
_LABEL_INDEX = {ch: i for i, ch in enumerate(LABELS)}
_SET_6 = set(ALPHABET_6)
#: the 36 possible fractionation digraphs over the row/col labels
_ALL_DIGRAPHS = [a + b for a in LABELS for b in LABELS]
#: plaintext symbols a 6x6 square holds (26 letters + 10 digits)
_TARGET_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#: below this many fractionation symbols, blind two-phase recovery is unreliable
BLIND_MIN_SYMBOLS = 200


def _clean_plaintext(text: str) -> str:
    """Uppercase, keep only A-Z and 0-9 (the 36 encodable symbols)."""
    return "".join(ch for ch in text.upper() if ch in _SET_6)


def _split_key(key: str) -> tuple[str, str]:
    if "/" not in key:
        raise ValueError("ADFGVX key must be 'SQUARE/COLKEY' (separator '/')")
    square_part, col_part = key.split("/", 1)
    square_part = square_part.strip()
    col_part = col_part.strip()
    if not square_part or not col_part:
        raise ValueError("ADFGVX key must be 'SQUARE/COLKEY' with both parts non-empty")
    return square_part, col_part


def _build_square(square_key: str) -> PolybiusSquare:
    """Return a 6x6 square from a keyword, or from an explicit 36-char grid.

    An explicit grid (a permutation of A-Z + 0-9) is passed straight through as
    the square's alphabet so a published vector can be reproduced exactly.
    """
    cleaned = "".join(ch for ch in square_key.upper() if ch in _SET_6)
    if len(cleaned) == 36 and set(cleaned) == _SET_6:
        # Explicit full grid: use it verbatim (no keyword, alphabet IS the grid).
        return PolybiusSquare("", size=6, alphabet=cleaned)
    return PolybiusSquare(square_key, size=6)


def _read_order(col_key: str) -> list[int]:
    """Column read order: explicit numeric list, or alphabetical rank of a keyword."""
    s = str(col_key).strip()
    if s and all(ch.isdigit() or ch in ", " for ch in s):
        order = [int(x) for x in s.replace(",", " ").split()]
    else:
        letters = [ch for ch in s.upper() if "A" <= ch <= "Z" or "0" <= ch <= "9"]
        if not letters:
            raise ValueError("ADFGVX columnar key must be a keyword or numeric read order")
        order = [idx for _, idx in sorted((ch, i) for i, ch in enumerate(letters))]
    if sorted(order) != list(range(len(order))):
        raise ValueError(f"ADFGVX read order must be a permutation of 0..{len(order) - 1}")
    return order


def _column_lengths(n: int, width: int) -> list[int]:
    full_rows, extra = divmod(n, width)
    return [full_rows + (1 if c < extra else 0) for c in range(width)]


def _fractionate(letters: str, sq: PolybiusSquare) -> str:
    out: list[str] = []
    for ch in letters:
        r, c = sq.rc(ch)
        out.append(LABELS[r])
        out.append(LABELS[c])
    return "".join(out)


def _transpose(stream: str, order: list[int]) -> str:
    width = len(order)
    columns: list[list[str]] = [[] for _ in range(width)]
    for i, ch in enumerate(stream):
        columns[i % width].append(ch)
    return "".join("".join(columns[c]) for c in order)


def _untranspose(cipher: str, order: list[int]) -> str:
    width = len(order)
    n = len(cipher)
    lengths = _column_lengths(n, width)
    columns: list[str] = [""] * width
    idx = 0
    for c in order:
        columns[c] = cipher[idx : idx + lengths[c]]
        idx += lengths[c]
    pos = [0] * width
    out: list[str] = []
    for i in range(n):
        c = i % width
        out.append(columns[c][pos[c]])
        pos[c] += 1
    return "".join(out)


def _defractionate(stream: str, sq: PolybiusSquare) -> str:
    out: list[str] = []
    # Consecutive pairs -> (row, col) -> letter. Drop a dangling odd char.
    for i in range(0, len(stream) - 1, 2):
        a, b = stream[i], stream[i + 1]
        if a not in _LABEL_INDEX or b not in _LABEL_INDEX:
            continue
        out.append(sq.at(_LABEL_INDEX[a], _LABEL_INDEX[b]))
    return "".join(out)


class ADFGVX(Cipher):
    name = "adfgvx"
    description = "6x6 fractionation (A-Z + 0-9) then columnar transposition (Nebel, 1918)."
    key_format = "square/colkey (6x6 keyword over A-Z0-9 and columnar transposition keyword)"
    key_example = "KEYWORD/PRIVACY"
    complexity = 6
    ciphertext_alphabet = "ADFGVX"

    # Fractionation + transposition cannot preserve word spacing; output is a
    # clean uppercase A/D/F/G/V/X stream (no reflow).
    def encode(self, text: str, key: str) -> str:
        square_key, col_key = _split_key(key)
        sq = _build_square(square_key)
        order = _read_order(col_key)
        stream = _fractionate(_clean_plaintext(text), sq)
        return _transpose(stream, order)

    def decode(self, text: str, key: str) -> str:
        square_key, col_key = _split_key(key)
        sq = _build_square(square_key)
        order = _read_order(col_key)
        stream = "".join(ch for ch in text.upper() if ch in _LABEL_INDEX)
        plain_stream = _untranspose(stream, order)
        return _defractionate(plain_stream, sq)

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

        ADFGVX is fractionation *then* transposition, and the two layers peel in
        order without a crib:

        1. **Transposition** by a mapping-independent statistic. When the columns are
           un-transposed correctly, consecutive symbol pairs reconstitute the original
           digraphs, whose frequency profile is a 1:1 substitution of English letters
           — so the **digraph index-of-coincidence ≈ 0.066**; a wrong order mixes the
           halves of different letters and flattens it. Annealing the column order to
           maximise digraph-IoC recovers the transposition *without knowing the
           square* (IoC is invariant to the digraph→letter relabelling).
        2. **Square** as a simple substitution. The recovered digraph stream is a
           monoalphabetic substitution over <=36 symbols; anneal the digraph→symbol
           map on the quadgram score.

        Nested and hard: reliable only on long messages (>= ~200 fractionation
        symbols, i.e. ~100 plaintext characters) and not guaranteed. Returns the best
        candidate found, or ``[]`` when too short.
        """
        stream = "".join(ch for ch in text.upper() if ch in _LABEL_INDEX)
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
