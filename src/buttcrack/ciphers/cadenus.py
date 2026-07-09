"""Cadenus cipher -- keyed columnar transposition with per-column rotation.

The Cadenus (Soudart & Lange, *Traite de Cryptographie*, 1925) is a
transposition whose message length must be an exact multiple of 25.  The
keyword length ``C`` equals ``len(text) / 25`` and a single keyword does
double duty: it sets BOTH the column read order AND each column's cyclic
rotation.

ALGORITHM (ACA / CryptoCrack convention)
----------------------------------------
A 25-letter row alphabet ``A B C ... U V X Y Z`` (the letter ``W`` is folded
into ``V`` so the alphabet has exactly 25 rows) is written down the side of a
25-row grid.

ENCRYPT:

1. Normalize the plaintext (uppercase, letters only, ``W`` -> ``V``); its
   length must be ``25 * C`` where ``C = len(keyword)``.
2. Write the plaintext into a 25-row by ``C``-column grid, row by row.
3. For each column ``i`` rotate it UPWARD by ``s_i`` rows, where
   ``s_i`` is the index of ``keyword[i]`` in the 25-letter row alphabet
   ``ABCDEFGHIJKLMNOPQRSTUVXYZ`` (``A``=0, ``B``=1, ... ``E``=4, ... ``S``=18,
   ``T``=19, ... ``Z``=24).
4. Permute the columns into ascending alphabetical order of their keyword
   letters (ties broken left-to-right).
5. Read the resulting grid off BY ROWS (left to right, top to bottom).

DECRYPT reverses steps 5->2: read the ciphertext into the (sorted) columns by
rows, rotate each column DOWNWARD by ``s_i``, restore the original column
order, then read by rows.

KEY FORMAT
----------
A single keyword of letters whose length is exactly ``len(text) / 25``.  The
keyword letters supply both the column permutation (alphabetical rank) and the
per-column rotation (index in ``ABCDEFGHIJKLMNOPQRSTUVXYZ``).  A duplicated
keyword letter is legal: identical letters keep their left-to-right order.

Verified against the ACA worked example (keyword ``SET``, "to all men there is
a season ...") from The Black Chamber.
"""

from __future__ import annotations

import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import reflow
from ..words import _words
from .base import Cipher

ROWS = 25
#: 25-letter row alphabet with W folded into V; index gives the rotation amount.
ROW_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVXYZ"


def _normalize(text: str) -> str:
    """Uppercase, keep A-Z only, fold W into V (the cipher has 25 rows)."""
    return "".join(("V" if ch == "W" else ch) for ch in text.upper() if "A" <= ch <= "Z")


def _key_letters(key: str) -> str:
    letters = _normalize(key)
    if not letters:
        raise ValueError("cadenus key must contain letters")
    return letters


def _column_order(kw: str) -> list[int]:
    """Indices of ``kw`` sorted by (letter, position) -> ascending column order."""
    return sorted(range(len(kw)), key=lambda i: (kw[i], i))


def _rotations(kw: str) -> list[int]:
    return [ROW_ALPHA.index(ch) for ch in kw]


def _encode_letters(letters: str, kw: str) -> str:
    width = len(kw)
    if len(letters) != ROWS * width:
        raise ValueError(
            f"cadenus needs len(text) == 25 * len(key); got {len(letters)} for key length {width}"
        )
    grid = [[letters[r * width + c] for c in range(width)] for r in range(ROWS)]
    order = _column_order(kw)
    rot = _rotations(kw)
    out_cols: list[list[str]] = []
    for i in order:
        col = [grid[r][i] for r in range(ROWS)]
        s = rot[i]
        out_cols.append(col[s:] + col[:s])  # rotate UP by s
    out: list[str] = []
    for r in range(ROWS):
        for col in out_cols:
            out.append(col[r])
    return "".join(out)


def _decode_letters(letters: str, kw: str) -> str:
    width = len(kw)
    if len(letters) != ROWS * width:
        raise ValueError(
            f"cadenus needs len(text) == 25 * len(key); got {len(letters)} for key length {width}"
        )
    # Read ciphertext into the sorted columns, row by row.
    sorted_cols = [[letters[r * width + c] for r in range(ROWS)] for c in range(width)]
    order = _column_order(kw)
    rot = _rotations(kw)
    grid: list[list[str]] = [["" for _ in range(width)] for _ in range(ROWS)]
    for pos, i in enumerate(order):
        col = sorted_cols[pos]
        s = rot[i]
        restored = (col[-s:] + col[:-s]) if s else col[:]  # rotate DOWN by s
        for r in range(ROWS):
            grid[r][i] = restored[r]
    out: list[str] = []
    for r in range(ROWS):
        for c in range(width):
            out.append(grid[r][c])
    return "".join(out)


class Cadenus(Cipher):
    """Cadenus: 25-row keyed columnar transposition with per-column rotation."""

    name = "cadenus"
    aliases = ()
    description = "Cadenus transposition: 25-row grid, keyword sets column order and rotation."
    key_format = "keyword (letters) of length exactly len(text)/25"
    key_example = "SET"
    complexity = 5

    # Transposition cannot preserve word spacing; operate on a clean uppercase
    # letter stream (W folded to V) so output never leaks word lengths.
    def encode(self, text: str, key: str) -> str:
        return _encode_letters(_normalize(text), _key_letters(key))

    def decode(self, text: str, key: str) -> str:
        return _decode_letters(_normalize(text), _key_letters(key))

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
        """Dictionary keyword search.

        The keyword length is fixed at ``len(text) / 25`` and a single keyword
        determines both the column order and every column's rotation, so the
        sound attack is a dictionary search over words of exactly that length.
        Caller-supplied ``keyword``/``keywords`` are tried first; otherwise the
        bundled English word list of the right length is swept.  Returns ``[]``
        for messages that are not a multiple of 25.
        """
        letters = _normalize(text)
        if not letters or len(letters) % ROWS != 0:
            return []
        width = len(letters) // ROWS
        if width < 1:
            return []

        keywords: list[str] = []
        if opts.get("keyword"):
            keywords.append(str(opts["keyword"]))
        keywords.extend(str(k) for k in opts.get("keywords", []))
        keywords = [k for k in keywords if _normalize(k) and len(_normalize(k)) == width]
        if not keywords:
            keywords = [w for w in _words() if len(w) == width]
        if not keywords:
            return []

        deadline = (time.monotonic() + timeout) if timeout else None
        results: list[tuple[float, str, str]] = []  # (score, plaintext, keyword)
        for keyword in keywords:
            if deadline and time.monotonic() > deadline:
                break
            kw = _normalize(keyword)
            if len(kw) != width:
                continue
            try:
                plain = _decode_letters(letters, kw)
            except ValueError:
                continue
            results.append((scorer.score(plain), plain, kw))

        results.sort(key=lambda r: r[0], reverse=True)
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for score, plain, keyword in results[: max(top, 1) * 4]:
            if plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=keyword,
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"keyword": keyword, "width": width},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
