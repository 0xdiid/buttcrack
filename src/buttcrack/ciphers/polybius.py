"""Polybius square cipher — every letter becomes its (row, column) coordinate pair.

The square itself is the workhorse behind Bifid, Trifid, ADFGX, Nihilist, Playfair
and the rest of the grid family, and ``PolybiusSquare`` has always been available
inside this package. What was missing was the *cipher* in its own right: the plain
"letter -> two digits" encoding with no fractionation, transposition or additive on
top. It is the most common outer wrapper in puzzle chains, and without a registry
entry it could not be reached from ``butt encode``/``decode``/``crack``.

ALGORITHM
---------
Build a keyed mixed 5x5 square (J merged into I, the ACA standard). Each plaintext
letter is replaced by the two digits ``(row)(column)``, both 1-indexed, row first.
``ZEBRAS`` square, letter ``D`` -> ``23``.

Decryption is the inverse lookup. The cipher is a pure monoalphabetic substitution
written in a two-digit alphabet — it hides nothing that frequency analysis of the
digit pairs does not immediately give back, which is exactly why ``crack`` can solve
it outright rather than searching keywords.

KEY FORMAT
----------
A keyword for the mixed square (``KRYPTOS``). An empty key gives the standard
A-Z (J->I) square.
"""

from __future__ import annotations

import re

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .squares import ALPHABET_5, PolybiusSquare

#: 25 proxy letters, one per possible coordinate pair, for the monoalphabetic solve.
_PROXY = ALPHABET_5


def _digits(text: str) -> str:
    return re.sub(r"[^1-5]", "", str(text))


def _pairs(text: str) -> list[tuple[int, int]]:
    d = _digits(text)
    return [(int(d[i]) - 1, int(d[i + 1]) - 1) for i in range(0, len(d) - 1, 2)]


class Polybius(Cipher):
    name = "polybius"
    aliases = ("polybius-square", "checkerboard-polybius")
    description = "5x5 keyed square; each letter becomes its 1-indexed row/column digit pair."
    key_format = "square keyword (empty = standard A-Z square, J->I)"
    key_example = "KRYPTOS"
    complexity = 2
    ciphertext_alphabet = "12345"

    def encode(self, text: str, key: str) -> str:
        square = PolybiusSquare(key or "")
        out = []
        for ch in square.prepare(text):
            row, col = square.rc(ch)
            out.append(f"{row + 1}{col + 1}")
        return " ".join(out)

    def decode(self, text: str, key: str) -> str:
        square = PolybiusSquare(key or "")
        return "".join(square.at(r, c) for r, c in _pairs(text))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Solve outright as a monoalphabetic substitution over the digit pairs.

        There is no keyword search here and there does not need to be one. Each
        distinct coordinate pair stands for exactly one plaintext letter, so relabel
        the pairs as proxy letters and the problem *is* the monoalphabetic problem the
        substitution solver already handles — recovering the plaintext without ever
        guessing the keyword. The square is then read straight off the solved map.

        Cells for pairs that never occur in the ciphertext are unrecoverable (nothing
        constrains them) and come back as ``?`` in the reported square.
        """
        from .substitution import Substitution

        pairs = _pairs(text)
        if len(pairs) < 40:
            return []

        # Relabel each distinct coordinate pair as a proxy letter, in order of first
        # appearance. More than 25 distinct pairs means this is not a 5x5 Polybius.
        order: list[tuple[int, int]] = []
        index: dict[tuple[int, int], int] = {}
        for p in pairs:
            if p not in index:
                if len(order) >= len(_PROXY):
                    return []
                index[p] = len(order)
                order.append(p)
        proxy_text = "".join(_PROXY[index[p]] for p in pairs)

        inner = Substitution().crack(proxy_text, scorer, top=top, rng=rng, timeout=timeout)

        out: list[Candidate] = []
        for cand in inner:
            plain = only_letters(cand.plaintext)
            if len(plain) != len(pairs):
                continue
            # Read the square off the solved map: the pair at (r, c) decrypts to the
            # plaintext letter that appeared wherever that pair occurred.
            grid = ["?"] * 25
            for pair, i in index.items():
                grid[pair[0] * 5 + pair[1]] = plain[proxy_text.index(_PROXY[i])]
            square = "".join(grid)
            out.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=square,
                    score=cand.score,
                    confidence=cand.confidence,
                    meta={
                        "square": square,
                        "cells_recovered": 25 - grid.count("?"),
                        "method": "monoalphabetic solve over coordinate pairs",
                    },
                )
            )
        return out[:top]
