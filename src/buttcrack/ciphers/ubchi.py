"""Ubchi (Übchi) — WWI German double columnar transposition with a REUSED key.

The distinguishing feature is that both passes use the *same* permutation, with a
handful of null letters inserted between them. Reusing the key is what makes it
weaker than a general double transposition: the keyspace is the permutations of one
width rather than of two, so ``crack`` can enumerate it outright below width 8 where
a two-key double transposition cannot be touched.

ALGORITHM
---------
1. Columnar-transpose the plaintext under the keyword.
2. Append ``n`` null letters to the intermediate result.
3. Columnar-transpose *that* under the SAME keyword.

Decryption undoes the outer transposition, drops the ``n`` trailing nulls, and
undoes the inner one.

KEY FORMAT
----------
``KEYWORD`` or ``KEYWORD/NULLS`` — e.g. ``UBER/1``. The permutation may also be
given as an explicit 0-based read order (``1,2,3,0/1``), which is what ``crack``
reports. ``NULLS`` defaults to 0.

Reference: dCode "Ubchi Cipher" — ``SECRET`` under key ``UBER`` with one null
encrypts to ``TECXRES``.
"""

from __future__ import annotations

import time
from itertools import permutations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .columnar import Columnar

#: Above this width the permutations stop being enumerable (9! = 362880 x null
#: counts is past the point of a useful interactive crack).
BRUTE_MAX_WIDTH = 8

#: The letter appended between the two passes when encoding.
NULL_LETTER = "X"


def _split_key(key: str) -> tuple[str, int]:
    s = str(key).strip()
    head, sep, tail = s.rpartition("/")
    if sep and tail.strip().isdigit():
        return head.strip(), int(tail)
    return s, 0


class Ubchi(Cipher):
    name = "ubchi"
    aliases = ("uebchi", "double-columnar-same-key")
    description = "WWI German double columnar transposition reusing one key, with nulls between."
    key_format = "keyword or read order, optionally '/NULLS' (e.g. UBER/1)"
    key_example = "UBER/1"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        keyword, nulls = _split_key(key)
        columnar = Columnar()
        first = columnar.encode(only_letters(text), keyword)
        return columnar.encode(first + NULL_LETTER * nulls, keyword)

    def decode(self, text: str, key: str) -> str:
        keyword, nulls = _split_key(key)
        columnar = Columnar()
        padded = columnar.decode(only_letters(text), keyword)
        inner = padded[: len(padded) - nulls] if nulls else padded
        return columnar.decode(inner, keyword)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Enumerate the permutation exhaustively — the key reuse makes this possible.

        A general double transposition needs two independent permutations, so even
        width 6 is 518400 pairs. Ubchi reuses one, which collapses the search to ``w!``
        per width; widths 2..8 together are under 46000 orders. Each is tried against
        every plausible null count, since the null count shifts the intermediate
        length and a wrong count decodes to garbage.
        """
        letters = only_letters(text)
        if len(letters) < 20:
            return []
        deadline = (time.monotonic() + timeout) if timeout else None
        max_width = min(int(opts.get("max_width", 7)), BRUTE_MAX_WIDTH)
        max_nulls = int(opts.get("max_nulls", 5))

        candidates: list[Candidate] = []
        for width in range(2, max_width + 1):
            if deadline and time.monotonic() > deadline:
                break
            if width > len(letters):
                break
            for order in permutations(range(width)):
                if deadline and time.monotonic() > deadline:
                    break
                order_key = ",".join(str(i) for i in order)
                best = None
                for nulls in range(max_nulls + 1):
                    if nulls >= len(letters):
                        break
                    try:
                        plain = self.decode(letters, f"{order_key}/{nulls}")
                    except ValueError:
                        continue
                    score = scorer.score(plain)
                    if best is None or score > best[0]:
                        best = (score, nulls, plain)
                if best is None:
                    continue
                score, nulls, plain = best
                candidates.append(
                    Candidate(
                        plaintext=plain,
                        cipher=self.name,
                        key=f"{order_key}/{nulls}",
                        score=score,
                        confidence=scorer.confidence(plain),
                        meta={"width": width, "order": list(order), "nulls": nulls},
                    )
                )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
