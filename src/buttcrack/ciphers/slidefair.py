"""Slidefair: an ACA periodic DIGRAPHIC cipher (Playfair pairing + a Vigenere slide).

A keyword sets the period (= its length). The plaintext letter stream is split
into consecutive DIGRAPHS (padded with a trailing ``X`` if odd). For digraph
``i`` the key letter is ``keyword[i mod period]``.

Each digraph is enciphered with an "abbreviated table": two strips, a top row of
the straight alphabet ``A..Z`` and a bottom row that depends on the chosen table
(Vigenere, Variant, or Beaufort). The first plaintext letter is found in the top
strip, the second in the bottom strip; they are opposite corners of a rectangle
and the ciphertext is the other two corners (top letter first). For the
Vigenere table this reduces to::

    C1 = (P2 - k) mod 26
    C2 = (P1 + k) mod 26

with a SAME-COLUMN special case: if the two plaintext letters sit in the same
column of the strips (``(P1 + k) mod 26 == P2``) the rectangle degenerates, and
the ciphertext is instead the pair one column to the RIGHT::

    C1 = (P1 + 1) mod 26,  C2 = (P2 + 1) mod 26

DECRYPT walks the same rectangle backwards, moving LEFT in the same-column case,
so Slidefair is NOT perfectly reciprocal (except the Beaufort table, which is).

Table variants (the bottom strip changes the slide direction):
  * Vigenere (ACA default): C1 = (P2 - k),  C2 = (P1 + k); vertical rule shifts right.
  * Variant:                C1 = (P2 + k),  C2 = (P1 - k); vertical when (P1 - k) == P2.
  * Beaufort:               C1 = (k - P2),  C2 = (k - P1); reciprocal, no vertical case.

Full 26-letter alphabet, no I/J merge.

KEY FORMAT
----------
``"KEYWORD"`` or ``"KEYWORD/TABLE"`` where TABLE is one of ``VIGENERE``
(default), ``VARIANT``, or ``BEAUFORT`` (case-insensitive; ``VIG``/``VAR``/
``BEA``/``BF`` accepted). Examples::

    "DIGRAPH"              # Vigenere table (ACA default)
    "DIGRAPH/VARIANT"
    "SECRET/BEAUFORT"
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

_TABLES = {
    "VIGENERE": "VIGENERE",
    "VIG": "VIGENERE",
    "V": "VIGENERE",
    "VARIANT": "VARIANT",
    "VAR": "VARIANT",
    "BEAUFORT": "BEAUFORT",
    "BEA": "BEAUFORT",
    "BF": "BEAUFORT",
    "B": "BEAUFORT",
}


def _parse_key(key: str) -> tuple[list[int], str]:
    """Return (key letter indices 0-25, table name) from a packed key string."""
    raw = key.strip()
    table = "VIGENERE"
    if "/" in raw:
        word_part, _, table_part = raw.partition("/")
        token = only_letters(table_part).upper()
        if token:
            if token not in _TABLES:
                raise ValueError(
                    f"slidefair table must be one of VIGENERE/VARIANT/BEAUFORT, got {table_part!r}"
                )
            table = _TABLES[token]
        raw = word_part
    letters = only_letters(raw)
    if not letters:
        raise ValueError("slidefair key must contain a keyword (letters)")
    return [ord(c) - 65 for c in letters], table


# --- per-digraph combiners (all args/results are 0-25 ints) -----------------


def _enc_pair(p1: int, p2: int, k: int, table: str) -> tuple[int, int]:
    if table == "BEAUFORT":
        return (k - p2) % 26, (k - p1) % 26
    if table == "VARIANT":
        if (p1 - k) % 26 == p2:  # same-column -> shift right
            return (p1 + 1) % 26, (p2 + 1) % 26
        return (p2 + k) % 26, (p1 - k) % 26
    # VIGENERE
    if (p1 + k) % 26 == p2:  # same-column -> shift right
        return (p1 + 1) % 26, (p2 + 1) % 26
    return (p2 - k) % 26, (p1 + k) % 26


def _dec_pair(c1: int, c2: int, k: int, table: str) -> tuple[int, int]:
    if table == "BEAUFORT":  # reciprocal
        return (k - c2) % 26, (k - c1) % 26
    if table == "VARIANT":
        p1 = (c2 + k) % 26
        p2 = (c1 - k) % 26
        if (p1 - k) % 26 == p2:  # the vertical rule fired on encrypt -> move left
            return (c1 - 1) % 26, (c2 - 1) % 26
        return p1, p2
    # VIGENERE
    p1 = (c2 - k) % 26
    p2 = (c1 + k) % 26
    if (p1 + k) % 26 == p2:  # the vertical rule fired on encrypt -> move left
        return (c1 - 1) % 26, (c2 - 1) % 26
    return p1, p2


def _transform(letters: str, key_idx: list[int], table: str, *, encrypt: bool) -> str:
    period = len(key_idx)
    s = letters
    if encrypt and len(s) % 2:
        s = s + "X"
    if not encrypt and len(s) % 2:
        s = s[:-1]
    fn = _enc_pair if encrypt else _dec_pair
    out: list[str] = []
    for d, i in enumerate(range(0, len(s), 2)):
        a, b = ord(s[i]) - 65, ord(s[i + 1]) - 65
        k = key_idx[d % period]
        x, y = fn(a, b, k, table)
        out.append(chr(x + 65))
        out.append(chr(y + 65))
    return "".join(out)


class Slidefair(Cipher):
    name = "slidefair"
    description = (
        "Periodic digraphic cipher: Playfair pairing with a Vigenere/Variant/Beaufort slide."
    )
    key_format = "keyword (optionally /TABLE: VIGENERE|VARIANT|BEAUFORT)"
    key_example = "DIGRAPH/VARIANT"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        key_idx, table = _parse_key(key)
        return _transform(only_letters(text), key_idx, table, encrypt=True)

    def decode(self, text: str, key: str) -> str:
        key_idx, table = _parse_key(key)
        return _transform(only_letters(text), key_idx, table, encrypt=False)

    def crack(
        self,
        text,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Best-effort keyless recovery.

        Strategy: for a forced/guessed period, each key position is an
        independent constant-shift digraphic cipher (only 26 key letters per
        column and 3 table types), so seed every column by the best per-column
        digraph score, then hill-climb the keyword against the full-text
        quadgram score. Without a forced period this is searched over a small
        range; long, period-aligned text recovers, short text often will not.
        """
        letters = only_letters(text)
        if len(letters) < 40:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        forced = opts.get("key_length")
        max_len = int(opts.get("max_key_length", 8))
        lengths = [int(forced)] if forced else range(2, max_len + 1)
        tables = list(opts.get("tables", ("VIGENERE", "VARIANT", "BEAUFORT")))

        def decode_with(idx: list[int], table: str) -> str:
            return _transform(letters, idx, table, encrypt=False)

        results: list[tuple[float, list[int], str]] = []
        for table in tables:
            for length in lengths:
                if length < 1:
                    continue
                if deadline and time.monotonic() > deadline:
                    break
                key_idx = self._solve_length(letters, length, table, scorer, deadline)
                score = scorer.score(decode_with(key_idx, table))
                results.append((score, key_idx, table))

        if not results:
            return []
        results.sort(key=lambda r: r[0], reverse=True)
        seen: set[str] = set()
        out: list[Candidate] = []
        for score, key_idx, table in results:
            plain = decode_with(key_idx, table)
            if plain in seen:
                continue
            seen.add(plain)
            keyword = "".join(chr(i + 65) for i in key_idx)
            out.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=f"{keyword}/{table}",
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"key_length": len(key_idx), "table": table},
                )
            )
            if len(out) >= top:
                break
        return out

    def _solve_length(
        self,
        letters: str,
        length: int,
        table: str,
        scorer: NgramScorer,
        deadline: float | None,
    ) -> list[int]:
        # Seed: pick each column's key letter by best per-column digraph score.
        key_idx = [self._seed_column(letters, length, pos, table, scorer) for pos in range(length)]
        best = scorer.score(_transform(letters, key_idx, table, encrypt=False))
        improved = True
        while improved:
            improved = False
            for pos in range(length):
                if deadline and time.monotonic() > deadline:
                    return key_idx
                cur = key_idx[pos]
                best_k = cur
                for k in range(26):
                    if k == cur:
                        continue
                    key_idx[pos] = k
                    sc = scorer.score(_transform(letters, key_idx, table, encrypt=False))
                    if sc > best:
                        best, best_k, improved = sc, k, True
                key_idx[pos] = best_k
        return key_idx

    def _seed_column(
        self,
        letters: str,
        length: int,
        pos: int,
        table: str,
        scorer: NgramScorer,
    ) -> int:
        # Digraph d uses column (d % length); score that column's decrypted
        # digraphs in isolation against the scorer to seed the key letter.
        s = letters if len(letters) % 2 == 0 else letters[:-1]
        col_pairs = [
            (ord(s[2 * d]) - 65, ord(s[2 * d + 1]) - 65)
            for d in range(len(s) // 2)
            if d % length == pos
        ]
        if not col_pairs:
            return 0
        best_k, best_sc = 0, float("-inf")
        for k in range(26):
            buf = []
            for a, b in col_pairs:
                x, y = _dec_pair(a, b, k, table)
                buf.append(chr(x + 65))
                buf.append(chr(y + 65))
            sc = scorer.score("".join(buf))
            if sc > best_sc:
                best_sc, best_k = sc, k
        return best_k
