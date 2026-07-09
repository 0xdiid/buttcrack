"""Nihilist substitution cipher.

A polyalphabetic cipher built on a keyed 5x5 Polybius square plus a repeating
numeric *additive* key.

Algorithm
---------
SETUP
    Build a 5x5 :class:`~buttcrack.ciphers.squares.PolybiusSquare` from a
    keyword (mixed alphabet, J merged into I, 25-letter alphabet). Rows and
    columns are labelled ``1..5``; each letter's value is the two-digit number
    ``(row)(col)``, **row first** (e.g. ZEBRAS square: ``D`` -> ``23``).

ENCRYPT
    1. Convert each plaintext letter to its two-digit square coordinate.
    2. Convert each letter of a *second* keyword (the additive key) to a
       two-digit coordinate using the **same** square; repeat this key sequence
       cyclically to the plaintext length.
    3. ``ciphertext_number = plaintext_number + key_number`` (ordinary decimal
       addition, *not* digit-wise mod). Sums range roughly 22..110 and are
       printed as 2- or 3-digit numbers separated by spaces.

DECRYPT
    Subtract the repeating key number from each ciphertext number; the result is
    a two-digit number whose tens digit is the row and units digit the column.

Because each plaintext/key coordinate digit lies in ``1..5``, the tens and units
of every coordinate group lie in ``2..10`` after addition, which lets an
adversary segment a run-together digit stream uniquely.

KEY FORMAT
----------
``"SQUAREKEY/ADDITIVEKEY"`` -- two keywords separated by ``/``. The first builds
the mixed 5x5 square; the second is the repeating additive key (also enciphered
through the same square). The period equals the length of the additive keyword
(after J->I merge / non-letter stripping).

Encrypt is **not** the same as decrypt (not reciprocal).

Reference: Wikipedia, "Nihilist cipher" -- worked example with square keyword
``ZEBRAS`` and additive key ``RUSSIAN`` enciphering ``DYNAMITE WINTER PALACE``.
"""

from __future__ import annotations

import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .squares import PolybiusSquare


def _split_key(key: str) -> tuple[str, str]:
    """Split ``"SQUAREKEY/ADDITIVEKEY"`` into its two keywords."""
    s = str(key)
    if "/" not in s:
        raise ValueError(
            "nihilist-substitution key must be 'SQUAREKEY/ADDITIVEKEY' (slash-separated)"
        )
    square_kw, _, additive_kw = s.partition("/")
    if not only_letters(additive_kw):
        raise ValueError("nihilist-substitution additive keyword must contain letters")
    return square_kw, additive_kw


def _coord(square: PolybiusSquare, letter: str) -> int:
    """Two-digit (row)(col) value, 1-indexed, row first."""
    r, c = square.rc(letter)
    return (r + 1) * 10 + (c + 1)


def _key_numbers(square: PolybiusSquare, additive_kw: str) -> list[int]:
    letters = square.prepare(additive_kw)
    return [_coord(square, ch) for ch in letters]


class NihilistSubstitution(Cipher):
    """Keyed Polybius square + repeating numeric additive (Nihilist substitution)."""

    name = "nihilist-substitution"
    aliases = ("nihilist-sub", "nihsub")
    description = (
        "Polybius-square coordinates added to a repeating numeric key built from "
        "a second keyword; output is space-separated 2-3 digit numbers."
    )
    key_format = (
        "squarekeyword/additivekeyword (square keyword builds 5x5; additive is the repeating key)"
    )
    key_example = "ZEBRAS/RUSSIAN"
    needs_key = True
    complexity = 4

    def encode(self, text: str, key: str) -> str:
        square_kw, additive_kw = _split_key(key)
        square = PolybiusSquare(square_kw)
        keynums = _key_numbers(square, additive_kw)
        if not keynums:
            raise ValueError("additive keyword produced no key numbers")
        letters = square.prepare(text)
        out = []
        for i, ch in enumerate(letters):
            out.append(str(_coord(square, ch) + keynums[i % len(keynums)]))
        return " ".join(out)

    def decode(self, text: str, key: str) -> str:
        square_kw, additive_kw = _split_key(key)
        square = PolybiusSquare(square_kw)
        keynums = _key_numbers(square, additive_kw)
        if not keynums:
            raise ValueError("additive keyword produced no key numbers")
        groups = [int(tok) for tok in str(text).split() if tok.strip()]
        out = []
        for i, num in enumerate(groups):
            coord = num - keynums[i % len(keynums)]
            row, col = divmod(coord, 10)
            if not (1 <= row <= 5 and 1 <= col <= 5):
                # Bad key/ciphertext for this position; emit a placeholder.
                out.append("?")
                continue
            out.append(square.at(row - 1, col - 1))
        return "".join(out)

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
        """Best-effort keyless crack.

        The full problem -- recover *both* an unknown mixed 5x5 square and an
        unknown additive keyword from coordinate sums -- is a hard joint search
        (a Polybius substitution layered under a polyalphabetic additive). We
        attempt only the tractable slice: assume the **standard** A..Z (J->I)
        square and try to recover a short additive key.

        For each candidate period ``p`` (1..MAXP), each of the ``p`` columns is
        an independent additive on a coordinate; the key number for a column is
        a square coordinate (two digits, each 1..5 -> 25 possibilities). We pick,
        per column, the key value whose recovered letters best fit English
        monogram frequencies (chi-squared), decode the whole message, and rank by
        the quadgram score. If nothing scores like English we return ``[]``.
        """
        from ..scoring import chi_squared, get_scorer

        groups = [int(tok) for tok in str(text).split() if tok.strip().isdigit()]
        if len(groups) < 8:  # not a numeric nihilist ciphertext
            return []

        scorer = scorer or get_scorer()
        square = PolybiusSquare("")  # standard A..Z, J->I
        deadline = (time.monotonic() + timeout) if timeout else None

        # Pre-compute, for each ciphertext group, which letter results from each
        # of the 25 possible key coordinates (None when out of range).
        key_values = [10 * r + c for r in range(1, 6) for c in range(1, 6)]  # 11..55, digits 1..5
        letter_for: list[dict[int, str]] = []
        for num in groups:
            row_map: dict[int, str] = {}
            for k in key_values:
                row, col = divmod(num - k, 10)
                if 1 <= row <= 5 and 1 <= col <= 5:
                    row_map[k] = square.at(row - 1, col - 1)
            letter_for.append(row_map)

        def decode_with(chosen: list[int], p: int) -> str:
            out = []
            for i, _ in enumerate(groups):
                ch = letter_for[i].get(chosen[i % p])
                out.append(ch if ch else "?")
            return "".join(out)

        max_p = int(opts.get("max_period", 12))
        candidates: list[Candidate] = []
        seen_keys: set[str] = set()

        for p in range(1, max_p + 1):
            if deadline and time.monotonic() > deadline:
                break
            if p > len(groups):
                break
            chosen: list[int] = []
            ok = True
            for col in range(p):
                idxs = range(col, len(groups), p)
                best_k = None
                best_chi = float("inf")
                for k in key_values:
                    letters = [letter_for[i].get(k) for i in idxs]
                    if any(ch is None for ch in letters):
                        continue  # this key value puts some group out of square
                    chi = chi_squared("".join(letters))  # type: ignore[arg-type]
                    if chi < best_chi:
                        best_chi, best_k = chi, k
                if best_k is None:
                    ok = False
                    break
                chosen.append(best_k)
            if not ok:
                continue
            plain = decode_with(chosen, p)
            key_repr = "STANDARD/" + "-".join(str(k) for k in chosen)
            if key_repr in seen_keys:
                continue
            seen_keys.add(key_repr)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=key_repr,
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"period": p, "square": "standard", "key_coords": chosen},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
