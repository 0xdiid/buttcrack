"""Morbit cipher: a two-tier (digraph) fractionation of International Morse.

Morbit ("MORse BIT") is the ACA cipher that sits between Morse and Fractionated
Morse. Plaintext is converted to a ``.-x`` Morse stream (``x`` between letters,
``xx`` between words), padded to an even length with a trailing ``x``, then split
into consecutive PAIRS of symbols. There are exactly nine ordered pairs over
``{'.', '-', 'x'}`` and each is replaced by one of the digits 1-9. The result is
a digit string (conventionally grouped in fives).

KEY format
----------
A 9-letter keyword. Its letters are numbered 1..9 in strict alphabetical order
(ties broken left-to-right); that number sequence is the digit assigned, in the
canonical pair order below, to each of the nine Morse pairs::

    pos  1    2    3    4    5    6    7    8    9
    pair '..' '.-' '.x' '-.' '--' '-x' 'x.' 'x-' 'xx'

For example ``MORSECODE`` -> sorted C,D,E,E,M,O,O,R,S -> the keyword letters get
ranks M=5 O=6 R=8 S=9 E=3 C=1 O=7 D=2 E=4, so the nine pairs map to the digits
``5 6 8 9 3 1 7 2 4``.

A direct key is also accepted: any permutation of the digits ``1``-``9`` (as a
9-character string), interpreted as the digit for each pair-position in order.
"""

from __future__ import annotations

import itertools
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher
from .morse import FROM_MORSE, morse_to_text, text_to_morse

# Canonical fixed order of the nine Morse-symbol pairs (dCode / ACA convention).
PAIR_ORDER: tuple[str, ...] = ("..", ".-", ".x", "-.", "--", "-x", "x.", "x-", "xx")


def _keyword_to_digits(key: str) -> list[str]:
    """Map a 9-letter keyword (or a 1-9 digit permutation) to the per-pair digits.

    Returns a list of nine single-character digits aligned with ``PAIR_ORDER``.
    """
    raw = str(key).strip()
    letters = [c for c in raw.upper() if c.isalpha()]
    digits = [c for c in raw if c.isdigit()]

    # Direct key: a permutation of the digits 1-9.
    if not letters and len(digits) == 9 and sorted(digits) == list("123456789"):
        return digits

    if len(letters) != 9:
        raise ValueError("Morbit key must be a 9-letter keyword or a 1-9 digit permutation")
    # Rank the keyword's letters 1..9 alphabetically, ties broken left-to-right.
    ordered = sorted(range(9), key=lambda i: (letters[i], i))
    ranks = [0] * 9
    for rank, idx in enumerate(ordered, start=1):
        ranks[idx] = rank
    return [str(r) for r in ranks]


class Morbit(Cipher):
    """ACA Morbit: pair-wise fractionation of a Morse ``.-x`` stream into digits."""

    name = "morbit"
    aliases = ("morse-bit",)
    description = "Morbit: digraph fractionation of Morse into digits 1-9 via a 9-letter keyword."
    key_format = "9-letter keyword (or a permutation of digits 1-9)"
    key_example = "MORSECODE"
    needs_key = True
    complexity = 5

    # -- transforms ------------------------------------------------------
    def encode(self, text: str, key: str) -> str:
        pair_to_digit = dict(zip(PAIR_ORDER, _keyword_to_digits(key), strict=True))
        stream = text_to_morse(text)
        if not stream:
            return ""
        if len(stream) % 2 == 1:
            stream += "x"
        out = [pair_to_digit[stream[i : i + 2]] for i in range(0, len(stream), 2)]
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        digit_to_pair = {d: p for p, d in zip(PAIR_ORDER, _keyword_to_digits(key), strict=True)}
        digits = [c for c in str(text) if c.isdigit()]
        stream = "".join(digit_to_pair[d] for d in digits if d in digit_to_pair)
        return self._stream_to_text(stream)

    @staticmethod
    def _stream_to_text(stream: str) -> str:
        # Trim any padding 'x' the encoder appended, then translate.
        stream = stream.strip("x")
        if not stream:
            return ""
        return morse_to_text(stream)

    # -- crack -----------------------------------------------------------
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

        The keyspace is the assignment of the nine pair-positions to the distinct
        digits seen in the ciphertext. When few distinct digits appear (k <= 8)
        the number of injective assignments is small enough to brute-force within
        the timeout; otherwise we sample permutations until the deadline. Each
        candidate map is expanded to a Morse stream, rejected if it yields illegal
        Morse (e.g. ``xxx`` or un-decodable groups), and scored with the n-gram
        fitness of the recovered text.
        """
        digits = [c for c in str(text) if c.isdigit()]
        if not digits:
            return []
        distinct = sorted(set(digits))
        k = len(distinct)
        # Each distinct ciphertext digit must come from exactly one pair-position;
        # we search injective maps {digit -> pair}. Pairs not used by any digit
        # are irrelevant. So we choose an ordered k-subset of the 9 pairs.
        deadline = None if timeout is None else time.monotonic() + timeout

        seen_plain: set[str] = set()
        candidates: list[Candidate] = []

        def try_map(digit_to_pair: dict[str, str]) -> None:
            stream = "".join(digit_to_pair[d] for d in digits)
            # ACA convention forbids three x's in a row in the plaintext Morse.
            if "xxx" in stream:
                return
            trimmed = stream.strip("x")
            if not trimmed:
                return
            # Every Morse group between separators must be a legal letter/digit;
            # otherwise the map is wrong (morse_to_text silently drops junk).
            for word in trimmed.split("xx"):
                if not word:
                    return  # came from xxx run -> 3+ word separators
                for code in word.split("x"):
                    if code not in FROM_MORSE:
                        return
            plain = morse_to_text(trimmed)
            if not plain or plain in seen_plain:
                return
            seen_plain.add(plain)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key="".join(f"{d}={p}" for d, p in sorted(digit_to_pair.items())),
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"distinct_digits": k},
                )
            )

        def out_of_time() -> bool:
            return deadline is not None and time.monotonic() > deadline

        # Exhaustive over injective digit->pair maps when tractable.
        # Number of ordered k-subsets of 9 pairs = 9!/(9-k)! ; <= 9! = 362880.
        pair_perms = itertools.permutations(PAIR_ORDER, k)
        if k <= 8:
            for assignment in pair_perms:
                if out_of_time():
                    break
                try_map(dict(zip(distinct, assignment, strict=False)))
        else:  # k == 9: full 9! permutations; bounded by timeout.
            for assignment in pair_perms:
                if out_of_time():
                    break
                try_map(dict(zip(distinct, assignment, strict=False)))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
