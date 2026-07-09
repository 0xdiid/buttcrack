"""Baconian (biliteral) cipher: each letter becomes a 5-symbol A/B group.

Uses the classic 24-letter table (I/J share a code, U/V share a code) in straight
binary order. Decoding/cracking reads an A/B stream in groups of five.
"""

from __future__ import annotations

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher

_ALPHA24 = "ABCDEFGHIKLMNOPQRSTUWXYZ"  # no J, no V
_ENCODE = {_ALPHA24[i]: format(i, "05b").translate(str.maketrans("01", "AB")) for i in range(24)}
_DECODE = {code: letter for letter, code in _ENCODE.items()}


def _fold(ch: str) -> str:
    ch = ch.upper()
    return {"J": "I", "V": "U"}.get(ch, ch)


def _decode_bits(bits: str) -> str:
    return "".join(_DECODE.get(bits[i : i + 5], "?") for i in range(0, len(bits) // 5 * 5, 5))


class Bacon(Cipher):
    name = "bacon"
    aliases = ("baconian", "biliteral")
    description = "Biliteral cipher: each letter -> a 5-symbol A/B group (24-letter table)."
    key_format = "(none)"
    key_example = ""
    needs_key = False
    complexity = 1

    def encode(self, text: str, key: str = "") -> str:
        groups = [_ENCODE[_fold(ch)] for ch in text if ch.isalpha()]
        return " ".join(groups)

    def decode(self, text: str, key: str = "") -> str:
        bits = "".join(c for c in text.upper() if c in "AB")
        return _decode_bits(bits)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        symbols = sorted({c for c in text.upper() if c.isalpha()})
        if len(symbols) != 2:
            return []  # a clean biliteral stream has exactly two distinct symbols
        candidates = []
        # Try both polarities (which symbol is 'A').
        for a, b in ((symbols[0], symbols[1]), (symbols[1], symbols[0])):
            bits = "".join({a: "A", b: "B"}.get(c, "") for c in text.upper() if c in (a, b))
            plain = _decode_bits(bits)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=f"{a}=A,{b}=B",
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"symbols": [a, b]},
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
