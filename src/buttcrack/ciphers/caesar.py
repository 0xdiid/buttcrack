"""Caesar (shift) cipher and the ROT13 special case."""

from __future__ import annotations

from ..result import Candidate
from ..scoring import NgramScorer, chi_squared
from ..text import only_letters, reflow
from .base import Cipher


def _parse_shift(key) -> int:
    s = str(key).strip()
    if len(s) == 1 and s.isalpha():
        return ord(s.upper()) - 65
    return int(s) % 26


def _shift(text: str, amount: int) -> str:
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + amount) % 26 + 65))
        elif "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + amount) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)


class Caesar(Cipher):
    name = "caesar"
    aliases = ("shift", "rot")
    description = "Monoalphabetic shift by a fixed amount (0-25)."
    key_format = "shift 0-25 or a single letter A-Z"
    key_example = "7"
    complexity = 1

    def encode(self, text: str, key: str) -> str:
        return _shift(text, _parse_shift(key))

    def decode(self, text: str, key: str) -> str:
        return _shift(text, -_parse_shift(key))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if not letters:
            return []
        candidates: list[Candidate] = []
        for shift in range(26):
            plain = _shift(letters, -shift)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=str(shift),
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"shift": shift, "chi2": round(chi_squared(plain), 2)},
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]


class Rot13(Cipher):
    name = "rot13"
    description = "Caesar with a fixed shift of 13 (self-inverse)."
    key_format = "(none)"
    key_example = ""
    needs_key = False
    complexity = 0

    def encode(self, text: str, key: str = "13") -> str:
        return _shift(text, 13)

    def decode(self, text: str, key: str = "13") -> str:
        return _shift(text, 13)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if not letters:
            return []
        plain = _shift(letters, 13)
        return [
            Candidate(
                plaintext=reflow(text, plain),
                cipher=self.name,
                key="13",
                score=scorer.score(plain),
                confidence=scorer.confidence(plain),
            )
        ]
