"""Atbash — reverse-alphabet substitution (A<->Z). Keyless and self-inverse."""

from __future__ import annotations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher


def _atbash(text: str) -> str:
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(90 - (ord(ch) - 65)))
        elif "a" <= ch <= "z":
            out.append(chr(122 - (ord(ch) - 97)))
        else:
            out.append(ch)
    return "".join(out)


class Atbash(Cipher):
    name = "atbash"
    description = "Reverse-alphabet substitution (A<->Z); keyless, self-inverse."
    key_format = "(none)"
    key_example = ""
    needs_key = False
    complexity = 0

    def encode(self, text: str, key: str = "") -> str:
        return _atbash(text)

    def decode(self, text: str, key: str = "") -> str:
        return _atbash(text)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if not letters:
            return []
        plain = _atbash(letters)
        return [
            Candidate(
                plaintext=reflow(text, plain),
                cipher=self.name,
                key=None,
                score=scorer.score(plain),
                confidence=scorer.confidence(plain),
            )
        ]
