"""Affine cipher: E(x) = (a*x + b) mod 26, with gcd(a, 26) = 1."""

from __future__ import annotations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

# a values coprime with 26
_COPRIMES = (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)
_A_INV = {a: pow(a, -1, 26) for a in _COPRIMES}


def _parse_key(key) -> tuple[int, int]:
    parts = str(key).replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError("affine key must be 'a,b' (e.g. '5,8')")
    a, b = int(parts[0]) % 26, int(parts[1]) % 26
    if a not in _A_INV:
        raise ValueError(f"affine 'a' must be coprime with 26; got {a}")
    return a, b


def _apply(text: str, fn) -> str:
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(fn(ord(ch) - 65) % 26 + 65))
        elif "a" <= ch <= "z":
            out.append(chr(fn(ord(ch) - 97) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)


class Affine(Cipher):
    name = "affine"
    description = "E(x) = (a*x + b) mod 26 with gcd(a,26)=1; key is 'a,b'."
    key_format = "a,b with a coprime to 26"
    key_example = "5,8"
    complexity = 2

    def encode(self, text: str, key: str) -> str:
        a, b = _parse_key(key)
        return _apply(text, lambda x: a * x + b)

    def decode(self, text: str, key: str) -> str:
        a, b = _parse_key(key)
        ai = _A_INV[a]
        return _apply(text, lambda y: ai * (y - b))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if not letters:
            return []
        candidates: list[Candidate] = []
        for a in _COPRIMES:
            ai = _A_INV[a]
            for b in range(26):
                plain = _apply(letters, lambda y, ai=ai, b=b: ai * (y - b))
                candidates.append(
                    Candidate(
                        plaintext=reflow(text, plain),
                        cipher=self.name,
                        key=f"{a},{b}",
                        score=scorer.score(plain),
                        confidence=scorer.confidence(plain),
                        meta={"a": a, "b": b},
                    )
                )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
