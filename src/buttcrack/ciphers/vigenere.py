"""Vigenere cipher: periodic shift by a repeating keyword. C = (P + K) mod 26."""

from __future__ import annotations

from ._periodic import PeriodicCipher


class Vigenere(PeriodicCipher):
    name = "vigenere"
    aliases = ("vig",)
    description = "Polyalphabetic shift by a repeating keyword; C = (P + K) mod 26."
    key_format = "keyword (letters)"
    key_example = "lemon"
    complexity = 3

    def _enc(self, shift: int, p: int) -> int:
        return p + shift

    def _dec(self, shift: int, c: int) -> int:
        return c - shift
