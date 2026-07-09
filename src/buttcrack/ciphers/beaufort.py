"""Beaufort cipher: reciprocal polyalphabetic. C = (K - P) mod 26 (encode == decode)."""

from __future__ import annotations

from ._periodic import PeriodicCipher


class Beaufort(PeriodicCipher):
    name = "beaufort"
    description = "Reciprocal polyalphabetic; C = (K - P) mod 26 (encode == decode)."
    key_format = "keyword (letters)"
    key_example = "fortify"
    complexity = 3

    def _enc(self, shift: int, p: int) -> int:
        return shift - p

    def _dec(self, shift: int, c: int) -> int:
        return shift - c
