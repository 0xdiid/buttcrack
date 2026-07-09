"""Gronsfeld cipher: Vigenere with a numeric key (digits 0-9). C = (P + d) mod 26."""

from __future__ import annotations

from collections.abc import Sequence

from ._periodic import PeriodicCipher


class Gronsfeld(PeriodicCipher):
    name = "gronsfeld"
    description = "Vigenere with a digit key (0-9); C = (P + digit) mod 26."
    key_format = "digits (0-9); period = number of digits"
    key_example = "31415"
    complexity = 2
    allowed_shifts = range(10)

    def _enc(self, shift: int, p: int) -> int:
        return p + shift

    def _dec(self, shift: int, c: int) -> int:
        return c - shift

    def _key_to_shifts(self, key: str) -> list[int]:
        shifts = [int(ch) for ch in str(key) if ch.isdigit()]
        if not shifts:
            raise ValueError("gronsfeld key must contain digits 0-9")
        return shifts

    def _shifts_to_key(self, shifts: Sequence[int]) -> str:
        return "".join(str(s) for s in shifts)
