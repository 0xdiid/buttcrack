"""Della Porta cipher: reciprocal periodic polyalphabetic over 13 tables.

Each key letter selects one of 13 reciprocal alphabets by its PAIR
(A/B -> table 0, C/D -> table 1, ... Y/Z -> table 12). Within a table the first
half A-M and the rotated second half N-Z swap, so encrypt == decrypt and no
letter ever maps to itself.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..text import only_letters
from ._periodic import PeriodicCipher


def _porta(t: int, x: int) -> int:
    """Reciprocal Porta map for table ``t`` (0-12) and letter index ``x`` (0-25)."""
    if x < 13:
        return 13 + (x + t) % 13
    return (x - 13 - t) % 13


class Porta(PeriodicCipher):
    name = "porta"
    description = "Reciprocal periodic polyalphabetic over 13 tables (encode == decode)."
    key_format = "keyword (letters); period = keyword length"
    key_example = "PORTA"
    complexity = 3
    allowed_shifts = range(13)

    def _enc(self, shift: int, p: int) -> int:
        return _porta(shift, p)

    def _dec(self, shift: int, c: int) -> int:
        return _porta(shift, c)

    def _key_to_shifts(self, key: str) -> list[int]:
        # Each key letter selects a table by its pair: t = (letter) // 2.
        shifts = [(ord(c) - 65) // 2 for c in only_letters(key)]
        if not shifts:
            raise ValueError("porta key must contain letters")
        return shifts

    def _shifts_to_key(self, shifts: Sequence[int]) -> str:
        # Canonical keyword: first letter of each pair (table t -> 'A'+2t).
        return "".join(chr(65 + 2 * t) for t in shifts)
