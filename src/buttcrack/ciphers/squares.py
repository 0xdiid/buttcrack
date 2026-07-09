"""Polybius square construction shared by the square/fractionation ciphers.

A keyed mixed alphabet laid into an N×N grid. The 5×5 form merges J→I (the ACA
standard); the 6×6 form uses A-Z plus 0-9. Coordinates are 0-indexed internally;
callers add 1 for the conventional 1..5 / 1..6 display.
"""

from __future__ import annotations

ALPHABET_5 = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # no J
ALPHABET_6 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class PolybiusSquare:
    """An N×N keyed square with letter<->coordinate lookup."""

    def __init__(self, keyword: str, *, size: int = 5, alphabet: str | None = None):
        if alphabet is None:
            alphabet = ALPHABET_5 if size == 5 else ALPHABET_6
        self.size = size
        self.alphabet = alphabet
        # 5x5 alphabets omit J, so J->I on input; 6x6 keeps all 26 letters.
        self._merge = ("J", "I") if "J" not in alphabet else None

        seq: list[str] = []
        for ch in self._clean(keyword) + alphabet:
            if ch in alphabet and ch not in seq:
                seq.append(ch)
        if len(seq) != size * size:
            raise ValueError(f"square needs {size * size} cells, built {len(seq)}")
        self.grid = seq
        self._pos = {ch: (i // size, i % size) for i, ch in enumerate(seq)}

    def _clean(self, text: str) -> str:
        out = []
        for ch in text.upper():
            if self._merge and ch == self._merge[0]:
                ch = self._merge[1]
            if ch in self.alphabet:
                out.append(ch)
        return "".join(out)

    def prepare(self, text: str) -> str:
        """Letters-only, uppercased, with the square's merge applied (J->I)."""
        return self._clean(text)

    def rc(self, letter: str) -> tuple[int, int]:
        """0-indexed (row, col) of ``letter`` (applies J->I merge)."""
        ch = letter.upper()
        if self._merge and ch == self._merge[0]:
            ch = self._merge[1]
        return self._pos[ch]

    def at(self, row: int, col: int) -> str:
        """Letter at 0-indexed (row, col)."""
        return self.grid[row * self.size + col]
