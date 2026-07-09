"""Text normalization helpers shared across ciphers and scoring."""

from __future__ import annotations

import string

ALPHABET = string.ascii_uppercase
A_ORD = ord("A")


def only_letters(text: str) -> str:
    """Uppercase, keeping only A-Z (drops spaces, punctuation, digits)."""
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def to_indices(letters: str) -> list[int]:
    """Map an A-Z string to 0-25 indices."""
    return [ord(ch) - A_ORD for ch in letters]


def from_indices(indices) -> str:
    """Map 0-25 indices back to an A-Z string."""
    return "".join(chr(A_ORD + (i % 26)) for i in indices)


def reflow(template: str, letters: str) -> str:
    """Re-insert non-letter characters from ``template`` around ``letters``.

    Walks ``template`` and, for every alphabetic position, consumes the next
    character from ``letters`` (preserving the template's original case),
    leaving spaces and punctuation untouched. Lets a cipher operate on the
    letters-only stream yet return output that lines up with the input layout.
    """
    out = []
    it = iter(letters)
    for ch in template:
        if ch.isalpha():
            try:
                nxt = next(it)
            except StopIteration:
                break
            out.append(nxt.lower() if ch.islower() else nxt.upper())
        else:
            out.append(ch)
    return "".join(out)
