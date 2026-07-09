"""Text utilities — buttcrack's answer to CryptoCrack's Convert Text / Format Text.

Pure, reversible string transforms an agent can use to prep ciphertext: convert
letters to/from A1Z26 numbers, regroup into blocks, strip/normalize whitespace.
"""

from __future__ import annotations

import re

from .text import only_letters


def to_numbers(text: str, *, pair: bool = False, divider: str = " ") -> str:
    """Letters -> A1Z26 numbers (1..26). ``pair`` zero-pads to two digits (01..26)."""
    nums = [f"{ord(ch) - 64:02d}" if pair else str(ord(ch) - 64) for ch in only_letters(text)]
    return divider.join(nums)


def from_numbers(text: str) -> str:
    """Numbers (1..26) -> letters. Ignores any value outside 1..26."""
    return "".join(chr(64 + int(n)) for n in re.findall(r"\d+", text) if 1 <= int(n) <= 26)


def group(text: str, size: int = 5, *, letters_only: bool = False, sep: str = " ") -> str:
    """Regroup into fixed-size blocks separated by ``sep`` (whitespace collapsed)."""
    if size < 1:
        raise ValueError("group size must be >= 1")
    body = only_letters(text) if letters_only else "".join(text.split())
    return sep.join(body[i : i + size] for i in range(0, len(body), size))


def strip_whitespace(text: str) -> str:
    return "".join(text.split())


def convert(text: str, target: str, *, divider: str = " ", group_size: int | None = None) -> str:
    """Dispatch for ``butt convert``: target in {numbers, pairs, letters}."""
    if target == "numbers":
        out = to_numbers(text, pair=False, divider=divider)
    elif target == "pairs":
        out = to_numbers(text, pair=True, divider=divider)
    elif target == "letters":
        out = from_numbers(text)
        if group_size:
            out = group(out, group_size)
    else:
        raise ValueError(f"unknown convert target {target!r} (use numbers|pairs|letters)")
    return out
