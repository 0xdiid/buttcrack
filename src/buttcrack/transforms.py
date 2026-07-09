"""Pre-flight transforms — undo format tricks before cipher analysis.

A "cipher" that resists every solver is often just *wrapped*: the letters are
reversed, padded with nulls, or the whole thing is a base64/hex/A1Z26 encoding of
the real ciphertext. These are cheap, mostly-deterministic un-wraps to try first.

Detection here is deliberately conservative — a plain uppercase A-Z ciphertext is
valid base64 and could pass as hex, so an encoding is only reported when the input
carries a signal a letter cipher wouldn't (digits, lowercase, base64 padding) and
decodes to plausible text. ``butt transform`` reports candidates; ``auto`` only
auto-applies the high-confidence ones.
"""

from __future__ import annotations

import base64
import binascii
import re

from .text import only_letters

_PRINTABLE = set(range(32, 127))


def reverse(text: str) -> str:
    """The letter stream, reversed (undoes plain reversal / a R-to-L write)."""
    return only_letters(text)[::-1]


def decimate(text: str, period: int, offset: int) -> str:
    """Drop every ``period``-th letter at ``offset`` (strip a regular null pattern)."""
    letters = only_letters(text)
    return "".join(c for i, c in enumerate(letters) if i % period != offset % period)


def _looks_like_text(b: bytes) -> bool:
    """True if bytes are mostly printable ASCII with a healthy share of letters."""
    if not b:
        return False
    printable = sum(1 for x in b if x in _PRINTABLE)
    alpha = sum(1 for x in b if chr(x).isalpha())
    return printable / len(b) > 0.9 and alpha / len(b) > 0.6


def detect_encoding(raw: str) -> list[dict]:
    """High-confidence nested encodings the input might be, decoded.

    Returns ``[{"kind", "decoded"}]`` — base64 / hex / A1Z26. Empty when the input
    is just letters (no encoding signal), to avoid mis-peeling real ciphertext.
    """
    s = raw.strip()
    out: list[dict] = []
    compact = re.sub(r"\s+", "", s)

    # base64: needs a non-A-Z signal (lowercase / digit / + / / / =) so a plain
    # uppercase ciphertext isn't mistaken for it.
    if (
        len(compact) >= 8
        and len(compact) % 4 == 0
        and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact)
        and re.search(r"[a-z0-9+/=]", compact)
    ):
        try:
            dec = base64.b64decode(compact, validate=True)
            if _looks_like_text(dec):
                out.append({"kind": "base64", "decoded": dec.decode("ascii", "replace")})
        except (binascii.Error, ValueError):
            pass

    # hex: needs at least one digit (pure A-F could be a real cipher).
    if len(compact) >= 8 and len(compact) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", compact):
        if re.search(r"[0-9]", compact):
            try:
                dec = bytes.fromhex(compact)
                if _looks_like_text(dec):
                    out.append({"kind": "hex", "decoded": dec.decode("ascii", "replace")})
            except ValueError:
                pass

    # A1Z26: separated numbers, all in 1..26 -> letters.
    nums = re.findall(r"\d+", s)
    if len(nums) >= 4 and re.fullmatch(r"[\d\s,.\-/]+", s) and all(1 <= int(x) <= 26 for x in nums):
        out.append({"kind": "a1z26", "decoded": "".join(chr(64 + int(x)) for x in nums)})

    return out


def candidates(raw: str) -> list[dict]:
    """All pre-flight transform candidates for ``raw`` (for ``butt transform``)."""
    out: list[dict] = [{"kind": "reverse", "text": reverse(raw)}]
    for enc in detect_encoding(raw):
        out.append({"kind": enc["kind"], "text": enc["decoded"]})
    return out
