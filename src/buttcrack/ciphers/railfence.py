"""Rail fence (zigzag) transposition cipher."""

from __future__ import annotations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher


def _pattern(n: int, rails: int) -> list[int]:
    """Row index for each of the n positions following the zigzag."""
    if rails < 2:
        return [0] * n
    pattern = []
    row, step = 0, 1
    for _ in range(n):
        pattern.append(row)
        if row == 0:
            step = 1
        elif row == rails - 1:
            step = -1
        row += step
    return pattern


def _encode_letters(letters: str, rails: int) -> str:
    rows: list[list[str]] = [[] for _ in range(max(rails, 1))]
    for ch, r in zip(letters, _pattern(len(letters), rails), strict=True):
        rows[r].append(ch)
    return "".join("".join(r) for r in rows)


def _decode_letters(cipher: str, rails: int) -> str:
    n = len(cipher)
    pattern = _pattern(n, rails)
    counts = [pattern.count(r) for r in range(max(rails, 1))]
    rows, idx = [], 0
    for c in counts:
        rows.append(list(cipher[idx : idx + c]))
        idx += c
    pos = [0] * len(rows)
    out = []
    for r in pattern:
        out.append(rows[r][pos[r]])
        pos[r] += 1
    return "".join(out)


class RailFence(Cipher):
    name = "railfence"
    aliases = ("rail", "zigzag")
    description = "Zigzag transposition across N rails; key is the rail count."
    key_format = "rail count (integer >= 2)"
    key_example = "3"
    complexity = 2

    # Transposition only reorders letters, so it cannot preserve word spacing;
    # encode/decode operate on a clean uppercase letter stream (no reflow, which
    # would leak the plaintext's word lengths into the ciphertext).
    def encode(self, text: str, key: str) -> str:
        return _encode_letters(only_letters(text), int(key))

    def decode(self, text: str, key: str) -> str:
        return _decode_letters(only_letters(text), int(key))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 3:
            return []
        max_rails = int(opts.get("max_rails", min(len(letters) - 1, 12)))
        candidates: list[Candidate] = []
        for rails in range(2, max_rails + 1):
            plain = _decode_letters(letters, rails)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=str(rails),
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"rails": rails},
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
