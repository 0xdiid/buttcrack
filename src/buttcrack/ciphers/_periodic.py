"""Shared solver for periodic polyalphabetic ciphers (Vigenere family).

Each such cipher is fully described by a per-letter decrypt ``dec(shift, c) -> p``
and the set of allowed shifts per key position. Key recovery is then identical
across the whole family: seed each period column by chi-squared, then hill-climb
the shifts against the full-text quadgram score (cross-column context is what
recovers short keys that per-column analysis alone misses).

``shift`` and the letter args are 0-25 ints; callers do their own A-Z mapping.
The helpers take a raw ``dec``/``fn`` returning an int and apply ``% 26``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..result import Candidate
from ..scoring import NgramScorer, chi_squared
from ..text import only_letters, reflow
from .base import Cipher

# (shift, letter_index) -> result_index (mod 26 applied by the helpers)
LetterFn = Callable[[int, int], int]


def columns(letters: str, length: int) -> list[str]:
    cols: list[list[str]] = [[] for _ in range(length)]
    for i, ch in enumerate(letters):
        cols[i % length].append(ch)
    return ["".join(c) for c in cols]


def _decrypt(letters: str, shifts: Sequence[int], dec: LetterFn) -> str:
    length = len(shifts)
    return "".join(
        chr(dec(shifts[i % length], ord(c) - 65) % 26 + 65) for i, c in enumerate(letters)
    )


def _seed_column(column: str, dec: LetterFn, allowed: Sequence[int]) -> int:
    """Pick the shift that makes one period-column look most like English."""
    best_shift, best_chi = allowed[0], float("inf")
    for s in allowed:
        plain = "".join(chr(dec(s, ord(c) - 65) % 26 + 65) for c in column)
        chi = chi_squared(plain)
        if chi < best_chi:
            best_chi, best_shift = chi, s
    return best_shift


def _solve_length(
    letters: str,
    length: int,
    scorer: NgramScorer,
    dec: LetterFn,
    allowed: Sequence[int],
    deadline: float | None,
) -> tuple[float, list[int], str]:
    shifts = [_seed_column(c, dec, allowed) for c in columns(letters, length)]
    best = scorer.score(_decrypt(letters, shifts, dec))
    improved = True
    while improved:
        improved = False
        for pos in range(length):
            if deadline and time.monotonic() > deadline:
                return best, shifts, _decrypt(letters, shifts, dec)
            best_s = shifts[pos]
            for s in allowed:
                if s == best_s:
                    continue
                shifts[pos] = s
                sc = scorer.score(_decrypt(letters, shifts, dec))
                if sc > best:
                    best, best_s, improved = sc, s, True
            shifts[pos] = best_s
    return best, shifts, _decrypt(letters, shifts, dec)


def solve_periodic(
    letters: str,
    scorer: NgramScorer,
    dec: LetterFn,
    *,
    max_len: int,
    forced: int | None = None,
    deadline: float | None = None,
    allowed: Sequence[int] = range(26),
) -> dict[str, tuple[float, list[int], int]]:
    """Return ``plaintext -> (score, shifts, key_length)``.

    Solves every key length (cheap), dedups by resulting plaintext keeping the
    SHORTEST key — so a length-5 key and its length-10 repeat collapse to the
    shorter, and the true length is never missed when IoC ranking is noisy.
    """
    allowed = list(allowed)
    lengths = [int(forced)] if forced else range(1, max_len + 1)
    by_plain: dict[str, tuple[float, list[int], int]] = {}
    for length in lengths:
        if length < 1:
            continue
        if deadline and time.monotonic() > deadline:
            break
        score, shifts, plain = _solve_length(letters, length, scorer, dec, allowed, deadline)
        prev = by_plain.get(plain)
        if prev is None or length < prev[2]:
            by_plain[plain] = (score, shifts, length)
    return by_plain


def stream(text: str, shifts: Sequence[int], fn: LetterFn) -> str:
    """Apply ``fn(shift, x)`` across the letters of ``text``.

    Preserves case and passes non-letters through unchanged.
    """
    length = len(shifts)
    out = []
    j = 0
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(fn(shifts[j % length], ord(ch) - 65) % 26 + 65))
            j += 1
        elif "a" <= ch <= "z":
            out.append(chr(fn(shifts[j % length], ord(ch) - 97) % 26 + 97))
            j += 1
        else:
            out.append(ch)
    return "".join(out)


class PeriodicCipher(Cipher):
    """Base for the Vigenere family.

    A subclass supplies only its two letter equations ``_enc(shift, p)`` and
    ``_dec(shift, c)`` (raw ints; ``% 26`` applied downstream). Keyword-keyed
    ciphers get key parsing for free; numeric-keyed ones override it.
    """

    #: shifts each key position may take (override e.g. range(10) for Gronsfeld)
    allowed_shifts: Sequence[int] = range(26)

    def _enc(self, shift: int, p: int) -> int:  # pragma: no cover - abstract-ish
        raise NotImplementedError

    def _dec(self, shift: int, c: int) -> int:  # pragma: no cover - abstract-ish
        raise NotImplementedError

    def _key_to_shifts(self, key: str) -> list[int]:
        shifts = [ord(c) - 65 for c in only_letters(key)]
        if not shifts:
            raise ValueError(f"{self.name} key must contain letters")
        return shifts

    def _shifts_to_key(self, shifts: Sequence[int]) -> str:
        return "".join(chr(s + 65) for s in shifts)

    def encode(self, text: str, key: str) -> str:
        return stream(text, self._key_to_shifts(key), self._enc)

    def decode(self, text: str, key: str) -> str:
        return stream(text, self._key_to_shifts(key), self._dec)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 4:
            return []
        max_len = int(opts.get("max_key_length", min(20, len(letters) // 2)))
        forced = opts.get("key_length")
        deadline = (time.monotonic() + timeout) if timeout else None

        by_plain = solve_periodic(
            letters,
            scorer,
            self._dec,
            max_len=max_len,
            forced=forced,
            deadline=deadline,
            allowed=self.allowed_shifts,
        )
        candidates = [
            Candidate(
                plaintext=reflow(text, plain),
                cipher=self.name,
                key=self._shifts_to_key(shifts),
                score=score,
                confidence=scorer.confidence(plain),
                meta={"key_length": length},
            )
            for plain, (score, shifts, length) in by_plain.items()
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
