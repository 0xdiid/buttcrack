"""Pollux cipher: a homophonic fractionation of International Morse.

The plaintext is first turned into a dot/dash stream with ``x`` separators
(single ``x`` between letters, ``xx`` between words) via :mod:`buttcrack.ciphers.morse`.
Each of the three morse symbols ``.`` ``-`` ``x`` is then represented by one or
more of the digits 0-9. Because several digits map to the same symbol, encryption
is *homophonic* (many-to-one) and therefore non-unique: at each step the
encipherer is free to pick any digit whose key value equals the current symbol.
Decryption is deterministic — each ciphertext digit has exactly one symbol, and
the resulting morse stream parses uniquely on ``x``/``xx``.

KEY FORMAT
----------
A 10-character string over ``{., -, x}`` indexed by the digits ``1 2 3 4 5 6 7 8 9 0``
(position 0 of the string is the value of digit ``1``; position 9 is the value of
digit ``0`` — the classic ACA layout where the columns are headed 1..9,0). Every
morse symbol must be used at least once; ACA guidance recommends at least four
digits mapping to dot. Example::

    key = ". . - x - - . x x -"   (spaces optional)
    # 1=. 2=. 3=- 4=x 5=- 6=- 7=. 8=x 9=x 0=-

Output is a digit string (the convention is to group in 5s; this implementation
emits ungrouped digits — strip/insert spaces as you like).
"""

from __future__ import annotations

import itertools
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher
from .morse import FROM_MORSE, morse_to_text, text_to_morse

_SYMBOLS = (".", "-", "x")
_DIGIT_ORDER = "1234567890"  # string index i corresponds to this digit


def _parse_key(key: str) -> dict[str, str]:
    """Parse a key into a ``digit -> symbol`` map.

    Accepts the 10-char form (over ``.-x``, indexed 1..9,0) with optional spaces.
    """
    cleaned = "".join(ch for ch in str(key) if ch in ".-x")
    if len(cleaned) != 10:
        raise ValueError(
            f"Pollux key must give 10 symbols over {{., -, x}} (got {len(cleaned)}): {key!r}"
        )
    mapping = {_DIGIT_ORDER[i]: cleaned[i] for i in range(10)}
    if set(mapping.values()) != set(_SYMBOLS):
        raise ValueError("Pollux key must use all three symbols . - x at least once")
    return mapping


def _invert(mapping: dict[str, str]) -> dict[str, list[str]]:
    """Build ``symbol -> [digits]`` from a ``digit -> symbol`` map."""
    out: dict[str, list[str]] = {s: [] for s in _SYMBOLS}
    for digit in _DIGIT_ORDER:
        out[mapping[digit]].append(digit)
    return out


class Pollux(Cipher):
    """Homophonic morse fractionation (ACA 'Pollux')."""

    name = "pollux"
    aliases = ()
    description = "Homophonic morse cipher: . - x each map to several digits 0-9."
    key_format = "10 symbols over {.,-,x} (one per digit 1..9,0; all three symbols used)"
    key_example = "..-x--.xx-"
    needs_key = True
    complexity = 4

    def encode(self, text: str, key: str, *, rng=None) -> str:
        """Encrypt ``text`` to a digit string.

        Homophonic: for each morse symbol a digit of that class is chosen. With
        ``rng=None`` selection is deterministic (round-robin through each class,
        in the digit order 1..9,0) so output is reproducible; pass a
        ``random.Random`` for a randomized realization.
        """
        mapping = _parse_key(key)
        by_symbol = _invert(mapping)
        stream = text_to_morse(text)
        cursors = {s: 0 for s in _SYMBOLS}
        out: list[str] = []
        for sym in stream:
            choices = by_symbol[sym]
            if rng is None:
                digit = choices[cursors[sym] % len(choices)]
                cursors[sym] += 1
            else:
                digit = rng.choice(choices)
            out.append(digit)
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        """Decrypt a digit string back to plaintext (deterministic)."""
        mapping = _parse_key(key)
        stream = "".join(mapping[ch] for ch in str(text) if ch in mapping)
        return morse_to_text(stream)

    # -- cracking -------------------------------------------------------
    def crack(
        self,
        text: str,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Best-effort keyless crack by searching digit->symbol assignments.

        Each of the 10 distinct digits is assigned to one of ``{., -, x}``; a
        valid morse stream must parse (no ``xxx``, every inter-``x`` run is a
        real morse code). We enumerate assignments consistent with the digits
        that actually appear, decode each, and score the recovered English.

        This is hard keyless (3^10 raw assignments, most yielding garbage). We
        prune with structural validity and rank by n-gram fitness, honoring the
        ``timeout`` deadline.
        """
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        present = sorted(set(digits))
        if not present or len(digits) < 4:
            return []

        deadline = None if timeout is None else time.monotonic() + timeout
        results: list[Candidate] = []
        seen: set[str] = set()

        # Assign each present digit to one of the three symbols. Require at least
        # one digit per symbol that is actually present in the text (otherwise
        # the symbol contributes nothing and the partition is degenerate).
        n = len(present)
        for combo in itertools.product(_SYMBOLS, repeat=n):
            if deadline is not None and time.monotonic() > deadline:
                break
            used = set(combo)
            # Need dot and dash to spell letters, and x to separate them.
            if used != set(_SYMBOLS):
                continue
            digit_to_sym = dict(zip(present, combo, strict=False))
            stream = "".join(digit_to_sym[d] for d in digits)
            # Structural filter: ACA bans xxx; reject impossible morse.
            if "xxx" in stream or stream.startswith("x") or stream.endswith("x"):
                continue
            # Strong constraint: a correct partition yields a stream whose every
            # inter-x run is a real morse code. Reject any with an invalid code.
            codes = [c for w in stream.split("xx") for c in w.split("x") if c]
            if not codes or any(c not in FROM_MORSE for c in codes):
                continue
            plain = morse_to_text(stream)
            letters = "".join(c for c in plain if c.isalpha())
            if len(letters) < 3:
                continue
            if plain in seen:
                continue
            seen.add(plain)
            # Reconstruct a representative key string for display.
            key_repr = "".join(digit_to_sym.get(d, ".") for d in _DIGIT_ORDER)
            results.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=key_repr,
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"assignment": digit_to_sym},
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top]
