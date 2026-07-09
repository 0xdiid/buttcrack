"""Bifid cipher (Delastelle, ~1901): fractionation over a 5x5 keyed square.

Each plaintext letter is replaced by its (row, col) coordinates in a keyed 5x5
Polybius square (J merged into I). Working one period-length block at a time, the
row digits and column digits are read off horizontally — all rows then all cols —
and the resulting 2P-digit stream is re-paired into (row, col) coordinates that
look up new ciphertext letters. Because each cipher letter then depends on two
plaintext letters, Bifid diffuses the message and is digraphic in effect.

KEY FORMAT
----------
A single ``--key`` of the form ``SQUAREKEY/PERIOD`` where the two components are
separated by ``/``:

  * ``SQUAREKEY`` — the keyword (or full 25-letter alphabet) used to build the
    5x5 square. The square is filled ROW-BY-ROW: the deduplicated keyword
    letters first, then the remaining A-Z letters in order (J->I). To reproduce
    a published square that uses a spiral/other fill, pass the square's letters
    read row-by-row as the keyword (e.g. ``EXTRAKLMPOHWZQDGVUSIFCBYN``).
  * ``PERIOD`` — the integer period P (block length) for the fractionation.

Examples: ``EXTRAKLMPOHWZQDGVUSIFCBYN/7`` or ``SECRET/5``.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .squares import PolybiusSquare

_FULL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def square_alphabet(drop_letter: str = "J") -> str:
    """The 25-letter square alphabet with ``drop_letter`` removed (default drops J).

    Dropping ``J`` reproduces the classic 5x5 bifid alphabet ``ABCDEFGHIKLMNOP...`` (and
    :func:`build_square` then applies the standard ``J->I`` merge on input). Any other
    drop letter is simply absent from the square; plaintext must avoid it (the square,
    like the J-dropped one, can never emit its dropped letter).
    """
    d = str(drop_letter).upper()[:1] or "J"
    if d not in _FULL_ALPHABET:
        raise ValueError(f"drop_letter must be a single A-Z letter; got {drop_letter!r}")
    return "".join(ch for ch in _FULL_ALPHABET if ch != d)


def build_square(keyword: str, drop_letter: str = "J") -> PolybiusSquare:
    """Build a 5x5 keyed :class:`PolybiusSquare` over the ``drop_letter`` alphabet.

    ``keyword`` may be a short keyword (keyed then filled A-Z minus the drop letter) or
    a full 25-letter permutation. For the default drop ``J`` the square merges ``J->I``
    on input, exactly as before.
    """
    return PolybiusSquare(keyword, size=5, alphabet=square_alphabet(drop_letter))


def bifid_encode(text: str, keyword: str, period: int, *, drop_letter: str = "J") -> str:
    """Bifid-encipher ``text`` with a keyed square (any ``drop_letter``) and period."""
    sq = build_square(keyword, drop_letter)
    return _encode_letters(sq.prepare(text), sq, period)


def bifid_decode(text: str, keyword: str, period: int, *, drop_letter: str = "J") -> str:
    """Bifid-decipher ``text`` with a keyed square (any ``drop_letter``) and period."""
    sq = build_square(keyword, drop_letter)
    return _decode_letters(sq.prepare(text), sq, period)


def _parse_key(key: str) -> tuple[str, int, str]:
    """Split ``SQUAREKEY/PERIOD[/DROP]`` into (keyword, period, drop_letter).

    The optional third component names the dropped letter (default ``J`` for
    back-compat); e.g. ``SECRET/5`` drops J, ``SECRET/5/Q`` drops Q.
    """
    s = str(key)
    if "/" not in s:
        raise ValueError("bifid key must be 'SQUAREKEY/PERIOD[/DROP]', e.g. 'SECRET/5'")
    parts = s.split("/")
    kw = parts[0]
    per = parts[1].strip()
    if not per.isdigit() or int(per) < 1:
        raise ValueError("bifid period must be a positive integer (key 'SQUAREKEY/PERIOD[/DROP]')")
    drop = "J"
    if len(parts) >= 3 and parts[2].strip():
        drop = parts[2].strip().upper()[:1]
    return kw, int(per), drop


def _encode_letters(letters: str, sq: PolybiusSquare, period: int) -> str:
    out: list[str] = []
    for b in range(0, len(letters), period):
        block = letters[b : b + period]
        rows: list[int] = []
        cols: list[int] = []
        for ch in block:
            r, c = sq.rc(ch)
            rows.append(r)
            cols.append(c)
        seq = rows + cols  # all row digits, then all column digits
        for i in range(0, len(seq), 2):
            out.append(sq.at(seq[i], seq[i + 1]))
    return "".join(out)


def _decode_letters(letters: str, sq: PolybiusSquare, period: int) -> str:
    out: list[str] = []
    for b in range(0, len(letters), period):
        block = letters[b : b + period]
        digits: list[int] = []
        for ch in block:
            r, c = sq.rc(ch)
            digits.append(r)
            digits.append(c)
        half = len(block)
        first = digits[:half]  # original row digits
        second = digits[half:]  # original column digits
        for i in range(half):
            out.append(sq.at(first[i], second[i]))
    return "".join(out)


class Bifid(Cipher):
    """Bifid fractionation cipher over a 5x5 keyed square (J->I).

    KEY FORMAT: ``SQUAREKEY/PERIOD`` (slash-separated), e.g. ``SECRET/5`` or the
    full row-by-row square ``EXTRAKLMPOHWZQDGVUSIFCBYN/7``. The square is built
    row-by-row from the deduplicated keyword then the remaining alphabet; the
    period is the block length used for the row/column fractionation.
    """

    name = "bifid"
    aliases = ("delastelle",)
    description = "Bifid fractionation over a 5x5 keyed square (J->I); key 'SQUAREKEY/PERIOD'."
    key_format = "squarekey/period (5x5 keyword and integer fractionation period)"
    key_example = "SECRET/5"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        kw, period, drop = _parse_key(key)
        sq = build_square(kw, drop)
        return _encode_letters(sq.prepare(text), sq, period)

    def decode(self, text: str, key: str) -> str:
        kw, period, drop = _parse_key(key)
        sq = build_square(kw, drop)
        return _decode_letters(sq.prepare(text), sq, period)

    @staticmethod
    def _drop_candidates(opts) -> list[str]:
        """Resolve the drop-letter(s) to try from ``opts``.

        ``drop_letters`` (a string of letters, or "all"/"sweep") requests a sweep; else
        the single ``drop_letter`` (default ``J``) is used. A wrong drop-letter makes the
        whole square search structurally blind, so sweeping it is a first-class option.
        """
        dl = opts.get("drop_letters")
        if dl is None:
            return [str(opts.get("drop_letter", "J")).upper()[:1] or "J"]
        if isinstance(dl, str):
            if dl.lower() in ("all", "sweep", "*"):
                return list(_FULL_ALPHABET)
            return [c for c in dl.upper() if c in _FULL_ALPHABET]
        return [str(x).upper()[:1] for x in dl]

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        """Keyless best-effort: brute-force the period, anneal the 5x5 square.

        For each candidate period the 25-letter square is recovered by simulated
        annealing against the quadgram score of the decrypt (the practical-
        cryptography approach), restarting from random squares to escape local
        optima. Returns the best decrypt per period, ranked; ``[]`` for inputs
        too short to fingerprint.

        ``drop_letter`` (default ``"J"``) chooses which letter the 25-cell square omits;
        ``drop_letters`` (a letter string, or ``"all"``) sweeps several — recovering a
        bifid that drops a letter other than J, which the J-only assumption misses.
        """
        raw = only_letters(text)
        if len(raw) < 40:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        drops = self._drop_candidates(opts)
        periods = opts.get("periods")
        if periods is None:
            max_p = min(int(opts.get("max_period", 15)), len(raw))
            periods = list(range(3, max_p + 1))
        restarts = int(opts.get("restarts", 2))
        temp0 = float(opts.get("temp", 10.0))
        step = float(opts.get("temp_step", 0.5))
        iters = int(opts.get("iters", 1500))

        candidates: list[Candidate] = []
        for drop in drops:
            if deadline and time.monotonic() > deadline:
                break
            alpha = square_alphabet(drop)
            base_alphabet = list(alpha)
            # Prepare the ciphertext for this square alphabet (J->I merge only when
            # dropping J; any letter outside the square alphabet is filtered — a wrong
            # drop-letter that the ciphertext contradicts therefore self-penalises).
            probe = PolybiusSquare("", size=5, alphabet=alpha)
            letters = probe.prepare(raw)

            def decrypt_with(square: str, period: int, _alpha=alpha, _letters=letters) -> str:
                sq = PolybiusSquare(square, size=5, alphabet=_alpha)
                return _decode_letters(_letters, sq, period)

            for period in periods:
                if deadline and time.monotonic() > deadline:
                    break
                best_sq = base_alphabet[:]
                best_score = float("-inf")
                for _ in range(restarts):
                    if deadline and time.monotonic() > deadline:
                        break
                    parent = base_alphabet[:]
                    rng.shuffle(parent)
                    cur = scorer.score(decrypt_with("".join(parent), period))
                    temp = temp0
                    while temp > 0:
                        if deadline and time.monotonic() > deadline:
                            break
                        for _ in range(iters):
                            i, j = rng.randrange(25), rng.randrange(25)
                            child = parent[:]
                            child[i], child[j] = child[j], child[i]
                            s = scorer.score(decrypt_with("".join(child), period))
                            delta = s - cur
                            if delta > 0 or rng.random() < math.exp(delta / temp):
                                parent, cur = child, s
                                if s > best_score:
                                    best_sq, best_score = child[:], s
                        temp -= step
                sq_str = "".join(best_sq)
                plain = decrypt_with(sq_str, period)
                key = f"{sq_str}/{period}" if drop == "J" else f"{sq_str}/{period}/{drop}"
                candidates.append(
                    Candidate(
                        plaintext=plain,
                        cipher=self.name,
                        key=key,
                        score=best_score,
                        confidence=scorer.confidence(plain),
                        meta={"period": period, "square": sq_str, "drop_letter": drop},
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
