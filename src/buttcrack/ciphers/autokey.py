"""Autokey cipher (Vigenere autokey): the keystream is the keyword then the plaintext.

Unlike Vigenere the key never repeats — after the primer keyword runs out, the
plaintext itself becomes the key. That kills the periodic structure, so cracking
hill-climbs the short primer against the full-text quadgram score.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher


def _key_indices(key: str) -> list[int]:
    idx = [ord(c) - 65 for c in only_letters(key)]
    if not idx:
        raise ValueError("autokey key must contain letters")
    return idx


def _encode_idx(plain: list[int], primer: list[int]) -> list[int]:
    keystream = primer + plain  # plaintext extends the key
    return [(plain[i] + keystream[i]) % 26 for i in range(len(plain))]


def _decode_idx(cipher: list[int], primer: list[int]) -> list[int]:
    m = len(primer)
    plain: list[int] = []
    for i, c in enumerate(cipher):
        k = primer[i] if i < m else plain[i - m]
        plain.append((c - k) % 26)
    return plain


def _to_letters(idx: list[int]) -> str:
    return "".join(chr(i + 65) for i in idx)


class Autokey(Cipher):
    name = "autokey"
    description = "Vigenere autokey: keystream is the keyword followed by the plaintext."
    key_format = "primer keyword (letters)"
    key_example = "queen"
    complexity = 4

    def encode(self, text: str, key: str) -> str:
        plain = [ord(c) - 65 for c in only_letters(text)]
        return reflow(text, _to_letters(_encode_idx(plain, _key_indices(key))))

    def decode(self, text: str, key: str) -> str:
        cipher = [ord(c) - 65 for c in only_letters(text)]
        return reflow(text, _to_letters(_decode_idx(cipher, _key_indices(key))))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 8:
            return []
        cipher = [ord(c) - 65 for c in letters]
        rng = rng or random.Random()
        max_primer = int(opts.get("max_key_length", min(12, len(letters) // 3)))
        forced = opts.get("key_length")
        primer_lengths = [int(forced)] if forced else range(1, max_primer + 1)
        restarts = int(opts.get("restarts", 3))
        deadline = (time.monotonic() + timeout) if timeout else None

        best_by_plain: dict[str, tuple[float, list[int], int]] = {}
        for m in primer_lengths:
            if m < 1:
                continue
            if deadline and time.monotonic() > deadline:
                break
            score, primer = self._solve_primer(cipher, m, scorer, rng, restarts, deadline)
            plain = _to_letters(_decode_idx(cipher, primer))
            prev = best_by_plain.get(plain)
            if prev is None or m < prev[2]:
                best_by_plain[plain] = (score, primer, m)

        candidates = [
            Candidate(
                plaintext=reflow(text, plain),
                cipher=self.name,
                key=_to_letters(primer),
                score=score,
                confidence=scorer.confidence(plain),
                meta={"key_length": m},
            )
            for plain, (score, primer, m) in best_by_plain.items()
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]

    @staticmethod
    def _solve_primer(cipher, m, scorer, rng, restarts, deadline):
        """Hill-climb an m-letter primer; a couple of restarts escape local optima."""
        best_overall_score = float("-inf")
        best_overall = [0] * m
        for r in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            primer = [0] * m if r == 0 else [rng.randrange(26) for _ in range(m)]
            best = scorer.score(_to_letters(_decode_idx(cipher, primer)))
            improved = True
            while improved:
                improved = False
                for pos in range(m):
                    if deadline and time.monotonic() > deadline:
                        break
                    best_k = primer[pos]
                    for k in range(26):
                        if k == best_k:
                            continue
                        primer[pos] = k
                        s = scorer.score(_to_letters(_decode_idx(cipher, primer)))
                        if s > best:
                            best, best_k, improved = s, k, True
                    primer[pos] = best_k
            if best > best_overall_score:
                best_overall_score, best_overall = best, primer
        return best_overall_score, best_overall
