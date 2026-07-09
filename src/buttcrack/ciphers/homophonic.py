"""ACA Homophonic cipher: each letter has four numeric codes (01..100).

A 4-letter keyword positions four numbered 25-letter alphabets (J->I), giving
ranges 1-25 / 26-50 / 51-75 / 76-100 (100 printed "00"). Encryption picks any of
a letter's four codes at random (flattening frequencies); decryption is unique.

Because each alphabet is merely a Caesar shift of A-Z, the keyless search is only
four shift parameters, so ``crack`` recovers it by coordinate ascent.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

_ALPHA = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 25 letters, J->I


def _idx(ch: str) -> int:
    return _ALPHA.index("I" if ch == "J" else ch)


def _shifts(key: str) -> list[int]:
    ks = [_idx(c) for c in key.upper() if c.isalpha()]
    if len(ks) != 4:
        raise ValueError("homophonic key must be a 4-letter keyword")
    return ks


def _code(letter_idx: int, k: int, shift: int) -> int:
    """The code for ``letter_idx`` in alphabet ``k`` (0-3) with keyword shift."""
    return k * 25 + ((letter_idx - shift) % 25) + 1


def _decode_num(num: int, shifts: list[int]) -> str:
    if num == 0:
        num = 100
    k = (num - 1) // 25
    offset = (num - 1) - k * 25
    return _ALPHA[(offset + shifts[k]) % 25]


class Homophonic(Cipher):
    name = "homophonic"
    description = "Homophonic substitution: four numeric codes per letter (key is a 4-letter word)."
    key_format = "4-letter keyword (positions the four numbered alphabets)"
    key_example = "CODE"
    complexity = 5

    def encode(self, text: str, key: str, *, rng: random.Random | None = None) -> str:
        shifts = _shifts(key)
        out = []
        for i, ch in enumerate(only_letters(text)):
            k = rng.randrange(4) if rng else i % 4  # round-robin is deterministic
            num = _code(_idx(ch), k, shifts[k]) % 100
            out.append(f"{num:02d}")
        return " ".join(out)

    def decode(self, text: str, key: str) -> str:
        shifts = _shifts(key)
        nums = [int(tok) for tok in text.split() if tok.isdigit() and len(tok) <= 3]
        return "".join(_decode_num(n, shifts) for n in nums)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        nums = [int(tok) for tok in str(text).split() if tok.isdigit() and len(tok) <= 3]
        if len(nums) < 12:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None
        # Per-number (alphabet, offset) is fixed; only the 4 shifts are unknown.
        cells = [((n if n else 100) - 1) for n in nums]
        meta = [(c // 25, c % 25) for c in cells]

        def decrypt(shifts: list[int]) -> str:
            return "".join(_ALPHA[(off + shifts[k]) % 25] for k, off in meta)

        best_shifts = [0, 0, 0, 0]
        best_score = scorer.score(decrypt(best_shifts))
        for r in range(int(opts.get("restarts", 6))):
            if deadline and time.monotonic() > deadline:
                break
            shifts = [0, 0, 0, 0] if r == 0 else [rng.randrange(25) for _ in range(4)]
            cur = scorer.score(decrypt(shifts))
            improved = True
            while improved:
                improved = False
                for k in range(4):
                    for s in range(25):
                        trial = shifts[:]
                        trial[k] = s
                        sc = scorer.score(decrypt(trial))
                        if sc > cur:
                            cur, shifts, improved = sc, trial, True
                if cur > best_score:
                    best_score, best_shifts = cur, shifts[:]

        plain = decrypt(best_shifts)
        key = "".join(_ALPHA[s] for s in best_shifts)
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=key,
                score=best_score,
                confidence=scorer.confidence(plain),
                meta={"shifts": best_shifts},
            )
        ]
