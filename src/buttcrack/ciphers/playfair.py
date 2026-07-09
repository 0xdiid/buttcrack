"""Playfair cipher: digraphic substitution over a 5x5 keyed square (J->I).

Cracking hill-climbs the square (swap letters / rows / cols / reflect) against
the quadgram score of the decrypted text — the standard shotgun approach.
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


def _square(key: str) -> str:
    return "".join(PolybiusSquare(key).grid)


def _prepare(letters: str) -> list[tuple[str, str]]:
    """Split into digraphs, splitting doubles and padding a lone final letter."""
    s = letters.replace("J", "I")
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        a = s[i]
        b = s[i + 1] if i + 1 < len(s) else ""
        if b == "" or a == b:
            pairs.append((a, "X" if a != "X" else "Q"))
            i += 1
        else:
            pairs.append((a, b))
            i += 2
    return pairs


def _transform(pairs: list[tuple[str, str]], sq: str, direction: int) -> str:
    pos = {c: i for i, c in enumerate(sq)}
    out = []
    for a, b in pairs:
        ia, ib = pos[a], pos[b]
        ra, ca = divmod(ia, 5)
        rb, cb = divmod(ib, 5)
        if ra == rb:
            out.append(sq[ra * 5 + (ca + direction) % 5])
            out.append(sq[rb * 5 + (cb + direction) % 5])
        elif ca == cb:
            out.append(sq[((ra + direction) % 5) * 5 + ca])
            out.append(sq[((rb + direction) % 5) * 5 + cb])
        else:
            out.append(sq[ra * 5 + cb])
            out.append(sq[rb * 5 + ca])
    return "".join(out)


def _pairs_from_text(letters: str) -> list[tuple[str, str]]:
    s = letters.replace("J", "I")
    if len(s) % 2:
        s = s[:-1]
    return [(s[i], s[i + 1]) for i in range(0, len(s), 2)]


def _mutate(sq: list[str], rng: random.Random) -> list[str]:
    new = sq[:]
    r = rng.random()
    if r < 0.8:  # swap two letters
        i, j = rng.randrange(25), rng.randrange(25)
        new[i], new[j] = new[j], new[i]
    elif r < 0.9:  # swap two rows
        a, b = rng.randrange(5), rng.randrange(5)
        for c in range(5):
            new[a * 5 + c], new[b * 5 + c] = new[b * 5 + c], new[a * 5 + c]
    else:  # swap two columns
        a, b = rng.randrange(5), rng.randrange(5)
        for rr in range(5):
            new[rr * 5 + a], new[rr * 5 + b] = new[rr * 5 + b], new[rr * 5 + a]
    return new


class Playfair(Cipher):
    name = "playfair"
    description = "Digraphic substitution over a 5x5 keyed square (J->I)."
    key_format = "keyword (letters; builds the 5x5 keyed square, J->I)"
    key_example = "MONARCHY"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        return _transform(_prepare(only_letters(text)), _square(key), +1)

    def decode(self, text: str, key: str) -> str:
        return _transform(_pairs_from_text(only_letters(text)), _square(key), -1)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 20:
            return []
        pairs = _pairs_from_text(letters)
        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 3))
        temp0 = float(opts.get("temp", 12.0))
        step = float(opts.get("temp_step", 0.3))
        iters = int(opts.get("iters", 3000))
        deadline = (time.monotonic() + timeout) if timeout else None

        def score_of(sq_list: list[str]) -> float:
            return scorer.score(_transform(pairs, "".join(sq_list), -1))

        best_sq = list("ABCDEFGHIKLMNOPQRSTUVWXYZ")
        best_score = float("-inf")
        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            # Simulated annealing: accept worse moves with prob exp(delta/T) so
            # the search escapes the many local optima of the Playfair landscape.
            parent = list("ABCDEFGHIKLMNOPQRSTUVWXYZ")
            rng.shuffle(parent)
            cur = score_of(parent)
            temp = temp0
            while temp > 0:
                if deadline and time.monotonic() > deadline:
                    break
                for _ in range(iters):
                    child = _mutate(parent, rng)
                    s = score_of(child)
                    delta = s - cur
                    if delta > 0 or rng.random() < math.exp(delta / temp):
                        parent, cur = child, s
                        if s > best_score:
                            best_sq, best_score = child[:], s
                temp -= step

        plain = _transform(pairs, "".join(best_sq), -1)
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key="".join(best_sq),
                score=best_score,
                confidence=scorer.confidence(plain),
                meta={"square": "".join(best_sq)},
            )
        ]
