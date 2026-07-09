"""Tri-Square (Three-Square) cipher.

A polygraphic cipher invented by ACA member THALES (The Cryptogram, Sep-Oct
1959). It enciphers each plaintext DIGRAPH into a ciphertext TRIGRAPH using three
keyed 5x5 squares, so the output is 50% longer than the input (3 letters out per
2 letters in).

Square roles (the dCode / ACA convention reproduced here):

* ``square1`` holds the FIRST plaintext letter ``P1``.
* ``square2`` holds the SECOND plaintext letter ``P2``.
* ``square3`` is the "middle" intersection square.

Encrypt a digraph ``(P1, P2)`` -> trigraph ``(C1, C2, C3)``:

* ``C1`` = any letter in ``square1`` that shares ``P1``'s COLUMN (a free /
  homophonic choice).
* ``C2`` = ``square3`` at the intersection of ``P1``'s ROW (from ``square1``)
  and ``P2``'s COLUMN (from ``square2``) -- the only fixed letter.
* ``C3`` = any letter in ``square2`` that shares ``P2``'s ROW (a free /
  homophonic choice).

Because ``C1`` and ``C3`` are free choices, encryption is NON-DETERMINISTIC
(homophonic): one plaintext has many valid ciphertexts. Decryption is still
unique -- only the COLUMN of ``C1``, the ROW of ``C3`` and the full position of
``C2`` carry information::

    P1 = square1[ row(C2 in square3), col(C1 in square1) ]
    P2 = square2[ row(C3 in square2), col(C2 in square3) ]

ALPHABET: by default the ACA/CryptoCrack 25-letter square (``J`` merged into
``I``, ``Z`` kept). dCode's reference vectors instead keep ``J`` and drop ``Z``;
pass ``alphabet="ABCDEFGHIJKLMNOPQRSTUVWXY"`` to reproduce those.

KEY FORMAT: three keywords separated by ``/`` --
``"KW1/KW2/KW3"`` (square1 / square2 / square3 keywords), e.g.
``"ONE/TWO/THREE"``. A keyed square = the keyword (duplicate letters dropped,
characters outside the alphabet skipped) followed by the remaining alphabet
letters in order, filled row by row.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher
from .squares import ALPHABET_5, PolybiusSquare


def _split_key(key: str) -> tuple[str, str, str]:
    """Split ``"KW1/KW2/KW3"`` into the three square keywords."""
    parts = str(key).split("/")
    if len(parts) != 3:
        raise ValueError(
            "tri-square key must be three keywords separated by '/', e.g. 'ONE/TWO/THREE'"
        )
    return parts[0], parts[1], parts[2]


class TriSquare(Cipher):
    """Tri-Square: digraph -> trigraph substitution over three keyed 5x5 squares."""

    name = "tri-square"
    aliases = ("trisquare", "three-square", "threesquare")
    description = "Polygraphic digraph->trigraph cipher over three keyed 5x5 squares."
    key_format = "kw1/kw2/kw3 (three keywords, one per 5x5 keyed square, J->I)"
    key_example = "ONE/TWO/THREE"
    complexity = 7

    def __init__(self, alphabet: str = ALPHABET_5) -> None:
        if len(alphabet) != 25 or len(set(alphabet)) != 25:
            raise ValueError("tri-square alphabet must be 25 distinct letters")
        self.alphabet = alphabet

    # -- square / text helpers ------------------------------------------
    def _squares(self, key: str) -> tuple[PolybiusSquare, PolybiusSquare, PolybiusSquare]:
        kw1, kw2, kw3 = _split_key(key)
        s1 = PolybiusSquare(kw1, alphabet=self.alphabet)
        s2 = PolybiusSquare(kw2, alphabet=self.alphabet)
        s3 = PolybiusSquare(kw3, alphabet=self.alphabet)
        return s1, s2, s3

    def _prepare(self, text: str) -> str:
        """Uppercase, apply the alphabet's merge, keep only alphabet letters."""
        probe = PolybiusSquare("", alphabet=self.alphabet)
        return probe.prepare(text)

    # -- encode / decode -------------------------------------------------
    def encode(self, text: str, key: str, *, rng: random.Random | None = None) -> str:
        """Encrypt ``text``; ``C1``/``C3`` are homophonic.

        Without ``rng`` the choice is deterministic (``C1`` = top of ``P1``'s
        column, ``C3`` = first letter of ``P2``'s row) so output is repeatable;
        pass an ``rng`` for genuinely random homophone selection.
        """
        s1, s2, s3 = self._squares(key)
        letters = self._prepare(text)
        if len(letters) % 2:
            letters += "X"
        out: list[str] = []
        for i in range(0, len(letters), 2):
            p1, p2 = letters[i], letters[i + 1]
            r1, c1 = s1.rc(p1)
            r2, c2 = s2.rc(p2)
            # C1: a letter sharing P1's column in square1.
            row_c1 = rng.randrange(5) if rng is not None else 0
            # C3: a letter sharing P2's row in square2.
            col_c3 = rng.randrange(5) if rng is not None else 0
            out.append(s1.at(row_c1, c1))
            out.append(s3.at(r1, c2))
            out.append(s2.at(r2, col_c3))
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        """Decrypt a trigraph stream back to the plaintext digraphs."""
        s1, s2, s3 = self._squares(key)
        # Filter ciphertext to the alphabet (apply the same merge) but do not pad.
        letters = self._prepare(text)
        # Trim any trailing partial trigraph so indexing is safe.
        letters = letters[: len(letters) - len(letters) % 3]
        out: list[str] = []
        for i in range(0, len(letters), 3):
            c1, c2, c3 = letters[i], letters[i + 1], letters[i + 2]
            _, p1col = s1.rc(c1)  # column of P1
            p2row, _ = s2.rc(c3)  # row of P2
            p1row, p2col = s3.rc(c2)  # row of P1 and column of P2
            out.append(s1.at(p1row, p1col))
            out.append(s2.at(p2row, p2col))
        return "".join(out)

    # -- crack -----------------------------------------------------------
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
        """Keyless best-effort: simulated annealing over all three squares.

        Decoding couples three 25-letter squares (75 letters of hidden state),
        and the homophonic ``C1``/``C3`` positions leak only a column / a row
        each, so attribution is weak. This is a hard recovery that needs a lot
        of ciphertext; the method anneals the three grids together, scoring
        candidate decryptions with the n-gram scorer, and returns the single
        best hypothesis (or ``[]`` if there is too little ciphertext).
        """
        letters = self._prepare(text)
        letters = letters[: len(letters) - len(letters) % 3]
        n_tri = len(letters) // 3
        if n_tri < 60:
            return []

        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 3))
        temp0 = float(opts.get("temp", 8.0))
        step = float(opts.get("temp_step", 0.4))
        iters = int(opts.get("iters", 1500))
        deadline = (time.monotonic() + timeout) if timeout else None

        # Pre-split trigraphs as (C1, C2, C3) for the fast decoder.
        tris = [(letters[i], letters[i + 1], letters[i + 2]) for i in range(0, len(letters), 3)]

        def decode_with(g1: list[str], g2: list[str], g3: list[str]) -> str:
            p1pos = {c: i for i, c in enumerate(g1)}
            p2pos = {c: i for i, c in enumerate(g2)}
            p3pos = {c: i for i, c in enumerate(g3)}
            out: list[str] = []
            for c1, c2, c3 in tris:
                _, p1col = divmod(p1pos[c1], 5)
                p2row, _ = divmod(p2pos[c3], 5)
                p1row, p2col = divmod(p3pos[c2], 5)
                out.append(g1[p1row * 5 + p1col])
                out.append(g2[p2row * 5 + p2col])
            return "".join(out)

        def mutate(sq: list[str]) -> list[str]:
            new = sq[:]
            r = rng.random()
            if r < 0.8:
                i, j = rng.randrange(25), rng.randrange(25)
                new[i], new[j] = new[j], new[i]
            elif r < 0.9:
                a, b = rng.randrange(5), rng.randrange(5)
                for c in range(5):
                    new[a * 5 + c], new[b * 5 + c] = new[b * 5 + c], new[a * 5 + c]
            else:
                a, b = rng.randrange(5), rng.randrange(5)
                for rr in range(5):
                    new[rr * 5 + a], new[rr * 5 + b] = new[rr * 5 + b], new[rr * 5 + a]
            return new

        base = list(self.alphabet)
        best = (base[:], base[:], base[:])
        best_score = float("-inf")

        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            g1, g2, g3 = base[:], base[:], base[:]
            rng.shuffle(g1)
            rng.shuffle(g2)
            rng.shuffle(g3)
            cur = scorer.score(decode_with(g1, g2, g3))
            temp = temp0
            while temp > 0:
                if deadline and time.monotonic() > deadline:
                    break
                for _ in range(iters):
                    which = rng.randrange(3)
                    cand1, cand2, cand3 = g1, g2, g3
                    if which == 0:
                        cand1 = mutate(g1)
                    elif which == 1:
                        cand2 = mutate(g2)
                    else:
                        cand3 = mutate(g3)
                    s = scorer.score(decode_with(cand1, cand2, cand3))
                    delta = s - cur
                    if delta > 0 or rng.random() < math.exp(delta / temp):
                        g1, g2, g3, cur = cand1, cand2, cand3, s
                        if s > best_score:
                            best = (g1[:], g2[:], g3[:])
                            best_score = s
                temp -= step

        bg1, bg2, bg3 = best
        plain = decode_with(bg1, bg2, bg3)
        key = f"{''.join(bg1)}/{''.join(bg2)}/{''.join(bg3)}"
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=key,
                score=best_score,
                confidence=scorer.confidence(plain),
                meta={
                    "square1": "".join(bg1),
                    "square2": "".join(bg2),
                    "square3": "".join(bg3),
                },
            )
        ]
