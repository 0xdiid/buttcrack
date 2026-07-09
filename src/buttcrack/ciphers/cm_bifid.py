"""Conjugated Matrix Bifid (CM Bifid): Bifid fractionation across TWO squares.

CM Bifid is an ACA Cipher-Exchange type. It fractionates exactly like a plain
:class:`~buttcrack.ciphers.bifid.Bifid` cipher, but the lookup square used to
read the *plaintext* coordinates (Square A) differs from the square used to write
the *ciphertext* letters (Square B). The two independent 5x5 keyed alphabets
(J merged into I) are what make this the "conjugated matrix" variant; if Square B
equals Square A it degenerates to plain Bifid.

ENCRYPT (one period-length block at a time):
  1. Look up each plaintext letter's 1-indexed (row, col) in SQUARE A.
  2. Read the block's row digits left-to-right, then its column digits, into a
     single 2P-digit stream (the standard Bifid fractionation).
  3. Re-pair the stream into consecutive (row, col) pairs and read each pair as a
     ciphertext letter out of SQUARE B (the conjugate matrix).
DECRYPT reverses this: SQUARE B turns each ciphertext letter into a (row, col)
pair, the 2P digits are split into the first-P rows and second-P cols, re-paired
position-wise, and looked up in SQUARE A.

No padding is used; a short final block is processed at its true length. Cipher
length always equals (prepared) plaintext length.

KEY FORMAT
----------
``SQUAREA/SQUAREB/PERIOD`` — three ``/``-separated components:

  * ``SQUAREA`` / ``SQUAREB`` — each builds a 5x5 keyed square (25 letters,
    J->I). A component may be a keyword (the square is filled ROW-BY-ROW: the
    deduplicated keyword then the remaining A-Z letters), or a full 25-letter
    string read row-by-row to reproduce any published square regardless of the
    fill convention used to draw it. For example the ACA CM Bifid sheet draws
    Square B by entering the NOVELTY keyed alphabet down alternating columns
    (boustrophedon); that grid read row-by-row is ``NCDRSOBFQUVAGPWEYHMXLTIKZ``.
  * ``PERIOD`` — the integer period P (block length) for the fractionation.

Examples: ``EXTRAKLMPOHWZQDGVUSIFCBYN/NCDRSOBFQUVAGPWEYHMXLTIKZ/7`` or
``SECRET/KEYWORD/5``.
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


def _parse_key(key: str) -> tuple[str, str, int]:
    """Split ``SQUAREA/SQUAREB/PERIOD`` into (keyA, keyB, period)."""
    s = str(key)
    parts = s.split("/")
    if len(parts) != 3:
        raise ValueError("cm-bifid key must be 'SQUAREA/SQUAREB/PERIOD', e.g. 'SECRET/KEYWORD/5'")
    key_a, key_b, per = parts[0], parts[1], parts[2].strip()
    if not per.isdigit() or int(per) < 1:
        raise ValueError(
            "cm-bifid period must be a positive integer (key 'SQUAREA/SQUAREB/PERIOD')"
        )
    return key_a, key_b, int(per)


def _encode_letters(letters: str, sq_a: PolybiusSquare, sq_b: PolybiusSquare, period: int) -> str:
    out: list[str] = []
    for b in range(0, len(letters), period):
        block = letters[b : b + period]
        rows: list[int] = []
        cols: list[int] = []
        for ch in block:
            r, c = sq_a.rc(ch)  # coordinates read from SQUARE A
            rows.append(r)
            cols.append(c)
        seq = rows + cols  # all row digits, then all column digits
        for i in range(0, len(seq), 2):
            out.append(sq_b.at(seq[i], seq[i + 1]))  # written out of SQUARE B
    return "".join(out)


def _decode_letters(letters: str, sq_a: PolybiusSquare, sq_b: PolybiusSquare, period: int) -> str:
    out: list[str] = []
    for b in range(0, len(letters), period):
        block = letters[b : b + period]
        digits: list[int] = []
        for ch in block:
            r, c = sq_b.rc(ch)  # coordinates read from SQUARE B
            digits.append(r)
            digits.append(c)
        half = len(block)
        first = digits[:half]  # original row digits
        second = digits[half:]  # original column digits
        for i in range(half):
            out.append(sq_a.at(first[i], second[i]))  # looked up in SQUARE A
    return "".join(out)


class CMBifid(Cipher):
    """Conjugated Matrix Bifid: Bifid fractionation across two keyed squares.

    KEY FORMAT: ``SQUAREA/SQUAREB/PERIOD`` (slash-separated), e.g.
    ``SECRET/KEYWORD/5`` or full row-by-row grids
    ``EXTRAKLMPOHWZQDGVUSIFCBYN/NCDRSOBFQUVAGPWEYHMXLTIKZ/7``. Each square is
    built row-by-row from the deduplicated keyword then the remaining alphabet
    (J->I); plaintext coordinates are read from Square A and ciphertext letters
    are written out of Square B. The period is the fractionation block length.
    """

    name = "cm-bifid"
    aliases = ("conjugated-bifid", "cmbifid")
    description = (
        "Conjugated Matrix Bifid: Bifid fractionation across two 5x5 keyed "
        "squares (J->I); key 'SQUAREA/SQUAREB/PERIOD'."
    )
    key_format = "squareA/squareB/period (two 5x5 keywords and integer period)"
    key_example = "SECRET/KEYWORD/5"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        key_a, key_b, period = _parse_key(key)
        sq_a = PolybiusSquare(key_a)
        sq_b = PolybiusSquare(key_b)
        return _encode_letters(sq_a.prepare(text), sq_a, sq_b, period)

    def decode(self, text: str, key: str) -> str:
        key_a, key_b, period = _parse_key(key)
        sq_a = PolybiusSquare(key_a)
        sq_b = PolybiusSquare(key_b)
        return _decode_letters(sq_b.prepare(text), sq_a, sq_b, period)

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
        """Keyless best-effort: brute-force the period, anneal BOTH squares.

        Two independent 25-letter alphabets must be recovered (keyspace ~
        (25!)^2), so this is materially harder than plain Bifid. For each
        candidate period we run simulated annealing that, each step, swaps two
        letters in one of the two squares (chosen at random) and scores the
        decrypt with the quadgram model. The deadline from ``timeout`` is
        honoured throughout; returns the best decrypt per period, ranked.
        Recovery needs a generous budget and ample ciphertext (ACA: ~150-200
        letters); ``[]`` is returned for inputs too short to fingerprint.
        """
        letters = only_letters(text).replace("J", "I")
        if len(letters) < 60:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        base_alphabet = list("ABCDEFGHIKLMNOPQRSTUVWXYZ")
        periods = opts.get("periods")
        if periods is None:
            max_p = min(int(opts.get("max_period", 15)), len(letters))
            periods = list(range(3, max_p + 1))
        restarts = int(opts.get("restarts", 2))
        temp0 = float(opts.get("temp", 10.0))
        step = float(opts.get("temp_step", 0.5))
        iters = int(opts.get("iters", 1500))

        def decrypt_with(square_a: str, square_b: str, period: int) -> str:
            # `letters` is already only-letters with J->I, so it is valid against
            # any PolybiusSquare without further preparation.
            sq_a = PolybiusSquare(square_a)
            sq_b = PolybiusSquare(square_b)
            return _decode_letters(letters, sq_a, sq_b, period)

        candidates: list[Candidate] = []
        for period in periods:
            if deadline and time.monotonic() > deadline:
                break
            best_a = base_alphabet[:]
            best_b = base_alphabet[:]
            best_score = float("-inf")
            for _ in range(restarts):
                if deadline and time.monotonic() > deadline:
                    break
                parent_a = base_alphabet[:]
                parent_b = base_alphabet[:]
                rng.shuffle(parent_a)
                rng.shuffle(parent_b)
                cur = scorer.score(decrypt_with("".join(parent_a), "".join(parent_b), period))
                temp = temp0
                while temp > 0:
                    if deadline and time.monotonic() > deadline:
                        break
                    for _ in range(iters):
                        child_a = parent_a[:]
                        child_b = parent_b[:]
                        if rng.random() < 0.5:
                            i, j = rng.randrange(25), rng.randrange(25)
                            child_a[i], child_a[j] = child_a[j], child_a[i]
                        else:
                            i, j = rng.randrange(25), rng.randrange(25)
                            child_b[i], child_b[j] = child_b[j], child_b[i]
                        s = scorer.score(decrypt_with("".join(child_a), "".join(child_b), period))
                        delta = s - cur
                        if delta > 0 or rng.random() < math.exp(delta / temp):
                            parent_a, parent_b, cur = child_a, child_b, s
                            if s > best_score:
                                best_a, best_b, best_score = child_a[:], child_b[:], s
                    temp -= step
            sq_a_str = "".join(best_a)
            sq_b_str = "".join(best_b)
            plain = decrypt_with(sq_a_str, sq_b_str, period)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=f"{sq_a_str}/{sq_b_str}/{period}",
                    score=best_score,
                    confidence=scorer.confidence(plain),
                    meta={"period": period, "square_a": sq_a_str, "square_b": sq_b_str},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
