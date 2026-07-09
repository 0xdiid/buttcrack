"""Seriated Playfair: standard Playfair over vertical pairs of a seriated block.

The Seriated Playfair (H. F. Gaines, *Cryptanalysis*, 1939; ACA cipher type) is
a periodic strengthening of ordinary Playfair. Instead of enciphering adjacent
plaintext letters, the message is *seriated* -- written into two-row blocks of a
fixed width (the *period* N) -- and the vertical pairs (top-row letter over the
bottom-row letter) are enciphered with the usual 5x5 keyed-square Playfair rules.
Breaking up the natural adjacency of Playfair digraphs is what hardens it.

ALGORITHM
---------
1. **Square.** Build one 5x5 keyed Polybius square from the keyword exactly as in
   Playfair (deduplicate the keyword, append the rest of the alphabet, J->I).
2. **Seriation.** Take the letters-only, uppercased, J->I plaintext and write it
   into blocks of two rows: the first N letters across the top row, the next N
   across the bottom row; the next N start a fresh block's top row, and so on.
3. **Vertical digraphs.** Within each block, column j gives the digraph
   ``(top[j], bottom[j])``. If a column would be a double letter, an ``X`` null
   is inserted at that point (shifting every following letter along, so the
   block layout is rebuilt) -- exactly the Playfair double-letter rule, applied
   vertically. A short final block is padded with ``X`` so its two rows are
   equal length.
4. **Encipher.** Each vertical digraph is enciphered with the standard Playfair
   rules: same row -> each letter moves one to the right (wrapping); same column
   -> one down (wrapping); otherwise the rectangle rule (keep the row, swap the
   column with the partner). Decryption moves left / up instead.
5. **Take-off.** For each block the enciphered top row is written out, then the
   enciphered bottom row, block after block -- so the ciphertext preserves the
   two-row, period-N structure.

KEY FORMAT
----------
``KEYWORD/N`` -- a Playfair keyword and the integer seriation period, e.g.
``SERIATEDPLAYFAIR/7``. The period is required (there is no meaningful default);
``KEYWORD`` alone raises an error.

VECTOR (CryptoCrack user guide, "Seriated Playfair"): keyword
``SERIATEDPLAYFAIR``, period 7, plaintext (Babbage's Rule, from *The
Codebreakers*) ``BABBAGESRULENO...DIFFICULTCIPHER X THECODEBREAKERSBYKAHN X``
enciphers to ``FSFGSCI EIVDROM QSWEFRL ... MDIECDQ RGQZ``.
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


def _parse_key(key: str) -> tuple[str, int]:
    """Return (square_string, period) from a ``KEYWORD/N`` key."""
    s = str(key).strip()
    if "/" not in s:
        raise ValueError("seriated-playfair key must be 'KEYWORD/N' (e.g. 'PLAYFAIR/7')")
    kw, _, period_part = s.rpartition("/")
    period_part = period_part.strip()
    if not period_part.isdigit() or int(period_part) < 1:
        raise ValueError("seriated-playfair period must be a positive integer (key 'KEYWORD/N')")
    period = int(period_part)
    square = "".join(PolybiusSquare(kw).grid)
    return square, period


def _seriate(letters: str, period: int) -> str:
    """Insert X nulls so no vertical pair doubles, then pad the final block.

    Blocks consume up to ``2 * period`` letters (``period`` to the top row, the
    next ``period`` to the bottom row). When column ``j`` of a block would be a
    double letter, an ``X`` is inserted at that bottom-row position, shifting the
    rest of the stream; the layout is then re-scanned from the start (so cascaded
    doubles are all caught). Finally a short trailing block is padded with ``X``
    until its two rows are equal length.
    """
    s = list(letters.replace("J", "I"))
    block = 2 * period
    while True:
        changed = False
        for bs in range(0, len(s), block):
            top = s[bs : bs + period]
            bot = s[bs + period : bs + block]
            for j in range(min(len(top), len(bot))):
                if top[j] == bot[j]:
                    s.insert(bs + period + j, "X")
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    rem = len(s) % block
    if rem:
        width = (rem + 1) // 2  # top-row width of the final partial block
        s.extend(["X"] * (2 * width - rem))
    return "".join(s)


def _enc_pair(square: str, pos: dict[str, int], a: str, b: str, direction: int) -> tuple[str, str]:
    ia, ib = pos[a], pos[b]
    ra, ca = divmod(ia, 5)
    rb, cb = divmod(ib, 5)
    if ra == rb:
        return square[ra * 5 + (ca + direction) % 5], square[rb * 5 + (cb + direction) % 5]
    if ca == cb:
        return square[((ra + direction) % 5) * 5 + ca], square[((rb + direction) % 5) * 5 + cb]
    return square[ra * 5 + cb], square[rb * 5 + ca]


def _encode_prepared(prepared: str, square: str, period: int) -> str:
    """Encipher an already-seriated stream and take it off two rows per block."""
    pos = {ch: i for i, ch in enumerate(square)}
    block = 2 * period
    out: list[str] = []
    i = 0
    n = len(prepared)
    while i < n:
        width = period if n - i >= block else (n - i) // 2
        top = prepared[i : i + width]
        bot = prepared[i + width : i + 2 * width]
        i += 2 * width
        ct_top: list[str] = []
        ct_bot: list[str] = []
        for j in range(width):
            a, b = _enc_pair(square, pos, top[j], bot[j], +1)
            ct_top.append(a)
            ct_bot.append(b)
        out.append("".join(ct_top))
        out.append("".join(ct_bot))
    return "".join(out)


def _decode_to_prepared(cipher: str, square: str, period: int) -> str:
    """Decipher to the seriated plaintext stream (nulls retained)."""
    pos = {ch: i for i, ch in enumerate(square)}
    block = 2 * period
    out: list[str] = []
    i = 0
    n = len(cipher)
    while i < n:
        width = period if n - i >= block else (n - i) // 2
        top = cipher[i : i + width]
        bot = cipher[i + width : i + 2 * width]
        i += 2 * width
        pt_top: list[str] = []
        pt_bot: list[str] = []
        for j in range(width):
            a, b = _enc_pair(square, pos, top[j], bot[j], -1)
            pt_top.append(a)
            pt_bot.append(b)
        out.append("".join(pt_top))
        out.append("".join(pt_bot))
    return "".join(out)


class SeriatedPlayfair(Cipher):
    """Seriated Playfair: Playfair over the vertical pairs of a period-N block.

    KEY FORMAT: ``KEYWORD/N`` -- a Playfair keyword (5x5 keyed square, J->I) and
    the integer seriation period, e.g. ``SERIATEDPLAYFAIR/7``. ``encode`` operates
    on a clean uppercase letter stream: it seriates the text into two-row blocks of
    width N, breaks vertical double letters with ``X`` nulls (padding a short final
    block), enciphers each vertical pair with the standard Playfair rules, and reads
    off each block's two enciphered rows in turn. ``decode`` inverts this to the
    seriated plaintext stream (the inserted/padded ``X`` nulls are retained, to be
    stripped by context).
    """

    name = "seriated-playfair"
    aliases = ("seriated_playfair",)
    description = "Playfair over the vertical pairs of a period-N seriated block (key 'KEYWORD/N')."
    key_format = "keyword/period (Playfair keyword and integer seriation period N)"
    key_example = "PLAYFAIR/7"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        square, period = _parse_key(key)
        prepared = _seriate(only_letters(text), period)
        return _encode_prepared(prepared, square, period)

    def decode(self, text: str, key: str) -> str:
        square, period = _parse_key(key)
        cipher = only_letters(text)
        if len(cipher) % 2:
            cipher = cipher[:-1]
        return _decode_to_prepared(cipher, square, period)

    def crack(
        self,
        text,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Best-effort keyless recovery: detect the period, then anneal the square.

        Seriation breaks the digraph adjacency that plain-Playfair cracking leans
        on, so square recovery is harder and only works once the period is right.
        The search therefore runs in two phases:

        1. **Period detection.** For each candidate period a short simulated-anneal
           is run and the best quadgram score recorded. The correct period scores
           noticeably higher because only then do the vertical pairs realign.
        2. **Square recovery.** The remaining time budget is concentrated on the
           top-scoring period(s), running the full shotgun simulated-anneal (swap
           letters / rows / columns) that plain Playfair uses.

        Pass ``period=N`` to skip phase 1. ``decode`` reproduces the seriated
        plaintext *with* nulls, which is what the scorer sees. Returns ``[]`` for
        inputs too short to fingerprint.
        """
        letters = only_letters(text).replace("J", "I")  # square has no J
        if len(letters) % 2:
            letters = letters[:-1]
        if len(letters) < 60:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        max_period = int(opts.get("max_period", 12))
        if opts.get("period"):
            periods: list[int] = [int(opts["period"])]
        else:
            # Only periods giving at least two full blocks (2*period each) carry
            # enough repeated structure for the score to discriminate them.
            periods = [p for p in range(3, max_period + 1) if 4 * p <= len(letters)]
        if not periods:
            periods = [3]
        restarts = int(opts.get("restarts", 8))
        iters = int(opts.get("iters", 3000))
        temp0 = float(opts.get("temp", 10.0))
        step = float(opts.get("temp_step", 0.4))
        base = list("ABCDEFGHIKLMNOPQRSTUVWXYZ")

        def anneal(period: int, n_restarts: int, n_iters: int) -> tuple[float, str]:
            best_sq = base[:]
            best_score = float("-inf")
            for _ in range(n_restarts):
                if deadline and time.monotonic() > deadline:
                    break
                parent = base[:]
                rng.shuffle(parent)
                cur = scorer.score(_decode_to_prepared(letters, "".join(parent), period))
                temp = temp0
                while temp > 0:
                    if deadline and time.monotonic() > deadline:
                        break
                    for _ in range(n_iters):
                        child = _mutate(parent, rng)
                        sc = scorer.score(_decode_to_prepared(letters, "".join(child), period))
                        delta = sc - cur
                        if delta > 0 or rng.random() < math.exp(delta / temp):
                            parent, cur = child, sc
                            if sc > best_score:
                                best_sq, best_score = child[:], sc
                    temp -= step
            return best_score, "".join(best_sq)

        # Phase 1: rank periods by a cheap anneal (skipped when a period is pinned).
        if len(periods) == 1:
            ranked_periods = periods
        else:
            scored = [(anneal(p, 1, 900)[0], p) for p in periods]
            scored.sort(reverse=True)
            keep = int(opts.get("keep_periods", 2))
            ranked_periods = [p for _, p in scored[:keep]]

        # Phase 2: concentrate the budget on the best period(s).
        candidates: list[Candidate] = []
        for period in ranked_periods:
            if deadline and time.monotonic() > deadline:
                break
            score, square = anneal(period, restarts, iters)
            plain = _decode_to_prepared(letters, square, period)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=f"{square}/{period}",
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"period": period, "square": square},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]


def _mutate(sq: list[str], rng: random.Random) -> list[str]:
    """Perturb a 5x5 square: swap two letters, two rows, or two columns."""
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
