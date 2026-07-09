"""Two-Square cipher (a.k.a. double Playfair).

Two independent 5x5 keyed squares (one keyword each, same construction as
Playfair). The squares are stacked either VERTICALLY (square 1 on top, square 2
on bottom -- the default) or HORIZONTALLY (square 1 left, square 2 right).

Encrypt a digraph (a, b):

* VERTICAL -- locate ``a`` in the TOP square and ``b`` in the BOTTOM square.
  They form opposite corners of a rectangle; the ciphertext is the other two
  corners, taking the TOP-square letter first and the BOTTOM-square letter
  second. If ``a`` and ``b`` share a COLUMN the rectangle is degenerate and the
  digraph passes through UNCHANGED (a "transparency").
* HORIZONTAL -- locate ``a`` in the LEFT square and ``b`` in the RIGHT square;
  ciphertext is the opposite corners (LEFT-square letter first). If they share
  a ROW the digraph passes through unchanged.

The transform is reciprocal: decoding applies the exact same rule, so encode and
decode share one implementation.

Alphabet: the published reference vectors (Wikipedia, Crypto Corner) build their
squares from the 25-letter alphabet that KEEPS J and DROPS Q. That is the
default here so the vectors reproduce exactly. Plaintext is uppercased, reduced
to that alphabet, split into digraphs, and a lone final letter is padded with X.
Unlike Playfair, doubled letters are allowed and a letter may encrypt to itself.

KEY FORMAT: two keywords separated by ``/`` -- ``"TOPKEY/BOTKEY"`` (top/left
first, bottom/right second). An optional third field selects the layout:
``"TOPKEY/BOTKEY/H"`` for horizontal (``V`` or omitted = vertical).
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher
from .squares import PolybiusSquare

# Reference vectors keep J and drop Q (25 letters), unlike Playfair's J->I.
ALPHABET_NO_Q = "ABCDEFGHIJKLMNOPRSTUVWXYZ"


def _parse_key(key: str) -> tuple[str, str, bool]:
    """Return (top_keyword, bottom_keyword, vertical) from ``"K1/K2[/H]"``."""
    parts = [p.strip() for p in str(key).split("/")]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError("two-square key must be 'TOPKEY/BOTKEY' (optionally '/V' or '/H')")
    vertical = True
    if len(parts) >= 3 and parts[2]:
        flag = parts[2].upper()[0]
        if flag == "H":
            vertical = False
        elif flag != "V":
            raise ValueError("layout flag must be 'V' (vertical) or 'H' (horizontal)")
    return parts[0], parts[1], vertical


def _squares(key: str) -> tuple[PolybiusSquare, PolybiusSquare, bool]:
    top_kw, bot_kw, vertical = _parse_key(key)
    top = PolybiusSquare(top_kw, alphabet=ALPHABET_NO_Q)
    bot = PolybiusSquare(bot_kw, alphabet=ALPHABET_NO_Q)
    return top, bot, vertical


def _digraphs(letters: str) -> list[tuple[str, str]]:
    """Split a clean letter stream into digraphs, padding a lone tail with X."""
    pairs: list[tuple[str, str]] = []
    i = 0
    n = len(letters)
    while i < n:
        a = letters[i]
        if i + 1 < n:
            pairs.append((a, letters[i + 1]))
            i += 2
        else:
            pairs.append((a, "X"))
            i += 1
    return pairs


def _transform(
    pairs: list[tuple[str, str]], top: PolybiusSquare, bot: PolybiusSquare, vertical: bool
) -> str:
    out: list[str] = []
    for a, b in pairs:
        ra, ca = top.rc(a)
        rb, cb = bot.rc(b)
        if vertical:
            if ca == cb:  # same column -> transparency, passes through unchanged
                out.append(a)
                out.append(b)
            else:
                out.append(top.at(ra, cb))
                out.append(bot.at(rb, ca))
        else:  # horizontal layout
            if ra == rb:  # same row -> passes through unchanged
                out.append(a)
                out.append(b)
            else:
                out.append(top.at(ra, cb))
                out.append(bot.at(rb, ca))
    return "".join(out)


class TwoSquare(Cipher):
    """Two-Square / double-Playfair digraphic substitution over two 5x5 squares."""

    name = "two-square"
    aliases = ("double-playfair", "doubleplayfair")
    description = "Digraphic substitution over two keyed 5x5 squares (vertical/horizontal)."
    key_format = "top/bottom[/V|H] (two keywords; optional layout flag, default vertical)"
    key_example = "EXAMPLE/KEYWORD"
    complexity = 6

    def _prepare(self, text: str, top: PolybiusSquare) -> str:
        # top/bottom share the same alphabet, so either square cleans identically.
        return top.prepare(text)

    def encode(self, text: str, key: str) -> str:
        top, bot, vertical = _squares(key)
        letters = self._prepare(text, top)
        return _transform(_digraphs(letters), top, bot, vertical)

    def decode(self, text: str, key: str) -> str:
        top, bot, vertical = _squares(key)
        letters = self._prepare(text, top)
        # Reciprocal: the same rectangle rule recovers the plaintext digraphs.
        return _transform(_digraphs(letters), top, bot, vertical)

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
        """Keyless hill-climb over BOTH squares (vertical layout assumed).

        Best-effort. The ~20% transparency rate gives a statistical foothold but
        the (25!)^2 keyspace is large; on short texts this often fails to fully
        recover. Returns the single best hypothesis found, or [] if too short.
        """
        # Clean to the cipher's own alphabet so coordinates line up.
        probe = PolybiusSquare("", alphabet=ALPHABET_NO_Q)
        letters = probe.prepare(text)
        if len(letters) % 2:
            letters = letters[:-1]
        if len(letters) < 24:
            return []
        pairs = [(letters[i], letters[i + 1]) for i in range(0, len(letters), 2)]

        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 4))
        iters = int(opts.get("iters", 4000))
        deadline = (time.monotonic() + timeout) if timeout else None

        def decode_with(top_grid: str, bot_grid: str) -> str:
            tp = PolybiusSquare("", alphabet=top_grid)
            bp = PolybiusSquare("", alphabet=bot_grid)
            return _transform(pairs, tp, bp, True)

        base = ALPHABET_NO_Q
        best_plain = ""
        best_score = float("-inf")
        best_key = ""

        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            tg = list(base)
            bg = list(base)
            rng.shuffle(tg)
            rng.shuffle(bg)
            cur = scorer.score(decode_with("".join(tg), "".join(bg)))
            no_improve = 0
            for _ in range(iters):
                if deadline and time.monotonic() > deadline:
                    break
                which = tg if rng.random() < 0.5 else bg
                i, j = rng.randrange(25), rng.randrange(25)
                which[i], which[j] = which[j], which[i]
                cand = scorer.score(decode_with("".join(tg), "".join(bg)))
                if cand >= cur:
                    cur = cand
                    no_improve = 0
                else:
                    which[i], which[j] = which[j], which[i]  # revert
                    no_improve += 1
                if cur > best_score:
                    best_score = cur
                    best_plain = decode_with("".join(tg), "".join(bg))
                    best_key = "".join(tg) + "/" + "".join(bg)
                if no_improve > 1500:
                    break

        if not best_plain:
            return []
        return [
            Candidate(
                plaintext=best_plain,
                cipher=self.name,
                key=best_key,
                score=best_score,
                confidence=scorer.confidence(best_plain),
                meta={"layout": "vertical"},
            )
        ]
