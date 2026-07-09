"""Four-square cipher: digraphic substitution over four 5x5 squares.

Layout (a 2x2 block of 5x5 squares)::

    [ PLAINTEXT ] [ CIPHER TR ]
    [ CIPHER BL ] [ PLAINTEXT ]

The top-left and bottom-right squares hold the straight alphabet; the top-right
and bottom-left squares are keyed mixed alphabets (one keyword each). To match
the canonical Wikipedia / practicalcryptography vector this uses the 25-letter
"Q omitted" alphabet (``ABCDEFGHIJKLMNOPRSTUVWXYZ`` -- I and J both kept, Q
dropped). Encryption maps a plaintext digraph (a, b) to the opposite corners of
the rectangle it spans: the first cipher letter is read from the top-right
square (row of a, column of b) and the second from the bottom-left square (row
of b, column of a). Decryption is the clean inverse over the same four squares.

KEY FORMAT: two keywords separated by ``/`` --
``"TOPRIGHTKEY/BOTTOMLEFTKEY"`` (e.g. ``"EXAMPLE/KEYWORD"``). The first keyword
builds the top-right cipher square, the second the bottom-left cipher square.
The plaintext squares are fixed straight alphabets.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher
from .squares import PolybiusSquare

# Canonical four-square alphabet: 25 letters with Q dropped (Wikipedia variant).
# I and J are BOTH present; this differs from the usual 5x5 J->I merge.
ALPHABET = "ABCDEFGHIJKLMNOPRSTUVWXYZ"


def _split_key(key: str) -> tuple[str, str]:
    """Split ``"TR/BL"`` into the top-right and bottom-left keywords."""
    if "/" not in key:
        raise ValueError(
            "four-square key must be two keywords separated by '/', e.g. 'EXAMPLE/KEYWORD'"
        )
    tr, bl = key.split("/", 1)
    return tr, bl


def _prepare(text: str) -> str:
    """Uppercase, keep only the 25 alphabet letters (drops Q), pad to even."""
    s = "".join(ch for ch in text.upper() if ch in ALPHABET)
    if len(s) % 2:
        s += "X"
    return s


def _squares(key: str) -> tuple[PolybiusSquare, PolybiusSquare]:
    tr_kw, bl_kw = _split_key(key)
    tr = PolybiusSquare(tr_kw, alphabet=ALPHABET)
    bl = PolybiusSquare(bl_kw, alphabet=ALPHABET)
    return tr, bl


# The plaintext squares are the fixed straight alphabet (empty keyword).
_PLAIN = PolybiusSquare("", alphabet=ALPHABET)


def _encode_pairs(letters: str, tr: PolybiusSquare, bl: PolybiusSquare) -> str:
    out: list[str] = []
    for i in range(0, len(letters), 2):
        a, b = letters[i], letters[i + 1]
        ra, ca = _PLAIN.rc(a)
        rb, cb = _PLAIN.rc(b)
        out.append(tr.at(ra, cb))
        out.append(bl.at(rb, ca))
    return "".join(out)


def _decode_pairs(letters: str, tr: PolybiusSquare, bl: PolybiusSquare) -> str:
    out: list[str] = []
    for i in range(0, len(letters) - 1, 2):
        c1, c2 = letters[i], letters[i + 1]
        r1, col1 = tr.rc(c1)  # top-right cipher letter
        r2, col2 = bl.rc(c2)  # bottom-left cipher letter
        # First plaintext from top-left (row of c1, col of c2);
        # second from bottom-right (row of c2, col of c1).
        out.append(_PLAIN.at(r1, col2))
        out.append(_PLAIN.at(r2, col1))
    return "".join(out)


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


def _decode_with_grids(letters: str, tr_grid: list[str], bl_grid: list[str]) -> str:
    """Fast decode straight from two flat 25-char grids (for the crack loop)."""
    plain = ALPHABET
    tr_pos = {c: i for i, c in enumerate(tr_grid)}
    bl_pos = {c: i for i, c in enumerate(bl_grid)}
    out: list[str] = []
    for i in range(0, len(letters) - 1, 2):
        i1 = tr_pos[letters[i]]
        i2 = bl_pos[letters[i + 1]]
        r1, col1 = divmod(i1, 5)
        r2, col2 = divmod(i2, 5)
        out.append(plain[r1 * 5 + col2])
        out.append(plain[r2 * 5 + col1])
    return "".join(out)


class FourSquare(Cipher):
    name = "four-square"
    aliases = ("foursquare", "4square")
    description = "Digraphic substitution over four 5x5 squares (Q-omitted alphabet)."
    key_format = "topright/bottomleft (two keywords for the two cipher squares)"
    key_example = "EXAMPLE/KEYWORD"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        tr, bl = _squares(key)
        return _encode_pairs(_prepare(text), tr, bl)

    def decode(self, text: str, key: str) -> str:
        tr, bl = _squares(key)
        # Ciphertext is already letters; filter to the alphabet but do not pad.
        letters = "".join(ch for ch in text.upper() if ch in ALPHABET)
        return _decode_pairs(letters, tr, bl)

    def crack(
        self,
        text,
        scorer: NgramScorer,
        *,
        top=5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ):
        """Keyless best-effort: simulated annealing over both cipher squares.

        The two plaintext squares are fixed, so only the two keyed cipher
        squares (50 letters of state) need to be recovered. We anneal both
        simultaneously, scoring candidate decryptions with the n-gram scorer.
        Four-square is stronger than Playfair, so long ciphertext is needed.
        """
        letters = "".join(ch for ch in text.upper() if ch in ALPHABET)
        if len(letters) % 2:
            letters = letters[:-1]
        if len(letters) < 60:
            return []
        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 4))
        temp0 = float(opts.get("temp", 10.0))
        step = float(opts.get("temp_step", 0.3))
        iters = int(opts.get("iters", 2000))
        deadline = (time.monotonic() + timeout) if timeout else None

        def score_of(tr_grid: list[str], bl_grid: list[str]) -> float:
            return scorer.score(_decode_with_grids(letters, tr_grid, bl_grid))

        base = list(ALPHABET)
        best_tr, best_bl = base[:], base[:]
        best_score = float("-inf")

        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            tr = base[:]
            bl = base[:]
            rng.shuffle(tr)
            rng.shuffle(bl)
            cur = score_of(tr, bl)
            temp = temp0
            while temp > 0:
                if deadline and time.monotonic() > deadline:
                    break
                for _ in range(iters):
                    # Perturb one of the two squares per step.
                    if rng.random() < 0.5:
                        child_tr = _mutate(tr, rng)
                        child_bl = bl
                    else:
                        child_tr = tr
                        child_bl = _mutate(bl, rng)
                    s = score_of(child_tr, child_bl)
                    delta = s - cur
                    if delta > 0 or rng.random() < math.exp(delta / temp):
                        tr, bl, cur = child_tr, child_bl, s
                        if s > best_score:
                            best_tr, best_bl, best_score = tr[:], bl[:], s
                temp -= step

        plain = _decode_with_grids(letters, best_tr, best_bl)
        key = f"{''.join(best_tr)}/{''.join(best_bl)}"
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=key,
                score=best_score,
                confidence=scorer.confidence(plain),
                meta={"top_right": "".join(best_tr), "bottom_left": "".join(best_bl)},
            )
        ]
