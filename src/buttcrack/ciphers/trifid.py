"""Trifid cipher: Delastelle's 3-D fractionation (Bifid extended to a cube).

A keyed 27-character alphabet (26 letters + one extra symbol) is laid into a
3x3x3 cube, giving every character a trigram (layer, row, col) each in 1..3 in
the ACA row-by-row convention: the character at 0-based position ``k`` has
trigram ``(k//9 + 1, (k//3) % 3 + 1, k % 3 + 1)``. There is NO I/J merge -- all
26 letters are present alongside the 27th symbol.

Within each period-P block the three coordinates of each letter are written
vertically, then read off horizontally (all P layer digits, then all P row
digits, then all P col digits). Consecutive triples of those 3P digits are
turned back into letters through the same cube. Ciphertext length equals
plaintext length.

KEY FORMAT
    "KEYWORD/PERIOD" or "KEYWORD/PERIOD/SYMBOL" -- the '/' separates a square
    keyword from an integer period P (5-15 conventional), with an optional third
    field naming the 27th symbol (default '#'). Examples::

        "EXTRAORDINARY/10"        # ACA-style, '#' as the 27th symbol
        "CRYPTOGRAPHY/5/+"        # '+' as the 27th symbol

    A bare keyword with no '/' is accepted with a default period of 5 and '#'.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_SYMBOL = "#"
DEFAULT_PERIOD = 5


def _parse_key(key: str) -> tuple[str, int, str]:
    """Return (keyword, period, symbol) from a "KEYWORD/PERIOD[/SYMBOL]" string."""
    parts = str(key).split("/")
    keyword = parts[0].strip()
    period = DEFAULT_PERIOD
    symbol = DEFAULT_SYMBOL
    if len(parts) >= 2 and parts[1].strip():
        period = int(parts[1].strip())
    if len(parts) >= 3 and parts[2].strip():
        symbol = parts[2].strip()[0]
    if period < 1:
        raise ValueError("trifid period must be >= 1")
    return keyword, period, symbol


def _build_alphabet(keyword: str, symbol: str) -> str:
    """27-char keyed alphabet: deduped keyword + remaining A-Z + the symbol."""
    seq: list[str] = []
    pool = LETTERS + symbol
    for ch in keyword.upper() + LETTERS + symbol:
        if ch in pool and ch not in seq:
            seq.append(ch)
    if len(seq) != 27:
        raise ValueError(f"trifid alphabet needs 27 cells, built {len(seq)}")
    return "".join(seq)


def _trigrams(alphabet: str) -> dict[str, tuple[int, int, int]]:
    """Map each character to its 1..3 (layer, row, col) trigram (row-by-row fill)."""
    out: dict[str, tuple[int, int, int]] = {}
    for k, ch in enumerate(alphabet):
        out[ch] = (k // 9 + 1, (k // 3) % 3 + 1, k % 3 + 1)
    return out


def _from_trigram(alphabet: str, layer: int, row: int, col: int) -> str:
    return alphabet[(layer - 1) * 9 + (row - 1) * 3 + (col - 1)]


def _prepare(text: str, alphabet: str) -> str:
    """Uppercase letters only, plus any occurrence of the 27th symbol if present."""
    extra = alphabet[26]
    out = []
    for ch in text.upper():
        if "A" <= ch <= "Z" or ch == extra:
            out.append(ch)
    return "".join(out)


def _encode_block(block: str, alphabet: str, tri: dict[str, tuple[int, int, int]]) -> str:
    layers = "".join(str(tri[ch][0]) for ch in block)
    rows = "".join(str(tri[ch][1]) for ch in block)
    cols = "".join(str(tri[ch][2]) for ch in block)
    digits = layers + rows + cols
    out = []
    for i in range(0, len(digits), 3):
        a, b, c = int(digits[i]), int(digits[i + 1]), int(digits[i + 2])
        out.append(_from_trigram(alphabet, a, b, c))
    return "".join(out)


def _decode_block(block: str, alphabet: str, tri: dict[str, tuple[int, int, int]]) -> str:
    digits = "".join(f"{tri[ch][0]}{tri[ch][1]}{tri[ch][2]}" for ch in block)
    p = len(block)
    layers = digits[0:p]
    rows = digits[p : 2 * p]
    cols = digits[2 * p : 3 * p]
    out = []
    for i in range(p):
        out.append(_from_trigram(alphabet, int(layers[i]), int(rows[i]), int(cols[i])))
    return "".join(out)


def _process(text: str, key: str, *, decoding: bool) -> str:
    keyword, period, symbol = _parse_key(key)
    alphabet = _build_alphabet(keyword, symbol)
    tri = _trigrams(alphabet)
    letters = _prepare(text, alphabet)
    fn = _decode_block if decoding else _encode_block
    out = []
    for i in range(0, len(letters), period):
        out.append(fn(letters[i : i + period], alphabet, tri))
    return "".join(out)


class Trifid(Cipher):
    name = "trifid"
    aliases = ()
    description = "Delastelle 3-D fractionation over a keyed 3x3x3 cube; key 'KEYWORD/PERIOD'."
    key_format = (
        "keyword/period[/symbol] (27-char cube keyword, integer period, optional 27th symbol)"
    )
    key_example = "EXTRAORDINARY/10"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        return _process(text, key, decoding=False)

    def decode(self, text: str, key: str) -> str:
        return _process(text, key, decoding=True)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        """Keyless best-effort: brute-force the period, anneal the cube alphabet.

        Trifid spreads each plaintext letter across 3 cipher letters, so it needs
        a fair amount of text (~120+ letters per ACA) and the correct period
        before the alphabet recovery converges. The 27th symbol does not appear in
        an A-Z-only ciphertext, so cracking assumes a 26-letter ciphertext and
        keeps '#' fixed in the last cube cell.
        """
        letters = only_letters(text)
        if len(letters) < 60:
            return []
        rng = rng or random.Random()
        symbol = DEFAULT_SYMBOL
        periods = opts.get("periods")
        if periods is None:
            periods = range(3, min(16, len(letters)) + 1)
        restarts = int(opts.get("restarts", 2))
        iters = int(opts.get("iters", 1500))
        temp0 = float(opts.get("temp", 8.0))
        step = float(opts.get("temp_step", 0.4))
        deadline = (time.monotonic() + timeout) if timeout else None

        base = LETTERS + symbol  # symbol stays pinned in cell 26

        def decode_with(alpha: str, period: int) -> str:
            tri = {ch: (k // 9 + 1, (k // 3) % 3 + 1, k % 3 + 1) for k, ch in enumerate(alpha)}
            out = []
            for i in range(0, len(letters), period):
                out.append(_decode_block(letters[i : i + period], alpha, tri))
            return "".join(out)

        best: Candidate | None = None
        best_score = float("-inf")

        for period in periods:
            if deadline and time.monotonic() > deadline:
                break
            for _ in range(restarts):
                if deadline and time.monotonic() > deadline:
                    break
                parent = list(base[:26])
                rng.shuffle(parent)
                parent_alpha = "".join(parent) + symbol
                cur = scorer.score(decode_with(parent_alpha, period))
                temp = temp0
                while temp > 0:
                    if deadline and time.monotonic() > deadline:
                        break
                    for _ in range(iters):
                        i, j = rng.randrange(26), rng.randrange(26)
                        if i == j:
                            continue
                        child = parent[:]
                        child[i], child[j] = child[j], child[i]
                        child_alpha = "".join(child) + symbol
                        s = scorer.score(decode_with(child_alpha, period))
                        delta = s - cur
                        if delta > 0 or rng.random() < math.exp(delta / temp):
                            parent, cur = child, s
                            if s > best_score:
                                best_score = s
                                plain = decode_with(child_alpha, period)
                                best = Candidate(
                                    plaintext=plain,
                                    cipher=self.name,
                                    key=f"{child_alpha}/{period}",
                                    score=s,
                                    confidence=scorer.confidence(plain),
                                    meta={"period": period, "alphabet": child_alpha},
                                )
                    temp -= step

        return [best] if best is not None else []
