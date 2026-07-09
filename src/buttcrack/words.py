"""Dictionary word-search tools — CryptoCrack's Word Search menu.

Backs `butt words`: literal/wildcard match, pattern (isomorph) match — the core
aristocrat-cribbing technique — anagram search, and substring (n-gram) search,
over a bundled public-domain English word list (Webster's 1913).
"""

from __future__ import annotations

import functools
import gzip
from importlib import resources


@functools.lru_cache(maxsize=1)
def _words() -> tuple[str, ...]:
    raw = resources.files("buttcrack.data").joinpath("words_en.txt.gz").read_bytes()
    return tuple(gzip.decompress(raw).decode("ascii").split())


def pattern_of(word: str) -> str:
    """Isomorph signature: 'NOON' -> '0.1.1.0', 'MISSISSIPPI' -> distinct repeats.

    Two words share a pattern iff one is a simple-substitution image of the other,
    which is exactly what makes a pattern word a usable aristocrat crib.
    """
    seen: dict[str, int] = {}
    out = []
    for ch in word:
        if ch not in seen:
            seen[ch] = len(seen)
        out.append(str(seen[ch]))
    return ".".join(out)


@functools.lru_cache(maxsize=1)
def _by_pattern() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for w in _words():
        index.setdefault(pattern_of(w), []).append(w)
    return index


@functools.lru_cache(maxsize=1)
def _by_anagram() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for w in _words():
        index.setdefault("".join(sorted(w)), []).append(w)
    return index


def match(template: str, *, limit: int = 200) -> list[str]:
    """Words matching a template where '?' or '.' is any letter (length fixed)."""
    t = template.upper()
    n = len(t)
    fixed = [(i, c) for i, c in enumerate(t) if c not in "?."]
    out = [w for w in _words() if len(w) == n and all(w[i] == c for i, c in fixed)]
    return out[:limit]


def pattern(word: str, *, limit: int = 200) -> list[str]:
    """Dictionary words with the same isomorph pattern as ``word``."""
    return _by_pattern().get(pattern_of(word.upper()), [])[:limit]


def anagram(letters: str, *, limit: int = 200) -> list[str]:
    """Dictionary words that are an exact anagram of ``letters``."""
    key = "".join(sorted(c for c in letters.upper() if c.isalpha()))
    return _by_anagram().get(key, [])[:limit]


def ngram(seq: str, *, limit: int = 200) -> list[str]:
    """Dictionary words containing ``seq`` as a substring."""
    s = seq.upper()
    out = [w for w in _words() if s in w]
    return out[:limit]


def contains_word(text: str) -> bool:
    return text.upper() in set(_words())


@functools.lru_cache(maxsize=4)
def _words_by_minlen(minlen: int) -> frozenset[str]:
    """Bundled words of at least ``minlen`` letters, as a fast-lookup set."""
    return frozenset(w for w in _words() if len(w) >= minlen)


def long_word_coverage(text: str, minlen: int = 5) -> float:
    """Fraction of letters coverable by dictionary words of >= ``minlen`` letters.

    A maximum-coverage DP segmentation. Genuine prose packs many long words and
    scores high (~0.5-0.8); quadgram "salad" — text that fools the n-gram model
    but isn't language, the classic stochastic-solver overfit — scores near zero
    because it cannot be tiled by long real words even though short fragments
    abound. (Short words are deliberately excluded: a rich enough dictionary can
    tile almost any letter string with 2-3 letter words, so they carry no signal.)

    This is the single most reliable cheap discriminator between a real solve and
    a confident-looking overfit, and the n-gram score alone cannot see it.
    """
    s = "".join(c for c in text.upper() if c.isalpha())
    n = len(s)
    if n == 0:
        return 0.0
    words = _words_by_minlen(minlen)
    maxlen = min(22, n)
    # best[i] = max letters of s[i:] coverable by >=minlen words (greedy gaps allowed)
    best = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        b = best[i + 1]  # leave s[i] uncovered
        cap = min(maxlen, n - i)
        for length in range(minlen, cap + 1):
            if s[i : i + length] in words:
                c = length + best[i + length]
                if c > b:
                    b = c
        best[i] = b
    return best[0] / n
