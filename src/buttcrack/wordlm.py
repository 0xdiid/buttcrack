"""Word-level scoring and reconstruction — where character n-grams fall short.

A quadgram model scores fluent *letter* sequences, so it happily accepts strings that
locally look English but are not words — ``THESTHATION``, ``INGTHEREOF`` — and a blind
solver's fitness peak can sit on exactly such "salad". Two word-level tools catch what
the character model cannot:

* :func:`word_segment` — the best segmentation of a letter string into dictionary words
  (a Viterbi DP rewarding real words and penalizing uncovered letters). Its reliable
  discriminator is ``long_coverage`` — the fraction of letters inside *long* (>= 5) real
  words: genuine running prose packs long words (~0.7–0.8) while salad, which a rich
  dictionary can only tile with short/obscure fragments, cannot (~0.2–0.3). Returns the
  segmentation, coverage, long-word coverage, a per-letter score, and a ``wordlike``
  verdict. (Like any long-word method it under-rates deliberately short-word text such as
  pangrams — use it on running text.)
* :func:`word_tiling` — reconstruct a plaintext from a *bag of letters* (a multiset) by
  tiling dictionary words that exactly consume it. This is the anagram/exact-cover step
  for recovering text from per-column or per-coset letter multisets left by a
  transposition; the number of full tilings found is a determinacy/unicity signal (one
  tiling ⇒ the anagram is essentially forced; many ⇒ ambiguous).

Both build on the bundled dictionary in :mod:`buttcrack.words`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .text import only_letters
from .words import _words_by_minlen

_MAX_WORD = 22  # longest dictionary word length we bother trying to match

# --- word-segmentation scoring ----------------------------------------------


@dataclass
class Segmentation:
    """A best segmentation of a letter string into dictionary words (+ leftover gaps)."""

    segments: list[tuple[str, str]]  # ("word", w) or ("gap", letters)
    words: list[str]
    coverage: float  # fraction of letters inside any dictionary word
    long_coverage: float  # fraction inside LONG (>= long_at) words — the real signal
    score: float
    score_per_letter: float

    @property
    def wordlike(self) -> bool:
        """True for genuine running prose: substantial long-word coverage, positive score.

        Raw coverage alone is fooled — a rich dictionary tiles almost anything with short
        words — so the verdict rests on *long*-word coverage, which n-gram salad cannot
        fake. (This under-rates deliberately short-word text such as pangrams.)
        """
        return self.long_coverage >= 0.4 and self.score_per_letter > 0.0

    def summary(self) -> str:
        joined = " ".join(w if t == "word" else f"[{w}]" for t, w in self.segments)
        return (
            f"cov={self.coverage:.2f} long={self.long_coverage:.2f} "
            f"spl={self.score_per_letter:+.2f} {joined}"
        )


def word_segment(
    text: str,
    *,
    minlen: int = 2,
    gap_penalty: float = 2.0,
    word_cost: float = 1.0,
    long_bonus: float = 1.0,
    long_at: int = 5,
) -> Segmentation:
    """Best segmentation of ``text`` into dictionary words, favoring few long real words.

    A Viterbi DP over letter positions: a covered dictionary word of length ``L`` earns
    ``L + long_bonus*(L>=long_at) - word_cost``; an uncovered letter costs ``gap_penalty``.
    The per-word ``word_cost`` biases toward the few long words of real prose over the many
    short fragments a dictionary uses to tile salad. Read the verdict off ``long_coverage``
    (the fraction of letters in >= ``long_at`` real words): genuine running prose scores
    high there and n-gram salad low, whereas raw coverage is fooled because a rich
    dictionary tiles almost anything. Words shorter than ``minlen`` are ignored.
    """
    s = only_letters(text).upper()
    n = len(s)
    if n == 0:
        return Segmentation([], [], 0.0, 0.0, 0.0, 0.0)
    words = _words_by_minlen(minlen)
    best = [0.0] * (n + 1)
    choice: list[tuple] = [("gap", 0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        b = best[i + 1] - gap_penalty  # leave s[i] uncovered
        ch: tuple = ("gap", i + 1)
        cap = min(_MAX_WORD, n - i)
        for length in range(minlen, cap + 1):
            w = s[i : i + length]
            if w in words:
                reward = length + (long_bonus if length >= long_at else 0.0) - word_cost
                c = reward + best[i + length]
                if c > b:
                    b = c
                    ch = ("word", i + length, w)
        best[i] = b
        choice[i] = ch
    segments: list[tuple[str, str]] = []
    i = 0
    while i < n:
        ch = choice[i]
        if ch[0] == "word":
            segments.append(("word", ch[2]))
            i = ch[1]
        else:
            segments.append(("gap", s[i : ch[1]]))
            i = ch[1]
    covered = sum(len(w) for t, w in segments if t == "word")
    long_covered = sum(len(w) for t, w in segments if t == "word" and len(w) >= long_at)
    return Segmentation(
        segments=segments,
        words=[w for t, w in segments if t == "word"],
        coverage=covered / n,
        long_coverage=long_covered / n,
        score=best[0],
        score_per_letter=best[0] / n,
    )


# --- multiset word-tiling (anagram / exact cover) ----------------------------


@dataclass
class Tiling:
    """A sequence of dictionary words that exactly consumes a letter multiset."""

    words: list[str]
    score: float

    @property
    def text(self) -> str:
        return "".join(self.words)


def _submultiset(need: Counter, have: Counter) -> bool:
    return all(have.get(c, 0) >= k for c, k in need.items())


def _freeze(counter: Counter) -> tuple:
    return tuple(sorted((c, k) for c, k in counter.items() if k > 0))


def word_tiling(
    letters: str,
    *,
    minlen: int = 3,
    beam: int = 200,
    max_solutions: int = 20,
    max_expansions: int = 300_000,
    long_at: int = 5,
    long_bonus: float = 1.0,
) -> list[Tiling]:
    """Reconstruct plaintext from a *bag* of ``letters`` by tiling dictionary words.

    Beam search that consumes the multiset of ``letters`` with dictionary words (length
    ``>= minlen``) until nothing remains, returning full tilings ranked by a word-length
    reward. This is the exact-cover / anagram step for turning per-column or per-coset
    letter multisets (what a transposition leaves) back into words. The number of tilings
    returned is a determinacy signal: one ⇒ the anagram is essentially forced, many ⇒ it
    is ambiguous. Bounded by ``beam`` / ``max_expansions`` so a large bag degrades to a
    best-effort search rather than exploding.
    """
    avail0 = Counter(only_letters(letters).upper())
    total = sum(avail0.values())
    if total == 0:
        return []
    # Candidate pool: dictionary words that fit within the full bag (small for real bags).
    pool = [(w, Counter(w)) for w in _words_by_minlen(minlen) if _submultiset(Counter(w), avail0)]
    pool.sort(key=lambda x: -len(x[0]))  # try long words first

    def _reward(w: str) -> float:
        return len(w) + (long_bonus if len(w) >= long_at else 0.0)

    solutions: list[Tiling] = []
    frontier: list[tuple[tuple, tuple[str, ...], float]] = [(_freeze(avail0), (), 0.0)]
    seen: set[tuple] = set()
    expansions = 0
    while frontier and len(solutions) < max_solutions and expansions < max_expansions:
        nxt: list[tuple[tuple, tuple[str, ...], float]] = []
        for rem_f, words, score in frontier:
            rem = Counter(dict(rem_f))
            if sum(rem.values()) == 0:
                solutions.append(Tiling(list(words), score))
                continue
            for w, wc in pool:
                if expansions >= max_expansions:
                    break
                if _submultiset(wc, rem):
                    newrem = rem.copy()
                    newrem.subtract(wc)
                    newrem = +newrem  # drop zero/negative counts
                    nxt.append((_freeze(newrem), (*words, w), score + _reward(w)))
                    expansions += 1
        # Prune: keep the states that consumed the most letters, then highest score.
        nxt.sort(key=lambda st: (total - sum(k for _, k in st[0]), st[2]), reverse=True)
        pruned: list[tuple[tuple, tuple[str, ...], float]] = []
        for st in nxt:
            key = (st[0], st[1])
            if key in seen:
                continue
            seen.add(key)
            pruned.append(st)
            if len(pruned) >= beam:
                break
        frontier = pruned
    solutions.sort(key=lambda t: t.score, reverse=True)
    return solutions
