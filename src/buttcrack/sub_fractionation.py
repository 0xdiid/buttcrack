"""Decoupled recovery of a periodic substitution laid OVER a bifid fractionation.

The construction this attacks is ``CT = outer( bifid_inner( PT ) )`` where

* ``bifid_inner`` is a classic 5x5 bifid over a keyed square that DROPS one letter
  (default ``J``) and seriates at a fixed ``inner_period``; and
* ``outer`` is a periodic substitution — a Vigenere / Quagmire-III shift of period
  ``p`` over a keyed ``outer_alphabet`` (default the KRYPTOS keyed alphabet).

A joint blind search over ``key x square`` is hopeless (two coupled isolated optima).
The decoupling that makes it tractable is a razor-clean STRUCTURAL CONSTRAINT:

    a 25-cell bifid can never emit its dropped letter, so the intermediate stream
    (the residual left after stripping the outer substitution) is DROP-LETTER-FREE.

For a period-``p`` outer shift, each key position ``j`` acts on its own coset of the
ciphertext. A candidate shift ``s`` for column ``j`` is only admissible if stripping it
leaves that coset free of the drop letter — which prunes each of the ``p`` positions to a
SMALL candidate set (often 1-6 shifts on a few-hundred-letter message). Within those
admissible shifts the outer key is then recovered by constrained coordinate descent (or an
exhaustive scan when the admissible product is small), scoring the FULL de-bifided decode.

The discriminator is sharp: with the CORRECT square the recovered decode reads as language
(quadgram average ~ -4.3/char); with any wrong square it plateaus in the noise (~ -6.9). So
:func:`crack_sub_over_bifid` scans candidate squares (a keyword/dictionary/thematic set),
recovers the outer key under each, and ranks by the recovered decode's objective score.

Payload-agnostic objectives (:func:`make_objective`): the correct key for a
route/coordinate/number payload scores badly on English quadgrams, so the recovery
objective is selectable — ``"fitness"`` (entropy-normalized quadgram, max over a language
set), ``"ioc"`` (index of coincidence with a small quadgram tiebreak, the "iocitq" shape
that avoids the degenerate high-IoC keys a pure-IoC descent finds), or ``"repeats"``
(repeated bigram+trigram density, which catches repetitive route text language-agnostically).

Public API
----------
``sub_encode`` / ``sub_decode``                  periodic shift over a keyed alphabet
``encrypt_sub_over_bifid``                        plant the full CT = outer(bifid(PT))
``recover_outer_key_over_bifid``                  key-given-structure recovery (one square)
``crack_sub_over_bifid``                          driver over candidate squares + periods
``make_objective`` / ``rank_key``                 payload-agnostic scoring + final ranking
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from itertools import product

from .ciphers.bifid import _decode_letters, bifid_encode, square_alphabet
from .ciphers.squares import PolybiusSquare
from .scoring import LANGUAGES, get_scorer, index_of_coincidence
from .text import only_letters

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"


# --------------------------------------------------------------------------- #
# Alphabet / square / outer-substitution primitives
# --------------------------------------------------------------------------- #
def resolve_alphabet(spec: str) -> str:
    """Resolve ``spec`` to a 26-letter keyed alphabet.

    Accepts ``"KRYPTOS"``/``"STD"``, a full 26-letter permutation, or a keyword
    (expanded to the standard keyed alphabet: deduped keyword then the rest of A-Z).
    """
    s = "".join(ch for ch in str(spec).upper() if "A" <= ch <= "Z")
    if s in ("KRYPTOS", "KRY"):
        return _KRYPTOS
    if s in ("STD", "STANDARD", ""):
        return _STD
    if len(s) == 26 and set(s) == set(_STD):
        return s
    seen: list[str] = []
    for ch in s + _STD:
        if ch not in seen:
            seen.append(ch)
    return "".join(seen)


def resolve_square(item: str, drop_letter: str = "J") -> str:
    """Return the 25-letter row-by-row square string for a keyword or full permutation."""
    sq = PolybiusSquare(item, size=5, alphabet=square_alphabet(drop_letter))
    return "".join(sq.grid)


def sub_encode(intermediate: str, alphabet: str, shifts: Sequence[int]) -> str:
    """Periodic shift ENCODE over a keyed alphabet: ``c = A[(A.index(p) + s_j) % 26]``.

    This is a Vigenere (Quagmire-III when ``alphabet`` is keyed) over ``alphabet`` with
    per-column shifts ``shifts`` (period = ``len(shifts)``).
    """
    aidx = {ch: i for i, ch in enumerate(alphabet)}
    p = len(shifts)
    return "".join(alphabet[(aidx[ch] + shifts[i % p]) % 26] for i, ch in enumerate(intermediate))


def sub_decode(ct: str, alphabet: str, shifts: Sequence[int]) -> str:
    """Inverse of :func:`sub_encode`: ``p = A[(A.index(c) - s_j) % 26]``."""
    aidx = {ch: i for i, ch in enumerate(alphabet)}
    p = len(shifts)
    return "".join(alphabet[(aidx[ch] - shifts[i % p]) % 26] for i, ch in enumerate(ct))


def encrypt_sub_over_bifid(
    pt: str,
    square: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    inner_period: int = 7,
    outer_shifts: Sequence[int],
    drop_letter: str = "J",
) -> str:
    """Plant ``CT = periodic_sub( bifid( PT ) )`` — the exact structure this module attacks."""
    alpha = resolve_alphabet(outer_alphabet)
    intermediate = bifid_encode(pt, square, inner_period, drop_letter=drop_letter)
    return sub_encode(intermediate, alpha, outer_shifts)


# --------------------------------------------------------------------------- #
# Payload-agnostic objectives
# --------------------------------------------------------------------------- #
def _qnorm_factory(scorer) -> Callable[[str], float]:
    """A 0..1 quadgram tiebreak: how English-average the text is (clamped)."""
    lo, hi = scorer.floor, scorer._english_ref

    def qn(pt: str) -> float:
        if hi <= lo:
            return 0.0
        a = scorer.average(pt)
        return max(0.0, min(1.0, (a - lo) / (hi - lo)))

    return qn


def repeat_density(pt: str) -> float:
    """Repeated-bigram + repeated-trigram count per character (language-agnostic).

    Counts occurrences BEYOND the first of every bigram and trigram, normalized by
    length. Repetitive route/keyword payloads (``NORDNORDPASSIPASSI``...) score high;
    random text scores near zero. Invariant to the actual letters used.
    """
    letters = only_letters(pt)
    n = len(letters)
    if n < 3:
        return 0.0
    bi = Counter(letters[i : i + 2] for i in range(n - 1))
    tri = Counter(letters[i : i + 3] for i in range(n - 2))
    rep = sum(c - 1 for c in bi.values() if c > 1) + sum(c - 1 for c in tri.values() if c > 1)
    return rep / n


#: objective names understood by :func:`make_objective`
OBJECTIVES = ("fitness", "ioc", "repeats")


def make_objective(
    objective: str = "fitness",
    *,
    languages: Sequence[str] = ("english",),
    ioc_tiebreak: float = 0.01,
    repeat_tiebreak: float = 0.001,
) -> Callable[[str], float]:
    """Build a ``callable(plaintext) -> float`` (higher = better) for key recovery/ranking.

    * ``"fitness"`` — entropy-normalized quadgram fitness, taken as the MAX over
      ``languages`` (so a French/Latin/Italian payload is recognised too). Default set is
      English only; pass ``languages=LANGUAGE_SET`` for multi-language.
    * ``"ioc"`` — index of coincidence plus a small quadgram tiebreak (the "iocitq"
      shape). Pure IoC descent drifts onto degenerate high-IoC keys; the tiebreak keeps a
      genuine (route) decode on top when IoCs are close.
    * ``"repeats"`` — :func:`repeat_density` plus a tiny quadgram tiebreak; the language-
      agnostic choice for repetitive route/keyword plaintext.
    """
    objective = objective.lower()
    if objective == "fitness":
        langs = [lang for lang in languages if lang in LANGUAGES] or ["english"]
        scorers = [get_scorer("quadgrams", lang) for lang in langs]
        return lambda pt: max(s.fitness(pt) for s in scorers)
    if objective == "ioc":
        qn = _qnorm_factory(get_scorer("quadgrams", "english"))
        return lambda pt: index_of_coincidence(pt) + ioc_tiebreak * qn(pt)
    if objective == "repeats":
        qn = _qnorm_factory(get_scorer("quadgrams", "english"))
        return lambda pt: repeat_density(pt) + repeat_tiebreak * qn(pt)
    raise ValueError(f"objective must be one of {OBJECTIVES}; got {objective!r}")


# --------------------------------------------------------------------------- #
# Key-given-structure recovery (one candidate square)
# --------------------------------------------------------------------------- #
def _admissible_shifts(c_idx: list[int], n: int, p: int, drop_pos: int) -> list[list[int]]:
    """Per-column shifts that leave the residual coset free of the drop letter.

    Stripping shift ``s`` maps ciphertext index ``v`` to residual index ``(v - s) % 26``;
    that equals the drop position iff ``s == (v - drop_pos) % 26``. So the shifts banned
    for a column are exactly those values over the column's distinct ciphertext indices.
    """
    allowed: list[list[int]] = []
    for j in range(p):
        col_vals = {c_idx[i] for i in range(j, n, p)}
        banned = {(v - drop_pos) % 26 for v in col_vals}
        allowed_j = [s for s in range(26) if s not in banned]
        allowed.append(allowed_j or list(range(26)))
    return allowed


def recover_outer_key_over_bifid(
    ciphertext: str,
    square: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    inner_period: int = 7,
    outer_period: int,
    drop_letter: str = "J",
    objective: str = "fitness",
    languages: Sequence[str] = ("english",),
    objective_fn: Callable[[str], float] | None = None,
    restarts: int = 6,
    brute_cap: int = 4096,
    exhaustive: bool | None = None,
    rng=None,
    max_passes: int = 8,
) -> tuple[list[int], str, float]:
    """Recover the outer period-``p`` shift key GIVEN a candidate inner square.

    Uses the drop-letter constraint to prune each column's shifts to a small admissible
    set (:func:`_admissible_shifts`), then finds the best combination by objective score of
    the FULL de-bifided decode — exhaustively when the admissible product is small
    (``<= brute_cap`` or ``exhaustive=True``), else by constrained coordinate descent with
    random restarts. Returns ``(shifts, plaintext, objective_score)``.
    """
    import random as _random

    rng = rng or _random.Random(0)
    A = resolve_alphabet(outer_alphabet)
    aidx = {ch: i for i, ch in enumerate(A)}
    letters = only_letters(ciphertext)
    n = len(letters)
    p = int(outer_period)
    if n < p or p < 1:
        return ([0] * max(p, 1), "", float("-inf"))
    c_idx = [aidx[ch] for ch in letters]
    drop = str(drop_letter).upper()[:1]
    drop_pos = aidx[drop]

    obj = objective_fn or make_objective(objective, languages=languages)
    sq = PolybiusSquare(square, size=5, alphabet=square_alphabet(drop))

    def decode_for(shifts: Sequence[int]) -> str:
        residual = "".join(A[(c_idx[i] - shifts[i % p]) % 26] for i in range(n))
        return _decode_letters(residual, sq, inner_period)

    allowed = _admissible_shifts(c_idx, n, p, drop_pos)

    prod = 1
    for a in allowed:
        prod *= len(a)
        if prod > brute_cap:
            break

    if exhaustive or (exhaustive is None and prod <= brute_cap):
        best: tuple[float, list[int], str] | None = None
        for combo in product(*allowed):
            pt = decode_for(combo)
            score = obj(pt)
            if best is None or score > best[0]:
                best = (score, list(combo), pt)
        assert best is not None
        return best[1], best[2], best[0]

    # Constrained coordinate descent with restarts (large admissible product).
    def descend(init: list[int]) -> tuple[list[int], str, float]:
        shifts = list(init)
        best_pt = decode_for(shifts)
        best_obj = obj(best_pt)
        improved, passes = True, 0
        while improved and passes < max_passes:
            improved, passes = False, passes + 1
            for j in range(p):
                cur = shifts[j]
                loc_s, loc_obj, loc_pt = cur, best_obj, best_pt
                for s in allowed[j]:
                    if s == cur:
                        continue
                    shifts[j] = s
                    pt = decode_for(shifts)
                    sc = obj(pt)
                    if sc > loc_obj:
                        loc_obj, loc_s, loc_pt = sc, s, pt
                shifts[j] = loc_s
                if loc_obj > best_obj + 1e-12:
                    best_obj, best_pt, improved = loc_obj, loc_pt, True
        return shifts, best_pt, best_obj

    overall = descend([col[0] for col in allowed])
    for _ in range(max(0, restarts - 1)):
        init = [col[rng.randrange(len(col))] for col in allowed]
        cand = descend(init)
        if cand[2] > overall[2]:
            overall = cand
    return overall


# --------------------------------------------------------------------------- #
# Driver: scan candidate squares, recover the outer key under each, rank
# --------------------------------------------------------------------------- #
def _square_iter(squares, drop_letter: str) -> list[tuple[str, str]]:
    """Return a list of ``(label, square25)`` from a spec ('dictionary' or an iterable)."""
    if isinstance(squares, str) and squares.lower() in ("dictionary", "dict", "keywords"):
        from .ciphers._quagmire_solver import BUILTIN_KEYWORDS

        words: Iterable[str] = BUILTIN_KEYWORDS
        return [(w, resolve_square(w, drop_letter)) for w in words]
    out: list[tuple[str, str]] = []
    for item in squares:
        label = str(item)
        out.append((label, resolve_square(label, drop_letter)))
    return out


def _shifts_to_keystr(shifts: Sequence[int]) -> str:
    """Render per-column shifts as additive-offset letters (A=0..Z=25)."""
    return "".join(chr(65 + (s % 26)) for s in shifts)


def _drop_list(drop_letter) -> list[str]:
    """Resolve ``drop_letter`` (None/'J', a letter, a letter string, or 'sweep'/'all')."""
    if drop_letter is None:
        return ["J"]
    if isinstance(drop_letter, str):
        s = drop_letter.strip()
        if s.lower() in ("sweep", "all", "*"):
            return list(_STD)
        letters = [c for c in s.upper() if "A" <= c <= "Z"]
        return letters or ["J"]
    return [str(x).upper()[:1] for x in drop_letter]


def crack_sub_over_bifid(
    ciphertext: str,
    *,
    outer_alphabet: str = "KRYPTOS",
    inner_period: int = 7,
    outer_period: int | Iterable[int] | None = None,
    squares="dictionary",
    objective: str = "fitness",
    drop_letter: str | None = None,
    languages: Sequence[str] = ("english",),
    top: int = 5,
    timeout: float | None = None,
    rng=None,
    brute_cap: int = 4096,
) -> list[tuple[str, str, str, float]]:
    """Crack ``CT = periodic_sub( bifid( PT ) )`` by scanning candidate inner squares.

    For every candidate ``drop_letter``, square (a keyword/permutation, or the built-in
    ``"dictionary"`` keyword set) and ``outer_period``, recover the outer key by the
    constrained descent of :func:`recover_outer_key_over_bifid`, then rank all hypotheses
    by the recovered decode's ``objective`` score. Returns up to ``top`` tuples
    ``(square, key, plaintext, score)`` best-first — the correct square/period recovers a
    readable (or, for a route payload under ``ioc``/``repeats``, a structured) decode that
    sits far above the wrong-square plateau.

    ``outer_period`` may be an int, an iterable of ints, or ``None`` (sweep 2..12).
    ``drop_letter`` defaults to ``"J"``; pass a letter, a letter string, or ``"sweep"``/
    ``"all"`` to attack (or search for) a bifid that drops a letter other than J.
    """
    drops = _drop_list(drop_letter)
    letters = only_letters(ciphertext)
    obj = make_objective(objective, languages=languages)

    if outer_period is None:
        periods = list(range(2, 13))
    elif isinstance(outer_period, int):
        periods = [outer_period]
    else:
        periods = [int(x) for x in outer_period]

    deadline = (time.monotonic() + timeout) if timeout else None

    results: list[tuple[float, str, str, str, str]] = []  # (score, square, keystr, pt, drop)
    for drop in drops:
        if deadline and time.monotonic() > deadline:
            break
        sq_list = _square_iter(squares, drop)
        for _label, sq25 in sq_list:
            if deadline and time.monotonic() > deadline:
                break
            for p in periods:
                if deadline and time.monotonic() > deadline:
                    break
                shifts, pt, score = recover_outer_key_over_bifid(
                    letters,
                    sq25,
                    outer_alphabet=outer_alphabet,
                    inner_period=inner_period,
                    outer_period=p,
                    drop_letter=drop,
                    objective_fn=obj,
                    rng=rng,
                    brute_cap=brute_cap,
                )
                if not pt:
                    continue
                results.append((score, sq25, _shifts_to_keystr(shifts), pt, drop))

    results.sort(key=lambda r: r[0], reverse=True)
    # dedup by plaintext, keep best-scoring
    seen: set[str] = set()
    out: list[tuple[str, str, str, float]] = []
    for score, sq25, keystr, pt, _drop in results:
        if pt in seen:
            continue
        seen.add(pt)
        out.append((sq25, keystr, pt, score))
        if len(out) >= top:
            break
    return out


# --------------------------------------------------------------------------- #
# Final ranking (fitness + IoC + optional vocab hits)
# --------------------------------------------------------------------------- #
def rank_key(
    candidates,
    *,
    languages: Sequence[str] = ("english",),
    vocab: Sequence[str] | None = None,
    fitness_weight: float = 1.0,
    ioc_weight: float = 10.0,
    vocab_weight: float = 1.0,
) -> list[dict]:
    """Rank final decode candidates by a composite of fitness + IoC + vocab hits.

    ``candidates`` is an iterable of plaintext strings OR ``(square, key, plaintext, score)``
    tuples (as returned by :func:`crack_sub_over_bifid`). Returns a list of dicts
    ``{plaintext, fitness, ioc, vocab_hits, composite, square, key}`` sorted by ``composite``
    (``fitness_weight*fitness + ioc_weight*ioc + vocab_weight*vocab_hits``). ``vocab`` is an
    optional list of expected words (e.g. thematic tokens); a hit is a substring match.
    """
    langs = [lang for lang in languages if lang in LANGUAGES] or ["english"]
    scorers = [get_scorer("quadgrams", lang) for lang in langs]
    vocab_u = [only_letters(w).upper() for w in (vocab or []) if only_letters(w)]

    rows: list[dict] = []
    for item in candidates:
        square = key = None
        if isinstance(item, str):
            pt = item
        else:
            seq = list(item)
            # (square, key, plaintext, score) or (plaintext,)
            if len(seq) >= 3:
                square, key, pt = seq[0], seq[1], seq[2]
            else:
                pt = seq[-1]
        letters = only_letters(pt).upper()
        fit = max(s.fitness(pt) for s in scorers)
        ioc = index_of_coincidence(pt)
        hits = sum(letters.count(w) for w in vocab_u)
        composite = fitness_weight * fit + ioc_weight * ioc + vocab_weight * hits
        rows.append(
            {
                "plaintext": pt,
                "fitness": fit,
                "ioc": ioc,
                "vocab_hits": hits,
                "composite": composite,
                "square": square,
                "key": key,
            }
        )
    rows.sort(key=lambda r: r["composite"], reverse=True)
    return rows


#: convenience: the full bundled language set for multi-language ``fitness``
LANGUAGE_SET = LANGUAGES
