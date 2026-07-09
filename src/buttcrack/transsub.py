"""Crack a columnar transposition applied OVER a periodic substitution.

This is the mirror of :mod:`buttcrack.layered` (which handles substitution-over-
transposition).  Here the transposition is the OUTER layer, so it *hides* the inner
substitution's period from the raw ciphertext.  The lever is a **mapping-independent
discriminator**: undo the right transposition and a clean periodic index-of-coincidence
spike reappears (because the inner substitution's per-period columns become monoalphabetic
again), regardless of which keyed alphabet the substitution used.

Single columnar orders are recovered by a keyword/dictionary sweep; double columnar
by simulated annealing over the two read-orders with the reveal-IoC fitness —
which, unlike a quadgram objective, has a usable gradient *before* the substitution is
solved.  Once the transposition is undone, the exposed periodic substitution is solved with
the existing Quagmire solver.

KEY LESSON (validated): the reveal-IoC discriminator cleanly separates the true transposition
on synthetic instances, but it only works when the *inner substitution's effective period
repeats* within the message (period < ~length/4).  When the effective keystream does not
repeat (a long random key, or two coprime substitutions whose lcm exceeds the length), the
construction is one-time-pad-grade and not recoverable blind.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import permutations
from typing import Any

from .analysis import search_aware_null
from .ciphers import incomplete_columnar as icol
from .ciphers._quagmire_solver import dictionary_attack
from .ciphers.columnar import _decode_letters, _decode_units, _read_order
from .ciphers.quagmire3 import keyed_alphabet
from .layered import solve_inner_periodic, solve_inner_periodic_screen
from .scoring import NgramScorer
from .text import only_letters
from .words import long_word_coverage


def _ioc(s: str) -> float:
    n = len(s)
    if n < 2:
        return 0.0
    counts = Counter(s)
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def _col_ioc(s: str, period: int) -> float:
    return sum(_ioc(s[j::period]) for j in range(period)) / period


def reliable_period_cap(n: int, lo: int = 3, min_col: int = 16) -> int:
    """Largest inner period whose per-column IoC is statistically trustworthy at length ``n``.

    Each column needs >=~``min_col`` letters or the IoC estimate is noise a blind order
    search will overfit. So the cap *scales with the message*: a period the text is simply
    too short to reveal (e.g. period 36 at 279 letters -> 7.8 letters/col) is an
    information limit, not a bug — surface it with this so callers don't mistake "capped"
    for "no long period exists". (See :func:`~buttcrack.analysis.crackability_cliff`, which
    estimates the effective period straight from IoC even when it's past this cap.)
    """
    return max(lo, n // min_col)


def reveal_score(text: str, lo: int = 3, hi: int | None = None) -> tuple[float, int]:
    """Best per-column IoC over candidate inner periods (English ~0.066, random ~0.038).
    A high score means undoing the transposition exposed a periodic substitution.

    The upper period is capped by :func:`reliable_period_cap` so every column keeps
    >=~16 letters: short columns make the IoC estimate noisy and a blind search will
    happily overfit a spurious high-period spike (validated failure mode) instead of the
    true low-period structure. ``hi`` defaults to that length-scaled cap (so long messages
    are NOT artificially blinded at 18); pass an explicit ``hi`` only to look lower.
    """
    cap = reliable_period_cap(len(text), lo)
    top = cap if hi is None else min(hi, cap)
    best = (0.0, 0)
    for p in range(lo, top + 1):
        v = _col_ioc(text, p)
        if v > best[0]:
            best = (v, p)
    return best


#: Widest column count whose full read-order enumeration (w! decodes) is tractable.
#: 8! = 40320 reveal evaluations per width is the ceiling; above it the order has to be
#: recovered by the SA path in :func:`crack_transposition_over_sub` instead.
ENUM_MAX_WIDTH = 8


def _best_reveal_for_width(letters: str, width: int, unit: int = 1) -> float:
    """Best reveal-IoC over every read-order of an (incomplete) width-``width`` columnar.

    This is the *search* statistic the enum attack maximises; phrased as a pure
    ``str -> float`` so :func:`~buttcrack.analysis.search_aware_null` can replay the same
    full enumeration on shuffles of the same letters to calibrate selection bias. ``unit``
    is the transposition granularity (``unit=3`` transposes trigraph blocks).
    """
    best = 0.0
    for perm in permutations(range(width)):
        rv, _ = reveal_score(_undo_columnar(letters, list(perm), incomplete=False, unit=unit))
        if rv > best:
            best = rv
    return best


def _calibrate_reveal_null(
    letters: str, search, *, samples: int, seed: int, unit: int = 1
) -> dict[str, Any]:
    """Run ``search`` (best reveal-IoC its order search achieves on a letter string) on the
    real text and on shuffles, and label the verdict.

    A high best-of-search reveal is selection-biased: trying many orders and keeping the
    maximum lifts even structureless text.  The reported transposition is only signal when
    it BEATS the shuffled-search null (``beats_null_max`` or a clear ``z``); otherwise the
    reveal sits ``within null (overfit)`` and no transposition layer was actually found.
    """
    null = search_aware_null(letters, search, samples=samples, seed=seed, unit=unit)
    null["verdict"] = (
        "beats null" if (null["beats_null_max"] or null["z"] >= 3.0) else "within null (overfit)"
    )
    return null


def _enum_reveal_null(
    letters: str, widths, *, samples: int, seed: int, unit: int = 1
) -> dict[str, Any]:
    """Search-aware null for the FULL w! order enumeration over ``widths`` (at ``unit``
    granularity)."""
    enum_widths = [w for w in widths if 2 <= w <= len(letters) and w <= ENUM_MAX_WIDTH]

    def search(s: str) -> float:
        return max((_best_reveal_for_width(s, w, unit) for w in enum_widths), default=0.0)

    return _calibrate_reveal_null(letters, search, samples=samples, seed=seed, unit=unit)


def _keyword_reveal_null(
    letters: str, orders: list[list[int]], *, samples: int, seed: int, unit: int = 1
) -> dict[str, Any]:
    """Search-aware null for a *keyword sweep*: replay the same fixed set of read-``orders``
    on shuffles (far cheaper than the factorial enumeration, and the correct null for the
    keyword-driven search :func:`crack_transposition_over_sub` actually runs). ``unit`` must
    match the search granularity so the null undoes the same (block) transposition."""

    def search(s: str) -> float:
        return max(
            (reveal_score(_undo_columnar(s, o, incomplete=False, unit=unit))[0] for o in orders),
            default=0.0,
        )

    return _calibrate_reveal_null(letters, search, samples=samples, seed=seed)


def _sa_reveal_null(
    ct: str, width: int, *, restarts: int, iters: int, samples: int, seed: int
) -> dict[str, Any]:
    """Search-aware null for the blind DOUBLE-columnar SA (the most overfit-prone path).

    The SA maximises reveal-IoC over two read-orders — a large space — so its best is
    selection-biased and looks like structure even on noise (the validated CODEX trap).
    The honest null is the SAME SA run on shuffles of the same letters; cheaper restart/
    iter budgets keep it affordable while still calibrating the maximum.
    """

    def search(s: str) -> float:
        rng = random.Random(seed)
        return _sa_double(s, width, restarts=restarts, iters=iters, rng=rng)[0]

    return _calibrate_reveal_null(ct, search, samples=samples, seed=seed)


def _undo_double(ct: str, o1: list[int], o2: list[int]) -> str:
    return _decode_letters(_decode_letters(ct, o2), o1)


def _solve_inner(
    stream: str,
    scorer: NgramScorer,
    *,
    alphabet: str,
    period_hint: int | None,
    periods=range(2, 16),
) -> dict[str, Any]:
    """Solve an exposed periodic substitution on a de-transposed ``stream``.

    Tries the fast Quagmire-III keyword ``dictionary_attack`` first (cheap, and the
    common Kryptos-family case), then falls back to the generic
    :func:`~buttcrack.layered.solve_inner_periodic` so the inner layer can also be a
    standard-alphabet Vigenere or a Beaufort/variant over the keyed alphabet. Returns a
    dict with ``plaintext``, ``score``, ``coverage`` and a ``substitution`` descriptor.
    """
    inner = dictionary_attack(
        stream,
        scorer,
        "Q3",
        forced_period=period_hint if period_hint and period_hint >= 2 else None,
    )
    if inner is not None:
        score, plaintext, sub_period, shifts, keyword = inner
        cov = long_word_coverage(plaintext)
        if cov >= 0.40:
            return {
                "plaintext": plaintext,
                "score": score,
                "coverage": cov,
                "substitution": f"quagmire/{alphabet}",
                "convention": "vigenere",
                "period": sub_period,
                "substitution_keyword": keyword,
                "shifts": shifts,
                "method": "dictionary",
            }
    # Generic fallback: any (alphabet, convention, period) — recovers Beaufort/variant
    # and standard-alphabet Vigenere that the keyed-Q3 dictionary cannot.
    use_periods = [period_hint] if period_hint and period_hint >= 2 else list(periods)
    per_char, plaintext, meta = solve_inner_periodic(
        stream, scorer, alphabets=(alphabet, "STD"), periods=use_periods
    )
    cov = long_word_coverage(plaintext)
    return {
        "plaintext": plaintext,
        "score": per_char * max(1, len(stream)),
        "coverage": cov,
        "substitution": f"{meta.get('convention', '?')}/{meta.get('alphabet', alphabet)}",
        "convention": meta.get("convention"),
        "period": meta.get("period"),
        "substitution_keyword": None,
        "shifts": meta.get("shifts"),
        "method": "generic-periodic",
    }


def _sa_double(
    ct: str, width: int, *, restarts: int, iters: int, rng: random.Random
) -> tuple[float, list[int], list[int], int]:
    """Blind simulated annealing over two width-``width`` read-orders, maximising the
    reveal-IoC of the doubly-untransposed text."""
    best = (-1.0, list(range(width)), list(range(width)), 0)
    for _ in range(restarts):
        o1 = list(range(width))
        o2 = list(range(width))
        rng.shuffle(o1)
        rng.shuffle(o2)
        cur, _ = reveal_score(_undo_double(ct, o1, o2))
        local = (cur, o1[:], o2[:])
        for it in range(iters):
            temp = max(0.0008, 0.012 * (1 - it / iters))
            which = o1 if rng.random() < 0.5 else o2
            a, b = rng.randrange(width), rng.randrange(width)
            which[a], which[b] = which[b], which[a]
            score, _ = reveal_score(_undo_double(ct, o1, o2))
            if score >= cur or rng.random() < math.exp((score - cur) / temp):
                cur = score
                if score > local[0]:
                    local = (score, o1[:], o2[:])
            else:
                which[a], which[b] = which[b], which[a]
        # deterministic polish on the best of this restart
        o1, o2 = local[1], local[2]
        cur = local[0]
        improved = True
        while improved:
            improved = False
            for which in (o1, o2):
                for a in range(width):
                    for b in range(a + 1, width):
                        which[a], which[b] = which[b], which[a]
                        score, _ = reveal_score(_undo_double(ct, o1, o2))
                        if score > cur + 1e-9:
                            cur, improved = score, True
                        else:
                            which[a], which[b] = which[b], which[a]
        if cur > best[0]:
            rv, period = reveal_score(_undo_double(ct, o1, o2))
            best = (cur, o1[:], o2[:], period)
    return best


def crack_transposition_over_sub(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    alphabet: str = "KRYPTOS",
    layers: int = 2,
    widths=range(7, 10),
    keywords: list[str] | None = None,
    sa_restarts: int = 200,
    sa_iters: int = 4000,
    null_samples: int = 24,
    null_seed: int = 20250615,
    unit: int = 1,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Recover a single or double columnar transposition layered OVER a periodic
    (Quagmire) substitution, then solve the exposed substitution.

    ``layers=1`` sweeps keyword/numeric single-columnar orders; ``layers=2`` runs the
    blind double-columnar SA.  Returns the best decrypt with a structure description and
    the reveal-IoC achieved (which gates confidence: < ~0.06 means no transposition was
    found and the construction is likely not recoverable blind).

    ``unit`` is the transposition granularity: ``unit=3`` moves 3-letter blocks (a
    trigraph-granular columnar, so trigrams survive intact). Supported on the ``layers=1``
    keyword sweep; the blind ``layers=2`` SA is letter-only (block-SA gains nothing — the
    objective is gradient-less) and rejects ``unit != 1``.
    """
    ct = only_letters(ciphertext)
    rng = rng or random.Random(0)
    if unit != 1 and layers != 1:
        raise ValueError("unit>1 (block transposition) is supported only with layers=1")

    best_reveal = -1.0
    candidates: list[tuple[str, dict[str, Any], int]] = []
    swept_orders: list[list[int]] = []  # every read-order tried (for the keyword null)

    if layers == 1:
        words = keywords or []
        for width in widths:
            orders: list[list[int]] = []
            for kw in words:
                if len(only_letters(kw)) == width:
                    orders.append(_read_order(kw))
            swept_orders.extend(orders)
            for order in orders:
                # unit=1 keeps the original complete-columnar behaviour; unit>1 uses the
                # ragged-aware block primitive (the incomplete flag is ignored there).
                undone = _undo_columnar(ct, order, incomplete=False, unit=unit)
                rv, period = reveal_score(undone)
                if rv > best_reveal:
                    best_reveal = rv
                    candidates = [
                        (
                            undone,
                            {
                                "layer_order": "transposition-over-substitution",
                                "transposition": "columnar",
                                "columnar_order": order,
                            },
                            period,
                        )
                    ]
    else:
        for width in widths:
            rv, o1, o2, period = _sa_double(
                ct, width, restarts=sa_restarts, iters=sa_iters, rng=rng
            )
            if rv > best_reveal:
                best_reveal = rv
                undone = _undo_double(ct, o1, o2)
                candidates = [
                    (
                        undone,
                        {
                            "layer_order": "transposition-over-substitution",
                            "transposition": "double-columnar",
                            "columnar_width": width,
                            "columnar_orders": [o1, o2],
                        },
                        period,
                    )
                ]

    if not candidates:
        return {
            "structure": None,
            "reveal_ioc": round(best_reveal, 4),
            "plaintext": "",
            "note": "no transposition candidate found",
        }

    undone, structure, period = candidates[0]
    header = keyed_alphabet(alphabet)
    inner = _solve_inner(
        undone,
        scorer,
        alphabet=alphabet,
        period_hint=period if period >= 2 else None,
    )
    plaintext = inner["plaintext"]
    score = inner["score"]
    keyword = inner["substitution_keyword"]
    structure["substitution"] = inner["substitution"]
    structure["period"] = inner["period"]
    structure["substitution_keyword"] = keyword

    # Honest confidence gate.  A high reveal-IoC alone is NOT a solve: a blind search over
    # a large order space (especially double columnar) can overfit a spurious periodicity.
    # The construction is only confirmed when the *exposed substitution solves to English*.
    coverage = long_word_coverage(plaintext)
    recovered = coverage >= 0.40
    result = {
        "structure": structure,
        "plaintext": plaintext,
        "score": score,
        "reveal_ioc": round(best_reveal, 4),
        "word_coverage": round(coverage, 3),
        "recovered": recovered,
        "header": header,
    }
    # Calibrate the reported reveal against a shuffled-search null only on a clean solve.
    # The keyword sweep tries a fixed set of read-orders, so the honest (and cheap) null
    # replays exactly those orders on shuffles — not a full factorial enumeration.
    if recovered and layers == 1 and swept_orders:
        result["reveal_null"] = _keyword_reveal_null(
            ct, swept_orders, samples=null_samples, seed=null_seed, unit=unit
        )
    elif layers == 2 and null_samples > 0 and "columnar_width" in structure:
        # The double-columnar SA is the most overfit-prone path, so ALWAYS calibrate it
        # (not just on a clean solve) — a tempting reveal that sits inside the shuffled-SA
        # null is the CODEX trap, and the caller needs that flag even when coverage is low.
        result["reveal_null"] = _sa_reveal_null(
            ct,
            structure["columnar_width"],
            restarts=min(sa_restarts, 30),
            iters=min(sa_iters, 600),
            samples=min(null_samples, 8),
            seed=null_seed,
        )
    return result


def crack_columnar_reveal_enum(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    widths=range(5, 11),
    alphabet: str = "KRYPTOS",
    unit: int = 1,
    top_orders: int = 8,
    null_samples: int = 24,
    null_seed: int = 20250615,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """FULLY enumerate the read-orders of an INCOMPLETE single columnar, ranked by the
    mapping-independent reveal-IoC, then solve the exposed periodic substitution.

    This closes the gap where the true columnar order is a *non-dictionary* permutation at
    a width that does not divide the length: a keyword sweep (see
    :func:`crack_transposition_over_sub` ``layers=1``) can never reach it, but for a small
    width ``w`` the whole ``w!`` order space is tractable to brute force.  For each width
    (``w <= 8``; wider factorials are guarded out) every read-order is undone and scored by
    :func:`reveal_score`; undoing the *true* order re-exposes the inner substitution's
    period as a per-column IoC spike regardless of the keyed alphabet used.

    The top ``top_orders`` reveal candidates are then each handed to the Quagmire
    ``dictionary_attack`` (the reveal-IoC ties across cyclic-equivalent orders, so the
    single best reveal is not always the one that solves) and the best by
    ``long_word_coverage`` is kept.  The result is gated honestly: ``recovered`` requires
    ``word_coverage >= 0.40`` *and* the best reveal must beat a shuffled-search null
    (``reveal_null``), else the spike is flagged ``within null (overfit)``.
    """
    ct = only_letters(ciphertext)
    rng = rng or random.Random(0)
    # Guard against w! blowups: ENUM_MAX_WIDTH (8) caps the per-width enumeration at 8!.
    enum_widths = [w for w in widths if 2 <= w <= len(ct) and w <= ENUM_MAX_WIDTH]

    # Collect (reveal, width, order, period, undone) for every read-order, keep the top few.
    scored: list[tuple[float, int, list[int], int, str]] = []
    best_reveal = -1.0
    for width in enum_widths:
        for perm in permutations(range(width)):
            order = list(perm)
            undone = _undo_columnar(ct, order, incomplete=False, unit=unit)
            rv, period = reveal_score(undone)
            scored.append((rv, width, order, period, undone))
            best_reveal = max(best_reveal, rv)
    scored.sort(key=lambda t: t[0], reverse=True)

    best: dict[str, Any] | None = None
    for rv, width, order, period, undone in scored[:top_orders]:
        inner = _solve_inner(
            undone,
            scorer,
            alphabet=alphabet,
            period_hint=period if period >= 2 else None,
        )
        plaintext = inner["plaintext"]
        score = inner["score"]
        keyword = inner["substitution_keyword"]
        coverage = inner["coverage"]
        if best is None or coverage > best["word_coverage"]:
            best = {
                "structure": {
                    "layer_order": "transposition-over-substitution",
                    "transposition": "columnar",
                    "columnar_width": width,
                    "columnar_order": order,
                    "substitution": inner["substitution"],
                    "period": inner["period"],
                    "substitution_keyword": keyword,
                },
                "plaintext": plaintext,
                "score": score,
                "reveal_ioc": round(rv, 4),
                "word_coverage": round(coverage, 3),
            }
        # Reveal-IoC ranks the true order at or near the top, so a clean English solve on an
        # early candidate is the answer — stop rather than burn the polish budget on the rest.
        if coverage >= 0.40:
            break

    if best is None:
        return {
            "structure": None,
            "plaintext": "",
            "reveal_ioc": round(best_reveal, 4),
            "recovered": False,
            "note": "no order exposed a solvable substitution",
        }

    # Search-aware null: a high best-of-enumeration reveal is only structure if it beats the
    # same full enumeration run on shuffles of these letters.
    null = _enum_reveal_null(ct, enum_widths, samples=null_samples, seed=null_seed, unit=unit)
    best["reveal_null"] = null
    best["header"] = keyed_alphabet(alphabet)
    best["structure"]["unit"] = unit
    best["recovered"] = best["word_coverage"] >= 0.40 and null["verdict"] == "beats null"
    return best


def _undo_columnar(text: str, order: list[int], *, incomplete: bool, unit: int = 1) -> str:
    """Undo one columnar layer, choosing the complete or incomplete primitive.

    ``unit > 1`` transposes ``unit``-letter blocks as indivisible tokens (e.g. ``unit=3``
    for a trigraph-granular columnar), via :func:`columnar._decode_units` which handles a
    ragged final block; ``unit == 1`` keeps the original letter-columnar behaviour exactly.
    """
    if unit != 1:
        return _decode_units(text, order, unit)
    if incomplete:
        return icol._decode_letters(text, order)
    return _decode_letters(text, order)


def sweep_known_alphabet(
    ct: str,
    orders,
    *,
    alphabet: str = "KRYPTOS",
    unit: int = 1,
    kind: str = "quagmire3",
    periods=range(6, 46),
    null_samples: int = 30,
    null_seed: int = 20250615,
    top: int = 10,
) -> dict[str, Any]:
    """Rank candidate transposition read-orders by fast-solving the inner sub under each.

    The non-overfitting decider for *recognition*-based transpositions: given a family of
    candidate read-orders (columnar widths, routes, key-derived orders), undo each at the
    chosen ``unit`` granularity and recover the inner periodic substitution's shifts with a
    KNOWN alphabet via :func:`quagmire_solver.solve_fixed_alphabet` (no annealing). Every
    candidate gets an instant, calibrated accept/reject: results are ranked by long-word
    coverage, and a shuffle null (solving random permutations of the same text) is the bar
    a genuine order must clear. This does NOT beat a flat blind objective by search — it
    makes any *hypothesised* low-entropy order confirm-or-die in one pass.

    Returns ``{"candidates": [...top...], "null": {...}}`` where each candidate carries
    ``order, score, period, word_coverage, recovered, plaintext``.
    """
    from .quagmire_solver import solve_fixed_alphabet

    text = only_letters(ct)
    if len(text) < 8:
        raise ValueError(
            "ciphertext too short for a meaningful inner-sub sweep (need >= 8 letters)"
        )
    periods = list(periods)
    cands: list[dict[str, Any]] = []
    for order in orders:
        order = list(order)
        undone = _undo_columnar(text, order, incomplete=False, unit=unit)
        r = solve_fixed_alphabet(undone, alphabet, kind=kind, periods=periods)
        cands.append(
            {
                "order": order,
                "score": round(r["score"], 4),
                "period": r["period"],
                "word_coverage": r["word_coverage"],
                "recovered": r["recovered"],
                "plaintext": r["plaintext"],
            }
        )
    cands.sort(key=lambda d: (-d["word_coverage"], -d["score"]))

    rng = random.Random(null_seed)
    toks = [text[i : i + unit] for i in range(0, len(text), unit)]
    null_cov: list[float] = []
    for _ in range(null_samples):
        shuffled = toks[:]
        rng.shuffle(shuffled)
        r = solve_fixed_alphabet("".join(shuffled), alphabet, kind=kind, periods=periods)
        null_cov.append(r["word_coverage"])
    null_cov.sort()
    null = {
        "samples": null_samples,
        "max": null_cov[-1] if null_cov else None,
        "p95": null_cov[min(len(null_cov) - 1, int(0.95 * len(null_cov)))] if null_cov else None,
        "mean": round(sum(null_cov) / len(null_cov), 4) if null_cov else None,
    }
    return {"candidates": cands[:top], "null": null}


def _keyword_orders(wordlist, length: int) -> list[tuple[str, list[int]]]:
    """``(keyword, read-order)`` pairs for every wordlist entry of the given length."""
    out: list[tuple[str, list[int]]] = []
    for kw in wordlist:
        letters = only_letters(kw)
        if len(letters) == length:
            out.append((kw.upper(), _read_order(letters)))
    return out


def crack_double_columnar_keywords(
    ct: str,
    scorer: NgramScorer,
    *,
    lengths,
    wordlist,
    alphabet: str = "KRYPTOS",
    period_band=range(11, 16),
    reveal_floor: float = 0.058,
    null_samples: int = 16,
    null_seed: int = 20250615,
    unit: int = 1,
) -> dict[str, Any]:
    """DIRECTED double-columnar keyword-pair sweep.

    ``unit`` is the transposition granularity (``unit=3`` = trigraph-granular block
    columnar, so trigrams survive both layers — block-transposition geometry).

    For each ordered pair of length-matched keywords drawn from ``wordlist``, undo both
    columnar layers (the OUTER ``o1`` then the inner ``o2`` — the encrypt applied them in
    the reverse order), pre-filter on the mapping-independent reveal-IoC (cheap, and high
    only when the *right* pair re-exposes the inner substitution's periodicity), then hand
    the survivors to :func:`~buttcrack.layered.solve_inner_periodic` via :func:`_solve_inner`
    so the exposed substitution can be Vigenere/Beaufort/variant over the keyed or standard
    alphabet. Gated honestly on ``word_coverage`` AND a search-aware null over the swept
    pairs. ``lengths`` is the set of keyword widths to consider (the two keywords share a
    width); pass several to sweep widths. Both complete and incomplete columnar are
    tried per pair (an inner layer is incomplete when the length is not a multiple of
    the width).

    Returns the best decrypt with its structure, reveal-IoC, coverage, and ``reveal_null``.
    """
    text = only_letters(ct)
    swept: list[tuple[list[int], list[int], bool]] = []  # (o1, o2, incomplete) tried
    best: dict[str, Any] | None = None
    best_reveal = -1.0

    period_list = [p for p in period_band if 2 <= p <= len(text)]

    for length in lengths:
        pairs = _keyword_orders(wordlist, length)
        for kw1, o1 in pairs:
            for kw2, o2 in pairs:
                for incomplete in (False,) if unit != 1 else (False, True):
                    inner_ct = _undo_columnar(text, o1, incomplete=incomplete, unit=unit)
                    undone = _undo_columnar(inner_ct, o2, incomplete=incomplete, unit=unit)
                    swept.append((o1, o2, incomplete))
                    rv, period = reveal_score(undone)
                    best_reveal = max(best_reveal, rv)
                    if rv < reveal_floor:
                        continue  # no periodicity re-exposed; not the right pair
                    # Lock the period to the reveal spike when it lands in the band (the
                    # cheap screen); otherwise scan the whole band with the full solve.
                    if period in period_list:
                        sc, plaintext, meta = solve_inner_periodic_screen(
                            undone,
                            scorer,
                            period=period,
                            alphabets=(alphabet, "STD"),
                        )
                    else:
                        sc, plaintext, meta = solve_inner_periodic(
                            undone,
                            scorer,
                            alphabets=(alphabet, "STD"),
                            periods=period_list or range(2, 16),
                        )
                    coverage = long_word_coverage(plaintext)
                    if best is None or coverage > best["word_coverage"]:
                        best = {
                            "structure": {
                                "layer_order": "transposition-over-substitution",
                                "transposition": "double-columnar",
                                "columnar_width": length,
                                "columnar_keywords": [kw1, kw2],
                                "columnar_orders": [o1, o2],
                                "incomplete": incomplete,
                                "substitution": f"{meta.get('convention', '?')}/"
                                f"{meta.get('alphabet', alphabet)}",
                                "convention": meta.get("convention"),
                                "period": meta.get("period"),
                            },
                            "plaintext": plaintext,
                            "score": sc * max(1, len(undone)),
                            "reveal_ioc": round(rv, 4),
                            "word_coverage": round(coverage, 3),
                            "shifts": meta.get("shifts"),
                        }
                    if coverage >= 0.45:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break

    if best is None:
        return {
            "structure": None,
            "plaintext": "",
            "reveal_ioc": round(best_reveal, 4),
            "recovered": False,
            "note": "no keyword pair re-exposed a solvable substitution",
        }

    # Search-aware null over the SAME swept pairs: a high reveal is only structure if it
    # beats undoing those same (o1, o2) on shuffles of these letters.
    triples = swept

    def search(s: str) -> float:
        return max(
            (
                reveal_score(
                    _undo_columnar(
                        _undo_columnar(s, o1, incomplete=inc, unit=unit),
                        o2,
                        incomplete=inc,
                        unit=unit,
                    )
                )[0]
                for o1, o2, inc in triples
            ),
            default=0.0,
        )

    null = _calibrate_reveal_null(text, search, samples=null_samples, seed=null_seed)
    best["reveal_null"] = null
    best["header"] = keyed_alphabet(alphabet)
    best["recovered"] = best["word_coverage"] >= 0.40 and null["verdict"] == "beats null"
    return best


def reveal_spectrum(
    ct: str,
    *,
    widths,
    periods=range(3, 16),
    units=(1,),
    null_samples: int = 16,
    null_seed: int = 20250615,
) -> dict[str, Any]:
    """Per-(width, unit) best reveal-IoC achievable by undoing a single columnar, with a verdict.

    For each width in ``widths`` (capped at :data:`ENUM_MAX_WIDTH` for the full ``w!``
    enumeration) and each granularity in ``units`` (1 = letters, 3 = trigraph blocks — the
    block-transposition shape), reports the best reveal-IoC over all read-orders and the period
    at which it peaks, plus a **granularity-matched** search-aware-null verdict (``beats null`` vs
    ``within null (overfit)``) so :command:`butt diagnose` can tell a real hidden-substitution-
    under-transposition layering from selection-bias noise, AND at what block size. Widths
    above the enumeration cap are skipped (recovered by SA, not enumerated).

    Returns ``{"widths": [{"width", "unit", "best_reveal", "period", "best_order", "verdict",
    "null"}], "raw_reveal": <reveal of untouched ct>, "best": <top row>}``.
    """
    text = only_letters(ct)
    raw_rv, _ = reveal_score(text)
    rows: list[dict[str, Any]] = []
    for unit in units:
        for width in widths:
            if not (2 <= width <= len(text)) or width > ENUM_MAX_WIDTH:
                continue
            best_rv, best_order, best_period = -1.0, list(range(width)), 0
            for perm in permutations(range(width)):
                order = list(perm)
                rv, period = reveal_score(_undo_columnar(text, order, incomplete=False, unit=unit))
                if rv > best_rv:
                    best_rv, best_order, best_period = rv, order, period
            null = _calibrate_reveal_null(
                text,
                lambda s, w=width, u=unit: _best_reveal_for_width(s, w, u),
                samples=null_samples,
                seed=null_seed,
                unit=unit,
            )
            rows.append(
                {
                    "width": width,
                    "unit": unit,
                    "best_reveal": round(best_rv, 4),
                    "period": best_period,
                    "best_order": best_order,
                    "verdict": null["verdict"],
                    "null": null,
                }
            )
    rows.sort(key=lambda r: r["best_reveal"], reverse=True)
    beats = [r for r in rows if r["verdict"] == "beats null"]
    best = beats[0] if beats else (rows[0] if rows else None)
    return {"raw_reveal": round(raw_rv, 4), "widths": rows, "best": best}
