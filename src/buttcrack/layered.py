"""Layered solver: periodic (Quagmire/Vigenere) substitution OVER a columnar
transposition — the class ``CT = Quag_p(columnar_W(PLAINTEXT))``.

This is the sibling of :func:`engine._layered_additive_crack` (which peels a
*short-period additive* substitution over a columnar by per-column chi-square)
and :func:`engine._layered_crib_crack` (mono-substitution over columnar). It
handles the harder case real layered puzzles exposed:

* a *keyed* alphabet (e.g. KRYPTOS) rather than the standard one, and
* a *long* period (key length comparable to ``len/5``), and
* an *unknown* columnar column-order.

The attack, validated end to end:

1. **De-sub seed (order-independent).** For a candidate period, recover each
   column's shift by chi-square against English monograms. A transposition
   preserves monogram frequencies, so this seed is *independent of the columnar
   order* — compute it once. It is only ~50% correct on short (≈5-letter)
   columns, but it is a free warm start.
2. **Columnar order by hill-climb.** For each candidate width, hill-climb the
   column read-order (random restarts; simulated annealing does NOT work here —
   the landscape is rugged but the true order is a strong local max) scoring each
   order by a fast quadgram refinement of the shifts from the seed. The true
   order's plaintext scores clearly highest.
3. **Shift recovery.** With the order fixed, recover the per-column shifts by
   quadgram coordinate-ascent plus the near-pair 2-opt finisher
   (:func:`_quagmire_solver._two_opt_polish`-style), with random restarts.

KEY HONEST LIMIT and the agent hook. When the period is long enough that each
column holds only ~5 letters, ~20% of columns are *entangled* (a quadgram window
straddles two columns) and their true shift is NOT the n-gram optimum, so no
amount of quadgram/5-gram search recovers them — recovery plateaus at ~80% and is
non-deterministic. Rather than silently emit the n-gram-optimal (wrong) letters,
:func:`column_alternatives` exposes, per substitution column, the top candidate
shifts and the exact plaintext slots that column controls *in context*, so the
driving agent (an LLM, which can judge real English where an n-gram counter
cannot) makes the final call. A crib (any contiguous true-plaintext span at a
known position) collapses the residual to an exact solve via
:func:`engine._layered_crib_crack`'s logic generalized here.
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import permutations
from typing import Any

from .ciphers import _quagmire_solver as qs
from .ciphers.columnar import _column_lengths
from .ciphers.quagmire3 import keyed_alphabet
from .scoring import NgramScorer, index_of_coincidence
from .text import only_letters
from .words import long_word_coverage

#: English monogram frequencies (%) for the order-independent chi-square seed.
_ENG = {
    "A": 8.2,
    "B": 1.5,
    "C": 2.8,
    "D": 4.3,
    "E": 12.7,
    "F": 2.2,
    "G": 2.0,
    "H": 6.1,
    "I": 7.0,
    "J": 0.15,
    "K": 0.77,
    "L": 4.0,
    "M": 2.4,
    "N": 6.7,
    "O": 7.5,
    "P": 1.9,
    "Q": 0.095,
    "R": 6.0,
    "S": 6.3,
    "T": 9.1,
    "U": 2.8,
    "V": 0.98,
    "W": 2.4,
    "X": 0.15,
    "Y": 2.0,
    "Z": 0.074,
}


def _out_to_pt(n: int, order: list[int]) -> list[int]:
    """Output position -> source plaintext index for a complete/incomplete columnar
    of the given column read-order (matches :mod:`buttcrack.ciphers.columnar`)."""
    width = len(order)
    lengths = _column_lengths(n, width)
    mapping: list[int] = []
    for c in order:
        for r in range(lengths[c]):
            mapping.append(c + r * width)
    return mapping


def _freqs_for(language: str | None) -> dict[str, float]:
    """Monogram frequencies (%) for ``language`` for the chi-square seed; English fallback.

    A transposition preserves monogram counts, so seeding the periodic substitution against the
    *payload's* language (not always English) matters for e.g. an Italian/Latin plaintext.
    """
    if not language or str(language).lower() == "english":
        return _ENG
    try:
        from .scoring import get_scorer

        sc = get_scorer("monograms", str(language).lower())
        freqs = {chr(65 + i): 0.0 for i in range(26)}
        for gram, lp in sc.log_probs.items():
            if len(gram) == 1 and "A" <= gram <= "Z":
                freqs[gram] = (10**lp) * 100.0
        for c, v in freqs.items():
            if v <= 0:  # chi-square divides by the expected count
                freqs[c] = 0.01
        return freqs
    except Exception:
        return _ENG


def _chi2(letters: str, freqs: dict[str, float] = _ENG) -> float:
    counts = Counter(letters)
    total = len(letters)
    if not total:
        return 0.0
    return (
        sum(
            (counts.get(chr(65 + i), 0) - total * freqs[chr(65 + i)] / 100) ** 2
            / (total * freqs[chr(65 + i)] / 100)
            for i in range(26)
        )
        / total
    )


def _chi_seed(
    ct: str, period: int, header: str, hpos: dict[str, int], freqs: dict[str, float] = _ENG
) -> list[int]:
    n = len(ct)
    seed = [0] * period
    for j in range(period):
        col = [ct[i] for i in range(j, n, period)]
        best = (1e18, 0)
        for sh in range(26):
            dec = "".join(header[(hpos[c] - sh) % 26] for c in col)
            score = _chi2(dec, freqs)
            if score < best[0]:
                best = (score, sh)
        seed[j] = best[1]
    return seed


def _fast_quad_table(scorer: NgramScorer) -> list[float]:
    table, n = qs._fast_table(scorer)
    if n != 4:
        raise ValueError("layered solver requires a quadgram scorer")
    return table


def _decoder(ct: str, header: str, period: int, order: list[int]):
    """Return (apply, n) where apply(shifts) -> list[int] standard-alphabet indices
    of the un-columnar'd plaintext for the given shifts."""
    n = len(ct)
    hpos = {c: i for i, c in enumerate(header)}
    hdr_std = [ord(header[k]) - 65 for k in range(26)]
    ctn = [hpos[c] for c in ct]
    o2p = _out_to_pt(n, order)
    pt = [0] * n

    def apply(shifts: list[int]) -> list[int]:
        for o in range(n):
            pt[o2p[o]] = hdr_std[(ctn[o] - shifts[o % period]) % 26]
        return pt

    return apply, n


def _qscore(idx: list[int], table: list[float]) -> float:
    s = 0.0
    for i in range(len(idx) - 3):
        s += table[((idx[i] * 26 + idx[i + 1]) * 26 + idx[i + 2]) * 26 + idx[i + 3]]
    return s


def _recover_shifts(
    ct: str,
    header: str,
    period: int,
    order: list[int],
    table: list[float],
    *,
    seed: list[int] | None = None,
    restarts: int = 0,
    rng=None,
    passes: int | None = None,
) -> tuple[float, list[int], list[int]]:
    """Quadgram coordinate-ascent (optionally bounded passes / random restarts) of the
    per-column shifts, scoring the full un-columnar'd plaintext. Returns
    ``(score, shifts, plain_idx)``."""
    apply, n = _decoder(ct, header, period, order)
    base = list(seed) if seed is not None else [0] * period

    def climb(init: list[int]) -> tuple[float, list[int]]:
        shifts = list(init)
        cur = _qscore(apply(shifts), table)
        improved, done = True, 0
        while improved and (passes is None or done < passes):
            improved = False
            done += 1
            for j in range(period):
                best = (cur, shifts[j])
                for x in range(26):
                    shifts[j] = x
                    s = _qscore(apply(shifts), table)
                    if s > best[0]:
                        best = (s, x)
                shifts[j] = best[1]
                if best[0] > cur:
                    cur, improved = best[0], True
        return cur, shifts

    best_s, best = climb(base)
    rng = rng or random.Random(0)
    for _ in range(restarts):
        s, sh = climb([rng.randrange(26) for _ in range(period)])
        if s > best_s:
            best_s, best = s, sh
    return best_s, best, apply(best)


def _hillclimb_order(
    ct: str,
    header: str,
    period: int,
    width: int,
    table: list[float],
    seed: list[int],
    *,
    restarts: int,
    rng,
) -> tuple[float, list[int]]:
    """Find the columnar read-order by swap hill-climb with random restarts, scoring
    each order by a **deterministic full** quadgram coordinate-ascent of the shifts
    from the (order-independent) chi-square seed.

    IMPORTANT: the recovery must run to *convergence* (``passes=None``), not a fixed
    few passes. A truncated recovery under-converges and ranks a near-miss order as
    best (it scores almost as high as the truth), so the true order is never found —
    this exact mistake made an earlier version plateau at ~80% on a real puzzle.
    With a full deterministic climb the true order separates
    cleanly (e.g. ~-4.1/quadgram vs ~-5.1 for wrong orders on a period-45 width-8
    instance), with no restart noise to muddy the comparison."""

    def objective(order: list[int]) -> float:
        s, _, _ = _recover_shifts(ct, header, period, order, table, seed=seed)
        return s

    def climb(order: list[int]) -> tuple[float, list[int]]:
        order = order[:]
        cur = objective(order)
        improved = True
        while improved:
            improved = False
            for a in range(width):
                for b in range(a + 1, width):
                    order[a], order[b] = order[b], order[a]
                    s = objective(order)
                    if s > cur:
                        cur, improved = s, True
                    else:
                        order[a], order[b] = order[b], order[a]
        return cur, order

    best = climb(list(range(width)))
    for _ in range(restarts):
        start = list(range(width))
        rng.shuffle(start)
        cand = climb(start)
        if cand[0] > best[0]:
            best = cand
    return best


#: Exhaustive order brute is used up to this width (8! = 40320). Wider columnars fall
#: back to hill-climb (the order space is too large to enumerate).
_BRUTE_MAX_WIDTH = 8


def _order_chunk(args):
    """Worker: best (score, order, plaintext) over all column orders that START with
    column ``first`` (one chunk of the exhaustive brute), scored by the deterministic
    full-climb objective. Top-level so it is picklable for ProcessPoolExecutor; rebuilds
    its scoring state from picklable args (spawn-safe)."""
    ct, alphabet, period, width, first, language = args
    from .scoring import get_scorer

    header = keyed_alphabet(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    table = _fast_quad_table(get_scorer("quadgrams", language or "english"))
    seed = _chi_seed(ct, period, header, hpos, _freqs_for(language))
    best: tuple[float, list[int], str] = (-1e18, [], "")
    rest = [c for c in range(width) if c != first]
    for tail in permutations(rest):
        order = [first, *tail]
        sc, _, plain = _recover_shifts(ct, header, period, order, table, seed=seed)
        if sc > best[0]:
            best = (sc, order, "".join(chr(65 + x) for x in plain))
    return best


def _brute_order(
    ct: str,
    alphabet: str,
    period: int,
    width: int,
    workers: int | None,
    language: str = "english",
):
    """Exhaustively find the best column read-order (deterministic full-climb objective),
    parallelised by first column. Returns (score, order, plaintext)."""
    args = [(ct, alphabet, period, width, f, language) for f in range(width)]
    if workers == 1 or width <= 5:
        results = [_order_chunk(a) for a in args]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_order_chunk, args))
    return max(results, key=lambda r: r[0])


def detect_periods(ciphertext: str, *, top: int = 3, z_min: float = 2.0) -> list[int]:
    """Candidate substitution periods for a substitution-OVER-transposition layering.
    Because the substitution is the outer layer, its period shows in the raw-ciphertext
    column-IoC spectrum (z-scored vs random). Returns the most significant periods."""
    from .analysis import calibrated_periods

    ranked = calibrated_periods(only_letters(ciphertext), top=max(top, 6))
    sig = [d["period"] for d in ranked if d["z"] >= z_min][:top]
    return sig or ([ranked[0]["period"]] if ranked else [])


def _lang_ioc(freqs: dict[str, float]) -> float:
    """Expected index of coincidence of a language from its monogram frequencies (%)."""
    return sum((v / 100.0) ** 2 for v in freqs.values())


def _ioc_max_peel(ct: str, header: str, period: int, *, restarts: int = 30, rng=None) -> float:
    """Best full-text IoC reachable by a free period-``p`` shift ascent (the DEGENERATE peel).

    Exposed only for contrast: pure IoC maximisation drifts onto a one-letter spike, so a high
    value here does NOT mean the substitution was recovered. Compare against the chi-square peel.
    """
    rng = rng or random.Random(0)
    hpos = {c: i for i, c in enumerate(header)}
    cn = [hpos[c] for c in ct]
    n = len(ct)

    def ioc_of(shifts: list[int]) -> float:
        dec = "".join(header[(cn[i] - shifts[i % period]) % 26] for i in range(n))
        return index_of_coincidence(dec)

    best = 0.0
    for _ in range(restarts):
        sh = [rng.randrange(26) for _ in range(period)]
        cur = ioc_of(sh)
        improved = True
        while improved:
            improved = False
            for j in range(period):
                bj, bs = sh[j], cur
                for s in range(26):
                    sh[j] = s
                    v = ioc_of(sh)
                    if v > bs:
                        bs, bj = v, s
                sh[j] = bj
                if bs > cur + 1e-12:
                    cur, improved = bs, True
        best = max(best, cur)
    return best


def substitution_over_transposition_test(
    ciphertext: str,
    *,
    period: int,
    alphabet: str = "KRYPTOS",
    language: str = "english",
    overfit_gap: float = 0.010,
) -> dict[str, Any]:
    """Is ``CT`` a periodic substitution laid OVER a transposition (substitution outer)?

    Peels the period-``p`` substitution **properly** — per-column chi-square against
    ``language``'s monogram frequencies, which recovers the English/Italian-*shaped* key rather
    than the degenerate one-letter spike — and reports whether the residual's whole-text IoC
    **snaps to language level** (⇒ the layer beneath the substitution is *transposed language*:
    finish with a transposition solve, e.g. :func:`crack_layered`) or **stays flat** (⇒ a
    *flattening* inner such as a fractionation/polygraphic/paired-square cipher, or the period is
    wrong).

    Also runs a free IoC-MAX peel for contrast (:func:`_ioc_max_peel`) and sets
    ``overfit_warning`` when it beats the chi-square peel by more than ``overfit_gap`` — the
    signature of the one-letter-spike overfit that fakes an IoC "snap". **Trust the chi-square
    number, never the IoC-max one.**

    Returns a dict: ``period, alphabet, language, chi_seed_ioc, ioc_max, language_ioc, floor,
    gap_to_language, verdict, overfit_warning, note``.
    """
    ct = only_letters(ciphertext)
    header = _ALPHABETS.get(alphabet.upper(), keyed_alphabet(alphabet))
    hpos = {c: i for i, c in enumerate(header)}
    freqs = _freqs_for(language)
    lang_ioc = _lang_ioc(freqs)
    floor = 1.0 / 26

    seed = _chi_seed(ct, period, header, hpos, freqs)
    residual = "".join(header[(hpos[c] - seed[i % period]) % 26] for i, c in enumerate(ct))
    chi_ioc = index_of_coincidence(residual)  # whole-text: "does it snap to language?"
    # coset IoC (shift-invariant, = raw ct's) is the "is there period-p structure?" signal;
    # it can exceed the whole-text IoC when the peeled cosets have divergent distributions.
    coset_ioc = sum(index_of_coincidence(ct[j::period]) for j in range(period)) / period
    ioc_max = _ioc_max_peel(ct, header, period)

    gap = (chi_ioc - floor) / (lang_ioc - floor) if lang_ioc > floor else 0.0
    flattener_floor = floor + 0.35 * (lang_ioc - floor)
    if chi_ioc >= lang_ioc - 0.010:
        verdict = "substitution-over-transposition"
        note = (
            f"chi-square peel snaps whole-text IoC to {chi_ioc:.4f} ~ {language} "
            f"{lang_ioc:.4f}: the layer under the period-{period} substitution is transposed "
            f"language — finish with a transposition solve (crack_layered)."
        )
    elif coset_ioc >= flattener_floor:
        verdict = "substitution-over-flattener"
        note = (
            f"period-{period} coset IoC {coset_ioc:.4f} is real (digraphic/flattener band, "
            f"above floor {floor:.4f}) but the peeled whole-text IoC {chi_ioc:.4f} does NOT "
            f"snap to {language} {lang_ioc:.4f}: the inner is a flattening cipher "
            f"(fractionation / polygraphic / paired-square), not language."
        )
    else:
        verdict = "no-structure-or-wrong-period"
        note = (
            f"period-{period} coset IoC {coset_ioc:.4f} ~ random floor {floor:.4f}: no real "
            f"period-{period} structure here (try another period/alphabet)."
        )

    overfit = ioc_max - chi_ioc > overfit_gap
    return {
        "period": period,
        "alphabet": alphabet,
        "language": language,
        "chi_seed_ioc": round(chi_ioc, 4),
        "coset_ioc": round(coset_ioc, 4),
        "ioc_max": round(ioc_max, 4),
        "language_ioc": round(lang_ioc, 4),
        "floor": round(floor, 4),
        "gap_to_language": round(gap, 3),
        "verdict": verdict,
        "overfit_warning": overfit,
        "note": note
        + (
            f" [OVERFIT WARNING: free IoC-max reaches {ioc_max:.4f} >> chi-square {chi_ioc:.4f} —"
            " a one-letter-spike artefact; ignore the IoC-max value.]"
            if overfit
            else ""
        ),
    }


# --------------------------------------------------------------------------- #
# Generic periodic-substitution inner solve (the workhorse the transposition   #
# crackers call once an outer transposition has been undone).                  #
# --------------------------------------------------------------------------- #

#: Alphabet names accepted by the generic inner solve. ``STD`` is the standard
#: A-Z; ``KRYPTOS`` is the keyed alphabet used throughout the Kryptos family.
_ALPHABETS: dict[str, str] = {
    "STD": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "KRYPTOS": keyed_alphabet("KRYPTOS"),
}


def _alphabet_header(name: str) -> str:
    """Resolve an alphabet *name* (``'STD'``/``'KRYPTOS'``/any keyword) to its header."""
    up = name.upper()
    if up in _ALPHABETS:
        return _ALPHABETS[up]
    return keyed_alphabet(up)


def _convention_index(convention: str, cidx: int, shift: int) -> int:
    """Plaintext alphabet-index for cipher alphabet-index ``cidx`` under ``shift``.

    Conventions match the family grammar (index space over the keyed alphabet):
    Vigenere ``p = c - k``; Beaufort ``p = k - c``; variant ``p = c + k``.
    """
    if convention == "vigenere":
        return (cidx - shift) % 26
    if convention == "beaufort":
        return (shift - cidx) % 26
    if convention == "variant":
        return (cidx + shift) % 26
    raise ValueError(f"unknown convention {convention!r}")


def _periodic_decoder(stream: str, header: str, convention: str, period: int):
    """Return ``(apply, n)`` where ``apply(shifts)`` is the standard-alphabet index
    list of the decrypt of ``stream`` under the given keyed alphabet + convention.

    ``stream`` is the (already de-transposed) ciphertext; column ``j`` uses
    ``shifts[j]``. The plaintext index is taken in the keyed alphabet then mapped
    back to a standard A-Z index for n-gram scoring.
    """
    hpos = {c: i for i, c in enumerate(header)}
    hdr_std = [ord(header[k]) - 65 for k in range(26)]
    cn = [hpos[c] for c in stream]
    n = len(stream)
    pt = [0] * n

    if convention == "vigenere":

        def apply(shifts: list[int]) -> list[int]:
            for i in range(n):
                pt[i] = hdr_std[(cn[i] - shifts[i % period]) % 26]
            return pt
    elif convention == "beaufort":

        def apply(shifts: list[int]) -> list[int]:
            for i in range(n):
                pt[i] = hdr_std[(shifts[i % period] - cn[i]) % 26]
            return pt
    elif convention == "variant":

        def apply(shifts: list[int]) -> list[int]:
            for i in range(n):
                pt[i] = hdr_std[(cn[i] + shifts[i % period]) % 26]
            return pt
    else:
        raise ValueError(f"unknown convention {convention!r}")

    return apply, n


def _chi_seed_generic(
    stream: str, header: str, convention: str, period: int, freqs: dict[str, float] = _ENG
) -> list[int]:
    """Per-column chi-square shift seed for an arbitrary keyed alphabet + convention.

    Each column is a monoalphabet, so the shift whose decrypt best matches English
    monogram frequencies is found independently per column. Order-independent for the
    transposition layer (a transposition preserves monogram counts), so a single pass
    over the columns warm-starts every order/width the caller tries.
    """
    hpos = {c: i for i, c in enumerate(header)}
    n = len(stream)
    cn = [hpos[c] for c in stream]
    seed = [0] * period
    for j in range(period):
        col = [cn[i] for i in range(j, n, period)]
        best = (1e18, 0)
        for sh in range(26):
            # plaintext letter = header[plaintext-index-in-keyed-alphabet]
            dec = "".join(header[_convention_index(convention, v, sh)] for v in col)
            score = _chi2(dec, freqs)
            if score < best[0]:
                best = (score, sh)
        seed[j] = best[1]
    return seed


def _refine_periodic(
    apply, period: int, table: list[float], seed: list[int]
) -> tuple[float, list[int], list[int]]:
    """Quadgram coordinate-ascent of the per-column shifts from ``seed`` to convergence.

    Returns ``(score, shifts, plain_idx)``. Deterministic — no restarts; the chi-square
    seed plus full coordinate-ascent is enough on a clean (single-layer) periodic stream.
    """
    shifts = list(seed)
    cur = _qscore(apply(shifts), table)
    improved = True
    while improved:
        improved = False
        for j in range(period):
            best = (cur, shifts[j])
            for x in range(26):
                shifts[j] = x
                s = _qscore(apply(shifts), table)
                if s > best[0]:
                    best = (s, x)
            shifts[j] = best[1]
            if best[0] > cur:
                cur, improved = best[0], True
    return cur, shifts, apply(shifts)


def _combo_solve(
    stream: str,
    scorer: NgramScorer,
    alphabet: str,
    convention: str,
    period: int,
    table: list[float],
) -> tuple[float, str, dict[str, Any]]:
    """Solve a single (alphabet, convention, period) combo on a de-transposed ``stream``.

    Returns ``(score_per_char, plaintext, meta)`` where ``meta`` carries the resolved
    header, the recovered per-column shifts, and the combo descriptor.
    """
    header = _alphabet_header(alphabet)
    seed = _chi_seed_generic(stream, header, convention, period)
    apply, n = _periodic_decoder(stream, header, convention, period)
    score, shifts, plain = _refine_periodic(apply, period, table, seed)
    plaintext = "".join(chr(65 + x) for x in plain)
    per_char = score / max(1, n)
    meta = {
        "alphabet": alphabet,
        "header": header,
        "convention": convention,
        "period": period,
        "shifts": shifts,
    }
    return per_char, plaintext, meta


def solve_inner_periodic(
    stream: str,
    scorer: NgramScorer,
    *,
    alphabets=("KRYPTOS", "STD"),
    conventions=("vigenere", "beaufort", "variant"),
    periods=range(2, 16),
) -> tuple[float, str, dict[str, Any]]:
    """Solve a periodic (Vigenere/Beaufort/variant) substitution over a *keyed or
    standard* alphabet on an already-de-transposed ``stream``.

    For every ``(alphabet, convention, period)`` combo: an order-independent per-column
    chi-square shift seed, then a deterministic quadgram coordinate-ascent refine; the
    best by quadgram-score-per-char wins. This is the inner-solve workhorse the
    transposition-over-substitution crackers call once the outer transposition is undone,
    generalising them beyond KRYPTOS-Q3-Vigenere. Returns ``(score_per_char, plaintext,
    meta)``; ``meta`` records the winning alphabet/convention/period/shifts/header.
    """
    table = _fast_quad_table(scorer)
    best: tuple[float, str, dict[str, Any]] | None = None
    for alphabet in alphabets:
        for convention in conventions:
            for period in periods:
                if period < 2 or period > len(stream):
                    continue
                cand = _combo_solve(stream, scorer, alphabet, convention, period, table)
                if best is None or cand[0] > best[0]:
                    best = cand
    if best is None:
        return float("-inf"), stream, {}
    return best


def solve_inner_periodic_screen(
    stream: str,
    scorer: NgramScorer,
    *,
    period: int,
    alphabets=("KRYPTOS", "STD"),
    conventions=("vigenere", "beaufort", "variant"),
) -> tuple[float, str, dict[str, Any]]:
    """Cheap variant of :func:`solve_inner_periodic` for a *locked* period.

    Chi-seed every (alphabet, convention) combo and rank them by the seed decrypt's
    quadgram score, then run the (expensive) coordinate-ascent refine on the single
    best-seeded combo only. Useful when an outer search has already fixed the period and
    each evaluation must be cheap (the inner loop of a keyword-pair sweep).
    """
    table = _fast_quad_table(scorer)
    if period < 2 or period > len(stream):
        return float("-inf"), stream, {}
    screened: tuple[float, str, str, list[int], str] | None = None
    for alphabet in alphabets:
        header = _alphabet_header(alphabet)
        for convention in conventions:
            seed = _chi_seed_generic(stream, header, convention, period)
            apply, n = _periodic_decoder(stream, header, convention, period)
            sc = _qscore(apply(seed), table) / max(1, n)
            if screened is None or sc > screened[0]:
                screened = (sc, alphabet, convention, seed, header)
    assert screened is not None
    _, alphabet, convention, _seed, _header = screened
    return _combo_solve(stream, scorer, alphabet, convention, period, table)


def crack_layered(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    alphabet: str = "KRYPTOS",
    periods: list[int] | None = None,
    widths=range(4, 9),
    workers: int | None = None,
    hill_restarts: int = 60,
    language: str | None = None,
    brute_max_width: int = 9,
    rng=None,
) -> dict[str, Any]:
    """Autonomous version of :func:`crack_quagmire_over_columnar`: auto-detect the
    substitution period from the raw ciphertext, then for each candidate period sweep
    columnar widths — bruting orders exhaustively up to ``brute_max_width`` (parallelised by
    first column), hill-climbing for wider — and return the best decrypt across the whole sweep.

    Defaults are tuned for the common layered shape (period up to ~len/5, small
    columnar). ``workers=None`` uses all but two cores for the brute.

    ``language`` seeds the chi-square peel against that language's monogram frequencies (a
    transposition preserves monograms, so an Italian/Latin payload needs its own seed); it
    defaults to ``scorer.lang`` so passing a matching scorer is enough. ``brute_max_width``
    (default 9) is the widest column count enumerated exhaustively — width 9 (9! = 362 880) is
    tractable parallelised by first column; raise it only for a targeted run.
    """
    ct = only_letters(ciphertext)
    table = _fast_quad_table(scorer)
    if not language:  # None or "" -> fall back to the scorer's language
        language = getattr(scorer, "lang", "english")
    freqs = _freqs_for(language)
    rng = rng or random.Random(0)
    if workers is None:
        import os

        workers = max(1, (os.cpu_count() or 4) - 2)
    if periods is None:
        periods = detect_periods(ct)

    if not periods:
        raise ValueError("no candidate periods detected; pass periods explicitly")
    header = alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    best: tuple[float, int, int, list[int], str] | None = None
    for period in periods:
        seed = _chi_seed(ct, period, header, hpos, freqs)
        for width in widths:
            if width <= brute_max_width:
                sc, order, plaintext = _brute_order(ct, alphabet, period, width, workers, language)
            else:
                sc, order = _hillclimb_order(
                    ct, header, period, width, table, seed, restarts=hill_restarts, rng=rng
                )
                _, _, plain = _recover_shifts(ct, header, period, order, table, seed=seed)
                plaintext = "".join(chr(65 + x) for x in plain)
            if best is None or sc > best[0]:
                best = (sc, period, width, order, plaintext)
            # Early exit: once a width/period yields clean readable English there is no
            # point trying weaker periods (saves the expensive brute on wrong periods).
            if long_word_coverage(plaintext) >= 0.45:
                break
        else:
            continue
        break

    assert best is not None
    score, period, width, order, plaintext = best
    coverage = long_word_coverage(plaintext)
    # re-derive shifts for the winner (for the residual report / key)
    _, shifts, _ = _recover_shifts(
        ct, header, period, order, table, seed=_chi_seed(ct, period, header, hpos, freqs)
    )
    residual = (
        [] if coverage >= 0.42 else column_alternatives(ct, header, period, order, shifts, table)
    )
    return {
        "structure": {
            "layer_order": "substitution-over-transposition",
            "substitution": f"quagmire/{alphabet}",
            "period": period,
            "columnar_width": width,
            "columnar_order": order,
        },
        "plaintext": plaintext,
        "score": score,
        "word_coverage": round(coverage, 3),
        "shifts": shifts,
        "residual": residual,
    }


def alphabet_header(alphabet: str) -> str:
    return keyed_alphabet(alphabet)


def crack_quagmire_over_columnar(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    alphabet: str = "KRYPTOS",
    period: int,
    widths=range(5, 13),
    order: list[int] | None = None,
    order_restarts: int = 60,
    shift_restarts: int = 25,
    rng=None,
) -> dict[str, Any]:
    """Crack ``CT = Quagmire(alphabet, period)( columnar_W( PT ) )``.

    ``period`` is required (recover it first from the raw-ciphertext column-IoC
    spectrum — e.g. ``analysis.calibrated_periods`` / ``butt stats``; for a
    substitution-outer layering the period is visible in the raw ciphertext). If
    ``order`` is given the columnar search is skipped. Returns a dict with the
    recovered structure, plaintext, score, and per-column residual report.
    """
    ct = only_letters(ciphertext)
    header = keyed_alphabet(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    table = _fast_quad_table(scorer)
    rng = rng or random.Random(0)
    seed = _chi_seed(ct, period, header, hpos)

    if order is None:
        best: tuple[float, int, list[int], list[int], list[int]] | None = None
        for width in widths:
            _, ordr = _hillclimb_order(
                ct, header, period, width, table, seed, restarts=order_restarts, rng=rng
            )
            # full shift recovery to compare widths fairly
            full, shifts, plain = _recover_shifts(
                ct, header, period, ordr, table, seed=seed, restarts=shift_restarts, rng=rng
            )
            if best is None or full > best[0]:
                best = (full, width, ordr, shifts, plain)
        if best is None:
            raise ValueError("no candidate widths to search")
        score, width, order, shifts, plain = best
    else:
        width = len(order)
        score, shifts, plain = _recover_shifts(
            ct, header, period, order, table, seed=seed, restarts=shift_restarts, rng=rng
        )

    plaintext = "".join(chr(65 + x) for x in plain)
    # The per-column residual report is only useful when the solve is NOT already clean
    # English — when it reads, every column is decided and the report is just noise. Gate
    # it on long-word coverage (a clean solve tiles into real >=5-letter words).
    coverage = long_word_coverage(plaintext)
    residual = (
        [] if coverage >= 0.42 else column_alternatives(ct, header, period, order, shifts, table)
    )
    return {
        "structure": {
            "layer_order": "substitution-over-transposition",
            "substitution": f"quagmire/{alphabet}",
            "period": period,
            "columnar_width": width,
            "columnar_order": order,
        },
        "plaintext": plaintext,
        "score": score,
        "word_coverage": round(coverage, 3),
        "shifts": shifts,
        "residual": residual,
    }


def column_alternatives(
    ct: str,
    header: str,
    period: int,
    order: list[int],
    shifts: list[int],
    table: list[float],
    *,
    top: int = 3,
    gap: float = 60.0,
) -> list[dict[str, Any]]:
    """Agent-native residual report. For each substitution column whose best shift is
    a near-tie (within ``gap`` quadgram log-prob of an alternative), emit the top
    candidate shifts with the exact plaintext slots that column controls, each shown
    in context. An LLM driving the tool reads these and picks the shift that yields
    real words — the judgement an n-gram model cannot make. Columns with a clear
    winner are omitted (they are already decided)."""
    n = len(ct)
    o2p = _out_to_pt(n, order)
    p2o = [0] * n
    for o, p in enumerate(o2p):
        p2o[p] = o
    apply, _ = _decoder(ct, header, period, order)

    report: list[dict[str, Any]] = []
    for j in range(period):
        ranked = []
        for x in range(26):
            trial = list(shifts)
            trial[j] = x
            ranked.append((_qscore(apply(trial), table), x))
        ranked.sort(reverse=True)
        if ranked[0][0] - ranked[1][0] > gap:
            continue  # clear winner — already decided
        positions = sorted(p for p in range(n) if p2o[p] % period == j)
        options = []
        for _q, x in ranked[:top]:
            trial = list(shifts)
            trial[j] = x
            dec = "".join(chr(65 + v) for v in apply(trial))
            ctxs = [
                dec[max(0, p - 4) : p] + "[" + dec[p] + "]" + dec[p + 1 : p + 5] for p in positions
            ]
            options.append(
                {
                    "shift": x,
                    "letters": "".join(dec[p] for p in positions),
                    "contexts": ctxs,
                    "current": x == shifts[j],
                }
            )
        report.append({"column": j, "options": options})
    return report
