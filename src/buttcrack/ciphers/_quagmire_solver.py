"""Keyword dictionary attack shared by the Quagmire family (K1/K2/K3).

Hill-climbing a keyed alphabet from scratch is infeasible at ACA message lengths
(~20 letters/column): the correct alphabet is an isolated optimum the climb never
reaches, even on a known plaintext. The reliable keyless route is a *keyword*
dictionary attack:

1. fix the period from the per-column index of coincidence (each column is a
   monoalphabet, so columns look English only at the true period);
2. for every candidate keyed-alphabet keyword, build the keyed alphabet and
   recover the per-column Vigenere shifts by **quadgram coordinate-ascent over the
   full text** — chi-squared per column is too noisy at ~20 letters/column, but
   the full-text quadgram signal pins the shifts exactly once the alphabet is right;
3. keep the best-scoring decrypt.

Famous cipher/puzzle keywords (KRYPTOS first) are tried before common words so the
typical puzzle falls out in milliseconds even without ``--wordlist``.

A genuine keyword produces real English (high score); every wrong keyword stays in
the noise, so the right one stands out by a wide margin — the same property that
makes this far more reliable than a blind hill-climb.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..result import Candidate
from ..scoring import ENGLISH_MONOGRAM_FREQ, NgramScorer
from ..text import only_letters, reflow

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Below this many letters, keyless Quagmire recovery (even the dictionary attack's
#: per-column shift recovery) is unreliable — too few letters per column. We still
#: try, but flag the result so callers/agents don't over-trust it.
RELIABLE_MIN = 100

#: Blind keyed-alphabet recovery (no keyword) needs a long message: the alphabet is
#: an isolated optimum the anneal only reaches when each column has enough letters
#: for the per-column cycleword recovery to lock. Below this it's a coin-flip, so we
#: don't bother (the dictionary attack remains the route at ACA lengths).
BLIND_MIN = 200

# Tried before the common-word list so on-theme puzzles solve instantly. KRYPTOS
# (the CIA sculpture's keyed-Vigenere alphabet) is the single most common keyed
# alphabet in the wild, hence first.
FAMOUS_KEYWORDS: tuple[str, ...] = (
    "KRYPTOS",
    "PALIMPSEST",
    "ABSCISSA",
    "CRYPTOGRAPHY",
    "CRYPTOGRAM",
    "CRYPTOGRAPHIC",
    "CIPHER",
    "VIGENERE",
    "ENIGMA",
    "ZIMMERMANN",
    "ALPHABET",
    "KEYWORD",
    "PASSWORD",
    "SUBSTITUTION",
    "TRANSPOSITION",
    "ARCHIVE",
    "MYSTERY",
    "SHADOW",
    "PHANTOM",
    "ORACLE",
    "SECRET",
    "HIDDEN",
    "PUZZLE",
    "RIDDLE",
    "DECIPHER",
    "ENCRYPT",
    "PLAINTEXT",
    "CIPHERTEXT",
    "POLYALPHABETIC",
    "FREEMASON",
    "TEMPLAR",
    "LABYRINTH",
    "COMPASS",
    "MERIDIAN",
    "LATITUDE",
    "LONGITUDE",
    "EQUATION",
    "VARIABLE",
    "FUNCTION",
)

# A small high-frequency English word list as a fallback so a non-themed keyword
# that happens to be common still falls out without the user supplying one.
COMMON_KEYWORDS: tuple[str, ...] = (
    "ABOUT",
    "ABOVE",
    "AFTER",
    "AGAIN",
    "ANOTHER",
    "BECAUSE",
    "BEFORE",
    "BETWEEN",
    "CHANGE",
    "COUNTRY",
    "DURING",
    "ENGLAND",
    "FAMILY",
    "FATHER",
    "FOLLOW",
    "FRIEND",
    "GARDEN",
    "GENERAL",
    "GROUND",
    "HISTORY",
    "HOUSE",
    "JOURNEY",
    "KINGDOM",
    "LETTER",
    "LITTLE",
    "MACHINE",
    "MARKET",
    "MASTER",
    "MEMORY",
    "MORNING",
    "MOTHER",
    "NATURE",
    "NUMBER",
    "OCEAN",
    "OFFICE",
    "PEOPLE",
    "PERSON",
    "PICTURE",
    "POWER",
    "QUESTION",
    "REASON",
    "RIVER",
    "SCHOOL",
    "SCIENCE",
    "SEASON",
    "SILVER",
    "SIMPLE",
    "SPRING",
    "SUMMER",
    "AUTUMN",
    "WINTER",
    "TABLE",
    "THOUGHT",
    "TROUBLE",
    "VILLAGE",
    "WEATHER",
    "WINDOW",
    "WONDER",
    "YELLOW",
    "JANUARY",
    "OCTOBER",
)

#: default candidate keywords when the caller passes none (KRYPTOS first)
BUILTIN_KEYWORDS: tuple[str, ...] = FAMOUS_KEYWORDS + COMMON_KEYWORDS


def keyed_alphabet(keyword: str) -> str:
    """Deduped keyword followed by the unused A-Z letters (standard ACA keying)."""
    out: list[str] = []
    for ch in keyword.upper():
        if "A" <= ch <= "Z" and ch not in out:
            out.append(ch)
    for ch in _STD:
        if ch not in out:
            out.append(ch)
    return "".join(out)


# n-gram tables don't change inside a process, so build the flat quadgram lookup
# (base-26 indexed) once per scorer — the hot loop runs it millions of times.
_FAST_CACHE: dict[int, tuple[list[float], int]] = {}


def _fast_table(scorer: NgramScorer) -> tuple[list[float], int]:
    key = id(scorer)
    cached = _FAST_CACHE.get(key)
    if cached is not None:
        return cached
    n = scorer.n
    size = 26**n
    table = [scorer.floor] * size
    for gram, lp in scorer.log_probs.items():
        if len(gram) != n:
            continue
        code = 0
        ok = True
        for ch in gram:
            if not ("A" <= ch <= "Z"):
                ok = False
                break
            code = code * 26 + (ord(ch) - 65)
        if ok:
            table[code] = lp
    _FAST_CACHE[key] = (table, n)
    return table, n


def candidate_periods(letters: str, max_period: int | None, forced: int | None) -> list[int]:
    """Periods to try, true period first.

    Forced wins. Otherwise rank by *calibrated* per-column IoC (z vs a random
    baseline), which — unlike a naive IoC threshold — surfaces a long period whose
    short columns depress the absolute IoC (e.g. period 40 on 280 letters). Keep
    the statistically significant periods (z >= 2.5), else the top few.
    """
    if forced:
        return [int(forced)]
    from ..analysis import calibrated_periods

    cands = calibrated_periods(letters, max_period=max_period, top=6)
    if not cands:
        return [1]
    significant = [c["period"] for c in cands if c["z"] >= 2.5]
    return (significant or [c["period"] for c in cands[:4]])[:5]


def _build_pre_post(model: str, ctn: Sequence[int], keyed: str) -> tuple[list[int], list[int]]:
    """Per-position base values and the output map, so plain[i] = post[(pre[i]-shift)%26].

    Q1: keyed plaintext, straight cipher   -> p = keyed[(c - shift)]
    Q2: straight plaintext, keyed cipher   -> p = (posKeyed(c) - shift)
    Q3: same keyed alphabet both sides      -> p = keyed[(posKeyed(c) - shift)]
    """
    kidx = [ord(c) - 65 for c in keyed]  # keyed alphabet as letter indices
    inv = [0] * 26
    for pos, ch in enumerate(keyed):
        inv[ord(ch) - 65] = pos
    if model == "Q1":
        return list(ctn), kidx
    if model == "Q2":
        return [inv[c] for c in ctn], list(range(26))
    if model == "Q3":
        return [inv[c] for c in ctn], kidx
    raise ValueError(f"unknown quagmire model {model!r}")


def _recover_shifts(
    pre: Sequence[int],
    post: Sequence[int],
    period: int,
    cols: Sequence[Sequence[int]],
    table: list[float],
    *,
    restarts: int = 1,
    rng=None,
    deadline: float | None = None,
) -> tuple[float, list[int], list[int]]:
    """Quadgram hill-climb on the per-column shifts, with random restarts.

    Plain greedy coordinate-ascent gets trapped on long keys (a period-40 key on
    280 letters means 40 columns of only 7 letters, with many local optima), so we
    restart from random shift vectors and keep the best — restart 0 is the
    deterministic all-zero start. Returns (score, shifts, plain).
    """
    n = len(pre)
    plain = [0] * n

    def fill(col: Sequence[int], sh: int) -> None:
        for i in col:
            plain[i] = post[(pre[i] - sh) % 26]

    def score() -> float:
        s = 0.0
        a, b, c = plain[0], plain[1], plain[2]
        for i in range(3, n):
            d = plain[i]
            s += table[((a * 26 + b) * 26 + c) * 26 + d]
            a, b, c = b, c, d
        return s

    def climb(init: list[int]) -> tuple[float, list[int]]:
        shifts = list(init)
        for j in range(period):
            fill(cols[j], shifts[j])
        cur = score()
        improved = True
        while improved:
            improved = False
            for j in range(period):
                col = cols[j]
                best_sh, best_s = shifts[j], None
                for sh in range(26):
                    fill(col, sh)
                    s = score()
                    if best_s is None or s > best_s:
                        best_s, best_sh = s, sh
                shifts[j] = best_sh
                fill(col, best_sh)
                if best_s is not None and best_s > cur:
                    cur, improved = best_s, True
        return cur, shifts

    best_score, best_shifts = climb([0] * period)
    for _ in range(1, restarts):
        if deadline and time.monotonic() > deadline:
            break
        init = [rng.randrange(26) for _ in range(period)] if rng else [0] * period
        sc_, sh_ = climb(init)
        if sc_ > best_score:
            best_score, best_shifts = sc_, sh_
    for j in range(period):
        fill(cols[j], best_shifts[j])
    return best_score, best_shifts, plain


#: Guaranteed wall-clock budget (seconds) for the deterministic 2-opt finisher.
#: It is bounded work — only near column pairs over a few passes — so it gets its
#: own budget even when the caller's main deadline has already elapsed.
_TWO_OPT_BUDGET_S = 4.0


def _two_opt_polish(
    pre: Sequence[int],
    post: Sequence[int],
    period: int,
    cols: Sequence[Sequence[int]],
    table: list[float],
    shifts: Sequence[int],
    *,
    max_passes: int = 4,
    deadline: float | None = None,
) -> tuple[float, list[int], list[int]]:
    """Joint two-column refinement that escapes the coupled local optima 1-opt misses.

    Single-column coordinate ascent (:func:`_recover_shifts`) traps on long keys:
    two nearby columns can be *jointly* wrong while no single-column move improves
    either, so even many random restarts fail to converge to clean English. A
    quadgram window spans only four consecutive positions, so the only column pairs
    that can be jointly coupled are those within cyclic distance 3 — refining just
    those pairs (each over all 26x26 shift combinations) breaks the trap. Because a
    column change only touches the handful of quadgram windows over that column, the
    pair score is recomputed incrementally, making the whole pass cheap and
    deterministic. Starts from ``shifts`` and only ever improves, so the result is
    never worse than the 1-opt seed. Returns ``(score, shifts, plain)``.
    """
    n = len(pre)
    plain = [0] * n

    def fillj(j: int, sh: int) -> None:
        for i in cols[j]:
            plain[i] = post[(pre[i] - sh) % 26]

    shifts = list(shifts)
    for j in range(period):
        fillj(j, shifts[j])

    def full_score() -> float:
        s = 0.0
        a, b, c = plain[0], plain[1], plain[2]
        for i in range(3, n):
            d = plain[i]
            s += table[((a * 26 + b) * 26 + c) * 26 + d]
            a, b, c = b, c, d
        return s

    def windows_for(a: int, b: int) -> list[int]:
        ws: set[int] = set()
        for i in list(cols[a]) + list(cols[b]):
            for w in (i - 3, i - 2, i - 1, i):
                if 0 <= w <= n - 4:
                    ws.add(w)
        return sorted(ws)

    def sum_windows(ws: Sequence[int]) -> float:
        s = 0.0
        for w in ws:
            a, b, c, d = plain[w], plain[w + 1], plain[w + 2], plain[w + 3]
            s += table[((a * 26 + b) * 26 + c) * 26 + d]
        return s

    if n < 4 or period < 2:
        # No quadgram windows / no column pairs to refine: nothing to do.
        return (full_score() if n >= 4 else 0.0), shifts, plain

    pairs = [
        (a, b, windows_for(a, b))
        for a in range(period)
        for b in range(a + 1, period)
        if 0 < min(b - a, period - (b - a)) <= 3
    ]
    cur = full_score()
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved = False
        passes += 1
        for a, b, ws in pairs:
            if deadline is not None and time.monotonic() > deadline:
                break
            base = cur - sum_windows(ws)
            best_a, best_b, best_s = shifts[a], shifts[b], cur
            for sa in range(26):
                fillj(a, sa)
                for sb in range(26):
                    fillj(b, sb)
                    s = base + sum_windows(ws)
                    if s > best_s:
                        best_s, best_a, best_b = s, sa, sb
            shifts[a], shifts[b] = best_a, best_b
            fillj(a, best_a)
            fillj(b, best_b)
            if best_s > cur + 1e-9:
                cur = best_s
                improved = True
    return cur, shifts, plain


_ENG_FREQ = [ENGLISH_MONOGRAM_FREQ[chr(65 + i)] for i in range(26)]


def _chi_shifts(
    pre: Sequence[int],
    post: Sequence[int],
    period: int,
    cols: Sequence[Sequence[int]],
) -> list[int]:
    """Per-column cycleword shifts recovered by chi-square (the deterministic step).

    Given a candidate keyed alphabet (folded into ``pre``/``post``), each column is a
    Caesar shift; the shift whose decrypt best matches English monogram frequencies
    is found independently per column. Cheap (O(period x 26 x 26)) — this is the inner
    "cycleword is deterministic" recovery, leaving only the keyed alphabet to search.
    """
    shifts: list[int] = []
    for j in range(period):
        col = cols[j]
        m = len(col)
        if m == 0:
            shifts.append(0)
            continue
        hist = [0] * 26
        for i in col:
            hist[pre[i]] += 1
        best_sh, best_chi = 0, None
        for sh in range(26):
            cnt = [0] * 26
            for v in range(26):
                h = hist[v]
                if h:
                    cnt[post[(v - sh) % 26]] += h
            chi = 0.0
            for letter in range(26):
                exp = _ENG_FREQ[letter] * m
                diff = cnt[letter] - exp
                chi += diff * diff / exp
            if best_chi is None or chi < best_chi:
                best_chi, best_sh = chi, sh
        shifts.append(best_sh)
    return shifts


def _decode_quad_score(
    pre: Sequence[int],
    post: Sequence[int],
    shifts: Sequence[int],
    period: int,
    table: list[float],
) -> float:
    """Quadgram fitness of the decrypt implied by (keyed alphabet, shifts)."""
    n = len(pre)
    plain = [post[(pre[i] - shifts[i % period]) % 26] for i in range(n)]
    s = 0.0
    a, b, c = plain[0], plain[1], plain[2]
    for i in range(3, n):
        d = plain[i]
        s += table[((a * 26 + b) * 26 + c) * 26 + d]
        a, b, c = b, c, d
    return s


def blind_attack(
    letters: str,
    scorer: NgramScorer,
    model: str,
    *,
    forced_period: int | None = None,
    max_period: int | None = None,
    deadline: float | None = None,
    rng=None,
    restarts: int = 6,
    iters_per_temp: int = 120,
) -> tuple[float, str, int, list[int], str] | None:
    """Blind keyed-alphabet recovery — no keyword needed (best-effort, long texts).

    The stblake/AZdecrypt-class shape: simulated-annealing search over the 26-letter
    *keyed alphabet* only (the hard, isolated optimum), with each candidate alphabet's
    per-column *cycleword* recovered **deterministically** by chi-square and the whole
    decrypt scored by quadgrams. Decoupling the deterministic cycleword from the
    searched alphabet is what makes this tractable where a joint (alphabet+rotations)
    anneal stalls. The winning alphabet's shifts are then polished by quadgram
    coordinate-ascent. Returns ``(score, plaintext, period, shifts, alphabet)`` or
    ``None``. Works for Q1/Q2/Q3 via ``model`` (``_build_pre_post``).
    """
    import random as _random

    from .. import search

    n = len(letters)
    if n < BLIND_MIN:
        return None
    rng = rng or _random.Random(0)
    table, _ = _fast_table(scorer)
    ctn = [ord(c) - 65 for c in letters]
    periods = candidate_periods(letters, max_period, forced_period)

    def make_fitness(period: int, cols: list[list[int]]) -> Callable[[list[str]], float]:
        def fitness(state: list[str]) -> float:
            pre, post = _build_pre_post(model, ctn, "".join(state))
            shifts = _chi_shifts(pre, post, period, cols)
            return _decode_quad_score(pre, post, shifts, period, table)

        return fitness

    def init_alphabet() -> list[str]:
        return search.shuffled(list(_STD), rng)

    # Bind callables to concrete list[str] types so anneal's generic infers cleanly.
    neighbour: Callable[[list[str], object], list[str]] = search.swap_neighbour

    best: tuple[float, str, int, list[int], str] | None = None
    for period in periods:
        if deadline and time.monotonic() > deadline:
            break
        cols = [[i for i in range(j, n, period)] for j in range(period)]
        alpha, _ = search.anneal(
            init=init_alphabet,
            neighbour=neighbour,
            score=make_fitness(period, cols),
            rng=rng,
            restarts=restarts,
            iters_per_temp=iters_per_temp,
            deadline=deadline,
        )
        keyed = "".join(alpha)
        pre, post = _build_pre_post(model, ctn, keyed)
        sc, shifts, plain = _recover_shifts(
            pre, post, period, cols, table, restarts=20, rng=rng, deadline=deadline
        )
        sc, shifts, plain = _two_opt_polish(
            pre, post, period, cols, table, shifts, deadline=time.monotonic() + _TWO_OPT_BUDGET_S
        )
        if best is None or sc > best[0]:
            best = (sc, "".join(chr(65 + x) for x in plain), period, shifts, keyed)
    return best


def blind_candidates(cipher, model: str, text: str, scorer, *, deadline, rng=None, **opts):
    """Run the blind keyed-alphabet anneal and return verified ``Candidate``s.

    The recovered alphabet is an arbitrary 26-letter permutation (no keyword), so the
    published key uses the full alphabet as the "keyword" (``keyed_alphabet`` leaves a
    complete alphabet unchanged) plus the derived indicator — which still round-trips
    through the cipher's own ``decode``. Returns ``[]`` when nothing is found.
    """
    letters = only_letters(text)
    if len(letters) < BLIND_MIN:
        return []
    hit = blind_attack(
        letters,
        scorer,
        model,
        forced_period=opts.get("key_length") or opts.get("period"),
        max_period=opts.get("max_period"),
        deadline=deadline,
        rng=rng,
    )
    if hit is None:
        return []
    score, plain, period, shifts, alphabet = hit
    key: str | None = build_key(model, alphabet, shifts)
    try:
        if only_letters(cipher.decode(letters, key)) != plain:
            key = None
    except Exception:
        key = None
    return [
        Candidate(
            plaintext=reflow(text, plain),
            cipher=cipher.name,
            key=key,
            score=score,
            confidence=scorer.confidence(plain),
            meta={
                "period": period,
                "keyed_alphabet": alphabet,
                "vigenere_key_offsets": shifts,
                "method": "blind-anneal",
            },
        )
    ]


def dictionary_attack(
    letters: str,
    scorer: NgramScorer,
    model: str,
    *,
    keywords: Sequence[str] | None = None,
    forced_period: int | None = None,
    max_period: int | None = None,
    deadline: float | None = None,
    rng=None,
    polish_restarts: int = 80,
) -> tuple[float, str, int, list[int], str] | None:
    """Best (score, plaintext, period, shifts, keyword) over the keyword list, or None.

    ``model`` is ``"Q1"``/``"Q2"``/``"Q3"``. ``shifts`` are the per-column additive
    shifts in the keyed-alphabet ordering (the recovered Vigenere key, as offsets).

    Two phases: a cheap one-greedy *filter* finds the right (keyword, period) —
    even a single greedy pass ranks the true keyword first by a wide margin — then
    a random-restart *polish* converges the winner to clean English (essential for
    long keys, where one greedy pass is "mostly right" but not readable). The
    filter runs period-by-significance, KRYPTOS-first, so the likely answer is the
    very first evaluation and survives an early deadline; the deadline is split so
    the polish always gets a share.
    """
    import random as _random

    n = len(letters)
    if n < 40:
        return None
    rng = rng or _random.Random(0)
    table, _ = _fast_table(scorer)
    ctn = [ord(c) - 65 for c in letters]
    words = list(keywords) if keywords is not None else list(BUILTIN_KEYWORDS)
    periods = candidate_periods(letters, max_period, forced_period)

    phase1_deadline = None
    if deadline:
        now = time.monotonic()
        phase1_deadline = now + 0.55 * max(0.0, deadline - now)

    colcache: dict[int, list[list[int]]] = {}
    best: tuple[float, int, list[int], str] | None = None  # (score, period, shifts, kw)
    for period in periods:
        cols = [[i for i in range(j, n, period)] for j in range(period)]
        colcache[period] = cols
        for kw in words:
            if (phase1_deadline and time.monotonic() > phase1_deadline) or (
                deadline and time.monotonic() > deadline
            ):
                break
            pre, post = _build_pre_post(model, ctn, keyed_alphabet(kw))
            sc_, shifts, _plain = _recover_shifts(pre, post, period, cols, table, restarts=1)
            if best is None or sc_ > best[0]:
                best = (sc_, period, shifts, kw)
        if deadline and time.monotonic() > deadline:
            break
    if best is None:
        return None

    # Polish the winning (keyword, period). First single-column ascent with whatever
    # restart budget remains, then a deterministic 2-opt finisher that escapes the
    # coupled near-column local optima 1-opt (even with restarts) gets stuck on for
    # long keys. The 2-opt is cheap and bounded, so it gets a guaranteed budget even
    # if the caller's deadline has already elapsed — it is the step that makes a
    # long-key solve clean and reproducible rather than RNG-dependent.
    _, period, _, kw = best
    pre, post = _build_pre_post(model, ctn, keyed_alphabet(kw))
    cols = colcache.get(period) or [[i for i in range(j, n, period)] for j in range(period)]
    sc_, shifts, plain = _recover_shifts(
        pre, post, period, cols, table, restarts=polish_restarts, rng=rng, deadline=deadline
    )
    sc_, shifts, plain = _two_opt_polish(
        pre, post, period, cols, table, shifts, deadline=time.monotonic() + _TWO_OPT_BUDGET_S
    )
    return (sc_, "".join(chr(65 + x) for x in plain), period, shifts, kw)


def _dedup_by_plaintext(candidates: list[Candidate]) -> list[Candidate]:
    """Highest-scoring first, with duplicate plaintexts removed."""
    candidates.sort(key=lambda c: c.score, reverse=True)
    seen: set[str] = set()
    unique: list[Candidate] = []
    for cand in candidates:
        if cand.plaintext not in seen:
            seen.add(cand.plaintext)
            unique.append(cand)
    return unique


def dictionary_candidates(cipher, model: str, text: str, scorer, *, deadline, rng=None, **opts):
    """Run the keyword dictionary attack and return it as verified ``Candidate``s.

    Shared by quagmire1/2/3 ``crack``; ``cipher`` supplies ``decode`` for the
    round-trip key check and its ``name``. Returns ``[]`` when nothing is found.
    """
    letters = only_letters(text)
    hit = dictionary_attack(
        letters,
        scorer,
        model,
        keywords=opts.get("keywords"),
        forced_period=opts.get("key_length") or opts.get("period"),
        deadline=deadline,
        rng=rng,
    )
    if hit is None:
        return []
    score, plain, period, shifts, keyword = hit
    # Publish a key only if it round-trips through the cipher's own decode.
    key: str | None = build_key(model, keyword, shifts)
    try:
        if only_letters(cipher.decode(letters, key)) != plain:
            key = None
    except Exception:
        key = None
    meta = {
        "period": period,
        "keyword": keyword,
        "keyed_alphabet": keyed_alphabet(keyword),
        "vigenere_key_offsets": shifts,
        "method": "keyword-dictionary",
    }
    if len(letters) < RELIABLE_MIN:
        meta["warning"] = (
            f"short input ({len(letters)} letters); keyless Quagmire recovery is "
            f"unreliable below ~{RELIABLE_MIN} letters"
        )
    return [
        Candidate(
            plaintext=reflow(text, plain),
            cipher=cipher.name,
            key=key,
            score=score,
            confidence=scorer.confidence(plain),
            meta=meta,
        )
    ]


def build_key(model: str, keyword: str, shifts: Sequence[int]) -> str:
    """A ``key`` string that round-trips through the cipher's own ``decode``.

    The recovered shifts are encoded as the indicator keyword (plus the alignment
    letter for Q3) so the published key reproduces the decrypt exactly.
    """
    keyed = keyed_alphabet(keyword)
    if model == "Q1":  # shift = (ord(ind) - ord(align)_in_KP); align = A
        base = keyed.index("A")
        indicator = "".join(chr((sh + base) % 26 + 65) for sh in shifts)
        return f"{keyword}/{indicator}"
    if model == "Q2":  # shift = index(ind in keyed) - (ord(align)-65); align = A
        indicator = "".join(keyed[sh] for sh in shifts)
        return f"{keyword}/{indicator}"
    if model == "Q3":  # shift = posA(ind) - posA(align); align = first keyed letter
        indicator = "".join(keyed[sh] for sh in shifts)
        return f"{keyword}/{indicator}/{keyed[0]}"
    raise ValueError(f"unknown quagmire model {model!r}")
