"""Fractionation crackers: Trifid (3x3x3 cube) and 6x6 Bifid/Polybius.

A *doubly-flattened* IoC is the decisive fingerprint these crackers target: a
panel whose single-letter IoC sits in the random band AND whose digraph IoC is
also random-flat. English would give a digraph IoC z-score around +30, so BOTH
being flat is the rare signature of FRACTIONATION — a cipher that diffuses each
plaintext letter across several ciphertext positions. The two live,
natively-26-letter (no I/J merge) candidates are Trifid and 6x6 fractionation,
which is what this module attacks.

CIPHERS (encrypt/decrypt round-trip verified, including partial final blocks):

* **Trifid** — Delastelle's 3-D fractionation. A keyed 27-cell alphabet (26
  letters + a 27th symbol, default ``#``) fills a 3x3x3 cube; each letter has a
  trigram ``(k//9+1, (k//3)%3+1, k%3+1)`` in the ACA row-by-row convention.
  Within each period-P block the three coordinates are written vertically then
  read horizontally (all P layer digits, then all P row, then all P col), and
  consecutive triples become new letters. Natively emits all 26 letters plus the
  27th cell.

* **6x6 Bifid (Polybius fractionation)** — a 36-cell keyed square holding the
  26 letters AND the 10 digits, so I and J stay DISTINCT (unlike 5x5 Bifid, which
  merges J->I and cannot emit a distinct J). Each letter has a digram
  ``(k//6+1, k%6+1)``; per block the row digits then col digits are read off and
  re-paired. Plaintext that is pure A-Z still produces a full-alphabet,
  both-I-and-J ciphertext over the 26 letters.

BLIND CRACKERS — period detection (sweep 3..22, rank by post-decode hexagram
fitness) then GREEDY best-improvement key hill-climb from MANY seeded restarts
(keyword-seeded + random). On these cubes *greedy beats SA* (SA wanders); so this
uses greedy with many restarts and per-restart period re-pick, never simulated
annealing.

CONTROL-GATING (the whole point): a null on a target ciphertext is only meaningful
if the blind cracker FIRST recovers a SYNTHETIC of the SAME cipher at the SAME
length (English plaintext, random/keyword key) to >=90% char-match under the SAME
blind budget. Always run the matched synthetic control alongside any real attempt.
If the control does NOT clear 90%, report the recovery CEILING honestly: the cipher
is then unfalsifiable-by-cracking at this length — do NOT claim the target refuted.
"""

from __future__ import annotations

import random
import time
from functools import partial

from .scoring import index_of_coincidence, resolve_scorer

KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
STANDARD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
SYMBOL = "#"

# Optional keyword seeds — supply suspected key words for the target here to add
# in-basin restarts. Empty by default; the cracker still seeds KRYPTOS + random.
THEMATIC_WORDS: tuple[str, ...] = ()

# Hexagram scorer sharpens the fractionation objective; quadgrams is the fallback
# that resolve_scorer guarantees. Both expose .score / .average / .fitness.
_SCORER = resolve_scorer("hexagrams", "english")


# --------------------------------------------------------------------------- #
# Alphabet / key construction
# --------------------------------------------------------------------------- #
def _dedup(seq: str) -> str:
    out: list[str] = []
    for ch in seq:
        if ch not in out:
            out.append(ch)
    return "".join(out)


def _keyed(keyword: str, pool: str) -> str:
    """Keyword-then-pool keyed alphabet over the given character pool."""
    kw = "".join(c for c in keyword.upper() if c in pool)
    return _dedup(kw + pool)


def trifid_alphabet(keyword: str = "KRYPTOS", symbol: str = SYMBOL) -> str:
    """27-cell Trifid alphabet (26 letters keyed + a 27th symbol appended)."""
    return _keyed(keyword, STANDARD) + symbol


def bifid6_alphabet(keyword: str = "KRYPTOS") -> str:
    """36-cell 6x6 alphabet: 26 letters + 10 digits, keyed (I and J distinct)."""
    return _keyed(keyword, STANDARD + DIGITS)


# --------------------------------------------------------------------------- #
# Trifid encrypt / decrypt (validated round-trip, incl. partial final block)
# --------------------------------------------------------------------------- #
def _trifid_tri(alpha: str):
    fwd = {ch: (k // 9 + 1, (k // 3) % 3 + 1, k % 3 + 1) for k, ch in enumerate(alpha)}
    inv = {(k // 9 + 1, (k // 3) % 3 + 1, k % 3 + 1): ch for k, ch in enumerate(alpha)}
    return fwd, inv


def trifid_encode(pt: str, alpha: str, period: int) -> str:
    fwd, inv = _trifid_tri(alpha)
    out: list[str] = []
    for i in range(0, len(pt), period):
        block = pt[i : i + period]
        p = len(block)
        layers = [fwd[c][0] for c in block]
        rows = [fwd[c][1] for c in block]
        cols = [fwd[c][2] for c in block]
        digits = "".join(str(x) for x in (layers + rows + cols))
        for j in range(0, 3 * p, 3):
            out.append(inv[(int(digits[j]), int(digits[j + 1]), int(digits[j + 2]))])
    return "".join(out)


def trifid_decode(ct: str, alpha: str, period: int) -> str:
    fwd, inv = _trifid_tri(alpha)
    out: list[str] = []
    for i in range(0, len(ct), period):
        block = ct[i : i + period]
        p = len(block)
        digits = "".join(f"{fwd[c][0]}{fwd[c][1]}{fwd[c][2]}" for c in block)
        layers, rows, cols = digits[0:p], digits[p : 2 * p], digits[2 * p : 3 * p]
        for j in range(p):
            out.append(inv[(int(layers[j]), int(rows[j]), int(cols[j]))])
    return "".join(out)


# --------------------------------------------------------------------------- #
# 6x6 Bifid encrypt / decrypt (validated round-trip, incl. partial final block)
# --------------------------------------------------------------------------- #
def _bifid6_di(alpha: str):
    fwd = {ch: (k // 6 + 1, k % 6 + 1) for k, ch in enumerate(alpha)}
    inv = {(k // 6 + 1, k % 6 + 1): ch for k, ch in enumerate(alpha)}
    return fwd, inv


#: Bifid seriation read-order variants: ``(half_order, reverse_rows, reverse_cols)``.
#: A Bifid gathers each block's row- and column-coordinates into one sequence, then
#: re-pairs it. Implementations differ in *how* they gather — rows-then-columns
#: (``"RC"``) vs columns-then-rows (``"CR"``), each optionally reversing the row and/or
#: column run. Several of these are statistically indistinguishable from one another,
#: so a blind solver that assumes only the standard order silently misses ciphertext
#: built with a different one. ``"std"`` is the classic row-major bifid (backward
#: compatible default).
GATHER_VARIANTS: dict[str, tuple[str, bool, bool]] = {
    "std": ("RC", False, False),
    "rc_ft": ("RC", False, True),
    "rc_tf": ("RC", True, False),
    "rc_tt": ("RC", True, True),
    "cr_ff": ("CR", False, False),
    "cr_ft": ("CR", False, True),
    "cr_tf": ("CR", True, False),
    "cr_tt": ("CR", True, True),
}


def _resolve_gather(gather: str | tuple[str, bool, bool]) -> tuple[str, bool, bool]:
    return GATHER_VARIANTS[gather] if isinstance(gather, str) else tuple(gather)  # type: ignore[return-value]


def _gather(rows: list[int], cols: list[int], spec: tuple[str, bool, bool]) -> list[int]:
    half, rev_r, rev_c = spec
    r = rows[::-1] if rev_r else list(rows)
    c = cols[::-1] if rev_c else list(cols)
    return (r + c) if half == "RC" else (c + r)


def _ungather(seq: list[int], p: int, spec: tuple[str, bool, bool]) -> tuple[list[int], list[int]]:
    half, rev_r, rev_c = spec
    if half == "RC":
        r, c = seq[:p], seq[p:]
    else:
        c, r = seq[:p], seq[p:]
    if rev_r:
        r = r[::-1]
    if rev_c:
        c = c[::-1]
    return r, c


def bifid6_encode(
    pt: str, alpha: str, period: int, gather: str | tuple[str, bool, bool] = "std"
) -> str:
    spec = _resolve_gather(gather)
    fwd, inv = _bifid6_di(alpha)
    out: list[str] = []
    for i in range(0, len(pt), period):
        block = pt[i : i + period]
        p = len(block)
        rows = [fwd[c][0] for c in block]
        cols = [fwd[c][1] for c in block]
        seq = _gather(rows, cols, spec)
        for j in range(0, 2 * p, 2):
            out.append(inv[(seq[j], seq[j + 1])])
    return "".join(out)


def bifid6_decode(
    ct: str, alpha: str, period: int, gather: str | tuple[str, bool, bool] = "std"
) -> str:
    spec = _resolve_gather(gather)
    fwd, inv = _bifid6_di(alpha)
    out: list[str] = []
    for i in range(0, len(ct), period):
        block = ct[i : i + period]
        p = len(block)
        seq: list[int] = []
        for c in block:
            r, cc = fwd[c]
            seq.append(r)
            seq.append(cc)
        rows, cols = _ungather(seq, p, spec)
        for j in range(p):
            out.append(inv[(rows[j], cols[j])])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Generic blind cracker (cube/square agnostic via decode + pool callbacks)
# --------------------------------------------------------------------------- #
_FLOOR = _SCORER.floor


def _score(pt: str) -> float:
    """Hexagram log-prob of a candidate decode, penalizing non-A-Z output.

    The 6x6 square holds digits, so a degenerate key can decode the ciphertext to
    a digit-heavy string; the n-gram scorer strips non-letters and would then score
    that short letter-residue deceptively high. Penalizing each non-letter by one
    ``floor`` keeps a clean A-Z English decode optimal and crushes such artifacts.
    (Pure-A-Z decodes — every Trifid case and the bifid6 true key — are unaffected.)
    """
    non_letters = sum(1 for c in pt if not ("A" <= c <= "Z"))
    return _SCORER.score(pt) + non_letters * _FLOOR


def _seed_keys(pool: str, symbol: str, rng: random.Random, n_random: int) -> list[str]:
    """KRYPTOS + thematic + random keyed alphabets over the given letter pool.

    For Trifid (pool = A-Z) ``symbol`` is appended; for 6x6 (pool = A-Z0-9) the
    pool already fills all 36 cells and ``symbol`` is empty.
    """
    seeds: list[str] = []
    for kw in ("KRYPTOS", "", *THEMATIC_WORDS):
        seeds.append(_keyed(kw, pool) + symbol)
    seeds = _dedup_keys(seeds)
    for _ in range(n_random):
        a = list(pool)
        rng.shuffle(a)
        seeds.append("".join(a) + symbol)
    return seeds


def _dedup_keys(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _local_climb(
    cur: list[str], tail: str, letters: str, period: int, decode, deadline: float
) -> float:
    """Best-improvement greedy swap climb (mutates ``cur`` in place to a local opt).

    Scans all transpositions of the ``cur`` cells, applies the single best-gaining
    swap, and repeats until no swap improves. Greedy beats SA on these cubes (the
    campaign finding), so the local move is strictly greedy. Returns the score of
    the local optimum reached.
    """
    n = len(cur)
    cur_score = _score(decode(letters, "".join(cur) + tail, period))
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        best_swap = None
        best_gain = 1e-9
        for i in range(n):
            for j in range(i + 1, n):
                cur[i], cur[j] = cur[j], cur[i]
                s = _score(decode(letters, "".join(cur) + tail, period))
                cur[i], cur[j] = cur[j], cur[i]
                if s - cur_score > best_gain:
                    best_gain = s - cur_score
                    best_swap = (i, j, s)
            if time.monotonic() >= deadline:
                break
        if best_swap is not None:
            i, j, s = best_swap
            cur[i], cur[j] = cur[j], cur[i]
            cur_score = s
            improved = True
    return cur_score


def _greedy_climb(
    letters: str,
    alpha: str,
    period: int,
    decode,
    deadline: float,
    rng: random.Random,
    kick_swaps: int = 3,
) -> tuple[float, str]:
    """Iterated local search: greedy local climbs chained by random kicks.

    Climb to a local optimum (best-improvement greedy), then repeatedly KICK the
    incumbent by ``kick_swaps`` random transpositions and re-climb; accept the kick
    if it does not worsen the best, otherwise restart the kick from the best basin.
    This escapes the shallow local optima a single greedy climb gets trapped in
    while keeping greedy as the local move. Only the first ``len(alpha) - n_sym``
    cells are permuted; a trailing symbol cell (Trifid's 27th) stays pinned.
    Returns the best (score, alphabet) reached before the deadline.
    """
    n_sym = 1 if alpha and alpha[-1] == SYMBOL else 0
    n = len(alpha) - n_sym
    tail = alpha[n:]
    cur = list(alpha[:n])
    cur_score = _local_climb(cur, tail, letters, period, decode, deadline)
    best_cur = cur[:]
    best_alpha, best_score = "".join(cur) + tail, cur_score
    while time.monotonic() < deadline:
        cur = best_cur[:]
        for _ in range(kick_swaps):
            i, j = rng.randrange(n), rng.randrange(n)
            cur[i], cur[j] = cur[j], cur[i]
        cur_score = _local_climb(cur, tail, letters, period, decode, deadline)
        if cur_score > best_score:
            best_score = cur_score
            best_cur = cur[:]
            best_alpha = "".join(cur) + tail
    return best_score, best_alpha


def _detect_period(
    letters: str,
    pool: str,
    symbol: str,
    decode,
    periods,
    rng: random.Random,
    probe_restarts: int,
    probe_seconds: float,
):
    """Rank candidate periods by the best seeded score.

    Robust under load: for every period each structured seed's RAW decode score is
    evaluated unconditionally (instant, no climbing), so a key-matched seed at the
    true period always surfaces even when the per-period climb budget is exhausted.
    A short greedy climb is layered on top, time permitting. The raw seed-decode
    signal is what pins the period when the plaintext key happens to be a seed.
    """
    ranked: list[tuple[float, int, str]] = []
    seeds = _seed_keys(pool, symbol, rng, probe_restarts)
    for P in periods:
        bs, ba = -1e18, seeds[0]  # seeds is non-empty; default keeps ba a real key, never None
        for s0 in seeds:
            sc = _score(decode(letters, s0, P))  # raw, always evaluated
            if sc > bs:
                bs, ba = sc, s0
        deadline = time.monotonic() + probe_seconds
        for s0 in seeds:
            if time.monotonic() >= deadline:
                break
            sc, a = _greedy_climb(letters, s0, P, decode, deadline, rng)
            if sc > bs:
                bs, ba = sc, a
        ranked.append((bs, P, ba))
    ranked.sort(key=lambda t: -t[0])
    return ranked


def _blind_solve(
    ct: str,
    pool: str,
    symbol: str,
    decode,
    *,
    periods,
    restarts,
    seconds_per_start,
    probe_seconds,
    seed,
):
    """Shared blind pipeline: detect period, then deep greedy climb from seeds."""
    # Keep the cipher's FULL character set: Trifid ciphertext can legitimately
    # contain the 27th symbol cell, so it must NOT be filtered out (that would
    # shorten the text and break block alignment / the round-trip).
    keep = pool + symbol
    letters = "".join(c for c in ct.upper() if c in keep)
    rng = random.Random(seed)
    periods = list(periods)

    ranked = _detect_period(
        letters,
        pool,
        symbol,
        decode,
        periods,
        rng,
        probe_restarts=min(restarts, 6),
        probe_seconds=probe_seconds,
    )
    # Keep the strongest few periods for the deep pass.
    top_periods = [P for _, P, _ in ranked[: min(3, len(ranked))]]

    # Seed the incumbent with the detection winner so the deep pass can never
    # regress below what period detection already found (key-matched in-basin case).
    best_score, best_P, best_key = ranked[0]
    best = {
        "score": best_score,
        "plaintext": decode(letters, best_key, best_P),
        "key": best_key,
        "period": best_P,
    }

    seeds = _seed_keys(pool, symbol, rng, restarts)
    for P in top_periods:
        for s0 in seeds:
            deadline = time.monotonic() + seconds_per_start
            sc, a = _greedy_climb(letters, s0, P, decode, deadline, rng)
            if sc > best["score"]:
                pt = decode(letters, a, P)
                best = {"score": sc, "plaintext": pt, "key": a, "period": P}
    best["period_ranking"] = [(round(s, 1), P) for s, P, _ in ranked]
    best["ioc"] = index_of_coincidence(best["plaintext"]) if best["plaintext"] else 0.0
    return best


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def solve_trifid(
    ct: str,
    periods=range(3, 23),
    restarts: int = 40,
    seconds_per_start: float = 4.0,
    probe_seconds: float = 2.0,
    seed: int = 0,
) -> dict:
    """Blind Trifid crack: detect period, greedy cube climb from many seeds.

    Returns ``dict(score, plaintext, key, period, ioc, period_ranking)`` where
    ``key`` is the recovered 27-cell alphabet (last cell is the symbol). ``score``
    is the hexagram log-prob total (higher is better; clean English >> gibberish).
    """
    return _blind_solve(
        ct,
        STANDARD,
        SYMBOL,
        trifid_decode,
        periods=periods,
        restarts=restarts,
        seconds_per_start=seconds_per_start,
        probe_seconds=probe_seconds,
        seed=seed,
    )


def solve_bifid6(
    ct: str,
    periods=range(3, 23),
    restarts: int = 40,
    seconds_per_start: float = 4.0,
    probe_seconds: float = 2.0,
    seed: int = 0,
    gather: str | tuple[str, bool, bool] = "std",
) -> dict:
    """Blind 6x6 Bifid/Polybius crack: detect period, greedy square climb.

    Returns ``dict(score, plaintext, key, period, ioc, period_ranking)`` where
    ``key`` is the recovered 36-cell alphabet (26 letters + 10 digits).

    ``gather`` selects the seriation read-order (see :data:`GATHER_VARIANTS`); the
    variants are statistically near-indistinguishable, so sweep them all — e.g.
    ``best = max((solve_bifid6(ct, gather=g) for g in GATHER_VARIANTS), key=lambda r: r["score"])``
    — when the standard order does not read out.
    """
    decode = partial(bifid6_decode, gather=gather)
    return _blind_solve(
        ct,
        STANDARD + DIGITS,
        "",
        decode,
        periods=periods,
        restarts=restarts,
        seconds_per_start=seconds_per_start,
        probe_seconds=probe_seconds,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Self-test: round-trip + matched-synthetic control recovery (the real ceiling)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    # Generic English plaintext source for the synthetic controls / demo.
    ENGLISH = (
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHEOLDCLOCKINTHEHALL"
        "STRUCKMIDNIGHTANDTHEWINDCARRIEDTHESCENTOFRAINACROSSTHEQUIET"
        "FIELDSWHERETHEHARVESTHADLONGSINCEBEENGATHEREDINTOTHEBARNS"
        "BESIDETHEWEATHEREDFENCETHATMARKEDTHEEDGEOFTHEFARMSTEADLANDS"
        "ANDTHESILENTWATCHERSTOODGUARDOVERTHESLEEPINGVILLAGEUNTILDAWN"
    )
    N = 272

    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0  # seconds per start
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(
        f"calibration: random_ref={_SCORER._random_ref:.3f} "
        f"english_ref={_SCORER._english_ref:.3f} (avg hexagram log-prob)",
        flush=True,
    )

    def match(a: str, b: str) -> float:
        return sum(x == y for x, y in zip(a, b, strict=False)) / max(1, len(b))

    # ---- 1. Round-trip verification (both ciphers, incl. partial final blocks)
    print("\n=== round-trip verification (N=272, partial final block) ===", flush=True)
    rng = random.Random(7)
    rt_ok = True
    pt272 = (ENGLISH * 2)[:N]
    for P in (5, 7, 11, 13, 17):
        ta = trifid_alphabet("KRYPTOS")
        if trifid_decode(trifid_encode(pt272, ta, P), ta, P) != pt272:
            rt_ok = False
            print(f"  Trifid P={P}: ROUND-TRIP FAIL", flush=True)
        ba = bifid6_alphabet("KRYPTOS")
        if bifid6_decode(bifid6_encode(pt272, ba, P), ba, P) != pt272:
            rt_ok = False
            print(f"  Bifid6 P={P}: ROUND-TRIP FAIL", flush=True)
    print(
        f"  round-trip {'OK' if rt_ok else 'FAILED'} for Trifid & Bifid6 at P in {{5,7,11,13,17}}",
        flush=True,
    )

    # The blind crack permutes a 26-/36-cell key; the basin of attraction around the
    # true key is NARROW (a 4-swap perturbation already collapses the hexagram score
    # to the random floor — verified in development). So two control tiers are run:
    #   IN-BASIN: planted key == a search seed (KRYPTOS / thematic). Proves the climb
    #             + period detection work end-to-end (best case; expected ~100%).
    #   HONEST:   planted key is a FRESH random permutation NOT in the seed set. This
    #             is the true blind ceiling and the ONLY tier the 90% gate keys on.
    def plant_decode_match(encode, decode, alpha, period, solver, seed):
        ct = encode(pt272, alpha, period)
        t0 = time.monotonic()
        res = solver(
            ct,
            periods=range(3, 23),
            restarts=restarts,
            seconds_per_start=budget,
            probe_seconds=budget,
            seed=seed,
        )
        m = match(res["plaintext"], pt272)
        return res, m, time.monotonic() - t0

    # ---- 2. Trifid controls (period 11) ------------------------------------
    print("\n=== CONTROL: blind TRIFID recovery (P=11, N=272) ===", flush=True)
    res_ti, rec_ti, dt = plant_decode_match(
        trifid_encode, trifid_decode, trifid_alphabet("KRYPTOS"), 11, solve_trifid, 1
    )
    print(
        f"  IN-BASIN (planted KRYPTOS seed): recovered P={res_ti['period']} "
        f"match={rec_ti:.0%} score={res_ti['score']:.0f} {dt:.0f}s "
        f"rank5={res_ti['period_ranking'][:5]}",
        flush=True,
    )
    rng_k = random.Random(99)
    rand_letters = list(STANDARD)
    rng_k.shuffle(rand_letters)
    rand_tri = "".join(rand_letters) + SYMBOL
    res_th, rec_th, dt = plant_decode_match(
        trifid_encode, trifid_decode, rand_tri, 11, solve_trifid, 11
    )
    print(
        f"  HONEST (fresh random key): recovered P={res_th['period']} "
        f"match={rec_th:.0%} score={res_th['score']:.0f} {dt:.0f}s",
        flush=True,
    )

    # ---- 3. 6x6 Bifid controls (period 9) ----------------------------------
    print("\n=== CONTROL: blind 6x6 BIFID recovery (P=9, N=272) ===", flush=True)
    res_bi, rec_bi, dt = plant_decode_match(
        bifid6_encode, bifid6_decode, bifid6_alphabet("KRYPTOS"), 9, solve_bifid6, 2
    )
    print(
        f"  IN-BASIN (planted KRYPTOS seed): recovered P={res_bi['period']} "
        f"match={rec_bi:.0%} score={res_bi['score']:.0f} {dt:.0f}s "
        f"rank5={res_bi['period_ranking'][:5]}",
        flush=True,
    )
    rng_k2 = random.Random(123)
    rand_pool = list(STANDARD + DIGITS)
    rng_k2.shuffle(rand_pool)
    res_bh, rec_bh, dt = plant_decode_match(
        bifid6_encode, bifid6_decode, "".join(rand_pool), 9, solve_bifid6, 21
    )
    print(
        f"  HONEST (fresh random key): recovered P={res_bh['period']} "
        f"match={rec_bh:.0%} score={res_bh['score']:.0f} {dt:.0f}s",
        flush=True,
    )

    # ---- 4. Target panel (interpret ONLY relative to the HONEST ceiling) ----
    # Stand-in for an unknown target: a held-out synthetic encrypted under a
    # FRESH random key not in the seed set. Swap in real ciphertext here.
    print("\n=== TARGET (interpret ONLY relative to the HONEST control ceiling) ===", flush=True)
    rng_t = random.Random(2024)
    rand_target = list(STANDARD)
    rng_t.shuffle(rand_target)
    TARGET = trifid_encode(pt272, "".join(rand_target) + SYMBOL, 11)
    res_p = solve_trifid(
        TARGET,
        periods=range(3, 23),
        restarts=restarts,
        seconds_per_start=budget,
        probe_seconds=budget,
        seed=3,
    )
    print(
        f"  Trifid target: best period={res_p['period']} score={res_p['score']:.0f} "
        f"ioc={res_p['ioc']:.4f}",
        flush=True,
    )
    print(f"    plain[:80]: {res_p['plaintext'][:80]}", flush=True)

    # ---- Verdict: the HONEST controls are the gate -------------------------
    CLEAR = 0.90
    honest_cleared = rec_th >= CLEAR and rec_bh >= CLEAR
    print("\n=== VERDICT ===", flush=True)
    print(
        f"  IN-BASIN recovery: Trifid={rec_ti:.0%}  Bifid6={rec_bi:.0%} "
        f"(sanity: climb+period-detection work end-to-end)",
        flush=True,
    )
    print(
        f"  HONEST recovery:   Trifid={rec_th:.0%}  Bifid6={rec_bh:.0%}  "
        f"(threshold {CLEAR:.0%}) <-- the gate",
        flush=True,
    )
    if honest_cleared:
        print(
            "  HONEST controls CLEAR 90% -> blind crack is trustworthy at this "
            "length; a flat target result here would be a meaningful negative.",
            flush=True,
        )
    else:
        print(
            "  HONEST controls do NOT clear 90% -> this is the recovery CEILING. "
            "The cube/square key basin is too narrow to recover blind at this "
            "length (more text does not help: diffusion is set by period, not "
            "length). Trifid/6x6 are UNFALSIFIABLE-BY-CRACKING here without a crib; "
            "do NOT claim the target refuted from a flat result.",
            flush=True,
        )
    sys.exit(0 if rt_ok else 1)
