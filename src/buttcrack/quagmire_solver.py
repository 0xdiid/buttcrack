"""Blind solver for periodic polyalphabetic ciphers with KEYED alphabets.

Covers Vigenere / Beaufort / Variant-Beaufort / Porta and the four Quagmire
ciphers (I-IV). A Quagmire is a *periodic shift cipher* between a (possibly
keyed) plaintext alphabet and a (possibly keyed) ciphertext alphabet:

    Quagmire I   : keyed plaintext alphabet, straight ciphertext alphabet
    Quagmire II  : straight plaintext alphabet, keyed ciphertext alphabet
    Quagmire III : same keyed alphabet on both sides (== Vigenere in that alphabet)
    Quagmire IV  : two independent keyed alphabets

All of these are a special case of: for period position ``j`` (0..p-1), a
plaintext letter ``P`` is enciphered as

    C = CA[ ( PA.index(P) + shift[j] ) mod 26 ]

where ``PA`` is the plaintext alphabet (a permutation of A-Z), ``CA`` is the
ciphertext alphabet, and ``shift[j]`` is the per-period offset.  Decryption is

    P = PA[ ( CA.index(C) - shift[j] ) mod 26 ]

The unknowns are therefore (PA, CA, shift[0..p-1]).  We search over them with a
slippery hill-climber / simulated-annealing engine, seeding the keyed alphabet
from KRYPTOS and maximizing ``scoring.resolve_scorer('hexagrams').fitness``.
Reference: stblake/polyalphabetic ("slippery hill climber with backtracking").

Public API::

    solve(ct, periods=range(2, 21), kind='quagmire3', restarts=30) -> dict(
        score, plaintext, key, alphabet, period)
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

try:  # normal package import
    from .scoring import index_of_coincidence, resolve_scorer
    from .text import only_letters
except ImportError:  # allow `python3 src/buttcrack/quagmire_solver.py`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from buttcrack.scoring import index_of_coincidence, resolve_scorer
    from buttcrack.text import only_letters

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
KRYPTOS_ALPHABET = "KRYPTOSABCDEFGHIJLMNQUVWXZ"

# Which alphabets are free to be permuted, per Quagmire/cipher kind. The first
# flag controls the plaintext alphabet, the second the ciphertext alphabet, and
# ``linked`` ties the two together (Quag III: PA == CA).
_KIND_SPEC = {
    # name        : (pt_keyed, ct_keyed, linked)
    "vigenere":     (False, False, False),
    "quagmire1":    (True,  False, False),
    "quagmire2":    (False, True,  False),
    "quagmire3":    (True,  True,  True),
    "quagmire4":    (True,  True,  False),
}


def _keyed_alphabet(keyword: str) -> str:
    """Standard keyed alphabet: keyword letters (deduped) then the rest A-Z."""
    seen: list[str] = []
    for ch in (keyword + ALPHABET).upper():
        if "A" <= ch <= "Z" and ch not in seen:
            seen.append(ch)
    return "".join(seen)


def _decrypt(ct: str, pa: str, ca: str, shifts: Sequence[int], beaufort: bool = False) -> str:
    """Decrypt ``ct`` given plaintext/ciphertext alphabets and per-period shifts.

    ``beaufort=True`` uses the reciprocal Beaufort rule ``p = key - c`` instead of the
    Vigenere rule ``p = c - key`` (indices in the keyed alphabet)."""
    p = len(shifts)
    ca_index = {c: i for i, c in enumerate(ca)}
    out = []
    for k, c in enumerate(ct):
        j = ca_index[c]
        s = shifts[k % p]
        out.append(pa[(s - j) % 26] if beaufort else pa[(j - s) % 26])
    return "".join(out)


def _encrypt(pt: str, pa: str, ca: str, shifts: Sequence[int], beaufort: bool = False) -> str:
    """Inverse of :func:`_decrypt` (used by the self-test to plant synthetics)."""
    p = len(shifts)
    pa_index = {c: i for i, c in enumerate(pa)}
    out = []
    for k, c in enumerate(pt):
        i = pa_index[c]
        s = shifts[k % p]
        out.append(ca[(s - i) % 26] if beaufort else ca[(i + s) % 26])
    return "".join(out)


def _best_shifts(ct: str, pa: str, ca: str, p: int, scorer, beaufort: bool = False) -> tuple[list[int], float]:
    """For fixed alphabets, pick the best per-period shift independently.

    Each period column is a simple substituted Caesar; we score each candidate
    shift's column with the n-gram-free index of coincidence against English and
    pick the per-column best, then refine the whole set with the real scorer.
    This makes the alphabet-perturbation step cheap and well-conditioned.
    """
    # Column-wise IoC pre-pick gives a strong starting set of shifts.
    cols: list[list[str]] = [[] for _ in range(p)]
    for k, c in enumerate(ct):
        cols[k % p].append(c)
    ca_index = {c: i for i, c in enumerate(ca)}
    shifts = []
    for j in range(p):
        col = cols[j]
        best_s, best_chi = 0, math.inf
        for s in range(26):
            # decode column with this shift, score by chi-squared-ish fit
            freq = [0] * 26
            for c in col:
                idx = (s - ca_index[c]) % 26 if beaufort else (ca_index[c] - s) % 26
                freq[idx] += 1
            # map decoded indices through pa to letters then chi vs English
            chi = _col_chi(freq, pa)
            if chi < best_chi:
                best_chi, best_s = chi, s
        shifts.append(best_s)
    score = scorer.fitness(_decrypt(ct, pa, ca, shifts, beaufort=beaufort))
    return shifts, score


# English letter frequencies for the cheap per-column chi-squared.
_ENG = {
    "A": 0.08167, "B": 0.01492, "C": 0.02782, "D": 0.04253, "E": 0.12702,
    "F": 0.02228, "G": 0.02015, "H": 0.06094, "I": 0.06966, "J": 0.00153,
    "K": 0.00772, "L": 0.04025, "M": 0.02406, "N": 0.06749, "O": 0.07507,
    "P": 0.01929, "Q": 0.00095, "R": 0.05987, "S": 0.06327, "T": 0.09056,
    "U": 0.02758, "V": 0.00978, "W": 0.02360, "X": 0.00150, "Y": 0.01974,
    "Z": 0.00074,
}


def _col_chi(decoded_freq: Sequence[int], pa: str) -> float:
    """Chi-squared of a decoded column's letter frequencies vs English.

    ``decoded_freq[i]`` is the count of plaintext-alphabet index ``i`` in the
    column; we translate through ``pa`` to actual letters before comparing.
    """
    n = sum(decoded_freq)
    if n == 0:
        return math.inf
    letter_counts = {}
    for i, cnt in enumerate(decoded_freq):
        if cnt:
            letter_counts[pa[i]] = letter_counts.get(pa[i], 0) + cnt
    total = 0.0
    for letter, exp_p in _ENG.items():
        expected = exp_p * n
        obs = letter_counts.get(letter, 0)
        total += (obs - expected) ** 2 / expected
    return total


def _anneal_one(
    ct: str,
    p: int,
    spec: tuple[bool, bool, bool],
    scorer,
    rng: random.Random,
    iters: int,
) -> tuple[float, str, str, list[int], str]:
    """One annealing restart for a fixed period. Returns (score, pa, ca, shifts, pt)."""
    pt_keyed, ct_keyed, linked = spec

    # --- seed alphabets (bias toward the KRYPTOS keyed alphabet) ---
    if pt_keyed:
        pa = list(KRYPTOS_ALPHABET) if rng.random() < 0.6 else _shuffled(rng)
    else:
        pa = list(ALPHABET)
    if linked:
        ca = pa
    elif ct_keyed:
        ca = list(KRYPTOS_ALPHABET) if rng.random() < 0.5 else _shuffled(rng)
    else:
        ca = list(ALPHABET)

    shifts, score = _best_shifts(ct, "".join(pa), "".join(ca), p, scorer)
    best = (score, list(pa), list(ca), list(shifts))

    # temperature schedule for simulated annealing
    T0, T1 = 8.0, 0.05
    cur_score = score
    cur_pa, cur_ca, cur_shifts = list(pa), list(ca), list(shifts)

    free_alpha = pt_keyed or ct_keyed
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))

        npa, nca, nshifts = list(cur_pa), list(cur_ca), list(cur_shifts)

        # choose a perturbation: swap two alphabet letters OR change one shift
        do_alpha = free_alpha and rng.random() < 0.6
        if do_alpha:
            # swap two letters in a randomly chosen keyed alphabet
            if linked:
                i, j = rng.randrange(26), rng.randrange(26)
                npa[i], npa[j] = npa[j], npa[i]
                nca = npa  # stay linked
            else:
                # pick which alphabet to perturb among the keyed ones
                targets = []
                if pt_keyed:
                    targets.append("pa")
                if ct_keyed:
                    targets.append("ca")
                which = rng.choice(targets)
                arr = npa if which == "pa" else nca
                i, j = rng.randrange(26), rng.randrange(26)
                arr[i], arr[j] = arr[j], arr[i]
            # re-pick shifts greedily for the new alphabets, then perturb them
            nshifts, _ = _best_shifts(ct, "".join(npa), "".join(nca), p, scorer)
        else:
            # nudge one or two shifts
            for _ in range(1 + (rng.random() < 0.3)):
                nshifts[rng.randrange(p)] = rng.randrange(26)

        pt = _decrypt(ct, "".join(npa), "".join(nca), nshifts)
        nscore = scorer.fitness(pt)

        d = nscore - cur_score
        if d >= 0 or rng.random() < math.exp(d / T):
            cur_pa, cur_ca, cur_shifts, cur_score = npa, nca, nshifts, nscore
            if nscore > best[0]:
                best = (nscore, list(npa), list(nca), list(nshifts))

    bscore, bpa, bca, bshifts = best
    bpt = _decrypt(ct, "".join(bpa), "".join(bca), bshifts)
    return bscore, "".join(bpa), "".join(bca), bshifts, bpt


def _shuffled(rng: random.Random) -> list[str]:
    a = list(ALPHABET)
    rng.shuffle(a)
    return a


def _polish(
    ct: str,
    pa: str,
    ca: str,
    shifts: list[int],
    spec: tuple[bool, bool, bool],
    scorer,
    rng: random.Random,
    iters: int = 4000,
) -> tuple[float, str, str, list[int], str]:
    """Deterministic-ish greedy polish of a good candidate (no uphill moves)."""
    pt_keyed, ct_keyed, linked = spec
    free_alpha = pt_keyed or ct_keyed
    cpa, cca, csh = list(pa), list(ca), list(shifts)
    p = len(shifts)
    best_score = scorer.fitness(_decrypt(ct, "".join(cpa), "".join(cca), csh))
    for _ in range(iters):
        npa, nca, nsh = list(cpa), list(cca), list(csh)
        if free_alpha and rng.random() < 0.6:
            if linked:
                i, j = rng.randrange(26), rng.randrange(26)
                npa[i], npa[j] = npa[j], npa[i]
                nca = npa
            else:
                arr = npa if (pt_keyed and (not ct_keyed or rng.random() < 0.5)) else nca
                i, j = rng.randrange(26), rng.randrange(26)
                arr[i], arr[j] = arr[j], arr[i]
            nsh, _ = _best_shifts(ct, "".join(npa), "".join(nca), p, scorer)
        else:
            nsh[rng.randrange(p)] = rng.randrange(26)
        s = scorer.fitness(_decrypt(ct, "".join(npa), "".join(nca), nsh))
        if s > best_score:
            best_score, cpa, cca, csh = s, npa, nca, nsh
    bpt = _decrypt(ct, "".join(cpa), "".join(cca), csh)
    return best_score, "".join(cpa), "".join(cca), csh, bpt


def _shifts_to_key(shifts: Sequence[int], ca: str, pa: str) -> str:
    """Render per-period shifts as a key indicator string (A=0..Z=25).

    For Quag III this is the Vigenere key in the keyed alphabet; for the others
    it is the offset indicator, which is the most useful human-readable summary.
    """
    return "".join(ALPHABET[s % 26] for s in shifts)


def solve(
    ct: str,
    periods: Iterable[int] = range(2, 21),
    kind: str = "quagmire3",
    restarts: int = 30,
    *,
    seed: int | None = None,
    iters: int | None = None,
    scorer=None,
    verbose: bool = False,
) -> dict:
    """Blindly solve a periodic keyed-alphabet polyalphabetic cipher.

    Parameters
    ----------
    ct        : ciphertext (non-letters ignored).
    periods   : a single ``int`` or an iterable of candidate periods to sweep.
    kind      : one of ``vigenere``, ``quagmire1..4`` (default ``quagmire3``).
    restarts  : annealing restarts *per period*.
    seed      : RNG seed for reproducibility.
    iters     : annealing iterations per restart (auto from length if None).
    scorer    : optional pre-built scorer (defaults to hexagrams fitness).

    Returns a dict with ``score, plaintext, key, alphabet, period`` (and
    ``pt_alphabet``, ``ct_alphabet``, ``shifts``, ``kind`` for completeness).
    """
    ct = only_letters(ct)
    if len(ct) < 8:
        raise ValueError("ciphertext too short")
    if kind not in _KIND_SPEC:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(_KIND_SPEC)}")
    spec = _KIND_SPEC[kind]
    if scorer is None:
        scorer = resolve_scorer("hexagrams")

    if isinstance(periods, int):
        periods = [periods]
    periods = list(periods)

    rng = random.Random(seed)
    n = len(ct)

    best = None  # (score, pa, ca, shifts, pt, period)
    for p in periods:
        if p < 1 or p > n:
            continue
        # auto iteration budget: longer alphabets / shorter text -> more iters
        it = iters if iters is not None else max(600, min(4000, 1500 + 40 * p))
        p_best = None
        for r in range(restarts):
            score, pa, ca, shifts, pt = _anneal_one(
                ct, p, spec, scorer, rng, it
            )
            if p_best is None or score > p_best[0]:
                p_best = (score, pa, ca, shifts, pt)
        # polish the period's best
        score, pa, ca, shifts, pt = _polish(
            ct, p_best[1], p_best[2], p_best[3], spec, scorer, rng
        )
        if verbose:
            ic = index_of_coincidence(pt)
            print(f"  period {p:2d}: score={score:.4f} IoC={ic:.4f} {pt[:48]}")
        if best is None or score > best[0]:
            best = (score, pa, ca, shifts, pt, p)

    score, pa, ca, shifts, pt, period = best
    alphabet = pa if spec[0] else (ca if spec[1] else ALPHABET)
    from .validate import solve_confidence
    conf = solve_confidence(pt, len(ct))
    return {
        "score": score,
        "plaintext": pt,
        "key": _shifts_to_key(shifts, ca, pa),
        "alphabet": alphabet,
        "period": period,
        "pt_alphabet": pa,
        "ct_alphabet": ca,
        "shifts": list(shifts),
        "kind": kind,
        "recovered": conf["recovered"],
        "word_coverage": conf["word_coverage"],
    }


def _as_alphabet(spec: str) -> str:
    """Accept a full 26-letter alphabet or a keyword; return the keyed alphabet."""
    s = "".join(ch for ch in str(spec).upper() if "A" <= ch <= "Z")
    if len(s) == 26 and set(s) == set(ALPHABET):
        return s
    return _keyed_alphabet(s)


def solve_fixed_alphabet(
    ct: str,
    alphabet: str = KRYPTOS_ALPHABET,
    *,
    kind: str = "quagmire3",
    periods: Iterable[int] = range(2, 21),
    ct_alphabet: str | None = None,
    scorer=None,
) -> dict:
    """Solve a periodic keyed-alphabet cipher when the alphabet is **known**.

    Unlike :func:`solve` (which anneals the 26-letter alphabet over many restarts),
    this fixes the alphabet(s) and recovers only the per-period shifts via the cheap
    per-column chi-squared fit — orders of magnitude faster, and the right tool
    whenever the alphabet is given (e.g. the KRYPTOS family, where every layer uses
    the ``KRYPTOS`` keyed alphabet). Sweeps ``periods`` and returns the best, in the
    same dict shape as :func:`solve`.

    ``alphabet`` may be a full 26-letter alphabet or a keyword (expanded to a keyed
    alphabet). For Quagmire IV pass a second alphabet via ``ct_alphabet``.
    """
    ct = only_letters(ct)
    if len(ct) < 8:
        raise ValueError("ciphertext too short")
    # "beaufort" is the reciprocal of vigenere, decoded with the Beaufort rule p = key - c.
    # Shape it like Quagmire III (one linked keyed alphabet) so the KRYPTOS default gives
    # beaufort-over-KRYPTOS (the family case); pass --alphabet ABC..Z for classic Beaufort.
    beaufort = kind == "beaufort"
    spec_kind = "quagmire3" if beaufort else kind
    if spec_kind not in _KIND_SPEC:
        raise ValueError(
            f"unknown kind {kind!r}; choose from {sorted(_KIND_SPEC) + ['beaufort']}"
        )
    if scorer is None:
        scorer = resolve_scorer("hexagrams")
    pt_keyed, ct_keyed, linked = _KIND_SPEC[spec_kind]
    keyed = _as_alphabet(alphabet)
    pa = keyed if pt_keyed else ALPHABET
    if linked:
        ca = pa
    elif ct_keyed:
        ca = _as_alphabet(ct_alphabet) if ct_alphabet is not None else keyed
    else:
        ca = ALPHABET

    if isinstance(periods, int):
        periods = [periods]
    n = len(ct)
    best = None  # (score, shifts, pt, period)
    for p in periods:
        if p < 1 or p > n:
            continue
        # cheap per-column chi-square pre-pick, then refine the shifts (alphabet stays
        # fixed) by coordinate ascent on the real scorer to fix any column mispicks.
        shifts, score = _best_shifts(ct, pa, ca, p, scorer, beaufort=beaufort)
        shifts = list(shifts)
        for _ in range(2):
            improved = False
            for j in range(p):
                cur = shifts[j]
                best_s, best_sc = cur, scorer.fitness(_decrypt(ct, pa, ca, shifts, beaufort=beaufort))
                for s in range(26):
                    if s == cur:
                        continue
                    shifts[j] = s
                    sc = scorer.fitness(_decrypt(ct, pa, ca, shifts, beaufort=beaufort))
                    if sc > best_sc:
                        best_sc, best_s = sc, s
                shifts[j] = best_s
                if best_s != cur:
                    improved = True
            if not improved:
                break
        score = scorer.fitness(_decrypt(ct, pa, ca, shifts, beaufort=beaufort))
        if best is None or score > best[0]:
            best = (score, shifts, _decrypt(ct, pa, ca, shifts, beaufort=beaufort), p)
    if best is None:
        raise ValueError("no valid period in range")
    score, shifts, pt, period = best
    alpha_out = pa if pt_keyed else (ca if ct_keyed else ALPHABET)
    from .validate import solve_confidence
    conf = solve_confidence(pt, len(ct))
    return {
        "score": score,
        "plaintext": pt,
        "key": _shifts_to_key(shifts, ca, pa),
        "alphabet": alpha_out,
        "period": period,
        "pt_alphabet": pa,
        "ct_alphabet": ca,
        "shifts": list(shifts),
        "kind": kind,
        "recovered": conf["recovered"],
        "word_coverage": conf["word_coverage"],
    }


# ---------------------------------------------------------------------------
# Self-test: plant synthetics of the exact target structure and assert recovery.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    REF = (
        "WHENINTHECOURSEOFHUMANEVENTSITBECOMESNECESSARYFORONEPEOPLETODISSOLVE"
        "THEPOLITICALBANDSWHICHHAVECONNECTEDTHEMWITHANOTHERANDTOASSUMEAMONGTHE"
        "POWERSOFTHEEARTHTHESEPARATEANDEQUALSTATIONTOWHICHTHELAWSOFNATUREANDOF"
        "NATURESGODENTITLETHEMADECENTRESPECTTOTHEOPINIONSOFMANKINDREQUIRESTHAT"
        "THEYSHOULDDECLARETHECAUSESWHICHIMPELTHEMTOTHESEPARATION"
    )

    def _charmatch(a: str, b: str) -> float:
        m = min(len(a), len(b))
        if m == 0:
            return 0.0
        return sum(x == y for x, y in zip(a[:m], b[:m])) / m

    scorer = resolve_scorer("hexagrams")
    print(f"scorer: {scorer.name} (n={scorer.n}); reference length: {len(REF)} letters")

    # ---- Test 1: Quagmire III, KRYPTOS keyed alphabet, random period-13 key ----
    rng = random.Random(20260622)
    pa = ca = KRYPTOS_ALPHABET  # Quag III: same keyed alphabet both sides
    p13 = 13
    true_shifts = [rng.randrange(26) for _ in range(p13)]
    ct1 = _encrypt(REF, pa, ca, true_shifts)
    print(f"\n[Test 1] Quagmire III  period={p13}  "
          f"true key={''.join(ALPHABET[s] for s in true_shifts)}")
    res1 = solve(ct1, periods=[p13], kind="quagmire3", restarts=24, seed=1,
                 scorer=scorer, verbose=True)
    cm1 = _charmatch(res1["plaintext"], REF)
    print(f"  recovered alphabet: {res1['alphabet']}")
    print(f"  recovered key     : {res1['key']}")
    print(f"  char-match        : {cm1:.1%}")
    print(f"  plaintext[:80]    : {res1['plaintext'][:80]}")

    # ---- Test 2: plain Vigenere, period 10 ----
    rng2 = random.Random(7)
    keyword = "CIPHERWORD"  # length 10
    vshifts = [ALPHABET.index(c) for c in keyword]
    ct2 = _encrypt(REF, ALPHABET, ALPHABET, vshifts)
    print(f"\n[Test 2] Vigenere      period=10  true key={keyword}")
    res2 = solve(ct2, periods=[10], kind="vigenere", restarts=12, seed=2,
                 scorer=scorer, verbose=True)
    cm2 = _charmatch(res2["plaintext"], REF)
    print(f"  recovered key     : {res2['key']}")
    print(f"  char-match        : {cm2:.1%}")
    print(f"  plaintext[:80]    : {res2['plaintext'][:80]}")

    # ---- Test 3 (bonus): blind period sweep for the Quag III sample ----
    print(f"\n[Test 3] Quagmire III blind period sweep 8..16")
    res3 = solve(ct1, periods=range(8, 17), kind="quagmire3", restarts=8, seed=3,
                 scorer=scorer)
    cm3 = _charmatch(res3["plaintext"], REF)
    print(f"  found period={res3['period']}  char-match={cm3:.1%}  "
          f"plaintext[:60]={res3['plaintext'][:60]}")

    print("\n=== RESULTS ===")
    print(f"Test1 (Quag III p13): {cm1:.1%}  -> {'PASS' if cm1 >= 0.85 else 'FAIL'}")
    print(f"Test2 (Vigenere p10): {cm2:.1%}  -> {'PASS' if cm2 >= 0.85 else 'FAIL'}")
    print(f"Test3 (blind sweep) : {cm3:.1%}  period={res3['period']}  "
          f"-> {'PASS' if cm3 >= 0.85 else 'FAIL'}")

    overall = cm1 >= 0.85 and cm2 >= 0.85
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    import sys
    sys.exit(0 if overall else 1)
