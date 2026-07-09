"""Homophonic / simple substitution solver (AZdecrypt-style).

A homophonic cipher maps *many* ciphertext symbols onto *one* plaintext letter
(many-to-one). When the ciphertext alphabet is plain A-Z and each symbol maps to
a distinct letter, this degenerates to a classic monoalphabetic substitution.
This module implements the general case with simulated annealing over the
``symbol -> plaintext letter`` mapping.

The objective is :meth:`buttcrack.scoring.NgramScorer.fitness` of the decrypt
using the bundled English **hexagram** model (``resolve_scorer('hexagrams')``),
which already folds in an entropy/multiplicity guard so degenerate low-entropy
mappings (everything -> ``E``) are penalised. For speed the inner annealing loop
scores incrementally against the same hexagram log-probability table and only
re-derives the canonical ``fitness`` at the end / at acceptance checkpoints, so
the reported ``score`` is exactly ``resolve_scorer('hexagrams').fitness(plaintext)``.

Public API
----------
``solve(ct, restarts=30, iters=20000) -> {"score", "plaintext", "mapping"}``
"""

from __future__ import annotations

import math
import random

try:  # normal package import
    from .scoring import ENGLISH_LETTER_ENTROPY, resolve_scorer
    from .text import only_letters
except ImportError:  # allow running this file directly for the self-test
    from buttcrack.scoring import ENGLISH_LETTER_ENTROPY, resolve_scorer
    from buttcrack.text import only_letters

_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _decrypt(symbols: list[int], mapping: list[int]) -> str:
    """Apply ``mapping`` (symbol-index -> plaintext-letter-index) to symbols."""
    return "".join(_ALPHA[mapping[s]] for s in symbols)


def _hexagram_tables(scorer):
    """Pack the scorer's loaded hexagram table into fast lookup structures.

    Returns ``(n, log_probs, floor)`` where ``log_probs`` keys are integer
    base-26 codes of the n letters (fast, no string slicing in the hot loop).
    """
    n = scorer.n
    floor = scorer.floor
    coded: dict[int, float] = {}
    for gram, lp in scorer.log_probs.items():
        code = 0
        for ch in gram:
            code = code * 26 + (ord(ch) - 65)
        coded[code] = lp
    return n, coded, floor


def _raw_ngram_total(plain_idx: list[int], n: int, coded: dict, floor: float) -> float:
    """Sum of hexagram log-probabilities over the decrypted index stream."""
    L = len(plain_idx)
    if L < n:
        return floor * max(1, L)
    total = 0.0
    # rolling base-26 code over a window of n letters
    base = 26 ** (n - 1)
    code = 0
    for i in range(n):
        code = code * 26 + plain_idx[i]
    total += coded.get(code, floor)
    for i in range(n, L):
        code = (code - plain_idx[i - n] * base) * 26 + plain_idx[i]
        total += coded.get(code, floor)
    return total


def _inner_fitness(plain_idx: list[int], n: int, coded: dict, floor: float) -> float:
    """Reproduce ``NgramScorer.fitness`` from index stream (entropy-guarded)."""
    L = len(plain_idx)
    windows = L - n + 1
    if windows <= 0:
        return 0.0
    avg = _raw_ngram_total(plain_idx, n, coded, floor) / windows
    counts = [0] * 26
    for v in plain_idx:
        counts[v] += 1
    H = -sum((c / L) * math.log2(c / L) for c in counts if c)
    return (avg - floor) * (H / ENGLISH_LETTER_ENTROPY)


def solve(ct: str, restarts: int = 30, iters: int = 20000) -> dict:
    """Solve a simple/homophonic substitution cipher by simulated annealing.

    Parameters
    ----------
    ct:
        Ciphertext. Non-letters are stripped; remaining A-Z are treated as the
        ciphertext symbols (general homophonic solvers accept arbitrary symbol
        sets — here the symbol set is whatever distinct A-Z letters appear).
    restarts:
        Number of independent annealing runs (best result kept).
    iters:
        Annealing iterations per restart.

    Returns
    -------
    dict with keys:
        ``score``     -- ``resolve_scorer('hexagrams').fitness(plaintext)``
        ``plaintext`` -- best decrypt (A-Z)
        ``mapping``   -- ``{ciphertext_symbol: plaintext_letter}``
    """
    letters = only_letters(ct)
    if not letters:
        return {
            "score": 0.0,
            "plaintext": "",
            "mapping": {},
            "recovered": False,
            "word_coverage": 0.0,
        }

    scorer = resolve_scorer("hexagrams")
    n, coded, floor = _hexagram_tables(scorer)

    # Distinct ciphertext symbols -> contiguous symbol indices.
    syms = sorted(set(letters))
    sym_index = {s: i for i, s in enumerate(syms)}
    n_syms = len(syms)
    symbols = [sym_index[ch] for ch in letters]
    L = len(symbols)

    rng = random.Random(0xC0FFEE)

    # Frequency-matched seed: most-frequent ciphertext symbol -> most-frequent
    # English letter (E T A O I N S H R D L C U M W F G Y P B V K J X Q Z).
    eng_order = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
    sym_counts = [0] * n_syms
    for s in symbols:
        sym_counts[s] += 1
    freq_seed = [0] * n_syms
    for rank, s in enumerate(sorted(range(n_syms), key=lambda i: -sym_counts[i])):
        freq_seed[s] = ord(eng_order[rank % 26]) - 65

    # Positions where each symbol occurs (for incremental rescoring of a move).
    positions: list[list[int]] = [[] for _ in range(n_syms)]
    for i, s in enumerate(symbols):
        positions[s].append(i)

    # Whether the cipher is (currently) a bijection: every symbol maps to a
    # distinct plaintext letter. Simple substitutions are; swap moves keep this.
    base = 26 ** (n - 1)

    def raw_total(plain_idx: list[int]) -> float:
        if L < n:
            return floor * max(1, L)
        total = 0.0
        code = 0
        for i in range(n):
            code = code * 26 + plain_idx[i]
        total += coded.get(code, floor)
        for i in range(n, L):
            code = (code - plain_idx[i - n] * base) * 26 + plain_idx[i]
            total += coded.get(code, floor)
        return total

    def windows_score(plain_idx: list[int], affected: set[int]) -> float:
        """Sum of log-probs of every n-window that overlaps an affected index."""
        wins = set()
        for p in affected:
            lo = max(0, p - n + 1)
            hi = min(L - n, p)
            for w in range(lo, hi + 1):
                wins.add(w)
        s = 0.0
        for w in wins:
            code = 0
            for k in range(w, w + n):
                code = code * 26 + plain_idx[k]
            s += coded.get(code, floor)
        return s

    def entropy_factor(mapping: list[int]) -> float:
        counts = [0] * 26
        for s in range(n_syms):
            counts[mapping[s]] += sym_counts[s]
        H = -sum((c / L) * math.log2(c / L) for c in counts if c)
        return H / ENGLISH_LETTER_ENTROPY

    def canonical_fitness(mapping: list[int]) -> float:
        plain_idx = [mapping[s] for s in symbols]
        windows = L - n + 1
        if windows <= 0:
            return 0.0
        avg = raw_total(plain_idx) / windows
        return (avg - floor) * entropy_factor(mapping)

    best_mapping = list(freq_seed)
    best_fit = canonical_fitness(best_mapping)

    for r in range(restarts):
        if r == 0:
            mapping = list(freq_seed)
        else:
            mapping = [rng.randrange(26) for _ in range(n_syms)]
        plain_idx = [mapping[s] for s in symbols]
        cur_raw = raw_total(plain_idx)

        # Temperatures in *raw log-prob* units. A single symbol reassignment
        # touches its occurrences * n windows; deltas are O(tens) of log10, so
        # we cool from a permissive start down to near-greedy.
        T0, T1 = 8.0, 0.05
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / iters)

            # Move: 80% swap two symbols' targets (keeps a bijection -> ideal
            # for simple substitution), 20% reassign one symbol to a random
            # letter (needed for true many-to-one homophonic recovery).
            if n_syms >= 2 and rng.random() < 0.80:
                a = rng.randrange(n_syms)
                b = rng.randrange(n_syms)
                while b == a:
                    b = rng.randrange(n_syms)
                affected = set(positions[a])
                affected.update(positions[b])
                before = windows_score(plain_idx, affected)
                va, vb = mapping[a], mapping[b]
                mapping[a], mapping[b] = vb, va
                for p in positions[a]:
                    plain_idx[p] = vb
                for p in positions[b]:
                    plain_idx[p] = va
                after = windows_score(plain_idx, affected)
                d = after - before
                if d >= 0 or rng.random() < math.exp(d / T):
                    cur_raw += d
                else:  # revert
                    mapping[a], mapping[b] = va, vb
                    for p in positions[a]:
                        plain_idx[p] = va
                    for p in positions[b]:
                        plain_idx[p] = vb
            else:
                s = rng.randrange(n_syms)
                old = mapping[s]
                new = rng.randrange(26)
                if new == old:
                    new = (old + 1 + rng.randrange(25)) % 26
                affected = set(positions[s])
                before = windows_score(plain_idx, affected)
                mapping[s] = new
                for p in positions[s]:
                    plain_idx[p] = new
                after = windows_score(plain_idx, affected)
                d = after - before
                if d >= 0 or rng.random() < math.exp(d / T):
                    cur_raw += d
                else:
                    mapping[s] = old
                    for p in positions[s]:
                        plain_idx[p] = old

        # Evaluate this restart's endpoint with the canonical entropy-guarded
        # fitness and keep the global best.
        cand_fit = canonical_fitness(mapping)
        if cand_fit > best_fit:
            best_fit = cand_fit
            best_mapping = list(mapping)

    plaintext = _decrypt(symbols, best_mapping)
    canonical = scorer.fitness(plaintext)  # faithful to the stated objective
    mapping_out = {syms[i]: _ALPHA[best_mapping[i]] for i in range(n_syms)}
    from .validate import solve_confidence

    conf = solve_confidence(plaintext, len(letters))
    return {
        "score": canonical,
        "plaintext": plaintext,
        "mapping": mapping_out,
        "recovered": conf["recovered"],
        "word_coverage": conf["word_coverage"],
    }


# ---------------------------------------------------------------------------
# Self-test: plant a simple monoalphabetic substitution and assert recovery.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    PLAIN = (
        "WE ARE NOT ALONE IN THE UNIVERSE AND THE STARS HAVE ALWAYS "
        "GUIDED THOSE WHO DARED TO WANDER BEYOND THE EDGE OF THE KNOWN "
        "WORLD WHERE THE OCEAN MEETS THE SKY AND THE WIND CARRIES OLD "
        "STORIES OF FORGOTTEN KINGS WHO ONCE RULED THESE QUIET LANDS "
        "WITH WISDOM AND COURAGE AND A STEADY HAND UPON THE WHEEL OF "
        "FATE THAT TURNS FOR EVERY LIVING SOUL BENEATH THE SILVER MOON "
        "AND SO WE SAIL AGAIN INTO THE QUIET DARK OF ANOTHER DISTANT DAWN"
    )
    plain = only_letters(PLAIN)
    assert len(plain) >= 300, f"need >=300 letters, got {len(plain)}"

    # Plant a random monoalphabetic substitution (plaintext letter -> cipher).
    rng = random.Random(1234)
    perm = list(_ALPHA)
    rng.shuffle(perm)
    enc = {p: c for p, c in zip(_ALPHA, perm, strict=False)}
    ct = "".join(enc[ch] for ch in plain)

    t0 = time.time()
    res = solve(ct, restarts=20, iters=12000)
    dt = time.time() - t0

    recovered = res["plaintext"]
    matches = sum(1 for a, b in zip(recovered, plain, strict=False) if a == b)
    pct = 100.0 * matches / len(plain)

    print(f"letters           : {len(plain)}")
    print(f"distinct symbols  : {len(set(ct))}")
    print(f"elapsed           : {dt:.1f}s")
    print(f"fitness score     : {res['score']:.4f}")
    print(f"char-match        : {matches}/{len(plain)} = {pct:.1f}%")
    print(f"plaintext (head)  : {plain[:80]}")
    print(f"recovered (head)  : {recovered[:80]}")
    print(f"verify fitness    : {resolve_scorer('hexagrams').fitness(recovered):.4f}")
    assert abs(res["score"] - resolve_scorer("hexagrams").fitness(recovered)) < 1e-9, (
        "reported score must equal resolve_scorer('hexagrams').fitness(plaintext)"
    )
    status = "PASS" if pct >= 90.0 else "FAIL"
    print(f"RESULT            : {status} (threshold 90%)")
    assert pct >= 90.0, f"recovery {pct:.1f}% below 90% threshold"
