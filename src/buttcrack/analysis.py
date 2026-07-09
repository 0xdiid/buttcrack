"""Statistical analysis tools — buttcrack's answer to CryptoCrack's analysis menus.

Pure, read-only reporting an agent (or human) can use to reason about a cipher
before/independent of cracking: letter & digraph frequencies, index of
coincidence, chi-squared fit, and Kasiski examination for period estimation.
"""

from __future__ import annotations

import functools
import itertools
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import cast

from .keysources import _alphabet
from .scoring import (
    ENGLISH_MONOGRAM_FREQ,
    chi_squared,
    index_of_coincidence,
    letter_entropy,
)
from .text import only_letters

try:  # optional acceleration only; the package itself stays dependency-free
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in dev/test
    _np = None  # type: ignore[assignment]  # optional-dependency fallback sentinel

_UNITS26 = tuple(u for u in range(26) if math.gcd(u, 26) == 1)


def _ngram_counts(letters: str, n: int, top: int) -> list[dict]:
    counts = Counter(letters[i : i + n] for i in range(len(letters) - n + 1))
    return [{"gram": gram, "count": c} for gram, c in counts.most_common(top)]


def _divisors(n: int, max_period: int = 20) -> list[int]:
    return [d for d in range(2, min(n, max_period) + 1) if n % d == 0]


def kasiski(letters: str, seq_len: int = 3, top: int = 12) -> tuple[list[dict], list[dict]]:
    """Repeated-sequence spacings and the key periods their factors suggest."""
    occ: dict[str, list[int]] = {}
    for i in range(len(letters) - seq_len + 1):
        occ.setdefault(letters[i : i + seq_len], []).append(i)

    repeats: list[dict] = []
    factor_tally: Counter = Counter()
    for gram, positions in occ.items():
        if len(positions) < 2:
            continue
        spacings = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
        for sp in spacings:
            for f in _divisors(sp):
                factor_tally[f] += 1
        repeats.append({"gram": gram, "count": len(positions), "spacings": spacings})

    repeats.sort(key=lambda r: r["count"], reverse=True)
    likely = [{"period": p, "weight": c} for p, c in factor_tally.most_common(5)]
    return repeats[:top], likely


def block_transposition_signal(
    text: str, *, ngram: int = 3, min_count: int = 2, max_block: int = 8
) -> dict:
    """Detect a BLOCK-granular cipher/transposition from the alignment of repeated n-grams.

    A block-of-``b`` transposition (or a ``b``-graph block cipher such as Hill) relocates or
    maps whole ``b``-letter blocks, so every repeated ciphertext n-gram must start at a
    position ``== 0 (mod b)``. Plaintext, single-letter substitutions, and running keys
    scatter repeats across all residues. This z-scores the observed mod-``b`` alignment of
    all repeated-n-gram occurrences against the uniform null and reports the largest block
    size at which every repeat aligns — the "trigraph-granular transposition" signature.
    (A trigraph block cipher fingerprints exactly this way: e.g. two trigrams whose six
    occurrences are all ``== 0 (mod 3)``.) Combined with a flat letter-IoC it also flags the
    classic "a periodic substitution is hidden INSIDE the transposition" structure.
    """
    letters = only_letters(text)
    occ: dict[str, list[int]] = {}
    for i in range(len(letters) - ngram + 1):
        occ.setdefault(letters[i : i + ngram], []).append(i)
    repeats = sorted(
        ((g, p) for g, p in occ.items() if len(p) >= min_count),
        key=lambda gp: -len(gp[1]),
    )
    positions = [i for _, ps in repeats for i in ps]
    k = len(positions)
    # Strongly-repeated grams (>=3x) are the reliable signal: boundary-spanning windows
    # coincidentally repeat 2x and add mod-b noise, but a 3x+ repeat that is block-aligned
    # is the block-cipher/transposition fingerprint (e.g. two distinct 3x trigrams, all == 0 mod 3).
    strong_pos = [i for _, ps in repeats if len(ps) >= 3 for i in ps]

    def _binom_tail(kk: int, a: int, p: float) -> float:
        """Exact upper tail P(X >= a) for X ~ Binomial(kk, p)."""
        if a <= 0:
            return 1.0
        return sum(math.comb(kk, j) * p**j * (1 - p) ** (kk - j) for j in range(a, kk + 1))

    # Test the RELIABLE (>=3x) repeats when we have a handful; else fall back to all repeats.
    # A block-of-b transposition relocates whole b-blocks, so every reliable repeat must share
    # ONE residue mod b (not necessarily residue 0 — a phase-offset grid aligns at r != 0, which
    # the old residue-0-only test missed entirely). We find the max-count residue per b and
    # score it with the EXACT binomial tail (the normal-approx z is unreliable at these small k
    # and its >=6-count gate silently missed legitimate small-k signals: a 5x trigram all-aligned
    # is p=(1/3)^5=0.004, a bridged 4x is p=0.012 — both real, both previously dropped).
    pos_set = strong_pos if len(strong_pos) >= 4 else positions
    kk = len(pos_set)
    ntests = max(1, max_block - 1)  # Bonferroni over the block-size range
    alignment: dict[int, dict] = {}
    best_block = None
    for b in range(2, max_block + 1):
        res_counts = Counter(i % b for i in pos_set)
        residue, aligned = res_counts.most_common(1)[0] if res_counts else (0, 0)
        exp = kk / b
        var = kk * (1.0 / b) * (1 - 1.0 / b)
        z = (aligned - exp) / math.sqrt(var) if var > 0 else 0.0
        p_raw = _binom_tail(kk, aligned, 1.0 / b) if kk else 1.0
        alignment[b] = {
            "aligned": aligned,
            "residue": residue,
            "total": kk,
            "expected": round(exp, 2),
            "z": round(z, 2),
            "p": round(p_raw, 6),
            "p_corrected": round(min(1.0, p_raw * ntests), 6),
        }
        # Require the reliable set to FULLY share a residue (aligned == kk) and clear an exact
        # significance bar; prefer the LARGEST such block (a genuine block-of-b aligns at b and
        # every divisor, so the largest aligning block is the true granularity).
        if kk >= 4 and aligned == kk and p_raw <= 0.02:
            best_block = b
    ioc = index_of_coincidence(letters)
    if best_block:
        ab = alignment[best_block]
        verdict = (
            f"repeated {ngram}-grams all start at position == {ab['residue']} (mod {best_block}) "
            f"(exact p={ab['p']}): a block-of-{best_block} transposition or a "
            f"{best_block}-graph block cipher (e.g. Hill) — single-letter ciphers cannot do this"
        )
        if ioc < 0.05:
            verdict += (
                f"; the flat letter-IoC ({ioc:.4f}) shows a substitution is present but not "
                f"outermost-periodic, i.e. a periodic substitution likely sits INSIDE the "
                f"block transposition"
            )
    elif k == 0:
        verdict = (
            f"no repeated {ngram}-grams — typical of a non-repeating/running key or a "
            f"thorough polyalphabetic with no block structure"
        )
    else:
        verdict = (
            f"repeated {ngram}-grams present but not block-aligned — likely plaintext repeats "
            f"(a transposition of readable text) or chance"
        )
    return {
        "ngram": ngram,
        "repeated_ngrams": [{"gram": g, "count": len(p), "positions": p} for g, p in repeats[:12]],
        "occurrences": k,
        "alignment": alignment,
        "best_block": best_block,
        "letter_ioc": round(ioc, 4),
        "verdict": verdict,
    }


def contacts(text: str) -> list[dict]:
    """Variety-of-contacts: distinct left/right neighbours per letter.

    Vowels contact a wide variety of letters (high variety); low-variety
    high-frequency letters tend to be consonants — the classic aid for spotting
    vowels in an Aristocrat.
    """
    letters = only_letters(text)
    n = len(letters)
    left: dict[str, set] = {}
    right: dict[str, set] = {}
    counts: Counter = Counter(letters)
    for i, ch in enumerate(letters):
        if i > 0:
            left.setdefault(ch, set()).add(letters[i - 1])
        if i < n - 1:
            right.setdefault(ch, set()).add(letters[i + 1])
    rows = [
        {
            "letter": ch,
            "count": counts[ch],
            "left_variety": len(left.get(ch, ())),
            "right_variety": len(right.get(ch, ())),
            "variety": len(left.get(ch, set()) | right.get(ch, set())),
        }
        for ch in sorted(counts, key=lambda c: -counts[c])
    ]
    return rows


def _mean_col_ioc(seq: list[int], p: int) -> float:
    """Mean per-column index of coincidence of integer sequence ``seq`` at period ``p``."""
    total = 0.0
    cols = 0
    for j in range(p):
        col = seq[j::p]
        m = len(col)
        if m < 2:
            continue
        counts = [0] * 26
        for x in col:
            counts[x] += 1
        total += sum(k * (k - 1) for k in counts) / (m * (m - 1))
        cols += 1
    return total / cols if cols else 0.0


@functools.lru_cache(maxsize=64)
def _ioc_baseline(n: int, max_period: int, samples: int, min_col: int, seed: int) -> dict:
    """Random-text mean per-column IoC (mean, std) per period — cached by length.

    The baseline depends only on the message length and split, not its content, so
    it's computed once per ``n`` and reused (the Monte-Carlo is the expensive part).
    """
    rng = random.Random(seed)
    pool = [[rng.randrange(26) for _ in range(n)] for _ in range(samples)]
    base: dict[int, tuple[float, float]] = {}
    for p in range(2, max_period + 1):
        if any(len(range(j, n, p)) < min_col for j in range(p)):
            continue
        vals = [_mean_col_ioc(s, p) for s in pool]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1e-9
        base[p] = (mu, sd)
    return base


def transposition_periods(
    letters: str,
    *,
    max_period: int | None = None,
    samples: int = 80,
    seed: int = 20250615,
    top: int = 4,
) -> list[dict]:
    """Periods at which whole *bigrams exactly recur* vs shuffles of the same letters.

    Counts positions where ``seq[i:i+2] == seq[i+p:i+p+2]`` and z-scores the count
    against shuffles of the same letter multiset. This detects *regular repeated
    structure at a fixed lag* — repeated-key / route / periodic-fill patterns, or a
    repeated plaintext under a monoalphabetic substitution. It is **not** a general
    columnar-transposition period finder: an ordinary keyed columnar does not leave
    exact bigram repeats at its width, so this stays silent on it (verified). Nor does
    it detect a homophonic layer, which destroys exact repeats by design. Conservative
    on purpose — it requires both a meaningful absolute count (>= max(6, 4% of length))
    and a high z, so plain English doesn't false-spike; it stays silent rather than
    guess. Surfaced as informational ``stats``, not a confident claim.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 80:
        return []
    if max_period is None:
        max_period = min(n // 4, 50)
    vals = [ord(c) - 65 for c in letters]
    min_count = max(6, int(0.04 * n))

    def bigram_autocorr(seq: list[int], p: int) -> int:
        return sum(
            1 for i in range(n - p - 1) if seq[i] == seq[i + p] and seq[i + 1] == seq[i + 1 + p]
        )

    rng = random.Random(seed)
    pool = []
    for _ in range(samples):
        s = vals[:]
        rng.shuffle(s)
        pool.append(s)
    out: list[dict] = []
    for p in range(2, max_period + 1):
        obs = bigram_autocorr(vals, p)
        if obs < min_count:
            continue
        base = [bigram_autocorr(s, p) for s in pool]
        mu = sum(base) / len(base)
        sd = (sum((b - mu) ** 2 for b in base) / len(base)) ** 0.5 or 1e-9
        z = (obs - mu) / sd
        if z >= 4.0:
            out.append({"period": p, "repeats": obs, "baseline": round(mu, 1), "z": round(z, 2)})
    out.sort(key=lambda r: r["z"], reverse=True)
    return out[:top]


def calibrated_periods(
    letters: str,
    *,
    max_period: int | None = None,
    samples: int = 60,
    min_col: int = 4,
    seed: int = 20250615,
    top: int = 6,
) -> list[dict]:
    """Most significant periods by per-column IoC vs a random-text baseline.

    For each period the mean per-column index of coincidence is z-scored against
    the same statistic on random text split the same way (Monte-Carlo). This
    matters because a *long* key on a not-very-long message gives few letters per
    column (280 = 7x40 -> 7 letters/column at period 40), which depresses the
    *absolute* IoC enough to look like noise — but the calibrated z-score still
    exposes the real spike. Catching that is what separates a long-key periodic
    cipher (Vigenere/Quagmire) from a (mis-diagnosed) "no period / running key".

    Returns up to ``top`` periods sorted by z descending; ``z`` > ~3 is a strong
    signal. Empty for very short inputs.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 12:
        return []
    if max_period is None:
        max_period = min(n // min_col, 50)
    baseline = _ioc_baseline(n, max_period, samples, min_col, seed)
    lett = [ord(ch) - 65 for ch in letters]
    out: list[dict] = []
    for p, (mu, sd) in baseline.items():
        obs = _mean_col_ioc(lett, p)
        out.append(
            {
                "period": p,
                "ioc": round(obs, 4),
                "baseline": round(mu, 4),
                "z": round((obs - mu) / sd, 2),
            }
        )
    out.sort(key=lambda r: r["z"], reverse=True)
    return out[:top]


def _merge_ioc(idx: list[int], p: int) -> float:
    """Greedily superimpose the ``p`` cosets under additive shifts; return the merged IoC.

    Coset 0 is the anchor; each later coset is appended at the additive shift (mod 26) that
    maximises the running merged IoC. When the columns are additive shifts of one *peaked*
    distribution (a periodic shift cipher, or a period-``p`` shift laid over a mildly-peaked
    inner), the greedy alignment stacks their peaks and the merged IoC climbs; when the columns
    are *flat* (a strong flattener) there is no peak to align and it stays near the shuffle floor.
    """
    cosets = [idx[j::p] for j in range(p)]
    counts = [0] * 26
    for x in cosets[0]:
        counts[x] += 1
    for col in cosets[1:]:
        best_counts, best_ioc = counts, -1.0
        for s in range(26):
            trial = counts[:]
            for x in col:
                trial[(x - s) % 26] += 1
            m = sum(trial)
            v = sum(k * (k - 1) for k in trial) / (m * (m - 1)) if m > 1 else 0.0
            if v > best_ioc:
                best_ioc, best_counts = v, trial
        counts = best_counts
    m = sum(counts)
    return sum(k * (k - 1) for k in counts) / (m * (m - 1)) if m > 1 else 0.0


def superimposition_periods(
    letters: str,
    *,
    alphabet: str = "STANDARD",
    max_period: int | None = None,
    samples: int = 80,
    seed: int = 20250615,
    top: int = 6,
) -> list[dict]:
    """Periods whose columns *superimpose* under additive shifts (Kerckhoffs / Kerckhoffs test).

    Complements :func:`calibrated_periods`. Where coset-IoC asks "does each column carry
    structure", this asks "are the columns additive shifts of ONE common distribution" — it
    greedily shifts the ``p`` columns onto one another (:func:`_merge_ioc`) and z-scores the
    merged IoC against the same greedy alignment on shuffles of the same letters. A high z means
    period ``p`` is a clean per-column additive **shift** over an alignable (peaked) inner — a
    periodic shift cipher, or a ``shift_p`` laid over a mildly-peaked inner. It stays near the
    floor for a strong flattener (flat columns, nothing to align) or for no period. It does not
    by itself separate a true Vigenere from a Quagmire with independent keyed alphabets — both
    have peaked columns; use it to confirm a period is a *shift* and to rule a *flattener* in/out.

    ``alphabet`` is the ring the shifts are measured in (``"STANDARD"``, ``"KRYPTOS"``, or a
    26-letter permutation); the merge realigns most cleanly in the ring the key actually used.

    Returns up to ``top`` periods sorted by z descending. **Detects period existence, not the
    shift values** — at short lengths the greedy optimum overfits, so trust the z, not any
    recovered key vector. Empty for very short inputs.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 20:
        return []
    ring = _alphabet(alphabet)
    ring_idx = {c: i for i, c in enumerate(ring)}
    idx = [ring_idx[c] for c in letters]
    if max_period is None:
        max_period = min(n // 4, 30)
    rng = random.Random(seed)
    pool = []
    for _ in range(samples):
        s = idx[:]
        rng.shuffle(s)
        pool.append(s)
    out: list[dict] = []
    for p in range(2, max_period + 1):
        if any(len(range(j, n, p)) < 2 for j in range(p)):
            continue
        obs = _merge_ioc(idx, p)
        base = [_merge_ioc(s, p) for s in pool]
        mu = sum(base) / len(base)
        sd = (sum((b - mu) ** 2 for b in base) / len(base)) ** 0.5 or 1e-9
        out.append(
            {
                "period": p,
                "merged_ioc": round(obs, 4),
                "baseline": round(mu, 4),
                "z": round((obs - mu) / sd, 2),
            }
        )
    out.sort(key=lambda r: r["z"], reverse=True)
    return out[:top]


def period_inner_content(
    letters: str,
    period: int,
    *,
    samples: int = 200,
    seed: int = 20250615,
) -> dict:
    """Classify the layer *underneath* a detected period: natural language or flattened?

    A period is only half the story. Detecting period ``p`` (via Kasiski / calibrated
    per-column IoC) tells you the outer/last-applied layer is a period-``p`` polyalphabetic —
    but not what it sits on. The mean per-column IoC at ``p`` is a **mapping-invariant ceiling**:
    it is unchanged by any monoalphabetic-per-column substitution (Vigenere/Beaufort/Porta/
    Quagmire all preserve it), so it measures the index-of-coincidence of the text *under* the
    periodic layer. Compare that to English:

    * ``coset_ioc`` ~ English (``~0.067``) → the inner layer is **natural language** (or a pure
      transposition of it, which keeps English monogram IoC). A plain periodic substitution will
      peel it to readable text — align the columns (mutual-IoC) or run a Quagmire solve.
    * ``coset_ioc`` clearly **below English but above the ``~0.0385`` floor** (with a positive
      z vs. shuffles) → the inner layer is **flattened**: it is NOT a simple language, so a plain
      Vigenere/Quagmire peel will *never* reach readable text. Expect a **polygraphic/digraphic
      inner** (Playfair/two-square/four-square/Hill/fractionation) or a **non-prose payload**
      (a key, coordinates, a route) under the periodic layer.
    * ``coset_ioc`` ~ floor → there is no real period-``p`` structure here.

    This is the single check that most cleanly separates "period-P over English" (easy) from
    "period-P over a flattening cipher / payload" (needs a two-layer or crib attack) — the trap
    that makes a weak periodic signal look like a solvable Vigenere when it is not.

    Returns the coset IoC, the English and floor references, the fraction of the English→floor gap,
    a shuffle-null (mean/sd/z), and a ``verdict`` string.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < max(period * 4, 24) or period < 2:
        return {}
    english_ioc = sum(f * f for f in ENGLISH_MONOGRAM_FREQ.values())
    floor = 1.0 / 26.0
    seq = [ord(ch) - 65 for ch in letters]
    coset = _mean_col_ioc(seq, period)
    rng = random.Random(seed)
    nulls = []
    for _ in range(samples):
        rng.shuffle(seq)
        nulls.append(_mean_col_ioc(seq, period))
    mu = sum(nulls) / len(nulls)
    sd = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** 0.5 or 1e-9
    z = (coset - mu) / sd
    # how far from English toward the random floor (0 = English, 1 = floor)
    gap = (english_ioc - coset) / (english_ioc - floor) if english_ioc > floor else 1.0

    # Reliability: with few letters per column even *random* text inflates the per-coset IoC
    # (e.g. ~4 letters/col pushes the null well above the 0.0385 floor), so an absolute compare
    # to English is meaningless. Only trust the language-vs-flattened verdict when the null sits
    # near the floor. This is what stops a small-sample harmonic (period 35 = 5x7 on a short text)
    # from masquerading as a "natural-language inner".
    letters_per_col = n / period
    reliable = mu < floor + 0.010 and letters_per_col >= 12
    if not reliable:
        verdict = (
            f"period-{period} coset IoC {coset:.4f} but only ~{letters_per_col:.0f} "
            f"letters/column: too few to judge the inner content (random itself sits at {mu:.4f} "
            "here). Re-check the FUNDAMENTAL period (a smaller divisor with >=12 letters/column) "
            "instead."
        )
    elif z < 2.0:
        verdict = (
            f"no real period-{period} structure (coset IoC {coset:.4f} within noise of the "
            f"shuffle floor {mu:.4f}); this period is not the lever."
        )
    elif coset >= english_ioc - 2.0 * sd:
        verdict = (
            f"NATURAL-LANGUAGE inner: coset IoC {coset:.4f} ~ English {english_ioc:.4f}. The layer "
            f"under the period-{period} substitution is plain text (or a transposition of it) — "
            f"peel it by column alignment / Quagmire (and undo any inner transposition)."
        )
    else:
        verdict = (
            f"FLATTENED inner: coset IoC {coset:.4f} is real (z={z:.1f}) but well below English "
            f"{english_ioc:.4f} ({gap * 100:.0f}% of the way to random). The text under the "
            f"period-{period} layer is NOT a simple language — a plain Vigenere/Quagmire peel will "
            "not read. Expect a polygraphic/digraphic inner (Playfair/Hill/fractionation) or a "
            "non-prose payload."
        )
    return {
        "period": period,
        "coset_ioc": round(coset, 4),
        "english_ioc": round(english_ioc, 4),
        "floor": round(floor, 4),
        "gap_to_random": round(gap, 3),
        "letters_per_col": round(letters_per_col, 1),
        "reliable": reliable,
        "null_mean": round(mu, 4),
        "null_sd": round(sd, 4),
        "z": round(z, 2),
        "verdict": verdict,
    }


_STD_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


#: the inner-cipher classes :func:`inner_class_coset_ioc` can simulate.
INNER_CLASSES = ("language", "playfair", "bifid", "four_square", "hill2", "uniform")


def _sample_language_letters(m: int, rng: random.Random, freqs: dict[str, float]) -> str:
    """``m`` letters drawn from a language's monogram distribution (reproduces its IoC exactly)."""
    letters = list(freqs)
    weights = list(freqs.values())
    return "".join(rng.choices(letters, weights, k=m))


def _rand_square(rng: random.Random, drop: str = "J") -> str:
    """A random 25-letter Polybius square string (uniform keyed square) omitting ``drop``."""
    cells = [c for c in _STD_ALPHABET if c != drop]
    rng.shuffle(cells)
    return "".join(cells)


def _hill2_encode(text: str, rng: random.Random) -> str:
    """Encipher ``text`` with a random invertible 2x2 Hill matrix over Z26 (no key recovery)."""
    while True:
        a, b, c, d = (rng.randrange(26) for _ in range(4))
        det = (a * d - b * c) % 26
        if det % 2 and det % 13:  # gcd(det, 26) == 1
            break
    s = text if len(text) % 2 == 0 else text[:-1]
    out = []
    for i in range(0, len(s), 2):
        x, y = ord(s[i]) - 65, ord(s[i + 1]) - 65
        out.append(chr(65 + (a * x + b * y) % 26))
        out.append(chr(65 + (c * x + d * y) % 26))
    return "".join(out)


def inner_class_coset_ioc(
    n: int,
    period: int = 7,
    *,
    samples: int = 200,
    language: str = "english",
    corpus: str | None = None,
    bifid_period: int | None = None,
    seed: int = 0,
    classes: Sequence[str] = INNER_CLASSES,
) -> dict[str, dict[str, float]]:
    """Calibrate the period-``period`` coset IoC of each inner-cipher CLASS at length ``n``.

    Answers "which flattener could have produced this observed coset IoC?" by Monte-Carlo:
    for each class it applies that inner cipher (random key/square/matrix) to a language sample
    and measures the mean±sd coset IoC over ``samples`` trials. Classes (:data:`INNER_CLASSES`):

    * ``language``   — plain language (coset IoC ≈ the language's IoC, ~0.066 EN);
    * ``playfair``   — 5x5 Playfair (digraphic; ~0.052);
    * ``bifid``      — bifid seriated at ``bifid_period`` (default = ``period``);
    * ``four_square``— four-square over two random squares (~0.052);
    * ``hill2``      — 2x2 Hill (flattens toward the floor, ~0.040);
    * ``uniform``    — random letters (the ~0.0385 floor).

    Input letters are sampled from ``language``'s monogram frequencies (reproducing its exact
    letter distribution) unless a real ``corpus`` is supplied. Returns
    ``{class: {"mean": float, "sd": float}}``. Pair with :func:`classify_coset_ioc`.
    """
    from .ciphers.bifid import bifid_encode
    from .ciphers.playfair import Playfair
    from .scoring import get_scorer
    from .sub_four_square import four_square_encode, fs_alphabet

    bp = period if bifid_period is None else bifid_period
    if language == "english":
        freqs = {k: v for k, v in ENGLISH_MONOGRAM_FREQ.items()}
    else:
        sc = get_scorer("monograms", language)
        freqs = {chr(65 + i): 0.0 for i in range(26)}
        for g, lp in sc.log_probs.items():
            if len(g) == 1 and "A" <= g <= "Z":
                freqs[g] = 10**lp
        for k, v in list(freqs.items()):
            if v <= 0:
                freqs[k] = 1e-4

    pf = Playfair()
    corpus_letters = only_letters(corpus.upper()) if corpus else None
    rng = random.Random(seed)

    def draw_input(m: int) -> str:
        if corpus_letters and len(corpus_letters) > m:
            s = rng.randrange(len(corpus_letters) - m)
            return corpus_letters[s : s + m]
        return _sample_language_letters(m, rng, freqs)

    out: dict[str, dict[str, float]] = {}
    for cls in classes:
        vals: list[float] = []
        for _ in range(samples):
            src = draw_input(int(n * 1.4) + 8)
            if cls == "language":
                text = src
            elif cls == "uniform":
                text = "".join(rng.choices(_STD_ALPHABET, k=len(src)))
            elif cls == "playfair":
                text = pf.encode(src.replace("J", "I"), _rand_square(rng, "J"))
            elif cls == "bifid":
                text = bifid_encode(src.replace("J", "I"), _rand_square(rng, "J"), bp)
            elif cls == "four_square":
                a25 = fs_alphabet("Q")
                text = four_square_encode(
                    src.replace("Q", ""), _rand_square(rng, "Q"), _rand_square(rng, "Q"), a25
                )
            elif cls == "hill2":
                text = _hill2_encode(src, rng)
            else:
                raise ValueError(f"unknown inner class {cls!r}; choose from {INNER_CLASSES}")
            text = text[:n]
            vals.append(sum(index_of_coincidence(text[j::period]) for j in range(period)) / period)
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        out[cls] = {"mean": round(mu, 4), "sd": round(sd, 4)}
    return out


def classify_coset_ioc(
    observed: float,
    n: int,
    period: int = 7,
    *,
    tol: float = 1.5,
    **kwargs,
) -> list[dict[str, object]]:
    """Rank inner-cipher classes by how well their calibrated coset IoC matches ``observed``.

    Runs :func:`inner_class_coset_ioc` and returns, best-first, the classes within ``tol``
    standard deviations of ``observed`` as ``{"class", "mean", "sd", "z"}`` (``z`` = signed
    deviation of ``observed`` from the class mean in class-sd units). ``kwargs`` pass through
    (``samples``, ``language``, ``corpus``, ``bifid_period``, ``seed``, ``classes``).
    """
    cal = inner_class_coset_ioc(n, period, **kwargs)
    rows: list[dict[str, object]] = []
    for cls, st in cal.items():
        sd = st["sd"] or 1e-9
        z = (observed - st["mean"]) / sd
        if abs(z) <= tol:
            rows.append({"class": cls, "mean": st["mean"], "sd": st["sd"], "z": round(z, 2)})
    rows.sort(key=lambda r: abs(cast(float, r["z"])))
    return rows


@functools.cache
def _projective_surjective_covectors(k: int) -> tuple[tuple[int, ...], ...]:
    """One representative per projective class of *surjective* k-covectors mod 26.

    A covector ``v`` acts on a k-letter block ``b`` as ``v . b (mod 26)``. Its channel is
    surjective onto Z26 — capable of the full ``log2(26)`` entropy — iff ``gcd(v_1..v_k, 26) == 1``.
    That guard is the whole trick: it EXCLUDES zero-divisor covectors like ``(13, 13)`` whose
    channel only ever reaches ``{0, 13}`` and would otherwise fake a ~1-bit "signal" on any text.
    Scalar multiples by a unit permute the channel labels but not its partition (same entropy), so
    we keep one canonical representative per projective class.

    Covectors with a single non-zero coordinate are EXCLUDED: they just pick one ciphertext
    position, whose channel is concentrated for any periodic *monoalphabetic* sub (Vigenere/
    Quagmire) — so they would flag those as "Hill". A genuine Hill mixes the block's letters, so
    its concentrated covector has ≥2 non-zero coordinates. Keeping only mixing covectors makes the
    test specific to *polygraphic linearity* (it drops only degenerate triangular Hills).
    """
    units = [u for u in range(26) if math.gcd(u, 26) == 1]
    seen: set = set()
    reps: list = []
    for v in itertools.product(range(26), repeat=k):
        if sum(1 for e in v if e) < 2:  # exclude zero and single-position covectors
            continue
        g = 0
        for e in v:
            g = math.gcd(g, e)
        if math.gcd(g, 26) != 1:
            continue
        canon = min(tuple((u * e) % 26 for e in v) for u in units)
        if canon not in seen:
            seen.add(canon)
            reps.append(canon)
    return tuple(reps)


def _min_channel_entropy(
    indices: list[int], k: int, covectors: tuple[tuple[int, ...], ...], decimated: bool
) -> float:
    """Minimum Shannon entropy (bits) of ``v . block`` over all covectors ``v``."""
    n = len(indices) // k
    if decimated:  # block j = (indices[j], indices[j+n], ...) — the 3xN grid read column-wise
        blocks = [tuple(indices[j + t * n] for t in range(k)) for j in range(n)]
    else:  # contiguous k-letter blocks
        blocks = [tuple(indices[j * k + t] for t in range(k)) for j in range(n)]
    best = math.log2(26)
    for v in covectors:
        counts = [0] * 26
        for b in blocks:
            s = 0
            for t in range(k):
                s += v[t] * b[t]
            counts[s % 26] += 1
        ent = 0.0
        for c in counts:
            if c:
                p = c / n
                ent -= p * math.log2(p)
        if ent < best:
            best = ent
    return best


def linear_channel(
    letters: str,
    *,
    block_sizes: tuple[int, ...] = (2, 3),
    alphabet: str = _STD_ALPHABET,
    null_samples: int = 20,
    seed: int = 20250615,
) -> dict:
    """Detect a *linear* polygraphic (Hill) channel — distinguishing Hill from Playfair.

    Once a text looks polygraphic (flat monograms, e.g. :func:`period_inner_content` reports a
    flattened inner), the next question is whether that layer is **linear**. For a Hill cipher
    ``C = M . P`` on k-letter blocks, every covector ``v`` gives ``v . C = (M^T v) . P``, so *some*
    covector isolates a single plaintext coordinate — a concentrated, low-entropy channel whenever
    the plaintext is structured (a language, or any non-uniform source). A **nonlinear** digraphic
    cipher (Playfair / two-square / four-square) has no such linear channel, so this cleanly
    separates the two. It is **language-independent** (entropy, not an English gate), so it still
    fires under an outer layer or on a non-English inner where a recovery attempt would fail.

    Reports, per block size and block reading (contiguous vs. decimated 3xN-grid), the minimum
    channel entropy over surjective covectors, calibrated against a shuffle null (mean/sd/z). A
    strongly negative ``z`` means a Hill channel is present; ``z ~ 0`` means no linear channel.

    Note it is **alphabet-sensitive**: a Hill over a keyed alphabet is nonlinear in the plain index,
    so pass that ``alphabet`` to see it (a fixed-alphabet run can be negative for a keyed Hill).

    Length note: reliable from ~50 blocks; below that the min-over-covectors multiple-testing floor
    approaches the signal, so a negative result is inconclusive (``reliable`` is then ``False``).
    """
    letters = only_letters(letters)
    # Scope guard: this test is only meaningful on FLAT (polygraphic-looking) text. On natural
    # language the block channels carry ordinary bigram/trigram correlations that read as a false
    # "linear channel" — so refuse English-level IoC and point at the intended entry (a flattened
    # inner). diagnose() only calls this in the flattened-inner branch, where IoC is already low.
    if index_of_coincidence(letters) >= 0.058:
        return {
            "reliable": False,
            "hit": False,
            "channels": [],
            "verdict": (
                f"input IoC {index_of_coincidence(letters):.4f} ~ English — not a flat/polygraphic "
                "text; the linear-channel test only applies to a flattened inner (see "
                "period_inner_content). Skipped."
            ),
        }
    idx_map = {c: i for i, c in enumerate(alphabet)}
    indices = [idx_map[c] for c in letters if c in idx_map]
    rng = random.Random(seed)
    channels: list[dict] = []
    for k in block_sizes:
        nblocks = len(indices) // k
        if nblocks < 20:
            continue
        cov = _projective_surjective_covectors(k)
        for decimated in (False, True):
            obs = _min_channel_entropy(indices, k, cov, decimated)
            nulls = []
            pool = list(indices)
            for _ in range(null_samples):
                rng.shuffle(pool)
                nulls.append(_min_channel_entropy(pool, k, cov, decimated))
            mu = sum(nulls) / len(nulls)
            sd = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** 0.5 or 1e-9
            channels.append(
                {
                    "block": k,
                    "reading": "decimated" if decimated else "contiguous",
                    "blocks": nblocks,
                    "min_entropy": round(obs, 4),
                    "null_mean": round(mu, 4),
                    "null_sd": round(sd, 4),
                    "z": round((obs - mu) / sd, 2),
                }
            )
    if not channels:
        return {"reliable": False, "channels": [], "verdict": "too short for a linear-channel test"}
    channels.sort(key=lambda c: c["z"])
    best = channels[0]
    reliable = best["blocks"] >= 50
    hit = best["z"] <= -4.0
    loc = (
        f"block {best['block']} {best['reading']} (entropy {best['min_entropy']} vs null "
        f"{best['null_mean']}, z={best['z']})"
    )
    if best["z"] <= -8.0:
        verdict = (
            f"STRONG linear channel — a HILL cipher: {loc}. Recover it with `butt crack hill`; if "
            "it sits under a periodic/transposition layer, peel that first. (Alphabet-sensitive — "
            "a keyed Hill needs its alphabet passed in.)"
        )
    elif hit:
        verdict = (
            f"moderate linear channel — likely a HILL: {loc}. At short length a partially-linear "
            "digraphic (Playfair, whose same-row/col cases are shifts) can also reach this, so "
            "confirm with `butt crack hill`."
        )
    elif reliable:
        verdict = (
            f"no linear channel (best z={best['z']}): NOT a Hill of a structured source in this "
            "alphabet. A polygraphic inner here would be NONLINEAR (Playfair/two-/four-square), "
            "or the cipher is not polygraphic. (A keyed-alphabet Hill needs its alphabet passed "
            "in.)"
        )
    else:
        verdict = (
            f"inconclusive: only {best['blocks']} blocks (<50) — the covector multiple-testing "
            "floor approaches the signal at this length, so a null result cannot rule a Hill out."
        )
    return {"reliable": reliable, "hit": hit, "channels": channels, "verdict": verdict}


@functools.cache
def _bounded_surjective_covectors(
    w: int, coeffs: tuple[int, ...], cap: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Deduped surjective, MIXING covectors drawn from a *bounded* coefficient grid.

    For block widths too large to enumerate the whole ``Z26`` covector space
    (:func:`_projective_surjective_covectors` is ``26**w``), the width detector searches a
    bounded-coefficient grid instead. Every kept covector is still SURJECTIVE onto ``Z26``
    (``gcd(entries, 26) == 1`` — guard (a): a non-surjective/zero-divisor functional collapses
    the output range and fakes a near-maximal concentration, so it is excluded) and MIXING
    (>= 2 non-zero coordinates, so it cannot merely pick one ciphertext position), deduped by
    projective class. The grid is enumerated in full when small; otherwise a seeded random
    sample of ``cap`` distinct covectors is drawn (matched-count: the same set is reused for the
    shuffle null).
    """
    seen: set = set()
    out: list = []

    def _ok(v: tuple[int, ...]) -> bool:
        if sum(1 for e in v if e) < 2:
            return False
        g = 0
        for e in v:
            g = math.gcd(g, e % 26)
        return math.gcd(g, 26) == 1

    def _canon(v: tuple[int, ...]) -> tuple[int, ...]:
        return min(tuple((u * (e % 26)) % 26 for e in v) for u in _UNITS26)

    total = len(coeffs) ** w
    if total <= max(4 * cap, 4096):
        for v in itertools.product(coeffs, repeat=w):
            if not _ok(v):
                continue
            c = _canon(v)
            if c in seen:
                continue
            seen.add(c)
            out.append(tuple(e % 26 for e in v))
        return tuple(out)
    rng = random.Random(seed)
    tries = 0
    while len(out) < cap and tries < cap * 80:
        tries += 1
        v = tuple(rng.choice(coeffs) for _ in range(w))
        if not _ok(v):
            continue
        c = _canon(v)
        if c in seen:
            continue
        seen.add(c)
        out.append(tuple(e % 26 for e in v))
    return tuple(out)


def _width_covectors(
    w: int, coeffs: tuple[int, ...], max_functionals: int, seed: int, enum_cap: int
) -> tuple[tuple[int, ...], ...]:
    """Covector set for a width-``w`` linear-channel probe.

    When the full ``Z26`` covector space is small enough (``26**w <= enum_cap``) it is
    enumerated EXHAUSTIVELY (so the Hill's leaked-coordinate covector is *guaranteed* to be in
    the search — a null result then genuinely rules out a width-``w`` Hill). For larger widths
    the space is astronomically bigger than any tractable search, so a bounded-coefficient
    sample is used and the caller must treat a null result as inconclusive (the blind spot).
    """
    if 26**w <= enum_cap:
        return _projective_surjective_covectors(w)
    return _bounded_surjective_covectors(w, coeffs, max_functionals, seed)


def _channel_best_ioc(
    indices: list[int], w: int, cov: tuple[tuple[int, ...], ...]
) -> tuple[float, int]:
    """Max index-of-coincidence of the per-block value stream ``v . block`` over covectors ``v``.

    Returns ``(best_ioc, best_covector_index)``. The per-block value stream of the most
    concentrated surjective functional is the candidate "leaked plaintext coordinate".
    """
    nb = len(indices) // w
    if nb < 2 or not cov:
        return 0.0, 0
    if _np is not None:
        b = _np.asarray(indices[: nb * w], dtype=_np.int64).reshape(nb, w)
        cov_np = _np.asarray(cov, dtype=_np.int64)
        ch = (b @ cov_np.T) % 26
        ncov = cov_np.shape[0]
        counts = _np.zeros((26, ncov), dtype=_np.int64)
        _np.add.at(counts, (ch, _np.broadcast_to(_np.arange(ncov), ch.shape)), 1)
        iocs = (counts * (counts - 1)).sum(0) / (nb * (nb - 1))
        j = int(iocs.argmax())
        return float(iocs[j]), j
    blocks = [indices[b * w : (b + 1) * w] for b in range(nb)]
    denom = nb * (nb - 1)
    best_io, best_j = -1.0, 0
    for j, v in enumerate(cov):
        counts = [0] * 26
        for blk in blocks:
            s = 0
            for k in range(w):
                s += v[k] * blk[k]
            counts[s % 26] += 1
        io = sum(c * (c - 1) for c in counts) / denom
        if io > best_io:
            best_io, best_j = io, j
    return best_io, best_j


def _channel_null_maxes(
    indices: list[int], w: int, cov: tuple[tuple[int, ...], ...], trials: int, seed: int
) -> list[float]:
    """Matched-count null: best-over-covectors IoC on ``trials`` shuffles of the SAME letters.

    Guard (b): the identical search (same covector set, same count) is re-run on each shuffle,
    so the null captures exactly the selection/overfitting inflation of a ``w``-coefficient
    search over few blocks — the only fair reference for the observed maximum.
    """
    nb = len(indices) // w
    m = nb * w
    if nb < 2 or not cov:
        return [0.0]
    if _np is not None:
        cov_np = _np.asarray(cov, dtype=_np.int64)
        ncov = cov_np.shape[0]
        cols = _np.broadcast_to(_np.arange(ncov), (nb, ncov))
        arr = _np.asarray(indices[:m], dtype=_np.int64).copy()
        g = _np.random.default_rng(seed)
        out: list[float] = []
        for _ in range(trials):
            g.shuffle(arr)
            ch = (arr.reshape(nb, w) @ cov_np.T) % 26
            counts = _np.zeros((26, ncov), dtype=_np.int64)
            _np.add.at(counts, (ch, cols), 1)
            iocs = (counts * (counts - 1)).sum(0) / (nb * (nb - 1))
            out.append(float(iocs.max()))
        return out
    rng = random.Random(seed)
    pool = list(indices[:m])
    out = []
    for _ in range(trials):
        rng.shuffle(pool)
        io, _j = _channel_best_ioc(pool, w, cov)
        out.append(io)
    return out


def linear_channel_width(
    text: str,
    *,
    alphabet: str = "KRYPTOS",
    widths: tuple[int, ...] = (2, 3),
    coeffs: tuple[int, ...] = (-2, -1, 0, 1, 2),
    max_functionals: int = 1500,
    null_trials: int = 200,
    z_hit: float = 4.0,
    min_blocks: int = 20,
    enum_cap: int = 40000,
    seed: int = 20250615,
) -> dict:
    """Width-parameterised linear-channel (Hill) detector with two hard-won disciplines.

    For each block width ``w`` this searches surjective linear functionals
    ``a0*c0 + ... + a(w-1)*c(w-1) (mod 26)`` over the ``w`` letters of every block and reports
    the one whose per-block value stream is most CONCENTRATED (max IoC) — the candidate
    "leaked plaintext coordinate" of a width-``w`` Hill (``v . C = (M^T v) . P``, and some
    covector isolates a single plaintext coordinate). Significance is a shuffle-null z AND an
    empirical p AND ``beats_null_max``.

    Two guards, both learned the hard way in a 150-hour effort:

    * (a) SURJECTIVE/unit-guarded functionals only — a covector whose gcd with 26 is not 1 is a
      zero-divisor channel (e.g. ``(13, 13)`` only reaches ``{0, 13}``) that fakes near-maximal
      concentration; only functionals surjective onto ``Z26`` are considered.
    * (b) MATCHED-COUNT shuffle null — ``w`` free coefficients over few blocks overfit badly, so
      a raw "max IoC" is meaningless. The identical search (same covector set/count) is re-run on
      shuffles of the same ciphertext; ``beats_null_max`` (obs strictly above every shuffle) is the
      clean discriminator that a raw z (inflated by a tiny null sd) is not.

    Crucially, this exposes the BLIND SPOT that caused a wrong "nonlinear" verdict: a width-``w'``
    probe cannot see a width-``w`` Hill when ``w'`` misaligns ``w``. Each width also reports
    ``search_exhaustive`` — ``True`` only when the FULL ``Z26`` covector space was enumerated
    (small ``w``), so the leaked coordinate was certainly searched and a null result really rules
    a Hill out. For large ``w`` the covector space (``26**w``) dwarfs any tractable search, so
    ``search_exhaustive`` is ``False`` and a null result there is INCONCLUSIVE — it must NOT be
    read as "nonlinear/no Hill".

    Note it is **alphabet-sensitive** (a Hill over a keyed alphabet is nonlinear in the plain
    index): pass the ``alphabet`` the Hill was built in (default ``"KRYPTOS"``; pass ``"STD"`` for
    a plain A–Z Hill).

    Returns ``{width: {best_functional, ioc, z, p, null_mean, null_max, beats_null_max, blocks,
    functionals, search_exhaustive, hit, reliable, note}}`` plus a top-level ``"verdict"`` and
    ``"best_width"`` (the strongest hitting width, or ``None``).
    """
    alpha = _alphabet(alphabet)
    idx_map = {c: i for i, c in enumerate(alpha)}
    indices = [idx_map[c] for c in only_letters(text) if c in idx_map]
    per_width: dict[int, dict] = {}
    for w in widths:
        if w < 2:
            continue
        nb = len(indices) // w
        exhaustive = 26**w <= enum_cap
        if nb < min_blocks:
            per_width[w] = {
                "width": w,
                "blocks": nb,
                "hit": False,
                "reliable": False,
                "search_exhaustive": exhaustive,
                "ioc": None,
                "z": None,
                "p": None,
                "note": f"too few blocks ({nb} < {min_blocks}) for a width-{w} probe",
            }
            continue
        cov = _width_covectors(w, tuple(coeffs), max_functionals, seed + w, enum_cap)
        obs, j = _channel_best_ioc(indices, w, cov)
        nulls = _channel_null_maxes(indices, w, cov, null_trials, seed + 1000 + w)
        mu = sum(nulls) / len(nulls)
        sd = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** 0.5 or 1e-9
        z = (obs - mu) / sd
        null_max = max(nulls)
        p = (1 + sum(1 for x in nulls if x >= obs)) / (null_trials + 1)
        beats = obs > null_max
        hit = beats and z >= z_hit
        if hit:
            note = (
                f"width-{w} linear channel present (IoC {obs:.4f} vs null max {null_max:.4f}, "
                f"z={z:.1f}) — a HILL of block width {w} (or a multiple)"
            )
        elif exhaustive:
            note = (
                f"no width-{w} linear channel — the FULL Z26 covector space was searched, so this "
                f"genuinely rules out a width-{w} Hill of a structured source in this alphabet"
            )
        else:
            note = (
                f"INCONCLUSIVE at width {w}: the covector space (26^{w}) is far larger than the "
                f"{len(cov)} functionals searched, so the leaked coordinate was likely never tried "
                f"— a null result CANNOT rule out a width-{w} Hill (this is the blind spot that "
                "mis-reads a wide Hill as 'nonlinear')"
            )
        per_width[w] = {
            "width": w,
            "best_functional": list(cov[j]),
            "ioc": round(obs, 4),
            "null_mean": round(mu, 4),
            "null_max": round(null_max, 4),
            "z": round(z, 2),
            "p": round(p, 4),
            "blocks": nb,
            "functionals": len(cov),
            "beats_null_max": beats,
            "search_exhaustive": exhaustive,
            "hit": hit,
            "reliable": nb >= min_blocks,
            "note": note,
        }
    hits = [d for d in per_width.values() if d.get("hit")]
    best_width = max(hits, key=lambda d: d["z"])["width"] if hits else None
    if best_width is not None:
        verdict = (
            f"LINEAR channel at block width {best_width} (z={per_width[best_width]['z']}): a HILL "
            "— recover it with `butt crack hill`. A mismatched-width probe is blind to it."
        )
    else:
        underpowered = [
            w for w, d in per_width.items() if not d.get("search_exhaustive") and d.get("reliable")
        ]
        if underpowered:
            verdict = (
                f"no linear channel at exhaustively-searched widths; widths {underpowered} are "
                f"INCONCLUSIVE (covector space too large to search) — do NOT conclude 'nonlinear' "
                f"from these. Widen the probe or bring a crib."
            )
        else:
            verdict = "no linear channel at any tested width (searched exhaustively where feasible)"
    return {"widths": per_width, "best_width": best_width, "verdict": verdict}


def ioc_decay(
    letters: str,
    *,
    segments: int = 8,
    samples: int = 200,
    seed: int = 20250615,
) -> dict:
    """Detect a *monotonic drift* in index-of-coincidence along the message.

    A periodic polyalphabetic (Vigenere/Quagmire) and a transposition both have a
    **stationary** IoC: every segment of the message reads the same in expectation.
    An **evolving / position-dependent keystream** does not — progressive-key,
    autokey, chain-addition (Gromark) and dynamic alphabets (Chaocipher/Hutton) all
    start more structured and grow toward random, so per-segment IoC *decreases*
    along the text.  This splits the text into equal segments, fits the slope of
    segment-IoC vs position, and z-scores that slope against shuffles of the same
    letter multiset (shuffling destroys positional structure -> flat slope in
    expectation).

    A strongly negative ``slope_z`` (<= -2.5) is the fingerprint of a non-stationary
    keystream: the period is *not* recoverable and periodic/transposition attacks
    will fail — a crib is the lever.  A near-zero ``slope_z`` says the structure is
    stationary, so a flat-IoC/no-period reading points instead to a long key or a
    transposition hiding an inner period.  Returns ``{}`` for inputs too short for
    the segmentation to be meaningful.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < segments * 12:
        return {}
    seg_len = n // segments

    def slope(seq: str) -> float:
        vals = [index_of_coincidence(seq[i * seg_len : (i + 1) * seg_len]) for i in range(segments)]
        xbar = (segments - 1) / 2
        ybar = sum(vals) / segments
        den = sum((i - xbar) ** 2 for i in range(segments))
        num = sum((i - xbar) * (vals[i] - ybar) for i in range(segments))
        return num / den if den else 0.0

    seg_ioc = [
        round(index_of_coincidence(letters[i * seg_len : (i + 1) * seg_len]), 4)
        for i in range(segments)
    ]
    obs_slope = slope(letters)

    rng = random.Random(seed)
    pool = list(letters)
    null = []
    for _ in range(samples):
        rng.shuffle(pool)
        null.append(slope("".join(pool)))
    mu = sum(null) / len(null)
    sd = (sum((v - mu) ** 2 for v in null) / len(null)) ** 0.5 or 1e-9
    slope_z = (obs_slope - mu) / sd

    q = n // 4
    return {
        "segments": segments,
        "segment_ioc": seg_ioc,
        "quarter_ioc": [
            round(index_of_coincidence(letters[i * q : (i + 1) * q]), 4) for i in range(4)
        ],
        "slope": round(obs_slope, 6),
        "slope_z": round(slope_z, 2),
        "monotonic_decreasing": all(seg_ioc[i] >= seg_ioc[i + 1] for i in range(segments - 1)),
        # A drifting (non-stationary) keystream: not periodic, not a transposition.
        "non_stationary": slope_z <= -2.5,
    }


def kappa_spectrum(letters: str, *, max_lag: int = 32) -> list[dict]:
    """Autocorrelation (Kappa) spectrum: coincidence rate of the text against itself
    shifted by each lag, z-scored against random text.

    For each ``lag`` in ``1..max_lag`` this measures ``kappa`` = the fraction of
    positions where ``text[i] == text[i + lag]`` (the index of coincidence of the
    text with a copy of itself slid by ``lag``).  A periodic polyalphabetic keystream
    of period ``p`` re-aligns the same alphabet at every multiple of ``p``, so kappa
    *spikes* at ``lag = p, 2p, 3p, ...`` — Friedman's kappa test, and the lever the
    campaign leaned on to read lag-8/16/17 structure straight off the ciphertext.

    ``z`` compares the observed coincidence count against the random-text expectation
    (``1/26`` per pair) using the binomial standard deviation, so it is comparable
    across lags despite the shrinking overlap.  Returns one row per lag sorted by
    ``z`` descending; the top rows (and their common divisors) are the candidate
    periods.  Empty for inputs too short to overlap.
    """
    letters = only_letters(letters)
    n = len(letters)
    out: list[dict] = []
    p_random = 1.0 / 26.0
    for lag in range(1, max_lag + 1):
        pairs = n - lag
        if pairs < 2:
            break
        hits = sum(1 for i in range(pairs) if letters[i] == letters[i + lag])
        kappa = hits / pairs
        sd = (pairs * p_random * (1 - p_random)) ** 0.5 or 1e-9
        z = (hits - pairs * p_random) / sd
        out.append(
            {
                "lag": lag,
                "kappa": round(kappa, 4),
                "z": round(z, 2),
            }
        )
    out.sort(key=lambda r: r["z"], reverse=True)
    return out


def crackability_cliff(letters: str, period: int) -> dict:
    """Go/no-go for blind recovery of a periodic (or product) keystream of ``period``.

    A periodic keystream is only recoverable blind while it *repeats enough* for each
    of its alphabets to accumulate statistics: with ``cycles = length / period``
    repetitions, recovery is reliable around ``cycles >= 2.5`` (equivalently the
    effective period is under ~a quarter of the length).  Below that the construction
    is one-time-pad-grade — each alphabet sees too few letters to solve from frequency
    alone, and a crib/opening is the only lever.

    Returns ``{effective_period, cycles, recoverable, verdict}``.  ``period`` is the
    effective keystream period (for a product keystream pass the ``lcm`` of the
    component periods).
    """
    letters = only_letters(letters)
    n = len(letters)
    period = max(1, int(period))
    cycles = n / period if period else 0.0
    recoverable = cycles >= 2.5 and period <= n / 4
    if n == 0:
        verdict = "no text"
    elif recoverable:
        verdict = f"recoverable: {round(cycles, 2)} cycles give each alphabet enough letters"
    elif cycles >= 1.5:
        verdict = (
            f"marginal: only {round(cycles, 2)} cycles — blind recovery unreliable, a crib helps"
        )
    else:
        verdict = (
            f"OTP-grade: {round(cycles, 2)} cycles — keystream barely repeats, needs a crib/opening"
        )
    return {
        "effective_period": period,
        "cycles": round(cycles, 2),
        "recoverable": recoverable,
        "verdict": verdict,
    }


def crackability_cliff_auto(letters: str) -> dict:
    """:func:`crackability_cliff` using the best calibrated period as the effective one.

    Picks the most significant period from :func:`calibrated_periods` and reports its
    crackability.  Adds ``period_z`` (how strong the period signal is) and falls back
    to ``effective_period = 0`` with a "no period detected" verdict when no calibrated
    period stands out (a long/non-repeating key or a transposition artifact).
    """
    letters = only_letters(letters)
    n = len(letters)
    periods = calibrated_periods(letters) if n >= 48 else []
    if not periods:
        return {
            "effective_period": 0,
            "cycles": 0.0,
            "recoverable": False,
            "period_z": None,
            "verdict": "no period detected: long/non-repeating key or transposition artifact",
        }
    best = periods[0]
    result = crackability_cliff(letters, int(best["period"]))
    result["period_z"] = best["z"]
    # A weak calibrated spike is selection bias, not a real keystream period: even if
    # the cycle arithmetic looks fine, a period that barely clears the random baseline
    # (z < 3) is not a recoverable signal — flag it down and say so.
    if best["z"] < 3.0:
        result["recoverable"] = False
        result["verdict"] = (
            f"weak period signal (z={best['z']}): no recoverable keystream — "
            "long/non-repeating key or transposition artifact"
        )
    return result


# Reference quarter-IoC decay shapes (normalised: first quarter = 1.0). An evolving
# keystream thins structure toward the tail at a family-specific rate; a stationary
# cipher (periodic substitution or transposition) stays flat. Profiles are coarse on
# purpose — this is a shape *hint*, not a classifier.
_DECAY_PROFILES: dict[str, list[float]] = {
    # progressive key: alphabet advances every position -> steady linear thinning.
    "progressive-key": [1.0, 0.78, 0.56, 0.34],
    # autokey: plaintext feeds the key -> fast initial decay then levels near random.
    "autokey": [1.0, 0.7, 0.52, 0.45],
    # chain/Gromark: additive chain diffuses quickly -> near-random after the head.
    "chain-gromark": [1.0, 0.55, 0.4, 0.35],
    # stationary (periodic substitution / transposition / none): flat in expectation.
    "stationary": [1.0, 1.0, 1.0, 1.0],
}


def decay_fingerprint(letters: str) -> list[dict]:
    """Rank evolving-keystream families by how well the quarter-IoC decay *shape* fits.

    Beyond :func:`ioc_decay`'s boolean ``non_stationary`` flag, this compares the
    message's normalised quarter-IoC curve (first quarter scaled to 1.0) against a few
    hardcoded reference profiles — progressive-key, autokey, chain/Gromark and the
    flat *stationary* profile — and returns them ranked by Euclidean ``curve_distance``
    (smaller = better fit).

    Crucially, most evolving ciphers are IoC-*stationary* in expectation, so a small
    distance to ``stationary`` is the common, valid verdict: it means "no evolving
    family matches — periodic substitution, transposition, or random."  The returned
    ``verdict`` says so explicitly when ``stationary`` wins or the curve is too flat to
    discriminate.  Lightweight: it reuses the quarter IoCs, no Monte-Carlo.  Empty for
    inputs too short to quarter meaningfully.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 96:
        return []
    q = n // 4
    quarter_ioc = [index_of_coincidence(letters[i * q : (i + 1) * q]) for i in range(4)]
    head = quarter_ioc[0]
    if head <= 0:
        return []
    norm = [v / head for v in quarter_ioc]
    ranked: list[dict] = []
    for family, profile in _DECAY_PROFILES.items():
        dist = (sum((norm[i] - profile[i]) ** 2 for i in range(4))) ** 0.5
        ranked.append({"family": family, "curve_distance": round(dist, 4)})
    ranked.sort(key=lambda r: r["curve_distance"])
    best = ranked[0]
    # How far the curve actually drops — if it's nearly flat, the shape can't tell the
    # evolving families apart and stationary/transposition is the honest read.
    drop = norm[0] - min(norm)
    if best["family"] == "stationary" or drop < 0.15:
        verdict = (
            "no evolving family matches: likely periodic substitution or transposition artifact"
        )
    else:
        verdict = f"best shape match: {best['family']} (distance {best['curve_distance']})"
    for row in ranked:
        row["verdict"] = verdict
    return ranked


def search_aware_null(
    letters: str,
    search: Callable[[str], float],
    *,
    samples: int = 40,
    seed: int = 20250615,
    unit: int = 1,
) -> dict:
    """Calibrate a *best-of-search* statistic against shuffles of the same letters.

    When an attack searches a large space (every column order, every keyword) and
    keeps the BEST score, that maximum is inflated by selection bias — structureless
    text scores surprisingly high simply because many candidates were tried.  So the
    honest null is not "one random text" but "the same search run on shuffled text."
    ``search`` takes a letter string and returns the best score its search achieves;
    this runs it on the real text and on ``samples`` shuffles (same multiset).

    Returns ``{observed, null_mean, null_max, z, beats_null_max}``.  Treat the result
    as signal only when ``beats_null_max`` (or ``z`` is large): a high raw score that
    sits inside the shuffled-search band is overfit, not structure.  This is the guard
    that distinguishes a real transposition/period from a fluke maximum — the failure
    mode that wastes the most effort on layered ciphers.

    ``unit`` sets the shuffle granularity: with ``unit>1`` the null shuffles ``unit``-letter
    BLOCKS as indivisible tokens (matching a block/unit transposition search), preserving the
    intra-block trigraph structure the real cipher preserves. Shuffling single letters under a
    block construction destroys that structure and makes the null too easy to beat — so a
    unit-``g`` reveal search MUST be calibrated with ``unit=g`` or ``beats_null_max`` inflates.
    """
    base = only_letters(letters)
    observed = search(base)
    rng = random.Random(seed)
    if unit > 1:
        pool = [base[i : i + unit] for i in range(0, len(base) - len(base) % unit, unit)]
    else:
        pool = list(base)
    null = []
    for _ in range(samples):
        rng.shuffle(pool)
        null.append(search("".join(pool)))
    mu = sum(null) / len(null)
    sd = (sum((v - mu) ** 2 for v in null) / len(null)) ** 0.5 or 1e-9
    null_max = max(null)
    return {
        "observed": round(observed, 4),
        "null_mean": round(mu, 4),
        "null_max": round(null_max, 4),
        "z": round((observed - mu) / sd, 2),
        "beats_null_max": observed > null_max,
    }


def _kappa_at(lett: list[int], lag: int) -> float:
    """Repeat rate at ``lag`` (x26; random ~1.0)."""
    n = len(lett)
    if n <= lag:
        return 0.0
    hits = sum(1 for i in range(n - lag) if lett[i] == lett[i + lag])
    return hits / (n - lag) * 26.0


#: named per-period statistics for :func:`period_family_significance`
_PERIOD_STAT_FNS = {
    "coset_ioc": _mean_col_ioc,
    "merged_ioc": _merge_ioc,
    "kappa": _kappa_at,
}


def period_family_significance(
    letters: str,
    *,
    statistic: str = "coset_ioc",
    max_period: int | None = None,
    min_col: int = 4,
    samples: int = 200,
    seed: int = 20250615,
) -> dict:
    """Look-elsewhere (family-wide) significance of the single strongest period in a scan.

    A per-period z (:func:`calibrated_periods`, :func:`kappa_spectrum`) is calibrated against
    *that one period's* baseline — but the scan keeps the best of many periods, and that maximum
    is inflated by selection bias exactly like any best-of-search statistic (see
    :func:`search_aware_null`). A per-period ``z`` of +3 across a 15- or 50-period grid is often
    multiplicity noise, not structure. This reports the honest family-wide significance: the best
    period's statistic vs the distribution of the *max over the same period grid* on shuffles of
    the same letters.

    ``statistic`` is ``"coset_ioc"`` (mean per-column IoC), ``"merged_ioc"`` (greedy
    superimposition), or ``"kappa"`` (repeat rate at the lag). Returns ``{best_period, observed,
    null_mean, null_max, z, family_p, beats_null_max, grid, samples}`` where ``family_p`` is the
    add-one-smoothed fraction of shuffles whose grid-max reaches ``observed`` (the corrected
    p-value). Treat the period as real only when it clears the family null (``beats_null_max`` /
    small ``family_p``), NOT merely a high per-period z.
    """
    stat = _PERIOD_STAT_FNS.get(statistic)
    if stat is None:
        raise ValueError(f"unknown statistic {statistic!r}; use coset_ioc/merged_ioc/kappa")
    lett = [ord(ch) - 65 for ch in only_letters(letters)]
    n = len(lett)
    if max_period is None:
        max_period = min(n // min_col, 50)
    grid = list(range(2, max_period + 1))
    if n < 12 or not grid:
        return {
            "statistic": statistic,
            "best_period": None,
            "observed": 0.0,
            "null_mean": 0.0,
            "null_max": 0.0,
            "z": 0.0,
            "family_p": 1.0,
            "beats_null_max": False,
            "grid": None,
            "samples": 0,
        }
    # Per-period statistics are NOT comparable across periods (raw coset-IoC rises as the
    # columns shrink), so calibrate each period against its OWN shuffle baseline first, then
    # family-correct the MAX calibrated z across the grid.
    obs_raw = [stat(lett, p) for p in grid]
    rng = random.Random(seed)
    pool = lett[:]
    shuffles = []
    for _ in range(samples):
        rng.shuffle(pool)
        shuffles.append([stat(pool, p) for p in grid])
    cols = list(zip(*shuffles, strict=True))  # one column of shuffle stats per period
    mus = [sum(c) / samples for c in cols]
    sds = [
        (sum((v - m) ** 2 for v in c) / samples) ** 0.5 or 1e-9
        for c, m in zip(cols, mus, strict=True)
    ]

    z_obs = [(obs_raw[i] - mus[i]) / sds[i] for i in range(len(grid))]
    best_i = max(range(len(grid)), key=lambda i: z_obs[i])
    observed_z = z_obs[best_i]
    null_maxz = [max((row[i] - mus[i]) / sds[i] for i in range(len(grid))) for row in shuffles]
    grid_maxz = max(null_maxz)
    ge = sum(1 for v in null_maxz if v >= observed_z)
    return {
        "statistic": statistic,
        "best_period": grid[best_i],
        "observed": round(obs_raw[best_i], 4),
        "z": round(observed_z, 2),
        "family_null_mean_z": round(sum(null_maxz) / samples, 2),
        "family_null_max_z": round(grid_maxz, 2),
        "family_p": round((ge + 1) / (samples + 1), 4),
        "beats_null_max": observed_z > grid_maxz,
        "grid": [grid[0], grid[-1]],
        "samples": samples,
    }


def _digraph_ioc(letters: str) -> float:
    """Index of coincidence over adjacent letter pairs (random ~1/676 = 0.00148)."""
    n = len(letters) - 1
    if n < 2:
        return 0.0
    counts = Counter(letters[i : i + 2] for i in range(n))
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def repeat_adjusted_stats(
    text: str, *, ngram: int = 3, min_count: int = 3, samples: int = 200, seed: int = 20250615
) -> dict:
    """Is the digraph structure *real*, or just a handful of exact repeats?

    A few exactly-repeated n-grams (a deterministic block cipher's identical blocks, a
    repeated plaintext word under transposition) inflate the digraph IoC and can be
    mistaken for diffuse pair-structure. This excises the *redundant* occurrences of every
    n-gram seen ``>= min_count`` times (keeping the first of each) and re-measures: if the
    digraph elevation **survives** excision it is real structure; if it **collapses to the
    random floor** it was purely the repeats. (A block cipher with a couple of identical blocks
    can show a digraph ratio ~9x -> ~1x random once its 3x-trigrams are removed — i.e. featureless
    apart from the repeats.)

    Returns full vs excised digraph IoC, the same as a ratio to the random floor, the
    repeated n-grams found, and a shuffle-null z for the excised digraph.
    """
    letters = only_letters(text)
    occ: dict[str, list[int]] = {}
    for i in range(len(letters) - ngram + 1):
        occ.setdefault(letters[i : i + ngram], []).append(i)
    repeats = {g: pos for g, pos in occ.items() if len(pos) >= min_count}

    drop: set[int] = set()
    for positions in repeats.values():
        for p in positions[1:]:  # keep the first occurrence; excise the rest
            drop.update(range(p, p + ngram))
    excised = "".join(ch for i, ch in enumerate(letters) if i not in drop)

    floor = 1.0 / (26 * 26)
    full_dig = _digraph_ioc(letters)
    exc_dig = _digraph_ioc(excised)

    rng = random.Random(seed)
    pool = list(excised)
    null = []
    for _ in range(samples):
        rng.shuffle(pool)
        null.append(_digraph_ioc("".join(pool)))
    mu = sum(null) / len(null) if null else floor
    sd = (sum((v - mu) ** 2 for v in null) / len(null)) ** 0.5 if null else 0.0
    z = (exc_dig - mu) / sd if sd else 0.0

    survives = z >= 2.5
    return {
        "ngram": ngram,
        "repeated_ngrams": [
            {"gram": g, "count": len(p)}
            for g, p in sorted(repeats.items(), key=lambda kv: -len(kv[1]))
        ],
        "digraph_ioc_full": round(full_dig, 6),
        "digraph_ioc_excised": round(exc_dig, 6),
        "digraph_ratio_full": round(full_dig / floor, 2),
        "digraph_ratio_excised": round(exc_dig / floor, 2),
        "excised_z_vs_shuffle": round(z, 2),
        "verdict": (
            "diffuse digraph structure is real (survives excision)"
            if survives
            else "digraph elevation was the repeats only (collapses to random when excised)"
        ),
    }


def _fingerprint(letters: str) -> dict:
    """Compact statistical fingerprint used for family/sibling comparison."""
    n = len(letters)
    ioc = index_of_coincidence(letters) if n >= 2 else 0.0
    return {
        "length": n,
        "ioc": round(ioc, 4),
        "digraph_ratio": round(_digraph_ioc(letters) / (1.0 / 676), 2) if n >= 3 else 0.0,
        "chi2_per_letter": round(chi_squared(letters) / n, 4) if n else 0.0,
    }


def family_baseline(target: str, corpus, *, labels=None) -> dict:
    """Is ``target``'s fingerprint *normal for this family*, or anomalous?

    The single most useful re-grounding when a ciphertext sits in a series of solved
    siblings: compute each sibling's IoC fingerprint and ask whether the target's flatness
    is ordinary-for-the-family (so a flat IoC is NOT evidence of an exotic construction) or
    a genuine outlier. ``corpus`` is an iterable of sibling *ciphertexts* (or a mapping
    ``label -> ciphertext``). Reports the target fingerprint, the family IoC band, an
    in-band verdict, and the closest sibling by feature distance.

    (Worked example: a target whose IoC 0.039 reads "anomalously flat" only against English
    0.066 — against its own family of sibling ciphertexts (≈0.038–0.039) it is identical, so it
    is **family-normal**, and the flatness is *not* evidence of an exotic construction.)
    """
    if isinstance(corpus, dict):
        items = list(corpus.items())
    else:
        corpus = list(corpus)
        labs = labels if labels is not None else [f"#{i}" for i in range(len(corpus))]
        items = list(zip(labs, corpus, strict=False))
    sib = [(lab, _fingerprint(only_letters(c))) for lab, c in items]
    tgt = _fingerprint(only_letters(target))

    iocs = [f["ioc"] for _, f in sib]
    lo, hi = (min(iocs), max(iocs)) if iocs else (0.0, 0.0)
    in_band = lo - 1e-9 <= tgt["ioc"] <= hi + 1e-9

    # closest sibling by normalised feature distance (ioc dominates; scaled to ~unit)
    def dist(f):
        return ((tgt["ioc"] - f["ioc"]) / 0.01) ** 2 + (
            (tgt["digraph_ratio"] - f["digraph_ratio"]) / 1.0
        ) ** 2

    closest = min(sib, key=lambda lf: dist(lf[1])) if sib else (None, None)

    return {
        "target": tgt,
        "family_ioc_band": [round(lo, 4), round(hi, 4)],
        "family_ioc_mean": round(sum(iocs) / len(iocs), 4) if iocs else None,
        "siblings": [{"label": lab, **f} for lab, f in sib],
        "ioc_family_normal": in_band,
        "closest_sibling": closest[0],
        "verdict": (
            f"IoC {tgt['ioc']} is FAMILY-NORMAL (band {round(lo, 4)}-{round(hi, 4)}); "
            "flatness is not evidence of an exotic construction"
            if in_band
            else (
                f"IoC {tgt['ioc']} is OUTSIDE the family band {round(lo, 4)}-{round(hi, 4)}: "
                "genuinely anomalous"
            )
        ),
    }


# =========================================================================== #
# Extended statistical fingerprints
#
# Period detectors, cross-text tests, and distributional discriminants that
# complement the IoC / kappa / Kasiski core above. All pure and read-only; where
# a period/relationship is claimed it is z-scored against a shuffle (or analytic)
# null of the same letters, so a short-message artifact can't masquerade as signal.
# =========================================================================== #

_LOG2_26 = math.log2(26)


def _indices(letters: str, alphabet: str = "STANDARD") -> list[int]:
    ring = _alphabet(alphabet)
    pos = {c: i for i, c in enumerate(ring)}
    return [pos[c] for c in letters if c in pos]


def _column_twist(col: list[int]) -> float:
    """The 'twist' of one column: (upper-13 minus lower-13) sorted frequency mass, in percent.

    A peaked (single-alphabet) column concentrates its mass so upper-lower is large; a flat
    column gives ~0. Barr & Simoson's twist index.
    """
    m = len(col)
    if m == 0:
        return 0.0
    counts = [0] * 26
    for x in col:
        counts[x] += 1
    pct = sorted(c / m for c in counts)
    return (sum(pct[13:]) - sum(pct[:13])) * 100.0


def twist_periods(
    letters, *, alphabet="STANDARD", max_period=None, samples=80, seed=20250615, top=6
):
    """Key-length candidates by the Twist+ test — robust at short lengths.

    For each period the mean per-column twist is z-scored against the same statistic on shuffles
    of the same letters (the '+' correction that removes short-column bias, the failure mode that
    makes a bare coset-IoC spike at a spurious small period). The true key length peaks. More
    reliable than IoC/Kasiski when columns are short (n ~ 150). Up to ``top`` periods, by z.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 20:
        return []
    idx = _indices(letters, alphabet)
    if max_period is None:
        max_period = min(n // 4, 30)
    rng = random.Random(seed)
    pool = [idx[:] for _ in range(samples)]
    for s in pool:
        rng.shuffle(s)
    out = []
    for p in range(2, max_period + 1):
        obs = sum(_column_twist(idx[j::p]) for j in range(p)) / p
        base = [sum(_column_twist(s[j::p]) for j in range(p)) / p for s in pool]
        mu = sum(base) / len(base)
        sd = (sum((b - mu) ** 2 for b in base) / len(base)) ** 0.5 or 1e-9
        out.append(
            {
                "period": p,
                "twist": round(obs, 2),
                "baseline": round(mu, 2),
                "z": round((obs - mu) / sd, 2),
            }
        )
    out.sort(key=lambda r: r["z"], reverse=True)
    return out[:top]


def spectral_periods(letters, *, max_period=None, max_lag=None, samples=60, seed=20250615, top=6):
    """Periods from the coincidence-autocorrelation comb — an independent lens from IoC/Kasiski.

    Builds kappa(lag) = coincidence rate at each lag; a period-p cipher makes it spike at *every*
    multiple of p. The comb score for a candidate p is the mean kappa over lags p, 2p, 3p, ...,
    z-scored against shuffles. Because it samples the autocorrelation at integer lags it avoids
    the harmonic/leakage aliasing an FFT periodogram suffers (a sub-period like 3 for a true 6
    mixes in the un-elevated lags 3, 9, ... and is suppressed). Up to ``top`` periods, by z.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 24:
        return []
    if max_lag is None:
        max_lag = min(n // 2, 96)
    if max_period is None:
        max_period = min(n // 4, 30)
    seq = [ord(c) - 65 for c in letters]

    def comb(s, p):
        lags = range(p, max_lag + 1, p)
        vals = [sum(1 for i in range(n - lag) if s[i] == s[i + lag]) / (n - lag) for lag in lags]
        return sum(vals) / len(vals) if vals else 0.0

    rng = random.Random(seed)
    pool = [seq[:] for _ in range(samples)]
    for s in pool:
        rng.shuffle(s)
    out = []
    for p in range(2, max_period + 1):
        obs = comb(seq, p)
        base = [comb(s, p) for s in pool]
        mu = sum(base) / len(base)
        sd = (sum((b - mu) ** 2 for b in base) / len(base)) ** 0.5 or 1e-9
        out.append(
            {
                "period": p,
                "comb_kappa": round(obs, 4),
                "baseline": round(mu, 4),
                "z": round((obs - mu) / sd, 2),
            }
        )
    out.sort(key=lambda r: r["z"], reverse=True)
    return out[:top]


def mutual_index_of_coincidence(a, b):
    """Position-independent mutual IoC: P(a random letter of ``a`` == a random letter of ``b``).

    ~0.066 for two same-language texts (or two ciphertexts under the SAME monoalphabetic key),
    ~0.0385 for unrelated/independent alphabets. Language- and position-agnostic.
    """
    a = only_letters(a)
    b = only_letters(b)
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    return sum(ca[c] * cb.get(c, 0) for c in ca) / (len(a) * len(b))


def mutual_kappa_scan(a, b, *, max_shift=None, min_overlap=20, top=6):
    """Slide ``b`` against ``a`` and z-score the coincidence rate (kappa) at each offset.

    Detects 'depth' or a derivative relationship: two messages sharing a keystream/key align at
    one offset where kappa jumps to the language coincidence (~0.066) far above the position-
    independent expectation. The cross-text analogue of the single-message period tests — the
    natural probe for a cross-referencing series (is one puzzle keyed by another?). Uses the
    analytic null E[kappa]=MIC(a,b), Var=MIC(1-MIC)/overlap. Offsets sorted by z (``shift`` is
    b's start position relative to a; may be negative).
    """
    a = only_letters(a)
    b = only_letters(b)
    na, nb = len(a), len(b)
    if na < min_overlap or nb < min_overlap:
        return []
    ai = [ord(c) - 65 for c in a]
    bi = [ord(c) - 65 for c in b]
    if max_shift is None:
        max_shift = max(na, nb) - min_overlap
    mic = mutual_index_of_coincidence(a, b) or 1e-9
    out = []
    for shift in range(-max_shift, max_shift + 1):
        lo, hi = max(0, shift), min(na, nb + shift)
        overlap = hi - lo
        if overlap < min_overlap:
            continue
        kappa = sum(1 for i in range(lo, hi) if ai[i] == bi[i - shift]) / overlap
        sd = (mic * (1 - mic) / overlap) ** 0.5 or 1e-9
        out.append(
            {
                "shift": shift,
                "overlap": overlap,
                "kappa": round(kappa, 4),
                "z": round((kappa - mic) / sd, 2),
            }
        )
    out.sort(key=lambda r: r["z"], reverse=True)
    return out[:top]


def trigraphic_ioc(letters, *, step=1):
    """IoC over trigrams (``step=1`` overlapping, ``step=3`` boundary-aligned).

    The trigram analogue of the digraphic IoC: elevated for trigraphic ciphers / trifid and for
    structured (English/transposition) text, near-floor for a strong flattener. Complements
    DIC/EDI for the fractionation family.
    """
    letters = only_letters(letters)
    grams = [letters[i : i + 3] for i in range(0, len(letters) - 2, step)]
    m = len(grams)
    if m < 2:
        return 0.0
    counts = Counter(grams)
    return sum(c * (c - 1) for c in counts.values()) / (m * (m - 1))


def conditional_entropy(letters):
    """Monogram / bigram / conditional entropy (bits) and redundancy — language-agnostic.

    Separates classes without any language table (matters for non-English payloads):
    plaintext/monoalphabetic -> low monogram AND low conditional; transposition -> low monogram,
    HIGH conditional (adjacency destroyed); polyalphabetic/fractionation -> both high.
    ``redundancy`` = 1 - monogram_entropy / log2(26).
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 3:
        return {
            "monogram_entropy": None,
            "bigram_entropy": None,
            "conditional_entropy": None,
            "redundancy": None,
        }
    h1 = letter_entropy(letters)
    nb = n - 1
    bi = Counter(letters[i : i + 2] for i in range(nb))
    first = Counter(letters[i] for i in range(nb))
    hjoint = -sum((c / nb) * math.log2(c / nb) for c in bi.values())
    hfirst = -sum((c / nb) * math.log2(c / nb) for c in first.values())
    return {
        "monogram_entropy": round(h1, 4),
        "bigram_entropy": round(hjoint, 4),
        "conditional_entropy": round(hjoint - hfirst, 4),
        "redundancy": round(1 - h1 / _LOG2_26, 4),
    }


def hamming_periods(letters, *, max_period=None, samples=60, seed=20250615, top=6):
    """Key-length candidates by minimal normalized Hamming distance between blocks.

    For each ``L`` the message is cut into length-``L`` blocks and the mean fraction of DIFFERING
    positions between consecutive blocks is measured; at the true key length corresponding
    positions share a Caesar alphabet, so they coincide more and the distance DIPS. Reported as a
    positive z of the dip vs a shuffle null — an independent vote alongside Kasiski/coset-IoC.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 40:
        return []
    seq = [ord(c) - 65 for c in letters]
    if max_period is None:
        max_period = min(n // 4, 30)

    def mean_norm_hamming(s, L, max_blocks=30):
        blocks = [s[i : i + L] for i in range(0, n - L + 1, L)][:max_blocks]
        if len(blocks) < 2:
            return None
        tot, cnt = 0.0, 0
        for x in range(len(blocks)):  # all pairs — far less noisy than consecutive-only
            for y in range(x + 1, len(blocks)):
                b1, b2 = blocks[x], blocks[y]
                w = min(len(b1), len(b2))
                tot += sum(1 for i in range(w) if b1[i] != b2[i]) / w
                cnt += 1
        return tot / cnt if cnt else None

    rng = random.Random(seed)
    pool = [seq[:] for _ in range(samples)]
    for s in pool:
        rng.shuffle(s)
    out = []
    for L in range(2, max_period + 1):
        obs = mean_norm_hamming(seq, L)
        if obs is None:
            continue
        base = [v for v in (mean_norm_hamming(s, L) for s in pool) if v is not None]
        mu = sum(base) / len(base)
        sd = (sum((b - mu) ** 2 for b in base) / len(base)) ** 0.5 or 1e-9
        out.append(
            {
                "period": L,
                "norm_hamming": round(obs, 4),
                "baseline": round(mu, 4),
                "z": round((mu - obs) / sd, 2),
            }
        )  # a dip below the null -> positive z
    out.sort(key=lambda r: r["z"], reverse=True)
    return out[:top]


def sukhotin_vowels(letters):
    """Sukhotin's algorithm: classify letters as vowels from adjacency alone (language-blind).

    Works on plaintext, transposition, or a monoalphabetic substitution (adjacency survives
    relabeling), so the recovered vowel set + count is a quick 'reads like a natural-language
    layer' check and a language hint. Meaningless on a flattener.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 20:
        return {"vowels": "", "count": 0}
    present = sorted(set(letters))
    pos = {c: i for i, c in enumerate(present)}
    k = len(present)
    adj = [[0] * k for _ in range(k)]
    for i in range(n - 1):
        x, y = pos[letters[i]], pos[letters[i + 1]]
        if x != y:
            adj[x][y] += 1
            adj[y][x] += 1
    rho = [sum(row) for row in adj]
    vowels: list[int] = []
    remaining = set(range(k))
    while remaining:
        v = max(remaining, key=lambda i: rho[i])
        if rho[v] <= 0:
            break
        vowels.append(v)
        remaining.discard(v)
        for j in remaining:
            rho[j] -= 2 * adj[v][j]
    return {"vowels": "".join(sorted(present[v] for v in vowels)), "count": len(vowels)}


def _profile_r(counts_sorted_desc: list[float], ref: list[float]) -> float:
    mo, mr = sum(counts_sorted_desc) / 26, sum(ref) / 26
    cov = sum((o - mo) * (r - mr) for o, r in zip(counts_sorted_desc, ref, strict=False))
    so = sum((o - mo) ** 2 for o in counts_sorted_desc) ** 0.5
    sr = sum((r - mr) ** 2 for r in ref) ** 0.5
    return cov / (so * sr) if so and sr else 0.0


def frequency_profile_match(letters, *, samples=200, seed=20250615):
    """Permutation-invariant monoalphabetic-ness: sorted-frequency-profile fit vs English, z-scored.

    A monoalphabetic cipher preserves the *shape* of the frequency curve (a few high, many low),
    so the sorted profile still matches English even though letters are permuted; a
    polyalphabetic/flattened text has a flat profile. The **raw** Pearson r is fooled at short n
    (sampling noise makes even random text look peaked), so it is z-scored against random text of
    the same length: ``z`` >> 0 means monoalphabetic-shaped, ``z`` ~ 0 means flattened.
    """
    letters = only_letters(letters)
    n = len(letters)
    if n < 26:
        return {"r": None, "z": None}
    ref = sorted(ENGLISH_MONOGRAM_FREQ.values(), reverse=True)
    counts = Counter(letters)
    obs_r = _profile_r(
        sorted((counts.get(c, 0) / n for c in ENGLISH_MONOGRAM_FREQ), reverse=True), ref
    )
    rng = random.Random(seed)
    null = []
    for _ in range(samples):
        rc = [0] * 26
        for _ in range(n):
            rc[rng.randrange(26)] += 1
        null.append(_profile_r(sorted((c / n for c in rc), reverse=True), ref))
    mu = sum(null) / len(null)
    sd = (sum((x - mu) ** 2 for x in null) / len(null)) ** 0.5 or 1e-9
    return {"r": round(obs_r, 4), "z": round((obs_r - mu) / sd, 2)}


def serial_correlation(letters):
    """Lag-1 serial correlation of the letter-index sequence (a quick randomness probe)."""
    letters = only_letters(letters)
    n = len(letters)
    if n < 3:
        return None
    x = [ord(c) - 65 for c in letters]
    mean = sum(x) / n
    num = sum((x[i] - mean) * (x[i + 1] - mean) for i in range(n - 1))
    den = sum((xi - mean) ** 2 for xi in x)
    return round(num / den, 4) if den else 0.0


def runs_test(letters):
    """Wald-Wolfowitz runs test on letter indices above/below the median (randomness z)."""
    letters = only_letters(letters)
    n = len(letters)
    if n < 20:
        return {"runs": None, "z": None}
    seq = [ord(c) - 65 for c in letters]
    median = sorted(seq)[n // 2]
    signs = [1 if v > median else 0 for v in seq if v != median]
    m = len(signs)
    n1 = sum(signs)
    n0 = m - n1
    if n1 == 0 or n0 == 0:
        return {"runs": 0, "z": 0.0}
    runs = 1 + sum(1 for i in range(1, m) if signs[i] != signs[i - 1])
    mu = 1 + 2 * n1 * n0 / m
    var = (2 * n1 * n0 * (2 * n1 * n0 - m)) / (m * m * (m - 1)) if m > 1 else 0.0
    return {"runs": runs, "z": round((runs - mu) / var**0.5, 2) if var > 0 else 0.0}


def friedman_period_estimate(letters, *, kappa_p=0.0667, kappa_r=1 / 26):
    """Friedman's closed-form key-length estimate ``(kp - kr) / (IC - kr)`` from the observed IoC.

    A coarse scalar (the integer key length is near this value); pair it with the calibrated
    period scans for a real answer. ``None`` when the IoC is at/below the random floor.
    """
    letters = only_letters(letters)
    if len(letters) < 20:
        return None
    denom = index_of_coincidence(letters) - kappa_r
    return round((kappa_p - kappa_r) / denom, 2) if denom > 1e-6 else None


def analyze(text: str, *, top_ngrams: int = 10, with_contacts: bool = False) -> dict:
    """Full statistical report for ``text`` (letters only)."""
    letters = only_letters(text)
    n = len(letters)
    counts = Counter(letters)

    # Order letters by count first, then build the rows (keeps the sort off the
    # mixed-type dicts).
    by_count = sorted((chr(65 + i) for i in range(26)), key=lambda ch: -counts.get(ch, 0))
    frequencies = [
        {
            "letter": ch,
            "count": counts.get(ch, 0),
            "percent": round(100 * counts.get(ch, 0) / n, 2) if n else 0.0,
            "english_percent": round(100 * ENGLISH_MONOGRAM_FREQ[ch], 2),
        }
        for ch in by_count
    ]

    repeats, likely_periods = kasiski(letters) if n >= 6 else ([], [])

    # Calibrated per-column IoC catches long periods (short columns) that Kasiski
    # and naive IoC miss — but it's only meaningful for *polyalphabetic* text. Skip
    # the Monte-Carlo when the overall IoC already reads monoalphabetic/transposition
    # (~0.066), where "period" isn't the question and the baseline would just be noise.
    ioc = index_of_coincidence(letters) if n >= 2 else 0.0
    polyalpha = n >= 48 and ioc < 0.058
    periodic = calibrated_periods(letters) if polyalpha else []
    # Kerckhoffs superimposition: of the periods present, which behave as a per-column
    # additive *shift* (columns realign to one pile) rather than a block/flattener. Same
    # polyalphabetic gate as the calibrated per-column IoC.
    superimposition = superimposition_periods(letters) if polyalpha else []
    # Twist+ (short-length-robust) and coincidence-comb period detectors — independent votes.
    twist = twist_periods(letters) if polyalpha else []
    spectral = spectral_periods(letters) if polyalpha else []
    friedman_len = friedman_period_estimate(letters) if polyalpha else None
    # Repeating-bigram spikes flag a periodic transposition layer (Zodiac-style),
    # which can sit under a substitution; chi^2/letter low (~<0.05) means letter
    # frequencies still match English => order scrambled => a transposition is present.
    chi2_per_letter = round(chi_squared(letters) / n, 4) if n else None
    transposition = transposition_periods(letters) if n >= 80 else []
    # Language-agnostic distributional discriminants (all cheap scalars).
    cond_entropy = conditional_entropy(letters) if n >= 24 else {}
    trigraph_ioc = round(trigraphic_ioc(letters), 5) if n >= 12 else None
    freq_profile = frequency_profile_match(letters) if n >= 48 else {}
    vowels = sukhotin_vowels(letters) if n >= 40 else {}
    # IoC drift along the message: a non-stationary keystream (progressive/autokey/
    # chain/dynamic) decays where a periodic or transposition cipher stays flat.
    decay = ioc_decay(letters) if n >= 96 and ioc < 0.058 else {}
    # Friedman kappa autocorrelation: top lags expose periodic re-alignment (lag-p
    # spikes) straight off the ciphertext; surface only the strongest few.
    kappa = kappa_spectrum(letters)[:6] if n >= 16 else []
    # OTP-grade go/no-go on the best calibrated period: does the keystream repeat
    # enough (>= ~2.5 cycles) to be blind-recoverable?
    cliff = crackability_cliff_auto(letters) if n >= 48 else {}
    # Quarter-IoC decay *shape* vs evolving-keystream reference profiles (stationary
    # winning is the common, valid "no evolving family / transposition" verdict).
    fingerprint = decay_fingerprint(letters) if n >= 96 and ioc < 0.058 else []
    # Distinguish genuine diffuse digraph structure from a few exact repeats inflating it
    # (deterministic block ciphers / repeated plaintext under transposition). Only run when
    # some trigram actually recurs >=3x, else there's nothing to excise.
    rep3 = (
        any(c >= 3 for c in Counter(letters[i : i + 3] for i in range(n - 2)).values())
        if n >= 12
        else False
    )
    repeat_adjusted = repeat_adjusted_stats(letters) if rep3 else {}

    # Block-of-b transposition / b-graph block cipher fingerprint (repeated n-grams that
    # all start at == 0 mod b). Cheap; only informative once there are a few repeats.
    block_signal = block_transposition_signal(letters) if n >= 12 else {}

    report = {
        "length": n,
        "index_of_coincidence": round(ioc, 4) if n >= 2 else None,
        "chi_squared": round(chi_squared(letters), 2) if n else None,
        "chi_squared_per_letter": chi2_per_letter,
        "frequencies": frequencies,
        "bigrams": _ngram_counts(letters, 2, top_ngrams),
        "trigrams": _ngram_counts(letters, 3, top_ngrams),
        "kasiski_repeats": repeats,
        "likely_periods": likely_periods,
        "periodic_ioc": periodic,
        "superimposition_periods": superimposition,
        "twist_periods": twist,
        "spectral_periods": spectral,
        "friedman_period_estimate": friedman_len,
        "transposition_periods": transposition,
        "conditional_entropy": cond_entropy,
        "trigraphic_ioc": trigraph_ioc,
        "frequency_profile_match": freq_profile,
        "sukhotin_vowels": vowels,
        "ioc_decay": decay,
        "kappa_spectrum": kappa,
        "crackability_cliff": cliff,
        "decay_fingerprint": fingerprint,
        "repeat_adjusted": repeat_adjusted,
        "block_transposition": block_signal,
    }
    if with_contacts:
        report["contacts"] = contacts(text)
    return report
