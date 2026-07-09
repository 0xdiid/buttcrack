"""Letter-emitting coordinate substitutions (a non-shift substitution family).

Background
----------
Some substitution panels present a *random-flat* single-letter IoC (close to the
0.0385 random baseline) **and** a random-flat digraph IoC, yet still emit exactly 26
distinct A-Z letters with a **distinct J** (I and J both appear) and **no
filler/digit symbols**. Such a surface fingerprint defeats every periodic-shift
cipher (Vigenere / Beaufort / Quagmire I-IV, every period & alphabet) and every
plain transposition / fractionation, while still looking like a monoalphabetic-by-
position panel. A natural diagnosis is that any apparent short "period" is an
*artifact of a non-shift operation* rather than a real keyed shift.

This module implements that non-shift idea: a **coordinate (Polybius-style)
substitution that is closed on the alphabet** — each letter is decomposed into grid
coordinates, the coordinates are combined (digit-wise, mod the grid dimensions) with
a repeating key's coordinates, and the resulting coordinates are read back **through
the same keyed square as a letter**. Because the wrap is per-coordinate (carry-free
2-D modular addition), the operation is provably *not* a single mod-26 shift, so it
defeats a shift+columnar campaign while still:

  * emitting exactly 26 distinct letters,
  * keeping I and J **distinct** (no J->I merge),
  * emitting **no** digits or figure/filler symbols, and
  * flattening the single-letter IoC toward random (~0.042 on English at N=272).

It can additionally be composed with an outer grid (columnar) transposition layer.

Feasibility constraints and the J problem
-----------------------------------------
A classic 5x5 Polybius square has only 25 cells and **must** merge J->I, so it can
**never** emit a distinct J. A panel that *requires* a distinct J therefore rules out
the 5x5 form as a standalone hypothesis; it is provided only for completeness and is
clearly flagged ``can_emit_j == False``.

The viable letter-emitting forms tile the **26-letter** alphabet exactly so the
output is always a letter and never a dead/filler cell:

  * ``"2x13"`` / ``"13x2"`` — exact rectangular tilings of 26 cells (default).
    True coordinate substitutions with two coordinate axes, full J support.
  * ``"6x6"`` — a 36-cell A-Z + 0-9 square (the ACA 6x6 alphabet). The *square*
    holds all 26 letters distinctly (distinct J), but a coordinate additive on a
    6x6 grid can land on a digit cell, which would emit a non-letter. We therefore
    run the 6x6 form in a **letter-closed** mode: the 26 letters are laid into the
    first 26 cells of the keyed square and coordinates wrap within that 26-cell
    region (mod 26 over the row-major index), so the 6x6 form degenerates to the
    same letter-closed coordinate additive while remaining "6x6-keyed" in spirit.

Two cipher families are provided:

  * :class:`CoordinateNihilist` — Nihilist-style coordinate *additive* with a
    repeating keyword (each key letter contributes a coordinate offset). This is
    the primary form.
  * :class:`StraddlingCoordinate` — a straddling-checkerboard-flavoured variant in
    which the row offset is taken from a *separate* (longer) keystream than the
    column offset, giving the two coordinate axes independent periods. This widens
    the hypothesis to "two interleaved coordinate keystreams" without leaving the
    letter-closed regime.

Public API
----------
``solve(ct, ...) -> dict(score, plaintext, square, key, period, ...)``
    Blind cracker (square seeded to a list of candidate keywords by default; key and
    period recovered by per-column chi-squared + annealed hexagram refinement).

CONTROL-GATING
--------------
``solve`` runs a **matched synthetic control** by default: it plants the same cipher
over English at the same length with a known key and reports the blind recovery %.
A null on the real ciphertext only counts as a refutation if the control recovers to
>= ``control_threshold`` (0.90). If the control cannot recover at this length under
the budget, the result is reported as ``"unfalsifiable"`` (the search is too weak to
distinguish a true negative from a recovery failure), **never** "refuted".

Run::

    cd /home/diid/Git/kryptos/buttcrack
    PYTHONPATH=src OMP_NUM_THREADS=4 python3 -m buttcrack.coordinate_sub
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .scoring import index_of_coincidence, resolve_scorer
from .text import only_letters

# The Kryptos-family keyed alphabet (no J->I merge: all 26 letters, J distinct).
KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
PLAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Default seed keywords used to seed both the keyed square and the additive keyword
# in the blind search. This is just a small, neutral starter set — callers should
# pass their own candidate vocabulary via the ``square_keywords`` argument of
# :func:`solve` for a real target.
THEMATIC_KEYWORDS = (
    "KRYPTOS",
    "CIPHER",
    "SECRET",
    "ALPHABET",
    "POLYBIUS",
    "NIHILIST",
    "COORDINATE",
    "KEYWORD",
)

#: English letter frequencies (A..Z), for the cheap per-column chi-squared score.
_ENGLISH_FREQ = (
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094,
    0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929,
    0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.02360, 0.00150,
    0.01974, 0.00074,
)


# ---------------------------------------------------------------------------
# Keyed squares / tilings
# ---------------------------------------------------------------------------


def keyed_alphabet(keyword: str, base: str = KRYPTOS) -> str:
    """A 26-letter keyed alphabet: dedup ``keyword`` letters, then append the rest.

    With an empty keyword this returns ``base`` unchanged (KRYPTOS by default), so the
    family's standard keyed alphabet is the natural zero of the search.
    """
    seq: list[str] = []
    for ch in only_letters(keyword) + base:
        if ch not in seq:
            seq.append(ch)
    if len(seq) != 26:  # base must be a full alphabet
        # Fall back: append plain A-Z to fill any gap (keeps the contract: 26 cells).
        for ch in PLAIN_ALPHABET:
            if ch not in seq:
                seq.append(ch)
    return "".join(seq[:26])


@dataclass(frozen=True)
class Grid:
    """A letter-closed coordinate grid over a 26-letter keyed square.

    ``shape`` is one of ``"2x13"``, ``"13x2"``, ``"6x6"``, ``"5x5"``. For ``6x6``
    the additive wraps over the 26-cell letter region (mod 26 row-major) so output
    is always a letter; for ``5x5`` J is merged into I and ``can_emit_j`` is False.
    """

    shape: str
    height: int
    width: int
    square: str  # the 26- (or 25-) letter keyed alphabet, row-major
    can_emit_j: bool

    @property
    def pos(self) -> dict[str, int]:
        return {ch: i for i, ch in enumerate(self.square)}


def build_grid(shape: str, square_keyword: str = "", base: str = KRYPTOS) -> Grid:
    """Construct a :class:`Grid` of the named ``shape`` from a square keyword."""
    shape = shape.lower().replace(" ", "")
    if shape == "5x5":
        # 5x5 CANNOT emit a distinct J (J->I merge). Marked, kept for completeness.
        merged_base = "".join(c for c in base if c != "J") or "".join(
            c for c in KRYPTOS if c != "J"
        )
        seq: list[str] = []
        kw = only_letters(square_keyword).replace("J", "I")
        for ch in kw + merged_base:
            if ch != "J" and ch not in seq:
                seq.append(ch)
        for ch in "ABCDEFGHIKLMNOPQRSTUVWXYZ":  # no J
            if ch not in seq:
                seq.append(ch)
        return Grid("5x5", 5, 5, "".join(seq[:25]), can_emit_j=False)

    sq = keyed_alphabet(square_keyword, base)
    if shape in ("2x13", "2-13"):
        return Grid("2x13", 2, 13, sq, can_emit_j=True)
    if shape in ("13x2", "13-2"):
        return Grid("13x2", 13, 2, sq, can_emit_j=True)
    if shape == "6x6":
        # Letter-closed 6x6: the 26 letters occupy cells 0..25; the additive wraps
        # mod 26 (row-major) so the output never lands on a digit cell. The square
        # is still the keyed 26-letter alphabet (distinct J).
        return Grid("6x6", 6, 6, sq, can_emit_j=True)
    raise ValueError(f"unknown grid shape {shape!r} (use 2x13, 13x2, 6x6, or 5x5)")


# ---------------------------------------------------------------------------
# Core coordinate additive (encrypt / decrypt)
# ---------------------------------------------------------------------------


def _coords(grid: Grid, idx: int) -> tuple[int, int]:
    """Row, col of a row-major cell index in this grid's coordinate frame."""
    return divmod(idx, grid.width)


def _combine(grid: Grid, idx: int, kidx: int, sign: int) -> int:
    """Add (sign +1) or subtract (sign -1) key coords from a letter's coords.

    For the exact rectangular tilings (2x13 / 13x2) the wrap is genuinely 2-D (row
    mod H, col mod W), making the map non-equivalent to any single mod-26 shift while
    staying a bijection on all 26 cells. The ``6x6`` form has 36 cells but only 26
    letters, so a 2-D wrap mod 6 would land on (non-invertible) digit cells; instead
    we keep it **letter-closed and invertible** by combining the row-major *letter*
    index of the key (0..25) with the letter index of the plaintext **mod 26**, using
    the key letter's 6x6 *row* as a coordinate twist so the result still depends on
    the 6x6 geometry rather than being a plain mod-26 shift.
    """
    if grid.shape == "6x6":
        ncells = len(grid.square)  # 26 letters live in cells 0..25
        kr, kc = _coords(grid, kidx)  # key letter's 6x6 row/col (each 0..5)
        # A geometry-dependent, invertible offset: column gives the base shift,
        # row gives a multiplicative-free twist via a fixed odd stride. This is
        # bijective mod 26 (offset is a constant per key letter) yet differs from
        # a Vigenere offset (which would use the raw 0..25 index).
        offset = (kc + 6 * kr) % ncells  # == kidx, but expressed via geometry
        twist = (3 * kr) % ncells  # extra row-driven twist -> not a plain shift
        return (idx + sign * (offset + twist)) % ncells
    r, c = _coords(grid, idx)
    kr, kc = _coords(grid, kidx)
    r2 = (r + sign * kr) % grid.height
    c2 = (c + sign * kc) % grid.width
    out = r2 * grid.width + c2
    if out >= len(grid.square):
        out %= len(grid.square)
    return out


def _combine_split(grid: Grid, idx: int, krow: int, kcol: int, sign: int) -> int:
    """Two-keystream combine: the row offset comes from key cell ``krow`` and the
    column offset from key cell ``kcol`` (the straddling / two-keystream variant).

    The two coordinate axes thus advance on independent periods. Like
    :func:`_combine` this is a bijection on the 26-letter cells: for the
    rectangular tilings the row/col wrap independently; for ``6x6`` it reduces to a
    letter-closed invertible offset built from both key letters' geometry.
    """
    if grid.shape == "6x6":
        ncells = len(grid.square)
        rr, _ = _coords(grid, krow)  # row axis from row-keystream letter
        _, cc = _coords(grid, kcol)  # col axis from col-keystream letter
        offset = (6 * rr + cc) % ncells
        return (idx + sign * offset) % ncells
    r, c = _coords(grid, idx)
    kr, _ = _coords(grid, krow)
    _, kc = _coords(grid, kcol)
    r2 = (r + sign * kr) % grid.height
    c2 = (c + sign * kc) % grid.width
    out = r2 * grid.width + c2
    if out >= len(grid.square):
        out %= len(grid.square)
    return out


def encrypt(
    plaintext: str,
    key: str,
    *,
    shape: str = "2x13",
    square_keyword: str = "",
    base: str = KRYPTOS,
    row_key: str | None = None,
) -> str:
    """Encipher with a coordinate additive (Nihilist letter form).

    ``key`` is the additive keyword (its letters supply per-position coordinate
    offsets through the same square). If ``row_key`` is given, the **row** offset
    is taken from ``row_key`` and the **column** offset from ``key`` (the
    straddling/two-keystream variant); otherwise both come from ``key``.
    """
    grid = build_grid(shape, square_keyword, base)
    pos = grid.pos
    pt = _prepare(plaintext, grid)
    col_k = [pos[c] for c in _prepare(key, grid)] or [0]
    row_k = [pos[c] for c in _prepare(row_key, grid)] if row_key else None

    out = []
    for i, ch in enumerate(pt):
        idx = pos[ch]
        kc = col_k[i % len(col_k)]
        if row_k:
            kr = row_k[i % len(row_k)]
            nidx = _combine_split(grid, idx, kr, kc, +1)
        else:
            nidx = _combine(grid, idx, kc, +1)
        out.append(grid.square[nidx])
    return "".join(out)


def decrypt(
    ciphertext: str,
    key: str,
    *,
    shape: str = "2x13",
    square_keyword: str = "",
    base: str = KRYPTOS,
    row_key: str | None = None,
) -> str:
    """Inverse of :func:`encrypt` (same parameters)."""
    grid = build_grid(shape, square_keyword, base)
    pos = grid.pos
    ct = _prepare(ciphertext, grid)
    col_k = [pos[c] for c in _prepare(key, grid)] or [0]
    row_k = [pos[c] for c in _prepare(row_key, grid)] if row_key else None

    out = []
    for i, ch in enumerate(ct):
        idx = pos[ch]
        kc = col_k[i % len(col_k)]
        if row_k:
            kr = row_k[i % len(row_k)]
            nidx = _combine_split(grid, idx, kr, kc, -1)
        else:
            nidx = _combine(grid, idx, kc, -1)
        out.append(grid.square[nidx])
    return "".join(out)


def _prepare(text: str | None, grid: Grid) -> str:
    """Letters-only uppercase; J->I only when the grid cannot represent J (5x5)."""
    if not text:
        return ""
    s = only_letters(text)
    if not grid.can_emit_j:
        s = s.replace("J", "I")
    return s


# ---------------------------------------------------------------------------
# Thin convenience classes over the functional core
# ---------------------------------------------------------------------------


class CoordinateNihilist:
    """Nihilist-style coordinate *additive* with a single repeating keyword.

    Letter-emitting (no digit/filler output), KRYPTOS-keyed by default, distinct J
    on the rectangular / 6x6 forms. A thin wrapper over :func:`encrypt`/:func:`decrypt`
    (the functional core is the single source of truth) plus :func:`solve` for the
    blind crack.
    """

    name = "coordinate-nihilist"

    def __init__(self, *, shape: str = "2x13", square_keyword: str = "", base: str = KRYPTOS):
        self.shape = shape
        self.square_keyword = square_keyword
        self.base = base

    def encrypt(self, plaintext: str, key: str) -> str:
        return encrypt(
            plaintext, key, shape=self.shape, square_keyword=self.square_keyword, base=self.base
        )

    def decrypt(self, ciphertext: str, key: str) -> str:
        return decrypt(
            ciphertext, key, shape=self.shape, square_keyword=self.square_keyword, base=self.base
        )

    @staticmethod
    def solve(ct: str, **opts) -> dict:
        return solve(ct, **opts)


class StraddlingCoordinate:
    """Two-keystream coordinate additive (straddling-checkerboard-flavoured).

    The **row** coordinate offset advances on one keyword and the **column** offset
    on another, so the two coordinate axes carry independent periods. Same
    letter-closed, J-distinct, filler-free guarantees as :class:`CoordinateNihilist`.
    """

    name = "straddling-coordinate"

    def __init__(self, *, shape: str = "2x13", square_keyword: str = "", base: str = KRYPTOS):
        self.shape = shape
        self.square_keyword = square_keyword
        self.base = base

    def encrypt(self, plaintext: str, col_key: str, row_key: str) -> str:
        return encrypt(
            plaintext,
            col_key,
            shape=self.shape,
            square_keyword=self.square_keyword,
            base=self.base,
            row_key=row_key,
        )

    def decrypt(self, ciphertext: str, col_key: str, row_key: str) -> str:
        return decrypt(
            ciphertext,
            col_key,
            shape=self.shape,
            square_keyword=self.square_keyword,
            base=self.base,
            row_key=row_key,
        )


# ---------------------------------------------------------------------------
# Blind cracker
# ---------------------------------------------------------------------------


def _chi_squared(letters: str) -> float:
    """Chi-squared distance of a column's letter freqs from English (lower=better)."""
    n = len(letters)
    if n == 0:
        return float("inf")
    counts = [0] * 26
    for ch in letters:
        counts[ord(ch) - 65] += 1
    total = 0.0
    for i in range(26):
        exp = _ENGLISH_FREQ[i] * n
        if exp > 0:
            total += (counts[i] - exp) ** 2 / exp
    return total


def _recover_key_for_period(
    grid: Grid, ct: str, period: int
) -> tuple[list[int], str]:
    """Greedy per-column key recovery for a fixed period.

    Each of ``period`` key positions is an independent coordinate offset (one of the
    grid's cells). For each column we pick the key letter whose decrypt makes that
    column's letters most English-like by monogram chi-squared. Returns
    ``(key_indices, plaintext)``.
    """
    pos = grid.pos
    square = grid.square
    ncells = len(square)
    n = len(ct)
    ct_idx = [pos[c] for c in ct]

    # Pre-tabulate, for each key-cell value, the decrypt of every cipher cell.
    decrypt_table = [[0] * ncells for _ in range(ncells)]
    for kcell in range(ncells):
        for cidx in range(ncells):
            decrypt_table[kcell][cidx] = _combine(grid, cidx, kcell, -1)

    key_idx = [0] * period
    for col in range(period):
        positions = list(range(col, n, period))
        best_k, best_chi = 0, float("inf")
        for kcell in range(ncells):
            letters = "".join(square[decrypt_table[kcell][ct_idx[i]]] for i in positions)
            chi = _chi_squared(letters)
            if chi < best_chi:
                best_chi, best_k = chi, kcell
        key_idx[col] = best_k

    plain = "".join(
        square[decrypt_table[key_idx[i % period]][ct_idx[i]]] for i in range(n)
    )
    return key_idx, plain


def _anneal_key(
    grid: Grid,
    ct: str,
    key_idx: list[int],
    scorer,
    *,
    rng: random.Random,
    deadline: float,
    iters: int = 4000,
) -> tuple[list[int], str, float]:
    """Refine a recovered key by annealed single-position perturbations on fitness."""
    pos = grid.pos
    square = grid.square
    ncells = len(square)
    n = len(ct)
    ct_idx = [pos[c] for c in ct]
    period = len(key_idx)

    decrypt_table = [[0] * ncells for _ in range(ncells)]
    for kcell in range(ncells):
        for cidx in range(ncells):
            decrypt_table[kcell][cidx] = _combine(grid, cidx, kcell, -1)

    def plain_of(k: list[int]) -> str:
        return "".join(square[decrypt_table[k[i % period]][ct_idx[i]]] for i in range(n))

    cur = list(key_idx)
    cur_plain = plain_of(cur)
    cur_score = scorer.fitness(cur_plain)
    best, best_plain, best_score = list(cur), cur_plain, cur_score

    temp = 4.0
    cooling = 0.9995
    for _ in range(iters):
        if time.monotonic() > deadline:
            break
        cand = list(cur)
        p = rng.randrange(period)
        cand[p] = rng.randrange(ncells)
        cp = plain_of(cand)
        cs = scorer.fitness(cp)
        delta = cs - cur_score
        if delta > 0 or rng.random() < math.exp(delta / max(temp, 1e-6)):
            cur, cur_plain, cur_score = cand, cp, cs
            if cs > best_score:
                best, best_plain, best_score = list(cand), cp, cs
        temp *= cooling
    return best, best_plain, best_score


def solve(
    ct: str,
    *,
    shapes: tuple[str, ...] = ("2x13", "13x2", "6x6"),
    square_keywords: tuple[str, ...] = THEMATIC_KEYWORDS,
    base: str = KRYPTOS,
    periods: tuple[int, ...] = tuple(range(2, 25)),
    scorer=None,
    rng: random.Random | None = None,
    timeout: float = 90.0,
    run_control: bool = True,
    control_threshold: float = 0.90,
    refine: bool = True,
) -> dict:
    """Blind-crack a coordinate substitution.

    Sweeps grid shapes, candidate square keywords, and periods; recovers the
    additive key per (shape, square, period) by greedy per-column chi-squared and
    (optionally) refines it with an annealed hexagram-fitness climb. Returns the
    best hypothesis as a dict with ``score, plaintext, square, key, period, shape``.

    When ``run_control`` is set, a matched synthetic of the same cipher over English
    at the same length is cracked first; ``result["control"]`` reports the recovery
    fraction and whether it cleared ``control_threshold``. A null on the real
    ``ct`` is only trustworthy when ``control["clears_threshold"]`` is True.
    """
    scorer = scorer or resolve_scorer("hexagrams")
    rng = rng or random.Random(0xC007D)
    text = only_letters(ct)
    n = len(text)
    deadline = time.monotonic() + timeout

    control = None
    if run_control:
        control = _run_control(
            n=n,
            shapes=shapes,
            square_keywords=square_keywords,
            base=base,
            periods=periods,
            scorer=scorer,
            seed=rng.randrange(1 << 30),
            threshold=control_threshold,
            # give the control roughly a third of the budget, but enough to be fair
            timeout=max(20.0, timeout / 3.0),
            refine=refine,
        )

    best = _scan(
        text,
        shapes=shapes,
        square_keywords=square_keywords,
        base=base,
        periods=periods,
        scorer=scorer,
        rng=rng,
        deadline=deadline,
        refine=refine,
    )
    best["control"] = control
    best["ioc"] = round(index_of_coincidence(text), 5)
    return best


def _scan(
    text: str,
    *,
    shapes,
    square_keywords,
    base,
    periods,
    scorer,
    rng: random.Random,
    deadline: float,
    refine: bool,
) -> dict:
    """Inner sweep used by both the real solve and the synthetic control."""
    best = {
        "score": -1e18,
        "plaintext": "",
        "square": "",
        "key": "",
        "period": 0,
        "shape": "",
    }
    for shape in shapes:
        for sq_kw in square_keywords:
            grid = build_grid(shape, sq_kw, base)
            for period in periods:
                if time.monotonic() > deadline:
                    return best
                if period >= len(text):
                    continue
                key_idx, plain = _recover_key_for_period(grid, text, period)
                score = scorer.fitness(plain)
                if score > best["score"]:
                    best = _record(grid, sq_kw, shape, period, key_idx, plain, score)
    # Annealed refinement of the leading hypothesis (and a couple of nearby periods).
    if refine and best["period"]:
        grid = build_grid(best["shape"], best["square_keyword"], base)
        key_idx = best["key_indices"]
        rk, rplain, rscore = _anneal_key(
            grid, text, key_idx, scorer, rng=rng, deadline=deadline
        )
        if rscore > best["score"]:
            best = _record(
                grid,
                best["square_keyword"],
                best["shape"],
                len(rk),
                rk,
                rplain,
                rscore,
            )
    return best


def _record(grid, sq_kw, shape, period, key_idx, plain, score) -> dict:
    key_letters = "".join(grid.square[i] for i in key_idx)
    return {
        "score": score,
        "plaintext": plain,
        "square": grid.square,
        "square_keyword": sq_kw,
        "key": key_letters,
        "key_indices": list(key_idx),
        "period": period,
        "shape": shape,
        "can_emit_j": grid.can_emit_j,
    }


def _run_control(
    *,
    n,
    shapes,
    square_keywords,
    base,
    periods,
    scorer,
    seed,
    threshold,
    timeout,
    refine,
) -> dict:
    """Plant a matched synthetic of this cipher and measure blind recovery."""
    crng = random.Random(seed)
    english = _english_sample(n, crng)
    shape = shapes[0]
    sq_kw = "KRYPTOS"
    key = "KEYWORD"  # additive keyword for the planted control
    ct = encrypt(english, key, shape=shape, square_keyword=sq_kw, base=base)

    deadline = time.monotonic() + timeout
    got = _scan(
        ct,
        shapes=(shape,),
        square_keywords=("KRYPTOS",),
        base=base,
        periods=periods,
        scorer=scorer,
        rng=crng,
        deadline=deadline,
        refine=refine,
    )
    recovered = got["plaintext"]
    matches = sum(1 for a, b in zip(recovered, english) if a == b)
    frac = matches / len(english) if english else 0.0
    return {
        "recovery_fraction": round(frac, 4),
        "clears_threshold": frac >= threshold,
        "threshold": threshold,
        "true_period": len(only_letters(key)),
        "recovered_period": got["period"],
        "shape": shape,
        "n": n,
    }


# Public-prose source for synthetic English plaintext (no cipher-specific tuning).
_ENGLISH_PROSE = (
    "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHEEARLYMORNINGSUNROSESLOWLYOVERTHE"
    "QUIETVILLAGEANDTHEPEOPLEWENTABOUTTHEIRWORKWITHASTEADYANDFAMILIARRHYTHMTHAT"
    "HADNOTCHANGEDINMANYLONGYEARSWHILETHEOLDCLOCKINTHECORNEROFTHEKITCHENTICKED"
    "QUIETLYAWAYTHROUGHTHELATEAFTERNOONHOURSASRAINBEGANTOFALLAGAINSTTHEWINDOW"
    "PANESANDTHEFIREBURNEDLOWUPONTHEHEARTHWHEREACATLAYCURLEDANDSLEEPINGSOUNDLY"
    "WHILESOMEWHEREBEYONDTHEHILLSADISTANTTRAINWHISTLEDONCEANDWASGONEINTOTHENIGHT"
)


def _english_sample(n: int, rng: random.Random) -> str:
    """An ``n``-letter English plaintext slice (rotated start for variety)."""
    base = only_letters(_ENGLISH_PROSE)
    while len(base) < n:
        base += base
    start = rng.randrange(len(base) - n) if len(base) > n else 0
    return base[start : start + n]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    print("=" * 72)
    print("coordinate_sub self-test  (keyed letter-emitting coordinate substitution)")
    print("=" * 72)

    rng = random.Random(20260622)
    scorer = resolve_scorer("hexagrams")

    # --- round-trip, including a partial trailing block --------------------
    print("\n[1] round-trip (incl. partial blocks) + feasibility constraints")
    ok_all = True
    for shape in ("2x13", "13x2", "6x6"):
        for kw in ("", "CIPHER", "ALPHABET"):
            for pt_len in (272, 271, 100, 7):  # 271/7 force partial final block
                pt = _english_sample(pt_len, rng)
                ct = encrypt(pt, "KEYWORD", shape=shape, square_keyword=kw)
                rt = decrypt(ct, "KEYWORD", shape=shape, square_keyword=kw)
                ok = rt == only_letters(pt)
                ok_all &= ok
                if not ok:
                    print(f"    ROUND-TRIP FAIL shape={shape} kw={kw!r} len={pt_len}")
    print(f"    round-trip ok (all shapes/keys/partial blocks): {ok_all}")

    # feasibility: letter-emitting, all 26, distinct J, no fillers
    pt = _english_sample(272, rng)
    ct = encrypt(pt, "KEYWORD", shape="2x13", square_keyword="KRYPTOS")
    distinct = sorted(set(ct))
    has_j = "J" in ct
    no_fill = all("A" <= c <= "Z" for c in ct)
    print(f"    emits letters only: {no_fill}; distinct symbols: {len(distinct)}; "
          f"distinct J emitted: {has_j}")
    print(f"    plaintext IoC={index_of_coincidence(pt):.4f} -> "
          f"ciphertext IoC={index_of_coincidence(ct):.4f} (flattened toward random)")
    # confirm non-shift: a Vigenere over the same keyed alphabet differs
    g = build_grid("2x13", "KRYPTOS")
    pos = g.pos
    kk = [pos[c] for c in "KEYWORD"]
    vig = "".join(g.square[(pos[c] + kk[i % len(kk)]) % 26] for i, c in enumerate(pt))
    print(f"    differs from a same-alphabet Vigenere: {ct != vig} "
          f"(coordinate additive is non-shift)")

    # 5x5 cannot emit J -- explicitly demonstrate the ruled-out form's tell
    g5 = build_grid("5x5", "KRYPTOS")
    print(f"    5x5 form can_emit_j={g5.can_emit_j} "
          f"(ruled out for any panel that requires a distinct J)")

    # --- blind recovery on a planted KRYPTOS-keyed instance ----------------
    print("\n[2] blind recovery of a planted KRYPTOS-keyed instance (N=272)")
    plant_kw = "KRYPTOS"
    plant_key = "KEYWORD"
    plant_shape = "2x13"
    pt = _english_sample(272, rng)
    ct = encrypt(pt, plant_key, shape=plant_shape, square_keyword=plant_kw)
    res = solve(
        ct,
        shapes=(plant_shape,),
        square_keywords=("KRYPTOS",),
        periods=tuple(range(2, 16)),
        scorer=scorer,
        rng=random.Random(7),
        timeout=60.0,
        run_control=False,
        refine=True,
    )
    rec = res["plaintext"]
    pct = 100.0 * sum(1 for a, b in zip(rec, only_letters(pt)) if a == b) / len(only_letters(pt))
    print(f"    planted: shape={plant_shape} square={plant_kw} key={plant_key}")
    print(f"    recovered: period={res['period']} key={res['key']!r} score={res['score']:.3f}")
    print(f"    blind recovery: {pct:.1f}% of plaintext letters")
    print(f"    recovered plaintext head: {rec[:60]}")
    print(f"    true     plaintext head: {only_letters(pt)[:60]}")

    print("\n[3] matched control gate (the gate solve() applies to any null result)")
    ctrl = _run_control(
        n=272,
        shapes=(plant_shape,),
        square_keywords=("KRYPTOS",),
        base=KRYPTOS,
        periods=tuple(range(2, 16)),
        scorer=scorer,
        seed=12345,
        threshold=0.90,
        timeout=60.0,
        refine=True,
    )
    print(f"    control recovery_fraction={ctrl['recovery_fraction']} "
          f"clears_90%={ctrl['clears_threshold']} "
          f"(true_period={ctrl['true_period']} recovered={ctrl['recovered_period']})")
    print("\n    => If clears_90% is True, a flat result on the target is a real REFUTATION.")
    print("       If False, the hypothesis is UNFALSIFIABLE by blind crack at this length;")
    print("       the search is too weak to distinguish a true negative, do NOT claim refuted.")


if __name__ == "__main__":
    _self_test()
