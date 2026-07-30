"""Fitness scoring for candidate plaintexts.

The :class:`NgramScorer` loads an n-gram log-probability table (built by
``scripts/build_ngrams.py``) and scores text by summing log-probabilities. It
self-calibrates a 0..1 confidence from the loaded table by scoring a known
English reference paragraph, so confidence is comparable across ciphers.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable
from importlib import resources
from typing import Any

from .text import only_letters

try:  # optional acceleration only; the package itself stays dependency-free
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in dev/test
    _np = None  # type: ignore[assignment]  # optional-dependency fallback sentinel

# Pseudo-count (in n-gram windows) for shrinking short-text confidence toward
# the random baseline. Larger => more skeptical of short inputs. At 12, a clean
# solve needs ~48 letters to read "solved", while a one-quadgram fluke stays near 0.
CONFIDENCE_PSEUDOCOUNT = 12.0

# Per-language reference prose to calibrate "what a good score looks like"
# against the loaded n-gram table. Not tuned to any cipher; only used for the
# confidence sigmoid. Accents are folded to A-Z by the scorer.
_REF_TEXTS = {
    "english": (
        "the quick brown fox jumps over the lazy dog while the early morning sun "
        "rose slowly over the quiet village and the people went about their work "
        "with a steady and familiar rhythm that had not changed in many years"
    ),
    "french": (
        "le petit prince traversa la grande plaine ou le vent soufflait doucement "
        "sur les herbes hautes pendant que le soleil descendait lentement derriere "
        "les collines et que les oiseaux rentraient vers leurs nids avant la nuit"
    ),
    "german": (
        "der junge mann ging langsam durch die stille strasse waehrend die sonne "
        "hinter den alten haeusern verschwand und die menschen nach einem langen "
        "tag voller arbeit endlich nach hause zu ihren familien zurueckkehrten"
    ),
    "spanish": (
        "el viajero camino despacio por el largo camino mientras el sol se ocultaba "
        "detras de las montanas y la gente del pueblo regresaba a sus casas despues "
        "de una larga jornada de trabajo bajo el cielo claro de la tarde tranquila"
    ),
    "italian": (
        "il giovane cammino lentamente lungo la strada mentre il sole tramontava "
        "dietro le antiche case e la gente del paese tornava finalmente a casa "
        "dopo una lunga giornata di lavoro sotto il cielo sereno della sera"
    ),
    "latin": (
        "gallia est omnis divisa in partes tres quarum unam incolunt belgae aliam "
        "aquitani tertiam qui ipsorum lingua celtae nostra galli appellantur hi omnes "
        "lingua institutis legibus inter se differunt et bellum gerere constituerunt"
    ),
}
_REF_TEXT = _REF_TEXTS["english"]

# Standard English letter frequencies (proportions), for chi-squared scoring.
ENGLISH_MONOGRAM_FREQ = {
    "A": 0.08167,
    "B": 0.01492,
    "C": 0.02782,
    "D": 0.04253,
    "E": 0.12702,
    "F": 0.02228,
    "G": 0.02015,
    "H": 0.06094,
    "I": 0.06966,
    "J": 0.00153,
    "K": 0.00772,
    "L": 0.04025,
    "M": 0.02406,
    "N": 0.06749,
    "O": 0.07507,
    "P": 0.01929,
    "Q": 0.00095,
    "R": 0.05987,
    "S": 0.06327,
    "T": 0.09056,
    "U": 0.02758,
    "V": 0.00978,
    "W": 0.02360,
    "X": 0.00150,
    "Y": 0.01974,
    "Z": 0.00074,
}


class NgramScorer:
    """Log-probability scorer over a fixed n-gram size."""

    def __init__(self, name: str = "quadgrams", lang: str = "english"):
        self.name = name
        self.lang = lang
        self.n, table = _load_table(name, lang)
        total = sum(table.values())
        self.log_probs = {gram: math.log10(count / total) for gram, count in table.items()}
        # Floor for unseen n-grams: rarer than the rarest observed.
        self.floor = math.log10(0.01 / total)
        # Calibration: per-n-gram score of clean reference prose vs. random noise.
        self._english_ref = self.average(_REF_TEXTS.get(lang, _REF_TEXT))
        self._random_ref = self.floor  # random text hits the floor almost everywhere

    def score(self, text: str) -> float:
        """Total log-probability of ``text`` (higher/less-negative is better)."""
        letters = only_letters(text)
        n = self.n
        if len(letters) < n:
            return self.floor * max(1, len(letters))
        log_probs = self.log_probs
        floor = self.floor
        total = 0.0
        for i in range(len(letters) - n + 1):
            total += log_probs.get(letters[i : i + n], floor)
        return total

    def average(self, text: str) -> float:
        """Mean log-probability per n-gram window (length-independent)."""
        letters = only_letters(text)
        windows = len(letters) - self.n + 1
        if windows <= 0:
            return self.floor
        return self.score(text) / windows

    def fitness(self, text: str) -> float:
        """Entropy-normalized n-gram fitness for hill-climbing / annealing.

        AZdecrypt-style: the raw n-gram score is multiplied by the letter entropy so
        that degenerate low-entropy "solutions" (which game the n-gram score by piling
        onto a few common letters) are penalized. Higher is better; >0 means English-ish.
        Both the n-gram term (shifted above its floor) and the entropy fraction are
        non-negative, so a readable, full-alphabet plaintext maximizes the product.
        """
        letters = only_letters(text)
        windows = len(letters) - self.n + 1
        if windows <= 0:
            return 0.0
        avg = self.score(text) / windows
        H = letter_entropy(letters)
        return (avg - self.floor) * (H / ENGLISH_LETTER_ENTROPY)

    def anchored(self, text: str) -> float:
        """Anchor-normalized score: 0.0 ≈ random text, 1.0 ≈ typical clean language.

        ``(average - random_anchor) / (language_anchor - random_anchor)``, using the
        scorer's own calibration anchors. Because every model is normalized to *its own*
        anchors, anchored scores are comparable ACROSS models — an English quadgram
        model, a terse/route genre model, a word LM — where raw log-probabilities are
        not. This is what exposes the non-prose blind spot: a true route-register
        decode sits at the English model's "ghost ceiling" (~0.5–0.7) while scoring
        ~1.0 under a register-matched model. See :func:`anchored_score` for models
        that are not :class:`NgramScorer`.
        """
        lo, hi = self._random_ref, self._english_ref
        if hi <= lo:
            return 0.0
        return (self.average(text) - lo) / (hi - lo)

    def confidence(self, text: str) -> float:
        """Map a candidate's average score to a calibrated 0..1 confidence.

        Confidence is *sample-size aware*: the mean log-probability is a noisy
        estimate on short text, so we shrink it toward the random baseline with a
        pseudo-count (Bayesian shrinkage). One lucky quadgram ("THEM") therefore
        scores near zero, while a long paragraph keeps its full confidence. This
        stops short/repetitive inputs and tiny brute-force hits from looking solved.
        """
        letters = only_letters(text)
        windows = len(letters) - self.n + 1
        lo, hi = self._random_ref, self._english_ref
        if windows <= 0 or hi <= lo:
            return 0.0
        avg = self.score(text) / windows
        # Regress toward the random baseline by CONFIDENCE_PSEUDOCOUNT windows.
        shrunk = (windows * avg + CONFIDENCE_PSEUDOCOUNT * lo) / (windows + CONFIDENCE_PSEUDOCOUNT)
        midpoint = (lo + hi) / 2.0
        scale = (hi - lo) / 6.0 or 1.0
        return 1.0 / (1.0 + math.exp(-(shrunk - midpoint) / scale))


#: languages with bundled n-gram tables
LANGUAGES = ("english", "french", "german", "spanish", "italian", "latin")


@functools.cache
def get_scorer(name: str = "quadgrams", lang: str = "english") -> NgramScorer:
    """Cached scorer accessor (tables are read once per process)."""
    return NgramScorer(name, lang)


def ngram_table_available(name: str, lang: str = "english") -> bool:
    """Whether the ``<lang>_<name>.txt`` n-gram table is bundled/available."""
    try:
        return resources.files("buttcrack.data").joinpath(f"{lang}_{name}.txt").is_file()
    except (FileNotFoundError, ModuleNotFoundError):
        return False


def resolve_scorer(prefer: str = "quadgrams", lang: str = "english") -> NgramScorer:
    """Scorer for the preferred n-gram model, gracefully falling back to quadgrams.

    Lets a caller ask for a richer model (e.g. ``quintgrams`` / ``hexagrams``, which
    sharpen the fitness for the hardest searches) that degrades cleanly when its table
    isn't present. buttcrack bundles mono..hexagrams for English, so ``quintgrams`` and
    ``hexagrams`` resolve directly; other languages ship up to quadgrams (build higher
    orders with ``scripts/build_ngrams.py --max-n 6``). Use :func:`ngram_table_available`
    first if you want to warn on the fallback.
    """
    if prefer != "quadgrams" and ngram_table_available(prefer, lang):
        return get_scorer(prefer, lang)
    return get_scorer("quadgrams", lang)


def _load_table(name: str, lang: str = "english") -> tuple[int, dict[str, int]]:
    fname = f"{lang}_{name}.txt"
    raw = None
    try:
        raw = resources.files("buttcrack.data").joinpath(fname).read_text(encoding="ascii")
    except (FileNotFoundError, ModuleNotFoundError, Exception):
        raw = None
    if raw is None:
        import os

        _p = os.path.join(os.path.dirname(__file__), "data", fname)
        if os.path.isfile(_p):
            with open(_p, encoding="ascii") as _f:
                raw = _f.read()
    if raw is None:
        raise FileNotFoundError(
            f"n-gram table {fname!r} not found (lang={lang!r}). "
            f"Run scripts/build_ngrams.py --lang {lang} to build it."
        )
    table: dict[str, int] = {}
    n = 0
    for line in raw.splitlines():
        if not line:
            continue
        gram, _, count = line.partition(" ")
        table[gram] = int(count)
        n = len(gram)
    return n, table


#: Shannon entropy (bits/letter) of typical English letter frequencies.
ENGLISH_LETTER_ENTROPY = 4.18


def letter_entropy(text: str) -> float:
    """Shannon entropy (bits per letter) of the letter distribution of ``text``.

    ~4.18 for English, ~4.7 (log2 26) for uniform/random, low for degenerate text.
    Used to keep hill-climbers from converging onto a handful of common letters.
    """
    letters = only_letters(text)
    n = len(letters)
    if n == 0:
        return 0.0
    counts = [0] * 26
    for ch in letters:
        counts[ord(ch) - 65] += 1
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def index_of_coincidence(text: str) -> float:
    """Index of coincidence of the letters in ``text``.

    ~0.066 for English (mono-alphabetic and transposition preserve it),
    ~0.038 for random/polyalphabetic text. Key signal for the ``identify`` step.
    """
    letters = only_letters(text)
    n = len(letters)
    if n < 2:
        return 0.0
    counts = [0] * 26
    for ch in letters:
        counts[ord(ch) - 65] += 1
    return sum(c * (c - 1) for c in counts) / (n * (n - 1))


def chi_squared(letters: str) -> float:
    """Chi-squared distance of letter frequencies from English (lower = closer).

    Used to score Caesar/affine/Vigenere-column guesses cheaply.
    """
    n = len(letters)
    if n == 0:
        return float("inf")
    counts = [0] * 26
    for ch in letters:
        counts[ord(ch) - 65] += 1
    total = 0.0
    for i in range(26):
        expected = ENGLISH_MONOGRAM_FREQ[chr(65 + i)] * n
        total += (counts[i] - expected) ** 2 / expected if expected else 0.0
    return total


def anchored_score(score: float, random_anchor: float, language_anchor: float) -> float:
    """Normalize any model's score to its own anchors: 0.0 ≈ random, 1.0 ≈ language.

    The cross-model comparability trick for mixed scorer pipelines (quadgram + genre
    model + word LM): compute each model's score on random text (``random_anchor``)
    and on typical in-register text (``language_anchor``) once, then compare models on
    the anchored fraction instead of raw log-probabilities. :meth:`NgramScorer.anchored`
    does this automatically for bundled n-gram models.
    """
    if language_anchor <= random_anchor:
        raise ValueError("language_anchor must exceed random_anchor")
    return (score - random_anchor) / (language_anchor - random_anchor)


def best_caesar_gauge(text: str) -> tuple[int, float]:
    """The Caesar shift whose application makes ``text`` best fit English letter
    frequencies; returns ``(shift, chi_squared_at_shift)``.

    Cheap (26 chi-squared evaluations on one count vector) — cheap enough to run
    inside a hill-climb objective. See :class:`GaugeNormalizedScorer` for why.
    """
    letters = only_letters(text)
    n = len(letters)
    if n == 0:
        return 0, float("inf")
    counts = [0] * 26
    for ch in letters:
        counts[ord(ch) - 65] += 1
    best_shift, best_chi = 0, float("inf")
    for shift in range(26):
        total = 0.0
        for i in range(26):
            expected = ENGLISH_MONOGRAM_FREQ[chr(65 + i)] * n
            c = counts[(i - shift) % 26]  # the letter that +shift maps onto i
            total += (c - expected) ** 2 / expected if expected else 0.0
        if total < best_chi:
            best_shift, best_chi = shift, total
    return best_shift, best_chi


class GaugeNormalizedScorer:
    """N-gram scoring that is invariant to a global Caesar (alphabet-labelling) gauge.

    Many key parameterizations carry a gauge: an omitted or relabelled alphabet letter,
    a rotated keyed alphabet, a shifted ring. The *correct* decode then emerges
    Caesar-shifted, an n-gram model scores it as junk, and the search has **no gradient
    toward the right answer** — a measured failure mode (one wrong gauge value was 84%
    of a score cliff). The fix is not to search the gauge as another axis but to
    normalize it away inside the objective: pick the shift by unigram (chi-squared)
    fit, THEN n-gram score the normalized text, making all 26 gauge frames reachable
    from a single climb.

    Wraps any :class:`NgramScorer`; ``score``/``average``/``fitness`` mirror its API.
    """

    def __init__(self, base: NgramScorer | None = None):
        self.base = base or get_scorer()

    def normalize(self, text: str) -> str:
        """``text`` with its best Caesar gauge applied (the shift chi-squared picks)."""
        letters = only_letters(text)
        shift, _ = best_caesar_gauge(letters)
        if shift == 0:
            return letters
        return "".join(chr((ord(c) - 65 + shift) % 26 + 65) for c in letters)

    def score(self, text: str) -> float:
        return self.base.score(self.normalize(text))

    def average(self, text: str) -> float:
        return self.base.average(self.normalize(text))

    def fitness(self, text: str) -> float:
        return self.base.fitness(self.normalize(text))


def excision_score(
    text: str,
    scorer: NgramScorer | None = None,
    *,
    excise_len: int | None = None,
    mode: str = "contiguous",
    width: int | None = None,
    step: int = 1,
) -> dict:
    """Score exactly the letters a contamination hypothesis claims are language.

    A plaintext with an embedded non-language block (an inserted key, a serial, a
    coordinate field) drags a whole-text n-gram score down, so a scan over "where is
    the insert?" hypotheses scores every one of them as junk. This scores the
    *complement* instead: remove the hypothesized insert and score what remains,
    maximizing over placements.

    ``mode="contiguous"`` removes a run of ``excise_len`` letters at every position
    (stride ``step``); ``mode="column"`` removes one residue class mod ``width``
    (column-shaped inserts in a grid write-in — ``excise_len`` is not used). Returns
    ``{"score", "at", "mode", "excised"}`` where ``score`` is the best per-window
    average of the remaining text and ``at`` is the winning position / residue.
    """
    scorer = scorer or get_scorer()
    letters = only_letters(text)
    if mode == "contiguous":
        if excise_len is None or not 0 < excise_len < len(letters):
            raise ValueError(f"excise_len must be in (0, {len(letters)}), got {excise_len}")
        best: tuple[float, int, str] | None = None
        for j in range(0, len(letters) - excise_len + 1, step):
            kept = letters[:j] + letters[j + excise_len:]
            s = scorer.average(kept)
            if best is None or s > best[0]:
                best = (s, j, letters[j:j + excise_len])
        assert best is not None
        return {"score": best[0], "at": best[1], "mode": mode, "excised": best[2]}
    if mode == "column":
        if not width or width < 2:
            raise ValueError("mode='column' requires width >= 2")
        best = None
        for r in range(width):
            kept = "".join(c for i, c in enumerate(letters) if i % width != r)
            s = scorer.average(kept)
            if best is None or s > best[0]:
                best = (s, r, letters[r::width])
        assert best is not None
        return {"score": best[0], "at": best[1], "mode": mode, "excised": best[2]}
    raise ValueError(f"mode must be 'contiguous' or 'column', got {mode!r}")


# --- vectorized batch scoring (optional numpy accelerator) -------------------

#: Largest ``26**n`` for which a dense n-gram LUT is built (26**5 ≈ 11.9M cells,
#: ~95 MB as float64). Beyond this the batch scorer falls back to per-row scoring.
_MAX_DENSE_LUT = 26**5


def text_to_ordinals(text: str) -> list[int]:
    """A-Z letters of ``text`` as 0..25 ordinals (non-letters dropped)."""
    return [ord(c) - 65 for c in only_letters(text)]


class BatchNgramScorer:
    """Score many equal-length candidates at once, matching :meth:`NgramScorer.score`.

    Cryptanalytic search loops (brute force, hill-climbing, annealing) evaluate the
    same quadgram log-probability sum over thousands of candidate plaintexts per
    second; calling :meth:`NgramScorer.score` — a Python ``dict.get`` per window — is
    the bottleneck. This wraps a scorer's table as a dense ``26**n`` lookup array so a
    whole batch of candidates is scored by array-indexing and a summed sliding window.

    ``score_batch`` reproduces :meth:`NgramScorer.score` **exactly** (same floor for
    unseen n-grams, same short-text branch), so a solver can hunt with the fast path
    and report scores that agree bit-for-bit with the rest of buttcrack. When numpy is
    absent — or the model is too large for a dense LUT (quint/hexagrams) — it falls
    back transparently to per-row scoring, so callers never need to branch on it.
    """

    def __init__(self, scorer: NgramScorer):
        self.scorer = scorer
        self.n = scorer.n
        self.floor = scorer.floor
        self._lut = None
        if _np is not None and 26**self.n <= _MAX_DENSE_LUT:
            lut = _np.full(26**self.n, scorer.floor, dtype=_np.float64)
            for gram, lp in scorer.log_probs.items():
                idx = 0
                for ch in gram:
                    idx = idx * 26 + (ord(ch) - 65)
                lut[idx] = lp
            self._lut = lut

    @property
    def vectorized(self) -> bool:
        """True when the fast numpy LUT path is active (else the per-row fallback)."""
        return self._lut is not None

    def score_batch(self, batch: Any) -> list[float]:
        """Total log-probability of each row, matching :meth:`NgramScorer.score`.

        ``batch`` is a rectangular ``(B, L)`` of 0..25 ordinals — a numpy array, or any
        sequence of equal-length int sequences. Returns a list of ``B`` floats (the
        vectorized path also accepts and is happiest with a numpy array). Rows shorter
        than the n-gram size ``n`` get ``floor * max(1, L)``, exactly as the scalar scorer.
        """
        n = self.n
        if self._lut is not None:
            arr = _np.asarray(batch)
            if arr.ndim == 1:
                arr = arr[None, :]
            B, length = arr.shape
            if length < n:
                return [self.floor * max(1, length)] * B
            idx = arr[:, 0 : length - n + 1].astype(_np.int64)
            for k in range(1, n):
                idx = idx * 26 + arr[:, k : length - n + 1 + k]
            return self._lut[idx].sum(axis=1).tolist()
        # Fallback: score each row via the scalar scorer.
        rows = list(batch)
        out: list[float] = []
        for row in rows:
            out.append(self.scorer.score("".join(chr(65 + int(o)) for o in row)))
        return out

    def score_texts(self, texts: Iterable[str]) -> list[float]:
        """Total log-probability of each text (any lengths), matching the scalar scorer.

        Convenience wrapper that normalizes each string to A-Z ordinals and groups equal
        lengths so the vectorized path still applies within each group.
        """
        items = [text_to_ordinals(t) for t in texts]
        scores: list[float | None] = [None] * len(items)
        by_len: dict[int, list[int]] = {}
        for i, ords in enumerate(items):
            by_len.setdefault(len(ords), []).append(i)
        for length, idxs in by_len.items():
            if self._lut is not None and length >= self.n:
                mat = _np.array([items[i] for i in idxs], dtype=_np.int64)
                for pos, s in zip(idxs, self.score_batch(mat), strict=True):
                    scores[pos] = s
            else:
                for i in idxs:
                    scores[i] = self.scorer.score("".join(chr(65 + o) for o in items[i]))
        return [s if s is not None else self.floor for s in scores]

    def fitness_batch(self, batch: Any) -> list[float]:
        """Entropy-weighted fitness of each row, matching :meth:`NgramScorer.fitness`.

        The AZdecrypt-style objective for hill-climbing: the mean n-gram log-prob shifted
        above the floor, scaled by the row's letter-entropy fraction, so degenerate
        low-entropy "solutions" are penalized. Rows too short to hold a window score 0.0.
        """
        n = self.n
        if self._lut is not None:
            arr = _np.asarray(batch)
            if arr.ndim == 1:
                arr = arr[None, :]
            B, length = arr.shape
            windows = length - n + 1
            if windows <= 0:
                return [0.0] * B
            totals = _np.asarray(self.score_batch(arr))
            avg = totals / windows
            counts = _np.zeros((B, 26), dtype=_np.int64)
            _np.add.at(counts, (_np.arange(B)[:, None], arr), 1)
            p = counts / length
            with _np.errstate(divide="ignore", invalid="ignore"):
                terms = _np.where(p > 0, -p * _np.log2(p), 0.0)
            H = terms.sum(axis=1)
            return ((avg - self.floor) * (H / ENGLISH_LETTER_ENTROPY)).tolist()
        return [self.scorer.fitness("".join(chr(65 + int(o)) for o in row)) for row in batch]


@functools.cache
def get_batch_scorer(name: str = "quadgrams", lang: str = "english") -> BatchNgramScorer:
    """Cached :class:`BatchNgramScorer` over the same tables as :func:`get_scorer`."""
    return BatchNgramScorer(get_scorer(name, lang))
