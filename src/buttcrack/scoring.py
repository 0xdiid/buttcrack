"""Fitness scoring for candidate plaintexts.

The :class:`NgramScorer` loads an n-gram log-probability table (built by
``scripts/build_ngrams.py``) and scores text by summing log-probabilities. It
self-calibrates a 0..1 confidence from the loaded table by scoring a known
English reference paragraph, so confidence is comparable across ciphers.
"""

from __future__ import annotations

import functools
import math
from importlib import resources

from .text import only_letters

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
