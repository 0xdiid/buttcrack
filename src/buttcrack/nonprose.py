"""Flag candidate plaintexts that read as *structured / route text* rather than prose.

Every fitness gate in ``butt`` bottoms out in standard English quadgrams, which
implicitly assume the plaintext is natural-language prose (a diary entry, a
letter, a narrative). A different, entirely valid class of plaintext —
directions, spelled-out coordinates, ordinal/unit lists, imperative
step-by-step instructions ("NORTH SEVEN PACES LEFT AT THE OAK") — is *not* prose
and scores in the English "ghost band": above random, but a full log-unit below
real prose. Ranked purely by an English objective, such a true decode loses to
noise and gets thrown away.

This module builds a second opinion. It trains a lightweight interpolated
character-n-gram :class:`GenreModel` on a synthetic route/instruction corpus and
another on ordinary English prose, then normalizes each model's raw log-prob
against its own genre-typical and random-text anchors so the two are directly
comparable (``frac`` = 1.0 means genre-typical, 0.0 means random). A candidate
whose *route* frac beats its *prose* frac leans non-prose and deserves human
eyes even when its English quadgram score looks mediocre.

The scorer and corpus generator are pure-stdlib re-implementations of the
``Tri`` interpolated trigram model and ``route_corpus`` generator; no external
n-gram tables or numpy are required.
"""

from __future__ import annotations

import functools
import math
import random
from collections import defaultdict

from .text import only_letters

#: Additive (Lidstone) smoothing count applied to every n-gram order, so unseen
#: contexts back off smoothly instead of collapsing to zero probability.
SMOOTHING = 0.4

#: Window length and sample count for the genre/random anchors (see
#: :meth:`GenreModel._calibrate`). Fixed so :func:`default_models` is deterministic.
_ANCHOR_WINDOW = 200
_ANCHOR_SAMPLES = 50
_ANCHOR_SEED = 20260709

#: How much a candidate's route frac must exceed its prose frac before it is
#: called non-prose (keeps ordinary prose, whose fracs are close, out of the net).
_LEAN_MARGIN = 0.03


def _default_lambdas(order: int) -> dict[int, float]:
    """Interpolation weights per n-gram order (highest order gets the most mass).

    Reproduces the canonical ``(trigram, bigram, unigram) = (0.6, 0.3, 0.1)`` mix
    for ``order == 3`` and falls back to a normalized geometric decay otherwise.
    """
    if order == 3:
        return {3: 0.6, 2: 0.3, 1: 0.1}
    raw = {order - i: 0.5**i for i in range(order)}
    total = sum(raw.values())
    return {m: w / total for m, w in raw.items()}


class GenreModel:
    """An interpolated, backoff-smoothed character-n-gram language model.

    The probability of each character given its preceding context is a weighted
    blend of the full-order conditional, all lower-order conditionals, and the
    unigram distribution; every order is additively smoothed by :data:`SMOOTHING`,
    so no context ever yields ``-inf``. :meth:`score` returns the mean natural-log
    probability per character (length-independent), and :meth:`frac` rescales that
    onto a genre-typical (1.0) .. random (0.0) axis for cross-model comparison.
    """

    def __init__(
        self,
        order: int,
        smoothing: float,
        lambdas: dict[int, float],
        gram_counts: dict[int, dict[str, int]],
        ctx_counts: dict[int, dict[str, int]],
    ) -> None:
        self.order = order
        self.smoothing = smoothing
        self.lambdas = lambdas
        self._gram_counts = gram_counts
        self._ctx_counts = ctx_counts
        #: mean :meth:`score` of genre-typical windows (set by :meth:`_calibrate`).
        self.genre_anchor: float = 0.0
        #: mean :meth:`score` of uniform-random windows (set by :meth:`_calibrate`).
        self.random_anchor: float = 0.0

    @classmethod
    def train(cls, corpus: str, *, order: int = 3) -> GenreModel:
        """Train a model on ``corpus`` (letters only, folded to upper case).

        ``order`` is the highest n-gram order used; the model interpolates every
        order from ``order`` down to unigrams. Returns a calibrated model.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        letters = only_letters(corpus)
        gram_counts: dict[int, dict[str, int]] = {}
        ctx_counts: dict[int, dict[str, int]] = {}
        for m in range(1, order + 1):
            grams: dict[str, int] = defaultdict(int)
            ctxs: dict[str, int] = defaultdict(int)
            for i in range(len(letters) - m + 1):
                gram = letters[i : i + m]
                grams[gram] += 1
                ctxs[gram[:-1]] += 1
            gram_counts[m] = dict(grams)
            ctx_counts[m] = dict(ctxs)
        model = cls(order, SMOOTHING, _default_lambdas(order), gram_counts, ctx_counts)
        model._calibrate(letters)
        return model

    def _cond_prob(self, m: int, context: str, char: str) -> float:
        """Smoothed conditional probability ``P(char | context)`` at order ``m``."""
        k = self.smoothing
        num = self._gram_counts[m].get(context + char, 0) + k
        den = self._ctx_counts[m].get(context, 0) + k * 26.0
        return num / den

    def _lambdas_for(self, max_order: int) -> dict[int, float]:
        """Interpolation weights restricted to orders ``1..max_order``, renormalized."""
        if max_order >= self.order:
            return self.lambdas
        sub = {m: self.lambdas[m] for m in range(1, max_order + 1)}
        total = sum(sub.values())
        return {m: w / total for m, w in sub.items()}

    def _logprob(self, context: str, char: str) -> float:
        """Interpolated log-probability of ``char`` following ``context``.

        ``context`` may be shorter than ``order - 1`` (start of text); the mix then
        uses only the available orders, so no position is ever left unscored.
        """
        max_order = min(self.order, len(context) + 1)
        lambdas = self._lambdas_for(max_order)
        mix = 0.0
        for m in range(1, max_order + 1):
            ctx = context[len(context) - (m - 1) :] if m > 1 else ""
            mix += lambdas[m] * self._cond_prob(m, ctx, char)
        return math.log(mix)

    def score(self, text: str) -> float:
        """Mean natural-log probability per character (higher/less-negative = better)."""
        letters = only_letters(text)
        if not letters:
            return math.log(1.0 / 26.0)
        span = self.order - 1
        total = 0.0
        for i in range(len(letters)):
            context = letters[max(0, i - span) : i]
            total += self._logprob(context, letters[i])
        return total / len(letters)

    def frac(self, text: str) -> float:
        """Rescale :meth:`score` onto a genre-typical (1.0) .. random (0.0) axis.

        Values above 1.0 are possible for text even more genre-typical than the
        training windows; values near or below 0.0 read as random noise.
        """
        spread = self.genre_anchor - self.random_anchor
        if spread <= 0:
            return 0.0
        return (self.score(text) - self.random_anchor) / spread

    def _calibrate(self, letters: str) -> None:
        """Set the genre-typical and random anchors used by :meth:`frac`."""
        rng = random.Random(_ANCHOR_SEED)
        window = _ANCHOR_WINDOW
        # Genre-typical anchor: mean score over sampled windows of the training text.
        if len(letters) <= window:
            self.genre_anchor = self.score(letters) if letters else math.log(1.0 / 26.0)
        else:
            starts = [rng.randint(0, len(letters) - window) for _ in range(_ANCHOR_SAMPLES)]
            self.genre_anchor = sum(self.score(letters[s : s + window]) for s in starts) / len(
                starts
            )
        # Random anchor: mean score over uniform-random letter windows.
        rand_scores = []
        for _ in range(_ANCHOR_SAMPLES):
            noise = "".join(chr(65 + rng.randrange(26)) for _ in range(window))
            rand_scores.append(self.score(noise))
        self.random_anchor = sum(rand_scores) / len(rand_scores)


# --- synthetic route / instruction corpus ------------------------------------------

_ONES = (
    "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN "
    "FOURTEEN FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN"
).split()
_TENS = "TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY".split()
_ORDINALS = (
    "FIRST SECOND THIRD FOURTH FIFTH SIXTH SEVENTH EIGHTH NINTH TENTH ELEVENTH TWELFTH"
).split()
_UNITS = "PACES STEPS FEET YARDS DEGREES ROWS COLUMNS LETTERS LINES STONES DOORS".split()
_VERBS = (
    "COUNT TURN TAKE READ REPEAT ADVANCE MOVE FOLLOW SKIP BEGIN FACE WALK CLIMB "
    "CROSS MARK ADD SUBTRACT"
).split()
_DIRECTIONS = "NORTH SOUTH EAST WEST LEFT RIGHT UP DOWN FORWARD BACK CLOCKWISE WIDDERSHINS".split()
_GLUE = "THEN AND THE TO FROM AT OF UNTIL PAST TOWARD BY".split()


def _numword(n: int) -> str:
    """Spell a positive integer below 100 as an unspaced word (e.g. 47 -> FORTYSEVEN)."""
    if n < 20:
        return _ONES[n - 1]
    tens, ones = divmod(n, 10)
    return _TENS[tens - 2] + (_ONES[ones - 1] if ones else "")


def route_corpus(sentences: int = 400, *, seed: int = 0) -> str:
    """Generate a deterministic synthetic route/instruction corpus.

    Each "sentence" is a short template of spelled numbers, ordinals, units,
    imperative verbs, compass directions and glue words — the vocabulary of
    directions, coordinates and step lists. The result is an unspaced upper-case
    A-Z string suitable for :meth:`GenreModel.train`. Fully determined by ``seed``.
    """
    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(sentences):
        r = rng.random()
        if r < 0.35:
            out += [
                rng.choice(_VERBS),
                _numword(rng.randint(1, 99)),
                rng.choice(_UNITS),
                rng.choice(_DIRECTIONS),
            ]
        elif r < 0.6:
            out += [
                _numword(rng.randint(1, 99)),
                rng.choice(_UNITS),
                rng.choice(_GLUE),
                rng.choice(_VERBS),
                rng.choice(_GLUE),
                rng.choice(_ORDINALS),
                rng.choice(_UNITS),
            ]
        elif r < 0.8:
            out += [
                rng.choice(_GLUE),
                rng.choice(_ORDINALS),
                rng.choice(_UNITS),
                rng.choice(_VERBS),
                _numword(rng.randint(1, 19)),
            ]
        else:
            out += [
                rng.choice(_VERBS),
                rng.choice(_GLUE),
                _numword(rng.randint(1, 59)),
                rng.choice(_GLUE),
                _numword(rng.randint(1, 59)),
                rng.choice(_UNITS),
            ]
    return "".join(out)


_MONTHS = (
    "JANUARY FEBRUARY MARCH APRIL MAY JUNE JULY AUGUST SEPTEMBER OCTOBER NOVEMBER DECEMBER"
).split()
_WEEKDAYS = "MONDAY TUESDAY WEDNESDAY THURSDAY FRIDAY SATURDAY SUNDAY".split()
_TELEGRAPH_VERBS = (
    "ARRIVE DEPART CONFIRM ADVISE SEND AWAIT PROCEED RETURN MEET DELAY CANCEL REQUEST"
).split()
_TELEGRAPH_NOUNS = (
    "PACKAGE FUNDS ORDERS PARTY SHIPMENT CONTACT LETTER TRAIN VESSEL CARGO SIGNAL PAPERS"
).split()

#: Registers :func:`register_corpus` can synthesize. Use them to build
#: register-diverse plant gates and register-matched n-gram tables.
REGISTERS = ("route", "wordlist", "dates", "telegraphic", "numeric")


def register_corpus(register: str, *, sentences: int = 400, seed: int = 0) -> str:
    """Deterministic synthetic corpus for a non-prose plaintext REGISTER.

    A plant gate's recall is register-specific: a solver tuned on prose recovers
    prose plants and silently loses wordlist/coded/telegraphic payloads, so a null
    proven on prose plants does not transfer (see
    :class:`buttcrack.evidence.PlantGate`). These corpora exist to (a) plant
    same-register synthetics and (b) train register-matched
    :class:`GenreModel`/n-gram tables that must beat the English model on their
    own register BEFORE any search time is spent behind them.

    Registers: ``route`` (directions/distances — :func:`route_corpus`),
    ``wordlist`` (concatenated dictionary words, no grammar), ``dates`` (spelled
    dates, ordinals, years), ``telegraphic`` (terse cable style: verbs, nouns,
    spelled figures, STOP), ``numeric`` (spelled numbers only). Unspaced A-Z,
    fully determined by ``seed``.
    """
    if register == "route":
        return route_corpus(sentences, seed=seed)
    rng = random.Random(seed)
    out: list[str] = []
    if register == "wordlist":
        from .words import _words

        pool = [w for w in _words() if len(w) >= 4]
        for _ in range(sentences):
            out += [rng.choice(pool) for _ in range(3)]
    elif register == "dates":
        for _ in range(sentences):
            out += [
                rng.choice(_WEEKDAYS),
                rng.choice(_ORDINALS),
                rng.choice(_MONTHS),
                _numword(rng.randint(1, 99)),
            ]
    elif register == "telegraphic":
        for _ in range(sentences):
            r = rng.random()
            if r < 0.5:
                out += [
                    rng.choice(_TELEGRAPH_VERBS),
                    rng.choice(_TELEGRAPH_NOUNS),
                    rng.choice(_WEEKDAYS),
                    "STOP",
                ]
            else:
                out += [
                    rng.choice(_TELEGRAPH_NOUNS),
                    rng.choice(_TELEGRAPH_VERBS),
                    _numword(rng.randint(1, 99)),
                    "STOP",
                ]
    elif register == "numeric":
        for _ in range(sentences):
            out.append(_numword(rng.randint(1, 99)))
    else:
        raise ValueError(f"unknown register {register!r}; expected one of {REGISTERS}")
    return "".join(out)


#: A public-domain-style English prose sample (original text, not puzzle material)
#: used to train the default prose model. Deliberately varied ordinary vocabulary so
#: the trigram statistics reflect natural narrative English rather than any one topic.
PROSE_SAMPLE = (
    "The morning came in slowly over the quiet town and the light spread across the "
    "rooftops until the whole street was awake. People opened their doors and stepped "
    "out into the cool air, greeting one another as they walked toward the market where "
    "the bakers had already set out their bread. A gentle wind moved through the trees "
    "and carried the smell of coffee from the little shop on the corner. Children ran "
    "along the pavement on their way to school, laughing at some joke that only they "
    "understood, while their parents followed behind carrying bags and talking about the "
    "day ahead. In the square an old man sat on his usual bench and watched the birds "
    "gather around his feet, and he thought about all the years he had spent in this "
    "place and how little it had really changed. The river ran along the edge of the "
    "town, calm and steady, reflecting the pale sky above it. Boats drifted past with "
    "their sails half open, and the fishermen called to each other across the water. By "
    "the middle of the day the sun had climbed high and warm, and the shadows grew short "
    "beneath the walls. Someone was cooking in a kitchen nearby, and the sound of a "
    "spoon against a pot mixed with the voices of the neighbors. When the evening finally "
    "arrived the lamps came on one by one, and the town settled into a soft and familiar "
    "quiet that had held it together for a very long time. The stars appeared above the "
    "hills, and a dog barked once in the distance before the whole valley grew still and "
    "waited patiently for another ordinary and welcome morning to begin again at last."
)


@functools.cache
def default_models() -> tuple[GenreModel, GenreModel]:
    """Return the cached ``(route_model, prose_model)`` pair of default models.

    The route model is trained on :func:`route_corpus`; the prose model on the
    bundled :data:`PROSE_SAMPLE`. Both are trigram models. Cached so repeated
    calls (and :func:`nonprose_flag` with default models) share one build.
    """
    route_model = GenreModel.train(route_corpus())
    prose_model = GenreModel.train(PROSE_SAMPLE)
    return route_model, prose_model


def nonprose_flag(
    text: str,
    *,
    route_model: GenreModel | None = None,
    prose_model: GenreModel | None = None,
) -> dict:
    """Compare ``text`` under a route model and a prose model, anchor-normalized.

    Returns a dict with the anchor-normalized fracs under each model, their
    difference, and a verdict:

    - ``route_score`` — frac (genre-typical 1.0 .. random 0.0) under the route model.
    - ``prose_score`` — frac under the prose model.
    - ``delta`` — ``route_score - prose_score`` (positive leans non-prose).
    - ``leans_nonprose`` — ``True`` when ``delta`` exceeds :data:`_LEAN_MARGIN`.
    - ``verdict`` — a short human-readable label.

    When a model is omitted the cached :func:`default_models` are used.
    """
    if route_model is None or prose_model is None:
        default_route, default_prose = default_models()
        route_model = route_model or default_route
        prose_model = prose_model or default_prose

    route_score = route_model.frac(text)
    prose_score = prose_model.frac(text)
    delta = route_score - prose_score
    leans_nonprose = delta > _LEAN_MARGIN
    if leans_nonprose:
        verdict = "leans non-prose (route/structured)"
    elif delta < -_LEAN_MARGIN:
        verdict = "reads as prose"
    else:
        verdict = "ambiguous"
    return {
        "route_score": route_score,
        "prose_score": prose_score,
        "delta": delta,
        "leans_nonprose": leans_nonprose,
        "verdict": verdict,
    }
