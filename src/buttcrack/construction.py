"""Forward-simulation model identification — which construction could manufacture this?

Discriminant cipher-ID (:mod:`buttcrack.cipher_id`, :mod:`buttcrack.identify`) reads a
ciphertext's statistics and votes a family. This module works the other way round —
*generatively*. It simulates many instances of each candidate construction (a cipher
family encrypting fresh English under random keys), measures a multi-statistic **dial
vector** on each, and then asks of the real ciphertext: *which families could have
produced these dials, and which provably could not?*

The payoff is refutation. If every simulated Vigenere in 200 tries lands its monogram
IoC in ``0.038–0.046`` and the real ciphertext sits at ``0.066``, no Vigenere key
manufactures that dial — the family is refuted, whatever a discriminant might guess.
Families that *can* reproduce the observed dials are ranked by how typical the
observation is of them (RMS z across dials), and the "joint hit" rate reports how often
a single simulated instance reproduces the whole fingerprint at once.

* :func:`dial_vector` — the cipher-agnostic fingerprint (monogram/digraph IoC, letter
  entropy, coset-IoC at a few periods).
* :func:`default_generators` — a spread of built-in family simulators (language,
  periodic polyalphabetic, transposition, monoalphabetic, digraphic, running-key,
  keystream) built on the cipher registry; pass your own to extend the bank.
* :func:`construction_baseline` — simulate, fit, rank, and refute.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from . import registry
from .scoring import index_of_coincidence, letter_entropy
from .text import ALPHABET, only_letters

#: A family simulator: encrypt ``plaintext`` under a fresh random key, return ciphertext.
Generator = Callable[[str, random.Random], str]

#: Coset periods probed by the default dial vector (harmonics of common key lengths).
_COSET_PERIODS = (2, 3, 5, 7)

#: English corpus (repeatable) that simulations draw fresh plaintext windows from.
_CORPUS = (
    "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLEINHERLEDGER"
    "WHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHTOFTHERISINGSUNOUTSIDEAS"
    "ABROADRIVERWOUNDPASTTHEOLDSTONEBRIDGEWHEREFARMERSCARRIEDBASKETSOFFRESHFRUITTOTOWNAND"
    "CHILDRENPLAYEDALONGTHEGRASSYBANKSLAUGHINGASTHEYCHASEDONEANOTHERTHROUGHTHEOPENFIELDS"
    "UNTILTHEDISTANTBELLOFTHECHURCHCALLEDTHEMHOMEFORTHEEVENINGMEALANDALONGNIGHTOFQUIETREST"
    "THEANCIENTMAPSHOWEDAHIDDENPATHBETWEENTWOMOUNTAINSWHERENOTRAVELERHADWALKEDINMANYYEARS"
) * 2


def _ngram_ioc(letters: str, n: int) -> float:
    """Index of coincidence over length-``n`` grams (n=1 monogram, n=2 digraph)."""
    grams = [letters[i : i + n] for i in range(len(letters) - n + 1)]
    total = len(grams)
    if total < 2:
        return 0.0
    counts = Counter(grams)
    return sum(c * (c - 1) for c in counts.values()) / (total * (total - 1))


def _coset_ioc(letters: str, p: int) -> float:
    """Mean over the ``p`` cosets of the monogram IoC — elevated by a period-``p`` key."""
    total, used = 0.0, 0
    for r in range(p):
        col = letters[r::p]
        if len(col) >= 2:
            total += index_of_coincidence(col)
            used += 1
    return total / used if used else 0.0


def dial_vector(text: str) -> dict[str, float]:
    """The cipher-agnostic statistical fingerprint used to compare constructions.

    Returns monogram IoC (``ic``), digraph IoC (``dic``), letter entropy (``entropy``),
    and coset IoC at each of :data:`_COSET_PERIODS` (``cic{p}``). Monogram IoC separates
    polyalphabetic (low) from monoalphabetic/transposition (English-like); digraph IoC
    flags digraphic ciphers; the coset dials expose periodic keys and their harmonics.
    """
    letters = only_letters(text)
    dials = {
        "ic": _ngram_ioc(letters, 1),
        "dic": _ngram_ioc(letters, 2),
        "entropy": letter_entropy(letters),
    }
    for p in _COSET_PERIODS:
        dials[f"cic{p}"] = _coset_ioc(letters, p)
    return dials


# --- built-in family simulators ----------------------------------------------


def _corpus_window(length: int, rng: random.Random) -> str:
    """A fresh ``length``-letter English window (wraps the corpus if needed)."""
    corpus = _CORPUS
    while len(corpus) < length + 1:
        corpus += _CORPUS
    start = rng.randrange(len(corpus) - length)
    return corpus[start : start + length]


def _random_keyword(rng: random.Random, lo: int = 5, hi: int = 8) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(lo, hi)))


def _cipher_generator(name: str, key_fn: Callable[[random.Random], str]) -> Generator:
    """Wrap a registry cipher's ``encode`` with a random-key sampler into a Generator."""
    cipher = registry.get(name)

    def _gen(plaintext: str, rng: random.Random) -> str:
        for _ in range(6):  # retry a few bad random keys before giving up
            try:
                return cipher.encode(plaintext, key_fn(rng))
            except Exception:
                continue
        raise RuntimeError(f"generator {name!r} could not produce a sample")

    return _gen


def default_generators() -> dict[str, Generator]:
    """A default bank of family simulators spanning the main structural classes.

    ``english`` (the language baseline — monoalphabetic and transposition both preserve
    its dials), ``vigenere``/``beaufort`` (periodic polyalphabetic), ``columnar``
    (transposition), ``substitution`` (monoalphabetic), ``playfair`` (digraphic),
    ``gromark`` (running key), and ``keystream`` (self-generating key). Pass a superset/
    subset to :func:`construction_baseline` to tailor the bank.
    """

    def _sub_key(rng: random.Random) -> str:
        perm = list(ALPHABET)
        rng.shuffle(perm)
        return "".join(perm)

    def _gromark_key(rng: random.Random) -> str:
        return "".join(str(rng.randrange(10)) for _ in range(5)) + "/" + _random_keyword(rng)

    def _keystream_key(rng: random.Random) -> str:
        c = f"{rng.randrange(26)},{rng.randrange(26)}"
        s = f"{rng.randrange(26)},{rng.randrange(26)}"
        return f"{c}/{s}"

    return {
        "english": lambda pt, rng: pt,
        "vigenere": _cipher_generator("vigenere", _random_keyword),
        "beaufort": _cipher_generator("beaufort", _random_keyword),
        "columnar": _cipher_generator("columnar", _random_keyword),
        "substitution": _cipher_generator("substitution", _sub_key),
        "playfair": _cipher_generator("playfair", _random_keyword),
        "gromark": _cipher_generator("gromark", _gromark_key),
        "keystream": _cipher_generator("keystream", _keystream_key),
    }


# --- the baseline report -----------------------------------------------------


@dataclass
class FamilyFit:
    """How well one construction family reproduces the observed dial vector."""

    name: str
    rms_z: float  # RMS of per-dial z-scores (lower = more typical of this family)
    max_abs_z: float
    worst_dial: str
    joint_hit: float  # fraction of sims reproducing ALL dials within band
    per_dial: dict[str, dict[str, float]]  # dial -> {obs, mean, sd, z}
    verdict: str  # "plausible" or "refuted"
    n_sims: int

    def summary(self) -> str:
        return (
            f"{self.name:14s} {self.verdict:9s} rms_z={self.rms_z:5.2f} "
            f"max|z|={self.max_abs_z:5.2f}@{self.worst_dial} joint_hit={self.joint_hit:.2f}"
        )


@dataclass
class ConstructionReport:
    """Ranked forward-simulation fit of a ciphertext to a bank of construction families."""

    observed: dict[str, float]
    families: list[FamilyFit]  # best structural fit first
    length: int = 0
    _by_name: dict[str, FamilyFit] = field(default_factory=dict, repr=False)

    @property
    def best(self) -> FamilyFit | None:
        return self.families[0] if self.families else None

    @property
    def plausible(self) -> list[FamilyFit]:
        return [f for f in self.families if f.verdict == "plausible"]

    @property
    def refuted(self) -> list[FamilyFit]:
        return [f for f in self.families if f.verdict == "refuted"]

    def get(self, name: str) -> FamilyFit | None:
        return self._by_name.get(name)

    def summary(self) -> str:
        head = "observed dials: " + ", ".join(f"{k}={v:.4f}" for k, v in self.observed.items())
        return head + "\n" + "\n".join(f.summary() for f in self.families)


def construction_baseline(
    ciphertext: str,
    generators: dict[str, Generator] | None = None,
    *,
    n_sims: int = 200,
    rng: random.Random | None = None,
    refute_z: float = 4.0,
    band_sigma: float = 1.5,
) -> ConstructionReport:
    """Rank candidate constructions by whether they can manufacture the observed dials.

    Simulates ``n_sims`` instances of each family in ``generators`` (default
    :func:`default_generators`) at the ciphertext's length, measures a
    :func:`dial_vector` on each, and compares the real ciphertext's dials against every
    family's simulated distribution. A family is **refuted** when the observation is more
    than ``refute_z`` standard deviations off on any dial (no key in the family produced
    anything like it); survivors are ranked by RMS z across dials (lower = the observation
    is more typical of that family). ``joint_hit`` is the fraction of a family's sims that
    land within ``band_sigma`` of the observation on *every* dial at once.
    """
    rng = rng or random.Random()
    gens = generators if generators is not None else default_generators()
    letters = only_letters(ciphertext)
    length = len(letters)
    observed = dial_vector(letters)
    dial_names = list(observed)

    fits: list[FamilyFit] = []
    for name, gen in gens.items():
        sims: list[dict[str, float]] = []
        for _ in range(n_sims):
            pt = _corpus_window(length, rng)
            try:
                ct = gen(pt, rng)
            except Exception:
                continue
            sims.append(dial_vector(ct))
        if not sims:
            continue
        per_dial: dict[str, dict[str, float]] = {}
        z_by_dial: dict[str, float] = {}
        for d in dial_names:
            vals = [s[d] for s in sims]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            sd = math.sqrt(var)
            obs = observed[d]
            if sd > 0:
                z = (obs - mean) / sd
            else:
                z = 0.0 if obs == mean else math.copysign(refute_z + 1.0, obs - mean)
            per_dial[d] = {"obs": obs, "mean": mean, "sd": sd, "z": z}
            z_by_dial[d] = z
        # joint hit: a sim reproduces the observation if within band on every dial.
        band = {d: band_sigma * per_dial[d]["sd"] for d in dial_names}
        hits = 0
        for s in sims:
            if all(abs(s[d] - observed[d]) <= band[d] for d in dial_names if band[d] > 0):
                hits += 1
        joint_hit = hits / len(sims)
        abs_zs = {d: abs(z) for d, z in z_by_dial.items()}
        worst_dial = max(abs_zs, key=lambda d: abs_zs[d])
        max_abs_z = abs_zs[worst_dial]
        rms_z = math.sqrt(sum(z * z for z in z_by_dial.values()) / len(z_by_dial))
        verdict = "refuted" if max_abs_z > refute_z else "plausible"
        fits.append(
            FamilyFit(
                name=name,
                rms_z=rms_z,
                max_abs_z=max_abs_z,
                worst_dial=worst_dial,
                joint_hit=joint_hit,
                per_dial=per_dial,
                verdict=verdict,
                n_sims=len(sims),
            )
        )
    fits.sort(key=lambda f: f.rms_z)
    return ConstructionReport(
        observed=observed,
        families=fits,
        length=length,
        _by_name={f.name: f for f in fits},
    )
