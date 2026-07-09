"""Statistical per-type cipher identifier — buttcrack's ``Identify Cipher``.

This is a *fast, non-cracking* classifier. Given ciphertext it computes a panel
of statistics (in the spirit of CryptoCrack's "Identify Cipher" / the ACA
reference-statistics tool) and routes the text to a ranked list of likely cipher
*types/families* with scores and human-readable reasons.

It never decrypts. It only looks at structure:

Features (CryptoCrack analogues, with their conventional multipliers):
- ``ioc``               index of coincidence x1000 (``ic``)
- ``max_period_ioc``    max average column IoC over periods 1-15 x1000 (``mic``)
  plus ``best_period`` — the smallest key length whose columns look English
- ``max_kappa``         max autocorrelation kappa over periods 1-15 x1000 (``mka``)
- ``digraphic_ioc``     digraphic IoC over adjacent overlapping pairs x10000 (``dic``)
- ``even_digraphic_ioc``digraphic IoC over boundary-aligned pairs x10000 (``edi``)
- ``log_digraph``       average log-digraph score vs English digraph frequencies
- ``repeat3_root``      sqrt(% of repeated trigrams) x1000 (``lr``)
- ``odd_repeat_pct``    % of repeats at odd spacing (``rod``)
- ``digit_fraction``    fraction of non-space chars that are digits
- ``alphabet_size``     number of distinct letter symbols used
- ``vowel_ratio``       vowels / letters
- doubled-letter and odd-length structure flags

Routed types: ``monoalphabetic``, ``polyalphabetic`` (annotated with period),
``transposition``, ``playfair`` (digraphic), ``bifid`` (fractionation),
``numeric`` (morse/nihilist/checkerboard families), ``homophonic``.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from .text import only_letters

# Reference IoC landmarks (x1000 in feature space; here as raw proportions).
ENGLISH_IOC = 0.0667
RANDOM_IOC = 0.0385

VOWELS = frozenset("AEIOU")

# Below this many letters the statistics are too noisy to route on.
MIN_RELIABLE = 24
# Below this many symbols of any kind we refuse to guess at all.
MIN_USABLE = 8

# Periods considered for the polyalphabetic / autocorrelation scans.
MAX_PERIOD = 15

# Below this many letters per mod-p coset, the per-coset IoC is dominated by small-sample
# inflation (few letters push even random cosets well above the 1/26 floor), so an elevated
# coset IoC there is UNRELIABLE — a "period" seen only at this granularity is likely an artifact
# that evaporates as the message lengthens (the trap that derailed a 150-hour effort: a
# "period-7" whose mod-7 cosets held ~6 letters each, gone by n=504).
SMALL_SAMPLE_COSET = 8


# --------------------------------------------------------------------------- #
# English digraph log-frequency model (for the average-log-digraph statistic).
# --------------------------------------------------------------------------- #
# A compact relative-frequency table of common English digraphs (counts per
# ~10000). Everything unlisted shares a small floor. This is enough to separate
# "reads like English digraphs" (monoalphabetic/transposition) from "doesn't"
# (polyalphabetic/fractionated), which is all the statistic is used for.
_ENGLISH_DIGRAPHS: dict[str, float] = {
    "TH": 271,
    "HE": 233,
    "IN": 203,
    "ER": 178,
    "AN": 161,
    "RE": 141,
    "ND": 135,
    "AT": 124,
    "ON": 123,
    "NT": 121,
    "HA": 112,
    "ES": 109,
    "ST": 109,
    "EN": 108,
    "ED": 107,
    "TO": 104,
    "IT": 103,
    "OU": 102,
    "EA": 100,
    "HI": 96,
    "IS": 93,
    "OR": 90,
    "TI": 89,
    "AS": 87,
    "TE": 85,
    "ET": 76,
    "NG": 73,
    "OF": 71,
    "AL": 69,
    "DE": 69,
    "SE": 66,
    "LE": 65,
    "SA": 62,
    "SI": 60,
    "AR": 59,
    "VE": 57,
    "RA": 56,
    "LD": 54,
    "UR": 54,
    "EL": 53,
    "ME": 52,
    "RO": 51,
    "NE": 49,
    "CH": 48,
    "WI": 48,
    "WH": 47,
    "TA": 47,
    "LL": 46,
    "CO": 45,
    "DT": 44,
    "RI": 43,
    "OW": 42,
    "HT": 42,
    "OT": 41,
    "EE": 39,
    "NI": 39,
    "PR": 39,
    "RT": 39,
    "MA": 38,
    "WA": 37,
    "GE": 36,
    "HO": 36,
    "TT": 35,
    "LO": 34,
    "OM": 34,
    "EC": 33,
    "OO": 33,
    "AC": 32,
    "EW": 32,
    "IO": 32,
    "NO": 32,
    "DI": 31,
    "EV": 31,
    "LI": 31,
    "FO": 30,
    "CA": 29,
    "HU": 29,
    "LA": 29,
    "PE": 29,
    "SO": 29,
    "FA": 28,
    "AI": 27,
    "BE": 27,
    "DA": 27,
    "DO": 27,
    "EI": 27,
    "GH": 27,
    "TR": 27,
}
_DIGRAPH_FLOOR = 1.0  # per ~10000, for any digraph not in the table


def _english_digraph_logs() -> dict[str, float]:
    total = sum(_ENGLISH_DIGRAPHS.values()) + _DIGRAPH_FLOOR * (676 - len(_ENGLISH_DIGRAPHS))
    floor_log = math.log10(_DIGRAPH_FLOOR / total)
    logs: dict[str, float] = {}
    for a in range(26):
        for b in range(26):
            gram = chr(65 + a) + chr(65 + b)
            count = _ENGLISH_DIGRAPHS.get(gram, _DIGRAPH_FLOOR)
            logs[gram] = math.log10(count / total)
    logs["__floor__"] = floor_log
    return logs


_DIGRAPH_LOGS = _english_digraph_logs()


def _english_log_digraph_reference() -> float:
    """Frequency-weighted mean log-digraph of clean English (a calibration landmark)."""
    total_weight = sum(_ENGLISH_DIGRAPHS.values()) + _DIGRAPH_FLOOR * (676 - len(_ENGLISH_DIGRAPHS))
    weighted = 0.0
    for a in range(26):
        for b in range(26):
            gram = chr(65 + a) + chr(65 + b)
            weight = _ENGLISH_DIGRAPHS.get(gram, _DIGRAPH_FLOOR)
            weighted += _DIGRAPH_LOGS[gram] * weight
    return weighted / total_weight


_ENGLISH_LOG_DIGRAPH = _english_log_digraph_reference()


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ioc(letters: str) -> float:
    n = len(letters)
    if n < 2:
        return 0.0
    counts = Counter(letters)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def _digraphic_ioc(letters: str, *, step: int) -> float:
    """IoC over digraphs. ``step=1`` overlapping (DIC); ``step=2`` boundary (EDI)."""
    pairs = [letters[i : i + 2] for i in range(0, len(letters) - 1, step)]
    n = len(pairs)
    if n < 2:
        return 0.0
    counts = Counter(pairs)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def _period_profile(letters: str, max_period: int = MAX_PERIOD) -> dict[int, float]:
    """Average per-column IoC for each candidate period 1..max_period.

    A polyalphabetic cipher of period ``p`` has English-like column IoC at ``p``
    (and its multiples); other periods sit near the random baseline.
    """
    profile: dict[int, float] = {}
    for p in range(1, max_period + 1):
        columns = [letters[i::p] for i in range(p)]
        iocs = [_ioc(col) for col in columns if len(col) >= 2]
        profile[p] = sum(iocs) / len(iocs) if iocs else 0.0
    return profile


def _best_period(profile: dict[int, float], base_ioc: float) -> tuple[int, float]:
    """Smallest period whose columns look clearly more English than period 1.

    Returns ``(period, max_profile_ioc)``. ``period`` is 1 when no period shows a
    meaningful jump (i.e. the text is not polyalphabetic).
    """
    max_ioc = max(profile.values()) if profile else base_ioc
    # A "spike" is a column IoC at least 40% of the way from the text's own
    # flat baseline up to English, and well above that baseline in absolute terms.
    target = base_ioc + 0.40 * max(0.0, ENGLISH_IOC - base_ioc)
    threshold = max(target, base_ioc + 0.010)
    for p in range(2, MAX_PERIOD + 1):
        if profile.get(p, 0.0) >= threshold and profile[p] >= base_ioc + 0.010:
            return p, max_ioc
    return 1, max_ioc


def _coset_mean_ioc(letters: str, period: int) -> float:
    """Mean per-coset (per-column) index of coincidence at ``period`` — the ``mic`` kernel."""
    cols = [letters[i::period] for i in range(period)]
    iocs = [_ioc(col) for col in cols if len(col) >= 2]
    return sum(iocs) / len(iocs) if iocs else 0.0


def period_significance(
    text: str,
    periods=None,
    *,
    alphabet: str | None = None,
    samples: int = 200,
    seed: int = 20250617,
    small_sample_letters: int = SMALL_SAMPLE_COSET,
) -> dict:
    """Length-aware significance of each candidate period's coset IoC.

    The per-column (coset) IoC at a period is the standard "is this the key length" signal, but
    it is **length-blind**: with only a handful of letters per coset even random text sits well
    above the ``1/26`` floor, so a long period on a short message shows an elevated coset IoC that
    is pure small-sample noise — the exact artifact that derailed a 150-hour effort (a "period-7"
    whose mod-7 cosets held ~6 letters each, which vanished once the message grew to n=504).

    This separates a *real* period from that artifact two independent ways, WITHOUT touching the
    existing feature panel:

    * ``letters_per_coset = n / p`` and ``small_sample_flag`` (``letters_per_coset <
      small_sample_letters``) — a structural reliability gate: below ~8 letters/coset the IoC is
      untrustworthy regardless of how elevated it looks.
    * ``z`` — the coset IoC vs a matched shuffle null of the SAME letters (shuffling destroys any
      period, so a genuine period's coset IoC sits far above its own shuffle band; a small-sample
      bump does not, because the shuffle is *equally* inflated by the thin cosets).

    ``alphabet`` is accepted for API symmetry but unused: the index of coincidence is invariant to
    any relabelling of the alphabet.

    Returns ``{p: (mean_coset_ioc, letters_per_coset, z, small_sample_flag)}``. A period is a
    *trustworthy* signal only when ``z`` is high AND ``small_sample_flag`` is ``False``.
    """
    letters = only_letters(text)
    n = len(letters)
    if periods is None:
        periods = range(2, MAX_PERIOD + 1)
    periods = list(periods)
    obs = {p: _coset_mean_ioc(letters, p) for p in periods}

    rng = random.Random(seed)
    pool = list(letters)
    null: dict[int, list[float]] = {p: [] for p in periods}
    for _ in range(samples):
        rng.shuffle(pool)
        shuffled = "".join(pool)
        for p in periods:
            null[p].append(_coset_mean_ioc(shuffled, p))

    result: dict[int, tuple] = {}
    for p in periods:
        lpc = n / p if p else 0.0
        vals = null[p]
        mu = sum(vals) / len(vals) if vals else 0.0
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0.0
        sd = sd or 1e-9
        z = (obs[p] - mu) / sd
        small = lpc < small_sample_letters
        result[p] = (round(obs[p], 5), round(lpc, 1), round(z, 2), small)
    return result


def _max_kappa(letters: str, max_period: int = MAX_PERIOD) -> tuple[float, int]:
    """Max autocorrelation kappa (coincidence rate at shift ``p``) over 1..max."""
    n = len(letters)
    best_kappa = 0.0
    best_period = 1
    for p in range(1, max_period + 1):
        if n - p < 1:
            break
        matches = sum(1 for i in range(n - p) if letters[i] == letters[i + p])
        kappa = matches / (n - p)
        if kappa > best_kappa:
            best_kappa = kappa
            best_period = p
    return best_kappa, best_period


def _repeat3_stats(letters: str) -> tuple[float, float]:
    """Trigram-repeat statistics: ``(sqrt(%repeated) x1000, %odd-spaced-repeats)``.

    The first mirrors CryptoCrack's 3-symbol-repeat measure (high for letter-
    preserving ciphers — monoalphabetic & transposition). The second is the
    fraction of repeat spacings that are odd, used as a transposition tell.
    """
    n = len(letters)
    windows = n - 2
    if windows < 2:
        return 0.0, 0.0
    positions: dict[str, list[int]] = {}
    for i in range(windows):
        positions.setdefault(letters[i : i + 3], []).append(i)
    repeated = 0
    odd_spaced = 0
    total_spacings = 0
    for occ in positions.values():
        if len(occ) < 2:
            continue
        repeated += len(occ)
        for j in range(len(occ) - 1):
            spacing = occ[j + 1] - occ[j]
            total_spacings += 1
            if spacing % 2 == 1:
                odd_spaced += 1
    repeat_pct = 100.0 * repeated / windows
    root = math.sqrt(repeat_pct) * 1000.0 / 100.0  # keep on a 0..~100 scale
    odd_pct = 100.0 * odd_spaced / total_spacings if total_spacings else 0.0
    return root, odd_pct


def _avg_log_digraph(letters: str) -> float:
    """Average log-frequency of adjacent digraphs against the English model."""
    windows = len(letters) - 1
    if windows < 1:
        return _DIGRAPH_LOGS["__floor__"]
    floor = _DIGRAPH_LOGS["__floor__"]
    total = 0.0
    for i in range(windows):
        total += _DIGRAPH_LOGS.get(letters[i : i + 2], floor)
    return total / windows


def _vowel_ratio(letters: str) -> float:
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch in VOWELS) / len(letters)


def _doubled_fraction(letters: str) -> float:
    """Fraction of adjacent letter pairs that are doubles (AA, LL, ...)."""
    pairs = len(letters) - 1
    if pairs < 1:
        return 0.0
    return sum(1 for i in range(pairs) if letters[i] == letters[i + 1]) / pairs


def compute_features(text: str) -> dict:
    """Compute the full statistical feature panel for ``text``."""
    upper = text.upper()
    letters = only_letters(text)
    n = len(letters)

    non_space = [ch for ch in upper if not ch.isspace()]
    digits = sum(1 for ch in non_space if ch.isdigit())
    digit_fraction = digits / len(non_space) if non_space else 0.0

    ioc = _ioc(letters)
    profile = _period_profile(letters) if n >= 2 else {}
    best_period, max_period_ioc = _best_period(profile, ioc) if profile else (1, ioc)
    # Length-aware reliability of the reported period: how many letters land in each mod-p coset,
    # and whether that is too few to trust the coset IoC (small-sample artifact). Cheap — no null.
    best_period_letters_per_coset = (n / best_period) if best_period >= 1 else 0.0
    best_period_small_sample = (
        best_period >= 2 and best_period_letters_per_coset < SMALL_SAMPLE_COSET
    )
    max_kappa, kappa_period = _max_kappa(letters) if n >= 2 else (0.0, 1)
    dic = _digraphic_ioc(letters, step=1)
    edi = _digraphic_ioc(letters, step=2)
    repeat3_root, odd_repeat_pct = _repeat3_stats(letters)
    log_digraph = _avg_log_digraph(letters)
    alphabet_size = len(set(letters))
    vowel_ratio = _vowel_ratio(letters)
    doubled_fraction = _doubled_fraction(letters)

    return {
        "length": n,
        "non_space_length": len(non_space),
        # Conventional CryptoCrack multipliers are applied to the *_x fields so a
        # reader sees familiar magnitudes; raw proportions are kept too.
        "ioc": round(ioc, 5),
        "ioc_x1000": round(ioc * 1000, 1),
        "max_period_ioc": round(max_period_ioc, 5),
        "max_period_ioc_x1000": round(max_period_ioc * 1000, 1),
        "best_period": best_period,
        "best_period_letters_per_coset": round(best_period_letters_per_coset, 1),
        "best_period_small_sample": best_period_small_sample,
        "max_kappa": round(max_kappa, 5),
        "max_kappa_x1000": round(max_kappa * 1000, 1),
        "kappa_period": kappa_period,
        "digraphic_ioc": round(dic, 6),
        "digraphic_ioc_x10000": round(dic * 10000, 1),
        "even_digraphic_ioc": round(edi, 6),
        "even_digraphic_ioc_x10000": round(edi * 10000, 1),
        "log_digraph": round(log_digraph, 4),
        "english_log_digraph": round(_ENGLISH_LOG_DIGRAPH, 4),
        "repeat3_root": round(repeat3_root, 2),
        "odd_repeat_pct": round(odd_repeat_pct, 1),
        "digit_fraction": round(digit_fraction, 4),
        "alphabet_size": alphabet_size,
        "vowel_ratio": round(vowel_ratio, 4),
        "doubled_fraction": round(doubled_fraction, 4),
        "even_length": n % 2 == 0,
    }


def _english_likeness(ioc: float) -> float:
    """0 (random/poly) .. 1 (English-preserving mono/transposition)."""
    return _clamp((ioc - RANDOM_IOC) / (ENGLISH_IOC - RANDOM_IOC))


def _score_types(f: dict) -> list[dict]:
    """Turn the feature panel into a ranked list of typed hypotheses."""
    ioc = f["ioc"]
    dic = f["digraphic_ioc"]
    edi = f["even_digraphic_ioc"]
    best_period = f["best_period"]
    eng = _english_likeness(ioc)

    # The digraphic IoC (DIC) is the key mono-vs-transposition-vs-poly signal and
    # is INVARIANT to monoalphabetic substitution (it counts repeat *structure*,
    # not digraph identity). Empirically (x10000): monoalphabetic ~100+,
    # transposition ~55-70, polyalphabetic/fractionated ~22-35.
    #   dic_struct: 1 at full English digraph structure, 0 at the poly/random floor.
    DIC_FLOOR = 0.0035  # ~35 x10000: polyalphabetic baseline
    DIC_ENGLISH = 0.0100  # ~100 x10000: monoalphabetic English structure
    dic_struct = _clamp((dic - DIC_FLOOR) / (DIC_ENGLISH - DIC_FLOOR))
    # Average-log-digraph against English: only PLAINTEXT-ordered, unsubstituted
    # text scores high, so this is a weak corroborator for true-plaintext-like
    # inputs, not a discriminator after substitution.
    log_eng = _clamp(
        (f["log_digraph"] - _DIGRAPH_LOGS["__floor__"])
        / (_ENGLISH_LOG_DIGRAPH - _DIGRAPH_LOGS["__floor__"])
    )

    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def add(t: str, s: float, why: str) -> None:
        scores[t] = scores.get(t, 0.0) + s
        reasons.setdefault(t, []).append(why)

    # ---- numeric families (digits / Morse-style) ------------------------ #
    if f["digit_fraction"] > 0.5:
        add("numeric", 1.5, f"digits dominate symbols ({f['digit_fraction']:.0%})")
    elif f["digit_fraction"] > 0.15:
        add("numeric", 0.6, f"substantial digit content ({f['digit_fraction']:.0%})")

    # Tiny alphabet over many symbols => fractionation written as letters
    # (e.g. ADFGX/morse-as-letters/checkerboard rendered alphabetically).
    if 0 < f["alphabet_size"] <= 6 and f["length"] >= 20:
        add("numeric", 1.2, f"only {f['alphabet_size']} distinct symbols (fractionation alphabet)")

    # ---- monoalphabetic -------------------------------------------------- #
    # High single-letter IoC AND preserved digraph repeat-structure (high DIC).
    # A log-digraph bonus rewards genuine plaintext-ordered text but is small so
    # substituted (caesar/keyword) text still ranks monoalphabetic on DIC alone.
    mono = eng * dic_struct
    if mono > 0:
        add(
            "monoalphabetic",
            1.7 * mono + 0.5 * eng * dic_struct * log_eng,
            f"English-level IoC ({f['ioc_x1000']:.0f}) with intact digraph "
            f"structure (DIC {f['digraphic_ioc_x10000']:.0f})",
        )

    # ---- transposition --------------------------------------------------- #
    # High single-letter IoC (letters preserved) but digraph structure broken:
    # DIC sits between the monoalphabetic and polyalphabetic levels.
    transpo = eng * _clamp(1.0 - dic_struct)
    if transpo > 0:
        add(
            "transposition",
            1.9 * transpo,
            f"English-level IoC ({f['ioc_x1000']:.0f}) but disrupted digraphs "
            f"(DIC {f['digraphic_ioc_x10000']:.0f} below the monoalphabetic level)",
        )

    # ---- polyalphabetic -------------------------------------------------- #
    poly = 1.0 - eng  # flattened single-letter distribution
    if best_period >= 2:
        add(
            "polyalphabetic",
            1.4 * max(poly, 0.5),
            f"flat IoC ({f['ioc_x1000']:.0f}) with period-{best_period} column structure "
            f"(col IoC {f['max_period_ioc_x1000']:.0f})",
        )
    elif poly > 0.4 and f["digit_fraction"] < 0.15 and f["alphabet_size"] > 6:
        # Flat distribution, no detected period: poly with long/odd key,
        # or a digraphic/fractionating cipher (handled below too).
        add(
            "polyalphabetic",
            0.9 * poly,
            f"flat IoC ({f['ioc_x1000']:.0f}) with no English structure",
        )

    # ---- playfair / digraphic ------------------------------------------- #
    # Playfair: no doubled letters, even length, and boundary-aligned digraphs
    # (EDI) repeat much more than the overlapping ones (DIC).
    if f["alphabet_size"] > 6 and f["length"] >= 20:
        edi_excess = edi - dic
        digraphic = 0.0
        if f["doubled_fraction"] < 0.005:
            digraphic += 0.5
        if edi_excess > 0.0008:
            digraphic += _clamp(edi_excess / 0.004) * 1.0
        if f["even_length"]:
            digraphic += 0.15
        if digraphic > 0:
            add(
                "playfair",
                digraphic,
                f"no doubled letters, even-aligned digraph IoC "
                f"(EDI {f['even_digraphic_ioc_x10000']:.0f}) exceeds overlapping "
                f"(DIC {f['digraphic_ioc_x10000']:.0f})",
            )

    # ---- bifid / fractionation ------------------------------------------ #
    # Flattened single AND digraph IoC (both DIC and EDI low), letters only.
    if f["alphabet_size"] > 6 and f["length"] >= 20:
        flat_di = _clamp(1.0 - dic_struct)
        flat_single = poly
        bifid = flat_single * flat_di
        # Distinguish from poly: no usable period signal but digraphs equally dead.
        if best_period < 2 and edi - dic < 0.0008:
            bifid += 0.4
        if bifid > 0:
            add(
                "bifid",
                1.1 * bifid,
                f"both single ({f['ioc_x1000']:.0f}) and digraph "
                f"({f['digraphic_ioc_x10000']:.0f}) IoC flattened (fractionation)",
            )

    # ---- homophonic ------------------------------------------------------ #
    # Very flat single-letter distribution over a large alphabet / many symbols,
    # often with digits. Lower priority unless the alphabet is unusually large.
    if f["alphabet_size"] > 26 or (f["digit_fraction"] > 0.0 and f["alphabet_size"] > 20):
        add(
            "homophonic",
            0.8,
            f"oversized symbol set ({f['alphabet_size']} symbols) with flat frequencies",
        )

    # Sort on the float score before building dicts (keeps the comparison off the
    # heterogeneously-typed result dicts).
    ordered = sorted(
        ((t, s) for t, s in scores.items() if s > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    ranked: list[dict] = []
    for type_name, raw_score in ordered:
        entry: dict = {
            "type": type_name,
            "score": round(raw_score, 3),
            "reasons": reasons[type_name],
        }
        # Annotate the polyalphabetic entry with its period for convenience.
        if type_name == "polyalphabetic" and best_period >= 2:
            entry["period"] = best_period
        ranked.append(entry)
    return ranked


def identify_types(text: str) -> dict:
    """Statistically identify likely cipher types/families for ``text``.

    Returns ``{"features": {...}, "ranked": [{"type", "score", "reasons"}, ...]}``
    sorted best-first. ``ranked`` is a single ``undetermined`` entry when the text
    is empty / too short / digit-only-but-too-sparse to classify.
    """
    features = compute_features(text)
    n = features["length"]
    non_space = features["non_space_length"]

    if non_space < MIN_USABLE:
        return {
            "features": features,
            "ranked": [
                {
                    "type": "undetermined",
                    "score": 0.0,
                    "reasons": ["too few symbols to compute reliable statistics"],
                }
            ],
        }

    # Pure-digit / near-pure-digit numeric ciphers: route straight to numeric
    # even when there aren't enough letters for the letter-based statistics.
    if n < MIN_RELIABLE:
        if features["digit_fraction"] > 0.5:
            return {
                "features": features,
                "ranked": [
                    {
                        "type": "numeric",
                        "score": 1.0,
                        "reasons": [
                            f"digit-dominated ({features['digit_fraction']:.0%}) "
                            "numeric cipher (morse/nihilist/checkerboard family)"
                        ],
                    }
                ],
            }
        return {
            "features": features,
            "ranked": [
                {
                    "type": "undetermined",
                    "score": 0.0,
                    "reasons": [f"only {n} letters; below the reliable threshold ({MIN_RELIABLE})"],
                }
            ],
        }

    ranked = _score_types(features)
    if not ranked:
        ranked = [
            {
                "type": "undetermined",
                "score": 0.0,
                "reasons": ["no family scored above zero"],
            }
        ]
    return {"features": features, "ranked": ranked}
