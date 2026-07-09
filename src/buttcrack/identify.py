"""Heuristic cipher-family identification — a routing hint for ``auto``.

This never decrypts; it inspects statistics (index of coincidence, letter-
frequency fit) to rank likely cipher families. ``auto`` still verifies by
actually cracking, so these weights only influence ordering/annotation.
"""

from __future__ import annotations

from .scoring import chi_squared, index_of_coincidence
from .text import only_letters

ENGLISH_IOC = 0.0667
RANDOM_IOC = 0.0385

FAMILIES = {
    "transposition": ["railfence", "columnar"],
    "monoalphabetic": ["caesar", "atbash", "affine", "substitution"],
    "polyalphabetic": ["vigenere"],
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# Below this many letters, IoC/chi-squared are too noisy to trust.
MIN_RELIABLE = 20


def identify(text: str) -> dict:
    """Return IoC/fit statistics plus a ranked list of likely families."""
    letters = only_letters(text)
    n = len(letters)

    # Too few letters to compute meaningful statistics: say so, don't guess.
    if n < 2:
        return {
            "length": n,
            "index_of_coincidence": None,
            "chi_squared_per_letter": None,
            "reliable": False,
            "note": "insufficient letters to identify",
            "likely_families": [{"family": "undetermined", "weight": 1.0, "ciphers": []}],
        }

    ioc = index_of_coincidence(letters)
    chi2 = chi_squared(letters)
    chi2_per_letter = chi2 / n if n else float("inf")

    # How "English-like" the IoC is (1 = monoalphabetic/transposition, 0 = random/poly).
    eng_ioc = _clamp((ioc - RANDOM_IOC) / (ENGLISH_IOC - RANDOM_IOC))

    # Low chi-squared => letters already follow English frequencies => order was
    # scrambled but letters preserved => transposition. High => letters remapped
    # => monoalphabetic substitution.
    transposition_affinity = _clamp(1.0 - chi2_per_letter / 0.6)

    weights = {
        "transposition": eng_ioc * transposition_affinity,
        "monoalphabetic": eng_ioc * (1.0 - transposition_affinity),
        "polyalphabetic": (1.0 - eng_ioc),
    }
    total = sum(weights.values()) or 1.0
    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [
        {"family": fam, "weight": round(w / total, 3), "ciphers": FAMILIES[fam]}
        for fam, w in ordered
    ]

    periodic, diagnosis = _diagnose(letters, n, str(ranked[0]["family"]))

    return {
        "length": n,
        "index_of_coincidence": round(ioc, 4),
        "chi_squared_per_letter": round(chi2_per_letter, 4) if n else None,
        "reliable": n >= MIN_RELIABLE,
        "likely_families": ranked,
        # Calibrated period spectrum (polyalphabetic only) + a plain-language read.
        "periodic_ioc": periodic,
        "diagnosis": diagnosis,
    }


def _diagnose(letters: str, n: int, top_family: str) -> tuple[list[dict], str]:
    """A calibrated period spectrum + a plain-language diagnosis with caveats.

    The point is to keep callers honest where naive analysis goes wrong: a long key
    on a short message has no *obvious* period and flat IoC, which is easy to
    misread as "running key / aperiodic" — so when nothing reads, say what to try
    next (long periods, crib-drag) rather than implying a dead end.
    """
    if top_family != "polyalphabetic" or n < 48:
        return [], ""
    from .analysis import calibrated_periods, ioc_decay

    spectrum = calibrated_periods(letters, top=4)
    significant = [p for p in spectrum if p["z"] >= 3.0]
    # A genuine periodic cipher is STATIONARY; an evolving keystream is not. Compute the
    # IoC drift first so a non-stationary keystream is never mislabelled "clean periodic"
    # on the strength of a marginal/spurious period spike.
    decay = ioc_decay(letters)
    nonstat = bool(decay.get("non_stationary"))

    if significant and not nonstat:
        best = significant[0]
        lpc = n // best["period"]
        msg = (
            f"polyalphabetic; significant period {best['period']} "
            f"(z={best['z']}), ~{lpc} letters/column"
        )
        if lpc < 12:
            msg += " — short columns (long key); recovery is uncertain, try a crib"
        msg += _layered_hint(letters, best["period"])
        return spectrum, msg

    # Either no period, or a period plus a real IoC drift. Distinguish the cases that
    # look alike but need different attacks — the trap that wastes the most time:
    #   (a) IoC drifts down the message -> NON-STATIONARY keystream (period not
    #       recoverable; a crib is the lever) — even if a marginal period shows;
    #   (b) stationary, no period -> a periodic substitution may be hidden UNDER a
    #       transposition (its period is scrambled) -> try `transsub`;
    #   (c) otherwise a long/aperiodic key -> crib-drag.
    if nonstat:
        spurious = (
            f" (a period {significant[0]['period']} shows at z={significant[0]['z']} but is "
            "likely spurious given the drift)"
            if significant
            else ""
        )
        msg = (
            f"flat IoC, IoC drifts down the message (slope_z={decay['slope_z']}): a "
            "NON-STATIONARY / evolving keystream (progressive-key, autokey, chain-addition/"
            "Gromark, or a dynamic alphabet like Chaocipher) — the period is not recoverable "
            "and periodic/transposition attacks will fail; a crib is the lever." + spurious
        )
        return spectrum, msg
    msg = (
        "flat IoC but no significant period up to ~length/5 — a long/aperiodic key, "
        "OR a periodic substitution hidden UNDER a transposition (the period is "
        "scrambled, so it can't show in the raw text). Try `transsub` (transposition-"
        "over-substitution) and a crib-drag (`butt crib`) before concluding running-key."
    )
    return spectrum, msg


def _layered_hint(letters: str, period: int) -> str:
    """Flag an *additive* substitution layered over a transposition (Nicodemus-like).

    Additively de-substitute each period column (best Caesar shift by chi-squared).
    If the result reaches English *letter frequencies* (IoC ~0.066) yet isn't
    readable English, the de-substituted text is scrambled — a transposition lurks
    underneath. This is the reliable layered signature; a *keyed* substitution over
    a transposition can't be told from a plain keyed one this cheaply, so we don't
    guess there (a confident-but-wrong "layered" label would mislead).
    """
    from .scoring import ENGLISH_MONOGRAM_FREQ, get_scorer, index_of_coincidence

    def chi(col: str, s: int) -> float:
        counts = [0] * 26
        for ch in col:
            counts[(ord(ch) - 65 - s) % 26] += 1
        return sum(
            (counts[i] - ENGLISH_MONOGRAM_FREQ[chr(65 + i)] * len(col)) ** 2
            / (ENGLISH_MONOGRAM_FREQ[chr(65 + i)] * len(col) or 1)
            for i in range(26)
        )

    def best_shift(col: str) -> int:
        return min(range(26), key=lambda s: chi(col, s))

    shifts = [best_shift(letters[j::period]) for j in range(period)]
    desub = "".join(
        chr((ord(c) - 65 - shifts[i % period]) % 26 + 65) for i, c in enumerate(letters)
    )
    scorer = get_scorer()
    readable = scorer.average(desub) >= scorer._english_ref - 1.2
    if index_of_coincidence(desub) >= 0.060 and not readable:
        return (
            " — de-substitutes to English letter frequencies but not readable text: "
            "likely an (additive) substitution OVER a transposition; de-substitute, "
            "then attack the inner transposition"
        )
    return ""


def ordered_ciphers(text: str) -> list[str]:
    """Cipher names ordered by the identify heuristic (highest-weight family first)."""
    info = identify(text)
    order: list[str] = []
    for fam in info["likely_families"]:
        for name in fam["ciphers"]:
            if name not in order:
                order.append(name)
    return order
