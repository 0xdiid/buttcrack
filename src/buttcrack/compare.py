"""Sibling-pair analysis: do two ciphertexts plausibly share a cipher construction?

In an "each puzzle builds on the last" series, two *unsolved* ciphertexts may be built the
same way — same period, same alphabet family, even the same keystream. Every other tool in
``butt`` reasons about a single ciphertext; :func:`compare` is the two-ciphertext companion.
It runs a battery of position-independent and position-dependent statistics on the PAIR and
returns a transparent, rule-based verdict about whether they were likely built the same way.

The evidence it weighs:

* **sorted letter-frequency profiles** — the L1 distance between the two texts' descending
  frequency profiles (permutation-invariant *shape*), compared to each text's distance from
  English. Two ciphertexts flattened/peaked to the SAME degree sit closer to each other than
  either does to English — the fingerprint of a shared cipher class / amount of flattening.
* **mutual index of coincidence** — same-alphabet coincidence between the two texts.
* **kappa (Friedman autocorrelation) per text** — do both wind at the same period (both
  polyalphabetic at period ``p``), or is one wound and the other flat (the SAME substitution,
  applied with a different winding)?
* **two-ciphertext superimposition / depth battery** — match-rate and difference-IoC over
  every cyclic rotation of one text against the other, the rotations themselves as the null.
  A spike at one rotation is the signature of a SHARED per-position keystream/tableau (an
  additive-translate depth). Ported to pure stdlib from a numpy shared-keystream test.

The verdict is deliberately conservative: it reports ``insufficient`` unless the frequency
profiles agree AND a period/kappa relationship (or an outright shared keystream) corroborates
it — it would rather under-claim than assert a shared construction from a single weak signal.
"""

from __future__ import annotations

from collections import Counter

from .analysis import (
    kappa_spectrum,
    mutual_index_of_coincidence,
    mutual_kappa_scan,
)
from .scoring import ENGLISH_MONOGRAM_FREQ, index_of_coincidence
from .text import only_letters

#: kappa z at which a text is considered to carry a periodic re-alignment at all.
KAPPA_ALIVE_Z = 3.0
#: how far the strongest kappa lag must stand above the median lag to read as a PEAK
#: (a monoalphabetic text has high kappa at *every* lag — flat, no winding — so absolute
#: z is not enough; a genuine periodic winding shows contrast).
KAPPA_PEAK_MARGIN = 2.5
#: the pair's inter-profile L1 must be under this fraction of each text's English L1 to count
#: as "closer to each other than to English".
PROFILE_RATIO = 0.6
#: absolute L1 gap required on top of the ratio, so two near-English texts (tiny L1s) do not
#: trip the ratio on noise alone.
PROFILE_MARGIN = 0.04
#: depth-battery z at which a shared per-position keystream is declared.
SHARED_KS_Z = 4.0

#: English letter frequencies, sorted descending — the reference profile shape.
_ENGLISH_PROFILE = sorted(ENGLISH_MONOGRAM_FREQ.values(), reverse=True)


def _sorted_freq_profile(letters: str) -> list[float]:
    """The 26 letter proportions of ``letters``, sorted descending (permutation-invariant)."""
    n = len(letters)
    if n == 0:
        return [0.0] * 26
    counts = Counter(letters)
    return sorted((counts.get(chr(65 + i), 0) / n for i in range(26)), reverse=True)


def _l1(p: list[float], q: list[float]) -> float:
    """L1 (sum of absolute differences) between two equal-length profiles."""
    return sum(abs(a - b) for a, b in zip(p, q, strict=True))


def _mean_std(xs: list[float]) -> tuple[float, float]:
    """Population mean and standard deviation of ``xs`` (matches the ported numpy null)."""
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, sd


def _median(xs: list[float]) -> float:
    """Median of ``xs`` (0.0 for the empty list)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _kappa_report(letters: str, max_period: int) -> dict:
    """Per-text kappa (Friedman autocorrelation) winding summary up to ``max_period``.

    Reuses :func:`buttcrack.analysis.kappa_spectrum` (kappa z per lag vs the random floor) and
    reduces it to a winding verdict: the strongest lag, whether it stands out as a PEAK above
    the median lag (contrast), the fundamental period behind the peak, and an ``alive`` flag.
    ``alive`` distinguishes a *periodically wound* text (peaked spectrum: near-floor at most
    lags, high at multiples of the period) from a monoalphabetic one (flat-high spectrum, no
    peak) and from noise (flat-low spectrum).
    """
    spectrum = kappa_spectrum(letters, max_lag=max_period)
    by_period = {row["lag"]: row["z"] for row in spectrum}
    per_period = [{"period": p, "z": by_period.get(p)} for p in range(1, max_period + 1)]
    zs = [row["z"] for row in spectrum]
    if not zs:
        return {
            "per_period": per_period,
            "strongest_period": None,
            "max_z": None,
            "contrast": None,
            "alive": False,
        }
    max_z = max(zs)
    contrast = max_z - _median(zs)
    # The fundamental period is the SMALLEST lag among the strong spikes (period-p winding
    # spikes at p, 2p, 3p, ...; we want p, not a harmonic that happened to score marginally
    # higher). "Strong" = clears the alive bar and stays within reach of the top spike.
    thresh = max(KAPPA_ALIVE_Z, 0.6 * max_z)
    spikes = sorted(row["lag"] for row in spectrum if row["z"] >= thresh)
    fundamental = spikes[0] if spikes else None
    # "Alive" means POLYALPHABETICALLY wound, not merely English-content. A monoalphabetic text
    # has kappa ~ its (English-level) IoC at EVERY lag, so its whole kappa spectrum floats above
    # the random floor and throws up noise peaks that fake a spike. Gate on a flattened IoC
    # (the same polyalphabetic threshold analyze() uses) so only a genuinely wound text — near
    # the floor at most lags, spiking at the period — reads as alive.
    flattened = index_of_coincidence(letters) < 0.058
    alive = bool(
        flattened and fundamental and max_z >= KAPPA_ALIVE_Z and contrast >= KAPPA_PEAK_MARGIN
    )
    return {
        "per_period": per_period,
        "strongest_period": fundamental,
        "max_z": round(max_z, 2),
        "contrast": round(contrast, 2),
        "alive": alive,
    }


def _depth_battery(a_idx: list[int], b_idx: list[int]) -> dict:
    """Match-rate + difference-IoC over all cyclic rotations of ``b`` against ``a``.

    Pure-stdlib port of the numpy ``depth_battery`` shared-keystream test. For every rotation
    ``d`` it measures (a) ``match`` — the fraction of positions where ``a[i] == b[i+d]`` — and
    (b) ``diffIC`` — the index of coincidence (x26) of the per-position difference
    ``(a[i] - b[i+d]) mod 26``. Under a SHARED per-position keystream the difference cancels the
    keystream and leaves plaintext-minus-plaintext (structured, not flat), so both statistics
    spike at the aligning rotation; every other rotation is the null. Per statistic it reports
    the rotation-0 value/z, its rank, and the best rotation's value/z.
    """
    n = len(a_idx)
    if n < 3:
        return {}
    denom = n * (n - 1)
    match: list[float] = []
    diffic: list[float] = []
    for d in range(n):
        hits = 0
        counts = [0] * 26
        for i in range(n):
            bj = b_idx[(i + d) % n]
            if a_idx[i] == bj:
                hits += 1
            counts[(a_idx[i] - bj) % 26] += 1
        match.append(hits / n)
        diffic.append(26.0 * sum(c * (c - 1) for c in counts) / denom)
    out: dict[str, dict] = {}
    for name, arr in (("match", match), ("diffIC", diffic)):
        rest = arr[1:] if len(arr) > 1 else arr
        mu, sd = _mean_std(rest)
        sd = sd or 1e-9
        amu, asd = _mean_std(arr)
        asd = asd or 1e-9
        mx = max(arr)
        out[name] = {
            "d0": round(arr[0], 4),
            "z0": round((arr[0] - mu) / sd, 2),
            "rank_of_d0": sum(1 for v in arr if v >= arr[0]),
            "best": round(mx, 4),
            "argmax": arr.index(mx),
            "scan_max_z": round((mx - amu) / asd, 2),
        }
    return out


def _shared_keystream_signal(battery: dict) -> tuple[bool, float | None]:
    """Decide whether the depth battery shows a shared per-position keystream.

    Fires on a decisive rotation-0 spike (``d0`` is the global max AND clears the z bar) — the
    same-phase aligned case — or on a best rotation that stands well clear of the rotation null.
    The rotation set is the null, which corrects for the scan-max multiplicity of trying every
    rotation (a raw cross-text kappa maximum does NOT, so it is reported but not trusted here).
    Returns ``(fired, best_z)``.
    """
    zs: list[float] = []
    for name in ("match", "diffIC"):
        st = battery.get(name)
        if not st:
            continue
        if st["rank_of_d0"] == 1 and st["z0"] >= SHARED_KS_Z:
            zs.append(st["z0"])
        if st["scan_max_z"] >= SHARED_KS_Z + 1.0:
            zs.append(st["scan_max_z"])
    return (len(zs) > 0), (round(max(zs), 2) if zs else None)


def compare(ct_a: str, ct_b: str, *, max_period: int = 16) -> dict:
    """Test whether two ciphertexts plausibly share a cipher construction, with the evidence.

    A sibling-pair analysis for an "each puzzle builds on the last" series: it does NOT try to
    crack either text, it asks whether the PAIR was built the same way. It combines a
    permutation-invariant frequency-shape comparison, per-text kappa winding, mutual IoC, and a
    two-ciphertext superimposition (depth battery) into a conservative, rule-based verdict.

    Returns a dict with:

    * ``len_a`` / ``len_b`` / ``ioc_a`` / ``ioc_b`` — length and index of coincidence per text.
    * ``freq_profile_l1`` — L1 distance between the two texts' descending frequency profiles;
      ``l1_a_english`` / ``l1_b_english`` — each text's L1 distance from the English profile
      (so "closer to each other than to English" is directly visible).
    * ``mutual_ioc`` — position-independent mutual IoC of the two texts.
    * ``kappa`` — per text (``a``/``b``) the kappa z at each period up to ``max_period`` with a
      per-text ``alive`` flag at its strongest period, plus ``same_strongest_period`` and
      ``one_alive_one_dead`` (the "do they differ in winding?" signals).
    * ``superimposition`` — the depth battery: ``match`` / ``diffIC`` rotation stats, the best
      rotation's match-rate (``best_match_rate``) and z (``best_match_z`` / ``best_diffic_z``),
      and a ``shared_keystream`` flag with its z.
    * ``cross_kappa`` — top offsets from :func:`mutual_kappa_scan` (cross-text depth at a shift).
    * ``verdict`` — ``{shared_construction: bool, confidence: str, evidence: [str, ...]}``.
    """
    a = only_letters(ct_a)
    b = only_letters(ct_b)
    len_a, len_b = len(a), len(b)
    ioc_a = round(index_of_coincidence(a), 4)
    ioc_b = round(index_of_coincidence(b), 4)

    # Frequency-shape comparison (permutation-invariant): are the two closer to each other than
    # either is to English? That is the fingerprint of a shared cipher class / flattening.
    prof_a = _sorted_freq_profile(a)
    prof_b = _sorted_freq_profile(b)
    freq_profile_l1 = round(_l1(prof_a, prof_b), 4)
    l1_a_english = round(_l1(prof_a, _ENGLISH_PROFILE), 4)
    l1_b_english = round(_l1(prof_b, _ENGLISH_PROFILE), 4)

    mutual_ioc = round(mutual_index_of_coincidence(a, b), 4)

    ka = _kappa_report(a, max_period)
    kb = _kappa_report(b, max_period)
    kappa = {
        "a": ka,
        "b": kb,
        "same_strongest_period": bool(
            ka["alive"]
            and kb["alive"]
            and ka["strongest_period"] is not None
            and ka["strongest_period"] == kb["strongest_period"]
        ),
        "one_alive_one_dead": ka["alive"] != kb["alive"],
    }

    # Superimposition / depth battery on the equal-length prefix, plus the offset-aware
    # cross-text kappa (handles unequal lengths and non-zero alignment offsets).
    n = min(len_a, len_b)
    a_idx = [ord(c) - 65 for c in a[:n]]
    b_idx = [ord(c) - 65 for c in b[:n]]
    battery = _depth_battery(a_idx, b_idx)
    cross_kappa = mutual_kappa_scan(a, b, top=3) if len_a >= 20 and len_b >= 20 else []
    shared_ks, shared_ks_z = _shared_keystream_signal(battery)
    superimposition = {
        "n": n,
        "match": battery.get("match"),
        "diffIC": battery.get("diffIC"),
        "best_match_rate": battery["match"]["best"] if battery else None,
        "best_match_z": battery["match"]["scan_max_z"] if battery else None,
        "best_diffic_z": battery["diffIC"]["scan_max_z"] if battery else None,
        "shared_keystream": shared_ks,
        "shared_keystream_z": shared_ks_z,
    }

    # --- rule-based verdict (transparent + conservative) -------------------------------------
    min_english_l1 = min(l1_a_english, l1_b_english)
    profiles_share = (
        freq_profile_l1 < PROFILE_RATIO * min_english_l1
        and (min_english_l1 - freq_profile_l1) >= PROFILE_MARGIN
    )
    period_match = kappa["same_strongest_period"]
    winding_diff = bool(profiles_share and kappa["one_alive_one_dead"])
    period_kappa_rel = period_match or winding_diff

    evidence: list[str] = []
    if profiles_share:
        evidence.append(
            f"sorted frequency profiles are closer to each other (L1={freq_profile_l1}) than "
            f"either is to English (L1 a={l1_a_english}, b={l1_b_english}) — same flattening class"
        )
    else:
        evidence.append(
            f"sorted frequency profiles are NOT closer to each other (L1={freq_profile_l1}) than "
            f"to English (min English L1={min_english_l1}) — different letter-frequency shapes"
        )
    if period_match:
        evidence.append(
            f"both texts wind at period {ka['strongest_period']} "
            f"(kappa alive: a z={ka['max_z']}, b z={kb['max_z']}) — same periodic structure"
        )
    elif winding_diff:
        alive, dead = ("a", "b") if ka["alive"] else ("b", "a")
        evidence.append(
            f"kappa is alive on {alive} (wound) but flat on {dead}, with matching profiles — "
            "consistent with the SAME substitution applied with a different winding"
        )
    if shared_ks:
        evidence.append(
            f"two-ciphertext superimposition spikes (z={shared_ks_z}) — a SHARED per-position "
            "keystream/tableau (additive-translate depth)"
        )
    if mutual_ioc >= 0.06:
        evidence.append(
            f"mutual IoC {mutual_ioc} is at the same-alphabet level (~0.066), not the "
            "independent-alphabet floor (~0.0385)"
        )

    shared = bool((profiles_share and period_kappa_rel) or shared_ks)
    if not shared:
        confidence = "insufficient"
        evidence.append(
            "insufficient evidence: need matching frequency shapes AND a period/kappa "
            "relationship (or a shared keystream) to claim a shared construction"
        )
    elif shared_ks and profiles_share:
        confidence = "high"
    elif period_match and profiles_share:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "len_a": len_a,
        "len_b": len_b,
        "ioc_a": ioc_a,
        "ioc_b": ioc_b,
        "freq_profile_l1": freq_profile_l1,
        "l1_a_english": l1_a_english,
        "l1_b_english": l1_b_english,
        "mutual_ioc": mutual_ioc,
        "kappa": kappa,
        "superimposition": superimposition,
        "cross_kappa": cross_kappa,
        "verdict": {
            "shared_construction": shared,
            "confidence": confidence,
            "evidence": evidence,
        },
    }
