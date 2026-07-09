"""Statistical analysis tool (`butt stats`)."""

import json

from buttcrack import registry
from buttcrack.analysis import analyze
from buttcrack.cli import main


def test_analyze_english_profile(plaintext):
    info = analyze(plaintext)
    assert info["length"] > 100
    # English IoC is ~0.066; the most frequent letter should be a common one.
    assert 0.05 < info["index_of_coincidence"] < 0.08
    assert info["frequencies"][0]["letter"] in "ETAOINS"


def test_kasiski_finds_vigenere_period(plaintext):
    # A length-4 key leaves repeated-sequence spacings that are multiples of 4.
    ct = registry.get("vigenere").encode(plaintext, "WXYZ")
    info = analyze(ct)
    periods = [p["period"] for p in info["likely_periods"]]
    assert 4 in periods or any(p % 4 == 0 for p in periods)


def test_ioc_decay_flat_on_periodic_vigenere(plaintext):
    # A periodic substitution has a STATIONARY IoC — no positional drift.
    from buttcrack.analysis import ioc_decay

    ct = registry.get("vigenere").encode(plaintext, "CIPHER")
    d = ioc_decay(ct)
    assert d  # input long enough to segment
    assert not d["non_stationary"]
    assert d["slope_z"] > -2.5


def test_inner_class_coset_ioc_orders_flatteners():
    """Calibration table orders the classes: language high, flatteners in the middle,
    uniform at the floor."""
    from buttcrack.analysis import inner_class_coset_ioc

    cal = inner_class_coset_ioc(150, 7, samples=120, seed=1)
    assert cal["language"]["mean"] > cal["playfair"]["mean"] > cal["uniform"]["mean"]
    assert cal["language"]["mean"] > 0.06  # ~ English IoC
    assert abs(cal["uniform"]["mean"] - 1 / 26) < 0.006  # ~ random floor
    # digraphic classes cluster in the flattener band, clear of both extremes
    for cls in ("playfair", "bifid", "four_square"):
        assert cal["uniform"]["mean"] < cal[cls]["mean"] < cal["language"]["mean"]


def test_classify_coset_ioc_picks_digraphic_band():
    """A digraphic-class coset IoC classifies to playfair/bifid/four_square, not language/uniform."""
    from buttcrack.analysis import classify_coset_ioc

    hits = classify_coset_ioc(0.0536, 153, 7, samples=120, seed=1, tol=1.5)
    assert hits, "expected at least one class within tolerance"
    top = hits[0]["class"]
    assert top in ("playfair", "bifid", "four_square")


def test_period_inner_content_language_vs_flattened(plaintext):
    # The layer UNDER a detected period is either natural language (plain Vigenere/Quagmire,
    # peelable) or already flattened (a digraphic/polygraphic inner or a non-prose payload).
    from buttcrack.analysis import period_inner_content
    from buttcrack.text import only_letters

    # (1) period-7 Vigenere over ENGLISH: coset IoC ~ English -> NATURAL-LANGUAGE.
    ct_lang = registry.get("vigenere").encode(plaintext, "SEVENTH")
    lang = period_inner_content(ct_lang, 7)
    assert lang["reliable"]
    assert lang["verdict"].startswith("NATURAL-LANGUAGE")

    # (2) period-7 Vigenere over a Playfair (flattened) inner -> FLATTENED.
    inner = registry.get("playfair").encode(plaintext, "KRYPTOS")
    ct_flat = registry.get("vigenere").encode(inner, "SEVENTH")
    flat = period_inner_content(ct_flat, 7)
    assert flat["reliable"]
    assert flat["verdict"].startswith("FLATTENED")
    assert flat["coset_ioc"] < lang["coset_ioc"]

    # A small-sample harmonic (few letters/column) is flagged unreliable, not classified.
    short = period_inner_content(ct_lang, len(only_letters(ct_lang)) // 5)
    assert not short["reliable"]


def _decaying_text(n: int = 320, seed: int = 1) -> str:
    """Text whose per-position randomness GROWS along the message: each quarter is
    drawn uniformly from a progressively larger alphabet, so IoC decays (~0.056 ->
    ~0.039) with no periodic structure — a stand-in for an evolving keystream."""
    import random

    rng = random.Random(seed)
    sizes = [18, 21, 24, 26]
    q = n // 4
    out = []
    for i, size in enumerate(sizes):
        count = q if i < 3 else n - 3 * q
        out.append("".join(chr(65 + rng.randrange(size)) for _ in range(count)))
    return "".join(out)


def test_ioc_decay_flags_evolving_keystream():
    # An evolving keystream thins structure toward the tail -> per-segment IoC
    # decreases, which ioc_decay must flag (and a stationary cipher must not).
    from buttcrack.analysis import ioc_decay

    d = ioc_decay(_decaying_text())
    assert d["non_stationary"]
    assert d["slope_z"] <= -2.5


def test_diagnosis_flags_non_stationary():
    # identify's no-period branch must name the evolving-keystream case.
    from buttcrack.identify import identify

    diag = identify(_decaying_text())["diagnosis"]
    assert "NON-STATIONARY" in diag


def test_search_aware_null_separates_structure_from_fluke(plaintext):
    # A best-of-search score is only signal if it beats the SAME search on shuffles.
    import random

    from buttcrack.analysis import _mean_col_ioc, search_aware_null

    def best_period_ioc(s: str) -> float:
        seq = [ord(c) - 65 for c in s]
        return max(_mean_col_ioc(seq, p) for p in range(2, 13))

    # Real period-5 structure clears the shuffled-search null.
    ct = registry.get("vigenere").encode(plaintext, "GHOST")
    real = search_aware_null(ct, best_period_ioc, samples=30)
    assert real["beats_null_max"]
    assert real["z"] > 3

    # Structureless text does NOT — its best-of-search sits inside the null band.
    rng = random.Random(0)
    noise = "".join(chr(65 + rng.randrange(26)) for _ in range(254))
    fluke = search_aware_null(noise, best_period_ioc, samples=30)
    assert not fluke["beats_null_max"]


def test_stats_cli_json(capsys, plaintext):
    main(["stats", plaintext, "--json"])
    info = json.loads(capsys.readouterr().out)
    for field in (
        "length",
        "index_of_coincidence",
        "chi_squared",
        "frequencies",
        "bigrams",
        "trigrams",
        "kasiski_repeats",
        "likely_periods",
    ):
        assert field in info
    assert len(info["frequencies"]) == 26


def test_stats_empty_input_is_safe(capsys):
    main(["stats", "12345", "--json"])
    info = json.loads(capsys.readouterr().out)
    assert info["length"] == 0
    assert info["index_of_coincidence"] is None


def _planted_period(period: int = 7, repeats: int = 30, seed: int = 3) -> str:
    """A length-``period`` block of random letters repeated ``repeats`` times — a
    pure periodic re-alignment that kappa must spike at ``lag = period``."""
    import random

    rng = random.Random(seed)
    block = "".join(chr(65 + rng.randrange(26)) for _ in range(period))
    return block * repeats


def test_kappa_spectrum_flags_planted_period():
    from buttcrack.analysis import kappa_spectrum

    spec = kappa_spectrum(_planted_period(period=7), max_lag=24)
    # Rows are sorted by z descending; the top lag is the planted period (or a
    # multiple of it — every multiple re-aligns the same block).
    top_lag = spec[0]["lag"]
    assert top_lag % 7 == 0
    assert spec[0]["z"] > 5
    # Each row is JSON-friendly with the documented keys.
    assert set(spec[0]) == {"lag", "kappa", "z"}


def test_kappa_spectrum_flat_on_random():
    import random

    from buttcrack.analysis import kappa_spectrum

    rng = random.Random(0)
    noise = "".join(chr(65 + rng.randrange(26)) for _ in range(300))
    spec = kappa_spectrum(noise, max_lag=20)
    # No planted period -> no lag should produce a large kappa z.
    assert max(r["z"] for r in spec) < 4


def test_kappa_spectrum_short_input_is_safe():
    from buttcrack.analysis import kappa_spectrum

    assert kappa_spectrum("") == []
    # One usable pair (lag 1) at most; never raises, always JSON-friendly rows.
    spec = kappa_spectrum("ABCAB", max_lag=8)
    assert all(set(r) == {"lag", "kappa", "z"} for r in spec)


def test_crackability_cliff_recoverable_when_period_short(plaintext):
    from buttcrack.analysis import crackability_cliff

    ct = registry.get("vigenere").encode(plaintext, "GHOST")  # period 5, long text
    res = crackability_cliff(ct, 5)
    assert res["recoverable"] is True
    assert res["cycles"] >= 2.5
    assert set(res) == {"effective_period", "cycles", "recoverable", "verdict"}


def test_crackability_cliff_not_recoverable_when_period_long(plaintext):
    from buttcrack.analysis import crackability_cliff
    from buttcrack.text import only_letters

    ct = only_letters(registry.get("vigenere").encode(plaintext, "GHOST"))
    n = len(ct)
    # period > length/4 must be flagged unrecoverable (OTP-grade).
    res = crackability_cliff(ct, n // 4 + 1)
    assert res["recoverable"] is False
    assert "OTP" in res["verdict"] or "marginal" in res["verdict"]


def test_crackability_cliff_auto_picks_period(plaintext):
    from buttcrack.analysis import crackability_cliff_auto

    ct = registry.get("vigenere").encode(plaintext, "GHOST")
    res = crackability_cliff_auto(ct)
    assert res["effective_period"] in (5, 10)  # period 5 or its first multiple
    assert res["recoverable"] is True
    assert res["period_z"] is not None


def test_crackability_cliff_auto_no_period_on_random():
    import random

    from buttcrack.analysis import crackability_cliff_auto

    rng = random.Random(0)
    noise = "".join(chr(65 + rng.randrange(26)) for _ in range(254))
    res = crackability_cliff_auto(noise)
    assert res["recoverable"] is False


def test_decay_fingerprint_stationary_on_periodic(plaintext):
    from buttcrack.analysis import decay_fingerprint

    ct = registry.get("vigenere").encode(plaintext, "CIPHER")
    ranked = decay_fingerprint(ct)
    assert ranked  # input long enough
    # A periodic substitution is IoC-stationary -> stationary wins or verdict says so.
    assert "no evolving family" in ranked[0]["verdict"]
    families = [r["family"] for r in ranked]
    assert set(families) == {"progressive-key", "autokey", "chain-gromark", "stationary"}


def test_decay_fingerprint_matches_evolving_shape():
    from buttcrack.analysis import decay_fingerprint

    ranked = decay_fingerprint(_decaying_text())
    assert ranked
    # The synthetic decaying text thins steadily -> stationary is NOT the best fit.
    assert ranked[0]["family"] != "stationary"
    assert ranked[-1]["family"] == "stationary"


def test_decay_fingerprint_short_input_is_safe():
    from buttcrack.analysis import decay_fingerprint

    assert decay_fingerprint("ABC") == []


def test_analyze_surfaces_new_diagnostics(plaintext):
    ct = registry.get("vigenere").encode(plaintext, "GHOST")
    info = analyze(ct)
    assert info["kappa_spectrum"]
    assert info["crackability_cliff"]
    assert "recoverable" in info["crackability_cliff"]
    assert "decay_fingerprint" in info


def test_linear_channel_detects_hill(plaintext):
    from buttcrack.analysis import linear_channel

    # A Hill cipher has a linear channel (some covector isolates a plaintext coordinate);
    # the language-independent test must fire on both 3x3 and 2x2 forms.
    res3 = linear_channel(registry.get("hill").encode(plaintext, "RIVERBANK"))
    assert res3["hit"] is True
    assert res3["channels"][0]["z"] <= -4.0
    assert "HILL" in res3["verdict"].upper()
    res2 = linear_channel(registry.get("hill").encode(plaintext, "3,2,5,7"))
    assert res2["hit"] is True


def test_linear_channel_silent_on_flat_nonlinear(plaintext):
    from buttcrack.analysis import linear_channel

    # A FLAT non-Hill cipher must not be mistaken for a Hill: a Quagmire is monoalphabetic-per-
    # column, so its varying key scrambles any bigram structure — no polygraphic linear channel.
    assert linear_channel(registry.get("quagmire3").encode(plaintext, "CRUCIBL/METALS"))["hit"] is False


def test_linear_channel_skips_non_flat_text(plaintext):
    from buttcrack.analysis import linear_channel

    # Out of scope: plain English is not flat, so the test refuses it rather than reporting the
    # ordinary bigram correlation as a spurious "linear channel".
    res = linear_channel(plaintext)
    assert res["hit"] is False
    assert res["reliable"] is False
    assert "not a flat" in res["verdict"]


def test_linear_channel_covectors_are_surjective_and_mixing():
    import math

    from buttcrack.analysis import _projective_surjective_covectors

    for k in (2, 3):
        reps = _projective_surjective_covectors(k)
        assert reps
        for v in reps:
            g = 0
            for e in v:
                g = math.gcd(g, e)
            # surjective: channel reaches all of Z26 (no zero-divisor-only covector like (13,13))
            assert math.gcd(g, 26) == 1
            # mixing: >= 2 non-zero coords (no single-position pickup that flags a periodic sub)
            assert sum(1 for e in v if e) >= 2


# --------------------------------------------------------------------------- #
# Width-parameterised linear-channel (Hill) detector
# --------------------------------------------------------------------------- #
_WIDE_PLAIN = "".join(
    ch
    for ch in (
        "the analysis routines need a healthy stretch of perfectly ordinary english "
        "prose so that the frequency statistics and quadgram fitness scores can lock "
        "onto the underlying message and recover the original text without any prior "
        "knowledge of the secret key that was chosen to encipher it in the beginning "
    ).upper()
    * 4
    if ch.isalpha()
)


def _wide_hill(letters: str, w: int, seed: int) -> str:
    """Encrypt ``letters`` (A-Z) with a random invertible w×w Hill matrix mod 26.

    The registry Hill is 2x2/3x3 only, so this builds an arbitrary-width Hill directly to
    exercise the width detector on wide (e.g. 9x9) maps.
    """
    import random

    def _det_field(m, p):  # nonzero iff M is invertible mod prime p (Gaussian elimination)
        n = len(m)
        a = [[x % p for x in row] for row in m]
        for col in range(n):
            piv = next((r for r in range(col, n) if a[r][col] % p), None)
            if piv is None:
                return 0
            a[col], a[piv] = a[piv], a[col]
            inv = pow(a[col][col], -1, p)
            for r in range(col + 1, n):
                f = (a[r][col] * inv) % p
                if f:
                    for c in range(col, n):
                        a[r][c] = (a[r][c] - f * a[col][c]) % p
        return 1

    rng = random.Random(seed)
    while True:
        matrix = [[rng.randrange(26) for _ in range(w)] for _ in range(w)]
        if _det_field(matrix, 2) and _det_field(matrix, 13):  # invertible mod 26
            break
    idx = [ord(c) - 65 for c in letters]
    idx = idx[: len(idx) // w * w]
    out = []
    for b in range(0, len(idx), w):
        block = idx[b : b + w]
        for row in matrix:
            out.append(chr(65 + sum(row[k] * block[k] for k in range(w)) % 26))
    return "".join(out)


def test_linear_channel_width_fires_at_true_block_width():
    from buttcrack.analysis import linear_channel_width

    # A width-3 Hill over English leaks a plaintext coordinate to the width-3 probe: the full Z26
    # covector search finds it, so the channel IoC lifts to ~ English and the matched null is beaten.
    ct3 = _wide_hill(_WIDE_PLAIN, 3, seed=0)
    res = linear_channel_width(ct3, alphabet="STD", widths=(2, 3), null_trials=120)
    w3 = res["widths"][3]
    assert w3["hit"] is True
    assert w3["z"] > 8
    assert w3["ioc"] > 0.06  # recovered the leaked coordinate ~ English IoC
    assert w3["beats_null_max"] is True
    assert w3["search_exhaustive"] is True  # the full covector space was enumerated at width 3
    assert res["best_width"] == 3


def test_linear_channel_width_blind_spot_and_honest_underpowered_flag():
    from buttcrack.analysis import linear_channel_width

    # THE blind spot that caused a wrong "nonlinear" verdict: a width-3 probe cannot see a
    # width-9 Hill (its 3-blocks straddle three different row-groups of the 9x9 map, washing out
    # every functional). The detector must NOT report a hit at width 3 here.
    ct9 = _wide_hill(_WIDE_PLAIN, 9, seed=0)
    res = linear_channel_width(ct9, alphabet="STD", widths=(3, 9), null_trials=120)
    assert res["widths"][3]["hit"] is False
    # ...and crucially it distinguishes "searched fully, genuinely no channel" (width 3, the full
    # Z26 covector space) from "search underpowered, cannot conclude" (width 9: 26**9 dwarfs any
    # tractable search). A null at width 9 is flagged INCONCLUSIVE, not "nonlinear".
    assert res["widths"][3]["search_exhaustive"] is True
    assert res["widths"][9]["search_exhaustive"] is False
    assert res["widths"][9]["hit"] is False
    assert "INCONCLUSIVE" in res["widths"][9]["note"]
    assert res["best_width"] is None


def test_linear_channel_width_silent_on_random():
    import random

    from buttcrack.analysis import linear_channel_width

    rng = random.Random(4)
    noise = "".join(chr(65 + rng.randrange(26)) for _ in range(len(_WIDE_PLAIN)))
    res = linear_channel_width(noise, alphabet="STD", widths=(2, 3, 9), null_trials=120)
    # Random text overfits the same amount under the matched null -> no width beats it.
    assert res["best_width"] is None
    assert all(not d["hit"] for d in res["widths"].values())


def test_bounded_surjective_covectors_guarded():
    import math

    from buttcrack.analysis import _bounded_surjective_covectors

    # Guard (a): the wide-width fallback set is surjective onto Z26 AND mixing (no zero-divisor
    # covector that would fake near-maximal concentration by collapsing the output range).
    for w in (4, 9):
        cov = _bounded_surjective_covectors(w, (-2, -1, 0, 1, 2), 800, 7)
        assert cov
        for v in cov:
            g = 0
            for e in v:
                g = math.gcd(g, e % 26)
            assert math.gcd(g, 26) == 1
            assert sum(1 for e in v if e) >= 2
