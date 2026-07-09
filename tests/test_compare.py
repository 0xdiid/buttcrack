"""Sibling-pair analysis (``buttcrack.compare.compare``).

Two ciphertexts built the SAME way (two different English plaintexts under one shared
period-7 keyed substitution) must read as a shared construction: their sorted frequency
profiles land closer to each other than to English, they wind at the same kappa period, and
the depth battery sees the shared keystream. Two unrelated ciphertexts (a Caesar of English
vs a random-letter string) must not.
"""

from __future__ import annotations

import random

import numpy as np

from buttcrack.compare import compare
from buttcrack.quagmire_solver import KRYPTOS_ALPHABET, _encrypt
from buttcrack.text import only_letters

# Two DIFFERENT, ordinary-English plaintexts (long enough for stable frequency + kappa stats).
PLAINTEXT_A = only_letters(
    "the old lighthouse keeper climbed the narrow spiral stair every evening at dusk to "
    "trim the great lamp and polish the brass fittings until they shone like gold against "
    "the fading light of the setting sun over the restless grey water of the northern bay "
    "he had kept the lonely watch for thirty years and knew each shifting current and every "
    "hidden reef that lay beneath the surface of the treacherous channel below the cliffs "
    "and the sailors who passed in the dark trusted his steady beam to guide them safely "
    "home through the fog and the driving rain of the long and bitter winter storms"
)
PLAINTEXT_B = only_letters(
    "high in the northern mountains a small village rested beneath the shadow of ancient "
    "peaks where the heavy winter snow lingered long into the reluctant spring and narrow "
    "stone paths wound between the crooked wooden houses whose steep roofs sagged under "
    "years of frost the people there raised sturdy goats and grew hardy barley in the thin "
    "soil of the terraced fields and gathered each autumn to share the harvest around great "
    "fires that burned through the cold clear nights while the elders told the old stories "
    "of travellers and traders who had crossed the high passes in the distant golden days"
)

# One shared period-7 keyed substitution (Quagmire-style over the KRYPTOS alphabet).
SHIFTS = [4, 17, 8, 21, 2, 11, 25]


def _keyed_sub(pt: str) -> str:
    """Encipher with the shared period-7 keyed substitution."""
    return _encrypt(pt, KRYPTOS_ALPHABET, KRYPTOS_ALPHABET, SHIFTS)


def test_shared_period7_keyed_substitution_is_detected():
    ct_a = _keyed_sub(PLAINTEXT_A)
    ct_b = _keyed_sub(PLAINTEXT_B)
    res = compare(ct_a, ct_b)

    # Profiles closer to EACH OTHER than to English (same flattening class).
    assert res["freq_profile_l1"] < res["l1_a_english"]
    assert res["freq_profile_l1"] < res["l1_b_english"]

    # Same periodic winding, and the verdict fires.
    assert res["kappa"]["a"]["strongest_period"] == 7
    assert res["kappa"]["b"]["strongest_period"] == 7
    assert res["kappa"]["same_strongest_period"]
    assert res["verdict"]["shared_construction"] is True
    assert res["verdict"]["confidence"] in ("moderate", "high")

    # The depth battery sees the shared periodic keystream: the best rotation's match rate is
    # lifted to plaintext-coincidence level (~0.066+), well above the ~0.0385 independent floor.
    # (It recurs at every multiple of the period rather than a single unique rotation, and a
    # keyed alphabet breaks the additive diffIC cancellation, so this is a rate/z signal, not a
    # unique-rotation shared_keystream flag.)
    assert res["superimposition"]["best_match_rate"] > 0.05
    assert res["superimposition"]["best_match_z"] > 3.0


def test_unrelated_ciphers_are_not_shared():
    # A Caesar (monoalphabetic, English-shaped) vs a uniform random-letter string.
    caesar = "".join(chr(65 + (ord(c) - 65 + 3) % 26) for c in PLAINTEXT_A)
    rng = random.Random(20240607)
    noise = "".join(chr(65 + rng.randrange(26)) for _ in range(len(PLAINTEXT_B)))
    res = compare(caesar, noise)

    # The English-shaped Caesar is far closer to English than to the flat noise.
    assert res["l1_a_english"] < res["freq_profile_l1"]
    assert res["verdict"]["shared_construction"] is False
    assert res["verdict"]["confidence"] == "insufficient"
    assert res["superimposition"]["shared_keystream"] is False


def test_return_shape_and_types():
    ct_a = _keyed_sub(PLAINTEXT_A)
    ct_b = _keyed_sub(PLAINTEXT_B)
    res = compare(ct_a, ct_b, max_period=12)

    for key in (
        "len_a",
        "len_b",
        "ioc_a",
        "ioc_b",
        "freq_profile_l1",
        "l1_a_english",
        "l1_b_english",
        "mutual_ioc",
        "kappa",
        "superimposition",
        "verdict",
    ):
        assert key in res, key
    # kappa per-period list spans 1..max_period for each text.
    assert [row["period"] for row in res["kappa"]["a"]["per_period"]] == list(range(1, 13))
    # verdict shape.
    v = res["verdict"]
    assert set(v) == {"shared_construction", "confidence", "evidence"}
    assert isinstance(v["shared_construction"], bool)
    assert isinstance(v["evidence"], list) and all(isinstance(s, str) for s in v["evidence"])


def test_numpy_cross_check_on_depth_battery():
    """The stdlib depth battery matches a straight numpy computation of the same statistic."""
    ct_a = _keyed_sub(PLAINTEXT_A)
    ct_b = _keyed_sub(PLAINTEXT_B)
    n = min(len(ct_a), len(ct_b))
    a = np.frombuffer(ct_a[:n].encode(), dtype=np.uint8).astype(np.int64) - 65
    b = np.frombuffer(ct_b[:n].encode(), dtype=np.uint8).astype(np.int64) - 65
    match_np = np.array([float(np.mean(a == np.roll(b, -d))) for d in range(n)])

    res = compare(ct_a, ct_b)
    m = res["superimposition"]["match"]
    assert abs(m["d0"] - round(float(match_np[0]), 4)) < 1e-9  # module rounds to 4 dp
    assert m["argmax"] == int(match_np.argmax())
    # The best rotation matches numpy's, and it lifts the match rate above the random floor.
    assert abs(m["best"] - round(float(match_np.max()), 4)) < 1e-9
    assert m["best"] > 1.0 / 26.0
