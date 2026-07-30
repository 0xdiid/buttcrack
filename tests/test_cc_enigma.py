"""Enigma I / M3 and the Gillogly-style analyzer.

The machine is checked against the published ``AAAAA -> BDZGO`` vector and the
canonical 26-letter output before any crack result is trusted; the crack is gated on
settings it planted itself.
"""

import time

import pytest

from buttcrack import registry
from buttcrack.ciphers.enigma import (
    REFLECTORS,
    ROTORS,
    EnigmaMachine,
    parse_plugboard,
    plugboard_repr,
)
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

CORPUS = (
    "the analysis routines need a healthy stretch of perfectly ordinary english prose so "
    "that the frequency statistics and quadgram fitness scores can lock onto the underlying "
    "message and recover the original text without any prior knowledge of the secret key "
    "that was chosen to encipher it in the beginning of this particular exercise for today"
)
TARGET = only_letters(CORPUS).upper()


# -- the machine ---------------------------------------------------------------


def test_enigma_published_vector():
    """Rotors I II III, reflector B, rings AAA, positions AAA: AAAAA -> BDZGO."""
    assert registry.get("enigma").encode("AAAAA", "I II III/B/AAA/AAA") == "BDZGO"


def test_enigma_canonical_26_letter_output():
    """The full first turnover cycle, which exercises the double-step anomaly."""
    assert (
        registry.get("enigma").encode("A" * 26, "I II III/B/AAA/AAA")
        == "BDZGOWCXLTKSBTMCDLPBMUQOFX"
    )


@pytest.mark.parametrize(
    "key",
    [
        "I II III/B/AAA/AAA",
        "IV V I/C/BQF/XYZ/AB CD EF GH",
        "VI VII VIII/B/AAA/AAA",
        "V III I/C/ZZZ/QMT/AB CD EF GH IJ KL MN OP QR ST",
    ],
)
def test_enigma_is_reciprocal(key):
    """The reflector makes encryption its own inverse — decode is encode."""
    e = registry.get("enigma")
    assert e.encode(e.encode(CORPUS, key), key) == TARGET


def test_enigma_never_enciphers_a_letter_to_itself():
    """The reflector's other consequence, and the one every crib attack rests on."""
    ct = registry.get("enigma").encode("A" * 300, "I II III/B/AAA/AAA")
    assert "A" not in ct


def test_enigma_rotor_and_reflector_tables_are_valid_permutations():
    for wiring, _ in ROTORS.values():
        assert set(wiring) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for wiring in REFLECTORS.values():
        assert set(wiring) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_enigma_reflectors_are_involutions():
    """A reflector must pair letters up, or the machine would not be reciprocal."""
    for wiring in REFLECTORS.values():
        for i, ch in enumerate(wiring):
            assert wiring[ord(ch) - 65] == chr(i + 65)


def test_naval_rotors_have_two_notches():
    for name in ("VI", "VII", "VIII"):
        assert len(ROTORS[name][1]) == 2


def test_ring_and_position_shift_together_are_the_same_machine():
    """Turning the ring and turning the rotor with it cancels, except at turnover.

    This equivalence is what lets the crack sweep positions with rings fixed at AAA in
    phase 1 and recover the rings separately in phase 2.
    """
    a = EnigmaMachine(("I", "II", "III"), "B", "AAA", "AAA").run("A" * 10)
    b = EnigmaMachine(("I", "II", "III"), "B", "AAB", "AAB").run("A" * 10)
    assert a == b


@pytest.mark.parametrize(
    "key",
    [
        "I II/B/AAA/AAA",  # only two rotors
        "I I III/B/AAA/AAA",  # a repeated rotor
        "I II IX/B/AAA/AAA",  # unknown rotor
        "I II III/D/AAA/AAA",  # unknown reflector
        "I II III/B/AA/AAA",  # short ring setting
        "I II III/B/AAA",  # missing a field
    ],
)
def test_enigma_rejects_bad_key(key):
    with pytest.raises(ValueError):
        registry.get("enigma").encode("TEST", key)


# -- plugboard -----------------------------------------------------------------


def test_plugboard_roundtrip():
    assert plugboard_repr(parse_plugboard("AB CD EF")) == "AB CD EF"


def test_plugboard_is_an_involution():
    board = parse_plugboard("AB CD")
    assert all(board[board[i]] == i for i in range(26))


@pytest.mark.parametrize("spec", ["ABC", "AB BC", "A"])
def test_plugboard_rejects_bad_spec(spec):
    with pytest.raises(ValueError):
        parse_plugboard(spec)


def test_empty_plugboard_is_the_identity():
    assert parse_plugboard("") == list(range(26))


# -- the analyzer --------------------------------------------------------------


def _crack(key, **opts):
    e = registry.get("enigma")
    return e.crack(e.encode(CORPUS, key), get_scorer(), top=1, timeout=1200, **opts)


def test_crack_recovers_positions_with_known_rotor_order():
    best = _crack("I II III/B/AAA/QMT", rotor_orders=[("I", "II", "III")])[0]
    assert best.plaintext[:60] == TARGET[:60]
    assert best.meta["positions"] == "QMT"


def test_crack_recovers_ring_settings():
    """Phase 2's job: a wrong ring mis-times the middle rotor's turnover, which the
    IoC of phase 1 barely sees but the n-gram score does."""
    best = _crack("I II III/B/ABQ/QMT", rotor_orders=[("I", "II", "III")])[0]
    assert best.plaintext[:60] == TARGET[:60]


def test_crack_recovers_plugboard():
    """Phase 3 grows the board greedily — no crib required."""
    best = _crack("I II III/B/AAA/QMT/AB CD EF", rotor_orders=[("I", "II", "III")])[0]
    assert best.plaintext[:60] == TARGET[:60]
    assert set(best.meta["plugboard"].split()) == {"AB", "CD", "EF"}


def test_crack_handles_reflector_c():
    best = _crack("II V I/C/AAA/XYZ", rotor_orders=[("II", "V", "I")], reflectors=["C"])[0]
    assert best.plaintext[:60] == TARGET[:60]


@pytest.mark.slow
def test_crack_recovers_unknown_rotor_order():
    """The full Enigma I sweep: 60 rotor orders x 17576 start positions."""
    best = _crack("IV I V/B/AAA/QMT")[0]
    assert best.plaintext[:60] == TARGET[:60]
    assert best.meta["rotors"] == ["IV", "I", "V"]
    assert best.meta["exhaustive"]


def test_crack_declines_short_input():
    assert registry.get("enigma").crack("ABCDEFGHIJ", get_scorer()) == []


def test_crack_reports_partial_coverage_on_timeout():
    """A search cut short must not be readable as an exhausted one."""
    e = registry.get("enigma")
    got = e.crack(e.encode(CORPUS, "IV I V/B/AAA/QMT"), get_scorer(), top=1, timeout=3)
    if got:
        assert got[0].meta["coverage"] < 1.0
        assert not got[0].meta["exhaustive"]


def test_enigma_excluded_from_auto():
    """The attack runs in minutes; `auto` budgets seconds per cipher."""
    assert registry.get("enigma").auto_crackable is False


@pytest.mark.parametrize("probe", [60, 120])
def test_short_probe_loses_a_non_trivial_ring_setting(probe):
    """Truncating the probe is a speed lever that costs phase-1 recall.

    Measured, and narrower than it first looks. With rings AAA a 60-letter probe still
    solves — every start position is equally (un)affected, so the IoC ranking holds up.
    It is a *non-trivial ring* that breaks it: the wrong ring mis-times the middle
    rotor's turnover, which only shows up in the letters after that turnover, so a
    short probe is scoring mostly the part of the message the error has not reached.
    On this corpus the true setting ranks ~2300th by IoC at 60 letters and 1st at full
    length.

    Asserted rather than just documented, because the failure is silent — a wrong
    setting still returns a confident-looking candidate.
    """
    e = registry.get("enigma")
    ct = e.encode(CORPUS, "I II III/B/ABQ/QMT")
    order = [("I", "II", "III")]
    starved = e.crack(ct, get_scorer(), top=1, timeout=600, rotor_orders=order, probe=probe)
    assert starved and starved[0].plaintext[:60] != TARGET[:60]


def test_full_probe_recovers_what_the_short_probe_loses():
    assert _crack("I II III/B/ABQ/QMT", rotor_orders=[("I", "II", "III")])[0].plaintext[
        :60
    ] == TARGET[:60]


def test_ring_sweep_recovers_order_and_rings_together():
    """The one case the default cannot reach: rotor order AND rings both unknown.

    Without ``ring_sweep`` the right-hand ring is fixed at A in phase 1, and a measured
    plant put the true setting 128720th of 1054560 by IoC — past any shortlist. The
    sweep costs 26x and is therefore opt-in.
    """
    e = registry.get("enigma")
    ct = e.encode(CORPUS, "IV I V/B/AGH/QMT")
    order = [("IV", "I", "V")]
    plain = e.crack(ct, get_scorer(), top=1, timeout=1200, rotor_orders=order, ring_sweep=True)
    assert plain and plain[0].plaintext[:60] == TARGET[:60]
    assert plain[0].meta["rings"] == "AGH"


def test_ring_sweep_is_off_by_default():
    """The 26x phase-1 cost is opt-in, so an ordinary crack stays in the seconds."""
    e = registry.get("enigma")
    ct = e.encode(CORPUS, "I II III/B/AAA/QMT")
    start = time.monotonic()
    e.crack(ct, get_scorer(), top=1, timeout=600, rotor_orders=[("I", "II", "III")])
    assert time.monotonic() - start < 60
