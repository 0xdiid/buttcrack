"""Josse cipher — round-trip, the mono-invariance that powers the attack, and crack."""

import random

from buttcrack.ciphers.josse import Josse, mixed_alphabet
from buttcrack.registry import get
from buttcrack.scoring import NgramScorer
from buttcrack.text import ALPHABET, only_letters

PT = "WELLDONEISBETTERTHANWELLSAID"


def test_registered():
    assert get("josse").name == "josse"


def test_round_trip():
    j = Josse()
    for key in ("CHIEN", "KRYPTOS/7", "MERIDIAN"):
        assert only_letters(j.decode(j.encode(PT, key), key)) == PT


def test_historical_25_letter_form_round_trips():
    j = Josse()
    pt = PT.replace("W", "V")
    assert only_letters(j.decode(j.encode(pt, "CHIEN", drop="W"), "CHIEN", drop="W")) == pt


def test_difference_sequence_is_monoalphabetic_at_true_numbering():
    """The whole attack rests on this: at the right numbering, D = num(plaintext)."""
    j = Josse()
    long_pt = (PT * 8)[:153]
    key = "CHIEN"
    ct = only_letters(j.encode(long_pt, key))
    alpha = mixed_alphabet(key, None)
    true_num = [alpha.index(ALPHABET[k]) for k in range(26)]
    d = j.difference_sequence(ct, true_num)
    # a monoalphabetic image: the map D -> plaintext must be a well-defined function
    pairs = dict()
    ok = True
    for dv, p in zip(d, long_pt, strict=True):
        if pairs.setdefault(dv, p) != p:
            ok = False
    assert ok


def test_digraph_ioc_separates_true_numbering():
    """Stage-1 objective sanity: mono-invariant, so it is high only at the truth."""
    j = Josse()
    long_pt = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 6)[:153]
    ct = only_letters(j.encode(long_pt, "CHIEN"))
    alpha = mixed_alphabet("CHIEN", None)
    true_num = [alpha.index(ALPHABET[k]) for k in range(26)]
    rnd = random.Random(0)
    wrong = list(range(26))
    rnd.shuffle(wrong)
    assert j.digraph_ioc(j.difference_sequence(ct, true_num)) > 2.0
    assert j.digraph_ioc(j.difference_sequence(ct, wrong)) < 2.0


def test_keyword_sweep_recovers_the_key():
    """The practical attack: Josse's numbering comes from a KEYWORD, so sweep words."""
    j = Josse()
    sc = NgramScorer()
    pt = (
        "THEREAREMANYWAYSTOWRITEASENTENCEBUTONLYAFEWOFTHEMWILLEVERBEREAD"
        "ANDFEWERSTILLWILLBEREMEMBEREDBYANYONEATALLAFTERTHEYEARSHAVEPASSED"
    )[:153]
    ct = only_letters(j.encode(pt, "LANTERN"))
    bank = ["HARBOUR", "MERIDIAN", "LANTERN", "COMPASS", "THICKET", "CHIEN"]
    best = max((sc.score(only_letters(j.decode(ct, w))), w) for w in bank)
    assert best[1] == "LANTERN"
