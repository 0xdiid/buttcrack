"""Chaocipher and the M-94 / M-138 cylinder — the two rotor-family additions.

Both are checked against published artefacts before any crack is trusted: Chaocipher
against Byrne's Exhibit 1, the M-94 against the documented disk table (disk 17 begins
``ARMYOFTHEUS``).
"""

import time

import pytest

from buttcrack import registry
from buttcrack.ciphers.m94 import m94_disks
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

CORPUS = (
    "the analysis routines need a healthy stretch of perfectly ordinary english prose so "
    "that the frequency statistics and quadgram fitness scores can lock onto the underlying "
    "message and recover the original text without any prior knowledge of the secret key "
    "that was chosen to encipher it in the beginning of this particular exercise today "
) * 2
TARGET = only_letters(CORPUS).upper()


# -- Chaocipher ----------------------------------------------------------------

CHAO_KEY = "HXUCZVAMDSLKPEFJRIGTWOBNYQ/PTLNBQDEOYSFAVZKGJRIHWXUMC"


def test_chaocipher_byrne_exhibit_1():
    """The canonical vector from Rubin's "Chaocipher Revealed"."""
    assert (
        registry.get("chaocipher").encode("WELLDONEISBETTERTHANWELLSAID", CHAO_KEY)
        == "OAHQHCNYNXTSZJRRHJBYHQKSOUJY"
    )


def test_chaocipher_byrne_exhibit_1_decode():
    assert (
        registry.get("chaocipher").decode("OAHQHCNYNXTSZJRRHJBYHQKSOUJY", CHAO_KEY)
        == "WELLDONEISBETTERTHANWELLSAID"
    )


def test_chaocipher_roundtrip_long():
    c = registry.get("chaocipher")
    assert c.decode(c.encode(CORPUS, CHAO_KEY), CHAO_KEY) == TARGET


def test_chaocipher_alphabets_never_settle():
    """The dynamic alphabets are the whole point: a repeated plaintext block does not
    give a repeated ciphertext block, which is why no period is recoverable."""
    c = registry.get("chaocipher")
    ct = c.encode("ATTACK" * 4, CHAO_KEY)
    assert ct[:6] != ct[6:12]


@pytest.mark.parametrize(
    "bad",
    [
        "ABC",  # not two alphabets
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # no separator
        "AABCDEFGHIJKLMNOPQRSTUVWXY/ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # left has a repeat
    ],
)
def test_chaocipher_rejects_bad_key(bad):
    with pytest.raises(ValueError):
        registry.get("chaocipher").encode("TEST", bad)


def test_chaocipher_declines_to_crack():
    """No keyless attack exists, so it must return nothing rather than noise.

    A near-correct key corrupts the entire remaining decrypt rather than one position,
    so the n-gram score of an almost-right key is indistinguishable from a random one —
    there is no gradient for a climber to follow.
    """
    ct = registry.get("chaocipher").encode(CORPUS, CHAO_KEY)
    assert registry.get("chaocipher").crack(ct, get_scorer()) == []


def test_chaocipher_excluded_from_auto():
    assert registry.get("chaocipher").auto_crackable is False


# -- M-94 ----------------------------------------------------------------------

FULL_KEY = "16,18,24,25,13,5,8,20,17,21,12,3,1,9,4,22,7,15,2,19,10,23,6,14,11/9"


def test_m94_disk_table_is_the_documented_one():
    """Disk 17 begins ARMYOFTHEUS — the one published check on the whole table."""
    disks = m94_disks()
    assert len(disks) == 25
    assert disks[16].startswith("ARMYOFTHEUS")


def test_m94_every_disk_is_a_full_alphabet():
    for disk in m94_disks():
        assert len(disk) == 26
        assert set(disk) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_m94_disks_are_all_distinct():
    assert len(set(m94_disks())) == 25


@pytest.mark.parametrize("key", [FULL_KEY, "17,4,9,22,1/6", "1,2/13"])
def test_m94_roundtrip(key):
    m = registry.get("m94")
    assert m.decode(m.encode(CORPUS, key), key) == TARGET


def test_m94_offset_zero_is_the_identity():
    """Reading the same row you aligned returns the plaintext — the degenerate key."""
    assert registry.get("m94").encode(CORPUS, "1,2,3/0") == TARGET


def test_m94_shares_a_disk_every_width_positions():
    """The structural fact the crack rests on: position j and j+width use one disk."""
    m = registry.get("m94")
    ct = m.encode("A" * 60, "5,9,14/7")
    assert ct[0] == ct[3] == ct[6]


@pytest.mark.parametrize("bad", ["1,2,3", "0,1/5", "26,1/5", "1,1,2/5"])
def test_m94_rejects_bad_key(bad):
    with pytest.raises(ValueError):
        registry.get("m94").encode("TEST", bad)


def test_m94_crack_recovers_full_25_disk_order():
    """The headline case: a permutation of 25 disks is 15.5 septillion arrangements,
    recovered exactly by Hungarian assignment plus an n-gram climb."""
    m = registry.get("m94")
    best = m.crack(m.encode(CORPUS, FULL_KEY), get_scorer(), top=1, timeout=200)[0]
    assert best.plaintext[:80] == TARGET[:80]
    assert best.key == FULL_KEY


def test_m94_crack_recovers_m138_style_subset():
    m = registry.get("m94")
    key = "17,4,9,22,1/6"
    best = m.crack(m.encode(CORPUS, key), get_scorer(), top=1, timeout=120, width=5)[0]
    assert best.plaintext[:80] == TARGET[:80]
    assert best.key == key


def test_m94_crack_declines_when_text_is_shorter_than_two_rows():
    assert registry.get("m94").crack("ABCDEFGHIJ", get_scorer()) == []


def test_m94_crack_is_fast_enough_for_auto():
    """`auto` gives each cipher a few seconds; a 25-column Hungarian plus climb fits."""
    m = registry.get("m94")
    start = time.monotonic()
    m.crack(m.encode(CORPUS, FULL_KEY), get_scorer(), top=1, timeout=20)
    assert time.monotonic() - start < 20
