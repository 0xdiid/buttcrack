"""The ring-indexed table must make the A-Z/ring confusion impossible, not merely documented."""

from __future__ import annotations

import pytest

from buttcrack.ring_tables import (
    ALPHABET_RING,
    KRYPTOS_RING,
    ring_flat_table,
    ring_letter_map,
    ring_ngram_table,
    ring_score,
    ring_to_letters,
)
from buttcrack.scoring import get_scorer

ENG = "ITISATRUTHUNIVERSALLYACKNOWLEDGEDTHATASINGLEMANINPOSSESSIONOFAGOODFORTUNE"


def test_identity_ring_reproduces_the_plain_table() -> None:
    """A caller not using a keyed alphabet must lose nothing."""
    tab = ring_ngram_table("bigrams", ALPHABET_RING)
    sc = get_scorer("bigrams")
    assert tab[ord("T") - 65][ord("H") - 65] == pytest.approx(sc.log_probs["TH"])
    assert tab[ord("Q") - 65][ord("X") - 65] == pytest.approx(sc.floor)


def test_kryptos_ring_table_is_indexed_by_ring_position() -> None:
    tab = ring_ngram_table("bigrams", KRYPTOS_RING)
    sc = get_scorer("bigrams")
    t, h = KRYPTOS_RING.index("T"), KRYPTOS_RING.index("H")
    assert tab[t][h] == pytest.approx(sc.log_probs["TH"])
    # and the A-Z position of T/H must NOT hold that value -- that is the bug being prevented
    assert tab[ord("T") - 65][ord("H") - 65] != pytest.approx(sc.log_probs["TH"])


def test_scoring_english_through_the_ring_matches_scoring_it_plainly() -> None:
    """The same text scores the same whichever index space it is carried in."""
    plain = ring_ngram_table("bigrams", ALPHABET_RING)
    keyed = ring_ngram_table("bigrams", KRYPTOS_RING)
    az = [ord(c) - 65 for c in ENG]
    kr = [KRYPTOS_RING.index(c) for c in ENG]
    assert ring_score(az, plain, 2) == pytest.approx(ring_score(kr, keyed, 2))


def test_the_bug_it_prevents_is_real_and_large() -> None:
    """Scoring ring indices against an A-Z table must be visibly, badly wrong."""
    plain = ring_ngram_table("bigrams", ALPHABET_RING)
    keyed = ring_ngram_table("bigrams", KRYPTOS_RING)
    kr = [KRYPTOS_RING.index(c) for c in ENG]
    correct = ring_score(kr, keyed, 2)
    wrong = ring_score(kr, plain, 2)  # the mistake: ring indices, A-Z table
    assert correct > wrong + 0.5, "the confusion must be large enough to matter"


def test_round_trip_ring_to_letters() -> None:
    kr = [KRYPTOS_RING.index(c) for c in ENG]
    assert ring_to_letters(kr, KRYPTOS_RING) == ENG


def test_letter_map_is_a_permutation() -> None:
    m = ring_letter_map(KRYPTOS_RING)
    assert sorted(m) == list(range(26))
    assert m[0] == ord("K") - 65


def test_rejects_a_bad_ring() -> None:
    with pytest.raises(ValueError, match="permutation"):
        ring_ngram_table("bigrams", "AABCDEFGHIJKLMNOPQRSTUVWXY")
    with pytest.raises(ValueError, match="permutation"):
        ring_letter_map("SHORT")


def test_refuses_a_table_too_large_to_nest() -> None:
    with pytest.raises(ValueError, match="ring_flat_table"):
        ring_ngram_table("quadgrams", KRYPTOS_RING)


class TestRingFlatTable:
    """High-order ring-folded tables — the orders that used to be hand-rolled at each call site."""

    def test_identity_ring_matches_plain_az_index(self):
        """With the identity ring, packing ring positions == packing A–Z indices."""
        tab = ring_flat_table("quadgrams", ALPHABET_RING)
        idx = tab.index([ord(c) - 65 for c in "TION"])
        assert tab[idx] > tab.floor

    def test_kryptos_ring_finds_the_same_gram_at_a_permuted_index(self):
        """The SAME text scores the same under both rings — only the index differs."""
        az = ring_flat_table("quadgrams", ALPHABET_RING)
        kr = ring_flat_table("quadgrams", KRYPTOS_RING)
        az_pos = [ord(c) - 65 for c in "TION"]
        kr_pos = [KRYPTOS_RING.index(c) for c in "TION"]
        assert az_pos != kr_pos, "indices must differ or the test proves nothing"
        assert az[az.index(az_pos)] == kr[kr.index(kr_pos)]

    def test_english_outscores_reverse_under_the_kryptos_ring(self):
        """The regression that keeps recurring: ring-indexed text vs an A–Z table scores garbage.

        Under a correctly ring-folded table, English must beat its own reverse. Indexing an A–Z
        table with ring positions makes this assertion fail.
        """
        kr = ring_flat_table("quadgrams", KRYPTOS_RING)
        text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
        fwd = [KRYPTOS_RING.index(c) for c in text]
        rev = list(reversed(fwd))
        assert kr.score(fwd) > kr.score(rev)

    def test_high_orders_are_available_and_sparse_above_the_dense_limit(self):
        five = ring_flat_table("quintgrams", KRYPTOS_RING)
        six = ring_flat_table("hexagrams", KRYPTOS_RING)
        assert five.n == 5 and six.n == 6
        assert five.dense and not six.dense, "26^6 dense would be 1.2GB"
        eng = [KRYPTOS_RING.index(c) for c in "THEREFORE"]
        assert six.score(eng) > six.floor * (len(eng) - six.n + 1)

    def test_unseen_ngram_returns_floor(self):
        tab = ring_flat_table("quadgrams", KRYPTOS_RING)
        unseen = tab[tab.index([KRYPTOS_RING.index(c) for c in "QQQQ"])]
        # dense tables are float32, so the floor round-trips with ~1e-7 relative error
        assert unseen == pytest.approx(tab.floor)
