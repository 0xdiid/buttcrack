"""Word-level LM: segmentation rejects fluent non-words; tiling reconstructs from bags."""

from __future__ import annotations

from buttcrack import wordlm


def test_real_prose_is_wordlike():
    seg = wordlm.word_segment("OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUME")
    assert seg.long_coverage > 0.5
    assert seg.score_per_letter > 0
    assert seg.wordlike
    assert "MORNING" in seg.words and "LIBRARIAN" in seg.words


def test_short_word_text_still_fully_segments():
    # A pangram is genuine English but almost all short words; segmentation still tiles
    # it fully and finds the words, even though (like any long-word method) it is not
    # rated "wordlike".
    seg = wordlm.word_segment("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG")
    assert seg.coverage == 1.0
    assert "QUICK" in seg.words and "BROWN" in seg.words


def test_fluent_nonword_is_rejected():
    # Running prose packs long real words; salad that a character n-gram model accepts
    # can only be tiled by short/obscure fragments -> low long-word coverage.
    salad = wordlm.word_segment("THESTHATIONESTINGTHEREOFTIONALEST")
    real = wordlm.word_segment("OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUME")
    assert real.long_coverage > salad.long_coverage
    assert real.wordlike
    assert not salad.wordlike


def test_empty_and_gap_only():
    assert wordlm.word_segment("").coverage == 0.0
    seg = wordlm.word_segment("ZXQWKVBJ")  # no long words tile this
    assert seg.coverage < 0.5


def test_word_tiling_reconstructs_from_bag():
    # The multiset of "COMPUTER" letters must tile back to COMPUTER (determinacy check).
    tilings = wordlm.word_tiling("COMPUTER", minlen=4)
    assert tilings, "expected at least one tiling"
    texts = {t.text for t in tilings}
    assert "COMPUTER" in texts


def test_word_tiling_multi_word():
    # A bag that anagrams to two words; the search should find a full-cover tiling.
    tilings = wordlm.word_tiling("ANOTHERWORLD", minlen=3, max_solutions=30)
    assert tilings
    # every returned tiling exactly consumes the input multiset
    from collections import Counter

    want = Counter("ANOTHERWORLD")
    for t in tilings[:5]:
        assert Counter(t.text) == want


def test_word_tiling_determinacy_signal():
    # A short unambiguous bag yields few tilings; the count is the determinacy signal.
    tilings = wordlm.word_tiling("RHYTHM", minlen=5)
    assert any(t.text == "RHYTHM" for t in tilings)
