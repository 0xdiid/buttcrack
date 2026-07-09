"""5-gram fitness infrastructure: build-script support + graceful scorer fallback."""

import sys
from pathlib import Path

from buttcrack.scoring import ngram_table_available, resolve_scorer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_ngrams  # noqa: E402


def test_higher_order_tables_are_bundled():
    # buttcrack now ships up to hexagrams for sharper hill-climbing fitness.
    assert ngram_table_available("quadgrams") is True
    assert ngram_table_available("quintgrams") is True
    assert ngram_table_available("hexagrams") is True


def test_resolve_scorer_uses_richest_available():
    # the requested richer model is used when its table is bundled
    assert resolve_scorer("quintgrams").n == 5
    assert resolve_scorer("hexagrams").n == 6
    assert resolve_scorer("trigrams").n == 3
    # the default is unchanged
    assert resolve_scorer("quadgrams").n == 4
    # a genuinely-absent model still falls back to quadgrams
    assert resolve_scorer("septgrams").n == 4


def test_build_script_supports_quintgrams(tmp_path):
    assert build_ngrams.NGRAMS.get(5) == "quintgrams"
    assert build_ngrams.count_ngrams("ABCDEABCDE", 5) == {
        "ABCDE": 2,
        "BCDEA": 1,
        "CDEAB": 1,
        "DEABC": 1,
        "EABCD": 1,
    }
    corpus = tmp_path / "c.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 50, encoding="utf-8")
    rc = build_ngrams.main([str(corpus), "--out", str(tmp_path), "--max-n", "5"])
    assert rc == 0
    assert (tmp_path / "english_quintgrams.txt").is_file()
    # --max-n 4 (default) does NOT emit a quintgram table
    out4 = tmp_path / "four"
    build_ngrams.main([str(corpus), "--out", str(out4)])
    assert not (out4 / "english_quintgrams.txt").exists()
    assert (out4 / "english_quadgrams.txt").is_file()
