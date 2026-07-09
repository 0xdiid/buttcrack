"""Dictionary word-search tools."""

import json

from buttcrack import words
from buttcrack.cli import main


def test_pattern_of():
    assert words.pattern_of("NOON") == "0.1.1.0"
    assert words.pattern_of("ABC") == "0.1.2"
    # Same isomorph => same pattern.
    assert words.pattern_of("PEEP") == words.pattern_of("NOON")


def test_pattern_search_finds_isomorphs():
    # PEOPLE has pattern 0.1.2.0.3.1; the dictionary should return PEOPLE itself.
    res = words.pattern("PEOPLE")
    assert "PEOPLE" in res
    assert all(words.pattern_of(w) == words.pattern_of("PEOPLE") for w in res)


def test_match_wildcards():
    res = words.match("HOUS?")
    assert "HOUSE" in res
    assert all(len(w) == 5 and w[0:4] == "HOUS" for w in res)


def test_anagram():
    res = words.anagram("LISTEN")
    assert "SILENT" in res and "LISTEN" in res


def test_ngram_substring():
    res = words.ngram("RYPTOGR")
    assert "CRYPTOGRAM" in res
    assert all("RYPTOGR" in w for w in res)


def test_words_cli_json(capsys):
    main(["words", "anagram", "listen", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and "SILENT" in out["results"]
