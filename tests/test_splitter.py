"""Tests for the cipher-file splitter (CryptoCrack's "Separate Cipher Files")."""

from __future__ import annotations

from buttcrack.splitter import split_ciphers


def test_empty_input_returns_empty_list():
    assert split_ciphers("") == []
    assert split_ciphers("   \n\n  \t\n") == []


def test_single_block_no_label():
    entries = split_ciphers("XLMV TQ AB CDEF GHIJ KLMN")
    assert len(entries) == 1
    # No label found anywhere -> numbered fallback gives "1".
    assert entries[0]["title"] == "1"
    assert entries[0]["body"] == "XLMV TQ AB CDEF GHIJ KLMN"


def test_multi_entry_label_on_own_line():
    text = (
        "A-1.\n"
        "XLMV TQABCD EFGHIJ\n"
        "KLMNOP QRSTUV\n"
        "\n"
        "A-2.\n"
        "ZZQW ERTY UIOP\n"
        "\n"
        "A-3.\n"
        "MMNB VCXZ LKJH GFDS\n"
    )
    entries = split_ciphers(text)
    assert len(entries) == 3
    assert [e["title"] for e in entries] == ["A-1", "A-2", "A-3"]
    assert entries[0]["body"] == "XLMV TQABCD EFGHIJ KLMNOP QRSTUV"
    assert entries[1]["body"] == "ZZQW ERTY UIOP"
    assert entries[2]["body"] == "MMNB VCXZ LKJH GFDS"


def test_label_on_same_line_as_body():
    text = "1. WKLMN OPQRS TUVWX\n\n2. YZABC DEFGH IJKLM\n"
    entries = split_ciphers(text)
    assert len(entries) == 2
    assert entries[0]["title"] == "1"
    assert entries[0]["body"] == "WKLMN OPQRS TUVWX"
    assert entries[1]["title"] == "2"
    assert entries[1]["body"] == "YZABC DEFGH IJKLM"


def test_letter_number_labels_without_hyphen():
    text = "K1. AAAA BBBB CCCC\n\nC13: DDDD EEEE FFFF\n"
    entries = split_ciphers(text)
    assert [e["title"] for e in entries] == ["K1", "C13"]
    assert entries[0]["body"] == "AAAA BBBB CCCC"
    assert entries[1]["body"] == "DDDD EEEE FFFF"


def test_shouting_title_line():
    text = "THE LOST KEY\nQWERT YUIOP ASDFG\n\nHIDDEN MESSAGE HERE\nZXCVB NMASD FGHJK\n"
    entries = split_ciphers(text)
    assert len(entries) == 2
    assert entries[0]["title"] == "THE LOST KEY"
    assert entries[0]["body"] == "QWERT YUIOP ASDFG"
    assert entries[1]["title"] == "HIDDEN MESSAGE HERE"
    assert entries[1]["body"] == "ZXCVB NMASD FGHJK"


def test_all_caps_ciphertext_is_not_mistaken_for_title():
    # A single all-caps grouped ciphertext block: the first line is one long
    # grouped run, so it must NOT be treated as a heading.
    text = "ABCDEFGHIJ KLMNOPQRST\nUVWXYZABCD EFGHIJKLMN\n"
    entries = split_ciphers(text)
    assert len(entries) == 1
    assert entries[0]["title"] == "1"
    assert entries[0]["body"] == "ABCDEFGHIJ KLMNOPQRST UVWXYZABCD EFGHIJKLMN"


def test_leading_and_trailing_blank_lines_are_ignored():
    text = "\n\n\nA-1. HELLO WORLD\n\n\n"
    entries = split_ciphers(text)
    assert len(entries) == 1
    assert entries[0]["title"] == "A-1"
    assert entries[0]["body"] == "HELLO WORLD"


def test_internal_whitespace_collapsed():
    text = "A-1.   FOO\t\tBAR   BAZ\n  QUX   QUUX  \n"
    entries = split_ciphers(text)
    assert len(entries) == 1
    assert entries[0]["body"] == "FOO BAR BAZ QUX QUUX"


def test_mixed_labeled_and_unlabeled_blocks():
    # When at least one block has a real label, unlabeled blocks keep title None
    # (no numbered fallback, since the fallback only fires when *nothing* is labeled).
    text = "A-1. FIRST CIPHER TEXT\n\nJUST SOME CIPHERTEXT WITH lowercase so not a title\n"
    entries = split_ciphers(text)
    assert len(entries) == 2
    assert entries[0]["title"] == "A-1"
    assert entries[1]["title"] is None
    assert entries[1]["body"] == "JUST SOME CIPHERTEXT WITH lowercase so not a title"
