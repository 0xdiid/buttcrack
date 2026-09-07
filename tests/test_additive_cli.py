"""Synthetic command-level coverage of the periodic additive workflow."""

import json

import pytest

from buttcrack.cli import main


def invoke(capsys, *args):
    code = main(["additive", *args, "--compact"])
    output = capsys.readouterr()
    return code, json.loads(output.out)


def test_analysis(capsys):
    code, envelope = invoke(capsys, "analyze", "--periods", "2", "3", "6", "8")
    assert code == 0
    assert envelope["result"]["effective_parameters"] == 12


def test_fixed_fragments_file_and_dictionary(capsys, tmp_path):
    ciphertext = tmp_path / "cipher.txt"
    ciphertext.write_text("LIPPS ASVPH")
    words = tmp_path / "words.txt"
    words.write_text("E\nF\n")
    code, envelope = invoke(
        capsys,
        "crib",
        "--file",
        str(ciphertext),
        "--periods",
        "1",
        "--fragment",
        "0:HE",
        "--fragment",
        "8:LD",
        "--wordlist",
        str(words),
    )
    result = envelope["result"]
    assert code == 0
    assert result["plaintext"] == "HELLOWORLD"
    assert result["word_recovery"]["candidates"] == [["E"]]
    assert result["word_recovery"]["unique_in_bank"]


def test_partial_and_contradiction(capsys):
    code, envelope = invoke(capsys, "crib", "ABCD", "--periods", "3", "--fragment", "0:A")
    assert code == 0
    assert envelope["result"]["plaintext"] == "A??D"
    code, envelope = invoke(
        capsys, "crib", "ABCD", "--periods", "1", "--fragment", "0:A", "--fragment", "1:A"
    )
    assert code == 1
    assert envelope["result"]["status"] == "inconsistent"


def test_drag_cap(capsys):
    code, envelope = invoke(
        capsys, "crib", "LIPPSASVPH", "--periods", "1", "--drag", "HELLO", "--max-placements", "1"
    )
    assert code == 1
    assert envelope["result"]["status"] == "capped"
    assert envelope["result"]["candidates"][0]["result"]["plaintext"] == "HELLOWORLD"


def test_bank_audit(capsys, tmp_path):
    words = tmp_path / "words.txt"
    words.write_text("E\ne\n!\nTO\n")
    code, envelope = invoke(capsys, "bank", "--wordlist", str(words))
    result = envelope["result"]
    assert code == 0
    assert result["accepted"] == ["E", "TO"]
    assert result["excluded_counts"] == {"duplicate": 1, "invalid_symbols": 1}
    assert len(result["sha256"]) == 64


@pytest.mark.parametrize("action", ["words", "crib"])
def test_dictionary_cap_propagates(capsys, tmp_path, action):
    words = tmp_path / "words.txt"
    words.write_text("E\nF\n")
    args = [action]
    args += ["--pad", "4,?,4"] if action == "words" else ["LIPPS", "--fragment", "0:HELLO"]
    code, envelope = invoke(
        capsys, *args, "--periods", "1", "--wordlist", str(words), "--max-results", "1"
    )
    result = envelope["result"]
    if action == "crib":
        result = result["word_recovery"]
    assert code == 1
    assert result["capped"]
    assert not result["unique_in_bank"]


@pytest.mark.parametrize(
    "args",
    [
        ["analyze"],
        ["crib", "ABCD", "--periods", "1"],
        ["crib", "ABCD", "--periods", "1", "--fragment", "broken"],
        ["bank"],
        ["words", "--periods", "1"],
    ],
)
def test_invalid_input(capsys, args):
    assert main(["additive", *args]) == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
