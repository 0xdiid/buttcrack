"""Text utilities: convert (A1Z26) and format (group/strip/recase)."""

import json

from buttcrack import textops
from buttcrack.cli import main


def test_to_numbers():
    assert textops.to_numbers("ABZ") == "1 2 26"
    assert textops.to_numbers("ABZ", pair=True) == "01 02 26"
    assert textops.to_numbers("hello world") == "8 5 12 12 15 23 15 18 12 4"


def test_from_numbers_round_trip():
    assert textops.from_numbers("8 5 12 12 15") == "HELLO"
    assert textops.from_numbers("01,02,26") == "ABZ"  # any divider
    assert textops.from_numbers("99 5 0") == "E"  # out-of-range ignored


def test_group_and_strip():
    assert textops.group("attackatdawn", 5) == "attac katda wn"
    assert textops.strip_whitespace("a b\tc\nd") == "abcd"


def test_convert_cli_json(capsys):
    main(["convert", "ABC", "--to", "pairs", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["output"] == "01 02 03"


def test_format_cli_group(capsys):
    main(["format", "the quick brown fox", "--strip", "--group", "5", "--case", "upper"])
    assert capsys.readouterr().out.strip() == "THEQU ICKBR OWNFO X"


def test_convert_round_trip_cli(capsys):
    main(["convert", "secretmessage", "--to", "numbers"])
    nums = capsys.readouterr().out.strip()
    main(["convert", nums, "--to", "letters"])
    assert capsys.readouterr().out.strip() == "SECRETMESSAGE"
