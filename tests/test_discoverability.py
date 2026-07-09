"""Everything an agent needs is discoverable from the CLI."""

import json

from buttcrack.ciphers import ALL_CIPHERS
from buttcrack.cli import main


def test_every_cipher_has_a_key_format():
    for c in ALL_CIPHERS:
        inst = c()
        assert inst.key_format, f"{inst.name} missing key_format"
        # keyed ciphers must offer a working example
        if inst.needs_key:
            assert inst.key_example, f"{inst.name} missing key_example"


def test_list_json_exposes_key_format(capsys):
    main(["list", "--json"])
    rows = json.loads(capsys.readouterr().out)
    by_name = {r["name"]: r for r in rows}
    for field in ("name", "aliases", "needs_key", "key_format", "key_example", "complexity"):
        assert field in by_name["vigenere"]
    assert by_name["vigenere"]["key_format"]


def test_help_command(capsys):
    main(["help", "gromark", "--json"])
    info = json.loads(capsys.readouterr().out)
    assert info["name"] == "gromark"
    assert "/" in info["key_format"]  # multi-part key documented
    assert info["key_example"]


def test_schema_manifest(capsys):
    main(["schema", "--compact"])
    m = json.loads(capsys.readouterr().out)
    assert m["schema_version"] >= 1
    assert {"encode", "crack", "auto", "schema", "help"} <= set(m["commands"])
    assert "n/a" in m["verdict_values"]  # encode/decode verdict is documented
    assert "crack" in m["commands"] and m["commands"]["crack"]["flags"]
    assert len(m["ciphers"]) == len(ALL_CIPHERS)


def test_result_carries_schema_version(capsys):
    main(["encode", "caesar", "hello", "--key", "3", "--json", "--compact"])
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] >= 1
    # encode field semantics: ciphertext is the output, plaintext the input
    assert out["plaintext"] == "hello" and out["ciphertext"] != "hello"
    assert out["verdict"] == "n/a"
