"""CLI surface: JSON schema, exit codes, and NDJSON batch mode."""

import json

import pytest

from buttcrack import registry
from buttcrack.cli import main
from conftest import letters


def test_encode_decode_json_fields_are_inverses(capsys):
    # `ciphertext`/`plaintext` must name the cipher-side/plaintext-side text in
    # BOTH directions (regression: encode used to swap them).
    main(["encode", "vigenere", "attack at dawn", "--key", "LEMON", "--json", "--compact"])
    enc = json.loads(capsys.readouterr().out)
    assert enc["plaintext"] == "attack at dawn"  # the input
    assert enc["ciphertext"] != "attack at dawn"  # the encoded output
    main(["decode", "vigenere", enc["ciphertext"], "--key", "LEMON", "--json", "--compact"])
    dec = json.loads(capsys.readouterr().out)
    assert dec["ciphertext"] == enc["ciphertext"]  # decode input is the ciphertext
    assert dec["plaintext"] == "attack at dawn"  # round-trips to the original


def test_subfoursq_cli_ranks_planted_pair(capsys):
    """The subfoursq command recovers a planted four-square pair and emits JSON rows."""
    from buttcrack.sub_four_square import encrypt_sub_over_four_square

    ct = encrypt_sub_over_four_square(
        "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGEACHTREE",
        "WATERMELON", "LAVENDER", outer_shifts=[3, 17, 0, 9, 22, 5, 14],
    )
    rc = main(["subfoursq", ct, "--squares", "WATERMELON,LAVENDER,MEADOW,SILVER",
               "--outer-period", "7", "--top", "3", "--json", "--compact"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0 and rows
    assert {"tr_square", "bl_square", "key", "plaintext", "score"} <= set(rows[0])
    # the planted pair scores best
    assert rows[0]["tr_square"].startswith("WATERM")


def test_subserpf_cli_runs_and_emits_json(capsys):
    from buttcrack.sub_seriated_playfair import encrypt_sub_over_seriated_playfair

    ct = encrypt_sub_over_seriated_playfair(
        "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGE",
        "BUTTERFLY", inner_period=7, outer_shifts=[3, 17, 0, 9, 22, 5, 14],
    )
    rc = main(["subserpf", ct, "--squares", "BUTTERFLY,MEADOW", "--inner-period", "7",
               "--outer-period", "7", "--top", "2", "--json", "--compact"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0 and rows
    assert {"square", "key", "plaintext", "score"} <= set(rows[0])


@pytest.mark.slow
def test_crack_json_schema(capsys, plaintext):
    ct = registry.get("caesar").encode(plaintext, "9")
    rc = main(["crack", "caesar", ct, "--json", "--seed", "1"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    # Stable top-level schema the agents rely on.
    for field in (
        "ok",
        "operation",
        "cipher",
        "plaintext",
        "key",
        "score",
        "confidence",
        "runtime_ms",
        "candidate_count",
        "candidates",
    ):
        assert field in out
    assert out["ok"] is True
    assert letters(out["plaintext"]) == letters(plaintext)


@pytest.mark.slow
def test_auto_json_includes_identify(capsys, plaintext):
    ct = registry.get("vigenere").encode(plaintext, "LEMON")
    main(["auto", ct, "--json", "--seed", "1"])
    out = json.loads(capsys.readouterr().out)
    assert out["cipher"] == "auto"
    assert "identify" in out
    assert out["candidates"][0]["cipher"] == "vigenere"


def test_encode_decode_roundtrip_cli(capsys):
    main(["encode", "vigenere", "attack at dawn", "--key", "LEMON"])
    encoded = capsys.readouterr().out.strip()
    main(["decode", "vigenere", encoded, "--key", "LEMON"])
    assert capsys.readouterr().out.strip() == "attack at dawn"


def test_identify_json(capsys, plaintext):
    ct = registry.get("caesar").encode(plaintext, "5")
    main(["identify", ct, "--json"])
    out = json.loads(capsys.readouterr().out)
    assert "index_of_coincidence" in out
    assert out["likely_families"][0]["family"] in {
        "monoalphabetic",
        "transposition",
        "polyalphabetic",
    }


def test_list_json(capsys):
    main(["list", "--json"])
    out = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in out}
    assert {"caesar", "vigenere", "substitution"} <= names


def test_solve_batch(tmp_path, capsys, plaintext):
    ct = registry.get("caesar").encode(plaintext, "3")
    batch = tmp_path / "jobs.ndjson"
    batch.write_text(
        json.dumps({"id": "a", "op": "crack", "cipher": "caesar", "text": ct, "seed": 1})
        + "\n"
        + json.dumps({"id": "b", "op": "decode", "cipher": "caesar", "text": ct, "key": "3"})
        + "\n"
    )
    main(["solve", "--batch", str(batch)])
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert {line["id"] for line in lines} == {"a", "b"}
    for line in lines:
        assert letters(line["plaintext"]) == letters(plaintext)


@pytest.mark.slow
def test_crack_exit_code_on_failure(capsys):
    # Too short to crack -> no candidates -> exit code 1.
    rc = main(["crack", "substitution", "abc", "--json"])
    capsys.readouterr()
    assert rc == 1
