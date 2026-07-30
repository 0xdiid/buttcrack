"""Transport-layer wrappers (`butt transform`) — the encoding shell around a cipher."""

import json

import pytest

from buttcrack import transforms, wrappers
from buttcrack.cli import main

# Long enough that a 12-byte key still gets ~25 samples per column, which is where
# the XOR column solve becomes reliable.
CORPUS = (
    b"the analysis routines need a healthy stretch of perfectly ordinary english prose so "
    b"that the frequency statistics and quadgram fitness scores can lock onto the underlying "
    b"message and recover the original text without any prior knowledge of the secret key "
    b"that was chosen to encipher it in the beginning of this particular exercise today "
) * 2


# -- XOR -----------------------------------------------------------------------


def test_xor_is_its_own_inverse():
    assert wrappers.xor_bytes(wrappers.xor_bytes(CORPUS, b"ICE"), b"ICE") == CORPUS


def test_xor_rejects_empty_key():
    with pytest.raises(ValueError):
        wrappers.xor_bytes(CORPUS, b"")


@pytest.mark.parametrize("key", [b"K", b"ICE", b"KRYPTOS", b"longersecret", b"\x1f\x00\xab"])
def test_crack_xor_recovers_key_and_plaintext(key):
    hits = wrappers.crack_xor(wrappers.xor_bytes(CORPUS, key))
    assert hits, "no XOR candidates"
    assert hits[0]["key"] == key
    assert hits[0]["plaintext"].encode() == CORPUS


def test_crack_xor_prefers_the_shortest_equivalent_key():
    """A doubled key decrypts identically, so it must not outrank the real one.

    Without the length penalty this is the default failure: every extra key byte is a
    free parameter, raw fitness rises monotonically with key length, and the search
    always returns a key of exactly ``max_keysize``.
    """
    hits = wrappers.crack_xor(wrappers.xor_bytes(CORPUS, b"ICE"), max_keysize=20)
    assert hits[0]["keysize"] == 3


def test_crack_xor_declines_tiny_input():
    assert wrappers.crack_xor(b"abc") == []


def test_crack_xor_needs_samples_per_column():
    """Accuracy is set by samples-per-column, not total length — 13 is not enough.

    Documented rather than fixed: the failure is quiet (a key that is mostly right),
    which is exactly the kind of result that gets trusted when it should not be.
    """
    short = CORPUS[:262]
    key = b"abcdefghijklmnopqrst"  # 20 bytes -> 13 samples per column
    hits = wrappers.crack_xor(wrappers.xor_bytes(short, key))
    assert hits[0]["key"] != key
    # The same key at ~26 samples per column does come back exactly.
    long = CORPUS[:524]
    assert wrappers.crack_xor(wrappers.xor_bytes(long, key))[0]["key"] == key


def test_xor_keysize_candidates_rank_the_true_length_highly():
    ranked = wrappers.xor_keysize_candidates(wrappers.xor_bytes(CORPUS, b"KRYPTOS"), top=5)
    assert any(r["keysize"] % 7 == 0 for r in ranked)


# -- base-N --------------------------------------------------------------------


@pytest.mark.parametrize("base", [2, 8, 16, 26, 36, 62])
def test_base_n_roundtrip(base):
    for number in (0, 1, 25, 12345, 987654321):
        assert wrappers.base_n_decode(wrappers.base_n_encode(number, base), base) == number


def test_base26_uses_letters_as_digits():
    assert wrappers.base_n_encode(0, 26) == "A"
    assert wrappers.base_n_encode(25, 26) == "Z"
    assert wrappers.base_n_encode(26, 26) == "BA"


def test_base_n_rejects_bad_digit():
    with pytest.raises(ValueError):
        wrappers.base_n_decode("9", 8)


def test_base_n_rejects_negative():
    with pytest.raises(ValueError):
        wrappers.base_n_encode(-1, 26)


def test_base32_and_base85_roundtrip():
    assert wrappers.base32_decode(wrappers.base32_encode("attack at dawn")) == b"attack at dawn"
    assert wrappers.base85_decode(wrappers.base85_encode("attack at dawn")) == b"attack at dawn"


def test_base32_decode_tolerates_missing_padding():
    encoded = wrappers.base32_encode("hello").rstrip("=")
    assert wrappers.base32_decode(encoded) == b"hello"


# -- ROT variants --------------------------------------------------------------


def test_rot47_is_an_involution_over_printable_ascii():
    text = "Hello, World! 42 ~`{}|"
    assert wrappers.rot47(wrappers.rot47(text)) == text


def test_rot47_moves_punctuation_and_rot13_does_not():
    assert wrappers.rot13("!") == "!"
    assert wrappers.rot47("!") != "!"


def test_rot5_and_rot18_are_involutions():
    assert wrappers.rot5(wrappers.rot5("2026")) == "2026"
    assert wrappers.rot18(wrappers.rot18("abc123")) == "abc123"


def test_rot18_moves_both_letters_and_digits():
    assert wrappers.rot18("a1") == "n6"


# -- keyboard geometry ---------------------------------------------------------


def test_keyboard_shift_roundtrip():
    assert wrappers.keyboard_shift(wrappers.keyboard_shift("hello", 1), -1) == "hello"


def test_keyboard_shift_wraps_at_the_row_end():
    assert wrappers.keyboard_shift("p", 1) == "q"


def test_keyboard_shift_leaves_non_letters_alone():
    assert wrappers.keyboard_shift("a-1", 1) == "s-1"


def test_keyboard_coordinates_roundtrip():
    assert wrappers.from_keyboard_coordinates(wrappers.keyboard_coordinates("hello")) == "HELLO"


def test_keyboard_coordinates_q_is_one_one():
    assert wrappers.keyboard_coordinates("q") == "11"


# -- phone keypad --------------------------------------------------------------


def test_multitap_roundtrip():
    assert wrappers.multitap_decode(wrappers.multitap_encode("cab")) == "CAB"


def test_multitap_published_positions():
    """C is the third letter on key 2; S is the fourth on key 7."""
    assert wrappers.multitap_encode("C") == "222"
    assert wrappers.multitap_encode("S") == "7777"


def test_t9_is_lossy_and_reports_its_ambiguity():
    assert wrappers.t9_encode("cab") == "222"
    assert wrappers.t9_candidates("222") == 27


# -- tap code ------------------------------------------------------------------


def test_tap_code_roundtrip():
    assert wrappers.tap_decode(wrappers.tap_encode("hello")) == "HELLO"


def test_tap_code_folds_k_into_c():
    assert wrappers.tap_encode("K") == wrappers.tap_encode("C")


def test_tap_code_a_is_one_one():
    assert wrappers.tap_encode("A") == ". ."


# -- NATO ----------------------------------------------------------------------


def test_nato_roundtrip():
    assert wrappers.nato_decode(wrappers.nato_encode("hello")) == "HELLO"


def test_nato_accepts_both_spellings():
    """Decoding by initial rather than by prefix table is what makes variants free."""
    assert wrappers.nato_decode("Alpha Bravo") == wrappers.nato_decode("Alfa Bravo") == "AB"
    assert wrappers.nato_decode("Juliet") == wrappers.nato_decode("Juliett") == "J"


# -- transcription -------------------------------------------------------------


def test_braille_roundtrip():
    t = wrappers.BRAILLE_TRANSCRIPTOR
    assert t.decode(t.encode("attack")) == "ATTACK"


def test_transcriptor_from_pairs():
    t = wrappers.Transcriptor.from_pairs("A=@,B=#,C=$")
    assert t.encode("CAB") == "$@#"
    assert t.decode("$@#") == "CAB"


def test_transcriptor_rejects_ambiguous_table():
    with pytest.raises(ValueError):
        wrappers.Transcriptor({"A": "@", "B": "@"})


def test_transcriptor_rejects_malformed_spec():
    with pytest.raises(ValueError):
        wrappers.Transcriptor.from_pairs("A@")


# -- detection -----------------------------------------------------------------


def test_detect_leaves_plain_ciphertext_alone():
    """The whole point of the gates: a plain A-Z ciphertext is valid base32 and valid
    NATO-prefix soup, and reporting either would bury the real cipher."""
    assert wrappers.detect("WKHTXLFNEURZQIRAMXPSVRYHUWKHODCBGRJ") == []


def test_detect_finds_braille():
    encoded = wrappers.BRAILLE_TRANSCRIPTOR.encode("attackatdawn")
    assert any(h["kind"] == "braille" for h in wrappers.detect(encoded))


def test_detect_finds_base32():
    hits = wrappers.detect(wrappers.base32_encode("attack at dawn now"))
    assert any(h["kind"] == "base32" for h in hits)


def test_detect_finds_nato():
    assert any(h["kind"] == "nato" for h in wrappers.detect(wrappers.nato_encode("attackatdawn")))


def test_transform_candidates_include_wrappers():
    cands = transforms.candidates(wrappers.BRAILLE_TRANSCRIPTOR.encode("attackatdawn"))
    assert any(c["kind"] == "braille" and c["text"] == "ATTACKATDAWN" for c in cands)


def test_transform_candidates_still_only_reverse_for_plain_text():
    cands = transforms.candidates("WKHTXLFNEURZQIRAMXPSVRYHUWKHODCBGRJ")
    assert [c["kind"] for c in cands] == ["reverse"]


# -- dispatch and CLI ----------------------------------------------------------


def test_apply_wrapper_peel_and_wrap():
    assert wrappers.apply_wrapper("nato", "ALFA BRAVO") == "AB"
    assert wrappers.apply_wrapper("nato", "AB", encode=True) == "ALFA BRAVO"


def test_apply_wrapper_parametric():
    assert wrappers.apply_wrapper("rot:3", "KHOOR") == "HELLO"
    assert wrappers.apply_wrapper("rot:3", "HELLO", encode=True) == "KHOOR"


def test_apply_wrapper_rejects_unknown_and_one_way():
    with pytest.raises(ValueError):
        wrappers.apply_wrapper("nonsense", "abc")
    with pytest.raises(ValueError):
        wrappers.apply_wrapper("t9", "222")


def test_apply_wrapper_parametric_needs_its_argument():
    with pytest.raises(ValueError):
        wrappers.apply_wrapper("rot", "abc")


def test_cli_transform_lists_wrappers(capsys):
    assert main(["transform", "--list-wrappers", "--json"]) == 0
    assert "rot47" in json.loads(capsys.readouterr().out)["wrappers"]


def test_cli_transform_apply(capsys):
    assert main(["transform", "--apply", "nato", "Alfa Bravo", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"][0]["text"] == "AB"


def test_cli_transform_xor(capsys):
    hex_ct = wrappers.xor_bytes(CORPUS, b"ICE").hex()
    assert main(["transform", "--xor", hex_ct, "--json"]) == 0
    best = json.loads(capsys.readouterr().out)["candidates"][0]
    assert best["key"] == "ICE"
    assert best["text"].encode() == CORPUS


def test_cli_transform_transcribe(capsys):
    assert main(["transform", "--transcribe", "A=@,B=#", "@#", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"][0]["text"] == "AB"
