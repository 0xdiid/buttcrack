"""Round-trip: decode(encode(text, key), key) recovers the plaintext."""

import pytest

from buttcrack import registry
from buttcrack.text import only_letters

ROUND_TRIP_KEYS = {
    "caesar": "7",
    "rot13": "13",
    "atbash": "",
    "affine": "5,8",
    "vigenere": "LEMON",
    "beaufort": "CIPHER",
    "variant-beaufort": "CIPHER",
    "gronsfeld": "31415",
    "autokey": "SECRET",
    "porta": "PORTA",
    "railfence": "4",
    "columnar": "ZEBRA",
    "substitution": "QWERTYUIOPASDFGHJKLZXCVBNM",
}

# Transposition reorders letters and cannot preserve spacing/case, so it only
# round-trips the letter stream; substitution-class ciphers preserve layout.
TRANSPOSITION = {"railfence", "columnar"}

SAMPLE = "Meet me at the old bridge at dawn, bring the map!"


@pytest.mark.parametrize("name,key", ROUND_TRIP_KEYS.items())
def test_round_trip(name, key):
    cipher = registry.get(name)
    decoded = cipher.decode(cipher.encode(SAMPLE, key), key)
    if name in TRANSPOSITION:
        assert only_letters(decoded) == only_letters(SAMPLE)
    else:
        assert decoded == SAMPLE


@pytest.mark.parametrize("name", sorted(TRANSPOSITION))
def test_transposition_encode_does_not_leak_word_lengths(name):
    cipher = registry.get(name)
    key = ROUND_TRIP_KEYS[name]
    out = cipher.encode("attack the bridge at noon today", key)
    assert out.isalpha() and out.isupper()  # clean letter stream, no spacing leak


def test_aliases_resolve_to_same_instance():
    assert registry.get("vig") is registry.get("vigenere")
    assert registry.get("shift") is registry.get("caesar")


def test_substitution_rejects_bad_key():
    with pytest.raises(ValueError):
        registry.get("substitution").encode("hello", "TOOSHORT")


def test_affine_rejects_non_coprime_a():
    with pytest.raises(ValueError):
        registry.get("affine").encode("hello", "2,5")


def test_known_vectors():
    assert registry.get("caesar").encode("ABC", "3") == "DEF"
    assert registry.get("atbash").encode("ABC", "") == "ZYX"
    assert registry.get("rot13").encode("Hello", "") == "Uryyb"
    # Beaufort C=(K-P): K-P for KEY over ABC -> [10,3,22]
    assert registry.get("beaufort").encode("ABC", "KEY") == "KDW"
    assert registry.get("beaufort").decode("KDW", "KEY") == "ABC"  # reciprocal
    # Variant Beaufort C=(P-K): [0-10,1-4,2-24] mod 26 -> [16,23,4]
    assert registry.get("variant-beaufort").encode("ABC", "KEY") == "QXE"
    # Gronsfeld C=(P+digit): ABC + 1,2,3 -> BDF
    assert registry.get("gronsfeld").encode("ABC", "123") == "BDF"
    # Autokey (Wikipedia vector): keystream = QUEENLY + plaintext
    assert registry.get("autokey").encode("ATTACKATDAWN", "QUEENLY") == "QNXEPVYTWTWP"
    assert registry.get("autokey").decode("QNXEPVYTWTWP", "QUEENLY") == "ATTACKATDAWN"
    # Porta (ACA vector, reciprocal)
    porta = registry.get("porta")
    assert porta.encode("ENCIPHERMENTISRECIPROCAL", "PORTA") == "YGXRCOYJVRGMQJEYWQGEHWVU"
    assert porta.decode("YGXRCOYJVRGMQJEYWQGEHWVU", "PORTA") == "ENCIPHERMENTISRECIPROCAL"


def test_bacon_vector_and_roundtrip():
    bacon = registry.get("bacon")
    # dCode vector (24-letter table)
    assert bacon.encode("DCODE").replace(" ", "") == "AAABBAAABAABBABAAABBAABAA"
    # round-trips letters (uppercased; J->I, V->U folded)
    assert bacon.decode(bacon.encode("attackatdawn")) == "ATTACKATDAWN"
    assert bacon.decode(bacon.encode("java")) == "IAUA"  # J->I, V->U


def test_beaufort_is_reciprocal():
    # Beaufort applied twice is the identity (encode == decode).
    msg = "the eagle lands at midnight"
    b = registry.get("beaufort")
    assert b.encode(b.encode(msg, "FALCON"), "FALCON") == msg
