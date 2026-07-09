"""Each cipher's crack() should recover the plaintext from its own ciphertext."""

import pytest

from buttcrack import engine, registry
from conftest import letters


def _encode(name, key, plaintext):
    return registry.get(name).encode(plaintext, key)


@pytest.mark.slow
def test_crack_caesar(plaintext):
    ct = _encode("caesar", "11", plaintext)
    result = engine.crack("caesar", ct, seed=1)
    assert letters(result.best().plaintext) == letters(plaintext)


@pytest.mark.slow
def test_crack_affine(plaintext):
    ct = _encode("affine", "7,12", plaintext)
    result = engine.crack("affine", ct, seed=1)
    assert letters(result.best().plaintext) == letters(plaintext)


@pytest.mark.slow
def test_crack_atbash(plaintext):
    ct = _encode("atbash", "", plaintext)
    result = engine.crack("atbash", ct)
    assert letters(result.best().plaintext) == letters(plaintext)


@pytest.mark.slow
def test_crack_vigenere(plaintext):
    ct = _encode("vigenere", "LEMON", plaintext)
    result = engine.crack("vigenere", ct, seed=1)
    best = result.best()
    assert letters(best.plaintext) == letters(plaintext)
    assert best.key == "LEMON"


@pytest.mark.parametrize("key", ["KEY", "NIGHT", "LIBERTY"])
@pytest.mark.slow
def test_crack_vigenere_short_keys(plaintext, key):
    # Regression: short keys on modest text must beat overfit longer-length keys.
    ct = _encode("vigenere", key, plaintext)
    result = engine.crack("vigenere", ct, seed=1)
    best = result.best()
    assert best.key == key
    assert letters(best.plaintext) == letters(plaintext)


@pytest.mark.parametrize(
    "name,key",
    [("beaufort", "SECRET"), ("variant-beaufort", "SECRET"), ("gronsfeld", "31415")],
)
@pytest.mark.slow
def test_crack_periodic_family(plaintext, name, key):
    ct = _encode(name, key, plaintext)
    result = engine.crack(name, ct, seed=1)
    best = result.best()
    assert best.key == key
    assert letters(best.plaintext) == letters(plaintext)


@pytest.mark.slow
def test_crack_autokey(plaintext):
    ct = _encode("autokey", "KEY", plaintext)
    result = engine.crack("autokey", ct, seed=1)
    best = result.best()
    assert best.key == "KEY"
    assert letters(best.plaintext) == letters(plaintext)


@pytest.mark.slow
def test_crack_bacon(plaintext):
    ct = _encode("bacon", "", plaintext)
    result = engine.crack("bacon", ct, seed=1)
    assert letters(result.best().plaintext) == letters(plaintext).replace("J", "I").replace(
        "V", "U"
    )


@pytest.mark.slow
def test_crack_porta(plaintext):
    ct = _encode("porta", "PORTA", plaintext)
    result = engine.crack("porta", ct, seed=1)
    best = result.best()
    assert letters(best.plaintext) == letters(plaintext)
    # Recovered key is canonical (pair-first letters) but must reproduce the ct.
    assert letters(registry.get("porta").encode(plaintext, best.key))[:20] == letters(ct)[:20]


@pytest.mark.slow
def test_crack_railfence(plaintext):
    ct = _encode("railfence", "5", plaintext)
    result = engine.crack("railfence", ct, seed=1)
    assert letters(result.best().plaintext) == letters(plaintext)


@pytest.mark.slow
def test_crack_columnar(plaintext):
    ct = _encode("columnar", "CRYPT", plaintext)  # width 5
    result = engine.crack("columnar", ct, seed=1, max_width=6)
    assert letters(result.best().plaintext) == letters(plaintext)


@pytest.mark.slow
def test_crack_substitution(plaintext):
    key = "PHQGIUMEAYLNOFDXJKRCVSTZWB"
    ct = _encode("substitution", key, plaintext)
    result = engine.crack("substitution", ct, seed=7, restarts=40)
    # Hill-climbing should recover almost all letters; require a strong match.
    recovered = letters(result.best().plaintext)
    truth = letters(plaintext)
    matches = sum(a == b for a, b in zip(recovered, truth, strict=True))
    assert matches / len(truth) >= 0.95


@pytest.mark.slow
def test_auto_routes_to_right_cipher(plaintext):
    ct = _encode("vigenere", "LEMON", plaintext)
    result = engine.auto(ct, seed=1)
    best = result.best()
    assert best.cipher == "vigenere"
    assert letters(best.plaintext) == letters(plaintext)
    assert result.identify is not None


@pytest.mark.slow
def test_crack_returns_confidence_in_range(plaintext):
    ct = _encode("caesar", "4", plaintext)
    result = engine.crack("caesar", ct)
    for c in result.candidates:
        assert 0.0 <= c.confidence <= 1.0
