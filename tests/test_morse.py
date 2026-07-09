"""International Morse table + stream helpers."""

from buttcrack.ciphers.morse import morse_to_text, text_to_morse


def test_known_codes():
    from buttcrack.ciphers.morse import MORSE

    assert MORSE["A"] == ".-"
    assert MORSE["Z"] == "--.."
    assert MORSE["5"] == "....."


def test_stream_round_trip():
    # SOS -> ...x---x...
    assert text_to_morse("SOS") == "...x---x..."
    assert morse_to_text("...x---x...") == "SOS"


def test_word_separator():
    stream = text_to_morse("HI THERE")
    assert "xx" in stream  # word gap
    assert morse_to_text(stream) == "HI THERE"
