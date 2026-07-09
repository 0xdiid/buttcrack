"""International Morse code table + helpers, shared by the Morse-family ciphers.

Fractionated Morse / Morbit / Pollux all convert text to a dot/dash stream with
``x`` separators (one ``x`` between letters, two between words) before fractionating.
"""

from __future__ import annotations

MORSE: dict[str, str] = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
}
FROM_MORSE: dict[str, str] = {code: letter for letter, code in MORSE.items()}


def text_to_morse(text: str) -> str:
    """Encode letters/digits to a ``.-x`` stream: 'x' between symbols, 'xx' between words.

    Leading/trailing separators are trimmed so the stream starts and ends on a
    dot/dash (the conventional fractionated-Morse form).
    """
    words = [w for w in "".join(c if c.isalnum() else " " for c in text.upper()).split() if w]
    encoded_words = ["x".join(MORSE[ch] for ch in word if ch in MORSE) for word in words]
    return "xx".join(encoded_words)


def morse_to_text(stream: str) -> str:
    """Decode a ``.-x`` stream back to text (single x = letter gap, xx = word gap)."""
    out = []
    for word in stream.split("xx"):
        letters = [FROM_MORSE.get(code, "") for code in word.split("x") if code]
        out.append("".join(letters))
    return " ".join(w for w in out if w)
