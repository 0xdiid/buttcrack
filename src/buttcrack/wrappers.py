"""Transport-layer wrappers — the encoding shell a puzzle puts *around* its cipher.

``transforms`` handles the three wrappers a classical ciphertext usually arrives in
(reverse, base64/hex, A1Z26). Real puzzle chains use a much wider shell: repeating-key
XOR, the base-N family, keyboard geometry, phone keypads, tap code, spelled-out
letters. None of these is a cipher in the cryptanalytic sense — there is no key to
recover except in the XOR case — but a solver that cannot peel them never reaches the
cipher underneath.

Everything here is a pure function over strings or bytes. The one real attack is
:func:`crack_xor`, which recovers a repeating XOR key without being given its length.

NOT IMPLEMENTED, deliberately: ROT8000 (its validity table is version-specific and an
approximation would silently produce wrong output), the PGP word list (512 entries,
mechanical), the periodic-table cipher (symbol segmentation is genuinely ambiguous —
``SNO`` is S+N+O or Sn+O), DTMF, semaphore and pigpen (no faithful text
representation). :class:`Transcriptor` covers any of these once given a table.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections import Counter

from .text import only_letters

# -- byte-level scoring --------------------------------------------------------

#: Relative frequency of each letter in English, plus space, as a byte-level target.
_ENGLISH_BYTE_FREQ = {
    ord(" "): 0.1800,
    ord("e"): 0.1041,
    ord("t"): 0.0729,
    ord("a"): 0.0651,
    ord("o"): 0.0645,
    ord("i"): 0.0602,
    ord("n"): 0.0575,
    ord("s"): 0.0537,
    ord("r"): 0.0498,
    ord("h"): 0.0480,
    ord("l"): 0.0325,
    ord("d"): 0.0335,
    ord("c"): 0.0223,
    ord("u"): 0.0227,
    ord("m"): 0.0203,
    ord("f"): 0.0197,
    ord("p"): 0.0161,
    ord("g"): 0.0159,
    ord("w"): 0.0169,
    ord("y"): 0.0145,
    ord("b"): 0.0124,
    ord("v"): 0.0082,
    ord("k"): 0.0056,
    ord("x"): 0.0014,
    ord("j"): 0.0010,
    ord("q"): 0.0008,
    ord("z"): 0.0005,
}


def _build_byte_weights() -> list[float]:
    weights = [-0.5] * 256  # a byte that cannot appear in text is disqualifying
    for byte in range(32, 127):
        weights[byte] = 0.005
    for byte in (9, 10, 13):
        weights[byte] = 0.01
    for byte, freq in _ENGLISH_BYTE_FREQ.items():
        weights[byte] = freq
        if chr(byte).isalpha():
            weights[ord(chr(byte).upper())] = freq * 0.6  # uppercase is rarer in prose
    return weights


#: Per-byte fitness, precomputed. The column solve evaluates 256 key bytes against a
#: byte histogram, so the score has to be a table lookup rather than a per-byte branch.
_BYTE_WEIGHT = _build_byte_weights()


def english_byte_score(data: bytes) -> float:
    """Higher is more English-like. Non-printable bytes are penalised hard.

    Used to pick XOR key bytes one column at a time, where each column is a Caesar-like
    single-byte problem with only ~n/keysize samples — too few for n-gram fitness, so
    this scores monogram fit plus a printability penalty instead.
    """
    if not data:
        return float("-inf")
    return sum(_BYTE_WEIGHT[b] for b in data) / len(data)


# -- repeating-key XOR ---------------------------------------------------------


def xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR ``data`` against ``key``, repeating the key. Its own inverse."""
    if not key:
        raise ValueError("xor key must be non-empty")
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def _hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=True))


def xor_keysize_candidates(data: bytes, *, max_keysize: int = 40, top: int = 5) -> list[dict]:
    """Rank likely repeating-XOR key lengths by normalised Hamming distance.

    Blocks that are a whole number of key lengths apart were XORed with the same key
    bytes, so their difference is the difference of two English fragments — much smaller
    than for two randomly-offset blocks. Averaging over several block pairs (rather than
    the classic single pair) is what makes this stable on short inputs.
    """
    out: list[dict] = []
    for keysize in range(2, min(max_keysize, len(data) // 4) + 1):
        blocks = [data[i : i + keysize] for i in range(0, len(data), keysize)]
        blocks = [b for b in blocks if len(b) == keysize]
        if len(blocks) < 2:
            continue
        pairs = [(blocks[i], blocks[i + 1]) for i in range(min(len(blocks) - 1, 10))]
        distance = sum(_hamming(a, b) for a, b in pairs) / (len(pairs) * keysize)
        out.append({"keysize": keysize, "normalised_distance": round(distance, 4)})
    out.sort(key=lambda r: r["normalised_distance"])
    return out[:top]


def _best_key_byte(column: bytes) -> int:
    """The single XOR byte that makes ``column`` most English.

    Scored against the column's byte *histogram* rather than the column itself, so the
    cost is 256 x (distinct bytes) instead of 256 x (column length) — the difference
    between sweeping every key length and only being able to afford a handful.
    """
    counts = Counter(column).items()
    return max(
        range(256),
        key=lambda k: sum(count * _BYTE_WEIGHT[byte ^ k] for byte, count in counts),
    )


#: Bits of evidence a key byte must earn to justify itself, expressed on the n-gram
#: scale. Without it the search always runs to ``max_keysize``: every extra key byte is
#: a free parameter that can repaint one ciphertext position per period into whatever
#: the fitness likes, so raw fitness rises monotonically with key length and the
#: "best" key is always the longest one allowed.
_KEYSIZE_PENALTY = 12.0


def _xor_result(data: bytes, key: bytes) -> dict:
    from .scoring import get_scorer

    plain = xor_bytes(data, key)
    printable = all(32 <= b < 127 or b in (9, 10, 13) for b in plain)
    text = plain.decode("ascii", "replace")
    letters = only_letters(text)
    # Quadgram fitness on the recovered letters, not the byte-monogram score used to
    # pick the key bytes: a single repainted position barely moves a monogram sum but
    # wrecks four quadgrams, which is exactly the overfitting this has to see through.
    fitness = get_scorer().score(letters) / max(len(letters), 1) if len(letters) >= 4 else -99.0
    if not printable:
        fitness -= 5.0
    return {
        "keysize": len(key),
        "key": key,
        "key_repr": key.decode("ascii") if printable and key.isascii() else key.hex(),
        "plaintext": text,
        "score": round(fitness, 5),
        "adjusted_score": round(fitness - _KEYSIZE_PENALTY * len(key) / max(len(data), 1), 5),
        "byte_score": round(english_byte_score(plain), 5),
        "printable": printable,
    }


def crack_xor(data: bytes, *, max_keysize: int = 40, top: int = 3) -> list[dict]:
    """Recover a repeating XOR key and plaintext without being told the key length.

    For each candidate length the problem splits into that many independent single-byte
    XORs (one per key position), each solved by maximising :func:`english_byte_score`
    over all 256 possibilities. Returns ``[{"key", "keysize", "plaintext", "score"}]``
    ranked by the score of the whole decrypt.

    Every key length up to ``max_keysize`` is tried, not just the ones the Hamming
    heuristic likes. That heuristic needs several full blocks to be stable and quietly
    prefers multiples of the true length on short inputs; since the histogram-based
    column solve is cheap enough to sweep exhaustively, the final ranking is done on
    the score of the actual decrypt, which cannot be fooled the same way.

    Shorter keys win ties: a key of length ``2k`` that repeats itself decrypts exactly
    as well as the length-``k`` key it is built from, and the shorter one is the answer.

    Each key byte is solved from the ``len(data) / keysize`` bytes in its column, so
    accuracy is set by that ratio and not by the total length: below roughly 25 samples
    per column individual key bytes start coming back wrong, and a 20-byte key therefore
    needs ~500 bytes of ciphertext. Short input does not fail loudly — it returns a key
    that is mostly right — so check the plaintext rather than trusting the rank.
    """
    if len(data) < 8:
        return []
    ceiling = max(1, min(max_keysize, len(data) // 2))
    results = [
        _xor_result(data, bytes(_best_key_byte(data[i::keysize]) for i in range(keysize)))
        for keysize in range(1, ceiling + 1)
    ]
    # Collapse a periodic key (KEYKEY) onto its own period before ranking.
    for result in results:
        key = result["key"]
        for period in range(1, len(key)):
            if len(key) % period == 0 and key == key[:period] * (len(key) // period):
                result.update(_xor_result(data, key[:period]))
                break
    best: dict[bytes, dict] = {}
    for result in results:
        if result["key"] not in best:
            best[result["key"]] = result
    ranked = sorted(best.values(), key=lambda r: (r["adjusted_score"], -r["keysize"]), reverse=True)
    return ranked[:top]


# -- the base-N family ---------------------------------------------------------

BASE26 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _alphabet(base: int) -> str:
    if base == 26:
        return BASE26
    if base <= 36:
        return BASE36[:base]
    if base <= 62:
        return BASE62[:base]
    raise ValueError(f"unsupported base {base} (2..62)")


def base_n_encode(number: int, base: int) -> str:
    """A non-negative integer in ``base``, most-significant digit first."""
    if number < 0:
        raise ValueError("base-N encoding is defined for non-negative integers")
    digits = _alphabet(base)
    if number == 0:
        return digits[0]
    out: list[str] = []
    while number:
        number, rem = divmod(number, base)
        out.append(digits[rem])
    return "".join(reversed(out))


def base_n_decode(text: str, base: int) -> int:
    """Inverse of :func:`base_n_encode`. Case-insensitive below base 37."""
    digits = _alphabet(base)
    body = re.sub(r"\s+", "", text)
    if base <= 36:
        body = body.upper()
    value = 0
    for ch in body:
        idx = digits.find(ch)
        if idx < 0:
            raise ValueError(f"{ch!r} is not a base-{base} digit")
        value = value * base + idx
    return value


def base32_encode(data: str | bytes) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return base64.b32encode(raw).decode("ascii")


def base32_decode(text: str) -> bytes:
    body = re.sub(r"\s+", "", text).upper()
    body += "=" * (-len(body) % 8)
    return base64.b32decode(body)


def base85_encode(data: str | bytes) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return base64.b85encode(raw).decode("ascii")


def base85_decode(text: str) -> bytes:
    return base64.b85decode(re.sub(r"\s+", "", text))


# -- ROT variants --------------------------------------------------------------


def rot_n(text: str, n: int) -> str:
    """Caesar rotation on letters only, leaving everything else alone."""
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + n) % 26 + 65))
        elif "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + n) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)


def rot5(text: str) -> str:
    """Rotate digits by 5. Its own inverse."""
    return "".join(chr((ord(c) - 48 + 5) % 10 + 48) if c.isdigit() else c for c in text)


def rot13(text: str) -> str:
    return rot_n(text, 13)


def rot18(text: str) -> str:
    """ROT13 on letters and ROT5 on digits at once. Its own inverse."""
    return rot5(rot13(text))


def rot47(text: str) -> str:
    """Rotate the printable ASCII range 33..126 by 47. Its own inverse.

    Unlike ROT13 this moves punctuation and digits too, so it is the variant that
    survives base64 or URL-encoded payloads intact.
    """
    return "".join(chr(33 + (ord(c) - 33 + 47) % 94) if 33 <= ord(c) <= 126 else c for c in text)


# -- keyboard geometry ---------------------------------------------------------

QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def keyboard_shift(text: str, n: int = 1) -> str:
    """Shift each letter ``n`` places along its QWERTY row (wrapping at the ends)."""
    out = []
    for ch in text:
        low = ch.lower()
        for row in QWERTY_ROWS:
            if low in row:
                moved = row[(row.index(low) + n) % len(row)]
                out.append(moved.upper() if ch.isupper() else moved)
                break
        else:
            out.append(ch)
    return "".join(out)


def keyboard_coordinates(text: str) -> str:
    """Each letter as ``row,column`` on QWERTY, 1-indexed (Q = 1,1)."""
    out = []
    for ch in only_letters(text).lower():
        for r, row in enumerate(QWERTY_ROWS, start=1):
            if ch in row:
                out.append(f"{r}{row.index(ch) + 1}")
                break
    return " ".join(out)


def from_keyboard_coordinates(text: str) -> str:
    out = []
    for token in re.findall(r"\d\d?", re.sub(r"[^\d]", " ", text)):
        if len(token) != 2:
            continue
        r, c = int(token[0]) - 1, int(token[1]) - 1
        if 0 <= r < len(QWERTY_ROWS) and 0 <= c < len(QWERTY_ROWS[r]):
            out.append(QWERTY_ROWS[r][c].upper())
    return "".join(out)


# -- phone keypad --------------------------------------------------------------

KEYPAD = {
    "2": "ABC",
    "3": "DEF",
    "4": "GHI",
    "5": "JKL",
    "6": "MNO",
    "7": "PQRS",
    "8": "TUV",
    "9": "WXYZ",
}
_KEYPAD_INDEX = {
    ch: (digit, i + 1) for digit, group in KEYPAD.items() for i, ch in enumerate(group)
}


def multitap_encode(text: str, *, sep: str = " ") -> str:
    """Old-phone multi-tap: C is the third letter on key 2, so ``222``."""
    taps = (_KEYPAD_INDEX[c] for c in only_letters(text))
    return sep.join(digit * count for digit, count in taps)


def multitap_decode(text: str) -> str:
    """Decode multi-tap runs. Needs the group separator that ``multitap_encode`` emits."""
    out = []
    for token in re.findall(r"\d+", text):
        digit = token[0]
        if digit in KEYPAD and all(c == digit for c in token) and len(token) <= len(KEYPAD[digit]):
            out.append(KEYPAD[digit][len(token) - 1])
    return "".join(out)


def t9_encode(text: str) -> str:
    """T9 predictive: one digit per letter, so the encoding is deliberately lossy."""
    return "".join(_KEYPAD_INDEX[c][0] for c in only_letters(text))


def t9_candidates(digits: str) -> int:
    """How many letter strings a T9 digit run could stand for (its ambiguity)."""
    total = 1
    for d in re.sub(r"[^2-9]", "", digits):
        total *= len(KEYPAD[d])
    return total


# -- tap code ------------------------------------------------------------------

_TAP_ALPHABET = "ABCDEFGHIJLMNOPQRSTUVWXYZ"  # K is folded into C, the POW convention


def tap_encode(text: str, *, dot: str = ".", sep: str = " ") -> str:
    """Prisoner-of-war tap code: row taps, pause, column taps. 5x5 grid, K -> C."""
    out = []
    for ch in only_letters(text).upper().replace("K", "C"):
        idx = _TAP_ALPHABET.index(ch)
        row, col = divmod(idx, 5)
        out.append(f"{dot * (row + 1)}{sep}{dot * (col + 1)}")
    return "  ".join(out)


def tap_decode(text: str) -> str:
    runs = re.findall(r"[.\-•]+", text)
    out = []
    for i in range(0, len(runs) - 1, 2):
        row, col = len(runs[i]) - 1, len(runs[i + 1]) - 1
        if 0 <= row < 5 and 0 <= col < 5:
            out.append(_TAP_ALPHABET[row * 5 + col])
    return "".join(out)


# -- spelled-out letters -------------------------------------------------------

NATO = (
    "ALFA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIETT KILO LIMA MIKE "
    "NOVEMBER OSCAR PAPA QUEBEC ROMEO SIERRA TANGO UNIFORM VICTOR WHISKEY XRAY YANKEE ZULU"
).split()
#: Prefixes that identify a NATO word, used only to *detect* a NATO payload.
#: Both spellings of the contested words are accepted (ALFA/ALPHA, JULIETT/JULIET,
#: WHISKEY/WHISKY, XRAY/X-RAY), which is why matching is by two letters, not three.
_NATO_PREFIX = {word[:2] for word in NATO} | {"AL", "JU", "WH", "XR"}


def nato_encode(text: str) -> str:
    return " ".join(NATO[ord(c) - 65] for c in only_letters(text))


def nato_decode(text: str) -> str:
    """Take the first letter of each word.

    Not a prefix table: every NATO word begins with the letter it stands for, so the
    initial *is* the decode and every spelling variant works for free — ALFA and ALPHA,
    JULIETT and JULIET, WHISKEY and WHISKY.
    """
    return "".join(w[0] for w in re.findall(r"[A-Za-z]+", text.upper()))


# -- generic symbol transcription ----------------------------------------------

#: Unicode Braille (U+2800 block), grade 1 letters a-z.
BRAILLE = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵",
        strict=True,
    )
)


class Transcriptor:
    """A symbol alphabet as a lookup table — the generic form of the whole novelty set.

    dCode carries ~180 separate symbol alphabets (Aurebesh, Hylian, Wingdings, runes,
    every fandom script). None of them is cryptanalysis: each is a bijection between a
    glyph set and A-Z, and a solver only needs to apply the table it is handed. One
    class with a supplied table covers all of them, so only Braille ships built in.
    """

    def __init__(self, table: dict[str, str]):
        self.table = {k.upper(): v for k, v in table.items()}
        self.inverse = {v: k for k, v in self.table.items()}
        if len(self.inverse) != len(self.table):
            raise ValueError("transcription table is not one-to-one")

    def encode(self, text: str) -> str:
        return "".join(self.table.get(ch, ch) for ch in text.upper())

    def decode(self, text: str) -> str:
        return "".join(self.inverse.get(ch, ch) for ch in text)

    @classmethod
    def from_pairs(cls, spec: str) -> Transcriptor:
        """Build from a ``"A=⠁,B=⠃"`` spec so a table can come off the command line."""
        table = {}
        for item in re.split(r"[,\n]", spec):
            item = item.strip()
            if not item:
                continue
            letter, _, glyph = item.partition("=")
            if not glyph:
                raise ValueError(f"transcription entry {item!r} must be LETTER=GLYPH")
            table[letter.strip()] = glyph.strip()
        return cls(table)


BRAILLE_TRANSCRIPTOR = Transcriptor(BRAILLE)


# -- detection -----------------------------------------------------------------


def _plausible(text: str) -> bool:
    """True when a decode looks like it went somewhere — letters, not line noise."""
    if len(text) < 4:
        return False
    letters = sum(c.isalpha() for c in text)
    printable = sum(32 <= ord(c) < 127 or c in "\n\r\t" for c in text)
    return printable / len(text) > 0.95 and letters / len(text) > 0.5


#: Dictionary coverage a letters-to-letters decode must reach to be reported. Ordinary
#: English without spaces scores ~0.43; unrelated letters score 0.0.
_ENGLISH_COVERAGE_MIN = 0.25


def _reads_as_english(text: str) -> bool:
    from . import words

    return words.long_word_coverage(text) >= _ENGLISH_COVERAGE_MIN


def detect(raw: str) -> list[dict]:
    """Every wrapper that plausibly decodes ``raw``, as ``[{"kind", "text", "note"}]``.

    Conservative on purpose. A plain uppercase A-Z ciphertext is valid base32, valid
    base36 and valid NATO-prefix soup, so each decoder must find a signal a letter
    cipher would not carry — a digit, a lowercase run, a glyph outside A-Z — before it
    is allowed to report. The alternative is burying the real cipher under a dozen
    confident non-answers.
    """
    s = raw.strip()
    compact = re.sub(r"\s+", "", s)
    out: list[dict] = []

    def offer(kind: str, text: str, note: str = "", *, needs_english: bool = False) -> None:
        if not text or not _plausible(text):
            return
        # Letters-in / letters-out wrappers (keyboard shift, ROT on plain text) always
        # produce something that "looks like" a decode, so structural plausibility says
        # nothing about them. They have to earn their place by reading as English —
        # otherwise every A-Z ciphertext comes back with two confident non-answers.
        if needs_english and not _reads_as_english(text):
            return
        out.append({"kind": kind, "text": text, "note": note})

    # base32: the padding and the A-Z2-7 restriction are the signal.
    if re.fullmatch(r"[A-Z2-7]+=*", compact) and re.search(r"[2-7=]", compact):
        try:
            offer("base32", base32_decode(compact).decode("ascii", "replace"))
        except (binascii.Error, ValueError):
            pass

    # base85 needs characters outside the base64 set to be distinguishable.
    if len(compact) >= 8 and re.search(r"[!#$%&()*+\-;<=>?@^_`{|}~]", compact):
        try:
            offer("base85", base85_decode(compact).decode("ascii", "replace"))
        except (binascii.Error, ValueError):
            pass

    # ROT47 only makes sense when there is punctuation to rotate.
    if re.search(r"[!-/:-@\[-`{-~]", s):
        offer("rot47", rot47(s), needs_english=True)

    # Multi-tap: runs of one repeated keypad digit.
    if re.fullmatch(r"[2-9\s]+", s) and re.search(r"(\d)\1", compact):
        offer("multitap", multitap_decode(s))

    # Keyboard coordinates: two-digit groups with a row digit of 1-3.
    if re.fullmatch(r"[1-3][0-9](\s+[1-3][0-9])+", s.strip()):
        offer("keyboard-coordinates", from_keyboard_coordinates(s))

    # Tap code: runs of dots.
    if re.search(r"[.\-•]{1,5}\s", s) and not re.search(r"[A-Za-z]", s):
        offer("tap-code", tap_decode(s))

    # NATO: spelled-out words that are (almost) all in the alphabet.
    words = re.findall(r"[A-Za-z]{3,}", s)
    if len(words) >= 3 and sum(w[:2].upper() in _NATO_PREFIX for w in words) >= 0.8 * len(words):
        offer("nato", nato_decode(s))

    # Braille glyphs are unmistakable — no ambiguity gate needed.
    if any(ch in BRAILLE_TRANSCRIPTOR.inverse for ch in s):
        offer("braille", BRAILLE_TRANSCRIPTOR.decode(s))

    # Keyboard shift: both directions, but only when the result reads as English.
    if re.fullmatch(r"[A-Za-z\s]+", s):
        for n in (-1, 1):
            offer("keyboard-shift", keyboard_shift(s, n), f"shift {n:+d}", needs_english=True)

    return out


def detect_xor(raw: str, *, max_keysize: int = 40) -> list[dict]:
    """XOR candidates for ``raw``, read as hex or base64 bytes then attacked.

    Kept out of :func:`detect` because it is a *search*, not a decode: it always
    returns its best guess, so folding it into the conservative detection pass would
    put a speculative answer next to deterministic ones.
    """
    compact = re.sub(r"\s+", "", raw.strip())
    data: bytes | None = None
    if re.fullmatch(r"[0-9a-fA-F]+", compact) and len(compact) % 2 == 0:
        data = bytes.fromhex(compact)
    elif re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact) and len(compact) % 4 == 0:
        try:
            data = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError):
            data = None
    if data is None:
        data = raw.encode("utf-8", "replace")
    return crack_xor(data, max_keysize=max_keysize)


def letter_histogram(text: str) -> dict[str, int]:
    """Letter counts, for callers deciding whether a peel produced cipherable text."""
    return dict(Counter(only_letters(text)))


# -- named dispatch ------------------------------------------------------------


def _b(data: bytes) -> str:
    return data.decode("ascii", "replace")


#: ``kind -> (peel, wrap)``. ``peel`` undoes the wrapper, ``wrap`` applies it. Kinds
#: taking a numeric argument are written ``kind:N`` and receive it as ``arg``.
_WRAPPERS: dict[str, tuple] = {
    "rot47": (rot47, rot47),
    "rot5": (rot5, rot5),
    "rot18": (rot18, rot18),
    "rot13": (rot13, rot13),
    "base32": (lambda t: _b(base32_decode(t)), base32_encode),
    "base85": (lambda t: _b(base85_decode(t)), base85_encode),
    "base64": (
        lambda t: _b(base64.b64decode(re.sub(r"\s+", "", t))),
        lambda t: base64.b64encode(t.encode()).decode(),
    ),
    "hex": (lambda t: _b(bytes.fromhex(re.sub(r"\s+", "", t))), lambda t: t.encode().hex()),
    "nato": (nato_decode, nato_encode),
    "tap-code": (tap_decode, tap_encode),
    "multitap": (multitap_decode, multitap_encode),
    "t9": (None, t9_encode),  # one-way: T9 is lossy by construction
    "keyboard-coordinates": (from_keyboard_coordinates, keyboard_coordinates),
    "braille": (BRAILLE_TRANSCRIPTOR.decode, BRAILLE_TRANSCRIPTOR.encode),
}

#: Kinds parameterised by an integer, written ``kind:N``.
_PARAMETRIC: dict[str, tuple] = {
    "rot": (lambda t, n: rot_n(t, -n), rot_n),
    "keyboard-shift": (lambda t, n: keyboard_shift(t, -n), keyboard_shift),
    "base": (
        lambda t, n: str(base_n_decode(t, n)),
        lambda t, n: base_n_encode(int(re.sub(r"\D", "", t) or 0), n),
    ),
}


def wrapper_kinds() -> list[str]:
    """Every ``--apply`` / ``--wrap`` name, parametric ones shown as ``kind:N``."""
    return sorted(_WRAPPERS) + sorted(f"{k}:N" for k in _PARAMETRIC)


def apply_wrapper(kind: str, text: str, *, encode: bool = False) -> str:
    """Peel (or with ``encode``, apply) the named wrapper. ``kind`` may be ``name:N``."""
    name, _, arg = kind.partition(":")
    name = name.strip().lower()
    if name in _PARAMETRIC:
        if not arg.strip().lstrip("-").isdigit():
            raise ValueError(f"{name} needs a numeric argument, e.g. '{name}:3'")
        peel, wrap = _PARAMETRIC[name]
        return (wrap if encode else peel)(text, int(arg))
    if name not in _WRAPPERS:
        raise ValueError(f"unknown wrapper {kind!r}; try one of: {', '.join(wrapper_kinds())}")
    peel, wrap = _WRAPPERS[name]
    fn = wrap if encode else peel
    if fn is None:
        raise ValueError(f"{name} is one-way (encode only) — it discards information")
    return fn(text)
