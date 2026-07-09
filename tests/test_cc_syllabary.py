"""Tests for the Syllabary cipher.

VECTOR source (self-consistent, verified by encode):

  Standard unmixed syllabary square with sequential coordinates 0-9. The square
  is documented identically by the ACA cipher sheet
  (cryptogram.org/.../aca.info/ciphers/Syllabary.pdf -- "Known coordinates should
  be entered using the sequence 0-9, left to right, top to bottom"; standard
  syllabary alphabet given in the appendices), by the CryptoCrack user guide
  (sites.google.com/site/cryptocrackprogram/.../other/syllabary) and by
  kopaldev.de / sites.google.com/site/bionspot (G-MAN, the cipher's author).

  Square (row-major), rows/cols 0-9:
      A 1 AL AN AND AR ARE AS AT ATE / ATI B 2 BE C 3 CA CE CO COM /
      D 4 DA DE E 5 EA ED EN ENT / ER ERE ERS ES EST F 6 G 7 H /
      8 HAS HE I 9 IN ING ION IS IT / IVE J 0 K L LA LE M ME N /
      ND NE NT O OF ON OR OU P Q / R RA RE RED RES RI RO S SE SH /
      ST STO T TE TED TER TH THE THI THR / TI TO U V VE W WE X Y Z

  With sequential coordinates the code for a cell is (row)(col). Greedy
  longest-match parse of "A FRIEND IS ONE" is A | F | RI | EN | D | IS | ON | E,
  i.e. cells (0,0)(3,5)(7,5)(2,8)(2,0)(4,8)(6,5)(2,4):

      encode("A FRIEND IS ONE", "") == "00 35 75 28 20 48 65 24"

  Independently corroborated: CryptoCrack's own worked example lists 35 75 28 20
  as a valid spelling of FRIEND (F=35, RI=75, EN=28, D=20), matching these cells.
"""

from buttcrack.ciphers.syllabary import Syllabary

VECTOR_PT = "A FRIEND IS ONE"
VECTOR_KEY = ""  # standard unmixed square, sequential coordinates 0-9
VECTOR_CT = "00 35 75 28 20 48 65 24"


def test_vector_standard_square():
    c = Syllabary()
    assert c.encode(VECTOR_PT, VECTOR_KEY) == VECTOR_CT


def test_round_trip_standard():
    c = Syllabary()
    msg = "Defend the east wall of the castle"
    prepared = "".join(ch for ch in msg.upper() if ch.isalnum())
    ct = c.encode(msg, VECTOR_KEY)
    assert c.decode(ct, VECTOR_KEY) == prepared


def test_round_trip_keyed_square_and_coordinates():
    # Mixed square (keyword) plus scrambled row/column coordinate digits --
    # exercises all three ACA keying degrees of freedom at once.
    c = Syllabary()
    key = "REPLACING 1829357046 7031465982"
    msg = "The quick brown fox jumps over the lazy dog"
    prepared = "".join(ch for ch in msg.upper() if ch.isalnum())
    ct = c.encode(msg, key)
    assert c.decode(ct, key) == prepared


def test_output_is_digit_pairs():
    c = Syllabary()
    ct = c.encode("A FRIEND IS ONE", VECTOR_KEY)
    pairs = ct.split()
    assert all(len(p) == 2 and p.isdigit() for p in pairs)
    # 8 tokens parsed from the plaintext -> 8 coordinate pairs.
    assert len(pairs) == 8
