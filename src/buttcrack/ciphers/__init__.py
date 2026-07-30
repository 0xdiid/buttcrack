"""Cipher implementations."""

from __future__ import annotations

from .adfgvx import ADFGVX
from .adfgx import ADFGX
from .affine import Affine
from .amsco import Amsco
from .atbash import Atbash
from .autokey import Autokey
from .bacon import Bacon
from .base import Cipher
from .bazeries import Bazeries
from .beaufort import Beaufort
from .bifid import Bifid
from .cadenus import Cadenus
from .chaocipher import Chaocipher
from .caesar import Caesar, Rot13
from .checkerboard import Checkerboard
from .cm_bifid import CMBifid
from .collon import Collon
from .columnar import Columnar
from .compressocrat import Compressocrat
from .condi import Condi
from .digrafid import Digrafid
from .enigma import Enigma
from .four_square import FourSquare
from .fractionated_morse import FractionatedMorse
from .grandpre import Grandpre
from .grille import Grille
from .gromark import Gromark
from .gronsfeld import Gronsfeld
from .headline import Headline
from .hill import Hill
from .homophonic import Homophonic
from .incomplete_columnar import IncompleteColumnar
from .interrupted_key import InterruptedKey
from .josse import Josse
from .key_phrase import KeyPhrase
from .keystream import LcgKeystream, LinearRecurrenceKeystream
from .monome_dinome import MonomeDinome
from .morbit import Morbit
from .m94 import M94
from .myszkowski import Myszkowski
from .nicodemus import Nicodemus
from .nihilist_substitution import NihilistSubstitution
from .nihilist_transposition import NihilistTransposition
from .null_cipher import NullCipher
from .numbered_key import NumberedKey
from .periodic_gromark import PeriodicGromark
from .phillips import Phillips
from .playfair import Playfair
from .pollux import Pollux
from .polybius import Polybius
from .porta import Porta
from .portax import Portax
from .progressive_key import ProgressiveKey
from .quagmire1 import QuagmireI
from .quagmire2 import QuagmireII
from .quagmire3 import QuagmireIII
from .quagmire4 import QuagmireIV
from .ragbaby import Ragbaby
from .railfence import RailFence
from .rectangular_playfair import RectangularPlayfair
from .redefence import Redefence
from .route import Route
from .running_key import RunningKey
from .sequence_transposition import SequenceTransposition
from .seriated_playfair import SeriatedPlayfair
from .slidefair import Slidefair
from .straddling_checkerboard import StraddlingCheckerboard
from .substitution import Substitution
from .swagman import Swagman
from .syllabary import Syllabary
from .tri_square import TriSquare
from .tridigital import Tridigital
from .trifid import Trifid
from .two_square import TwoSquare
from .ubchi import Ubchi
from .variant_beaufort import VariantBeaufort
from .vigenere import Vigenere

#: All cipher classes, in a sensible default crack order (cheap/common first).
ALL_CIPHERS: tuple[type[Cipher], ...] = (
    Caesar,
    Rot13,
    Atbash,
    Affine,
    Vigenere,
    Beaufort,
    VariantBeaufort,
    Gronsfeld,
    Autokey,
    RunningKey,
    ProgressiveKey,
    InterruptedKey,
    Porta,
    Portax,
    QuagmireI,
    QuagmireII,
    QuagmireIII,
    QuagmireIV,
    Slidefair,
    Bacon,
    Playfair,
    RectangularPlayfair,
    SeriatedPlayfair,
    FourSquare,
    TwoSquare,
    TriSquare,
    Bifid,
    CMBifid,
    Trifid,
    ADFGX,
    ADFGVX,
    FractionatedMorse,
    Morbit,
    Pollux,
    Tridigital,
    StraddlingCheckerboard,
    NihilistSubstitution,
    Checkerboard,
    MonomeDinome,
    Digrafid,
    Polybius,
    Collon,
    Phillips,
    Compressocrat,
    Syllabary,
    NumberedKey,
    Grandpre,
    Homophonic,
    KeyPhrase,
    Headline,
    NullCipher,
    RailFence,
    Columnar,
    IncompleteColumnar,
    Myszkowski,
    Amsco,
    Route,
    Redefence,
    Ubchi,
    NihilistTransposition,
    Nicodemus,
    Cadenus,
    Swagman,
    Grille,
    SequenceTransposition,
    Hill,
    Bazeries,
    M94,
    Chaocipher,
    Enigma,
    Gromark,
    PeriodicGromark,
    LinearRecurrenceKeystream,
    LcgKeystream,
    Ragbaby,
    Condi,
    Substitution,
    Josse,
)

__all__ = [
    "Cipher",
    "ALL_CIPHERS",
    *[c.__name__ for c in ALL_CIPHERS],
]
