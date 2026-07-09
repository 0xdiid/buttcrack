"""Variant Beaufort cipher: C = (P - K) mod 26; decrypt P = (C + K) mod 26.

The inverse-direction Vigenere (encrypt and decrypt swap the Vigenere roles).
"""

from __future__ import annotations

from ._periodic import PeriodicCipher


class VariantBeaufort(PeriodicCipher):
    name = "variant-beaufort"
    aliases = ("variant", "varbeaufort")
    description = "Polyalphabetic; C = (P - K) mod 26 (inverse-direction Vigenere)."
    key_format = "keyword (letters)"
    key_example = "cipher"
    complexity = 3

    def _enc(self, shift: int, p: int) -> int:
        return p - shift

    def _dec(self, shift: int, c: int) -> int:
        return c + shift
