"""The Cipher contract every solver implements."""

from __future__ import annotations

import abc

from ..result import Candidate
from ..scoring import NgramScorer


class Cipher(abc.ABC):
    """Base class for a classical cipher.

    Subclasses provide reversible ``encode``/``decode`` and a keyless (or
    near-keyless) ``crack`` that returns ranked :class:`Candidate` hypotheses.
    """

    #: canonical lowercase name, used as the subcommand (e.g. "vigenere")
    name: str = ""
    #: alternative names accepted on the CLI
    aliases: tuple[str, ...] = ()
    #: one-line human description
    description: str = ""
    #: concise, human/agent-readable description of the --key format, e.g.
    #: "keyword" or "primer (5 digits)/keyword". Surfaced by `list`/`help`.
    key_format: str = ""
    #: a concrete, working example key (surfaced by `butt help <cipher>`)
    key_example: str = ""
    #: whether encode/decode require a --key
    needs_key: bool = True
    #: relative keyspace/overfitting risk; used as an Occam prior in `auto`
    #: (lower = simpler, preferred when confidences are close)
    complexity: int = 3
    #: whether `auto` should run this cipher's keyless crack. Set False when the
    #: keyless attack is ill-posed and yields confident garbage (e.g. running key,
    #: where the key is as long as the text) — still usable via explicit `crack`.
    auto_crackable: bool = True
    #: the alphabet a *ciphertext* of this cipher must be written in, when it's a
    #: strict small set (e.g. ADFGX over "ADFGX"). Used to reject input that can't
    #: be this cipher, so the cracker can't "solve" the few matching letters of an
    #: unrelated message and post a confident false positive. None = unrestricted.
    ciphertext_alphabet: str | None = None

    @abc.abstractmethod
    def encode(self, text: str, key: str) -> str:
        """Encrypt ``text`` with ``key``."""

    @abc.abstractmethod
    def decode(self, text: str, key: str) -> str:
        """Decrypt ``text`` with ``key``."""

    @abc.abstractmethod
    def crack(
        self,
        text: str,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Return ranked candidate decryptions without being given the key."""

    # -- shared helpers --------------------------------------------------
    def ciphertext_alphabet_ok(self, text: str, threshold: float = 0.9) -> bool:
        """False iff ``ciphertext_alphabet`` is set and ``text`` isn't (mostly) over it.

        Cheap structural sanity check: a 280-letter message with only 46 A/D/F/G/X
        letters cannot be ADFGX, so don't let the cracker pretend it solved those 46.
        """
        alpha = self.ciphertext_alphabet
        if not alpha:
            return True
        chars = [c for c in text.upper() if c.isalnum()]
        if not chars:
            return False
        return sum(c in alpha for c in chars) / len(chars) >= threshold

    def _candidate(self, scorer: NgramScorer, plaintext: str, key, **meta) -> Candidate:
        return Candidate(
            plaintext=plaintext,
            cipher=self.name,
            key=None if key is None else str(key),
            score=scorer.score(plaintext),
            confidence=scorer.confidence(plaintext),
            meta=meta,
        )
