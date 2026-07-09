"""Structured results — the machine-readable contract that makes butt agent-friendly.

Every command resolves to a :class:`CrackResult` that serializes to a stable JSON
schema. Agents consume the JSON; humans get a rendered table from the same object.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

# Bump when the JSON envelope shape changes; stamped on every result so agents
# can guard against a contract they don't understand.
# v2: added `word_coverage` (long-word language-fit signal) to candidates + envelope.
# v3: `butt schema` manifest now documents the stats/identify output contracts
#     (command_outputs) and the conditional crib_confirmed/ambiguous_with fields.
SCHEMA_VERSION = 3

#: every value `verdict` can take (encode/decode emit "n/a" — there's nothing to crack)
VERDICT_VALUES = ("solved", "likely", "ambiguous", "unlikely", "no-candidates", "n/a")

# Verdict thresholds on the best candidate's calibrated confidence. `margin` is
# the confidence gap to the runner-up; a small margin means two ciphers fit
# about equally well, so we never call it "solved".
VERDICT_SOLVED = 0.85
VERDICT_LIKELY = 0.65
VERDICT_AMBIGUOUS = 0.40
SOLVED_MARGIN = 0.05


@dataclass
class Candidate:
    """A single decryption hypothesis."""

    plaintext: str
    cipher: str
    key: str | None = None
    score: float = 0.0  # raw fitness (n-gram log-prob; higher = better)
    confidence: float = 0.0  # calibrated 0..1 (already deflated by word_coverage)
    # Fraction of the plaintext tiled by real >=5-letter words: a language-fit
    # signal the n-gram score can't see. ~0.5-0.8 for real English, ~0 for
    # quadgram "salad". None when not computed (non-English lang, or text too
    # short to judge). Read it to tell "low confidence: too short" from
    # "low confidence: scores like English but isn't language".
    word_coverage: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Ranking-only nudge (e.g. Occam prior in `auto`); never displayed.
    rank_bias: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plaintext": self.plaintext,
            "cipher": self.cipher,
            "key": self.key,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "word_coverage": (
                round(self.word_coverage, 4) if self.word_coverage is not None else None
            ),
            "meta": self.meta,
        }


@dataclass
class CrackResult:
    """The outcome of an encode/decode/crack/auto invocation."""

    cipher: str  # cipher attempted, or "auto"
    ciphertext: str
    operation: str = "crack"  # encode | decode | crack | auto
    candidates: list[Candidate] = field(default_factory=list)
    runtime_ms: float = 0.0
    notes: list[str] = field(default_factory=list)
    identify: dict[str, Any] | None = None  # routing hints, when present

    def sorted_candidates(self) -> list[Candidate]:
        # rank_bias breaks near-ties (0 for single-cipher cracks); score is the
        # final tiebreak. Displayed confidence is untouched.
        return sorted(
            self.candidates,
            key=lambda c: (c.confidence + c.rank_bias, c.score),
            reverse=True,
        )

    def best(self) -> Candidate | None:
        ranked = self.sorted_candidates()
        return ranked[0] if ranked else None

    def margin(self) -> float | None:
        """Confidence gap between the best and runner-up candidate."""
        ranked = self.sorted_candidates()
        if not ranked:
            return None
        if len(ranked) == 1:
            return ranked[0].confidence
        return ranked[0].confidence - ranked[1].confidence

    def verdict(self) -> str:
        """Honest trust signal: solved | likely | ambiguous | unlikely | no-candidates.

        Not meaningful for encode/decode (there is nothing to crack).
        """
        if self.operation in ("encode", "decode"):
            return "n/a"
        best = self.best()
        if best is None:
            return "no-candidates"
        conf, margin = best.confidence, (self.margin() or 0.0)
        if conf >= VERDICT_SOLVED and margin >= SOLVED_MARGIN:
            return "solved"
        if conf >= VERDICT_LIKELY:
            return "likely"
        if conf >= VERDICT_AMBIGUOUS:
            return "ambiguous"
        return "unlikely"

    def to_dict(self, top: int | None = None) -> dict[str, Any]:
        ranked = self.sorted_candidates()
        if top is not None:
            ranked = ranked[:top]
        best = ranked[0] if ranked else None
        verdict = self.verdict()
        margin = self.margin()
        # `plaintext`/`ciphertext` always name the plaintext-side / cipher-side
        # text regardless of direction: for encode the input is the plaintext and
        # the result is the ciphertext; for decode/crack/auto it is the reverse.
        result_text = best.plaintext if best else None
        plaintext_field: str | None
        ciphertext_field: str | None
        if self.operation == "encode":
            plaintext_field, ciphertext_field = self.ciphertext, result_text
        else:
            plaintext_field, ciphertext_field = result_text, self.ciphertext
        out: dict[str, Any] = {
            "ok": best is not None,
            "schema_version": SCHEMA_VERSION,
            "operation": self.operation,
            "verdict": verdict,
            "cipher": self.cipher,
            "ciphertext": ciphertext_field,
            "plaintext": plaintext_field,
            "key": best.key if best else None,
            "score": round(best.score, 4) if best else None,
            "confidence": round(best.confidence, 4) if best else None,
            "word_coverage": (
                round(best.word_coverage, 4)
                if best is not None and best.word_coverage is not None
                else None
            ),
            "margin": round(margin, 4) if margin is not None else None,
            "runtime_ms": round(self.runtime_ms, 3),
            "candidate_count": len(ranked),
            "candidates": [c.to_dict() for c in ranked],
        }
        # A crib match is strong, cipher-agnostic confirmation — surface it.
        if best is not None and isinstance(best.meta, dict) and best.meta.get("crib_confirmed"):
            out["crib_confirmed"] = True
        # When the top two ciphers are within SOLVED_MARGIN, name the rival.
        if best is not None and len(ranked) > 1 and verdict in ("ambiguous", "unlikely"):
            if ranked[0].confidence - ranked[1].confidence < SOLVED_MARGIN:
                out["ambiguous_with"] = ranked[1].cipher
        if self.identify is not None:
            out["identify"] = self.identify
        if self.notes:
            out["notes"] = self.notes
        return out


def asdict(obj: Any) -> Any:
    """dataclasses.asdict shim used by tests/tools."""
    return dataclasses.asdict(obj)
