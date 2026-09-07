"""Replayable candidate audits and narrowly scoped search completion receipts.

These records describe a supplied candidate universe, not a cipher family. An
exhausted bank is a bookkeeping result, not a statistical or mathematical proof.
No solver success rate or probability is inferred from these records.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal

Outcome = Literal[
    "exhausted_bank",
    "gate_failed",
    "timed_out",
    "capped",
    "interrupted",
    "exact_contradiction",
    "candidate_found",
    "statistical_mismatch",
]
_OUTCOMES = {
    "exhausted_bank",
    "gate_failed",
    "timed_out",
    "capped",
    "interrupted",
    "exact_contradiction",
    "candidate_found",
    "statistical_mismatch",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CandidateDecision:
    """One input occurrence, including rejected and duplicate occurrences."""

    index: int
    supplied: str | tuple[int, ...]
    normalized: str | tuple[int, ...]
    reason: str | None = None


@dataclass(frozen=True)
class CandidateAudit:
    """Complete ordered input and decisions, sufficient to replay the audit.

    ``sha256`` covers the kind, filters, raw inputs, and decisions. Thus even a
    change to an excluded input changes the digest. Accepted order is preserved.
    """

    kind: str
    filters: tuple[tuple[str, str | int | bool | None], ...]
    decisions: tuple[CandidateDecision, ...]

    @property
    def accepted(self) -> tuple[str | tuple[int, ...], ...]:
        return tuple(d.normalized for d in self.decisions if d.reason is None)

    @property
    def excluded_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(d.reason for d in self.decisions if d.reason).items()))

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "accepted": list(self.accepted),
            "supplied_count": len(self.decisions),
            "accepted_count": len(self.accepted),
            "excluded_counts": self.excluded_counts,
            "sha256": self.sha256,
        }


def audit_word_bank(
    words: Iterable[str],
    *,
    min_length: int = 1,
    max_length: int | None = None,
    alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
) -> CandidateAudit:
    """Trim and uppercase words, recording the first applicable exclusion.

    Rejection precedence is empty, invalid symbols, too short, too long, duplicate.
    Internal punctuation/whitespace and non-ASCII text are rejected, not silently
    stripped or transliterated. The alphabet must contain distinct ASCII capitals.
    """
    _positive_int(min_length, "min_length")
    if max_length is not None:
        _positive_int(max_length, "max_length")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
    if (
        not alphabet
        or len(set(alphabet)) != len(alphabet)
        or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in alphabet)
    ):
        raise ValueError("alphabet must contain distinct ASCII uppercase letters")
    seen: set[str] = set()
    decisions = []
    for index, supplied in enumerate(words):
        if not isinstance(supplied, str):
            raise TypeError("word bank entries must be strings")
        normalized = supplied.strip().upper()
        reason = None
        if not normalized:
            reason = "empty"
        elif not supplied.strip().isascii() or any(c not in alphabet for c in normalized):
            reason = "invalid_symbols"
        elif len(normalized) < min_length:
            reason = "too_short"
        elif max_length is not None and len(normalized) > max_length:
            reason = "too_long"
        elif normalized in seen:
            reason = "duplicate"
        else:
            seen.add(normalized)
        decisions.append(CandidateDecision(index, supplied, normalized, reason))
    return CandidateAudit(
        "word_bank",
        (
            ("normalization", "strip_upper_ascii"),
            ("alphabet", alphabet),
            ("min_length", min_length),
            ("max_length", max_length),
        ),
        tuple(decisions),
    )


def audit_period_tuples(
    candidates: Iterable[Iterable[int]],
    *,
    require_pairwise_coprime: bool = False,
) -> CandidateAudit:
    """Audit explicitly supplied period tuples; shared factors are allowed by default.

    Preserve tuple order: component ordering may matter to a caller. Only identical
    ordered tuples count as duplicates. Malformed periods raise instead of creating
    an apparently valid search universe.
    """
    if not isinstance(require_pairwise_coprime, bool):
        raise TypeError("require_pairwise_coprime must be boolean")
    decisions = []
    seen: set[tuple[int, ...]] = set()
    for index, candidate in enumerate(candidates):
        periods = tuple(candidate)
        if not periods:
            raise ValueError("period tuples must not be empty")
        for period in periods:
            _positive_int(period, "period")
        reason = None
        if require_pairwise_coprime and any(
            math.gcd(a, b) != 1 for a, b in combinations(periods, 2)
        ):
            reason = "not_pairwise_coprime"
        elif periods in seen:
            reason = "duplicate"
        else:
            seen.add(periods)
        decisions.append(CandidateDecision(index, periods, periods, reason))
    return CandidateAudit(
        "period_tuples",
        (
            ("require_pairwise_coprime", require_pairwise_coprime),
            ("ordering", "preserved"),
        ),
        tuple(decisions),
    )


@dataclass(frozen=True)
class SearchReceipt:
    """Completion accounting for exactly the accepted entries in one audit.

    Counts are disjoint terminal dispositions of candidates. ``evaluated`` means
    completed, not attempted. ``exact_contradiction`` requires a proof reference
    and model assumptions; their validity still requires external verification.
    A receipt never certifies a proof, a recovery gate, or family-wide exclusion.
    """

    audit: CandidateAudit
    status: Outcome
    evaluated: int
    assumptions: tuple[str, ...]
    timeouts: int = 0
    capped: int = 0
    proof_reference: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _OUTCOMES:
            raise ValueError("unknown search outcome")
        for name in ("evaluated", "timeouts", "capped"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.assumptions or any(not a.strip() for a in self.assumptions):
            raise ValueError("declare nonempty model assumptions")
        intended = len(self.audit.accepted)
        if self.evaluated + self.timeouts + self.capped > intended:
            raise ValueError("candidate dispositions exceed the accepted manifest")
        if self.status == "exhausted_bank" and (
            self.evaluated != intended or self.timeouts or self.capped
        ):
            raise ValueError("exhausted_bank requires all accepted candidates completed")
        if self.status == "timed_out" and not self.timeouts:
            raise ValueError("timed_out requires a timeout count")
        if self.status == "capped" and not self.capped:
            raise ValueError("capped requires a cap count")
        if self.status == "candidate_found" and not self.evaluated:
            raise ValueError("candidate_found requires an evaluated candidate")
        if self.status == "exact_contradiction":
            if not self.proof_reference or not self.proof_reference.strip():
                raise ValueError("exact_contradiction requires a proof reference")
            if not intended or self.evaluated != intended or self.timeouts or self.capped:
                raise ValueError("exact_contradiction requires all accepted candidates completed")
        if self.status == "statistical_mismatch" and not self.evaluated:
            raise ValueError("statistical_mismatch requires an evaluated candidate")

    def to_dict(self) -> dict[str, Any]:
        intended = len(self.audit.accepted)
        return {
            "status": self.status,
            "scope": "accepted entries of the supplied candidate manifest only",
            "audit": self.audit.to_dict(),
            "assumptions": list(self.assumptions),
            "intended": intended,
            "evaluated": self.evaluated,
            "timeouts": self.timeouts,
            "capped": self.capped,
            "unattempted": intended - self.evaluated - self.timeouts - self.capped,
            "proof_reference": self.proof_reference,
            "proof_verified": False,
            "family_exclusion": False,
        }
