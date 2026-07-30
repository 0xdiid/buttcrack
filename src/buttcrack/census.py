"""Construction census — which cipher families can *produce* an observed statistic?

Cracking asks "what key?". This asks the question that comes first: **what family?**
Given a target statistic measured on a ciphertext, encrypt known plaintext with every
registered cipher under many random keys, build each family's empirical distribution
of that statistic at the *observed length*, and report which families could plausibly
have emitted the target.

It is an exclusion instrument. A family whose distribution is nowhere near the target
is ruled out without ever running its cracker; the survivors are the search space.

Two things make it honest rather than decorative:

*Length feasibility.* A family that cannot even encrypt ``n`` letters (Playfair needs
even length, 3x3 Hill needs a multiple of 3, a complete columnar needs a factor) is
reported as INFEASIBLE, not as a bad statistical fit. Structural exclusion is much
stronger evidence than a distributional miss, and conflating the two hides it.

*Invariance.* The caller chooses the statistic. Pick one that is invariant under the
layers you are unsure about and it constrains only the layer you care about — e.g.
monogram IoC is invariant under transposition and under any per-coset substitution,
so it reads *through* an outer polyalphabetic to fingerprint the inner layer alone.

Nothing here decrypts, and nothing here is a null test: a family surviving the census
is a family not yet excluded, never a family confirmed.
"""

from __future__ import annotations

import math
import random
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from .registry import all_ciphers, get
from .text import only_letters

Statistic = Callable[[str], float]


def ioc(text: str, *, normalised: bool = True) -> float:
    """Index of coincidence. ``normalised`` puts random at 1.0, English at ~1.73."""
    a = only_letters(text)
    counts = Counter(a)
    n = len(a)
    if n < 2:
        return 0.0
    raw = sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))
    return raw * 26.0 if normalised else raw


def coset_ioc(text: str, period: int, *, normalised: bool = True) -> float:
    """Pooled per-coset IoC at ``period``.

    Invariant under any monoalphabetic substitution applied per coset — including a
    Vigenere/Quagmire of that period, mixed alphabets and all — because IoC does not
    see which symbol is which. So this measures what lies *inside* a periodic
    substitution, not the substitution.
    """
    a = only_letters(text)
    num = den = 0
    for r in range(period):
        counts = Counter(a[r::period])
        n = sum(counts.values())
        if n < 2:
            continue
        num += sum(v * (v - 1) for v in counts.values())
        den += n * (n - 1)
    if not den:
        return 0.0
    return (num / den) * (26.0 if normalised else 1.0)


@dataclass
class FamilyProfile:
    """One cipher family's empirical distribution of the statistic at length ``n``."""

    name: str
    n_samples: int
    values: list[float] = field(default_factory=list)
    infeasible: str | None = None
    errors: int = 0

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else float("nan")

    @property
    def sd(self) -> float:
        if len(self.values) < 2:
            return float("nan")
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / (len(self.values) - 1))

    def z(self, target: float) -> float:
        """Standard scores of ``target`` against this family. nan if degenerate."""
        s = self.sd
        if not s or math.isnan(s):
            return float("nan")
        return (target - self.mean) / s

    def contains(self, target: float, *, tail: float = 0.025) -> bool:
        """Is ``target`` inside the central ``1 - 2*tail`` of the family's values?"""
        if not self.values:
            return False
        vs = sorted(self.values)
        lo = vs[int(tail * (len(vs) - 1))]
        hi = vs[int((1 - tail) * (len(vs) - 1))]
        return lo <= target <= hi


def _random_key(example: str, rng: random.Random, words: list[str]) -> str:
    """A fresh key with the same character class and length as ``example``.

    Key *shape* is what most ciphers validate; using the registry's own working
    example as the template avoids hand-maintaining a schema per cipher.
    """
    if not example:
        return ""
    if all(c.isdigit() for c in example):
        return "".join(rng.choice(string.digits) for _ in example)
    if "/" in example or "," in example:
        # composite key (e.g. "KEYONE/KEYTWO") — regenerate each part in place
        sep = "/" if "/" in example else ","
        return sep.join(_random_key(p, rng, words) for p in example.split(sep))
    if example.isalpha():
        same_len = [w for w in words if len(w) == len(example)]
        if same_len:
            return rng.choice(same_len).upper()
        return "".join(rng.choice(string.ascii_uppercase) for _ in example)
    return example


def profile_family(
    name: str,
    plaintexts: list[str],
    statistic: Statistic,
    *,
    n_samples: int = 200,
    rng: random.Random | None = None,
    words: list[str] | None = None,
) -> FamilyProfile:
    """Encrypt ``plaintexts`` under random keys and profile ``statistic`` on the output.

    A family that raises on *every* attempt is marked infeasible with the reason —
    that is a structural exclusion at this length, which is stronger than a poor fit.
    """
    rng = rng or random.Random(0)
    words = words or ["KEYWORD", "CIPHER", "SECRET", "PORTAL", "ANNEAL"]
    cipher = get(name)
    prof = FamilyProfile(name=cipher.name, n_samples=n_samples)
    first_error: str | None = None

    for _ in range(n_samples):
        pt = rng.choice(plaintexts)
        key = _random_key(cipher.key_example, rng, words) if cipher.needs_key else ""
        try:
            ct = cipher.encode(pt, key)
        except Exception as exc:  # a family that cannot express this length
            prof.errors += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
            continue
        letters = only_letters(ct)
        if len(letters) < 2:
            prof.errors += 1
            if first_error is None:
                first_error = "encode produced no letters"
            continue
        prof.values.append(statistic(letters))

    if not prof.values:
        prof.infeasible = first_error or "no successful encodings"
    return prof


def census(
    plaintexts: list[str],
    statistic: Statistic,
    target: float,
    *,
    families: list[str] | None = None,
    n_samples: int = 200,
    seed: int = 0,
    words: list[str] | None = None,
) -> list[FamilyProfile]:
    """Profile every family and rank by how well it explains ``target``.

    Survivors sort first (nearest in |z|), then families that miss, then the
    structurally infeasible. Read it as an exclusion list, not a ranking of guesses.
    """
    rng = random.Random(seed)
    names = families if families is not None else [c.name for c in all_ciphers()]
    profs = [
        profile_family(n, plaintexts, statistic, n_samples=n_samples, rng=rng, words=words)
        for n in names
    ]

    def sort_key(p: FamilyProfile) -> tuple[int, float]:
        if p.infeasible:
            return (2, 0.0)
        z = abs(p.z(target))
        return (0 if p.contains(target) else 1, 0.0 if math.isnan(z) else z)

    return sorted(profs, key=sort_key)
