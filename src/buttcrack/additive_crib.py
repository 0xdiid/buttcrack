"""Exact known-plaintext recovery for sums of periodic shifts modulo 26.

The model is C[i] = P[i] + sum(K[j][i % periods[j]]) modulo 26, in a
specified alphabet. Linear systems over the fields of size 2 and 13 avoid
invalid division modulo 26. Only values fixed by *both* systems are disclosed;
a representative component key is not a claim of unique key recovery.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from math import gcd, isqrt, lcm

from .keysources import _alphabet
from .text import only_letters

MAX_PARAMETERS = 512


def _periods(periods: Sequence[int]) -> tuple[int, ...]:
    result = tuple(periods)
    if not result or any(type(p) is not int or p < 1 for p in result):
        raise ValueError("periods must be a nonempty sequence of positive integers")
    if sum(result) > MAX_PARAMETERS:
        raise ValueError(f"at most {MAX_PARAMETERS} raw parameters are supported")
    return result


def _phi(n: int) -> int:
    result, left = n, n
    for prime in range(2, isqrt(n) + 1):
        if left % prime == 0:
            result -= result // prime
            while left % prime == 0:
                left //= prime
    if left > 1:
        result -= result // left
    return result


def analyze_periods(periods: Sequence[int]) -> dict:
    """Describe the full sequence space without expanding its potentially huge LCM.

    Its dimension is the degree of lcm(x**p - 1), equivalently the sum of
    Euler's totient over the union of the periods' divisors. This counts shared
    components of non-coprime periods, including repeated or nested periods.
    The LCM is a repetition bound; particular keys can have a smaller period.
    """
    ps = _periods(periods)
    divisors = {d for p in ps for d in range(1, p + 1) if p % d == 0}
    dimension = sum(_phi(d) for d in divisors)
    return {
        "periods": list(ps),
        "raw_parameters": sum(ps),
        "effective_parameters": dimension,
        "gauge_dimensions": sum(ps) - dimension,
        "period_bound": lcm(*ps),
    }


def period_sets(
    rank: int, lo: int, hi: int, max_effective: int, *, coprime_only: bool = False
) -> list[tuple[int, ...]]:
    """Enumerate distinct increasing periods, including non-coprime tuples by default.

    Intended for small declared ranges. Reject oversized enumerations explicitly;
    the returned list is exhaustive only inside the supplied bounds and filters.
    """
    from math import comb

    if any(type(x) is not int or x < 1 for x in (rank, lo, hi, max_effective)) or lo > hi:
        raise ValueError("rank, bounds and max_effective must be positive; lo <= hi")
    if rank > hi - lo + 1:
        return []
    if sum(range(hi - rank + 1, hi + 1)) > MAX_PARAMETERS:
        raise ValueError(f"period tuples exceed the {MAX_PARAMETERS} raw-parameter limit")
    if comb(hi - lo + 1, rank) > 100_000:
        raise ValueError("period range exceeds 100000 candidate tuples; narrow the bounds")
    return [
        ps
        for ps in combinations(range(lo, hi + 1), rank)
        if (not coprime_only or all(gcd(a, b) == 1 for a, b in combinations(ps, 2)))
        and analyze_periods(ps)["effective_parameters"] <= max_effective
    ]


def _row(position: int, periods: Sequence[int]) -> list[int]:
    return [int(j == position % p) for p in periods for j in range(p)]


class _System:
    def __init__(self, prime: int):
        self.prime = prime
        self.basis: dict[int, tuple[list[int], int]] = {}

    def reduce(self, row: list[int]) -> tuple[list[int], int]:
        row = [x % self.prime for x in row]
        value = 0
        for pivot in sorted(self.basis):
            factor = row[pivot]
            if factor:
                known, rhs = self.basis[pivot]
                row = [(a - factor * b) % self.prime for a, b in zip(row, known, strict=True)]
                value = (value + factor * rhs) % self.prime
        return row, value

    def add(self, row: list[int], value: int) -> bool:
        row, known = self.reduce(row)
        rhs = (value - known) % self.prime
        pivot = next((j for j, x in enumerate(row) if x), None)
        if pivot is None:
            return rhs == 0
        inverse = pow(row[pivot], -1, self.prime)
        self.basis[pivot] = ([x * inverse % self.prime for x in row], rhs * inverse % self.prime)
        return True

    def determined(self, row: list[int]) -> int | None:
        remaining, value = self.reduce(row)
        return None if any(remaining) else value

    def representative(self, width: int) -> list[int]:
        out = [0] * width
        for pivot in sorted(self.basis, reverse=True):
            row, rhs = self.basis[pivot]
            out[pivot] = (rhs - sum(a * b for a, b in zip(row, out, strict=True))) % self.prime
        return out


def solve_additive_crib(
    ciphertext: str,
    periods: Sequence[int],
    fragments: Iterable[tuple[int, str]],
    *,
    alphabet: str = "STANDARD",
) -> dict:
    """Solve fixed-position cribs, reporting partial plaintext with '?' for unknowns.

    Offsets count letters in normalized ciphertext. Overlapping fragments are
    allowed; conflicting letters make the system inconsistent. Empty fragment
    lists describe an unconstrained model. No scoring or language assumptions.
    """
    ps = _periods(periods)
    alpha = _alphabet(alphabet)
    ct = only_letters(ciphertext)
    if not ct:
        raise ValueError("ciphertext must contain letters")
    index = {c: i for i, c in enumerate(alpha)}
    constraints: list[tuple[int, str]] = []
    for start, text in fragments:
        crib = only_letters(text)
        if type(start) is not int or start < 0 or not crib or start + len(crib) > len(ct):
            raise ValueError("fragments need nonnegative letter offsets and nonempty in-range text")
        constraints.extend((start + j, c) for j, c in enumerate(crib))
    systems = [_System(2), _System(13)]
    conflicts = []
    for number, (position, char) in enumerate(constraints):
        rhs = (index[ct[position]] - index[char]) % 26
        for system in systems:
            if not system.add(_row(position, ps), rhs):
                conflicts.append(
                    {"constraint": number, "position": position, "modulus": system.prime}
                )
    result = {
        **analyze_periods(ps),
        "alphabet": alpha,
        "constraints": len(constraints),
        "constraint_ranks": {str(s.prime): len(s.basis) for s in systems},
        "consistent": not conflicts,
        "status": "inconsistent" if conflicts else "partial",
        "conflicts": conflicts,
        "keystream": [None] * len(ct),
        "plaintext": "?" * len(ct),
        "determined_positions": [],
        "representative_keys": None,
    }
    if conflicts:
        return result
    pad: list[int | None] = []
    for position in range(len(ct)):
        values = [s.determined(_row(position, ps)) for s in systems]
        a, b = values
        pad.append(None if a is None or b is None else (13 * a + 14 * b) % 26)
    x2, x13 = [s.representative(sum(ps)) for s in systems]
    representative = [(13 * a + 14 * b) % 26 for a, b in zip(x2, x13, strict=True)]
    keys = []
    offset = 0
    for period in ps:
        keys.append(representative[offset : offset + period])
        offset += period
    determined = [i for i, value in enumerate(pad) if value is not None]
    result.update(
        status="determined" if len(determined) == len(ct) else "partial",
        keystream=pad,
        plaintext="".join(
            "?" if value is None else alpha[(index[c] - value) % 26]
            for c, value in zip(ct, pad, strict=True)
        ),
        determined_positions=determined,
        representative_keys=keys,
    )
    return result


def drag_additive_crib(
    ciphertext: str,
    periods: Sequence[int],
    crib: str,
    *,
    fragments: Iterable[tuple[int, str]] = (),
    alphabet: str = "STANDARD",
    max_placements: int = 1000,
) -> dict:
    """Try one floating crib against fixed fragments, without ranking compatible guesses.

    Compatibility is not confirmation. Placement caps are reported separately
    from exact inconsistency; every retained placement includes its partial solve.
    """
    _periods(periods)
    ct, word = only_letters(ciphertext), only_letters(crib)
    if not ct or not word or len(word) > len(ct):
        raise ValueError("floating crib must be nonempty and fit the ciphertext")
    if type(max_placements) is not int or max_placements < 1:
        raise ValueError("max_placements must be positive")
    fixed = list(fragments)
    total = len(ct) - len(word) + 1
    candidates = []
    evaluated = min(total, max_placements)
    for offset in range(evaluated):
        result = solve_additive_crib(ct, periods, [*fixed, (offset, word)], alphabet=alphabet)
        if result["consistent"]:
            candidates.append({"offset": offset, "result": result})
    return {
        "status": "exhausted_placements" if evaluated == total else "capped",
        "declared": total,
        "evaluated": evaluated,
        "rejected": evaluated - len(candidates),
        "candidates": candidates,
    }
