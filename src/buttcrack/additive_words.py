"""Bounded dictionary decomposition of a known or partial additive keystream."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product
from math import prod
from typing import Any


def recover_word_keys(
    keystream: Sequence[int | None],
    periods: Sequence[int],
    words: Iterable[str],
    *,
    alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    max_nodes: int = 1_000_000,
    max_results: int = 100,
) -> dict[str, Any]:
    """Find ordered word keys whose repeating sum matches every known position.

    Words are stripped and uppercased; non-ASCII text and symbols outside ``alphabet``
    are rejected, never silently removed. Only lengths in ``periods`` are retained.
    Repeated periods and equivalent decompositions are preserved. ``None`` marks an
    unknown pad position, not zero. All component keys start at pad position zero.

    The last component is indexed by its observed residues. Prefix combinations
    are streamed, so memory is linear in the accepted word bank, not its Cartesian
    product. Each bank item, indexed word, prefix combination, and verified tuple
    consumes one node. This also bounds ingestion of an unbounded iterable. Reaching
    a limit is conservative: the result is capped even if the last accepted result
    happened to be the final solution. Exhaustion only describes the supplied bank.
    """
    if len(alphabet) < 2 or len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must contain at least two distinct symbols")
    if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in alphabet):
        raise ValueError("alphabet must contain ASCII uppercase letters")
    if not periods or any(type(p) is not int or p < 1 for p in periods):
        raise ValueError("periods must contain positive integers")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    if type(max_results) is not int or max_results < 1:
        raise ValueError("max_results must be a positive integer")
    modulus = len(alphabet)
    if any(v is not None and (type(v) is not int or not 0 <= v < modulus) for v in keystream):
        raise ValueError("keystream entries must be None or alphabet indices")
    period_list = list(periods)
    known = [(i, v) for i, v in enumerate(keystream) if v is not None]
    indices = {c: i for i, c in enumerate(alphabet)}
    banks: dict[int, list[str]] = {p: [] for p in period_list}
    encoded: dict[str, tuple[int, ...]] = {}
    counts = {"bank_items": 0, "indexed_words": 0, "prefixes": 0, "verified_tuples": 0}
    filters = {"invalid_symbols": 0, "length_not_requested": 0, "duplicate": 0, "accepted": 0}
    candidates: list[list[str]] = []
    nodes = 0
    bank_complete = False

    def consume(kind: str) -> bool:
        nonlocal nodes
        if nodes >= max_nodes:
            return False
        nodes += 1
        counts[kind] += 1
        return True

    def result(status: str, reason: str | None = None) -> dict[str, Any]:
        return {
            "candidates": candidates,
            "status": status,
            "exhausted": status == "exhausted",
            "capped": status == "capped",
            "limit_reason": reason,
            "unique_in_bank": status == "exhausted" and len(candidates) == 1,
            "periods": period_list,
            "known_positions": len(known),
            "bank_complete": bank_complete,
            "bank_filters": filters,
            "bank_sizes": {str(p): len(bank) for p, bank in banks.items()},
            "ordered_tuple_space": prod(len(banks[p]) for p in period_list)
            if bank_complete
            else None,
            "counts": counts,
            "nodes_used": nodes,
            "limits": {"max_nodes": max_nodes, "max_results": max_results},
            "scope": "supplied dictionary; component phases fixed at zero",
        }

    iterator = iter(words)
    while True:
        # Do not request an additional element from an unbounded iterator at the cap.
        if nodes >= max_nodes:
            return result("capped", "max_nodes:bank")
        try:
            raw = next(iterator)
        except StopIteration:
            break
        consume("bank_items")
        if not isinstance(raw, str):
            raise ValueError("dictionary entries must be strings")
        word = raw.strip().upper()
        if not word or not raw.strip().isascii() or any(c not in indices for c in word):
            filters["invalid_symbols"] += 1
        elif len(word) not in banks:
            filters["length_not_requested"] += 1
        elif word in encoded:
            filters["duplicate"] += 1
        else:
            filters["accepted"] += 1
            banks[len(word)].append(word)
            encoded[word] = tuple(indices[c] for c in word)
    bank_complete = True
    if any(not banks[p] for p in period_list):
        return result("exhausted")

    last_period = period_list[-1]
    residues = sorted({i % last_period for i, _ in known})
    lookup: dict[tuple[int, ...], list[str]] = {}
    for word in banks[last_period]:
        if not consume("indexed_words"):
            return result("capped", "max_nodes:index")
        signature = tuple(encoded[word][r] for r in residues)
        lookup.setdefault(signature, []).append(word)

    for prefix in product(*(banks[p] for p in period_list[:-1])):
        if not consume("prefixes"):
            return result("capped", "max_nodes:search")
        required: dict[int, int] = {}
        for i, target in known:
            remainder = (target - sum(encoded[w][i % len(w)] for w in prefix)) % modulus
            residue = i % last_period
            if residue in required and required[residue] != remainder:
                break
            required[residue] = remainder
        else:
            signature = tuple(required[r] for r in residues)
            for last in lookup.get(signature, []):
                if not consume("verified_tuples"):
                    return result("capped", "max_nodes:verification")
                candidate = (*prefix, last)
                if all(
                    sum(encoded[w][i % len(w)] for w in candidate) % modulus == target
                    for i, target in known
                ):
                    candidates.append(list(candidate))
                    if len(candidates) >= max_results:
                        return result("capped", "max_results")
    return result("exhausted")
