"""Self-generating keystream ciphers: linear-recurrence and LCG additive keys.

Where a Vigenere key is a short word repeated and a running key is borrowed prose,
this family *manufactures* its keystream from a tiny seed by a fixed rule, so the
key is as long as the message yet specified by only a handful of numbers:

* **Linear recurrence over Z26** (:class:`LinearRecurrenceKeystream`) — an LFSR-style
  generator ``k[i] = (c1*k[i-1] + c2*k[i-2] + ... + cr*k[i-r]) mod 26`` seeded by the
  first ``r`` values. Lagged-Fibonacci / chain-addition keys are the special case
  ``coeffs = 1,1``; a maximal-length LFSR is another. The seed and the ``r``
  coefficients are the whole key.
* **Linear congruential generator** (:class:`LcgKeystream`) — ``k[i+1] = (a*k[i] + c)
  mod 26`` from seed ``k[0]``. Three numbers ``(a, c, s0)`` define the entire stream.

Each generator is combined with the plaintext by one of the standard additive rules
(Vigenere ``C=P+K``, Beaufort ``C=K-P``, variant-Beaufort ``C=P-K``).

The blind cracks exploit that these keyspaces are *tiny* — a recurrence of order 2
over Z26 has only ``26**4`` (coeffs × seed) settings, an LCG only ``26**3`` — so the
whole space is brute-forced, every candidate decrypted, and scored with the
vectorized n-gram scorer (:class:`buttcrack.scoring.BatchNgramScorer`). For a fixed
coefficient set the recurrence is *linear in the seed*, so a whole batch of seeds
becomes one matrix multiply (``K = seeds @ M.T mod 26``) — the trick that keeps the
order-2 blind search interactive.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..result import Candidate
from ..scoring import BatchNgramScorer, NgramScorer, get_batch_scorer
from ..text import ALPHABET, only_letters, reflow
from .base import Cipher

try:  # optional acceleration; falls back to pure Python when absent
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in dev/test
    _np = None  # type: ignore[assignment]

MOD = 26

# --- combiners: how a keystream value is mixed with a plaintext value --------

#: name -> (encode(p, k), decode(c, k)), all mod 26. ``vigenere`` C=P+K,
#: ``beaufort`` C=K-P (reciprocal), ``variant`` C=P-K.
COMBINERS: dict[str, tuple[Callable[[int, int], int], Callable[[int, int], int]]] = {
    "vigenere": (lambda p, k: (p + k) % MOD, lambda c, k: (c - k) % MOD),
    "beaufort": (lambda p, k: (k - p) % MOD, lambda c, k: (k - c) % MOD),
    "variant": (lambda p, k: (p - k) % MOD, lambda c, k: (c + k) % MOD),
}
_COMBINER_ALIASES = {
    "vig": "vigenere",
    "v": "vigenere",
    "beau": "beaufort",
    "b": "beaufort",
    "variant-beaufort": "variant",
    "varbeau": "variant",
}


def _resolve_combiner(name: str) -> str:
    key = name.strip().lower()
    key = _COMBINER_ALIASES.get(key, key)
    if key not in COMBINERS:
        raise ValueError(f"unknown combiner {name!r}; choose from {sorted(COMBINERS)}")
    return key


# --- keystream generators (pure, over Z_m) -----------------------------------


def linrec_stream(coeffs: Sequence[int], seed: Sequence[int], n: int, mod: int = MOD) -> list[int]:
    """Linear-recurrence keystream: ``k[i] = sum_j coeffs[j]*k[i-1-j] mod mod``.

    ``seed`` supplies the first ``len(coeffs)`` values ``k[0..r-1]``. Lagged-Fibonacci
    is ``coeffs = [1, 1]``; a general LFSR over Z26 is any coefficient vector.
    """
    r = len(coeffs)
    if len(seed) < r:
        raise ValueError(f"seed needs >= {r} values for order-{r} recurrence, got {len(seed)}")
    k = [int(s) % mod for s in seed[:r]]
    while len(k) < n:
        acc = 0
        for j in range(r):
            acc += coeffs[j] * k[-1 - j]
        k.append(acc % mod)
    return k[:n]


def lcg_stream(a: int, c: int, s0: int, n: int, mod: int = MOD) -> list[int]:
    """LCG keystream: ``k[0] = s0``, ``k[i+1] = (a*k[i] + c) mod mod``."""
    k = [s0 % mod]
    for _ in range(n - 1):
        k.append((a * k[-1] + c) % mod)
    return k[:n]


def _linrec_matrix(coeffs: Sequence[int], n: int, mod: int = MOD):
    """(n, r) matrix ``M`` with ``k = (seed @ M.T) mod mod`` for the given ``coeffs``.

    Because the recurrence is linear, every keystream value is a fixed linear
    combination of the seed; ``M`` collects those combinations so a batch of seeds is
    turned into a batch of keystreams by one matrix multiply.
    """
    r = len(coeffs)
    M = _np.zeros((n, r), dtype=_np.int64)
    for i in range(min(r, n)):
        M[i, i] = 1
    for i in range(r, n):
        row = _np.zeros(r, dtype=_np.int64)
        for j in range(r):
            row = (row + int(coeffs[j]) * M[i - 1 - j]) % mod
        M[i] = row
    return M


# --- combining a keystream batch with the ciphertext -------------------------


def _decode_batch(c_ords, streams, combiner: str):
    """Decrypt a batch of keystreams ``(B, n)`` against ciphertext ``c_ords`` ``(n,)``."""
    if _np is not None:
        c = _np.asarray(c_ords)
        if combiner == "vigenere":
            return (c[None, :] - streams) % MOD
        if combiner == "beaufort":
            return (streams - c[None, :]) % MOD
        return (c[None, :] + streams) % MOD  # variant
    _, dec = COMBINERS[combiner]
    return [[dec(c, k) for c, k in zip(c_ords, stream, strict=True)] for stream in streams]


def _topk_indices(scores, keep: int) -> list[int]:
    """Indices of the ``keep`` highest scores (numpy array or plain list)."""
    if _np is not None and isinstance(scores, _np.ndarray):
        if keep >= scores.shape[0]:
            return _np.argsort(scores)[::-1].tolist()
        top = _np.argpartition(scores, -keep)[-keep:]
        return top[_np.argsort(scores[top])[::-1]].tolist()
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return idx[:keep]


# --- shared helpers ----------------------------------------------------------


def _to_ords(letters: str) -> list[int]:
    return [ord(c) - 65 for c in letters]


def _parse_int_list(s: str) -> list[int]:
    """Parse ``"1,2,3"`` (or letters ``"BCD"``) into 0..25 ordinals."""
    s = s.strip()
    if not s:
        return []
    if any(ch.isdigit() for ch in s):
        return [int(tok) % MOD for tok in s.replace(" ", ",").split(",") if tok != ""]
    return [ord(ch.upper()) - 65 for ch in s if ch.isalpha()]


def _encode_ords(p_ords: list[int], stream: list[int], combiner: str) -> str:
    enc, _ = COMBINERS[combiner]
    return "".join(ALPHABET[enc(p, k)] for p, k in zip(p_ords, stream, strict=True))


def _decode_ords(c_ords: list[int], stream: list[int], combiner: str) -> str:
    _, dec = COMBINERS[combiner]
    return "".join(ALPHABET[dec(c, k)] for c, k in zip(c_ords, stream, strict=True))


def _plain_str(plain_ords) -> str:
    return "".join(ALPHABET[int(o)] for o in plain_ords)


def _rank_candidates(
    scored: list[tuple[float, str, str, dict]],
    text: str,
    cipher_name: str,
    scorer: NgramScorer,
    top: int,
) -> list[Candidate]:
    """Sort (score, plaintext, key_repr, meta) tuples into ranked, deduped Candidates."""
    scored.sort(key=lambda r: r[0], reverse=True)
    out: list[Candidate] = []
    seen: set[str] = set()
    for score, plain, key_repr, meta in scored:
        if plain in seen:
            continue
        seen.add(plain)
        out.append(
            Candidate(
                plaintext=reflow(text, plain),
                cipher=cipher_name,
                key=key_repr,
                score=score,
                confidence=scorer.confidence(plain),
                meta=meta,
            )
        )
        if len(out) >= top:
            break
    return out


def _batch_scorer(scorer: NgramScorer) -> BatchNgramScorer:
    """A BatchNgramScorer over the same table as ``scorer`` (cached by name/lang)."""
    return get_batch_scorer(scorer.name, scorer.lang)


def _all_vectors(r: int) -> Any:
    """All ``26**r`` vectors in Z26^r as an ``(26**r, r)`` array (or list of tuples)."""
    if _np is not None:
        return _np.array(list(itertools.product(range(MOD), repeat=r)), dtype=_np.int64)
    return list(itertools.product(range(MOD), repeat=r))


# --- linear-recurrence cipher ------------------------------------------------


class LinearRecurrenceKeystream(Cipher):
    name = "keystream"
    aliases = ("lfsr", "linrec", "linear-recurrence")
    description = "Self-generating linear-recurrence (LFSR-style) additive keystream over Z26."
    key_format = "coeffs/seed[/combiner], e.g. '1,1/7,4' or '2,0,1/5,5,5/beaufort'"
    key_example = "1,1/7,4"
    complexity = 7
    # Blind crack is a bounded brute force, well-posed but not something `auto`
    # should spend its budget on by default; drive it explicitly via `crack`.
    auto_crackable = False

    def _parse_key(self, key: str) -> tuple[list[int], list[int], str]:
        parts = str(key).split("/")
        if len(parts) < 2:
            raise ValueError(f"keystream key must be '{self.key_format}'")
        coeffs = _parse_int_list(parts[0])
        seed = _parse_int_list(parts[1])
        combiner = _resolve_combiner(parts[2]) if len(parts) > 2 else "vigenere"
        if not coeffs or not seed:
            raise ValueError(f"keystream key must be '{self.key_format}'")
        return coeffs, seed, combiner

    def encode(self, text: str, key: str) -> str:
        coeffs, seed, combiner = self._parse_key(key)
        letters = only_letters(text)
        stream = linrec_stream(coeffs, seed, len(letters))
        return _encode_ords(_to_ords(letters), stream, combiner)

    def decode(self, text: str, key: str) -> str:
        coeffs, seed, combiner = self._parse_key(key)
        letters = only_letters(text)
        stream = linrec_stream(coeffs, seed, len(letters))
        return _decode_ords(_to_ords(letters), stream, combiner)

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
        """Blind brute force over (coefficients × seed) for a small recurrence order.

        Options: ``order`` (default 2) the recurrence order ``r``; ``coeffs`` a fixed
        coefficient list to search only seeds; ``combiner`` one of vigenere/beaufort/
        variant (default: try all three). The full brute runs when the keyspace
        ``26**(2r)`` is small (order 2 = 456976); for larger orders supply ``coeffs`` or a
        ``timeout`` (then it random-samples within the budget and flags partial coverage).
        """
        letters = only_letters(text)
        if len(letters) < 12:
            return []
        n = len(letters)
        c_ords = _to_ords(letters)
        order = int(opts.get("order", 2))
        fixed = _parse_int_list(opts["coeffs"]) if opts.get("coeffs") else None
        if fixed is not None:
            order = len(fixed)
        combiners = (
            [_resolve_combiner(opts["combiner"])] if opts.get("combiner") else list(COMBINERS)
        )
        batch = _batch_scorer(scorer)
        deadline = (time.monotonic() + timeout) if timeout else None
        import random as _random

        rng = rng or _random.Random(0)

        coeff_space = MOD**order
        total = coeff_space if fixed is not None else coeff_space * coeff_space
        budget = 600_000
        partial = False
        keep = max(top, 5)
        scored: list[tuple[float, str, str, dict]] = []

        if fixed is not None:
            coeff_sets: Any = [tuple(fixed)]
        elif total <= budget:
            coeff_sets = itertools.product(range(MOD), repeat=order)
        else:
            partial = True  # too large to enumerate: sample coeff sets within budget
            n_sample = max(1, budget // coeff_space)
            coeff_sets = [
                tuple(rng.randrange(MOD) for _ in range(order)) for _ in range(n_sample)
            ]

        seeds = _all_vectors(order)
        for coeffs in coeff_sets:
            if deadline and time.monotonic() > deadline:
                partial = True
                break
            coeffs = list(coeffs)
            streams = self._streams(coeffs, seeds, n)
            for combiner in combiners:
                plains = _decode_batch(c_ords, streams, combiner)
                scores = batch.score_batch(plains)
                for i in _topk_indices(scores, keep):
                    key_repr = self._key_repr(coeffs, seeds[i], combiner)
                    meta = {"combiner": combiner, "order": order}
                    if partial:
                        meta["coverage"] = "partial"
                    scored.append((float(scores[i]), _plain_str(plains[i]), key_repr, meta))
        return _rank_candidates(scored, text, self.name, scorer, top)

    @staticmethod
    def _streams(coeffs, seeds, n):
        if _np is not None:
            M = _linrec_matrix(coeffs, n)
            return (seeds @ M.T) % MOD
        return [linrec_stream(coeffs, seed, n) for seed in seeds]

    @staticmethod
    def _key_repr(coeffs, seed, combiner) -> str:
        cr = ",".join(str(int(x)) for x in coeffs)
        sr = ",".join(str(int(x)) for x in seed)
        return f"{cr}/{sr}/{combiner}"


# --- LCG cipher --------------------------------------------------------------


class LcgKeystream(Cipher):
    name = "lcg"
    aliases = ("congruential", "lcg-keystream")
    description = "Linear congruential generator additive keystream over Z26 (a,c,s0)."
    key_format = "a,c,s0[/combiner], e.g. '7,3,11' or '5,1,0/beaufort'"
    key_example = "7,3,11"
    complexity = 6
    auto_crackable = False

    def _parse_key(self, key: str) -> tuple[int, int, int, str]:
        parts = str(key).split("/")
        nums = _parse_int_list(parts[0])
        if len(nums) != 3:
            raise ValueError(f"lcg key must be '{self.key_format}'")
        combiner = _resolve_combiner(parts[1]) if len(parts) > 1 else "vigenere"
        a, c, s0 = nums
        return a, c, s0, combiner

    def encode(self, text: str, key: str) -> str:
        a, c, s0, combiner = self._parse_key(key)
        letters = only_letters(text)
        stream = lcg_stream(a, c, s0, len(letters))
        return _encode_ords(_to_ords(letters), stream, combiner)

    def decode(self, text: str, key: str) -> str:
        a, c, s0, combiner = self._parse_key(key)
        letters = only_letters(text)
        stream = lcg_stream(a, c, s0, len(letters))
        return _decode_ords(_to_ords(letters), stream, combiner)

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
        """Full brute force over all ``26**3`` (a, c, s0) settings and each combiner.

        Option ``combiner`` restricts to one of vigenere/beaufort/variant (default: all).
        The whole keyspace is tiny, so this needs no hints and no timeout.
        """
        letters = only_letters(text)
        if len(letters) < 12:
            return []
        n = len(letters)
        c_ords = _to_ords(letters)
        combiners = (
            [_resolve_combiner(opts["combiner"])] if opts.get("combiner") else list(COMBINERS)
        )
        batch = _batch_scorer(scorer)
        deadline = (time.monotonic() + timeout) if timeout else None
        keep = max(top, 5)
        scored: list[tuple[float, str, str, dict]] = []

        params = _all_vectors(3)  # columns a, c, s0
        streams = self._streams(params, n)
        for combiner in combiners:
            if deadline and time.monotonic() > deadline:
                break
            plains = _decode_batch(c_ords, streams, combiner)
            scores = batch.score_batch(plains)
            for i in _topk_indices(scores, keep):
                a, c, s0 = (int(x) for x in params[i])
                key_repr = f"{a},{c},{s0}/{combiner}"
                scored.append(
                    (float(scores[i]), _plain_str(plains[i]), key_repr, {"combiner": combiner})
                )
        return _rank_candidates(scored, text, self.name, scorer, top)

    @staticmethod
    def _streams(params, n):
        if _np is not None:
            a, c, s0 = params[:, 0], params[:, 1], params[:, 2]
            B = params.shape[0]
            out = _np.empty((B, n), dtype=_np.int64)
            out[:, 0] = s0 % MOD
            for i in range(1, n):
                out[:, i] = (a * out[:, i - 1] + c) % MOD
            return out
        return [lcg_stream(a, c, s0, n) for a, c, s0 in params]
