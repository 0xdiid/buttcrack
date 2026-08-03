"""Blind joint running-key recovery — recover *both* streams when neither is known.

THE PROBLEM
-----------
A running key is ``ct = f(pt, key)`` where the key stream is a text as long as the message. When
the key text is *known up to a small candidate set*, this is trivial: try each candidate. That is
the only case classical tooling handles, and the only case this repo could previously test.

The hard case is a **blind** running key: an arbitrary English key text, no known-text handle,
~4.7 bits/char of entropy. Recovery means searching for the decomposition where *both* streams
look like English at once.

WHY IT USUALLY FAILS, AND WHAT ACTUALLY FIXES IT
------------------------------------------------
The received wisdom is that blind recovery is "below unicity" — that there is not enough
ciphertext. **That is the wrong diagnosis**, and it sends you looking for more ciphertext or a
wider beam, neither of which helps.

Two English streams cost ``2 x (model bits/char)`` against the 4.7 bits/char the ciphertext
supplies. Whether the *true* decomposition is the maximum-likelihood one is decided entirely by
that margin:

===========================  ==========  =====  =================================================
model                        bits/char   2x     is the truth the optimum?
===========================  ==========  =====  =================================================
quadgram + backoff           2.84        5.68   no — an impostor wins by 0.55-0.60 log10/char
quintgram                    —           —      no — impostor wins by 0.28-0.43
hexagram + backoff           2.74        5.48   no — impostor wins by 0.955
char-level neural LM         2.07        4.14   **yes at n=144** (truth ahead by 0.008-0.012)
===========================  ==========  =====  =================================================

Measured facts that follow, each of which contradicts an intuition worth naming:

* **Length does not help.** The truth-vs-impostor gap is *flat* in n (−0.548/−0.595/−0.574
  log10/char at n=60/100/144 under quadgrams). More ciphertext never closes it.
* **Raising n-gram order does not help by itself** — a better model finds a better impostor.
  Hexagrams beat the *quadgram beam's* impostor by +0.19/char, but the hexagram beam's *own*
  impostor beats the truth by 0.955/char. Only a model near true English entropy flips it.
* **Beam width is only the right lever once the objective is right.** Under a good enough model
  the truth becomes the optimum and width pays; under a bad one, widening the beam finds a
  *higher-scoring, less-correct* answer.

So: pass a strong scorer. :class:`NgramStreamScorer` is provided because it is always available
and is the correct instrument for *demonstrating* the limitation, but it will not recover a blind
running key. :class:`NeuralStreamScorer` will, imperfectly (~72% of both streams at n=144).

TWO TRAPS
---------
* **The streams are interchangeable.** ``pt + key == key + pt``, so a recovery with the streams
  swapped is a *correct* recovery. Always score both orientations — see :func:`charmatch`.
  Failing to do so once made a 17-character exact hit read as 0.222 noise.
* **A beam with no early signal is already lost.** Scoring with quadgrams alone gives 0.0 for the
  first three positions, so 26**3 = 17,576 prefixes tie and an arbitrary slice survives; the true
  prefix is pruned *before any evidence arrives*. The tell is the solver returning the identical
  plaintext for every input length. All scorers here use **progressive order** — order 1 at
  position 0, 2 at position 1, and so on — so the beam discriminates from the very first letter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .ring_tables import KRYPTOS_RING, ring_flat_table

__all__ = [
    "CONVENTIONS",
    "JointRecovery",
    "NeuralStreamScorer",
    "NgramStreamScorer",
    "StreamScorer",
    "charmatch",
    "joint_beam",
    "key_stream",
]

_TORCH_HINT = (
    "neural scoring needs PyTorch. Install with `pip install buttcrack[neural]` "
    "or `pip install torch`."
)

# ct = f(pt, key)  =>  key = CONVENTIONS[name](ct, pt). All positions are RING positions.
CONVENTIONS: dict[str, Callable[[int, int], int]] = {
    "vig": lambda c, p: (c - p) % 26,
    "beaufort": lambda c, p: (c + p) % 26,
    "variant": lambda c, p: (p - c) % 26,
}


class StreamScorer(Protocol):
    """Scores a next-letter distribution given a context, in RING positions.

    Scores from different scorers are **not comparable** (n-gram tables are log10, neural
    log-softmax is natural log). Compare only within one scorer, against its own null.
    """

    def next_logprobs(self, contexts: Sequence[Sequence[int]]) -> list[list[float]]:
        """``contexts[i]`` is a prefix in ring positions; return ``[i][r]`` for each ring pos."""
        ...


class NgramStreamScorer:
    """Progressive-order n-gram scoring, ring-folded. Always available; provably insufficient.

    Kept because it is the honest baseline: it demonstrates *why* blind recovery needs a stronger
    model, and it is a working scorer for every other purpose. Do not expect it to recover a blind
    running key — see the module docstring.
    """

    def __init__(self, ring: str = KRYPTOS_RING, max_order: int = 6) -> None:
        names = ["monograms", "bigrams", "trigrams", "quadgrams", "quintgrams", "hexagrams"]
        self.ring = ring
        self.max_order = max_order
        self._tabs = [ring_flat_table(names[k], ring) for k in range(max_order)]

    def next_logprobs(self, contexts: Sequence[Sequence[int]]) -> list[list[float]]:
        out: list[list[float]] = []
        for ctx in contexts:
            order = min(len(ctx) + 1, self.max_order)
            tab = self._tabs[order - 1]
            base = tab.index(ctx[len(ctx) - (order - 1) :]) * 26 if order > 1 else 0
            out.append([tab[base + r] for r in range(26)])
        return out


class NeuralStreamScorer:
    """A trained character-level LM, wrapped so its output is indexed by RING position.

    The model's vocabulary is A-Z; the cipher arithmetic is ring positions. The conversion happens
    here and nowhere else — the output distribution is column-permuted into ring order before it
    leaves this class. Indexing a model's A-Z output with a ring position is the single most
    repeated bug in this codebase (see :mod:`buttcrack.ring_tables`).

    Load a checkpoint written by :mod:`buttcrack.neural_scorer`.
    """

    def __init__(self, model: Any, ring: str = KRYPTOS_RING, device: str | None = None) -> None:
        torch = _torch()
        self.ring = ring
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model
        self._r2a = torch.tensor([ord(c) - 65 for c in ring], device=self.device)

    @classmethod
    def load(
        cls, path: str, ring: str = KRYPTOS_RING, device: str | None = None
    ) -> NeuralStreamScorer:
        from .neural_scorer import load_char_lm

        model, dev = load_char_lm(path, device=device)
        return cls(model, ring=ring, device=dev)

    def next_logprobs(self, contexts: Sequence[Sequence[int]]) -> list[list[float]]:
        return [row.tolist() for row in self._next_batch(contexts)]

    def _next_batch(self, contexts: Sequence[Sequence[int]], chunk: int = 2048) -> Any:
        torch = self._torch
        f = torch.nn.functional
        ctx_len = int(getattr(self._model, "ctx", 128))
        rows = [list(c)[-ctx_len:] for c in contexts]
        width = max(1, max(len(r) for r in rows))
        # left-pad short prefixes; the model is causal so only the tail matters
        padded = [[0] * (width - len(r)) + r for r in rows]
        idx = torch.tensor(padded, device=self.device)
        outs = []
        with torch.no_grad():
            for i in range(0, idx.shape[0], chunk):
                logits = self._model(self._r2a[idx[i : i + chunk]])[:, -1, :].float()
                outs.append(f.log_softmax(logits, -1)[:, self._r2a])
        return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(_TORCH_HINT) from exc
    return torch


@dataclass(frozen=True)
class JointRecovery:
    """A recovered decomposition. ``score`` is total, ``per_char`` is what to compare."""

    score: float
    plaintext: list[int]
    key: list[int]
    mode: str

    @property
    def per_char(self) -> float:
        return self.score / len(self.plaintext) if self.plaintext else float("nan")


def key_stream(ct: Sequence[int], plaintext: Sequence[int], mode: str = "vig") -> list[int]:
    """The key stream implied by a plaintext, in ring positions."""
    f = CONVENTIONS[mode]
    return [f(int(c), int(p)) % 26 for c, p in zip(ct, plaintext, strict=True)]


def charmatch(
    got_pt: Sequence[int],
    got_key: Sequence[int],
    true_pt: Sequence[int],
    true_key: Sequence[int],
) -> tuple[float, bool]:
    """Best agreement over BOTH stream orientations. Returns ``(fraction, swapped)``.

    ``pt + key == key + pt``, so recovering the pair with the streams exchanged is a correct
    recovery. Scoring only the direct orientation understates a real hit by ~2x.
    """
    n = len(true_pt)
    if n == 0:
        return float("nan"), False

    def agree(a: Sequence[int], b: Sequence[int]) -> float:
        return sum(int(x) == int(y) for x, y in zip(a, b, strict=True)) / n

    direct = (agree(got_pt, true_pt) + agree(got_key, true_key)) / 2
    swapped = (agree(got_pt, true_key) + agree(got_key, true_pt)) / 2
    return (swapped, True) if swapped > direct else (direct, False)


def joint_beam(
    ct: Sequence[int],
    scorer: StreamScorer,
    *,
    beam: int = 2048,
    mode: str = "vig",
    progress: Callable[[int, int, float], None] | None = None,
) -> JointRecovery:
    """Beam search for the decomposition maximising both streams' likelihood.

    ``ct`` is in RING positions. Position 0 contributes a constant (both streams unconstrained),
    which keeps all 26 openings alive rather than pruning on no evidence.

    Cost is ``O(n * beam * 26)`` scorer queries, batched per position. A wider beam only helps if
    the scorer is strong enough to put the optimum on the truth — otherwise it finds a
    higher-scoring wrong answer. See the module docstring.
    """
    if mode not in CONVENTIONS:
        raise ValueError(f"mode must be one of {sorted(CONVENTIONS)}, got {mode!r}")
    n = len(ct)
    if n == 0:
        return JointRecovery(0.0, [], [], mode)
    f = CONVENTIONS[mode]

    states: list[tuple[float, list[int]]] = [(0.0, [r]) for r in range(26)]
    for i in range(1, n):
        c = int(ct[i])
        pts = [s[1] for s in states]
        keys = [key_stream(ct[:i], p, mode) for p in pts]
        lp_p = scorer.next_logprobs(pts)
        lp_k = scorer.next_logprobs(keys)
        cand: list[tuple[float, list[int]]] = []
        for (sc, pt), rp, rk in zip(states, lp_p, lp_k, strict=True):
            for p in range(26):
                cand.append((sc + rp[p] + rk[f(c, p)], [*pt, p]))
        cand.sort(key=lambda t: -t[0])
        states = cand[:beam]
        if progress is not None:
            progress(i, n, states[0][0])

    score, pt = states[0]
    return JointRecovery(score, pt, key_stream(ct, pt, mode), mode)
