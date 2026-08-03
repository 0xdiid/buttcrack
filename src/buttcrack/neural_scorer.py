"""A character-level neural language model over the 26-letter alphabet.

WHY THIS EXISTS
---------------
Every search in this package that is guided by "does this look like English" was running on
n-gram tables. Measured on identical held-out text (200k characters of letters-only English):

===============================  ==========
scorer                           bits/char
===============================  ==========
quadgram + stupid backoff        2.8400
hexagram + stupid backoff        2.7411   (29% of positions back off)
**char LM, 10.6M params**        **2.0686**
===============================  ==========

WHERE IT HELPS, AND WHERE IT DOES NOT — MEASURED
------------------------------------------------
**Do not read the bits/char table as "better at everything".** Perplexity and *discrimination* are
different quantities, and the honest measurement (d′, n=128, 120 trials each) is:

=================================================  ============  ==========
task — what is the competing hypothesis?           quadgram d′   neural d′
=================================================  ============  ==========
English vs a **same-multiset shuffle**             **13.23**     10.90
English vs **trigram-Markov text**                 **−0.11**     **+2.61**
=================================================  ============  ==========

So:

* **For transposition-style search, keep using quadgrams.** When the rival hypothesis is a
  *shuffle*, quadgrams separate it better (and far more cheaply). Their floor effects are an
  advantage there: scrambled text hits table floors hard and consistently. Hexagrams are worse
  still (d′ 7.18) because sparse tables floor *real* English too.
* **For telling a real decode from an overfit one, n-gram scoring has essentially no power at
  all** (d′ = −0.11 — the fake scored marginally *higher*). Text produced by fitting free
  parameters is, by construction, n-gram-plausible; a quadgram scorer cannot see the difference.
  This is the gap the neural model fills, and it is the reason blind joint running-key recovery is
  impossible under any n-gram model here and possible under this one — the margin
  ``4.7 - 2 x bits/char`` changes sign (see :mod:`buttcrack.running_key`).

Rule of thumb: **choose the scorer by what you are trying to rule out, not by its bits/char.**
Rival is noise → quadgrams. Rival is something that already looks like English → this.

The model is deliberately domain-matched: **uppercase letters only, no spaces or punctuation**,
which is the form classical ciphertext takes. A general-purpose LM with a byte-pair vocabulary is
badly out of distribution on spaceless uppercase text and is *not* a shortcut here.

TRAINING
--------
``train_char_lm`` wants a plain string of A-Z. ~14M characters and ~16 minutes on a mid-range GPU
reaches ~2.07 bits/char; it is data-limited well before it is compute-limited, so more corpus is
the lever, not more steps. The result is a checkpoint you load with :func:`load_char_lm` and wrap
in :class:`buttcrack.running_key.NeuralStreamScorer`.

PyTorch is an optional dependency; import it only through :func:`_torch` so a missing install
raises a pointed error instead of an ImportError from three frames down.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["CharLMConfig", "bits_per_char", "build_char_lm", "load_char_lm", "train_char_lm"]

_TORCH_HINT = (
    "the neural scorer needs PyTorch. Install with `pip install buttcrack[neural]` "
    "or `pip install torch`."
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(_TORCH_HINT) from exc
    return torch


@dataclass(frozen=True)
class CharLMConfig:
    """Architecture. Defaults are the configuration measured at 2.0686 bits/char."""

    ctx: int = 128
    layers: int = 6
    heads: int = 6
    dim: int = 384

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.ctx, self.layers, self.heads, self.dim)


def build_char_lm(cfg: CharLMConfig | None = None) -> Any:
    """Construct an untrained model. Vocabulary is exactly the 26 letters, A=0 … Z=25."""
    torch = _torch()
    nn = torch.nn
    conf = cfg if cfg is not None else CharLMConfig()

    class Block(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(conf.dim)
            self.attn = nn.MultiheadAttention(conf.dim, conf.heads, batch_first=True)
            self.ln2 = nn.LayerNorm(conf.dim)
            self.mlp = nn.Sequential(
                nn.Linear(conf.dim, 4 * conf.dim), nn.GELU(), nn.Linear(4 * conf.dim, conf.dim)
            )

        def forward(self, x: Any, mask: Any) -> Any:
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class CharGPT(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.ctx = conf.ctx
            self.tok = nn.Embedding(26, conf.dim)
            self.pos = nn.Embedding(conf.ctx, conf.dim)
            self.blocks = nn.ModuleList([Block() for _ in range(conf.layers)])
            self.ln = nn.LayerNorm(conf.dim)
            self.head = nn.Linear(conf.dim, 26)
            self.register_buffer(
                "mask", torch.triu(torch.ones(conf.ctx, conf.ctx, dtype=torch.bool), 1)
            )

        def forward(self, idx: Any) -> Any:
            t = idx.shape[1]
            x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
            m = self.mask[:t, :t]
            for b in self.blocks:
                x = b(x, m)
            return self.head(self.ln(x))

    return CharGPT()


def _encode(text: str) -> Any:
    """A-Z string -> tensor of 0..25. Anything else is a caller error, not silently dropped."""
    torch = _torch()
    bad = {c for c in text if not ("A" <= c <= "Z")}
    if bad:
        raise ValueError(
            f"corpus must be uppercase A-Z only; found {sorted(bad)[:8]}. "
            f"Strip and upper-case it first — the model has no other vocabulary."
        )
    return torch.tensor([ord(c) - 65 for c in text], dtype=torch.long)


def load_char_lm(path: str, device: str | None = None) -> tuple[Any, str]:
    """Load a checkpoint written by :func:`train_char_lm`. Returns ``(model, device)``."""
    torch = _torch()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=dev)
    cfg = CharLMConfig(*ck["cfg"])
    model = build_char_lm(cfg).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, dev


def bits_per_char(model: Any, text: str, device: str | None = None, batch: int = 64) -> float:
    """Held-out bits/char — the number that decides whether a search objective can work.

    Compare against 2.84 (quadgram+backoff) and 2.74 (hexagram+backoff) on the *same* text.
    """
    torch = _torch()
    dev = device or next(model.parameters()).device
    data = _encode(text).to(dev)
    ctx = int(model.ctx)
    if len(data) <= ctx:
        raise ValueError(f"need more than ctx={ctx} characters to evaluate, got {len(data)}")
    starts = list(range(0, len(data) - ctx - 1, ctx))
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = starts[i : i + batch]
            x = torch.stack([data[s : s + ctx] for s in chunk])
            y = torch.stack([data[s + 1 : s + ctx + 1] for s in chunk])
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 26), y.reshape(-1), reduction="sum"
            )
            total += float(loss)
            count += y.numel()
    return total / count / math.log(2)


def train_char_lm(
    corpus: str,
    out_path: str,
    *,
    cfg: CharLMConfig | None = None,
    max_minutes: float = 16.0,
    batch_size: int = 96,
    lr: float = 3e-4,
    eval_every: int = 200,
    on_progress: Callable[[dict[str, float]], None] | None = None,
    device: str | None = None,
) -> float:
    """Train on ``corpus`` (uppercase A-Z), checkpointing the best model. Returns best bits/char.

    Bounded by wall clock rather than steps, because the useful question is always "how good a
    scorer can I have in the time I am willing to spend". ``on_progress`` receives a dict per
    evaluation (``step``, ``train_loss``, ``val_bits_per_char``, ``elapsed_s``, ``remaining_s``) —
    wire it to a logger; a training run with no live progress is a job you cannot supervise.
    """
    torch = _torch()
    conf = cfg if cfg is not None else CharLMConfig()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = _encode(corpus)
    n_val = min(1_000_000, len(data) // 10)
    if n_val <= conf.ctx:
        raise ValueError(f"corpus too small: need >> {conf.ctx} characters, got {len(data)}")
    train, val = data[:-n_val].to(dev), data[-n_val:].to(dev)

    model = build_char_lm(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scaler = torch.amp.GradScaler(dev)

    def sample(src: Any) -> tuple[Any, Any]:
        ix = torch.randint(len(src) - conf.ctx - 1, (batch_size,))
        x = torch.stack([src[i : i + conf.ctx] for i in ix])
        y = torch.stack([src[i + 1 : i + conf.ctx + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def val_loss(iters: int = 20) -> float:
        model.eval()
        tot = 0.0
        for _ in range(iters):
            x, y = sample(val)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 26), y.reshape(-1))
            tot += float(loss)
        model.train()
        return tot / iters

    t0 = time.time()
    deadline = t0 + max_minutes * 60
    best = math.inf
    step = 0
    while time.time() < deadline:
        x, y = sample(train)
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 26), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % eval_every == 0:
            vl = val_loss()
            if vl < best:
                best = vl
                torch.save({"state_dict": model.state_dict(), "cfg": conf.as_tuple()}, out_path)
            if on_progress is not None:
                on_progress(
                    {
                        "step": float(step),
                        "train_loss": float(loss),
                        "val_bits_per_char": vl / math.log(2),
                        "elapsed_s": time.time() - t0,
                        "remaining_s": max(0.0, deadline - time.time()),
                    }
                )
    return best / math.log(2)
