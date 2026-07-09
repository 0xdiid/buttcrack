"""buttcrack — a CLI for cracking classical ciphers that agents can drive easily.

The ``butt`` command exposes encode/decode/crack/auto/identify over a roster of
classical ciphers, with machine-readable JSON output on every command.
"""

from __future__ import annotations

from .engine import auto, crack, decode, encode
from .identify import identify
from .result import Candidate, CrackResult

__version__ = "0.1.0"

__all__ = [
    "auto",
    "crack",
    "decode",
    "encode",
    "identify",
    "Candidate",
    "CrackResult",
    "__version__",
]
