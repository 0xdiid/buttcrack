"""Live progress, stall detection, and cost prediction for long solver runs.

WHY
---
A solver that prints nothing until it finishes is indistinguishable from a solver that has
hung. That is not a hypothetical: a columnar sweep was handed width 51 (a legal divisor of
153), which asks Held-Karp for 2^51 states, and the only symptom was silence. Three
restarts later the actual cause was still unknown.

Three things fix that class of failure, and this module provides all three:

**1. Predict before you run.** Every stage declares the work it is about to do, in units it
can count, BEFORE starting. A stage that announces "2^51 states" is diagnosed in the log
line rather than in a post-mortem.

**2. Heartbeat, not tail-watching.** A background thread emits elapsed / rate / ETA on a
fixed interval regardless of where the loop is. It reports from inside numpy and torch
calls too, since those release the GIL.

**3. Stall detection.** If a stage stops ticking for longer than its budget, the heartbeat
says so explicitly instead of continuing to print a stale rate. Silence becomes a message.

Off by default so library use stays quiet; enable per-call with ``progress=Progress()`` or
globally with ``BUTT_PROGRESS=1`` (and ``BUTT_PROGRESS_INTERVAL`` / ``BUTT_PROGRESS_STALL``).
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def _fmt_units(n: float) -> str:
    for cut, suf in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= cut:
            return f"{n / cut:.1f}{suf}"
    return f"{n:.0f}"


@dataclass
class _Stage:
    name: str
    units: float | None
    detail: str
    started: float
    done: float = 0.0
    last_tick: float = field(default_factory=time.time)
    stalled_reported: bool = False


class Progress:
    """Heartbeat progress reporter with stall detection.

    ``interval`` seconds between heartbeats; ``stall_after`` seconds without a tick before
    the reporter says so. ``sink`` receives formatted lines.
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        interval: float = 5.0,
        stall_after: float = 60.0,
        sink=None,
        prefix: str = "",
    ):
        self.enabled = enabled
        self.interval = interval
        self.stall_after = stall_after
        self.sink = sink or (lambda m: print(m, file=sys.stderr, flush=True))
        self.prefix = prefix
        self.t0 = time.time()
        self._stack: list[_Stage] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> Progress:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._beat, daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> Progress:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reporting ------------------------------------------------------------

    def note(self, msg: str) -> None:
        if self.enabled:
            self.sink(f"{self.prefix}[t+{_fmt_secs(time.time() - self.t0)}] {msg}")

    def predict(self, name: str, ops: float, *, limit: float | None = None) -> bool:
        """Announce a stage's predicted cost before running it.

        Returns False when ``ops`` exceeds ``limit`` — the caller should skip rather than
        start something that cannot finish. This is the check that turns an unbounded hang
        into one log line.
        """
        ok = limit is None or ops <= limit
        if self.enabled:
            verdict = "" if ok else f"  >> EXCEEDS LIMIT {_fmt_units(limit)} — SKIPPING"
            self.note(f"plan {name}: ~{_fmt_units(ops)} ops{verdict}")
        return ok

    @contextmanager
    def stage(self, name: str, *, units: float | None = None, detail: str = ""):
        st = _Stage(name, units, detail, time.time())
        with self._lock:
            self._stack.append(st)
        self.note(
            f"START {name}"
            + (f" ({detail})" if detail else "")
            + (f"  units={_fmt_units(units)}" if units else "")
        )
        try:
            yield st
        finally:
            el = time.time() - st.started
            with self._lock:
                if st in self._stack:
                    self._stack.remove(st)
            rate = f", {_fmt_units(st.done / el)}/s" if st.done and el > 0 else ""
            self.note(f"END   {name}  {_fmt_secs(el)}{rate}")

    def tick(self, n: float = 1.0) -> None:
        with self._lock:
            if self._stack:
                st = self._stack[-1]
                st.done += n
                st.last_tick = time.time()
                st.stalled_reported = False

    # -- the heartbeat --------------------------------------------------------

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            now = time.time()
            with self._lock:
                stack = list(self._stack)
            if not stack:
                continue
            st = stack[-1]
            el = now - st.started
            quiet = now - st.last_tick
            if quiet > self.stall_after and not st.stalled_reported:
                st.stalled_reported = True
                self.sink(
                    f"{self.prefix}[t+{_fmt_secs(now - self.t0)}] !! STALL: '{st.name}' has "
                    f"not ticked for {_fmt_secs(quiet)} (running {_fmt_secs(el)}, "
                    f"{_fmt_units(st.done)} done"
                    + (f"/{_fmt_units(st.units)}" if st.units else "")
                    + ")"
                )
                continue
            rate = st.done / el if el > 0 else 0.0
            msg = f"{st.name}: {_fmt_units(st.done)}"
            if st.units:
                pct = 100.0 * st.done / st.units
                eta = (st.units - st.done) / rate if rate > 0 else math.inf
                msg += f"/{_fmt_units(st.units)} ({pct:.0f}%)"
                if math.isfinite(eta):
                    msg += f"  eta {_fmt_secs(eta)}"
            if rate:
                msg += f"  {_fmt_units(rate)}/s"
            msg += f"  [{_fmt_secs(el)}]"
            self.sink(f"{self.prefix}[t+{_fmt_secs(now - self.t0)}] {msg}")


_NULL = Progress(enabled=False)


def from_env(prefix: str = "") -> Progress:
    """A reporter configured from ``BUTT_PROGRESS`` / ``_INTERVAL`` / ``_STALL``."""
    on = os.environ.get("BUTT_PROGRESS", "").lower() in ("1", "true", "yes", "on")
    return Progress(
        enabled=on,
        interval=float(os.environ.get("BUTT_PROGRESS_INTERVAL", 5.0)),
        stall_after=float(os.environ.get("BUTT_PROGRESS_STALL", 60.0)),
        prefix=prefix,
    )


def resolve(progress: Progress | None) -> Progress:
    """Caller-supplied reporter, else the environment's, else a silent one."""
    if progress is not None:
        return progress
    env = from_env()
    return env if env.enabled else _NULL
