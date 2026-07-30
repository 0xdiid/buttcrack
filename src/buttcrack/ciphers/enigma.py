"""Enigma I / Wehrmacht M3 — three rotors, a reflector, a plugboard.

ALGORITHM
---------
A key press steps the rotors, then the current runs plugboard -> right rotor -> middle
-> left -> reflector -> left -> middle -> right -> plugboard. The reflector guarantees
the machine is reciprocal (encrypting the ciphertext under the same settings returns
the plaintext) and guarantees no letter ever encrypts to itself — the two facts every
attack on it is built from.

Stepping includes the double-step anomaly: the middle rotor advances both when the
right rotor passes its notch and, on the very next press, together with the left rotor
when the middle rotor is itself sitting on its notch.

A rotor at position ``p`` with ring setting ``r`` applies ``shift = p - r``::

    forward(c)  = (wiring[(c + shift) % 26] - shift) % 26
    backward(c) = (inverse[(c + shift) % 26] - shift) % 26

Turnover is read off the window letter, so a rotor is at its notch when its *position*
equals the notch letter, independent of the ring.

KEY FORMAT
----------
``ROTORS/REFLECTOR/RINGS/POSITIONS[/PLUGBOARD]`` — e.g. ``I II III/B/AAA/AAA/AB CD EF``.
Rotors are listed left to right (the leftmost is the slowest). Rings and positions are
three letters each. The plugboard is space-separated letter pairs and may be omitted.

Rotor wirings and notches are the standard Enigma I (I-V) and Kriegsmarine M3 (VI-VIII)
sets; reflectors are UKW-B and UKW-C.

Vector: rotors I II III, reflector B, rings AAA, positions AAA, no plugboard —
``AAAAA`` enciphers to ``BDZGO``.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import Counter

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

A = ord("A")

#: name -> (wiring, notch letters). Notches are where the rotor turns the next one over.
ROTORS: dict[str, tuple[str, str]] = {
    "I": ("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q"),
    "II": ("AJDKSIRUXBLHWTMCQGZNPYFVOE", "E"),
    "III": ("BDFHJLCPRTXVZNYEIWGAKMUSQO", "V"),
    "IV": ("ESOVPZJAYQUIRHXLNFTGKDCMWB", "J"),
    "V": ("VZBRGITYUPSDNHLXAWMJQOFECK", "Z"),
    # The naval rotors each carry two notches, which is why they step twice as often.
    "VI": ("JPGVOUMFYQBENHZRDKASXLICTW", "ZM"),
    "VII": ("NZJHGRCXMYSWBOUFAIVLPEKQDT", "ZM"),
    "VIII": ("FKQHTLXOCBJSPDZRAMEWNIUYGV", "ZM"),
}

REFLECTORS: dict[str, str] = {
    "B": "YRUHQSLDPXNGOKMIEBFZCWVJAT",
    "C": "FVPJIAOYEDRZXWGCTKUQSBNMHL",
}

#: The five rotors an Enigma I operator chose from; the default search space.
ARMY_ROTORS = ("I", "II", "III", "IV", "V")


def _to_ints(s: str) -> list[int]:
    return [ord(c) - A for c in s]


def _inverse(wiring: list[int]) -> list[int]:
    out = [0] * 26
    for i, w in enumerate(wiring):
        out[w] = i
    return out


def parse_plugboard(spec: str) -> list[int]:
    """``"AB CD"`` -> a 26-entry involution. Unpaired letters map to themselves."""
    board = list(range(26))
    seen: set[int] = set()
    for pair in str(spec).replace(",", " ").split():
        letters = only_letters(pair).upper()
        if len(letters) != 2:
            raise ValueError(f"plugboard entry {pair!r} must be exactly two letters")
        a, b = ord(letters[0]) - A, ord(letters[1]) - A
        if a in seen or b in seen:
            raise ValueError(f"plugboard letter reused in {pair!r}")
        seen.update((a, b))
        board[a], board[b] = b, a
    return board


def plugboard_repr(board: list[int]) -> str:
    return " ".join(f"{chr(a + A)}{chr(board[a] + A)}" for a in range(26) if board[a] > a)


class EnigmaMachine:
    """A configured machine. ``run`` is reciprocal, so it both encrypts and decrypts."""

    def __init__(
        self,
        rotors: tuple[str, str, str],
        reflector: str,
        rings: str,
        positions: str,
        plugboard: list[int] | None = None,
    ):
        for name in rotors:
            if name not in ROTORS:
                raise ValueError(f"unknown rotor {name!r}; have {', '.join(ROTORS)}")
        if reflector not in REFLECTORS:
            raise ValueError(f"unknown reflector {reflector!r}; have {', '.join(REFLECTORS)}")
        if len(set(rotors)) != 3:
            raise ValueError("the three rotor slots must hold three different rotors")
        rings, positions = only_letters(rings).upper(), only_letters(positions).upper()
        if len(rings) != 3 or len(positions) != 3:
            raise ValueError("ring settings and positions must be three letters each")

        # Stored right-to-left: index 0 is the fast rotor, which matches the stepping
        # rules and the signal path, so neither has to reverse the list at runtime.
        self.names = tuple(reversed(rotors))
        self.wiring = [_to_ints(ROTORS[n][0]) for n in self.names]
        self.inverse = [_inverse(w) for w in self.wiring]
        self.notches = [{ord(c) - A for c in ROTORS[n][1]} for n in self.names]
        self.rings = list(reversed(_to_ints(rings)))
        self.start = list(reversed(_to_ints(positions)))
        self.reflector = _to_ints(REFLECTORS[reflector])
        self.plugboard = plugboard if plugboard is not None else list(range(26))

    def run(self, text: str) -> str:
        pos = list(self.start)
        wiring, inverse, notches = self.wiring, self.inverse, self.notches
        rings, reflector, plug = self.rings, self.reflector, self.plugboard
        shift = [(pos[i] - rings[i]) % 26 for i in range(3)]
        out: list[str] = []

        for ch in only_letters(text).upper():
            # Step first, then encipher — the rotors move before the current flows.
            if pos[1] in notches[1]:
                pos[1] = (pos[1] + 1) % 26
                pos[2] = (pos[2] + 1) % 26
            elif pos[0] in notches[0]:
                pos[1] = (pos[1] + 1) % 26
            pos[0] = (pos[0] + 1) % 26
            shift[0] = (pos[0] - rings[0]) % 26
            shift[1] = (pos[1] - rings[1]) % 26
            shift[2] = (pos[2] - rings[2]) % 26

            c = plug[ord(ch) - A]
            for i in range(3):
                c = (wiring[i][(c + shift[i]) % 26] - shift[i]) % 26
            c = reflector[c]
            for i in (2, 1, 0):
                c = (inverse[i][(c + shift[i]) % 26] - shift[i]) % 26
            out.append(chr(plug[c] + A))
        return "".join(out)


def _parse_key(key: str) -> EnigmaMachine:
    parts = [p.strip() for p in str(key).split("/")]
    if len(parts) not in (4, 5):
        raise ValueError(
            "enigma key must be 'ROTORS/REFLECTOR/RINGS/POSITIONS[/PLUGBOARD]', "
            "e.g. 'I II III/B/AAA/AAA/AB CD'"
        )
    rotors = tuple(parts[0].replace(",", " ").split())
    if len(rotors) != 3:
        raise ValueError("enigma takes exactly three rotors, left to right")
    plug = parse_plugboard(parts[4]) if len(parts) == 5 else None
    return EnigmaMachine(rotors, parts[1].upper(), parts[2], parts[3], plug)


def _ioc(letters: str) -> float:
    n = len(letters)
    if n < 2:
        return 0.0
    return sum(c * (c - 1) for c in Counter(letters).values()) / (n * (n - 1))


class Enigma(Cipher):
    name = "enigma"
    aliases = ("enigma-m3", "enigma-i")
    description = "Wehrmacht M3: three rotors from I-VIII, reflector B/C, rings and plugboard."
    key_format = "'ROTORS/REFLECTOR/RINGS/POSITIONS[/PLUGBOARD]'"
    key_example = "I II III/B/AAA/AAA/AB CD EF"
    complexity = 10
    # The keyless attack is a minutes-to-hours search, not the seconds `auto` budgets
    # per cipher. Reachable through `butt crack enigma`, which is where it belongs.
    auto_crackable = False

    def encode(self, text: str, key: str) -> str:
        return _parse_key(key).run(text)

    #: Reciprocal by construction — the reflector makes encryption its own inverse.
    decode = encode

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Gillogly's three-phase attack: positions by IoC, then rings, then plugboard.

        Phase 1 sweeps every rotor order and all 17576 start positions with ring
        settings AAA and an empty plugboard, scoring the decrypt's index of
        coincidence. This works even though the rings and plugboard are wrong because
        neither destroys the IoC: the plugboard is a fixed substitution (IoC is
        invariant under it) and a wrong ring only mis-times the middle rotor's
        turnover, which spoils a fraction of the message rather than all of it.

        Phase 2 takes the surviving settings and sweeps the right and middle ring
        settings on the n-gram score, which does see the turnover error.

        Phase 3 grows the plugboard greedily: try all 325 unused letter pairs, keep the
        one that most improves the n-gram score, repeat until nothing helps. This is the
        step that recovers the plugboard without the cribs Bletchley relied on.

        This is expensive — 60 rotor orders x 17576 positions is over a million trial
        decrypts (~2.5 minutes). ``rotor_orders``/``reflectors`` narrow the sweep and
        ``probe`` caps how many letters phase 1 scores.

        ``ring_sweep=True`` adds the right-hand ring to phase 1, at 26x the cost. It is
        needed only when the rotor order AND a non-trivial ring setting are both
        unknown: the right ring decides when the middle rotor turns over, and a wrong
        turnover corrupts the message from that point on. With the order known, phase 2
        recovers the ring on its own; with both unknown, a measured plant put the true
        setting 128720th of 1054560 by IoC — far past any shortlist.

        The returned candidates carry ``coverage`` so a timeout that cut the sweep short
        is visible rather than being mistaken for an exhausted search.
        """
        letters = only_letters(text).upper()
        if len(letters) < 40:
            return []
        deadline = (time.monotonic() + timeout) if timeout else None

        rotor_set = tuple(opts.get("rotor_set", ARMY_ROTORS))
        orders = opts.get("rotor_orders") or list(itertools.permutations(rotor_set, 3))
        reflectors = opts.get("reflectors") or ["B"]
        # Phase 1 scores IoC, and IoC is a whole-message statistic: on this corpus the
        # true setting ranks 1070th when only 120 letters are scored, 4th at 200 and
        # 1st at 300. Truncating the probe is a real speed lever but it degrades the
        # ranking fast, so the default is the whole message (capped, since the cost is
        # linear in it and the ranking has saturated by ~400 letters).
        probe_len = int(opts.get("probe", 0)) or min(len(letters), 400)
        probe = letters[:probe_len]
        keep = int(opts.get("keep", 20))

        # -- phase 1: rotor order and start positions, ranked by IoC ---------------
        # A bounded heap, not a full list: the sweep is over a million trials and
        # keeping them all costs more memory than the search costs time.
        heap: list[tuple[float, tuple, str, str, str]] = []
        limit = max(keep * 4, 32)
        tried = 0
        # The right-hand ring decides WHEN the middle rotor turns over, and a wrong
        # turnover corrupts everything after the first one — most of the message. That
        # is invisible when the rotor order is known (phase 2 recovers it) but fatal
        # when it is not: on a 350-letter plant with rings AGH and 60 unknown orders,
        # the true setting ranked 128720th of 1054560 by IoC. Sweeping it here fixes
        # that at 26x the cost, so it is opt-in rather than the default.
        right_rings = range(26) if opts.get("ring_sweep") else [0]
        planned = len(reflectors) * len(orders) * 26**3 * len(right_rings)
        for reflector in reflectors:
            for order in orders:
                if deadline and time.monotonic() > deadline:
                    break
                for ring_right in right_rings:
                    rings = f"AA{chr(ring_right + A)}"
                    for a in range(26):
                        if deadline and time.monotonic() > deadline:
                            break
                        for b in range(26):
                            for c in range(26):
                                positions = f"{chr(a + A)}{chr(b + A)}{chr(c + A)}"
                                machine = EnigmaMachine(order, reflector, rings, positions)
                                entry = (
                                    _ioc(machine.run(probe)),
                                    order,
                                    reflector,
                                    positions,
                                    rings,
                                )
                                if len(heap) < limit:
                                    heapq.heappush(heap, entry)
                                elif entry[0] > heap[0][0]:
                                    heapq.heapreplace(heap, entry)
                            tried += 26
        if not heap:
            return []
        shortlist = sorted(heap, key=lambda r: r[0], reverse=True)[:keep]

        # -- phase 2: ring settings, on the n-gram score ---------------------------
        best: list[tuple[float, tuple, str, str, str]] = []
        for _, order, reflector, positions, found_rings in shortlist:
            if deadline and time.monotonic() > deadline:
                break
            # The right ring is already pinned when phase 1 swept it; otherwise it is
            # still open and gets swept here alongside the middle one.
            right_range = [ord(found_rings[2]) - A] if opts.get("ring_sweep") else list(range(26))
            for ring_mid in range(26):
                for ring_right in right_range:
                    rings = f"A{chr(ring_mid + A)}{chr(ring_right + A)}"
                    # Turning a ring and turning the rotor with it are the same machine,
                    # so the position has to follow the ring to explore anything new.
                    # Only the middle ring moves here when phase 1 already fixed the
                    # right one — its position is already correct for that ring.
                    delta_right = 0 if opts.get("ring_sweep") else ring_right
                    shifted = (
                        positions[0]
                        + chr((ord(positions[1]) - A + ring_mid) % 26 + A)
                        + chr((ord(positions[2]) - A + delta_right) % 26 + A)
                    )
                    machine = EnigmaMachine(order, reflector, rings, shifted)
                    plain = machine.run(letters)
                    best.append((scorer.score(plain), order, reflector, rings, shifted))
        if not best:
            best = [(-1e18, o, r, rg, p) for _, o, r, p, rg in shortlist]
        best.sort(key=lambda r: r[0], reverse=True)

        # -- phase 3: greedy plugboard ---------------------------------------------
        out: list[Candidate] = []
        for score, order, reflector, rings, positions in best[: max(top, 3)]:
            if deadline and time.monotonic() > deadline:
                break
            board, score = _grow_plugboard(
                order, reflector, rings, positions, letters, scorer, deadline
            )
            plain = EnigmaMachine(order, reflector, rings, positions, board).run(letters)
            spec = plugboard_repr(board)
            key = f"{' '.join(order)}/{reflector}/{rings}/{positions}"
            out.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=f"{key}/{spec}" if spec else key,
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={
                        "rotors": list(order),
                        "reflector": reflector,
                        "rings": rings,
                        "positions": positions,
                        "plugboard": spec,
                        # Fraction of the phase-1 space actually swept, so a truncated
                        # run cannot be read as an exhausted one.
                        "coverage": round(tried / planned, 4) if planned else 0.0,
                        "exhaustive": tried >= planned,
                    },
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:top]


def _grow_plugboard(
    order, reflector, rings, positions, letters, scorer, deadline
) -> tuple[list[int], float]:
    """Add the single best letter pair until none improves the score."""
    board = list(range(26))
    best = scorer.score(EnigmaMachine(order, reflector, rings, positions, board).run(letters))
    for _ in range(10):  # a Wehrmacht board carried ten leads
        if deadline and time.monotonic() > deadline:
            break
        candidate = None
        for a in range(26):
            if board[a] != a:
                continue
            for b in range(a + 1, 26):
                if board[b] != b:
                    continue
                trial = list(board)
                trial[a], trial[b] = b, a
                score = scorer.score(
                    EnigmaMachine(order, reflector, rings, positions, trial).run(letters)
                )
                if score > best:
                    best, candidate = score, (a, b)
        if candidate is None:
            break
        a, b = candidate
        board[a], board[b] = b, a
    return board, best
