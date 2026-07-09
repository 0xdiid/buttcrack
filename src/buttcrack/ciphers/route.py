"""Route transposition cipher.

The plaintext is written into a complete rectangle of a given width ``W``
following one geometric *write-in* route, then read back out following a
different geometric *read-out* route (spirals, diagonals, serpentine columns,
etc.). The grid width and the two named routes together form the key.

Key format (case-insensitive, semicolon- or comma-separated)::

    "width 4; write rows; read spiral-ccw-tr"

* ``width N`` / ``w N`` / a bare integer  -- grid width ``W`` (3..20)
* ``write <route>``                       -- write-in route (default ``rows``)
* ``read <route>``                        -- read-out route (default ``rows``)

A *route* names a geometric ordering of the cells of the rectangle. The
catalogue covers row/column scans (optionally serpentine/boustrophedon) from
each corner and spirals inwards (clockwise / counter-clockwise) from each
corner. See :data:`ROUTE_NAMES` for the full list.

The published CryptoCrack vector -- plaintext ``WE ARE DISCOVERED``, width 4,
write-in by rows, read-out a spiral inwards counter-clockwise from the
top-right corner -- encodes to ``RAEWECREDXESIDOV`` (``RAEWE CREDX ESIDO V``).
"""

from __future__ import annotations

import math
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

#: pad letter used to complete the final row of the rectangle
PAD = "X"

# direction deltas indexed 0=right, 1=down, 2=left, 3=up
_DIRS = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}


def _dims(n: int, width: int) -> tuple[int, int]:
    """Rows, cols for ``n`` letters in a rectangle of the given width."""
    cols = width
    rows = math.ceil(n / cols) if cols else 0
    return rows, cols


def _scan_route(
    rows: int, cols: int, *, start_corner: str, vertical: bool, serpentine: bool
) -> list[tuple[int, int]]:
    """Cell order for a straight row/column scan from a corner.

    ``start_corner`` is one of ``tl``/``tr``/``bl``/``br``. ``vertical`` scans
    by columns instead of rows. ``serpentine`` alternates the direction of
    successive lines (boustrophedon).
    """
    top = start_corner[0] == "t"
    left = start_corner[1] == "l"
    cells: list[tuple[int, int]] = []
    if not vertical:
        line_order = range(rows) if top else range(rows - 1, -1, -1)
        for i, r in enumerate(line_order):
            forward = range(cols) if left else range(cols - 1, -1, -1)
            seq = list(forward)
            if serpentine and i % 2 == 1:
                seq = seq[::-1]
            for c in seq:
                cells.append((r, c))
    else:
        line_order = range(cols) if left else range(cols - 1, -1, -1)
        for i, c in enumerate(line_order):
            forward = range(rows) if top else range(rows - 1, -1, -1)
            seq = list(forward)
            if serpentine and i % 2 == 1:
                seq = seq[::-1]
            for r in seq:
                cells.append((r, c))
    return cells


def _spiral_route(
    rows: int, cols: int, *, start_corner: str, clockwise: bool
) -> list[tuple[int, int]]:
    """Cell order for a spiral inwards from a corner.

    The starting heading runs along the rectangle edge away from the adjacent
    corner; the spiral then turns inwards. ``clockwise`` selects the turn
    direction (and, with it, the initial heading) so that the path stays on the
    grid.
    """
    r = 0 if start_corner[0] == "t" else rows - 1
    c = 0 if start_corner[1] == "l" else cols - 1

    # Initial heading: move along whichever edge leaves the grid valid, picking
    # the heading consistent with the requested turn sense.
    if clockwise:
        # turn right (+1) at each wall
        heading = {"tl": 0, "tr": 1, "br": 2, "bl": 3}[start_corner]
        turn = 1
    else:
        # turn left (-1) at each wall
        heading = {"tl": 1, "tr": 2, "br": 3, "bl": 0}[start_corner]
        turn = -1

    visited = [[False] * cols for _ in range(rows)]
    cells: list[tuple[int, int]] = []
    for _ in range(rows * cols):
        cells.append((r, c))
        visited[r][c] = True
        dr, dc = _DIRS[heading]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]):
            heading = (heading + turn) % 4
            dr, dc = _DIRS[heading]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]):
                break
        r, c = nr, nc
    return cells


def _diagonal_route(
    rows: int, cols: int, *, start_corner: str, anti: bool, serpentine: bool
) -> list[tuple[int, int]]:
    """Cell order reading successive diagonals of the grid.

    ``anti`` selects the diagonal family: anti-diagonals (constant ``r + c``, the "/"
    family) when true, main diagonals (constant ``r - c``, the "\\" family) when false.
    ``start_corner`` fixes the corner the traversal begins at: its row half (``t``/``b``)
    picks whether successive diagonals are visited top-down or bottom-up, and its column
    half (``l``/``r``) picks the within-diagonal direction. ``serpentine`` flips the
    within-diagonal direction on alternate diagonals (boustrophedon).
    """
    if anti:
        groups = [
            [(r, s - r) for r in range(rows) if 0 <= s - r < cols]
            for s in range(rows + cols - 1)
        ]
    else:
        groups = [
            [(r, r - d) for r in range(rows) if 0 <= r - d < cols]
            for d in range(-(cols - 1), rows)
        ]
    top = start_corner[0] == "t"
    left = start_corner[1] == "l"
    if not top:
        groups = groups[::-1]
    cells: list[tuple[int, int]] = []
    for i, diag in enumerate(groups):
        # each group is listed by increasing row; increasing row means decreasing column
        # on an anti-diagonal and increasing column on a main diagonal
        rightward = diag if (left == anti) else diag[::-1]
        if serpentine and i % 2 == 1:
            rightward = rightward[::-1]
        cells.extend(rightward)
    return cells


# route name -> builder; each builder takes (rows, cols) -> list of cells
def _make_route(rows: int, cols: int, name: str) -> list[tuple[int, int]]:
    name = name.strip().lower().replace("_", "-").replace(" ", "-")
    spec = _ROUTE_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown route {name!r}; known routes: {', '.join(ROUTE_NAMES)}")
    kind, kwargs = spec
    if kind == "scan":
        return _scan_route(rows, cols, **kwargs)
    if kind == "diagonal":
        return _diagonal_route(rows, cols, **kwargs)
    return _spiral_route(rows, cols, **kwargs)


def _build_route_specs() -> dict[str, tuple[str, dict]]:
    specs: dict[str, tuple[str, dict]] = {}
    corners = ("tl", "tr", "bl", "br")
    # plain rows (left-to-right) from the top-left is the canonical "rows"
    specs["rows"] = ("scan", {"start_corner": "tl", "vertical": False, "serpentine": False})
    specs["cols"] = ("scan", {"start_corner": "tl", "vertical": True, "serpentine": False})
    # horizontal scans
    for corner in corners:
        specs[f"rows-{corner}"] = (
            "scan",
            {"start_corner": corner, "vertical": False, "serpentine": False},
        )
        specs[f"serp-rows-{corner}"] = (
            "scan",
            {"start_corner": corner, "vertical": False, "serpentine": True},
        )
    # vertical scans
    for corner in corners:
        specs[f"cols-{corner}"] = (
            "scan",
            {"start_corner": corner, "vertical": True, "serpentine": False},
        )
        specs[f"serp-cols-{corner}"] = (
            "scan",
            {"start_corner": corner, "vertical": True, "serpentine": True},
        )
    # spirals inwards, both senses, from every corner
    for corner in corners:
        specs[f"spiral-cw-{corner}"] = ("spiral", {"start_corner": corner, "clockwise": True})
        specs[f"spiral-ccw-{corner}"] = ("spiral", {"start_corner": corner, "clockwise": False})
    # diagonal scans: both families ("/" = diag, "\" = maindiag), plain + serpentine
    for corner in corners:
        for anti, label in ((True, "diag"), (False, "maindiag")):
            specs[f"{label}-{corner}"] = (
                "diagonal",
                {"start_corner": corner, "anti": anti, "serpentine": False},
            )
            specs[f"serp-{label}-{corner}"] = (
                "diagonal",
                {"start_corner": corner, "anti": anti, "serpentine": True},
            )
    return specs


_ROUTE_SPECS = _build_route_specs()
#: every route name the cipher understands (write-in or read-out)
ROUTE_NAMES: tuple[str, ...] = tuple(_ROUTE_SPECS)

#: small, high-value catalogue the keyless crack searches over for the read-out
_CRACK_ROUTES: tuple[str, ...] = (
    "rows",
    "cols",
    "serp-rows-tl",
    "serp-cols-tl",
    "spiral-cw-tl",
    "spiral-ccw-tl",
    "spiral-cw-tr",
    "spiral-ccw-tr",
    "spiral-cw-bl",
    "spiral-ccw-bl",
    "spiral-cw-br",
    "spiral-ccw-br",
    "cols-tr",
    "cols-bl",
    "rows-br",
)


def _parse_key(key: str) -> tuple[int, str, str]:
    """Return ``(width, write_route, read_route)`` from a key string."""
    s = str(key).strip().lower()
    if not s:
        raise ValueError("route key must specify a width")
    width: int | None = None
    write_route = "rows"
    read_route = "rows"
    # split into tokens on ; and ,
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    for part in parts:
        tokens = part.split()
        if not tokens:
            continue
        head = tokens[0]
        rest = " ".join(tokens[1:]).strip()
        if head in ("width", "w") and rest:
            width = int(rest.split()[0])
        elif head in ("write", "write-in", "writein", "in") and rest:
            write_route = rest
        elif head in ("read", "read-out", "readout", "out") and rest:
            read_route = rest
        elif len(tokens) == 1 and head.isdigit():
            width = int(head)
        elif len(tokens) == 1 and head in _ROUTE_SPECS:
            # a bare route name applies to the read-out
            read_route = head
    if width is None:
        raise ValueError("route key must specify a width, e.g. 'width 4'")
    if width < 1:
        raise ValueError("route width must be >= 1")
    return width, write_route, read_route


def _encode_letters(letters: str, width: int, write_route: str, read_route: str) -> str:
    n = len(letters)
    if n == 0:
        return ""
    rows, cols = _dims(n, width)
    total = rows * cols
    padded = letters + PAD * (total - n)
    # place letters into the grid following the write-in route
    write_cells = _make_route(rows, cols, write_route)
    grid = [[""] * cols for _ in range(rows)]
    for ch, (r, c) in zip(padded, write_cells, strict=True):
        grid[r][c] = ch
    # read the grid out following the read-out route
    read_cells = _make_route(rows, cols, read_route)
    return "".join(grid[r][c] for (r, c) in read_cells)


def _decode_letters(cipher: str, width: int, write_route: str, read_route: str) -> str:
    n = len(cipher)
    if n == 0:
        return ""
    rows, cols = _dims(n, width)
    total = rows * cols
    # the ciphertext (possibly with pad letters) fills exactly the grid
    text = cipher
    if len(text) < total:
        text = text + PAD * (total - len(text))
    # place ciphertext back following the read-out route
    read_cells = _make_route(rows, cols, read_route)
    grid = [[""] * cols for _ in range(rows)]
    for ch, (r, c) in zip(text, read_cells, strict=True):
        grid[r][c] = ch
    # read the grid following the write-in route to recover the plaintext
    write_cells = _make_route(rows, cols, write_route)
    return "".join(grid[r][c] for (r, c) in write_cells)


class Route(Cipher):
    """Route transposition: write into a grid by one route, read by another."""

    name = "route"
    aliases = ("routetrans", "routes")
    description = "Route transposition; key is a grid width plus write-in and read-out routes."
    key_format = "width N; write <route>; read <route> (semicolon-separated, routes named)"
    key_example = "width 4; write rows; read spiral-ccw-tr"
    complexity = 3

    # Transposition only reorders letters, so it cannot preserve word spacing;
    # encode/decode operate on a clean uppercase letter stream (no reflow, which
    # would leak the plaintext's word lengths into the ciphertext).
    def encode(self, text: str, key: str) -> str:
        width, write_route, read_route = _parse_key(key)
        return _encode_letters(only_letters(text), width, write_route, read_route)

    def decode(self, text: str, key: str) -> str:
        width, write_route, read_route = _parse_key(key)
        return _decode_letters(only_letters(text), width, write_route, read_route)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        """Keyless brute force over (width, read-out route) with a rows write-in.

        The route catalogue is small and finite, so for each plausible width we
        invert every catalogued read-out route (assuming the standard rows
        write-in) and rank the recovered plaintexts by the n-gram scorer.
        """
        letters = only_letters(text)
        if len(letters) < 4:
            return []
        max_width = int(opts.get("max_width", min(len(letters), 20)))
        if opts.get("width"):
            widths = [int(opts["width"])]
        else:
            widths = list(range(3, max_width + 1))
        write_route = str(opts.get("write_route", "rows"))
        read_routes = opts.get("read_routes") or _CRACK_ROUTES
        deadline = (time.monotonic() + timeout) if timeout else None

        candidates: list[Candidate] = []
        truncated = False
        for width in widths:
            if width < 1 or width > len(letters):
                continue
            for route in read_routes:
                if deadline and time.monotonic() > deadline:
                    truncated = True
                    break
                plain = _decode_letters(letters, width, write_route, route)
                # the grid may carry trailing pad letters; the scorer is robust
                # to them, but trim a run of trailing pads for cleaner output.
                clean = plain.rstrip(PAD) or plain
                key = f"width {width}; write {write_route}; read {route}"
                candidates.append(
                    Candidate(
                        plaintext=clean,
                        cipher=self.name,
                        key=key,
                        score=scorer.score(clean),
                        confidence=scorer.confidence(clean),
                        meta={"width": width, "read_route": route, "write_route": write_route},
                    )
                )
            if truncated:
                break
        candidates.sort(key=lambda c: c.score, reverse=True)
        out = candidates[:top]
        if truncated and out:
            out[-1].meta["timeout_truncated"] = True
        return out
