"""Exact crib algebra for a *superimposed* (word-sum) additive cipher.

Some layered puzzles hide a long, random-looking shift sequence behind two short
additive keys summed position-wise: the shift at position ``i`` is

    ``key_i = a[i mod P] + b[i mod Q]  (mod 26)``          (a "superimposition")

so encryption is ``c_i = (p_i + a[i mod P] + b[i mod Q]) mod 26`` and decryption is
``p_i = (c_i - a[i mod P] - b[i mod Q]) mod 26``. With ``P`` and ``Q`` coprime the
combined key does not repeat for ``lcm(P, Q) = P*Q`` characters, so it defeats
period-finding and looks flat under kappa / index-of-coincidence tests even though it
carries only ``P + Q`` unknowns.

The saving grace is that it is *linear*. A known-plaintext crib at known positions turns
into one equation per covered position::

    a[i mod P] + b[i mod Q] = (c_i - p_i) mod 26

These are the edges of a BIPARTITE graph on ``P`` a-nodes and ``Q`` b-nodes. The two
keys are individually undetermined -- the gauge freedom "add ``t`` to every a-node and
subtract ``t`` from every b-node" leaves every ``a + b`` sum unchanged -- but that same
gauge cancels in the decryption, so **within a connected component every crib-covered
position (and every other position whose a-node and b-node both fall in that component)
decrypts EXACTLY**, with neither key ever guessed. Surplus edges -- any edge that closes
a cycle -- are independent mod-26 consistency checks: a correct crib passes all of them,
and a single wrong crib letter that lands on a cycle fails one, rejecting the crib
outright.

The solver is a weighted union-find (disjoint set) over the ``P + Q`` nodes carrying a
per-node potential relative to its component root, so the ``a + b`` sums stay exact and a
cycle-closing edge with the wrong residue is detected immediately.

Public API
----------
``crib_solve(ct_idx, crib_idx, positions, p_period, q_period) -> dict``
    Solve from 0-25 index lists; reports consistency, checks, and every determined
    position's plaintext index.
``crib_solve_letters(ciphertext, crib, start, p_period, q_period, *, alphabet) -> dict``
    Letter-level convenience: a contiguous crib from ``start``, plus a filled ``plaintext``.
"""

from __future__ import annotations

from .keysources import _alphabet
from .text import only_letters


class _SignedDSU:
    """Weighted disjoint-set over signed potentials for the bipartite sum constraint.

    Each node ``x`` carries ``offset[x] = psi[x] - psi[parent[x]] (mod 26)`` where the
    signed potential ``psi`` is ``+phi`` on a-nodes and ``-phi`` on b-nodes. The crib
    edge ``a[u] + b[v] = s`` (``u`` an a-node, ``v`` a b-node) is thereby the difference
    constraint ``psi[u] - psi[v] = s``, and for any two nodes sharing a root the pinned
    sum ``phi[u] + phi[v]`` is recovered as ``psi[u] - psi[v]``.
    """

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.offset = [0] * size  # offset[x] = psi[x] - psi[parent[x]]  (mod 26)

    def find(self, x: int) -> tuple[int, int]:
        """Return ``(root, psi[x] - psi[root] mod 26)``, compressing the path."""
        parent = self.parent[x]
        if parent == x:
            return x, 0
        root, off_parent = self.find(parent)
        off = (self.offset[x] + off_parent) % 26
        self.parent[x] = root
        self.offset[x] = off
        return root, off

    def union(self, u: int, v: int, s: int) -> bool | None:
        """Impose ``psi[u] - psi[v] = s``.

        Returns ``None`` when this edge merged two components (a spanning-tree edge), or
        a ``bool`` when it closed a cycle: ``True`` if the existing potentials already
        satisfy ``s`` (a passed check), ``False`` if they contradict it (a failed check).
        """
        ru, ou = self.find(u)
        rv, ov = self.find(v)
        if ru == rv:
            return (ou - ov) % 26 == s % 26
        # attach rv under ru: psi[rv] - psi[ru] = (psi[u]-ou) - (psi[v]-ov) rearranged
        self.parent[rv] = ru
        self.offset[rv] = (ou - ov - s) % 26
        return None


def crib_solve(
    ct_idx: list[int],
    crib_idx: list[int],
    positions: list[int],
    p_period: int,
    q_period: int,
) -> dict:
    """Solve the superimposed additive cipher from a crib, exactly, via crib algebra.

    ``ct_idx`` is the whole ciphertext as 0-25 indices; ``crib_idx[k]`` is the known
    plaintext index at ``positions[k]``. ``p_period`` (``P``) and ``q_period`` (``Q``) are
    the two additive-key lengths. Each crib position contributes the edge
    ``a[pos mod P] + b[pos mod Q] = (ct - pt) mod 26`` to a bipartite graph; the solver
    pins every ``a + b`` sum that the crib determines and validates every cycle.

    Returns a dict with:

    - ``consistent`` -- ``False`` iff some cycle-closing edge contradicted the crib.
    - ``determined_positions`` -- sorted positions of ``ct_idx`` whose a-node and b-node
      are both crib-covered and in one component (so they decrypt exactly).
    - ``decrypted`` -- ``{position: plaintext index}`` for each determined position.
    - ``components`` -- number of connected components among crib-touched nodes.
    - ``checks_passed`` / ``checks_failed`` -- surplus-edge (cycle) consistency counts.
    """
    if len(crib_idx) != len(positions):
        raise ValueError("crib_idx and positions must have equal length")
    if p_period < 1 or q_period < 1:
        raise ValueError("p_period and q_period must be positive")

    n = len(ct_idx)
    dsu = _SignedDSU(p_period + q_period)
    touched: set[int] = set()
    checks_passed = 0
    checks_failed = 0

    for pt_val, pos in zip(crib_idx, positions, strict=True):
        if not 0 <= pos < n:
            raise ValueError(f"crib position {pos} outside ciphertext of length {n}")
        a_node = pos % p_period
        b_node = p_period + (pos % q_period)
        s = (ct_idx[pos] - pt_val) % 26
        result = dsu.union(a_node, b_node, s)
        touched.add(a_node)
        touched.add(b_node)
        if result is True:
            checks_passed += 1
        elif result is False:
            checks_failed += 1

    components = len({dsu.find(node)[0] for node in touched})

    determined_positions: list[int] = []
    decrypted: dict[int, int] = {}
    for i in range(n):
        a_node = i % p_period
        b_node = p_period + (i % q_period)
        if a_node not in touched or b_node not in touched:
            continue
        root_a, off_a = dsu.find(a_node)
        root_b, off_b = dsu.find(b_node)
        if root_a != root_b:
            continue
        shift = (off_a - off_b) % 26  # phi[a] + phi[b] = psi[a] - psi[b]
        determined_positions.append(i)
        decrypted[i] = (ct_idx[i] - shift) % 26

    return {
        "consistent": checks_failed == 0,
        "determined_positions": determined_positions,
        "decrypted": decrypted,
        "components": components,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
    }


def crib_solve_letters(
    ciphertext: str,
    crib: str,
    start: int,
    p_period: int,
    q_period: int,
    *,
    alphabet: str = "STANDARD",
) -> dict:
    """Letter-level :func:`crib_solve` for a contiguous crib.

    ``ciphertext`` and ``crib`` are letter strings mapped to indices via ``alphabet``
    (``"STANDARD"``, ``"KRYPTOS"``, or a 26-letter permutation; see
    :func:`keysources._alphabet`). The crib is taken as contiguous, covering positions
    ``start, start + 1, ...`` of the letters-only ciphertext.

    Returns everything :func:`crib_solve` does, plus ``"plaintext"``: the letters-only
    ciphertext with every crib-determined position replaced by its recovered plaintext
    letter and all other positions left as the ciphertext letter.
    """
    alpha = _alphabet(alphabet)
    pos_of = {ch: i for i, ch in enumerate(alpha)}
    ct = only_letters(ciphertext)
    cr = only_letters(crib)
    ct_idx = [pos_of[ch] for ch in ct]
    crib_idx = [pos_of[ch] for ch in cr]
    positions = list(range(start, start + len(crib_idx)))

    result = crib_solve(ct_idx, crib_idx, positions, p_period, q_period)

    decrypted = result["decrypted"]
    plaintext = [alpha[decrypted[i]] if i in decrypted else ch for i, ch in enumerate(ct)]
    result["plaintext"] = "".join(plaintext)
    return result
