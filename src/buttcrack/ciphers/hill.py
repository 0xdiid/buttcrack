"""Hill cipher — a polygraphic substitution by matrix multiplication mod 26.

Supports the 2x2 and 3x3 forms in one class, auto-detecting the matrix size
``n`` from the key. Letters map A..Z -> 0..25. Plaintext is split into ``n``-letter
blocks (the final short block padded with ``X``); each block is treated as a
COLUMN vector ``p`` and enciphered as ``c = K p mod 26``. Decryption multiplies by
the matrix inverse mod 26, ``K^-1 = det^-1 * adj(K) mod 26``; if ``K`` is not
invertible mod 26 (``gcd(det, 26) != 1``) decode raises ``ValueError``.

KEY FORMAT (auto-detected):
  * An explicit matrix as comma/space-separated integers, row by row, e.g.
    ``"3,3,2,5"`` (2x2) or nine ints (3x3). 4 ints -> 2x2, 9 ints -> 3x3.
  * A keyword whose letters fill the n×n matrix row by row (A=0..Z=25); a
    4-letter keyword builds a 2x2, a 9-letter keyword a 3x3.

Encode/decode operate on a clean uppercase letter stream (non-letters dropped).
"""

from __future__ import annotations

import math
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

# Residues invertible mod 26 (coprime to 26: odd and not a multiple of 13).
_INV_MOD_26 = {d: pow(d, -1, 26) for d in range(1, 26) if math.gcd(d, 26) == 1}


def _parse_key(key: str) -> tuple[list[list[int]], int]:
    """Return ``(matrix, n)`` from an int-list key or a keyword key."""
    s = str(key).strip()
    # Try explicit integers first (comma- and/or whitespace-separated).
    tokens = [t for t in s.replace(",", " ").split() if t]
    nums: list[int] | None = None
    if tokens and all(_is_int(t) for t in tokens):
        nums = [int(t) % 26 for t in tokens]
    else:
        letters = only_letters(s)
        if letters:
            nums = [ord(ch) - 65 for ch in letters]
    if not nums:
        raise ValueError("Hill key must be integers or a keyword of letters")

    n = math.isqrt(len(nums))
    if n < 2 or n * n != len(nums):
        raise ValueError(
            f"Hill key must give a square n x n matrix (n*n values, n >= 2); got {len(nums)} values"
        )
    matrix = [nums[i * n : (i + 1) * n] for i in range(n)]
    return matrix, n


def _is_int(token: str) -> bool:
    t = token[1:] if token[:1] in "+-" else token
    return t.isdigit()


def _det2(m: list[list[int]]) -> int:
    return (m[0][0] * m[1][1] - m[0][1] * m[1][0]) % 26


def _det3(m: list[list[int]]) -> int:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)) % 26


def _det_int(m: list[list[int]]) -> int:
    """Exact integer determinant by cofactor expansion (no floats). Small n only."""
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    total = 0
    for c in range(n):
        minor = [row[:c] + row[c + 1 :] for row in m[1:]]
        total += ((-1) ** c) * m[0][c] * _det_int(minor)
    return total


def _adjugate_general(m: list[list[int]]) -> list[list[int]]:
    """Adjugate (transposed cofactor matrix) for any n, entries mod 26 (exact ints)."""
    n = len(m)
    adj = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            minor = [[m[rr][cc] for cc in range(n) if cc != c] for rr in range(n) if rr != r]
            cof = ((-1) ** (r + c)) * _det_int(minor)
            adj[c][r] = cof % 26  # transpose of the cofactor matrix
    return adj


def _determinant(m: list[list[int]], n: int) -> int:
    """Determinant mod 26 (exact integer arithmetic; 2x2/3x3 use closed forms)."""
    if n == 2:
        return _det2(m)
    if n == 3:
        return _det3(m)
    return _det_int(m) % 26


def _inverse(m: list[list[int]], n: int) -> list[list[int]]:
    """Matrix inverse mod 26 via the exact integer adjugate, or raise if singular.

    Uses ``K^-1 = det^-1 * adj(K) mod 26`` with det^-1 the modular inverse of the
    determinant (``pow(det, -1, 26)``). All arithmetic is integer/modular — there is
    NO float ``round(det * numpy.inv())`` path — so the result is exact and
    ``K . K^-1 == I (mod 26)`` holds identically (see :func:`verify_inverse`).
    """
    det = _determinant(m, n)
    if det not in _INV_MOD_26:
        raise ValueError(
            f"Hill key matrix is not invertible mod 26 (det={det}); "
            "choose a matrix whose determinant is coprime to 26"
        )
    det_inv = _INV_MOD_26[det]
    if n == 2:
        a, b = m[0]
        c, d = m[1]
        adj = [[d, -b], [-c, a]]
    elif n == 3:
        adj = _adjugate3(m)
    else:
        adj = _adjugate_general(m)
    return [[(det_inv * adj[r][c]) % 26 for c in range(n)] for r in range(n)]


def _adjugate3(m: list[list[int]]) -> list[list[int]]:
    """Adjugate (transposed cofactor matrix) of a 3x3 matrix, entries mod 26."""
    cof = [[0] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            minor = [[m[rr][cc] for cc in range(3) if cc != c] for rr in range(3) if rr != r]
            sub_det = minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]
            cof[r][c] = ((-1) ** (r + c)) * sub_det
    # adjugate = transpose of cofactor matrix
    return [[cof[c][r] % 26 for c in range(3)] for r in range(3)]


def _apply(matrix: list[list[int]], letters: str, n: int) -> str:
    """Multiply ``matrix`` by each n-letter COLUMN block of ``letters`` mod 26."""
    padded = letters
    if len(padded) % n != 0:
        padded += "X" * (n - len(padded) % n)
    out: list[str] = []
    for start in range(0, len(padded), n):
        block = [ord(padded[start + k]) - 65 for k in range(n)]
        for row in matrix:
            val = sum(row[k] * block[k] for k in range(n)) % 26
            out.append(chr(65 + val))
    return "".join(out)


class Hill(Cipher):
    name = "hill"
    aliases = ("hill2x2", "hill3x3")
    description = "Polygraphic matrix cipher mod 26 (2x2 or 3x3, auto-detected from key)."
    key_format = "matrix ints row-by-row (4=2x2, 9=3x3) or keyword; det coprime to 26"
    key_example = "3,3,2,5"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        matrix, n = _parse_key(key)
        # Validate invertibility eagerly so an un-decryptable key fails on encode.
        _inverse(matrix, n)
        return _apply(matrix, only_letters(text), n)

    def decode(self, text: str, key: str) -> str:
        matrix, n = _parse_key(key)
        inverse = _inverse(matrix, n)
        return _apply(inverse, only_letters(text), n)

    def crack(
        self,
        text,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Keyless crack of the 2x2 form (brute) and the 3x3 form (row recovery).

        Ciphertext-only key recovery is the hard case for the Hill cipher. The
        2x2 invertible matrix space is small (~157k matrices), so we brute-force
        every invertible decryption matrix in two stages: score each on a short
        prefix of the text (cheap), keep the most English-looking handful, then
        re-score those fully and rank by quadgram fitness.

        The 3x3 keyspace (~26^9) is far too large for a matrix brute, so the 3x3
        path instead decomposes the key by ROWS: each decrypt-matrix row is a
        3-covector (only ~1471 up to scale), a partially-correct key shows up as
        one or two high-scoring rows, and the strongest rows are assembled into
        readable plaintext. See :mod:`buttcrack.ciphers._hill_recover`.

        Both sizes are attempted and their candidates merged. Recognised ``opts``
        for the 3x3 path: ``alphabet`` (index alphabet, default A-Z; e.g.
        ``"KRYPTOS"``), ``q_values`` (additive-schedule periods to try, default
        ``(1,)``), ``pair_brute`` (rescue a monogram-outlier row on short text).
        """
        letters = only_letters(text)
        results = self._crack3(text, letters, scorer, top=top, **opts)
        # A confident 3x3 recovery is the answer; skip the (slow) 2x2 brute.
        if results and results[0].confidence >= 0.5:
            return results[:top]

        # 2x2 brute needs an even-length prefix.
        if len(letters) % 2:
            letters = letters[:-1]
        if len(letters) < 12:
            return results[:top]

        deadline = (time.monotonic() + timeout) if timeout else None
        invertibles = {d for d in range(1, 26) if math.gcd(d, 26) == 1}

        # Stage 1: cheap prefix score over every invertible decryption matrix.
        # A ~120-letter prefix gives enough quadgram signal to find the key while
        # keeping the full sweep fast.
        prefix = letters[: min(len(letters), 120)]
        shortlist: list[tuple[float, tuple[int, int, int, int]]] = []
        keep = max(top, 8)
        checked = 0
        for a in range(26):
            for b in range(26):
                for c in range(26):
                    for d in range(26):
                        if (a * d - b * c) % 26 not in invertibles:
                            continue
                        checked += 1
                        # Budget check periodically rather than every matrix.
                        if deadline and checked % 4096 == 0 and time.monotonic() > deadline:
                            return self._merge(
                                results, self._finish(letters, text, shortlist, scorer, top), top
                            )
                        plain = _apply([[a, b], [c, d]], prefix, 2)
                        s = scorer.score(plain)
                        if len(shortlist) < keep:
                            shortlist.append((s, (a, b, c, d)))
                            if len(shortlist) == keep:
                                shortlist.sort(key=lambda rs: rs[0])
                        elif s > shortlist[0][0]:
                            shortlist[0] = (s, (a, b, c, d))
                            shortlist.sort(key=lambda rs: rs[0])

        return self._merge(results, self._finish(letters, text, shortlist, scorer, top), top)

    def _crack3(self, text, letters, scorer, *, top, **opts) -> list[Candidate]:
        """3x3 recovery via row decomposition; returns round-trippable candidates."""
        if len(letters) < 30:
            return []
        from ._hill_recover import recover

        alphabet = opts.get("alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if len(alphabet) != 26:  # accept a name like "KRYPTOS"
            from ..keysources import _alphabet

            alphabet = _alphabet(alphabet)
        q_values = tuple(opts.get("q_values", (1,)))
        recs = recover(
            text,
            scorer,
            alphabet=alphabet,
            q_values=q_values,
            top=top,
            pair_brute=bool(opts.get("pair_brute", False)),
        )
        out: list[Candidate] = [self._candidate3(text, r, scorer) for r in recs]
        return out

    def _candidate3(self, text, r, scorer) -> Candidate:
        """Build a Candidate from a recovered 3x3 hypothesis.

        Prefer a plain, round-trippable Hill key: over the standard alphabet an
        invertible decrypt matrix inverts to the encryption key, and its exact
        ``decode`` is used whenever it reads as well as the recovered text (this
        also cleans up a plain Hill that the shift search over-fit into a spurious
        per-row additive). An affine or keyed-alphabet recovery that can't round-
        trip through a plain key reports the decrypt matrix and extra parameters.
        """
        std = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        dec = [list(row) for row in r.decrypt_matrix]
        if r.alphabet == std:
            try:
                enc = _inverse(dec, 3)
                key = ",".join(str(enc[i][j]) for i in range(3) for j in range(3))
                clean = self.decode(text, key)
                if scorer.confidence(clean) + 1e-9 >= scorer.confidence(r.plaintext):
                    return Candidate(
                        plaintext=reflow(text, clean),
                        cipher=self.name,
                        key=key,
                        score=scorer.score(clean),
                        confidence=scorer.confidence(clean),
                        meta={"n": 3, "decrypt_matrix": dec},
                    )
            except ValueError:
                pass
        # affine / keyed-alphabet: describe the full construction in the key + meta
        key = "decrypt " + ",".join(str(v) for row in dec for v in row)
        if r.alphabet != std:
            key += f" alphabet {r.alphabet}"
        if not (r.q == 1 and all(o == 0 for row in r.offsets for o in row)):
            key += f" offsets {r.offsets}"
        return Candidate(
            plaintext=reflow(text, r.plaintext),
            cipher=self.name,
            key=key,
            score=r.score,
            confidence=scorer.confidence(r.plaintext),
            meta={
                "n": 3,
                "decrypt_matrix": dec,
                "alphabet": r.alphabet,
                "q": r.q,
                "offsets": r.offsets,
            },
        )

    @staticmethod
    def _merge(a: list[Candidate], b: list[Candidate], top: int) -> list[Candidate]:
        """Merge two candidate lists, dedup by plaintext, rank by score."""
        out: list[Candidate] = []
        seen: set[str] = set()
        for cand in sorted([*a, *b], key=lambda c: c.score, reverse=True):
            if cand.plaintext in seen:
                continue
            seen.add(cand.plaintext)
            out.append(cand)
            if len(out) >= top:
                break
        return out

    def _finish(
        self,
        letters: str,
        text: str,
        shortlist: list[tuple[float, tuple[int, int, int, int]]],
        scorer: NgramScorer,
        top: int,
    ) -> list[Candidate]:
        """Re-score the shortlisted 2x2 decryption matrices on the full text."""
        rescored: list[tuple[float, tuple[int, int, int, int], str]] = []
        for _prefix_score, mat in shortlist:
            a, b, c, d = mat
            plain = _apply([[a, b], [c, d]], letters, 2)
            rescored.append((scorer.score(plain), mat, plain))
        rescored.sort(key=lambda rs: rs[0], reverse=True)

        seen: set[str] = set()
        candidates: list[Candidate] = []
        for score, mat, plain in rescored:
            if plain in seen:
                continue
            seen.add(plain)
            a, b, c, d = mat
            # Report the *encryption* key (inverse of the decryption matrix).
            try:
                enc = _inverse([[a, b], [c, d]], 2)
            except ValueError:
                continue
            key_str = ",".join(str(enc[r][col]) for r in range(2) for col in range(2))
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=key_str,
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"n": 2},
                )
            )
            if len(candidates) >= top:
                break
        return candidates


# --------------------------------------------------------------------------- #
# Reusable matrix helpers: exact modular inverse self-check + keyword builders.
# --------------------------------------------------------------------------- #
_STD_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def matmul_mod26(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """Matrix product ``A @ B`` reduced mod 26 (exact integer arithmetic)."""
    n, k, m = len(a), len(b), len(b[0])
    if len(a[0]) != k:
        raise ValueError("matmul_mod26: inner dimensions do not match")
    return [[sum(a[i][t] * b[t][j] for t in range(k)) % 26 for j in range(m)] for i in range(n)]


def determinant_mod26(matrix: list[list[int]]) -> int:
    """Determinant of a square matrix reduced mod 26 (exact, no floats)."""
    return _determinant(matrix, len(matrix))


def is_invertible_mod26(matrix: list[list[int]]) -> bool:
    """Whether ``matrix`` has an inverse mod 26 (det coprime to 26)."""
    return _determinant(matrix, len(matrix)) in _INV_MOD_26


def inverse_mod26(matrix: list[list[int]]) -> list[list[int]]:
    """Exact modular inverse of a square matrix mod 26 (raises if singular).

    Public wrapper over :func:`_inverse`; auto-detects the size and works for any
    ``n >= 2`` (2x2/3x3 fast paths, general integer adjugate otherwise).
    """
    return _inverse(matrix, len(matrix))


def verify_inverse(matrix: list[list[int]]) -> bool:
    """Assert ``matrix . matrix^-1 == I (mod 26)`` and return True (raises otherwise).

    A cheap self-check that the modular inverse is exact — the discipline the Hill
    routines rely on (integer adjugate, never a float ``round(det*inv())``).
    """
    n = len(matrix)
    inv = _inverse(matrix, n)
    prod = matmul_mod26(matrix, inv)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    if prod != identity:
        raise AssertionError(f"modular inverse check failed: M . M^-1 = {prod}")
    return True


def _word_indices(word: str, alphabet: str) -> list[int]:
    """Letters of ``word`` mapped to their 0-based positions in ``alphabet``."""
    pos = {ch: i for i, ch in enumerate(alphabet)}
    return [pos[ch] for ch in only_letters(word) if ch in pos]


def matrix_from_word(word: str, alphabet: str = _STD_ALPHABET) -> list[list[int]]:
    """Build the ``n x n`` matrix filled row-by-row from an ``n*n``-letter keyword.

    Letters map to indices in ``alphabet`` (default A=0..Z=25; pass e.g. the KRYPTOS
    keyed alphabet to index in that order). The keyword length must be a perfect
    square. This is the plain keyword->matrix used by the Hill key parser, exposed
    as a reusable helper.
    """
    v = _word_indices(word, alphabet)
    n = math.isqrt(len(v))
    if n < 1 or n * n != len(v):
        raise ValueError(f"matrix_from_word needs a perfect-square keyword length; got {len(v)}")
    return [v[i * n : (i + 1) * n] for i in range(n)]


def circulant_matrix(word: str, alphabet: str = _STD_ALPHABET) -> list[list[int]]:
    """Circulant ``n x n`` matrix from an ``n``-letter word: ``M[i][j] = v[(j - i) mod n]``.

    Each row is the numeric vector of ``word`` cyclically shifted, so a short keyword
    derives a wide, highly structured Hill matrix. Entries are mod 26.
    """
    v = _word_indices(word, alphabet)
    n = len(v)
    if n < 2:
        raise ValueError("circulant_matrix needs a word of length >= 2")
    return [[v[(j - i) % n] % 26 for j in range(n)] for i in range(n)]


def companion_matrix(word: str, alphabet: str = _STD_ALPHABET) -> list[list[int]]:
    """Companion ``n x n`` matrix of the monic polynomial with coefficients ``word``.

    For coefficients ``a[0..n-1]`` (the word's letters as indices) the companion form
    has a sub-diagonal of ones and last column ``-a`` (mod 26). Another compact
    keyword->wide-matrix derivation; ``det = (-1)^n * a[0] (mod 26)``, so pick a word
    whose first letter's index is coprime to 26 (odd, not M) to stay invertible.
    """
    a = _word_indices(word, alphabet)
    n = len(a)
    if n < 2:
        raise ValueError("companion_matrix needs a word of length >= 2")
    m = [[0] * n for _ in range(n)]
    for i in range(1, n):
        m[i][i - 1] = 1
    for i in range(n):
        m[i][n - 1] = (-a[i]) % 26
    return m


def kronecker_matrix(word_a: str, word_b: str, alphabet: str = _STD_ALPHABET) -> list[list[int]]:
    """Kronecker product ``M(word_a) ⊗ M(word_b)`` mod 26 for a wide Hill from two small keys.

    ``M(word)`` is :func:`matrix_from_word` (an ``sa x sa`` / ``sb x sb`` matrix from a
    perfect-square keyword), so the result is ``(sa*sb) x (sa*sb)``. Its determinant is
    ``det(A)^sb * det(B)^sa (mod 26)`` — invertible iff both factors are — giving an
    invertible wide Hill built from two short, memorable keywords.
    """
    a = matrix_from_word(word_a, alphabet)
    b = matrix_from_word(word_b, alphabet)
    ra, ca, rb, cb = len(a), len(a[0]), len(b), len(b[0])
    out = [[0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for lb in range(cb):
                    out[i * rb + k][j * cb + lb] = (a[i][j] * b[k][lb]) % 26
    return out


def matrix_to_key(matrix: list[list[int]]) -> str:
    """Render a matrix as the comma-separated integer Hill key string (row by row)."""
    return ",".join(str(v % 26) for row in matrix for v in row)
