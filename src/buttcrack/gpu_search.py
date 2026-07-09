"""GPU transposition-hypothesis scanner for double columnar over a periodic substitution.

A reusable GPU-accelerated scanner for double columnar over a periodic substitution.

The problem it attacks: a plaintext is enciphered as periodic substitution (Vigenere
OR Beaufort in a keyed alphabet) and then run through a *double* width-8 complete
columnar transposition. There are ``8! x 8! ~ 1.6e9`` ordered transposition pairs, far
too many for a per-candidate quadgram CPU climb. The GPU scores every composed pair
cheaply and keeps a global top-K; a decoupled CPU chi2+quadgram pass "finishes" the
survivors into readable English.

GPU scoring per candidate transposition ``S[i] = CT[perm[i]]``:
  * partition ``S`` into residue classes mod p (for p in 9..16),
  * for each class pick the best shift two ways -- Vigenere and Beaufort -- by
    correlating the class histogram against the (alphabet-reindexed) English frequency
    vector,
  * de-substitute and take the BIGRAM log-prob of the de-subbed stream,
  * keep the max over periods and over {vig, beaufort}.
The bigram-of-de-sub discriminator beats the selection-bias wall that made plain
reveal-IoC tests unreliable at scale (validated: planted double-w8 true pair -> global
rank 0 of 1.6e9).

CRITICAL -- everything happens in *keyed-alphabet index space*. ``CT`` is mapped to
keyed-alphabet indices (``aidx[c]``); the English monogram frequency vector and the
bigram log-prob matrix are both *reindexed by the alphabet* so that index ``k`` means
"the k-th letter of the keyed alphabet" consistently across the histogram correlation,
the shift de-sub, and the bigram lookup. Getting this wrong silently destroys the
signal.

Public API:
    scan_double_w8(ct, alphabet='KRYPTOS', keep=20000, finish=64)
        -> list of (score, orderA, orderB, plaintext) for the CPU-finished survivors.
        ``orderA`` is the inner read-order, ``orderB`` the outer (encode order is
        inner-then-outer; ``S = decode(decode(CT, B), A)``).

Falls back to CPU (pure-torch on device='cpu') when CUDA is unavailable; the algorithm
is identical, just slower.
"""
from __future__ import annotations

import itertools
import math
import os
from importlib import resources

from buttcrack.ciphers.columnar import _decode_letters, _encode_letters
from buttcrack.scoring import ENGLISH_MONOGRAM_FREQ, get_scorer

# KRYPTOS keyed alphabet; aliases accepted by _resolve_alphabet.
KRY = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Substitution periods scored on the GPU (and used by the CPU finish).
PERIODS = (9, 10, 11, 12, 13, 14, 15, 16)

# English monogram frequencies indexed A..Z (from buttcrack.scoring).
_ENG_AZ = [ENGLISH_MONOGRAM_FREQ[chr(65 + k)] for k in range(26)]


# --------------------------------------------------------------------------- #
# alphabet handling
# --------------------------------------------------------------------------- #
def _resolve_alphabet(alphabet: str) -> str:
    """Accept 'KRYPTOS'/'KRY', 'STD'/'ABC...', a full 26-letter alphabet, or a
    keyword to be turned into a keyed alphabet."""
    a = alphabet.upper()
    if a in ("KRY", "KRYPTOS"):
        return KRY
    if a in ("STD", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        return STD
    if len(a) == 26 and sorted(a) == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        return a
    # treat as a keyword -> keyed alphabet
    seen: list[str] = []
    for c in a + STD:
        if c.isalpha() and c not in seen:
            seen.append(c)
    if len(seen) != 26:
        raise ValueError(f"cannot build a 26-letter alphabet from {alphabet!r}")
    return "".join(seen)


# --------------------------------------------------------------------------- #
# undo-index permutation for complete width-8 columnar (validated vs buttcrack)
# --------------------------------------------------------------------------- #
def _undo_complete(order: list[int], width: int, n: int) -> list[int]:
    """Index map so that ``"".join(CT[idx[i]] for i)`` == decode(CT, order).

    Only valid for complete columnar (width | n)."""
    height = n // width
    idx = [0] * n
    for slot in range(width):
        for r in range(height):
            idx[r * width + order[slot]] = slot * height + r
    return idx


# --------------------------------------------------------------------------- #
# substitution helpers (CT and PT live in keyed-alphabet index space)
# --------------------------------------------------------------------------- #
def _sub_encode(pt: str, shifts: list[int], alph: str, variant: str = "vig") -> str:
    """Encode plaintext with a periodic Vigenere/Beaufort substitution in ``alph``."""
    idx = {c: i for i, c in enumerate(alph)}
    p = len(shifts)
    out = []
    for i, ch in enumerate(pt):
        j = idx[ch]
        s = shifts[i % p]
        k = (j + s) % 26 if variant == "vig" else (s - j) % 26
        out.append(alph[k])
    return "".join(out)


# --------------------------------------------------------------------------- #
# GPU scoring tables (ALL reindexed into keyed-alphabet space)
# --------------------------------------------------------------------------- #
def _build_tables(torch, dev, alph: str, n: int):
    aidx = {c: i for i, c in enumerate(alph)}
    # perm_raw[k] = A-Z index of the k-th keyed-alphabet letter (used to reindex
    # the A-Z English statistics into keyed-alphabet index space).
    perm_raw = [ord(alph[k]) - 65 for k in range(26)]

    eng = torch.tensor([_ENG_AZ[perm_raw[k]] for k in range(26)], dtype=torch.float32, device=dev)

    L = torch.arange(26, device=dev)
    # Vigenere: a class enciphered with shift s has letter l = (plain + s); de-sub
    # plain = (l - s). Correlating histogram against eng rolled by s. match_v[s,l] = eng[(l-s)].
    sheng_v = torch.stack([torch.roll(eng, s) for s in range(26)], dim=0)
    # Beaufort: cipher l = (s - plain); de-sub plain = (s - l). match_b[s,l] = eng[(s-l)%26].
    # (rows = shift s, cols = letter l; element [s,l] = eng[(s-l)%26])
    sheng_b = eng[((L.view(26, 1) - L.view(1, 26)) % 26)]

    # bigram log-prob matrix in A-Z space, then reindex rows+cols into alphabet space.
    counts = [[0.0] * 26 for _ in range(26)]
    raw = resources.files("buttcrack.data").joinpath("english_bigrams.txt").read_text("ascii")
    for line in raw.splitlines():
        p = line.split()
        if len(p) >= 2 and len(p[0]) == 2 and p[0].isalpha():
            a0, b0 = p[0].upper()
            counts[ord(a0) - 65][ord(b0) - 65] += float(p[1])
    tot = sum(sum(r) for r in counts) or 1.0
    floor = math.log(0.01 / tot)
    bg_raw = torch.tensor(
        [[math.log(counts[i][j] / tot) if counts[i][j] > 0 else floor for j in range(26)]
         for i in range(26)],
        dtype=torch.float32, device=dev,
    )
    pidx = torch.tensor(perm_raw, device=dev)
    BG = bg_raw[pidx][:, pidx]  # BG[a,b] = log P(alph[a] alph[b])

    cls = {p: (torch.arange(n, device=dev) % p) for p in PERIODS}
    return aidx, sheng_v, sheng_b, BG, cls


def _score_batch(torch, S, sheng_v, sheng_b, BG, cls, ones, n):
    """S: [B, n] of keyed-alphabet indices. Returns [B] best bigram-of-de-sub score."""
    B = S.shape[0]
    best = torch.full((B,), -1e9, device=S.device)
    for p in PERIODS:
        clsp = cls[p].unsqueeze(0).expand(B, n)
        flat = clsp * 26 + S
        hist = torch.zeros(B, p * 26, device=S.device)
        hist.scatter_add_(1, flat, ones[:B])
        hist = hist.view(B, p, 26)
        # vigenere: best shift per class, then de-sub & bigram score
        bs_v = (hist @ sheng_v.T).argmax(2)                       # [B,p]
        desub_v = (S - bs_v.gather(1, clsp)) % 26
        sc_v = BG[desub_v[:, :-1], desub_v[:, 1:]].sum(1)
        # beaufort
        bs_b = (hist @ sheng_b.T).argmax(2)
        desub_b = (bs_b.gather(1, clsp) - S) % 26
        sc_b = BG[desub_b[:, :-1], desub_b[:, 1:]].sum(1)
        best = torch.maximum(best, torch.maximum(sc_v, sc_b))
    return best


# --------------------------------------------------------------------------- #
# CPU chi2 + quadgram finish (per surviving composed permutation)
# --------------------------------------------------------------------------- #
_EXPECT = ENGLISH_MONOGRAM_FREQ  # {A..Z: freq}
_QUAD = get_scorer("quadgrams", "english")


def _chi2_best_shift(cl: list[str], aidx: dict, alph: str, variant: str) -> int:
    n = len(cl)
    if n == 0:
        return 0
    best = (0, 1e18)
    for s in range(26):
        cnt = [0] * 26
        for ch in cl:
            c = aidx[ch]
            j = (c - s) % 26 if variant == "vig" else (s - c) % 26
            cnt[j] += 1
        chi = 0.0
        for j in range(26):
            e = _EXPECT[alph[j]] * n
            chi += (cnt[j] - e) ** 2 / e
        if chi < best[1]:
            best = (s, chi)
    return best[0]


def _recover_rotations(stream: str, aidx: dict, alph: str, variant: str, period: int) -> list[int]:
    cls = [[] for _ in range(period)]
    for i, ch in enumerate(stream):
        cls[i % period].append(ch)
    return [_chi2_best_shift(c, aidx, alph, variant) for c in cls]


def _finish_stream(stream: str, alphabets=(KRY, STD), periods=PERIODS):
    """chi2 shift recovery + quadgram score over alphabets x {vig,beaufort} x periods."""
    best = (-1e18, None, None)
    for alph in alphabets:
        aidx = {c: i for i, c in enumerate(alph)}
        for var in ("vig", "beaufort"):
            for p in periods:
                sh = _recover_rotations(stream, aidx, alph, var, p)
                pt = "".join(
                    alph[((aidx[c] - sh[i % p]) % 26) if var == "vig" else ((sh[i % p] - aidx[c]) % 26)]
                    for i, c in enumerate(stream)
                )
                s = _QUAD.score(pt)
                if s > best[0]:
                    best = (s, pt, (("KRY" if alph == KRY else "STD"), var, p))
    return best


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def scan_double_w8(ct, alphabet: str = "KRYPTOS", keep: int = 20000, finish: int = 64,
                   _validate_plant=None, _progress=False, _only_outer=None):
    """Scan all 8! x 8! width-8 double columnar transpositions of ``ct``.

    Each composed permutation is GPU-scored (bigram of best-shift de-sub, max over
    periods 9-16 and over vig/beaufort, all in keyed-alphabet index space). The global
    top-``keep`` are CPU-finished (chi2 + quadgram) and the best ``finish`` survivors
    returned.

    Args:
        ct: ciphertext (length must be a multiple of 8).
        alphabet: 'KRYPTOS'/'KRY', 'STD', a full 26-letter alphabet, or a keyword.
        keep: GPU global top-K retained for the finish.
        finish: number of top survivors to CPU-finish and return.
        _validate_plant: internal -- (true_outer_idx, true_inner_idx) for the self-test
            rank check; returns extra diagnostics when set.

    Returns:
        list of (score, orderA, orderB, plaintext) sorted best-first, where ``orderA``
        is the inner read-order and ``orderB`` the outer. If ``_validate_plant`` is set,
        returns ``(results, diag_dict)``.
    """
    import torch

    ct = "".join(ct).upper()
    n = len(ct)
    if n % 8 != 0:
        raise ValueError(f"scan_double_w8 needs len(ct) divisible by 8, got {n}")
    alph = _resolve_alphabet(alphabet)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aidx, sheng_v, sheng_b, BG, cls = _build_tables(torch, dev, alph, n)

    perms = [list(p) for p in itertools.permutations(range(8))]
    fac = torch.tensor([_undo_complete(o, 8, n) for o in perms], dtype=torch.long, device=dev)
    M = fac.shape[0]  # 40320

    ct_idx = torch.tensor([aidx[c] for c in ct], dtype=torch.long, device=dev)
    ones = torch.ones(M, n, device=dev)

    bs_score = torch.full((keep,), -1e9, device=dev)
    bs_o = torch.full((keep,), -1, dtype=torch.long, device=dev)
    bs_i = torch.full((keep,), -1, dtype=torch.long, device=dev)
    arangeI = torch.arange(M, device=dev)

    true_o = true_i = None
    diag = {"true_pair_in_topk": None, "true_pair_global_rank": None, "global_top": None}
    if _validate_plant is not None:
        true_o, true_i = _validate_plant

    # outer loop over the outer (B) factor; inner factor (A) is the whole batch.
    # composed[a] = fac[outer] applied to fac[a]  ->  S = CT[composed].
    # ``_only_outer`` restricts the outer loop to a single row (used by the self-test to
    # validate the discriminator on the true-outer row in seconds instead of the full grid).
    outer_iter = range(M) if _only_outer is None else [int(_only_outer)]
    for o in outer_iter:
        composed = fac[o][fac]              # [M, n]
        S = ct_idx[composed]
        sc = _score_batch(torch, S, sheng_v, sheng_b, BG, cls, ones, n)
        cat_s = torch.cat([bs_score, sc])
        cat_o = torch.cat([bs_o, torch.full((M,), o, dtype=torch.long, device=dev)])
        cat_i = torch.cat([bs_i, arangeI])
        top = torch.topk(cat_s, keep)
        bs_score = top.values
        bs_o = cat_o[top.indices]
        bs_i = cat_i[top.indices]
        if true_o is not None and o == true_o:
            tp = sc[true_i].item()
            diag["true_pair_row_rank"] = int((sc > sc[true_i]).sum().item())
            diag["true_pair_score"] = tp
        if _progress and o % 4000 == 0:
            print(f"  o={o}/{M} top={bs_score[0].item():.1f}", flush=True)

    if true_o is not None:
        ink = bool(((bs_o == true_o) & (bs_i == true_i)).any().item())
        diag["true_pair_in_topk"] = ink
        diag["global_top"] = bs_score[0].item()

    # rank surviving composed perms, CPU-finish the best `finish`.
    ranked = sorted(zip(bs_score.tolist(), bs_o.tolist(), bs_i.tolist()), reverse=True)
    results = []
    for s, o, i in ranked[:finish]:
        if o < 0 or i < 0:
            continue
        comp = fac[o][fac[i]].tolist()
        stream = "".join(ct[comp[k]] for k in range(n))
        qsc, pt, meta = _finish_stream(stream)
        results.append((qsc, perms[i], perms[o], pt))
    results.sort(key=lambda t: -t[0])

    if _validate_plant is not None:
        # global rank of the true pair (linear scan over the full grid would be costly;
        # report whether it survived into the kept top-K and its row rank instead).
        return results, diag
    return results


def scan_double_w8_cpu(ct, alphabet: str = "KRYPTOS", keep: int = 20000, finish: int = 64,
                       _validate_plant=None):
    """Force the CPU path (torch on device='cpu'); identical algorithm. Used as the
    fallback when CUDA is unavailable and exercised by the self-test."""
    import torch
    real = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        return scan_double_w8(ct, alphabet, keep, finish, _validate_plant=_validate_plant)
    finally:
        torch.cuda.is_available = real


# --------------------------------------------------------------------------- #
# self-test: plant a double-w8 sub-inner synthetic, assert GPU top-K rank + finish
# --------------------------------------------------------------------------- #
def _english_sample(n: int) -> str:
    base = ("OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLEINLEDGER"
            "WHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHTOFTHERISINGSUNOUTSIDE"
            "ABROADRIVERWOUNDPASTTHEOLDSTONEBRIDGEWHEREFARMERSCARRIEDBASKETSOFFRESHFRUITTOTOWN"
            "ANDCHILDRENPLAYEDALONGTHEGRASSYBANKSLAUGHINGASTHEYCHASEDONEANOTHERTHROUGHTHEFIELDS")
    return (base * 3)[:n]


def _selftest():
    import random
    import itertools as _it

    try:
        import torch  # noqa: F401
        have_torch = True
    except Exception as e:  # pragma: no cover
        print(f"[selftest] torch import failed ({e}); cannot run.")
        return False

    import torch
    cuda = torch.cuda.is_available()
    print(f"[selftest] torch={torch.__version__} cuda={cuda}")

    N = 272
    alph = KRY
    pt = _english_sample(N)
    perms = [list(p) for p in _it.permutations(range(8))]
    keep = 20000
    fn = scan_double_w8 if cuda else scan_double_w8_cpu
    label = "GPU" if cuda else "CPU-fallback"
    # Set GPU_FULLSCAN=1 to run the whole 8!x8! grid (slow); else the fast ROW
    # validation (score the true-outer row, all 40320 inner perms) exercises the identical
    # scoring+finish path. We validate BOTH the Vigenere and Beaufort branches, because a
    # sign error in one match-table is invisible if only the other variant is planted.
    full = os.environ.get("GPU_FULLSCAN") == "1"

    def _trial(variant: str) -> bool:
        rng = random.Random(5)
        period = 11
        shifts = [rng.randrange(26) for _ in range(period)]
        Ssub = _sub_encode(pt, shifts, alph, variant)
        oa = list(range(8)); rng.shuffle(oa)   # inner read-order
        ob = list(range(8)); rng.shuffle(ob)   # outer read-order
        ct = _encode_letters(_encode_letters(Ssub, oa), ob)
        true_inner = perms.index(oa)
        true_outer = perms.index(ob)
        print(f"\n[selftest:{variant}] planted period={period} inner={oa} outer={ob} "
              f"(true_outer={true_outer} true_inner={true_inner})", flush=True)
        assert _decode_letters(_decode_letters(ct, ob), oa) == Ssub, "plant/undo mismatch"

        if full:
            print(f"[selftest:{variant}] {label} FULL scan over {40320 * 40320} pairs ...",
                  flush=True)
            results, diag = fn(ct, alphabet="KRYPTOS", keep=keep, finish=64,
                               _validate_plant=(true_outer, true_inner))
            rank_ok = bool(diag["true_pair_in_topk"])
        else:
            print(f"[selftest:{variant}] {label} ROW validation (true-outer row) ...", flush=True)
            results, diag = fn(ct, alphabet="KRYPTOS", keep=keep, finish=64,
                               _validate_plant=(true_outer, true_inner), _only_outer=true_outer)
            rank_ok = diag.get("true_pair_row_rank", 99999) <= 1
        print(f"[selftest:{variant}] true-pair score={diag.get('true_pair_score'):.1f} "
              f"row-rank={diag.get('true_pair_row_rank')}/40320 "
              f"global-top={diag.get('global_top'):.1f}", flush=True)
        score, _, _, recovered = results[0]
        match = sum(a == b for a, b in zip(recovered, pt)) / len(pt)
        read_ok = match >= 0.85
        passed = rank_ok and read_ok
        print(f"[selftest:{variant}] CPU-finish quad={score:.0f} char-match={match:.0%} "
              f"rank={'PASS' if rank_ok else 'FAIL'} read={'PASS' if read_ok else 'FAIL'} -> "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
        return passed

    return all(_trial(v) for v in ("vig", "beaufort"))


if __name__ == "__main__":
    import sys
    ok = _selftest()
    print("SELFTEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
