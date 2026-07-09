"""Exact crib solver for the superimposed-additive family shift_i = a[i%P] + b[i%Q]."""

import random

from buttcrack.crib_algebra import crib_solve, crib_solve_letters


def _plant(p_period, q_period, n, seed):
    rng = random.Random(seed)
    a = [rng.randrange(26) for _ in range(p_period)]
    b = [rng.randrange(26) for _ in range(q_period)]
    pt = [rng.randrange(26) for _ in range(n)]
    ct = [(pt[i] + a[i % p_period] + b[i % q_period]) % 26 for i in range(n)]
    return pt, ct


def test_crib_pins_positions_and_decrypts_exactly():
    p_period, q_period, n = 5, 7, 40
    pt, ct = _plant(p_period, q_period, n, 7)
    positions = list(range(30))  # a crib covering all P and Q residues -> one component
    res = crib_solve(ct, [pt[i] for i in positions], positions, p_period, q_period)
    assert res["consistent"] is True and res["checks_failed"] == 0
    # every pinned position decrypts to the TRUE plaintext (gauge cancels in a+b)
    for pos, val in res["decrypted"].items():
        assert val == pt[pos]
    # positions beyond the 30-letter crib are pinned purely by the component algebra
    assert any(pos >= 30 for pos in res["determined_positions"])


def test_corrupted_crib_is_flagged_inconsistent():
    p_period, q_period, n = 5, 7, 40
    pt, ct = _plant(p_period, q_period, n, 7)
    positions = list(range(30))
    crib = [pt[i] for i in positions]
    crib[10] = (crib[10] + 1) % 26  # a single wrong letter closes a cycle with a bad residue
    res = crib_solve(ct, crib, positions, p_period, q_period)
    assert res["consistent"] is False and res["checks_failed"] > 0


def test_crib_solve_letters_fills_plaintext():
    kry = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
    idx = {c: i for i, c in enumerate(kry)}
    p_period, q_period, n = 5, 7, 40
    pt, ct = _plant(p_period, q_period, n, 3)
    ciphertext = "".join(kry[c] for c in ct)
    crib = "".join(kry[pt[i]] for i in range(30))
    res = crib_solve_letters(ciphertext, crib, 0, p_period, q_period, alphabet="KRYPTOS")
    truth = "".join(kry[c] for c in pt)
    assert res["consistent"] is True and res["plaintext"] == truth
    # sanity: the crib really maps back through the KRYPTOS alphabet
    assert idx[crib[0]] == pt[0]
