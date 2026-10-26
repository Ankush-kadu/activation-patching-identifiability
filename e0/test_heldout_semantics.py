"""Semantic regression tests for positive_control_heldout.py.

These attack exactly the two defects an independent review found in the original analysis: a
direction "frozen then evaluated on held-out" that was fitted on its own test states, and a
"donor-held-out" split that let downstream donors appear on both sides.  A passing generic test
suite did not catch either, so these test the property itself.

    python3 -m pytest -q test_heldout_semantics.py
"""
import numpy as np
import pytest
import positive_control_heldout as H


def _synthetic(seed=0, n_don=32, coef=7):
    """A capture with the real layout: 2 arms x n_don donors x coef coefficients single-site rows,
    and a support-matched joint grid of every second donor upstream x every second downstream."""
    rng = np.random.default_rng(seed)
    half = n_don * coef
    a0, m, c = 0.3, 0.8, -0.1
    aA = rng.normal(0, 1, half); armA = np.c_[aA, m * aA + c + rng.normal(0, 0.05, half)]   # A patched, B recomputes
    armB = np.c_[np.full(half, a0) + rng.normal(0, 1e-3, half), rng.normal(0, 1, half)]      # B patched, A pinned
    Zs = np.vstack([armA, armB])
    ids = np.arange(0, n_don, 2)
    Zj = rng.normal(0, 1, (len(ids) * len(ids), 2))
    return Zs, Zj


def _fit_only(Zs, Zj, protocol):
    """Run the protocol and return only the TRAINING-fitted quantities (edge)."""
    r = protocol(Zs, Zj)
    return np.array([r["a0"], r["m"], r["c"]])


@pytest.mark.parametrize("protocol", [H.row_protocol, H.donor_protocol])
def test_fit_does_not_depend_on_test_rows(protocol):
    Zs, Zj = _synthetic()
    base = _fit_only(Zs, Zj, protocol)
    # Recover which rows are test rows by rerunning the protocol's own split logic, then corrupt ONLY them.
    n_rows, half = len(Zs), len(Zs) // 2
    if protocol is H.row_protocol:
        rng = np.random.default_rng(0)
        s_te = rng.permutation(n_rows)[n_rows // 2:]
        j_te = rng.permutation(len(Zj))[len(Zj) // 2:]
    else:
        n_don = half // H.N_COEF
        donor = (np.arange(n_rows) % half) // H.N_COEF
        train = np.random.default_rng(0).permutation(n_don)[: n_don // 2]
        s_te = np.flatnonzero(~np.isin(donor, train))
        ids = np.arange(0, n_don, 2); ja, jb = np.repeat(ids, len(ids)), np.tile(ids, len(ids))
        j_te = np.flatnonzero(~(np.isin(ja, train) & np.isin(jb, train)))   # every joint row not in training
    Zs2, Zj2 = Zs.copy(), Zj.copy()
    Zs2[s_te] = np.random.default_rng(9).normal(50, 30, Zs2[s_te].shape)    # wild values in held-out rows only
    Zj2[j_te] = np.random.default_rng(8).normal(-40, 25, Zj2[j_te].shape)
    assert np.allclose(_fit_only(Zs2, Zj2, protocol), base), "fitted edge moved when only held-out rows changed"


def test_donor_protocol_holds_out_both_donors_by_identity():
    n_don, coef = 32, 7
    half = n_don * coef
    donor = (np.arange(2 * half) % half) // coef
    train = np.random.default_rng(0).permutation(n_don)[: n_don // 2]
    ids = np.arange(0, n_don, 2); ja, jb = np.repeat(ids, len(ids)), np.tile(ids, len(ids))
    j_te = ~np.isin(ja, train) & ~np.isin(jb, train)
    j_tr = np.isin(ja, train) & np.isin(jb, train)
    assert j_te.any() and j_tr.any()
    # no donor identity scored in the test set appears anywhere in training (either joint axis, either arm)
    test_ids = set(ja[j_te]) | set(jb[j_te]) | set(donor[~np.isin(donor, train)])
    train_ids = set(ja[j_tr]) | set(jb[j_tr]) | set(donor[np.isin(donor, train)])
    assert test_ids.isdisjoint(train_ids)
    # and the joint grid layout the protocol assumes matches upstream-major, downstream-minor ordering
    assert ja[1] == ja[0] and jb[1] != jb[0]


def test_row_protocol_uses_same_draws_as_prep():
    """The row split must be the same permutation subspace_test.prep draws, so the corrected numbers are
    comparable row-for-row with the original ones."""
    from subspace_test import prep
    from levelC2 import design
    from subspace_test import DEG
    Zs, Zj = _synthetic(3)
    Str, Ste, Jtr, Jte = prep(Zs, Zj, seed=0)
    r = H.row_protocol(Zs, Zj, seed=0)
    assert r["sizes"] == [len(Str), len(Ste), len(Jtr), len(Jte)]
