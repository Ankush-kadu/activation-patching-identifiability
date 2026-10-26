"""STRESS TEST: is the single-site rejection rate real detection, or an ill-conditioning artifact?

The calibration already in ioi_query_chart.py permutes the readout WHOLESALE, which destroys the
main effects too.  That is a weak null: it only shows the test has correct size when nothing at
all is going on.  The question that matters is different -- when the main effects and the real
(near-degenerate) design are PRESENT and only the interaction is absent, does the single-site
Wald test still reject at its nominal rate?

Residual bootstrap under the additive null:  Y* = (additive fit) + resampled full-model residuals.
Main effects, design conditioning and error scale are all preserved exactly; the interaction is
zero by construction.
"""
import glob, hashlib, sys
import numpy as np
from identifiability import ADD, CROSS, dm, partial_info, wald

B = 400


def null_reject_rate(Z, Y, W, rng):
    Ma, Mc = dm(W(Z), ADD), dm(W(Z), CROSS)
    G, R = partial_info(Z, W)
    n = len(Z)
    ya, *_ = np.linalg.lstsq(Ma, Y, rcond=None)
    yr = Y - Ma @ ya
    try:
        c = np.linalg.solve(G * n, R.T @ yr)
    except np.linalg.LinAlgError:
        return None
    fit_add = Ma @ ya                      # additive-only fitted values
    e = yr - R @ c                         # residuals from the FULL model
    k = 0
    for _ in range(B):
        Ys = fit_add + rng.permutation(e)  # additive truth + real noise, NO interaction
        ya2, *_ = np.linalg.lstsq(Ma, Ys, rcond=None)
        yr2 = Ys - Ma @ ya2
        try:
            c2 = np.linalg.solve(G * n, R.T @ yr2)
        except np.linalg.LinAlgError:
            continue
        r2 = yr2 - R @ c2
        s2 = float(r2 @ r2) / max(n - len(ADD) - len(CROSS), 1)
        _, p = wald(c2, s2 * np.linalg.inv(G * n))
        k += (p < 0.05)
    return k / B


import json
OUT = {}
for tag, pat in [("ioi", "qchart_*.npz"), ("induction", "qind_*.npz"), ("control", "qctrl_*.npz")]:
    rs, rj = [], []
    for f in sorted(glob.glob(pat)):
        d = np.load(f)
        allZ = np.vstack([d["Zs"], d["Zjm"]]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
        W = lambda Z: (Z - mu) / sd
        # DETERMINISTIC seed: Python's str hash is salted per process (PYTHONHASHSEED), so
        # hash(f) would make this artifact irreproducible and break ledger re-verification.
        rng = np.random.default_rng(int(hashlib.sha256(f.encode()).hexdigest()[:8], 16))
        a = null_reject_rate(d["Zs"], d["Ys"], W, rng)
        b = null_reject_rate(d["Zjm"], d["Yjm"], W, rng)
        if a is not None: rs.append(a)
        if b is not None: rj.append(b)
    OUT[tag] = dict(n=len(rs), B=B, size_S=float(np.mean(rs)), size_J=float(np.mean(rj)),
                    max_S=float(max(rs)), max_J=float(max(rj)))
    print(f"{tag:>10}  n={len(rs):>3}   single-site size {100*np.mean(rs):5.1f}%  "
          f"(max {100*max(rs):5.1f}%)   joint size {100*np.mean(rj):5.1f}%  (max {100*max(rj):5.1f}%)"
          f"   [nominal 5.0%]")
json.dump(OUT, open("additive_null.json", "w"), indent=1)
print("\nsaved -> additive_null.json")
