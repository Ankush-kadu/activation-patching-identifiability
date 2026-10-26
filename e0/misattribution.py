"""WHAT DOES THE UNIDENTIFIED INTERACTION COST A PER-COMPONENT PATCHING NUMBER?

Section 7 establishes that the interaction coefficient is not identified by single-site evidence.
That is a statement about a design.  This turns it into a statement about a REPORTED NUMBER.

A per-component patching number is obtained by patching B alone.  On that arm the upstream
coordinate is pinned, a = a0, so the number measured is

    reported  =  Y(a0, b) - Y(a0, b0).

It is then read as "B contributes this much" -- a claim about B's effect wherever A happens to be,
not only at a = a0.  Under a mechanism with an interaction those differ, and the gap is

    error(a)  =  c . [ h(a, b) - h(a, b0) ]  -  c . [ h(a0, b) - h(a0, b0) ],

which for the bilinear term is exactly c(a - a0)(b - b0): the blind generator of Section 2.  The
misattribution IS the direction the single-site family cannot see.

So we can price it.  Take the confidence region each family licenses for c, and ask how large
error(a) can be over the states the JOINT family actually reaches, as a fraction of the reported
number.  The bound is the exact maximum of a linear functional over the Wald ellipsoid,

    max |D c|  over  (c - c_hat)' V^{-1} (c - c_hat) <= chi2_{3,0.95}
        =  |D c_hat| + sqrt( chi2_{3,0.95} * D' V D ),

with no box relaxation, so it uses the full covariance including the correlations among the cross
terms.  Reported as a percentage of the measured per-component swing.
"""
import glob
import json

import numpy as np
from scipy.stats import chi2

from identifiability import ADD, CROSS, dm, partial_info

CRIT = float(chi2.ppf(0.95, len(CROSS)))


def fit(Z, Y, W):
    """Cross coefficients, their covariance, and the FWL information -- as in Section 7."""
    Ma = dm(W(Z), ADD)
    G, R = partial_info(Z, W)
    n = len(Z)
    ya, *_ = np.linalg.lstsq(Ma, Y, rcond=None)
    yr = Y - Ma @ ya
    c = np.linalg.solve(G * n, R.T @ yr)
    resid = yr - R @ c
    s2 = float(resid @ resid) / max(n - len(ADD) - len(CROSS), 1)
    return c, s2 * np.linalg.inv(G * n)


def worst_case(D, c, V):
    """Exact max of |D c| over the 95% Wald ellipsoid, per row of D."""
    centre = np.abs(D @ c)
    rad = np.sqrt(np.maximum(CRIT * np.einsum("ij,jk,ik->i", D, V, D), 0.0))
    return centre + rad


def one(f):
    d = np.load(f)
    Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
    allZ = np.vstack([Zs, Zj])
    mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd

    n = len(Zs) // 2                      # rows 0:n patch A, rows n: patch B
    armB = Zs[n:]
    a0 = float(np.median(armB[:, 0]))     # the value A is pinned at while B is patched

    # the per-component number a practitioner would report: the readout swing from patching B
    # alone.  This is the quantity that gets quoted as "head B contributes X".
    reported = float(np.ptp(Ys[n:]))
    if reported < 1e-9:
        return None

    # how the cross terms move when A leaves its pinned value, over the joint reachable set
    Zj0 = Zj.copy()
    Zj0[:, 0] = a0
    D = dm(W(Zj), CROSS) - dm(W(Zj0), CROSS)

    out = {"file": f, "reported": reported, "a0": a0}
    for tag, (Z, Y) in (("S", (Zs, Ys)), ("J", (Zj, Yj))):
        try:
            c, V = fit(Z, Y, W)
        except np.linalg.LinAlgError:
            return None
        w = worst_case(D, c, V)
        out[f"max_{tag}"] = float(np.max(w) / reported)
        out[f"med_{tag}"] = float(np.median(w) / reported)
    return out


def coverage_selftest(trials=300, seed=0):
    """The bound is only worth quoting if it covers the truth. Build data with a KNOWN
    interaction and measure realised coverage of the 95% ellipsoid bound."""
    rng = np.random.default_rng(seed)
    cov = k = 0
    for _ in range(trials):
        ga, gb = rng.uniform(-1, 1, 16), rng.uniform(-1, 1, 16)
        Z = np.array([[x, y] for x in ga for y in gb])
        mu, sd = Z.mean(0), Z.std(0) + 1e-12
        W = lambda q: (q - mu) / sd
        c_true, beta = rng.normal(0, 0.5, len(CROSS)), rng.normal(0, 1, len(ADD))
        Y = dm(W(Z), ADD) @ beta + dm(W(Z), CROSS) @ c_true + rng.normal(0, 0.05, len(Z))
        Z0 = Z.copy(); Z0[:, 0] = float(np.median(ga))
        D = dm(W(Z), CROSS) - dm(W(Z0), CROSS)
        try:
            ch, V = fit(Z, Y, W)
        except np.linalg.LinAlgError:
            continue
        cov += bool(np.all(np.abs(D @ c_true) <= worst_case(D, ch, V) + 1e-12))
        k += 1
    return cov / max(k, 1)


if __name__ == "__main__":
    rows = [r for r in (one(f) for f in sorted(glob.glob("qchart_*.npz"))) if r]
    if not rows:
        raise SystemExit("no qchart_*.npz captures found -- run ioi_query_chart.py first")

    g = lambda k: np.array([r[k] for r in rows])
    res = dict(n=len(rows),
               med_S=float(np.median(g("med_S"))), med_J=float(np.median(g("med_J"))),
               max_S=float(np.median(g("max_S"))), max_J=float(np.median(g("max_J"))),
               frac_S_over_1=float(np.mean(g("med_S") > 1.0)),
               frac_J_over_1=float(np.mean(g("med_J") > 1.0)),
               ratio=float(np.median(g("med_S")) / max(np.median(g("med_J")), 1e-300)),
               rows=rows)
    res["coverage"] = coverage_selftest()
    res["over_tenth_S"] = int(np.sum(g("med_S") > 0.10))
    res["over_tenth_J"] = int(np.sum(g("med_J") > 0.10))
    res["over_one_S"] = int(np.sum(g("med_S") > 1.0))
    res["max_J_any"] = float(np.max(g("med_J")))
    json.dump(res, open("misattribution.json", "w"), indent=1, default=float)

    print(f"  n = {res['n']} published S-Inhibition -> Name Mover captures\n")
    print(f"{'':>34}{'single-site':>14}{'joint':>12}")
    print(f"{'typical un-excluded misattribution':>34}{100*res['med_S']:>13.0f}%{100*res['med_J']:>11.1f}%")
    print(f"{'worst case over reachable states':>34}{100*res['max_S']:>13.0f}%{100*res['max_J']:>11.1f}%")
    print(f"{'captures where it exceeds 100%':>34}{100*res['frac_S_over_1']:>13.0f}%"
          f"{100*res['frac_J_over_1']:>11.0f}%")
    print(f"\n  ratio of the two licences: {res['ratio']:.0f}x")
    print(f"  bound self-test: covers the true misattribution in "
          f"{100*res['coverage']:.1f}% of synthetic trials (nominal 95%)")
    print("\nsaved -> misattribution.json")
