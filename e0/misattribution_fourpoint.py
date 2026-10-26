"""Misattribution bound with the contrast misattribution.py's own docstring specifies.

WHY THIS FILE EXISTS (2026-09-23).  misattribution.py defines the gap between the measured
per-component number and its transported reading as

    error(a) = c . [ h(a, b) - h(a, b0) ]  -  c . [ h(a0, b) - h(a0, b0) ]          (its docstring)

but its implementation computes only  h(a, b) - h(a0, b)  -- the two baseline-B terms are missing.
Pooled standardisation does not put the clean B value at zero, so the two differ; for h = ab with
a0 = 0, b0 = 1, a = b = 1 the documented contrast is 0 and the implemented one is 1.  An independent
pre-submission review found this by re-running the supplement.  The same two-term shortcut is in the
coverage self-test.  misattribution.py is left untouched as the historical record; this file computes
the documented four-point contrast with the SAME estimator, covariance, Wald region, normalisation
and aggregation, and is what the ledger now reads.

The clean B value b0 is read from the clamped captures: on their patch-A arm B is held at its clean
value, so that coordinate is constant there (its spread is checked and recorded).

    python3 misattribution_fourpoint.py      # -> misattribution_fourpoint.json
"""
import glob
import json
import numpy as np
from identifiability import ADD, CROSS, dm
from misattribution import fit, worst_case


def four_point(W, Z, a0, b0):
    """D = H(a,b) - H(a0,b) - H(a,b0) + H(a0,b0), row by row, in the standardised chart."""
    Za0 = Z.copy(); Za0[:, 0] = a0
    Zb0 = Z.copy(); Zb0[:, 1] = b0
    Z00 = Z.copy(); Z00[:, 0] = a0; Z00[:, 1] = b0
    return dm(W(Z), CROSS) - dm(W(Za0), CROSS) - dm(W(Zb0), CROSS) + dm(W(Z00), CROSS)


def one(f):
    d = np.load(f)
    Zs, Ys, Zj, Yj, Zc = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"], d["Zc"]
    allZ = np.vstack([Zs, Zj])
    mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    n = len(Zs) // 2
    a0 = float(np.median(Zs[n:, 0]))                 # A pinned while B is patched (as misattribution.py)
    clamped_arm = Zc[: len(Zc) // 2, 1]
    b0, b0_spread = float(np.median(clamped_arm)), float(np.std(clamped_arm))
    reported = float(np.ptp(Ys[n:]))                 # the B-only readout swing, as misattribution.py
    if reported < 1e-9:
        return None
    D = four_point(W, Zj, a0, b0)
    out = {"file": f, "reported": reported, "a0": a0, "b0": b0, "b0_spread": b0_spread}
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
    """As misattribution.coverage_selftest, but checking coverage of the FOUR-POINT contrast.  Like
    the original, the synthetic data are drawn from the declared class with independent Gaussian
    noise, so this checks the calculation under the model's own assumptions; it does not establish
    coverage under dependent donors or a misspecified class."""
    rng = np.random.default_rng(seed)
    cov = k = 0
    for _ in range(trials):
        ga, gb = rng.uniform(-1, 1, 16), rng.uniform(-1, 1, 16)
        Z = np.array([[x, y] for x in ga for y in gb])
        mu, sd = Z.mean(0), Z.std(0) + 1e-12
        W = lambda q: (q - mu) / sd
        c_true, beta = rng.normal(0, 0.5, len(CROSS)), rng.normal(0, 1, len(ADD))
        Y = dm(W(Z), ADD) @ beta + dm(W(Z), CROSS) @ c_true + rng.normal(0, 0.05, len(Z))
        D = four_point(W, Z, float(np.median(ga)), float(np.median(gb)))
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
        raise SystemExit("no qchart_*.npz captures found")
    g = lambda k: np.array([r[k] for r in rows])
    res = dict(n=len(rows),
               med_S=float(np.median(g("med_S"))), med_J=float(np.median(g("med_J"))),
               max_S=float(np.median(g("max_S"))), max_J=float(np.median(g("max_J"))),
               frac_S_over_1=float(np.mean(g("med_S") > 1.0)), frac_J_over_1=float(np.mean(g("med_J") > 1.0)),
               ratio=float(np.median(g("med_S")) / max(np.median(g("med_J")), 1e-300)),
               ordered=int(np.sum(g("med_S") > g("med_J"))),
               max_b0_spread=float(np.max(g("b0_spread"))),
               rows=rows)
    res["coverage"] = coverage_selftest()
    res["over_tenth_S"] = int(np.sum(g("med_S") > 0.10)); res["over_tenth_J"] = int(np.sum(g("med_J") > 0.10))
    res["over_one_S"] = int(np.sum(g("med_S") > 1.0)); res["max_J_any"] = float(np.max(g("med_J")))
    json.dump(res, open("misattribution_fourpoint.json", "w"), indent=1, default=float)
    print(f"n={res['n']}  median single-site {100*res['med_S']:.2f}%  joint {100*res['med_J']:.3f}%  "
          f"ratio {res['ratio']:.1f}  ordered {res['ordered']}/{res['n']}")
    print(f"worst state: single-site {100*res['max_S']:.0f}%  joint {100*res['max_J']:.1f}%")
    print(f"self-test coverage (four-point, model-conditional) {100*res['coverage']:.1f}%  "
          f"| max clamped-B spread {res['max_b0_spread']:.2e}")
