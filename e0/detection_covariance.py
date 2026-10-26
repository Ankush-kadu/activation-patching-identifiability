"""The IOI attribution analysis's interaction tests under covariance estimators that respect the design.

WHY THIS FILE EXISTS (2026-09-23).  The paper contrasted single-site rejections of zero interaction
on the published IOI head pairs (26/31) with none on control heads (0/28).  That contrast uses the
homoskedastic covariance s^2 (R'R)^{-1} with one pooled residual variance.  But the two single-site
arms have very different residual variances (arm A / arm B median 0.52 for IOI and 0.0023 for
controls), so the pooled s^2 inflates the control covariance and deflates its test statistic.  An
independent check re-ran the tests with arm-appropriate covariances and found the controls rejecting
in roughly a third to a half of configurations.  This script computes those tests as an artifact the
ledger can read, and re-derives the shipped homoskedastic counts first.

Everything is in the paper's own chart (analyse(): centre/scale on the pooled single-site and joint
states), with the additive block partialled out as in misattribution.fit (Frisch-Waugh-Lovell).

Single-site covariance estimators for the interaction block:
  homo    s^2 (R'R)^{-1}                               (the shipped test; reproduces 26/31, 0/28)
  hc0     (R'R)^{-1} R' diag(e^2) R (R'R)^{-1}
  hc3     as hc0 with e_i^2 / (1 - h_ii)^2, h from the full [additive, interaction] design
  arm     sandwich with one residual variance per arm
  fgls    refit weighting each row by 1 / (its arm's residual variance), model-based covariance
  cr1     donor-clustered (a donor feeds both arms, so 32 clusters), CR1 small-sample factor
Joint family: homo, hc0, and a conservative two-way (upstream donor x downstream donor) cluster-robust
covariance V_up + V_down, since each joint row pairs two donors.  Also the power of the joint Wald
test at the single-site estimate under that two-way covariance.

    python3 detection_covariance.py      # -> detection_covariance.json
"""
import glob, json, os
import numpy as np
from scipy.stats import chi2, ncx2
from identifiability import ADD, CROSS, dm
from detection_consistency import GROUPS, chart, N_AL

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
K = len(CROSS)
CRIT = float(chi2.ppf(1 - ALPHA, K))


def _fwl(Z, Y, W, w=None):
    """Partial the additive block out of the interaction block and the readout; optional row weights."""
    Ma, Mc = dm(W(Z), ADD), dm(W(Z), CROSS)
    if w is not None:
        s = np.sqrt(w)[:, None]; Ma, Mc, Y = Ma * s, Mc * s, Y * s[:, 0]
    ca, *_ = np.linalg.lstsq(Ma, Mc, rcond=None); R = Mc - Ma @ ca
    ya, *_ = np.linalg.lstsq(Ma, Y, rcond=None); yr = Y - Ma @ ya
    G = R.T @ R; c = np.linalg.solve(G, R.T @ yr); e = yr - R @ c
    return c, R, e, G, np.hstack([Ma, Mc])


def _wald(c, V):
    T = float(c @ np.linalg.solve(V, c)); return T, float(chi2.sf(T, K))


def _sandwich(R, G, meat):
    Gi = np.linalg.inv(G); return Gi @ meat @ Gi


def single_site(Z, Y, W):
    n = len(Z); half = n // 2
    arm = np.r_[np.zeros(half, int), np.ones(n - half, int)]
    donor = (np.arange(n) % half) // N_AL                     # same donor feeds both arms
    c, R, e, G, X = _fwl(Z, Y, W)
    p = X.shape[1]
    out = {}
    s2 = float(e @ e) / (n - p)
    out["homo"] = _wald(c, s2 * np.linalg.inv(G))[1]
    out["hc0"] = _wald(c, _sandwich(R, G, (R * (e ** 2)[:, None]).T @ R))[1]
    h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)
    out["hc3"] = _wald(c, _sandwich(R, G, (R * (e ** 2 / np.clip(1 - h, 1e-8, None) ** 2)[:, None]).T @ R))[1]
    s2a = np.array([float(e[arm == a] @ e[arm == a]) / max((arm == a).sum() - p / 2, 1) for a in (0, 1)])
    out["arm"] = _wald(c, _sandwich(R, G, (R * s2a[arm][:, None]).T @ R))[1]
    cw, Rw, ew, Gw, _ = _fwl(Z, Y, W, w=1.0 / s2a[arm])
    out["fgls"] = _wald(cw, (float(ew @ ew) / (n - p)) * np.linalg.inv(Gw))[1]
    g_ids = np.unique(donor); meat = np.zeros((K, K))
    for g in g_ids:
        u = R[donor == g].T @ e[donor == g]; meat += np.outer(u, u)
    ng = len(g_ids); fac = ng / (ng - 1) * (n - 1) / (n - p)
    out["cr1"] = _wald(c, fac * _sandwich(R, G, meat))[1]
    return c, out


def _cluster_meat(R, e, ids):
    meat = np.zeros((K, K))
    for g in np.unique(ids):
        u = R[ids == g].T @ e[ids == g]; meat += np.outer(u, u)
    return meat


def joint(Z, Y, W, n_donor):
    n = len(Z); ne = len(range(0, n_donor, 2)); assert ne * ne == n, (ne, n)
    gi, gj = np.arange(n) // ne, np.arange(n) % ne            # upstream-major, downstream-minor
    c, R, e, G, X = _fwl(Z, Y, W)
    p = X.shape[1]; s2 = float(e @ e) / (n - p)
    out = {"homo": _wald(c, s2 * np.linalg.inv(G))[1],
           "hc0": _wald(c, _sandwich(R, G, (R * (e ** 2)[:, None]).T @ R))[1]}
    # Two-way clustering. The Cameron-Gelbach-Miller form V_i + V_j - V_ij is not positive
    # semi-definite in 27 of the 59 configurations here, and clipping its negative eigenvalues leaves
    # near-zero variance in those directions, which inflates the test. We use the conservative
    # V_i + V_j instead: always PSD and never smaller, so its power at c_S is a lower bound.
    V2 = _sandwich(R, G, _cluster_meat(R, e, gi) + _cluster_meat(R, e, gj))
    out["cr2way"] = _wald(c, V2)[1]
    return c, out, V2


def main():
    res = {}
    for grp, (pat, _, exp_S, exp_J, exp_n) in GROUPS.items():
        rows = []
        for f in sorted(glob.glob(os.path.join(HERE, pat))):
            d = np.load(f); Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
            W = chart(np.vstack([Zs, Zj]))
            n_donor = (len(Zs) // 2) // N_AL
            cS, pS = single_site(Zs, Ys, W)
            cJ, pJ, V2 = joint(Zj, Yj, W, n_donor)
            power = float(ncx2.sf(CRIT, K, float(cS @ np.linalg.solve(V2, cS))))
            rows.append(dict(file=os.path.basename(f), p_S=pS, p_J=pJ, power_J2way_at_cS=power))
        cnt = lambda fam, est: int(sum(r[f"p_{fam}"][est] < ALPHA for r in rows))
        summ = {"n": len(rows),
                "rej_S": {e: cnt("S", e) for e in ("homo", "hc0", "hc3", "arm", "fgls", "cr1")},
                "rej_J": {e: cnt("J", e) for e in ("homo", "hc0", "cr2way")}}
        rejS = [r for r in rows if r["p_S"]["homo"] < ALPHA]
        summ["min_power_J2way_at_cS_among_rej_S"] = float(min(r["power_J2way_at_cS"] for r in rejS)) if rejS else None
        assert summ["n"] == exp_n and summ["rej_S"]["homo"] == exp_S and summ["rej_J"]["homo"] == exp_J, \
            f"[REPRO] {grp}: {summ['n']}, {summ['rej_S']['homo']}, {summ['rej_J']['homo']} vs {exp_n}, {exp_S}, {exp_J}"
        res[grp] = dict(summary=summ, rows=rows)
        print(f"{grp:8s} n={summ['n']}  single-site rejections {summ['rej_S']}  joint {summ['rej_J']}  "
              f"min joint power at c_S (2-way) {summ['min_power_J2way_at_cS_among_rej_S']}")
    json.dump(res, open(os.path.join(HERE, "detection_covariance.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
