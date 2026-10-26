"""IS THE INTERACTION COEFFICIENT IDENTIFIED?  (added 2026-08-27, round-5 audit)

WHAT THIS REPLACES.  The submitted draft reported that the interaction coefficient inferred from
single-site evidence differs from its joint-evidence value by a median factor of 410x (GPT-2),
44.5x (Qwen) and 716x on the published IOI heads.  Those are ratios of two POINT ESTIMATES whose
denominator is ~0: on the S-Inhibition -> Name Mover heads the joint-evidence interaction
coefficient is 0.001-0.014 in whitened units, i.e. the mechanism is ADDITIVE in that chart.  A
ratio whose denominator is consistent with zero is not evidence, and the project's own logging
rule says so.

WHAT IS ACTUALLY TRUE, AND IS STRONGER.  The interaction coefficients are NOT IDENTIFIED by
single-site evidence.  Write the degree-3 mechanism class as [additive columns | cross columns].
Partialling the additive columns out of the cross columns (Frisch-Waugh-Lovell) gives the
residualised design R whose Gram R^T R is the Fisher information for the interaction
coefficients GIVEN the additive ones.  Under single-site evidence that matrix is near-singular:
the cross terms a*b, a^2 b, a b^2 are confounded with the additive ones because one coordinate is
pinned on each arm.  The consequences are measurable without ever dividing by a near-zero number:

    se(c_x) under S  vs  under J      a ratio of STANDARD ERRORS  (well defined, both > 0)
    sign-flip rate of c_x across random halves of the SAME family
    |c_x| / se(c_x) under J           is the interaction identified, and is it zero?

The scientific statement then does not depend on the interaction being large:
**single-site patching evidence cannot license the additivity assumption even when additivity
happens to be true**, because the evidence carries almost no information about the coefficient
that assumption sets to zero.  A practitioner reading a per-component patching number as a
contribution is relying on an assumption their evidence could not have checked.
"""
import argparse, glob, json
from scipy.stats import chi2
import numpy as np
from cells import CELLS

ADD = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (3, 0), (0, 3)]     # no cross terms
FULL = [(i, t - i) for t in range(4) for i in range(t + 1)]
CROSS = [e for e in FULL if e not in ADD]                          # ab, a^2 b, a b^2


def dm(Z, exps):
    return np.array([[a ** i * b ** j for (i, j) in exps] for a, b in Z], float)


def partial_info(Z, W):
    """Fisher information for the cross coefficients GIVEN the additive ones (FWL)."""
    Ma, Mc = dm(W(Z), ADD), dm(W(Z), CROSS)
    # residualise the cross columns on the additive ones
    coef, *_ = np.linalg.lstsq(Ma, Mc, rcond=None)
    R = Mc - Ma @ coef
    return R.T @ R / len(Z), R


def wald(coef, cov):
    """Joint Wald test of H0: every interaction coefficient is zero.

    ||coef||/||se||, used in an earlier version, is a ratio of vector NORMS, not a t-statistic:
    it ignores the correlations among the cross terms and its null expectation is ~0.8, not 0.
    The correct object for a 3-dimensional coefficient block is coef' Cov^{-1} coef, chi-square
    with 3 degrees of freedom under the null.
    """
    try:
        W2 = float(coef @ np.linalg.solve(cov, coef))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return W2, float(chi2.sf(W2, len(coef)))


def coef_and_se(Z, Y, W):
    """Interaction coefficients, their standard errors, and the partial information spectrum."""
    G, R = partial_info(Z, W)
    Ma = dm(W(Z), ADD)
    ya, *_ = np.linalg.lstsq(Ma, Y, rcond=None)
    yr = Y - Ma @ ya                                    # readout, additive part removed
    n, p = len(Z), len(CROSS)
    ev = np.linalg.eigvalsh(G)
    lam_min, lam_max = float(ev.min()), float(ev.max())
    try:
        c = np.linalg.solve(G * n, R.T @ yr)
        resid = yr - R @ c
        s2 = float(resid @ resid) / max(n - len(ADD) - p, 1)
        Vc = s2 * np.linalg.inv(G * n)                  # covariance of the cross coefficients
        se = np.sqrt(np.maximum(np.diag(Vc), 0.0))
        W_, pW = wald(c, Vc)
        ok = True
    except np.linalg.LinAlgError:
        c, se, ok = np.full(p, np.nan), np.full(p, np.inf), False
        W_, pW = float("nan"), float("nan")
    return dict(coef=c, se=se, lam_min=lam_min, lam_max=lam_max, wald=W_, wald_p=pW,
                cond=float(lam_max / max(lam_min, 1e-300)), invertible=ok)


def sign_flips(Z, Y, W, reps=40, seed=0):
    """How often does the fitted interaction coefficient flip sign between random halves?"""
    rng = np.random.default_rng(seed)
    flips = 0
    for r in range(reps):
        i = rng.permutation(len(Z))
        a, b = i[: len(i) // 2], i[len(i) // 2:]
        ca = coef_and_se(Z[a], Y[a], W)["coef"]
        cb = coef_and_se(Z[b], Y[b], W)["coef"]
        if np.isfinite(ca[0]) and np.isfinite(cb[0]) and np.sign(ca[0]) != np.sign(cb[0]):
            flips += 1
    return flips / reps


def analyse(Zs, Ys, Zj, Yj):
    allZ = np.vstack([Zs, Zj])
    mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    S, J = coef_and_se(Zs, Ys, W), coef_and_se(Zj, Yj, W)
    seS, seJ = float(np.linalg.norm(S["se"])), float(np.linalg.norm(J["se"]))
    cS, cJ = float(np.linalg.norm(S["coef"])), float(np.linalg.norm(J["coef"]))
    return dict(
        info_min_S=S["lam_min"], info_min_J=J["lam_min"],
        info_ratio=J["lam_min"] / max(S["lam_min"], 1e-300),
        cond_S=S["cond"], cond_J=J["cond"],
        se_S=seS, se_J=seJ, se_ratio=seS / max(seJ, 1e-300),
        coef_S=cS, coef_J=cJ,
        # is the interaction identified, and is it distinguishable from zero, per family?
        t_S=cS / max(seS, 1e-300), t_J=cJ / max(seJ, 1e-300),
        wald_p_S=S["wald_p"], wald_p_J=J["wald_p"],
        signflip_S=sign_flips(Zs, Ys, W), signflip_J=sign_flips(Zj, Yj, W),
        std_Yj=float(np.std(Yj)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", action="store_true",
                    help="use the support-matched captures (Zjm) instead of the frozen ones")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = {}
    print(f"{'cell':<24}{'n':>3}{'lam_min S':>12}{'lam_min J':>12}{'se_S/se_J':>11}"
          f"{'|c| S':>9}{'|c| J':>9}{'t_S':>7}{'t_J':>7}{'flip S':>8}{'flip J':>8}")
    print("-" * 112)
    for cell in CELLS:
        if A.matched:
            task = cell["label"].split()[-1].lower()
            tag = "gpt2" if cell["label"].startswith("GPT-2") else "qwen35"
            files = sorted(glob.glob(f"{'ioi' if task=='ioi' else task}_matched_{tag}_*.npz"))
            jz, jy = "Zjm", "Yjm"
        else:
            files, jz, jy = sorted(glob.glob(cell["pattern"])), "Zj", "Yj"
        rows = []
        for f in files:
            d = np.load(f)
            if "Ys" not in d:
                continue
            r = analyse(d["Zs"], d["Ys"], d[jz], d[jy])
            r["file"] = f
            rows.append(r)
        if not rows:
            continue
        med = lambda k: float(np.median([r[k] for r in rows]))
        out[cell["label"]] = dict(n=len(rows), rows=rows,
                                  **{k: med(k) for k in
                                     ("info_min_S", "info_min_J", "info_ratio", "se_ratio",
                                      "coef_S", "coef_J", "t_S", "t_J",
                                      "signflip_S", "signflip_J", "cond_S", "cond_J",
                                      "wald_p_S", "wald_p_J")},
                                  wald_rej_S=int(np.sum([r["wald_p_S"] < 0.05 for r in rows])),
                                  wald_rej_J=int(np.sum([r["wald_p_J"] < 0.05 for r in rows])))
        o = out[cell["label"]]
        print(f"{cell['label']:<24}{o['n']:>3}{o['info_min_S']:>12.2e}{o['info_min_J']:>12.2e}"
              f"{o['se_ratio']:>11.1f}{o['coef_S']:>9.3f}{o['coef_J']:>9.4f}"
              f"{o['t_S']:>7.2f}{o['t_J']:>7.2f}{o['signflip_S']:>8.2f}{o['signflip_J']:>8.2f}")
    fn = A.out or ("identifiability_matched.json" if A.matched else "identifiability.json")
    json.dump(out, open(fn, "w"), indent=1, default=float)
    print(f"\nsaved -> {fn}")
    print("\nlam_min = smallest eigenvalue of the Fisher information for the interaction")
    print("coefficients given the additive ones (Frisch-Waugh-Lovell).  se_S/se_J is a ratio of")
    print("STANDARD ERRORS, so neither denominator is near zero.")
    print("Whether a family can distinguish the interaction from zero is the joint WALD test")
    print("(coef' Cov^-1 coef ~ chi2_3), NOT |c|/se, which is a ratio of vector norms whose null")
    print("expectation is ~0.8.  Rejections at the 0.05 level:")
    for k, v in out.items():
        print(f"   {k:<24} joint {v['wald_rej_J']:>3}/{v['n']:<3}   single-site {v['wald_rej_S']:>3}/{v['n']:<3}")
