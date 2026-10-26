"""IS THE SINGLE-SITE "DETECTION OF NON-ADDITIVITY" REAL?  c_S versus c_J  (added 2026-09-23)

The IOI attribution analysis reported that single-site evidence rejects additivity -- the joint Wald test that the three
cross coefficients (ab, a^2 b, a b^2) of the cubic working model are zero -- in 26/31 IOI
configurations against 0/28 controls, while joint evidence, with ~72x smaller standard errors,
rejects in 7/31 and 5/28.  If the cubic working model were correct on both designs, the single-site
estimate c_S and the joint estimate c_J would estimate the SAME coefficient vector and would have to
agree to within V_S + V_J.  This script, for every configuration behind those four counts:

  1. reproduces the counts from the shipped captures with the paper's own analyse() and stops if it
     cannot;
  2. tests H0: c_S = c_J by (c_S - c_J)'(V_S + V_J)^{-1}(c_S - c_J) ~ chi2_3, with (c, V) from
     misattribution.fit() in the paper's pooled chart.  The statistic is invariant to the choice of
     a COMMON affine chart: whitened cross coefficients are an invertible (triangular) linear map of
     raw cross coefficients, and additive terms never generate cross terms under a shift/scale;
  3. tests the cubic working model OUT OF SAMPLE in each family (single-site S, joint J): rows are
     split into donor-disjoint halves, the chart (centring/scaling) and all ten coefficients are
     estimated on the training half only, and Chow's predictive-failure test is applied to the
     held-out half.  Under the working model's own assumptions (the ones fit()'s covariance and every
     Wald p-value rely on: cubic mean, iid homoscedastic Gaussian error), with e the held-out
     prediction errors, s2 the training residual variance on n - p df, and h = X_te(X'X)^{-1}X_te',
         F = [e'(I + h)^{-1} e / m] / s2  ~  F(m, n - p)       exactly.
     Also reported: the literal "half the rows" version (random row split, which leaks donors), the
     reverse direction, and the raw ratio mean(e^2)/s2 against its null expectation 1 + mean(diag h);
  4. supplementary diagnostics that bear on the same question: (a) split-half replication of c
     within each family on donor-disjoint halves, (b) power of the joint Wald test at c = c_S, (c) the
     single-site and joint Wald tests re-run with the additive block enlarged from degree 3 to
     degree 5 (a^4, a^5, b^4, b^5), cross block unchanged, and (d) with BOTH blocks at degree 5
     (10 cross terms, chi2_10), so that a real non-additivity of higher order than the cubic
     working model can express would have a chance to show up in the joint family.

The clamped single-site family (Zc) is not analysed: on it b is pinned on arm A and a on arm B, so
the cross columns are an exact linear combination of the additive ones (partial information ~1e-30);
its info is recorded per row as info_min_C to show this.

Unit of analysis throughout: one configuration = one capture file (head pair x prompt seed).
Aggregation: counts of configurations at alpha = 0.05 and medians over configurations, per group.
Run from anywhere:  python e0/detection_consistency.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2
from scipy.stats import f as fdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import N_AL                                          # noqa: E402  row = donor*N_AL + alpha
from identifiability import ADD, CROSS, FULL, analyse, dm, partial_info, wald  # noqa: E402
from misattribution import fit                                  # noqa: E402

ALPHA = 0.05
# group -> (capture glob, shipped per-configuration JSON, expected rej_S, expected rej_J, n)
GROUPS = {"ioi": ("qchart_*.npz", "ioi_query_chart.json", 26, 7, 31),
          "control": ("qctrl_*.npz", "ioi_query_chart_control.json", 0, 5, 28)}
# head lists as defined in ioi_query_chart.py (S_INHIB/NAME_MOVE l.33-34, CTRL_A/CTRL_B l.194-195)
HEADS = {"ioi": ([(7, 3), (7, 9), (8, 6), (8, 10)], [(9, 9), (9, 6), (10, 0)]),
         "control": ([(0, 5), (1, 2), (2, 9), (3, 4)], [(9, 9), (9, 6), (10, 0)])}
ADD5 = ADD + [(4, 0), (0, 4), (5, 0), (0, 5)]
CROSS5 = [(i, t - i) for t in range(2, 6) for i in range(1, t)]   # every a^i b^j, i,j>=1, i+j<=5
MATCH_KEYS = ("coef_S", "coef_J", "se_S", "se_J", "info_min_S", "info_min_J")
CRIT3 = float(chi2.ppf(1 - ALPHA, len(CROSS)))
NCP50 = float(brentq(lambda l: ncx2.sf(CRIT3, len(CROSS), l) - 0.5, 1e-3, 1e3))  # 50% power


def chart(Z):
    """Centre/scale the two coordinates on the rows given (the same map analyse() builds)."""
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    return lambda Q: (Q - mu) / sd


def label(group, f):
    """Head pair and seed from a capture name such as qchart_810_100_2.npz."""
    _, sa, sb, seed = os.path.basename(f)[:-4].split("_")
    A = {f"{l}{h}": (l, h) for l, h in HEADS[group][0]}
    B = {f"{l}{h}": (l, h) for l, h in HEADS[group][1]}
    return list(A[sa]), list(B[sb]), int(seed)


def donor_halves_S(n):
    """Zs = [arm A; arm B], each donor-major (ioi_query_chart.collect l.151).  Donor v feeds both
    arms (dS[v], dN[v] come from the same donor prompt), so a split by donor parity is disjoint."""
    arm = n // 2
    donor = (np.arange(n) % arm) // N_AL
    return donor % 2 == 0, donor % 2 == 1


def donor_halves_J(n, n_donor):
    """Zjm rows are pairs (i, j) over EVEN donors, i-major (ioi_query_chart.collect l.161).  A
    crossed design: train = both donors in half 0, test = both in half 1; the 50% of rows with one
    donor in each half are dropped, so no donor prompt appears in both halves."""
    ne = len(range(0, n_donor, 2))
    assert ne * ne == n, (ne, n)
    k = np.arange(n)
    gi, gj = (k // ne) % 2, (k % ne) % 2
    return (gi == 0) & (gj == 0), (gi == 1) & (gj == 1)


def random_halves(n, key):
    """The literal 'fit on half the rows' split (leaks donors across halves).  Deterministic seed."""
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    tr = np.zeros(n, bool)
    tr[rng.permutation(n)[: n // 2]] = True
    return tr, ~tr


def chow(Ztr, Ytr, Zte, Yte, exps=FULL):
    """Chow predictive-failure test of the polynomial working model `exps`.  Chart and
    coefficients are estimated on the training rows ONLY."""
    W = chart(Ztr)
    X, Xt = dm(W(Ztr), exps), dm(W(Zte), exps)
    n, p, m = len(X), X.shape[1], len(Xt)
    Q, R = np.linalg.qr(X)
    beta = solve_triangular(R, Q.T @ Ytr)
    e_tr = Ytr - X @ beta
    s2 = float(e_tr @ e_tr) / (n - p)
    e = Yte - Xt @ beta
    A = solve_triangular(R, Xt.T, trans="T")          # Xt (X'X)^{-1} Xt' = A'A
    h = np.einsum("ij,ij->j", A, A)                    # prediction leverages of held-out rows
    q = float(e @ cho_solve(cho_factor(np.eye(m) + A.T @ A), e))
    F = q / m / s2
    return dict(F=F, p=float(fdist.sf(F, m, n - p)), ratio=float(np.mean(e ** 2) / s2),
                ratio_null=float(1.0 + h.mean()), mse_test=float(np.mean(e ** 2)),
                rss_train=float(e_tr @ e_tr), s2_train=s2, n_train=n, n_test=m)


def lof(Z, Y, tr, te):
    """Held-out lack of fit of the cubic model, plus the held-out value of its cross block."""
    full, add = chow(Z[tr], Y[tr], Z[te], Y[te], FULL), chow(Z[tr], Y[tr], Z[te], Y[te], ADD)
    full["gain_test"] = 1.0 - full["mse_test"] / add["mse_test"]
    full["gain_train"] = 1.0 - full["rss_train"] / add["rss_train"]
    return full


def replicate(Z, Y, tr, te):
    """Split-half replication of c on donor-disjoint halves: (c1-c2)'(V1+V2)^{-1}(c1-c2) ~ chi2_3.
    Common chart estimated on half 1 only (the statistic does not depend on it)."""
    W = chart(Z[tr])
    c1, V1 = fit(Z[tr], Y[tr], W)
    c2, V2 = fit(Z[te], Y[te], W)
    return wald(c1 - c2, V1 + V2)


def fit_basis(Z, Y, W, add, cross=CROSS):
    """misattribution.fit() with blocks `add`/`cross` in place of ADD/CROSS (same FWL algebra)."""
    Ma, Mc = dm(W(Z), add), dm(W(Z), cross)
    coef, *_ = np.linalg.lstsq(Ma, Mc, rcond=None)
    R = Mc - Ma @ coef
    ya, *_ = np.linalg.lstsq(Ma, Y, rcond=None)
    yr = Y - Ma @ ya
    G = R.T @ R
    c = np.linalg.solve(G, R.T @ yr)
    res = yr - R @ c
    s2 = float(res @ res) / (len(Z) - len(add) - len(cross))
    return c, s2 * np.linalg.inv(G)


def reproduce(group):
    """Re-run the paper's analyse() on every capture and match each to its shipped JSON row."""
    pat, jfile, _, _, _ = GROUPS[group]
    files = sorted(glob.glob(os.path.join(HERE, pat)))
    ref = json.load(open(os.path.join(HERE, jfile)))["rows"]
    out, used = [], set()
    for f in files:
        d = np.load(f)
        r = analyse(d["Zs"], d["Ys"], d["Zjm"], d["Yjm"])
        dist = [max(abs(r[k] - q[k]) / max(abs(q[k]), 1e-300) for k in MATCH_KEYS) for q in ref]
        k = int(np.argmin(dist))
        if dist[k] > 1e-6 or k in used:
            raise SystemExit(f"[REPRO] {os.path.basename(f)}: no unique shipped row matches "
                             f"(best rel diff {dist[k]:.2e}, row {k}, already used={k in used})")
        used.add(k)
        out.append(dict(file=os.path.basename(f), ref_row=k, wald_p_S=r["wald_p_S"],
                        wald_p_J=r["wald_p_J"], ref_p_S=ref[k]["wald_p_S"],
                        ref_p_J=ref[k]["wald_p_J"], se_ratio=r["se_ratio"],
                        ref_nm=ref[k]["nm"], match_rel_diff=float(dist[k])))
    return out, len(ref)


def analyse_config(group, f):
    d = np.load(f)
    Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
    si, nm, seed = label(group, f)
    n_donor = (len(Zs) // 2) // N_AL

    # --- c_S vs c_J in the paper's pooled chart (identical to analyse()) ---------------------
    W = chart(np.vstack([Zs, Zj]))
    cS, VS = fit(Zs, Ys, W)
    cJ, VJ = fit(Zj, Yj, W)
    WS, pS = wald(cS, VS)
    WJ, pJ = wald(cJ, VJ)
    T, p_cons = wald(cS - cJ, VS + VJ)
    try:
        lam = float(eigh(VJ, VS, eigvals_only=True).max())   # largest eigenvalue of V_S^{-1} V_J
    except np.linalg.LinAlgError:
        lam = float("nan")
    ncp = float(cS @ np.linalg.solve(VJ, cS))                   # J's noncentrality if c = c_S
    power_J = float(ncx2.sf(CRIT3, len(CROSS), ncp))
    info_C = float(np.linalg.eigvalsh(partial_info(d["Zc"], W)[0]).min())
    # fit() pools ONE residual variance over both single-site arms; the cross block is identified
    # only on arm A (a is pinned on arm B).  Per-arm residual variance of the full cubic fit:
    X = dm(W(Zs), FULL)
    e = Ys - X @ np.linalg.lstsq(X, Ys, rcond=None)[0]
    h = len(Zs) // 2
    var_ratio = float(e[:h].var() / e[h:].var())

    # --- degree-5 additive block ---------------------------------------------------------------
    c3, V3 = fit_basis(Zs, Ys, W, ADD)                          # must equal fit(): self-check
    selfcheck = float(np.max(np.abs(c3 - cS)) / max(np.max(np.abs(cS)), 1e-300))
    _, pS5 = wald(*fit_basis(Zs, Ys, W, ADD5))
    _, pJ5 = wald(*fit_basis(Zj, Yj, W, ADD5))
    _, pS55 = wald(*fit_basis(Zs, Ys, W, ADD5, CROSS5))         # any non-additivity to degree 5
    _, pJ55 = wald(*fit_basis(Zj, Yj, W, ADD5, CROSS5))

    # --- held-out lack of fit + split-half replication, per family -----------------------------
    splits = {"S": (Zs, Ys, donor_halves_S(len(Zs))),
              "J": (Zj, Yj, donor_halves_J(len(Zj), n_donor))}
    row = dict(group=group, file=os.path.basename(f), si=si, nm=nm, seed=seed,
               n_S=len(Zs), n_J=len(Zj), wald_S=WS, wald_p_S=pS, wald_J=WJ, wald_p_J=pJ,
               c_S=cS.tolist(), c_J=cJ.tolist(), se_S=np.sqrt(np.diag(VS)).tolist(),
               se_J=np.sqrt(np.diag(VJ)).tolist(), cons_stat=T, cons_p=p_cons,
               lam_max_VSinv_VJ=lam, ncp_J_at_cS=ncp, power_J_at_cS=power_J,
               VJ_inflation_for_50pct_power=ncp / NCP50, resid_var_armA_over_armB=var_ratio,
               wald_p_S_add5=pS5, wald_p_J_add5=pJ5, wald_p_S_deg5=pS55, wald_p_J_deg5=pJ55,
               info_min_C=info_C,
               fit_basis_selfcheck=selfcheck)
    for fam, (Z, Y, (h0, h1)) in splits.items():
        g = lof(Z, Y, h0, h1)
        rv = lof(Z, Y, h1, h0)
        rr = lof(Z, Y, *random_halves(len(Z), row["file"] + fam))
        _, prep = replicate(Z, Y, h0, h1)
        row.update({f"lof_{fam}_F": g["F"], f"lof_{fam}_p": g["p"],
                    f"lof_{fam}_ratio": g["ratio"], f"lof_{fam}_ratio_null": g["ratio_null"],
                    f"lof_{fam}_n_train": g["n_train"], f"lof_{fam}_n_test": g["n_test"],
                    f"lof_{fam}_p_reverse": rv["p"], f"lof_{fam}_ratio_reverse": rv["ratio"],
                    f"lof_{fam}_p_rowsplit": rr["p"], f"lof_{fam}_ratio_rowsplit": rr["ratio"],
                    f"gain_{fam}_test": g["gain_test"], f"gain_{fam}_train": g["gain_train"],
                    f"rep_{fam}_p": prep})
    return row


def summarise(rows):
    n = len(rows)
    g = lambda k: np.array([r[k] for r in rows], float)
    rejS, rejJ = g("wald_p_S") < ALPHA, g("wald_p_J") < ALPHA
    inc = g("cons_p") < ALPHA
    kS = int(rejS.sum())
    med = lambda k, m=None: (float(np.median(g(k)[m])) if (m is None or m.any()) else None)
    s = dict(n=n, rej_S=kS, rej_J=int(rejJ.sum()),
             inconsistent_all=int(inc.sum()),
             inconsistent_among_rej_S=int((inc & rejS).sum()),
             inconsistent_among_rej_S_bonferroni=int((g("cons_p") < ALPHA / max(kS, 1))[rejS].sum()),
             inconsistent_among_nonrej_S=int((inc & ~rejS).sum()),
             median_cons_p_rej_S=med("cons_p", rejS),
             median_lam_max_VSinv_VJ=med("lam_max_VSinv_VJ"),
             median_power_J_at_cS_rej_S=med("power_J_at_cS", rejS),
             min_power_J_at_cS_rej_S=(float(g("power_J_at_cS")[rejS].min()) if kS else None),
             min_VJ_inflation_for_50pct_power_rej_S=(
                 float(g("VJ_inflation_for_50pct_power")[rejS].min()) if kS else None),
             median_resid_var_armA_over_armB=med("resid_var_armA_over_armB"),
             median_p_S=med("wald_p_S"), min_p_S=float(g("wald_p_S").min()),
             rej_S_add5=int((g("wald_p_S_add5") < ALPHA).sum()),
             rej_S_add5_among_rej_S=int((g("wald_p_S_add5") < ALPHA)[rejS].sum()),
             rej_J_add5=int((g("wald_p_J_add5") < ALPHA).sum()),
             rej_S_deg5=int((g("wald_p_S_deg5") < ALPHA).sum()),
             rej_S_deg5_among_rej_S=int((g("wald_p_S_deg5") < ALPHA)[rejS].sum()),
             rej_J_deg5=int((g("wald_p_J_deg5") < ALPHA).sum()))
    for fam in ("S", "J"):
        lo = g(f"lof_{fam}_p") < ALPHA
        s[f"lof_{fam}"] = dict(
            reject=int(lo.sum()), reject_among_rej_S=int(lo[rejS].sum()),
            reject_reverse=int((g(f"lof_{fam}_p_reverse") < ALPHA).sum()),
            reject_rowsplit=int((g(f"lof_{fam}_p_rowsplit") < ALPHA).sum()),
            median_ratio=med(f"lof_{fam}_ratio"), median_ratio_null=med(f"lof_{fam}_ratio_null"),
            median_ratio_reverse=med(f"lof_{fam}_ratio_reverse"),
            median_ratio_rowsplit=med(f"lof_{fam}_ratio_rowsplit"),
            median_gain_test=med(f"gain_{fam}_test"), median_gain_train=med(f"gain_{fam}_train"),
            gain_test_positive=int((g(f"gain_{fam}_test") > 0).sum()),
            replication_reject=int((g(f"rep_{fam}_p") < ALPHA).sum()),
            replication_reject_among_rej_S=int((g(f"rep_{fam}_p") < ALPHA)[rejS].sum()))
    return s


ESTIMANDS = {
    "unit": "one configuration = one capture file (head pair x prompt seed); counts are numbers "
            "of configurations with p < 0.05, medians are over configurations within a group",
    "wald_p_S / wald_p_J": "p of the joint Wald test H0: c = 0 (3 cross coefficients ab, a^2b, ab^2 "
                           "of the cubic working model, FWL fit, iid-error covariance), chi2_3; "
                           "single-site rows Zs,Ys (448) / support-matched joint rows Zjm,Yjm (256); "
                           "chart = centring/scaling on all S+J rows, as in identifiability.analyse",
    "cons_p": "p of H0: c_S = c_J, statistic (c_S-c_J)'(V_S+V_J)^{-1}(c_S-c_J) ~ chi2_3, same chart",
    "lam_max_VSinv_VJ": "largest eigenvalue of V_S^{-1} V_J; << 1 means the consistency test is "
                        "driven almost entirely by V_S",
    "power_J_at_cS": "power of the 5% joint Wald test if the true c were c_S: noncentral chi2_3 "
                     "with ncp = c_S' V_J^{-1} c_S",
    "VJ_inflation_for_50pct_power": "ncp_J_at_cS / ncp giving 50% power: the factor by which V_J "
                                    "would have to be understated for J to miss c = c_S half the "
                                    "time",
    "resid_var_armA_over_armB": "variance of full-cubic single-site residuals on arm A (source "
                                "head patched, where the cross block is identified) over arm B "
                                "(target head patched, a pinned); fit() pools one s2 over both",
    "lof_{S,J}_p":"Chow predictive-failure F test of the cubic working model, trained on donor "
                   "half 0 (chart + 10 coefficients fit there only), tested on donor half 1; "
                   "S: 224/224 rows by donor parity; J: 64/64 rows, both donors in the same half",
    "lof_{S,J}_ratio": "mean squared held-out prediction error / training residual variance "
                       "(RSS/(n-p)); ratio_null = its expectation under the working model, "
                       "1 + mean held-out leverage",
    "lof_*_reverse / _rowsplit": "same test with halves swapped / with a random half of rows "
                                 "(rows of one donor in both halves)",
    "gain_{S,J}_test": "1 - MSE_test(full cubic)/MSE_test(additive cubic), both trained on donor "
                       "half 0: held-out value of the cross block; gain_train is its in-sample "
                       "analogue 1 - RSS_full/RSS_add on the training half",
    "rep_{S,J}_p": "split-half replication: c fit separately on the two donor halves, "
                   "(c1-c2)'(V1+V2)^{-1}(c1-c2) ~ chi2_3",
    "wald_p_{S,J}_add5": "joint Wald p for the same 3 cross coefficients with the additive block "
                         "enlarged to degree 5 (adds a^4,a^5,b^4,b^5)",
    "wald_p_{S,J}_deg5": "joint Wald p, chi2_10, that ALL 10 cross coefficients a^i b^j (i,j>=1, "
                         "i+j<=5) are zero, degree-5 additive block: non-additivity of higher "
                         "order than the cubic working model can express",
    "dependence": "configurations are not independent units: for a given (target head, seed) the "
                  "prompt, donors, and all arm-B single-site rows are bit-identical across the 4 "
                  "source heads, and in the control group the joint readouts are ~identical too",
    "info_min_C": "smallest eigenvalue of the cross-block partial information on the clamped "
                  "single-site rows Zc (why the clamped family is not analysed)",
}

if __name__ == "__main__":
    # ---- 1. reproduction gate -------------------------------------------------------------------
    repro, repro_rows = {}, {}
    for grp, (pat, jfile, eS, eJ, en) in GROUPS.items():
        rr, nref = reproduce(grp)
        repro_rows[grp] = {x["file"]: x for x in rr}
        kS = sum(r["wald_p_S"] < ALPHA for r in rr)
        kJ = sum(r["wald_p_J"] < ALPHA for r in rr)
        dp = max(max(abs(r["wald_p_S"] - r["ref_p_S"]), abs(r["wald_p_J"] - r["ref_p_J"])) for r in rr)
        repro[grp] = dict(n_files=len(rr), n_shipped_rows=nref, rej_S=kS, rej_J=kJ,
                          expected=(eS, eJ, en), max_abs_p_diff_vs_shipped=dp,
                          max_match_rel_diff=max(r["match_rel_diff"] for r in rr))
        print(f"[REPRO] {grp:<8} files={len(rr)} shipped rows={nref}  single-site rej {kS}/{len(rr)}"
              f" (paper {eS}/{en})  joint rej {kJ}/{len(rr)} (paper {eJ}/{en})  "
              f"max|dp| vs shipped = {dp:.2e}")
        if (kS, kJ, len(rr), nref) != (eS, eJ, en, en):
            json.dump(dict(status="NOT_REPRODUCED", reproduction=repro),
                      open(os.path.join(HERE, "detection_consistency.json"), "w"), indent=1)
            raise SystemExit(f"[REPRO] {grp}: counts not reproduced -- stopping")
        bad_nm = [r["file"] for r in rr if r["ref_nm"] == [1, 0]]
        if bad_nm:
            print(f"[REPRO] note: {len(bad_nm)} shipped {jfile} rows record nm=[1,0]; the capture "
                  f"files are _100_ i.e. head (10,0)")
            repro[grp]["shipped_rows_with_nm_1_0"] = len(bad_nm)

    # ---- 2-4. per configuration -----------------------------------------------------------------
    rows, summ = [], {}
    hdr = (f"{'config':<22}{'p_S':>9}{'p_J':>9}{'p_cons':>9}{'powJ@cS':>9}{'lamVJ/VS':>10}"
           f"{'LOF_S p':>9}{'ratio':>7}{'LOF_J p':>9}{'ratio':>7}{'rep_S p':>9}{'p_S add5':>10}"
           f"{'p_S deg5':>10}{'p_J deg5':>10}")
    for grp, (pat, *_rest) in GROUPS.items():
        print(f"\n[{grp.upper()}]\n{hdr}\n" + "-" * len(hdr))
        R = [analyse_config(grp, f) for f in sorted(glob.glob(os.path.join(HERE, pat)))]
        for r in R:
            cfg = f"{tuple(r['si'])}->{tuple(r['nm'])} s{r['seed']}"
            print(f"{cfg:<22}{r['wald_p_S']:>9.2g}{r['wald_p_J']:>9.2g}{r['cons_p']:>9.2g}"
                  f"{r['power_J_at_cS']:>9.3f}{r['lam_max_VSinv_VJ']:>10.1e}"
                  f"{r['lof_S_p']:>9.2g}{r['lof_S_ratio']:>7.2f}{r['lof_J_p']:>9.2g}"
                  f"{r['lof_J_ratio']:>7.2f}{r['rep_S_p']:>9.2g}{r['wald_p_S_add5']:>10.2g}"
                  f"{r['wald_p_S_deg5']:>10.2g}{r['wald_p_J_deg5']:>10.2g}")
        # the paper's analyse() p and fit()+wald() p are the same computation: confirm
        rp = repro_rows[grp]
        dmax = max(max(abs(r["wald_p_S"] - rp[r["file"]]["wald_p_S"]),
                       abs(r["wald_p_J"] - rp[r["file"]]["wald_p_J"])) for r in R)
        summ[grp] = summarise(R)
        summ[grp]["max_abs_p_diff_fit_vs_analyse"] = float(dmax)
        summ[grp]["max_fit_basis_selfcheck"] = float(max(r["fit_basis_selfcheck"] for r in R))
        summ[grp]["max_info_min_C"] = float(max(r["info_min_C"] for r in R))
        rows += R

    # ---- report -----------------------------------------------------------------------------------
    for grp, s in summ.items():
        n, kS = s["n"], s["rej_S"]
        print(f"\n[SUMMARY {grp}]  n = {n}")
        print(f"  single-site Wald rejects c=0          : {kS}/{n}    joint: {s['rej_J']}/{n}")
        print(f"  c_S inconsistent with c_J (p<.05)      : {s['inconsistent_all']}/{n} overall; "
              f"{s['inconsistent_among_rej_S']}/{kS} of single-site rejections "
              f"({s['inconsistent_among_rej_S_bonferroni']}/{kS} at Bonferroni .05/{max(kS,1)}); "
              f"{s['inconsistent_among_nonrej_S']}/{n-kS} of non-rejections")
        print(f"  median lam_max(V_S^-1 V_J)             : {s['median_lam_max_VSinv_VJ']:.2e}")
        if kS:
            print(f"  power of joint test if c = c_S         : median {s['median_power_J_at_cS_rej_S']:.4f}, "
                  f"min {s['min_power_J_at_cS_rej_S']:.4f} over the {kS} single-site rejections; "
                  f"V_J would have to be understated >= "
                  f"{s['min_VJ_inflation_for_50pct_power_rej_S']:.1f}x for that power to fall to 50%")
        print(f"  single-site p_S: median {s['median_p_S']:.3g}, min {s['min_p_S']:.3g}; "
              f"residual variance arm A / arm B (full cubic S fit), median "
              f"{s['median_resid_var_armA_over_armB']:.2e}")
        for fam in ("S", "J"):
            L = s[f"lof_{fam}"]
            print(f"  held-out LOF, family {fam} (donor split)  : rejects {L['reject']}/{n} "
                  f"(reverse {L['reject_reverse']}/{n}; row split {L['reject_rowsplit']}/{n}); "
                  f"median MSE_test/s2_train {L['median_ratio']:.2f} vs null {L['median_ratio_null']:.2f} "
                  f"(reverse {L['median_ratio_reverse']:.2f}, row split {L['median_ratio_rowsplit']:.2f})"
                  + (f"; among single-site rejections {L['reject_among_rej_S']}/{kS}" if kS else ""))
            print(f"      cross block held-out gain         : median test {L['median_gain_test']:+.4f} "
                  f"vs train {L['median_gain_train']:+.4f}; test gain > 0 in {L['gain_test_positive']}/{n}")
            print(f"      split-half replication of c       : rejects {L['replication_reject']}/{n}"
                  + (f" ({L['replication_reject_among_rej_S']}/{kS} of single-site rejections)" if kS else ""))
        print(f"  additive block to degree 5            : single-site rejects {s['rej_S_add5']}/{n} "
              f"({s['rej_S_add5_among_rej_S']}/{kS} of the original rejections survive); "
              f"joint rejects {s['rej_J_add5']}/{n}")
        print(f"  degree-5 additive + degree-5 cross    : single-site rejects {s['rej_S_deg5']}/{n} "
              f"({s['rej_S_deg5_among_rej_S']}/{kS} of the original rejections); "
              f"joint rejects {s['rej_J_deg5']}/{n}  [chi2_{len(CROSS5)}]")
        print(f"  checks: max|p(fit+wald) - p(analyse)| = {s['max_abs_p_diff_fit_vs_analyse']:.1e}, "
              f"fit_basis(ADD) vs fit rel diff = {s['max_fit_basis_selfcheck']:.1e}, "
              f"max clamped info = {s['max_info_min_C']:.1e}")

    json.dump(dict(status="OK", alpha=ALPHA, reproduction=repro, groups=summ,
                   estimands=ESTIMANDS, rows=rows),
              open(os.path.join(HERE, "detection_consistency.json"), "w"), indent=1, default=float)
    print("\nsaved -> e0/detection_consistency.json")
