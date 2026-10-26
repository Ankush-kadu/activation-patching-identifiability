"""TRACK B — RESOLVING POWER OF INTERPRETABILITY CLAIM LANGUAGES.

Pre-registration in RESEARCH_LOG.md, committed before this file was written.

FORMULATION.  Logit lens / tuned lens / SAE are NOT mechanism classes -- saying so is a category
mistake (a logit lens is a readout, an SAE is a learned dictionary).  Each is a REPRESENTATION
rho that INDUCES a class once a downstream family G is declared:

    rho : z -> rho(z)     F_rho = { g o rho : g in G }     object:  (Q, rho, G) -> N(Q, F_rho)

MATCHED CHARTS.  Every rho is reduced to a chart of the SAME small dimension k (k/2 per site, so
the two-site structure the S-vs-J analysis rests on is preserved).  C_rho is fitted on the
TRAIN/CALIBRATION split of the SINGLE family ONLY, without Y, then FROZEN before any held-out
evaluation -- PCA on held-out states would invite a transductive-leakage objection even though Y
is never touched.

EXACT REDUCTION FOR THE LOGIT LENS.  rho_LL(A) = LN(A) W_U^T is vocabulary-dimensional (50257),
but rank(W_U) <= d_model.  With the thin SVD W_U = U_w S_w V_w^T and U_w orthonormal,
    rho_LL(A) = LN(A) V_w S_w U_w^T ,
and U_w is an isometry, so ALL Euclidean geometry -- hence PCA and every quadratic form we
compute -- is preserved exactly by working with LN(A) V_w S_w in R^{d_model}.  This is an exact
reduction, not an approximation.

TIERS (the controls are the point):
    B0 invariance          rho2 = A rho1, A invertible => spectrum UNCHANGED (positive control)
    B1 matched compression random projection / PCA at matched k -> effect of losing dims alone
    B2 real representations raw / logit lens / tuned lens (/ SAE only if a fixed released one
                           exists; otherwise OMITTED by protocol, never trained ad hoc)
    B2 counts ONLY if it separates from B0/B1.

COMPARISON RULE.  Cross-representation eigenvalue MAGNITUDES are never compared directly:
different rho induce different F_rho hence different geometries, and Prop reparam says
magnitudes are class-relative.  We compare WITHIN-class S-vs-J profiles, null-calibrated
blindness, and whether a representation exceeds its matched compression controls.
"""
import argparse, glob, json, os
import numpy as np
from scipy.linalg import eigh

LAM = 1e-6


# ------------------------------------------------------------ design matrix ---
def design(Z, deg):
    n, k = Z.shape
    cols, cur = [np.ones(n)], [()]
    for _ in range(deg):
        nxt = []
        for t in cur:
            start = t[-1] if t else 0
            for j in range(start, k):
                nt = t + (j,)
                nxt.append(nt)
                cols.append(np.prod(Z[:, list(nt)], axis=1))
        cur = nxt
    return np.stack(cols, 1)


def gram(M):
    return M.T @ M / max(M.shape[0], 1)


# ------------------------------------------------------------------ charts ---
def fit_chart(Xtr, k, kind="pca", seed=0):
    """Fitted on TRAIN only, never sees Y. Returns (k, d)."""
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    kk = min(k, d)
    if kind == "pca":
        mu = Xtr.mean(0)
        V = np.linalg.svd(Xtr - mu, full_matrices=False)[2]
        return V[:kk], mu
    if kind == "random":
        C = rng.normal(size=(kk, d))
        return np.linalg.qr(C.T)[0].T, Xtr.mean(0)
    raise ValueError(kind)


def apply_chart(chart, X):
    C, mu = chart
    return (X - mu) @ C.T


# ------------------------------------------------------ representations rho ---
def ln(A):
    mu = A.mean(-1, keepdims=True)
    sd = A.std(-1, keepdims=True) + 1e-6
    return (A - mu) / sd


def make_rhos(ctx, which):
    R = {}
    if "raw" in which:
        R["raw"] = lambda A: A
    if "logitlens" in which:
        VS = ctx["VS"]                       # V_w @ S_w, exact isometric reduction
        R["logitlens"] = lambda A: ln(A) @ VS
    if "tunedlens" in which:
        T, bb = ctx["TL"]
        R["tunedlens"] = lambda A: A @ T + bb
    if "sae" in which and ctx.get("SAE") is not None:
        D, thr = ctx["SAE"]
        R["sae"] = lambda A: np.maximum(A @ D.T - thr, 0.0)
    return R


# ----------------------------------------------------------------- scoring ---
def resolving(Zs_tr, Zs_te, Zj_tr, Zj_te, deg):
    Ms_tr, Ms_te = design(Zs_tr, deg), design(Zs_te, deg)
    Mj_tr, Mj_te = design(Zj_tr, deg), design(Zj_te, deg)
    cs = np.linalg.norm(np.vstack([Ms_tr, Mj_tr]), axis=0); cs[cs == 0] = 1.0
    Ms_tr, Ms_te, Mj_tr, Mj_te = Ms_tr / cs, Ms_te / cs, Mj_tr / cs, Mj_te / cs
    Gs, Gj = gram(Ms_tr), gram(Mj_tr)
    Gref = 0.5 * (Gs + Gj)
    try:
        lam, V = eigh(Gs + LAM * Gref, Gj + LAM * Gref)
    except np.linalg.LinAlgError:
        return None
    def share(h):
        s = float(np.mean((Ms_te @ h) ** 2)); j = float(np.mean((Mj_te @ h) ** 2))
        t = s + j
        return (s / t, j / t) if t > 0 else (np.nan, np.nan)
    hS = V[:, 0] / np.linalg.norm(V[:, 0])
    hJ = V[:, -1] / np.linalg.norm(V[:, -1])
    (vSs, vSj), (vJs, vJj) = share(hS), share(hJ)
    return dict(vS_on_Sblind=vSs, vJ_on_Sblind=vSj, vS_on_Jblind=vJs, vJ_on_Jblind=vJj,
                p=int(Gs.shape[0]))


def null_same_family(Zs_tr, Zs_te, deg, reps=5, seed=0):
    rng = np.random.default_rng(seed)
    allS = np.vstack([Zs_tr, Zs_te])
    out = []
    for _ in range(reps):
        i = rng.permutation(len(allS))
        a, b = allS[i[:len(i) // 2]], allS[i[len(i) // 2:]]
        na, nb = len(a) // 2, len(b) // 2
        r = resolving(a[:na], a[na:], b[:nb], b[nb:], deg)
        if r:
            out.append(r)
    return out


def chart_pair(rho, As, Bs, Aj, Bj, k, kind, seed):
    """Fit per-site charts on the TRAIN split of the SINGLE family only, then freeze.

    BUG FIXED 2026-08-19.  The first version split X[:n]/X[n:], but the single-site family is
    stored as [patch-A arm ; patch-B arm] -- site A has EXACTLY ZERO variance in the second half
    (it is pinned there).  That trained on one arm and tested on the other, and destroyed the
    phenomenon for every representation including raw (v_S = 0.63 instead of the frozen 7.4e-06).
    The frozen `subspace_test.prep` uses a random permutation across both arms and whitens Z
    before building the design matrix; both are restored here, with whitening statistics taken
    from TRAIN ONLY (stricter than the frozen protocol, which pools).
    """
    rng = np.random.default_rng(seed)
    RAs, RBs, RAj, RBj = rho(As), rho(Bs), rho(Aj), rho(Bj)
    i = rng.permutation(len(RAs)); j = rng.permutation(len(RAj))
    itr, ite = i[:len(i) // 2], i[len(i) // 2:]
    jtr, jte = j[:len(j) // 2], j[len(j) // 2:]

    kh = max(k // 2, 1)
    cA = fit_chart(RAs[itr], kh, kind, seed)          # TRAIN only, no Y
    cB = fit_chart(RBs[itr], kh, kind, seed + 1)
    Z = lambda X, Y: np.hstack([apply_chart(cA, X), apply_chart(cB, Y)])
    Zs_tr, Zs_te = Z(RAs[itr], RBs[itr]), Z(RAs[ite], RBs[ite])
    Zj_tr, Zj_te = Z(RAj[jtr], RBj[jtr]), Z(RAj[jte], RBj[jte])

    # whitening statistics from TRAIN ONLY (both families), then frozen
    ref = np.vstack([Zs_tr, Zj_tr])
    mu, sd = ref.mean(0), ref.std(0) + 1e-12
    W = lambda Z_: (Z_ - mu) / sd
    return W(Zs_tr), W(Zs_te), W(Zj_tr), W(Zj_te)


def run_file(npz, rhos, k, degs, seed=0, kind="pca"):
    d = np.load(npz)
    As, Bs, Aj, Bj = d["A_single"], d["B_single"], d["A_joint"], d["B_joint"]
    out = {}
    for name, rho in rhos.items():
        Zs_tr, Zs_te, Zj_tr, Zj_te = chart_pair(rho, As, Bs, Aj, Bj, k, kind, seed)
        per = {}
        for dg in degs:
            r = resolving(Zs_tr, Zs_te, Zj_tr, Zj_te, dg)
            if r is None:
                continue
            nl = null_same_family(Zs_tr, Zs_te, dg, seed=seed)
            if nl:
                r["null_vS_p05"] = float(np.percentile([x["vS_on_Sblind"] for x in nl], 5))
                r["null_vJ_p05"] = float(np.percentile([x["vJ_on_Jblind"] for x in nl], 5))
            per[str(dg)] = r
        out[name] = per
    return out


def build_ctx(WU, Atr):
    """Anything LEARNED is fitted on the TRAIN split only, then frozen."""
    Uw, Sw, Vwt = np.linalg.svd(WU, full_matrices=False)
    ctx = {"VS": (Vwt.T * Sw)}                       # exact isometric reduction of the logit lens
    tgt = ln(Atr) @ ctx["VS"]                        # tuned lens target = logit-lens readout
    X = np.hstack([Atr, np.ones((len(Atr), 1))])
    W, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    ctx["TL"] = (W[:-1], W[-1])
    ctx["SAE"] = None                                # no fixed released SAE available -> OMITTED
    return ctx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="actsg")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--degs", default="1,2,3")
    ap.add_argument("--out", default="mech_classes.json")
    A = ap.parse_args()
    degs = [int(x) for x in A.degs.split(",")]
    files = sorted(glob.glob(f"{A.tag}_*_*_*.npz"))
    WU = np.load(f"{A.tag}_WU.npy")

    # tuned lens calibrated on the TRAIN half of the FIRST file only, then frozen for all
    d0 = np.load(files[0])
    ctx = build_ctx(WU, d0["A_single"][: len(d0["A_single"]) // 2])
    rhos = make_rhos(ctx, ("raw", "logitlens", "tunedlens", "sae"))
    print(f"representations: {list(rhos)}"
          f"{'   [SAE omitted: no fixed released SAE available, per protocol]' if 'sae' not in rhos else ''}")

    RES = {"B2": {}, "B1": {}, "B0": {}}
    for f in files:
        RES["B2"][os.path.basename(f)] = run_file(f, rhos, A.k, degs)
        RES["B1"][os.path.basename(f)] = run_file(
            f, {"randproj": rhos["raw"]}, A.k, degs, kind="random")

    # ---- B0: invertible reparameterization of raw must leave the result unchanged ----
    rng = np.random.default_rng(0)
    d = np.load(files[0])
    p = d["A_single"].shape[1]
    Q = np.linalg.qr(rng.normal(size=(p, p)))[0] * np.logspace(0, 1, p)
    base = run_file(files[0], {"raw": rhos["raw"]}, A.k, degs)["raw"]
    repar = run_file(files[0], {"raw": lambda X: X @ Q.T}, A.k, degs)["raw"]
    RES["B0"] = {"base": base, "reparameterized": repar}

    print("\n" + "=" * 104)
    print(f"TRACK B — resolving power by representation   (k={A.k}, charts frozen on TRAIN, no Y)")
    print("=" * 104)
    print(f"  {'representation':<16}{'deg':>5}{'v_S on S-blind':>17}{'null p05':>11}"
          f"{'v_J on J-blind':>16}{'null p05':>11}   n")
    print("-" * 104)
    summary = {}
    for rep in list(rhos) + ["randproj"]:
        for dg in degs:
            src = RES["B1"] if rep == "randproj" else RES["B2"]
            key = "randproj" if rep == "randproj" else rep
            vals = [v[key][str(dg)] for v in src.values() if key in v and str(dg) in v[key]]
            if not vals:
                continue
            mS = float(np.median([x["vS_on_Sblind"] for x in vals]))
            mJ = float(np.median([x["vJ_on_Jblind"] for x in vals]))
            nS = float(np.median([x.get("null_vS_p05", np.nan) for x in vals]))
            nJ = float(np.median([x.get("null_vJ_p05", np.nan) for x in vals]))
            summary[f"{rep}/d{dg}"] = dict(vS=mS, vJ=mJ, null_vS=nS, null_vJ=nJ, n=len(vals))
            print(f"  {rep:<16}{dg:>5}{mS:>17.3e}{nS:>11.3f}{mJ:>16.3e}{nJ:>11.3f}   {len(vals)}")

    print("\n  B0 invariance check (invertible reparameterization of the SAME class):")
    for dg in degs:
        a1 = base.get(str(dg)); a2 = repar.get(str(dg))
        if a1 and a2:
            rel = abs(a1["vS_on_Sblind"] - a2["vS_on_Sblind"]) / max(a1["vS_on_Sblind"], 1e-30)
            print(f"    deg {dg}: v_S base={a1['vS_on_Sblind']:.4e}  "
                  f"reparam={a2['vS_on_Sblind']:.4e}   rel diff={rel:.2e}")

    json.dump({"summary": summary, "raw": RES}, open(A.out, "w"), indent=1, default=float)
    print(f"\nsaved -> {A.out}")
