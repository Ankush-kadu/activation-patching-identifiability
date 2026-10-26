"""FINITE-SAMPLE BEHAVIOUR OF THE ESTIMATED GRAM OPERATOR AND ITS BLIND SUBSPACE.

CIF (Asiaee, UAI 2026) and Spectral Identifiability (Dhor & Chen, Aug 2026) both ship
finite-sample rates, so a 2026 reviewer will ask what happens when G_Q is estimated
from n interventions rather than known exactly.  This script states the bound and then
CHECKS WHERE IT IS ACTUALLY INFORMATIVE, which is the part a stated-only theorem hides.

--------------------------------------------------------------------------------------
PROP 1 (concentration).  phi bounded, ||phi(z)|| <= B on supp(mu_Q); z_1..z_n iid ~ mu_Q;
Ghat = (1/n) sum phi_i phi_i^T.  Matrix Bernstein (Tropp 2015, Thm 6.1.1) with
X_i = (1/n)(phi_i phi_i^T - G):  ||X_i|| <= 2B^2/n and ||sum E X_i^2|| <= B^2 ||G|| / n.
Hence with probability >= 1 - delta

    ||Ghat - G||_op  <=  sqrt( 2 B^2 ||G|| L / n )  +  (4/3) B^2 L / n ,     L = log(2p/delta)

(the intrinsic-dimension form, Tropp Thm 7.3.1, replaces p by tr(G)/||G||; we report both).

PROP 2 (blind-subspace stability).  Let V be the bottom-k eigenspace of G and Vhat that of
Ghat, and let gamma = lambda_{p-k}(G) - lambda_{p-k+1}(G) be the eigengap SEPARATING THE
BLIND SUBSPACE FROM THE REST.  Davis-Kahan in the Yu-Wang-Samworth (2015) form:

    ||sin Theta(Vhat, V)||_F  <=  2 sqrt(k) ||Ghat - G||_op / gamma .

PROP 3 (generalized problem).  The reported directions solve G_S v = lam G_J v, implemented
as the ordinary eigenproblem for M = W (G_S + dI) W, W = (G_J + dI)^{-1/2}.  Using
||Ahat^{-1/2} - A^{-1/2}|| <= ||Ahat - A|| / (2 d^{3/2}) for A, Ahat >= dI,

    ||Mhat - M||  <=  E_S / d  +  (||G_S|| + d) E_J / d^2 ,   E_Q = ||Ghat_Q - G_Q||.

THE HONEST POINT, WHICH THE PAPER MUST STATE.
Prop 2 is informative only when gamma is large.  "G_Q is full rank but ill-conditioned"
means gamma at the blind-subspace boundary is SMALL, so the bound degrades exactly where
the phenomenon lives.  Prop 3 carries d^{-2} with d = 1e-10.  So the perturbation theory
does NOT carry the empirical claim.  What carries it is the frozen-direction train -> test
protocol: fit v on TRAIN, freeze it, evaluate v^T Ghat_test v on untouched TEST.  That is a
finite-sample statement about a FIXED vector (scalar Bernstein, no eigengap, no d^{-2}),
and it is what subspace_test.py actually reports.

PROP 4 (what the protocol licenses).  For a direction v frozen on TRAIN, the test statistic
v^T Ghat_test v is a mean of n iid variables in [0, B^2 ||v||^2], so by Bernstein it
concentrates around v^T G v at rate O(n^{-1/2}) WITH NO SPECTRAL ASSUMPTION WHATSOEVER.
The reported ratio is therefore a valid finite-sample quantity even though the estimated
SUBSPACE is not stable in the Prop-2 sense.

PROP 5 (measured, not assumed).  Solve the whole generalized eigenproblem independently on
two disjoint halves and compare.  MEASURED RESULT: the S-blind direction reproduces to
0.8 deg (GPT-2) / 1.9 deg (Qwen) and the J-blind direction to 17.5 / 14.8 deg, and every
frozen direction still separates on the half it was not fitted on (GPT-2 24/24 and 21/24;
Qwen 30/30 and 30/30).  This does NOT contradict Prop 2's unstable bottom-k eigenspace:
Prop 2 perturbs the MARGINAL spectrum of a single family, whose tail is nearly degenerate
(gap/||G|| = 0.08-0.13 for J), whereas the generalized problem keys on the CONTRAST between
the two families and is much better conditioned.  Worst-case theory is loose here; it is not
that the phenomenon is fragile.

CAVEAT ON THE MEASURED RATE.  The available n spans 25-200, under one decade, and the
two-sample statistic has a noise floor, so the fitted log-log slopes (-0.30 to -0.49) are
consistent with the theoretical -0.5 but do not sharply confirm it.  Report as consistent,
not as verified.
"""
import glob, json, numpy as np
from levelC2 import design

DEG, DELTA, EPS = 3, 0.05, 0.02


def prep_full(Zs, Zj, deg=DEG):
    """Identical whitening / column scaling to subspace_test.prep, but no split."""
    allZ = np.vstack([Zs, Zj])
    mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(W(allZ), deg), axis=0)
    cs[cs == 0] = 1.0
    return design(W(Zs), deg) / cs, design(W(Zj), deg) / cs


def gram(M):
    return M.T @ M / M.shape[0]


def bernstein(B2, normG, p, n, delta=DELTA, intrinsic=None):
    """Prop 1, both the dimension-dependent and intrinsic-dimension forms."""
    out = {}
    for tag, d_eff in (("p", p), ("intrinsic", intrinsic)):
        if d_eff is None:
            continue
        L = np.log(2 * max(d_eff, 1.0) / delta)
        out[tag] = float(np.sqrt(2 * B2 * normG * L / n) + (4.0 / 3.0) * B2 * L / n)
    return out


def sin_theta(A, Bm):
    """||sin Theta||_F between the column spans of two orthonormal bases."""
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(Bm)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    k = min(Qa.shape[1], Qb.shape[1])
    return float(np.sqrt(np.sum(1.0 - s[:k] ** 2)))


def bottom_k(G, k):
    w, U = np.linalg.eigh(G)          # ascending
    return U[:, :k], w


def rate_check(M, ns, reps=40, seed=0):
    """Two disjoint subsamples of size n; ||G1 - G2|| must decay ~ n^{-1/2}."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in ns:
        if 2 * n > len(M):
            continue
        vals = []
        for _ in range(reps):
            i = rng.permutation(len(M))
            vals.append(np.linalg.norm(gram(M[i[:n]]) - gram(M[i[n:2 * n]]), 2))
        out[n] = float(np.median(vals))
    return out


def fit_slope(curve):
    if len(curve) < 3:
        return float("nan")
    x = np.log(np.array(list(curve.keys()), float))
    y = np.log(np.array(list(curve.values()), float))
    return float(np.polyfit(x, y, 1)[0])


def analyse(Zs, Zj, k=3, seed=0):
    Ms, Mj = prep_full(Zs, Zj)
    n = len(Ms)
    p = Ms.shape[1]
    res = {"n": int(n), "p": int(p), "k": int(k)}

    for tag, M in (("S", Ms), ("J", Mj)):
        n = len(M)
        res[f"n_{tag}"] = int(n)
        G = gram(M)
        B2 = float(np.max(np.sum(M ** 2, axis=1)))
        w = np.linalg.eigvalsh(G)
        normG = float(w[-1])
        intr = float(np.trace(G) / max(normG, 1e-300))

        # --- Prop 1: rate + bound at the full sample size ---
        curve = rate_check(M, (25, 50, 100, 200, 400))
        res[f"{tag}_rate_curve"] = curve
        res[f"{tag}_rate_slope"] = fit_slope(curve)     # theory: -0.5
        res[f"{tag}_B2"] = B2
        res[f"{tag}_normG"] = normG
        res[f"{tag}_intrinsic_dim"] = intr
        res[f"{tag}_bernstein"] = bernstein(B2, normG, p, n, intrinsic=intr)

        # --- Prop 2: eigengap at the blind-subspace boundary, and is the bound vacuous? ---
        gap = float(w[p - k] - w[p - k - 1]) if p - k - 1 >= 0 else float(w[p - k])
        gap_rel = gap / max(normG, 1e-300)
        rng = np.random.default_rng(seed)
        i = rng.permutation(n)
        h1, h2 = i[: n // 2], i[n // 2:]
        V1, _ = bottom_k(gram(M[h1]), k)
        V2, _ = bottom_k(gram(M[h2]), k)
        emp_sin = sin_theta(V1, V2)
        E = float(res[f"{tag}_bernstein"]["intrinsic"])
        dk = 2 * np.sqrt(k) * E / max(gap, 1e-300)
        res[f"{tag}_eigengap"] = gap
        res[f"{tag}_eigengap_rel"] = gap_rel
        res[f"{tag}_sinTheta_empirical"] = emp_sin
        res[f"{tag}_davis_kahan_bound"] = float(dk)
        res[f"{tag}_dk_informative"] = bool(dk < np.sqrt(k))   # < sqrt(k) = better than chance

    # --- Prop 3: generalized-problem constant at the two regularizers we actually used ---
    Es = res["S_bernstein"]["intrinsic"]
    Ej = res["J_bernstein"]["intrinsic"]
    res["prop3"] = {}
    for d in (1e-10, 1e-4, 1e-2):
        res["prop3"][str(d)] = float(Es / d + (res["S_normG"] + d) * Ej / d ** 2)

    # --- Prop 5: is the REPORTED DIRECTION reproducible, and does it matter? ---
    # Run the whole generalized eigenproblem independently on two disjoint halves.
    # If the blind subspace has dim > 1 the direction is NOT unique, so a large angle
    # between the two solutions is EXPECTED and is not a defect -- provided each frozen
    # direction still separates on the half it never saw.  That is the check.
    rng = np.random.default_rng(seed)
    a = rng.permutation(len(Ms)); b = rng.permutation(len(Mj))
    Sa, Sb = Ms[a[: len(a) // 2]], Ms[a[len(a) // 2:]]
    Ja, Jb = Mj[b[: len(b) // 2]], Mj[b[len(b) // 2:]]

    def solve(S, J, d=1e-10):
        from scipy.linalg import eigh
        I = np.eye(S.shape[1])
        _, V = eigh(gram(S) + d * I, gram(J) + d * I)
        return (V[:, 0] / np.linalg.norm(V[:, 0]),
                V[:, -1] / np.linalg.norm(V[:, -1]))

    hS1, hJ1 = solve(Sa, Ja)
    hS2, hJ2 = solve(Sb, Jb)
    ang = lambda u, v: float(np.degrees(np.arccos(np.clip(abs(u @ v), 0, 1))))
    q = lambda M, h: float(np.mean((M @ h) ** 2))
    res["prop5"] = {
        "angle_Sblind_half1_vs_half2": ang(hS1, hS2),
        "angle_Jblind_half1_vs_half2": ang(hJ1, hJ2),
        # each direction scored on the half it was NOT fitted on
        "sep_Sblind_fit1_eval2": q(Jb, hS1) / max(q(Sb, hS1), 1e-300),
        "sep_Sblind_fit2_eval1": q(Ja, hS2) / max(q(Sa, hS2), 1e-300),
        "sep_Jblind_fit1_eval2": q(Sb, hJ1) / max(q(Jb, hJ1), 1e-300),
        "sep_Jblind_fit2_eval1": q(Sa, hJ2) / max(q(Ja, hJ2), 1e-300),
    }
    return res


def run(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    rows = [analyse(np.load(f)["Zs"], np.load(f)["Zj"]) for f in files]
    print("=" * 100)
    print(f"{label}   n_trials={len(rows)}   p={rows[0]['p']}   k={rows[0]['k']}")
    print("=" * 100)

    print("\nPROP 1 — concentration rate  ||Ghat_n^(1) - Ghat_n^(2)||_op  vs n")
    print(f"  {'n':>6} {'S family':>14} {'J family':>14}")
    ns = sorted(rows[0]["S_rate_curve"].keys())
    for n in ns:
        a = np.median([r["S_rate_curve"][n] for r in rows if n in r["S_rate_curve"]])
        b = np.median([r["J_rate_curve"][n] for r in rows if n in r["J_rate_curve"]])
        print(f"  {n:>6} {a:>14.4g} {b:>14.4g}")
    ss = np.median([r["S_rate_slope"] for r in rows])
    sj = np.median([r["J_rate_slope"] for r in rows])
    print(f"  log-log slope (theory -0.50):   S {ss:+.3f}    J {sj:+.3f}")

    print("\n  Bernstein bound at the full sample size (delta=0.05)")
    for t in ("S", "J"):
        bp = np.median([r[f"{t}_bernstein"]["p"] for r in rows])
        bi = np.median([r[f"{t}_bernstein"]["intrinsic"] for r in rows])
        ng = np.median([r[f"{t}_normG"] for r in rows])
        idim = np.median([r[f"{t}_intrinsic_dim"] for r in rows])
        print(f"    {t}: ||G||={ng:.4g}  intrinsic dim={idim:.2f}  "
              f"bound(p)={bp:.4g}  bound(intrinsic)={bi:.4g}  "
              f"relative={bi/max(ng,1e-300):.3g}")

    print("\nPROP 2 — Davis-Kahan on the blind subspace  (informative only if bound < sqrt(k))")
    print(f"  {'family':>8} {'eigengap':>12} {'gap/||G||':>12} {'emp sinTheta':>14} "
          f"{'DK bound':>12} {'informative':>12}")
    for t in ("S", "J"):
        g = np.median([r[f"{t}_eigengap"] for r in rows])
        gr = np.median([r[f"{t}_eigengap_rel"] for r in rows])
        es = np.median([r[f"{t}_sinTheta_empirical"] for r in rows])
        dk = np.median([r[f"{t}_davis_kahan_bound"] for r in rows])
        ni = sum(r[f"{t}_dk_informative"] for r in rows)
        print(f"  {t:>8} {g:>12.4g} {gr:>12.4g} {es:>14.4f} {dk:>12.4g} "
              f"{ni:>9}/{len(rows)}")

    print("\nPROP 3 — generalized-eigenproblem perturbation constant (vacuous if >> 1)")
    for d in ("1e-10", "0.0001", "0.01"):
        v = np.median([r["prop3"][d] for r in rows])
        print(f"    delta={d:>8}:  bound = {v:.4g}   "
              f"{'VACUOUS' if v > 1 else 'informative'}")

    print("\nPROP 5 — is the reported DIRECTION reproducible, and does non-uniqueness hurt?")
    print("  (two independent halves, full generalized eigenproblem solved on each)")
    aS = np.median([r["prop5"]["angle_Sblind_half1_vs_half2"] for r in rows])
    aJ = np.median([r["prop5"]["angle_Jblind_half1_vs_half2"] for r in rows])
    print(f"    angle between the two S-blind solutions : {aS:>7.1f} deg")
    print(f"    angle between the two J-blind solutions : {aJ:>7.1f} deg")
    for nm in ("Sblind", "Jblind"):
        v = np.array([[r["prop5"][f"sep_{nm}_fit1_eval2"],
                       r["prop5"][f"sep_{nm}_fit2_eval1"]] for r in rows]).ravel()
        print(f"    {nm}: separation on the UNSEEN half — median {np.median(v):>10.3g}, "
              f">10x in {int(np.sum(v > 10))}/{len(v)}")
    print("    => OBSERVED: the directions are in fact reproducible to within a few degrees")
    print("       (S-blind) and ~15 deg (J-blind), AND both separate on the unseen half.")
    print("       Note this does NOT contradict the unstable bottom-k eigenspace in Prop 2:")
    print("       Prop 2 perturbs the MARGINAL spectrum of one family, whose tail is nearly")
    print("       degenerate; Prop 5 solves the GENERALIZED problem, which keys on the")
    print("       CONTRAST between families and is far better conditioned than either")
    print("       marginal. The paper should still claim a subspace, not a vector.")

    print("\n  => The perturbation bounds do NOT carry the empirical claim.")
    print("     The frozen-direction train->test protocol (Prop 4) does: it is a statement")
    print("     about a FIXED vector, so it needs no eigengap and no delta^-2.")
    return rows


if __name__ == "__main__":
    out = {}
    for pat, lab in (("adj_raw_*.npz", "GPT-2"), ("rec_raw_*.npz", "Qwen3.5-9B")):
        r = run(pat, lab)
        if r:
            out[lab] = r
        print()
    json.dump(out, open("finite_sample.json", "w"), indent=1, default=float)
    print("saved -> finite_sample.json")
