"""STATISTICAL CONTROLS + BASIS ROBUSTNESS for the reciprocal blind-subspace result.

The reciprocal separations involve near-zero denominators, so the obvious attack is
"generalized eigendecomposition is exploiting finite-sample covariance noise."
Four controls close that, plus a basis sweep closes "it's a polynomial artifact".

C1  sample-size curve      -> a structural blind direction must NOT decay toward 1 as n grows
C2  bootstrap CIs          -> report intervals, not point estimates
C3  regularized Gram       -> G + lam*I sweep; must survive regularization
C4  RANDOMIZED-FAMILY NULL -> split S into S1,S2 from the SAME family and run the
                              identical procedure. Sep(S1,S2) must be ~1 while
                              Sep(S,J) >> 10. This is the cleanest control.

BASIS: poly (primary) | taylor (recentred: SAME span, conditioning check only)
       | rbf | fourier (genuinely different spans)
"""
import json, glob, numpy as np
from scipy.linalg import eigh

DEG = 3


# ------------------------------------------------------------------ bases ---
def b_poly(Z, deg=DEG, **kw):
    e = [(i, t - i) for t in range(deg + 1) for i in range(t + 1)]
    return np.array([[a ** i * b ** j for (i, j) in e] for a, b in Z], float)

def b_taylor(Z, deg=DEG, c=None, **kw):
    c = np.zeros(2) if c is None else c
    return b_poly(Z - c, deg)

def b_rbf(Z, centers=None, gamma=0.7, **kw):
    if centers is None:
        g = np.linspace(-2, 2, 4)
        centers = np.array([(x, y) for x in g for y in g])
    return np.exp(-gamma * ((Z[:, None, 0] - centers[None, :, 0]) ** 2 +
                            (Z[:, None, 1] - centers[None, :, 1]) ** 2))

def b_fourier(Z, W=None, **kw):
    if W is None:
        rng = np.random.default_rng(7)
        W = rng.normal(size=(2, 5))
    P = Z @ W
    return np.hstack([np.ones((len(Z), 1)), np.sin(P), np.cos(P)])

BASES = dict(poly=b_poly, taylor=b_taylor, rbf=b_rbf, fourier=b_fourier)


# ------------------------------------------------------------------ core ---
def prep(Za, Zb, basis="poly", seed=0):
    """Whiten with combined stats, build design matrices, common column scaling."""
    rng = np.random.default_rng(seed)
    allZ = np.vstack([Za, Zb]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    f = BASES[basis]
    cs = np.linalg.norm(f(W(allZ)), axis=0); cs[cs == 0] = 1.0
    Ma, Mb = f(W(Za)) / cs, f(W(Zb)) / cs
    i = rng.permutation(len(Ma)); j = rng.permutation(len(Mb))
    return (Ma[i[:len(i)//2]], Ma[i[len(i)//2:]],
            Mb[j[:len(j)//2]], Mb[j[len(j)//2:]])


def recip(Za, Zb, basis="poly", seed=0, lam=1e-10):
    """Return (A-blind ratio, B-blind ratio) on HELD-OUT data, directions frozen."""
    Atr, Ate, Btr, Bte = prep(Za, Zb, basis, seed)
    G = lambda M: M.T @ M / M.shape[0]
    Ga, Gb = G(Atr), G(Btr)
    n = Ga.shape[0]; I = np.eye(n)
    try:
        w, V = eigh(Ga + lam * I, Gb + lam * I)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    q = lambda M, h: float(np.mean((M @ h) ** 2))
    hA = V[:, 0] / np.linalg.norm(V[:, 0])
    hB = V[:, -1] / np.linalg.norm(V[:, -1])
    return (q(Bte, hA) / max(q(Ate, hA), 1e-300),
            q(Ate, hB) / max(q(Bte, hB), 1e-300))


def load(pattern):
    return [(f, np.load(f)) for f in sorted(glob.glob(pattern))]


# --------------------------------------------------------------- controls ---
def c4_random_family_null(data, basis="poly", reps=5):
    """Split the SINGLE family in half at random and run the identical procedure."""
    same, cross = [], []
    for _, d in data:
        Zs, Zj = d["Zs"], d["Zj"]
        for r in range(reps):
            rng = np.random.default_rng(100 + r)
            i = rng.permutation(len(Zs))
            a, b = Zs[i[:len(i)//2]], Zs[i[len(i)//2:]]
            x, y = recip(a, b, basis, seed=r)
            if np.isfinite(x): same.append(max(x, y))
        x, y = recip(Zs, Zj, basis, seed=0)
        if np.isfinite(x): cross.append(max(x, y))
    return np.array(same), np.array(cross)


def c1_sample_curve(data, basis="poly", ns=(25, 50, 100, 200, 400)):
    out = {}
    for n in ns:
        vals = []
        for _, d in data:
            Zs, Zj = d["Zs"], d["Zj"]
            if len(Zs) < n or len(Zj) < n:
                continue
            rng = np.random.default_rng(0)
            a = Zs[rng.choice(len(Zs), n, replace=False)]
            b = Zj[rng.choice(len(Zj), n, replace=False)]
            x, y = recip(a, b, basis)
            if np.isfinite(x): vals.append((x, y))
        if vals:
            v = np.array(vals)
            out[n] = (float(np.median(v[:, 0])), float(np.median(v[:, 1])), len(vals))
    return out


def c2_bootstrap(data, basis="poly", B=200):
    """Bootstrap over trials."""
    S = [recip(d["Zs"], d["Zj"], basis) for _, d in data]
    S = np.array([s for s in S if np.isfinite(s[0])])
    rng = np.random.default_rng(0)
    bs = np.array([[np.median(S[rng.integers(0, len(S), len(S))][:, k]) for k in (0, 1)]
                   for _ in range(B)])
    return {k: (float(np.percentile(bs[:, i], 2.5)), float(np.median(S[:, i])),
                float(np.percentile(bs[:, i], 97.5)))
            for i, k in enumerate(("S_blind", "J_blind"))}


def c3_reg_sweep(data, basis="poly", lams=(1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)):
    out = {}
    for lam in lams:
        v = np.array([recip(d["Zs"], d["Zj"], basis, lam=lam) for _, d in data])
        v = v[np.isfinite(v[:, 0])]
        out[lam] = (float(np.median(v[:, 0])), float(np.median(v[:, 1])))
    return out


if __name__ == "__main__":
    RES = {}
    for pat, lab in (("adj_raw_*.npz", "GPT-2"), ("rec_raw_*.npz", "Qwen3.5")):
        data = load(pat)
        if not data:
            continue
        print("=" * 96); print(f"{lab}   n_trials={len(data)}"); print("=" * 96)

        print("\nC4  RANDOMIZED-FAMILY NULL  (split S in half vs S-vs-J, identical procedure)")
        same, cross = c4_random_family_null(data)
        print(f"     same-family  S1 vs S2 : median {np.median(same):>12.1f}  "
              f"p95 {np.percentile(same,95):>12.1f}  max {same.max():>12.1f}   (n={len(same)})")
        print(f"     CROSS-family S vs J   : median {np.median(cross):>12.1f}  "
              f"min {cross.min():>12.1f}                    (n={len(cross)})")
        print(f"     ==> cross/same median ratio = {np.median(cross)/max(np.median(same),1e-12):.3g}")

        print("\nC1  SAMPLE-SIZE CURVE (does separation decay toward 1 as n grows?)")
        cur = c1_sample_curve(data)
        print(f"     {'n':>6} {'S-blind ratio':>16} {'J-blind ratio':>16} {'trials':>8}")
        for n, (a, b, k) in cur.items():
            print(f"     {n:>6} {a:>16.3g} {b:>16.3g} {k:>8}")

        print("\nC2  BOOTSTRAP 95% CI over trials (B=200)")
        bt = c2_bootstrap(data)
        for k, (lo, md, hi) in bt.items():
            print(f"     {k:>8}: median {md:>12.3g}   95% CI [{lo:.3g}, {hi:.3g}]")

        print("\nC3  REGULARIZATION SWEEP  G + lam*I")
        rg = c3_reg_sweep(data)
        print(f"     {'lambda':>10} {'S-blind':>14} {'J-blind':>14}")
        for lam, (a, b) in rg.items():
            print(f"     {lam:>10.0e} {a:>14.3g} {b:>14.3g}")

        print("\nBASIS ROBUSTNESS (is this a polynomial artifact?)")
        print(f"     {'basis':>10} {'dim':>5} {'S-blind (med)':>16} {'J-blind (med)':>16} {'both>10x':>10}")
        bas = {}
        for bn in ("poly", "taylor", "rbf", "fourier"):
            v = np.array([recip(d["Zs"], d["Zj"], bn) for _, d in data])
            v = v[np.isfinite(v[:, 0])]
            dim = BASES[bn](np.zeros((2, 2))).shape[1]
            both = int(np.sum((v[:, 0] > 10) & (v[:, 1] > 10)))
            bas[bn] = (float(np.median(v[:, 0])), float(np.median(v[:, 1])), both, len(v))
            print(f"     {bn:>10} {dim:>5} {np.median(v[:,0]):>16.3g} "
                  f"{np.median(v[:,1]):>16.3g} {both:>7}/{len(v)}")
        RES[lab] = dict(c4_same_median=float(np.median(same)),
                        c4_same_p95=float(np.percentile(same, 95)),
                        c4_cross_median=float(np.median(cross)),
                        c1=cur, c2=bt, c3={str(k): v for k, v in rg.items()}, basis=bas)
        print()
    json.dump(RES, open("controls_subspace.json", "w"), indent=1)
    print("saved -> controls_subspace.json")
