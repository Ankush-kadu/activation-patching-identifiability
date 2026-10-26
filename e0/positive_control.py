"""POSITIVE CONTROL FOR THE NULLSPACE FINDER  (owed since bug #8, 2026-08-19 06:00)

Bug #8: the earlier attempt planted the CLAMPED kernel (a-a0)(b-b0) into RECOMPUTE data,
so it tested nothing (it is correctly NOT null there).  A properly matched control was
recorded as owed.  This is it, plus the stronger test it makes possible.

PART A -- estimator recovery on synthetic data with known ground truth.
Generate an exact union-of-slices reachable set (what single-site patching produces in the
ideal case), add noise, and ask whether the procedure recovers the KNOWN generator.
If it cannot recover a planted direction, every reported nullspace is suspect.

PART B -- does the EXACT TOY THEORY PREDICT THE REAL DISCOVERED DIRECTION?
Under recompute semantics the single-site reachable set is
    patch-A arm : {(a, h(a))}      a swept, b RECOMPUTED
    patch-B arm : {(a0, b)}        a pinned at its clean value, b swept
so the theory's blind generator is  g = (a - a0) * (b - h(a)),  affine h  =>  (a-a0)(b-ma-c).

Every ingredient is estimable from the data WITHOUT looking at the discovered direction:
    a0     = mean of a on the patch-B arm      (a is constant there by construction)
    (m, c) = least squares of b on a along the patch-A arm

So g_pred is a PRE-SPECIFIED direction.  Two things are then asked of it:
    B1  alignment      |cos(g_pred, h_discovered)|   vs a random-direction null
    B2  ** the real test **  is g_pred ITSELF S-blind and J-visible on HELD-OUT data?
        No fitting, no discovery, no eigenproblem -- the exact theory names a direction in
        advance and the trained model is asked whether it is blind there.

B2 is the strongest available link from the toy algebra to a 9B transformer.  If it fails,
the exact theory is an analogy rather than a prediction, and the paper must say so.
"""
import glob, json
import numpy as np
from levelC2 import design, exps
from subspace_test import prep, gram, analyse
from scipy.linalg import eigh

DEG = 3


def vec(coefs, d=DEG):
    e = exps(d)
    v = np.zeros(len(e))
    for k, c in coefs.items():
        v[e.index(k)] += c
    return v


def g_dict(a0, m, c):
    """(a - a0)(b - m a - c) as {(i,j): coef} over a^i b^j."""
    return {(0, 0): a0 * c, (1, 0): a0 * m - c, (0, 1): -a0,
            (2, 0): -m, (1, 1): 1.0}


def g_affine(a0, m, c, d=DEG):
    return vec(g_dict(a0, m, c), d)


def shift(p, di, dj):
    return {(i + di, j + dj): c for (i, j), c in p.items()}


def blind_basis(a0, m, c, cs, d=DEG):
    """The TRUE blind SUBSPACE at degree d: g * P_<=(d-2) = span{g, a g, b g} for d=3.

    THIS IS THE OBJECT THE THEORY NAMES.  An earlier version of this control compared the
    single found direction to the single polynomial g and reported 0/5 -- that test is
    mis-specified: at degree 3 the blind set is 3-DIMENSIONAL, so no individual direction
    should be expected to equal g.  (Exactly the 'claim a subspace, never a vector' rule.)
    """
    g = g_dict(a0, m, c)
    cols = [vec(g, d), vec(shift(g, 1, 0), d), vec(shift(g, 0, 1), d)]
    B = np.stack([v * cs for v in cols], 1)
    return np.linalg.qr(B)[0]


def resid(B, h):
    """Component of unit vector h outside span(B); 0 = fully inside the blind subspace."""
    return float(np.linalg.norm(h - B @ (B.T @ h)))


# ------------------------------------------------------------------ PART A ---
def part_a(noise_levels=(0.0, 1e-4, 1e-3, 1e-2, 1e-1), n=200, seed=0):
    rng = np.random.default_rng(seed)
    a0, b0, m, c = 0.3, -0.7, 1.7, 0.4
    rows = []
    for s in noise_levels:
        av = rng.uniform(-2, 2, n)
        bv = rng.uniform(-2, 2, n)
        # recompute single-site family: arm 1 = (a, m a + c), arm 2 = (a0, b)
        Zs = np.vstack([np.stack([av, m * av + c], 1),
                        np.stack([np.full(n, a0), bv], 1)])
        Zs = Zs + rng.normal(0, s, Zs.shape)
        # joint family: independent a and b
        Zj = np.stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)], 1)

        Str, Ste, Jtr, Jte = prep(Zs, Zj, seed=seed)
        lam, V = eigh(gram(Str) + 1e-10 * np.eye(Str.shape[1]),
                      gram(Jtr) + 1e-10 * np.eye(Str.shape[1]))
        h = V[:, 0] / np.linalg.norm(V[:, 0])

        # ground truth in the SAME scaled basis as prep()
        allZ = np.vstack([Zs, Zj]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
        Wn = lambda Z: (Z - mu) / sd
        cs = np.linalg.norm(design(Wn(allZ), DEG), axis=0); cs[cs == 0] = 1.0
        a0w = (a0 - mu[0]) / sd[0]
        mw = m * sd[0] / sd[1]
        cw = (m * mu[0] + c - mu[1]) / sd[1]
        B = blind_basis(a0w, mw, cw, cs)
        r = resid(B, h)
        rnd = []
        for _ in range(2000):
            g = rng.normal(size=B.shape[0]); g /= np.linalg.norm(g)
            rnd.append(resid(B, g))
        p1 = float(np.percentile(rnd, 1))
        rows.append(dict(noise=s, resid=r, rnd_p1=p1, rnd_median=float(np.median(rnd)),
                         recovered=bool(r < p1)))
    return rows


# ------------------------------------------------------------------ PART B ---
def part_b(Zs, Zj, seed=0):
    """Theory-predicted generator from the data alone, then tested on held-out."""
    n = len(Zs)
    h1, h2 = Zs[: n // 2], Zs[n // 2:]                 # patch-A arm, patch-B arm
    # identify which arm has (near-)constant a; that one is the patch-B arm
    if np.std(h1[:, 0]) < np.std(h2[:, 0]):
        armA, armB = h2, h1
    else:
        armA, armB = h1, h2
    a0 = float(np.mean(armB[:, 0]))
    m, c = np.polyfit(armA[:, 0], armA[:, 1], 1)       # b ~ m a + c on the swept arm

    allZ = np.vstack([Zs, Zj]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    Wn = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(Wn(allZ), DEG), axis=0); cs[cs == 0] = 1.0
    a0w = (a0 - mu[0]) / sd[0]
    mw = m * sd[0] / sd[1]
    cw = (m * mu[0] + c - mu[1]) / sd[1]
    gp = g_affine(a0w, mw, cw) * cs
    gp /= np.linalg.norm(gp)
    Bpred = blind_basis(a0w, mw, cw, cs)

    Str, Ste, Jtr, Jte = prep(Zs, Zj, seed=seed)
    lam, V = eigh(gram(Str) + 1e-10 * np.eye(Str.shape[1]),
                  gram(Jtr) + 1e-10 * np.eye(Str.shape[1]))
    hd = V[:, 0] / np.linalg.norm(V[:, 0])

    q = lambda M, v: float(np.mean((M @ v) ** 2))
    rng = np.random.default_rng(seed)
    rnd_res, rnd_sep = [], []
    for _ in range(2000):
        g = rng.normal(size=len(gp)); g /= np.linalg.norm(g)
        rnd_res.append(resid(Bpred, g))
    for _ in range(200):
        g = rng.normal(size=len(gp)); g /= np.linalg.norm(g)
        rnd_sep.append(q(Jte, g) / max(q(Ste, g), 1e-300))

    return dict(
        a0=a0, m=float(m), c=float(c),
        resid_discovered_in_predicted=resid(Bpred, hd),
        resid_rnd_p1=float(np.percentile(rnd_res, 1)),
        resid_rnd_median=float(np.median(rnd_res)),
        # B2: the pre-specified direction's own held-out behaviour
        S_test=q(Ste, gp), J_test=q(Jte, gp),
        sep_pred=q(Jte, gp) / max(q(Ste, gp), 1e-300),
        rnd_sep_p95=float(np.percentile(rnd_sep, 95)),
        rnd_sep_median=float(np.median(rnd_sep)),
        sep_discovered=q(Jte, hd) / max(q(Ste, hd), 1e-300))


if __name__ == "__main__":
    out = {}
    print("=" * 96)
    print("PART A — can the finder recover a PLANTED generator? (synthetic, known truth)")
    print("=" * 96)
    print(f"  {'noise sd':>10} {'residual outside blind subspace':>32} "
          f"{'random p1':>11} {'random med':>11} {'recovered':>11}")
    A = part_a()
    for r in A:
        print(f"  {r['noise']:>10.0e} {r['resid']:>32.5f} {r['rnd_p1']:>11.4f} "
              f"{r['rnd_median']:>11.4f} {str(r['recovered']):>11}")
    out["part_a"] = A
    print(f"\n  recovered above the random-direction 99th pct at "
          f"{sum(r['recovered'] for r in A)}/{len(A)} noise levels")

    print("\n" + "=" * 96)
    print("PART B — does the EXACT THEORY predict the real blind direction?")
    print("  g_pred = (a-a0)(b - m a - c), all terms estimated from data, NOT from the")
    print("  discovered direction. B2 is the decisive column: g_pred's OWN held-out ratio.")
    print("=" * 96)
    from cells import CELLS                     # 2026-08-27: one cell per model x task
    for c in CELLS:
        pat, lab = c["pattern"], c["label"]
        files = sorted(glob.glob(pat))
        if not files:
            continue
        rows = [part_b(np.load(f)["Zs"], np.load(f)["Zj"]) for f in files]
        cos = np.array([r["resid_discovered_in_predicted"] for r in rows])
        p99 = np.array([r["resid_rnd_p1"] for r in rows])
        sp = np.array([r["sep_pred"] for r in rows])
        sd_ = np.array([r["sep_discovered"] for r in rows])
        rp = np.array([r["rnd_sep_p95"] for r in rows])
        print(f"\n  {lab}  (n={len(rows)})")
        print(f"    B1 resid(discovered outside predicted subspace): median {np.median(cos):.4f}"
              f"   random p1 {np.median(p99):.4f}   inside-null {int(np.sum(cos<p99))}/{len(rows)}")
        print(f"    B2 g_pred held-out J/S ratio : median {np.median(sp):.4g}   "
              f">10x in {int(np.sum(sp>10))}/{len(rows)}")
        print(f"       random-direction p95      : median {np.median(rp):.4g}")
        print(f"       (discovered direction)    : median {np.median(sd_):.4g}")
        out[lab] = dict(n=len(rows), resid_median=float(np.median(cos)),
                        resid_below_null=int(np.sum(cos < p99)),
                        sep_pred_median=float(np.median(sp)),
                        sep_pred_gt10=int(np.sum(sp > 10)),
                        rnd_sep_p95_median=float(np.median(rp)),
                        sep_discovered_median=float(np.median(sd_)))
    json.dump(out, open("positive_control.json", "w"), indent=1)
    print("\nsaved -> positive_control.json")
