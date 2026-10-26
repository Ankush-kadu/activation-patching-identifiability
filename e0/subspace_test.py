"""DECISIVE LEVEL-C TEST: are the S and J approximate blind subspaces DIFFERENT?

Reformulated around the intervention Gram / information operator

    G_Q = E_{z ~ mu_Q} [ phi(z) phi(z)^T ]

Exact toy case:  h in ker G_Q  => h exactly invisible under Q.
Real models:     G_Q is full rank but ill-conditioned; h^T G_Q h ~ 0 means h is
                 PRACTICALLY invisible to the evidence that family provides.

Two things are measured, both with TRAIN/TEST separation and frozen directions:

1. GENERALIZED EIGENPROBLEM  G_S v = lam G_J v
     smallest lam -> direction most invisible under S, visible under J
     largest  lam -> direction most invisible under J, visible under S
   If BOTH exist with large margins, the two families constrain DIFFERENT
   mechanism directions -- the real-model analogue of the exact Level-B theorem.

2. PRINCIPAL ANGLES between the eps-blind eigenspaces of G_S and G_J.
   Angles near 0 => same blind subspace. Angles near 90 deg => disjoint.

Reports NRMSE and correlation alongside R^2, because a hugely negative R^2 can be
produced by a small denominator and must not be the headline.
"""
import json, glob, sys, numpy as np
from scipy.linalg import eigh, subspace_angles
from levelC2 import design

DEG, EPS = 3, 0.02


def gram(M):
    return M.T @ M / M.shape[0]


def prep(Zs, Zj, deg=DEG, seed=0):
    """Whiten with combined stats; column-scale with combined design; split."""
    rng = np.random.default_rng(seed)
    allZ = np.vstack([Zs, Zj]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(W(allZ), deg), axis=0); cs[cs == 0] = 1.0
    Ms, Mj = design(W(Zs), deg) / cs, design(W(Zj), deg) / cs
    i = rng.permutation(len(Ms)); a, b = i[:len(i)//2], i[len(i)//2:]
    j = rng.permutation(len(Mj)); c, d = j[:len(j)//2], j[len(j)//2:]
    return Ms[a], Ms[b], Mj[c], Mj[d]


def q(M, h):
    """h^T G h  = mean squared response of direction h under that family."""
    return float(np.mean((M @ h) ** 2))


def analyse(Zs, Zj, seed=0, delta=1e-10):
    Str, Ste, Jtr, Jte = prep(Zs, Zj, seed=seed)
    Gs, Gj = gram(Str), gram(Jtr)
    n = Gs.shape[0]
    I = np.eye(n)

    # ---- generalized eigenproblem on TRAIN only ----
    lam, V = eigh(Gs + delta * I, Gj + delta * I)
    h_Sblind = V[:, 0] / np.linalg.norm(V[:, 0])      # min ratio: S-blind, J-visible
    h_Jblind = V[:, -1] / np.linalg.norm(V[:, -1])    # max ratio: J-blind, S-visible

    def report(h):
        return dict(S_train=q(Str, h), S_test=q(Ste, h),
                    J_train=q(Jtr, h), J_test=q(Jte, h))

    rS, rJ = report(h_Sblind), report(h_Jblind)

    # ---- principal angles between eps-blind eigenspaces (TRAIN) ----
    def blind_space(G, eps=EPS):
        w, U = np.linalg.eigh(G)
        keep = U[:, w / max(w.max(), 1e-300) < eps]
        return keep if keep.shape[1] else None
    Bs, Bj = blind_space(Gs), blind_space(Gj)
    if Bs is not None and Bj is not None:
        ang = np.degrees(subspace_angles(Bs, Bj))
    else:
        ang = None

    return dict(
        lam_min=float(lam[0]), lam_max=float(lam[-1]),
        dimS=int(Bs.shape[1]) if Bs is not None else 0,
        dimJ=int(Bj.shape[1]) if Bj is not None else 0,
        angles=[float(x) for x in ang] if ang is not None else None,
        Sblind=rS, Jblind=rJ,
        # held-out ratios: the frozen direction's visibility under each family
        sep_Sblind=rS["J_test"] / max(rS["S_test"], 1e-300),
        sep_Jblind=rJ["S_test"] / max(rJ["J_test"], 1e-300))


def run(pattern, label):
    print("=" * 104)
    print(f"{label}   (generalized eigenproblem  G_S v = lam G_J v ; TRAIN-fit, TEST-evaluated)")
    print("=" * 104)
    print(f"{'trial':>12} | {'S-BLIND dir: S_test':>20} {'J_test':>10} {'ratio':>10} "
          f"| {'J-BLIND dir: J_test':>20} {'S_test':>10} {'ratio':>10} | {'angles(deg)':>14}")
    print("-" * 104)
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        r = analyse(d["Zs"], d["Zj"])
        rows.append(r)
        ang = ("/".join(f"{a:.0f}" for a in r["angles"][:3]) if r["angles"] else "n/a")
        t = f.split("raw_")[1].replace(".npz", "")
        print(f"{t:>12} | {r['Sblind']['S_test']:>20.3e} {r['Sblind']['J_test']:>10.3e} "
              f"{r['sep_Sblind']:>10.1f} | {r['Jblind']['J_test']:>20.3e} "
              f"{r['Jblind']['S_test']:>10.3e} {r['sep_Jblind']:>10.1f} | {ang:>14}")
    if not rows:
        return None
    sS = np.array([r["sep_Sblind"] for r in rows])
    sJ = np.array([r["sep_Jblind"] for r in rows])
    print(f"\n  MEDIAN held-out ratio   S-blind dir (J_test/S_test) = {np.median(sS):>10.1f}")
    print(f"  MEDIAN held-out ratio   J-blind dir (S_test/J_test) = {np.median(sJ):>10.1f}")
    both = np.sum((sS > 10) & (sJ > 10))
    print(f"  BOTH directions exist with ratio >10x in {both}/{len(rows)} configs")
    allang = [a for r in rows if r["angles"] for a in r["angles"]]
    if allang:
        print(f"  principal angles between eps-blind eigenspaces: "
              f"median {np.median(allang):.1f} deg, range [{min(allang):.1f}, {max(allang):.1f}]")
    return dict(n=len(rows), med_sS=float(np.median(sS)), med_sJ=float(np.median(sJ)),
                both=int(both), angles_med=float(np.median(allang)) if allang else None)


if __name__ == "__main__":
    out = {}
    out["gpt2"] = run("adj_raw_*.npz", "GPT-2")
    print()
    out["qwen"] = run("rec_raw_*.npz", "Qwen3.5-9B (ACTIVE pairs)")
    json.dump(out, open("subspace_test.json", "w"), indent=1)
    print("\n" + "=" * 104)
    print("INTERPRETATION")
    print("=" * 104)
    print("If BOTH an S-blind/J-visible AND a J-blind/S-visible direction exist with")
    print("large held-out ratios, then neither family's blind subspace contains the")
    print("other -- the real-model analogue of the exact Level-B incomparability.")
    print("If only the S-blind direction exists, J is strictly more informative.")
