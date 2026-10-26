"""REPARAMETERIZATION INVARIANCE — and the two places where it FAILS.

External review (2026-08-19) raised three related claims. All three are checked here on the
frozen real-model data rather than asserted.

CLAIM 1 (invariance).  If phi'(z) = A phi(z) for invertible A -- i.e. the SAME mechanism class
F described in different feature coordinates -- then G'_Q = A G_Q A^T, and for v' = A^{-T} v
    v'^T G'_Q v'  =  v^T G_Q v .
Hence the generalized Rayleigh quotient R(v) = v^T G_{Q1} v / v^T G_{Q2} v is unchanged, and
the generalized eigenvalues of the UNREGULARIZED pair (G_{Q1}, G_{Q2}) are invariant.
=> poly vs taylor is a REPARAMETERIZATION of one F (a conditioning check).
   poly vs rbf/fourier are DIFFERENT declared mechanism classes (a real robustness test).
   Calling all four "a basis sweep" conflates the two.

CLAIM 2 (ridge is NOT invariant).  Under G -> A G A^T the regularizer G + lam*I does not
transform covariantly for non-orthogonal A.  So the lam sweep is a numerical robustness check
IN THE CHOSEN COORDINATES, not an intrinsic statement about (Q,F).

CLAIM 3 (Euclidean principal angles are NOT invariant).  Angles between COEFFICIENT subspaces
depend on the inner product.  The reported 68.4 deg / 62.9 deg are coordinate-space
diagnostics unless a metric is declared.

A fourth, independent point worth verifying: if G_J = c G_S for a scalar c, every direction has
the same ratio, so RECIPROCITY IS IMPOSSIBLE.  Observing both an S-blind and a J-blind
direction therefore rules out a global family-scale difference as the explanation.
"""
import glob, json
import numpy as np
from scipy.linalg import eigh, subspace_angles
from levelC2 import design
from subspace_test import prep, gram

DEG = 3


def gen_eigs(Gs, Gj, lam=0.0):
    n = Gs.shape[0]
    I = np.eye(n)
    return eigh(Gs + lam * I, Gj + lam * I, eigvals_only=True)


def rand_invertible(p, rng, cond_target=50.0):
    """A well-conditioned but decidedly non-orthogonal invertible matrix."""
    U = np.linalg.qr(rng.normal(size=(p, p)))[0]
    V = np.linalg.qr(rng.normal(size=(p, p)))[0]
    s = np.logspace(0, np.log10(cond_target), p)
    return U @ np.diag(s) @ V.T


def analyse(Zs, Zj, seed=0):
    rng = np.random.default_rng(seed)
    Str, Ste, Jtr, Jte = prep(Zs, Zj, seed=seed)
    Gs, Gj = gram(Str), gram(Jtr)
    p = Gs.shape[0]
    A = rand_invertible(p, rng)

    # phi' = A phi  =>  design' = design @ A.T  =>  G' = A G A^T
    Gs2, Gj2 = A @ Gs @ A.T, A @ Gj @ A.T

    out = {"cond_A": float(np.linalg.cond(A)),
           "cond_Gj": float(np.linalg.cond(Gj))}

    # ---- CLAIM 1: unregularized generalized eigenvalues are invariant ----
    try:
        l1, l2 = gen_eigs(Gs, Gj), gen_eigs(Gs2, Gj2)
        rel = np.abs(l1 - l2) / np.maximum(np.abs(l1), 1e-300)
        out["unreg_max_rel_diff"] = float(np.max(rel))
        out["unreg_ratio_span_before"] = float(l1[-1] / max(l1[0], 1e-300))
        out["unreg_ratio_span_after"] = float(l2[-1] / max(l2[0], 1e-300))
    except np.linalg.LinAlgError:
        out["unreg_max_rel_diff"] = float("nan")

    # ---- CLAIM 2: ridge-regularized eigenvalues are NOT invariant ----
    for lam in (1e-8, 1e-4, 1e-2):
        l1, l2 = gen_eigs(Gs, Gj, lam), gen_eigs(Gs2, Gj2, lam)
        rel = np.abs(l1 - l2) / np.maximum(np.abs(l1), 1e-300)
        out[f"ridge_{lam:g}_max_rel_diff"] = float(np.max(rel))

    # ---- CLAIM 3: Euclidean principal angles are NOT invariant ----
    def blind(G, eps=0.02):
        w, U = np.linalg.eigh(G)
        k = U[:, w / max(w.max(), 1e-300) < eps]
        return k if k.shape[1] else None
    b1, b2 = blind(Gs), blind(Gj)
    c1, c2 = blind(Gs2), blind(Gj2)
    if b1 is not None and b2 is not None and c1 is not None and c2 is not None:
        a_before = np.degrees(subspace_angles(b1, b2))
        a_after = np.degrees(subspace_angles(c1, c2))
        out["angles_before_med"] = float(np.median(a_before))
        out["angles_after_med"] = float(np.median(a_after))
        out["angles_max_abs_shift"] = float(np.max(np.abs(
            np.sort(a_before)[:min(len(a_before), len(a_after))] -
            np.sort(a_after)[:min(len(a_before), len(a_after))])))

    # ---- global-scale check: is G_J close to a scalar multiple of G_S? ----
    c = float(np.trace(Gj) / max(np.trace(Gs), 1e-300))
    out["scale_c"] = c
    out["rel_resid_Gj_minus_cGs"] = float(
        np.linalg.norm(Gj - c * Gs, 2) / max(np.linalg.norm(Gj, 2), 1e-300))
    return out


if __name__ == "__main__":
    from cells import CELLS                     # 2026-08-27: one cell per model x task
    RES = {}
    for c in CELLS:
        pat, lab = c["pattern"], c["label"]
        files = sorted(glob.glob(pat))
        if not files:
            continue
        rows = [analyse(np.load(f)["Zs"], np.load(f)["Zj"]) for f in files]
        med = lambda k: float(np.median([r[k] for r in rows if k in r and np.isfinite(r[k])]))
        print("=" * 92)
        print(f"{lab}   n={len(rows)}   median cond(A)={med('cond_A'):.1f}   "
              f"median cond(G_J)={med('cond_Gj'):.3g}")
        print("=" * 92)
        print(f"  CLAIM 1  unregularized gen-eigenvalues invariant under phi -> A phi")
        print(f"           max relative difference (median over trials): "
              f"{med('unreg_max_rel_diff'):.3e}   <-- must be ~0")
        print(f"  CLAIM 2  ridge-regularized eigenvalues under the same A")
        for lam in (1e-8, 1e-4, 1e-2):
            print(f"           lambda={lam:<8g} max rel diff = "
                  f"{med(f'ridge_{lam:g}_max_rel_diff'):.3e}   <-- NOT invariant")
        print(f"  CLAIM 3  Euclidean principal angles between blind eigenspaces")
        print(f"           before {med('angles_before_med'):.1f} deg   "
              f"after {med('angles_after_med'):.1f} deg   "
              f"max shift {med('angles_max_abs_shift'):.1f} deg  <-- NOT invariant")
        print(f"  SCALE    is G_J ~ c G_S?  c={med('scale_c'):.3g}   "
              f"relative residual {med('rel_resid_Gj_minus_cGs'):.3f}")
        print(f"           (a large residual means the families differ in SHAPE, not just scale;")
        print(f"            and reciprocity is impossible under an exact global scale anyway)\n")
        RES[lab] = dict(n=len(rows), **{k: med(k) for k in rows[0]})
    json.dump(RES, open("reparam.json", "w"), indent=1)
    print("saved -> reparam.json")
