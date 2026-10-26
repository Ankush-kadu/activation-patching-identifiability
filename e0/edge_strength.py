"""EMPIRICAL TEST OF THEOREM 1 IN A TRAINED TRANSFORMER (added 2026-08-27).

Theorem 1 compares two single-site families on the same two-site slice:

    recompute :  R_R = {(a, m a + c)} u {(a_0, b)}   ->  g_R = (a - a_0)(b - m a - c)
    clamped   :  R_C = {(a, b_0)}     u {(a_0, b)}   ->  g_C = (a - a_0)(b - b_0)

and says the two blind spaces are INCOMPARABLE -- provided the two linear forms are
non-associate, i.e. provided (m, c) != (0, b_0).  With m = 0 the recompute edge is flat, the
two reachable sets coincide, and the theorem's hypothesis fails.

That hypothesis is a MEASURABLE property of a trained model: m is the slope of the realised
A -> B edge in the chart, and the audit of 2026-08-27 showed it is near zero in most of our
site pairs (median sd(b | arm A) / sd(b | arm B) ~ 0.02).  Where it is near zero, g_R and g_C
are nearly the same direction and our "theory-named" direction is really the trivial cross
generator -- a fair criticism of the headline cells.  But the frozen captures also contain
pairs with a genuinely strong edge, and there the theorem's prediction is sharp and testable:

    g_R is blind to recompute evidence   (it vanishes on R_R by construction)
    g_C is NOT blind to recompute evidence, and increasingly visible as |m| grows

So this script measures, per capture: edge strength, |cos(g_R, g_C)|, and the held-out
response ratios of BOTH generators under the recompute single-site family.  A separation
between them that grows with edge strength is Theorem 1 observed in a trained transformer,
not in a toy -- and it is the one place where the recompute-vs-clamp distinction that the
whole framing rests on is exercised empirically.
"""
import glob, json
import numpy as np
from levelC2 import design, exps
from subspace_test import prep, gram
from cells import CELLS

DEG = 3


def _vec(coefs, cs):
    """Coefficient vector of a polynomial in the SAME scaled basis prep() produces."""
    e = exps(DEG)
    v = np.zeros(len(e))
    for k, c in coefs.items():
        v[e.index(k)] += c
    v = v * cs
    return v / np.linalg.norm(v)


def generators(Zs, Zj):
    """(g_R, g_C, diagnostics) estimated from the intervention STATES alone -- no readout."""
    n = len(Zs)
    h1, h2 = Zs[: n // 2], Zs[n // 2:]
    armA, armB = (h2, h1) if np.std(h1[:, 0]) < np.std(h2[:, 0]) else (h1, h2)
    a0 = float(np.mean(armB[:, 0]))
    m, c = np.polyfit(armA[:, 0], armA[:, 1], 1)
    b0 = float(m * a0 + c)                      # clean downstream value = h(a_0)

    # edge diagnostics
    sd_ratio = float(np.std(armA[:, 1]) / max(np.std(armB[:, 1]), 1e-300))
    r = np.corrcoef(armA[:, 0], armA[:, 1])[0, 1]
    edge_r2 = float(r ** 2) if np.isfinite(r) else 0.0

    allZ = np.vstack([Zs, Zj])
    mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(W(allZ), DEG), axis=0); cs[cs == 0] = 1.0
    a0w = (a0 - mu[0]) / sd[0]
    b0w = (b0 - mu[1]) / sd[1]
    mw = m * sd[0] / sd[1]
    cw = (m * mu[0] + c - mu[1]) / sd[1]

    # (a - a0)(b - m a - c)  and  (a - a0)(b - b0), in whitened coordinates
    gR = _vec({(0, 0): a0w * cw, (1, 0): a0w * mw - cw, (0, 1): -a0w,
               (2, 0): -mw, (1, 1): 1.0}, cs)
    gC = _vec({(0, 0): a0w * b0w, (1, 0): -b0w, (0, 1): -a0w, (1, 1): 1.0}, cs)
    return gR, gC, dict(slope=float(m), slope_whitened=float(mw), a0=a0, b0=b0,
                        sd_ratio=sd_ratio, edge_r2=edge_r2,
                        cos_gR_gC=float(abs(gR @ gC)))


def analyse(Zs, Zj, seed=0, n_rand=400):
    Str, Ste, Jtr, Jte = prep(Zs, Zj, seed=seed)
    gR, gC, diag = generators(Zs, Zj)
    q = lambda M, v: float(np.mean((M @ v) ** 2))
    ratio = lambda v: q(Jte, v) / max(q(Ste, v), 1e-300)
    rng = np.random.default_rng(seed)
    rnd = []
    for _ in range(n_rand):
        g = rng.normal(size=len(gR)); g /= np.linalg.norm(g)
        rnd.append(ratio(g))
    diag.update(sep_gR=ratio(gR), sep_gC=ratio(gC),
                rnd_p95=float(np.percentile(rnd, 95)), rnd_median=float(np.median(rnd)))
    diag["gR_over_gC"] = diag["sep_gR"] / max(diag["sep_gC"], 1e-300)
    return diag


if __name__ == "__main__":
    out, allrows = {}, []
    print(f"{'capture':<32}{'sd(b|A)/sd(b|B)':>16}{'R2(b~a)':>9}{'|cos(gR,gC)|':>14}"
          f"{'sep g_R':>12}{'sep g_C':>12}{'g_R/g_C':>10}{'rnd p95':>9}")
    print("-" * 114)
    for cell in CELLS:
        rows = []
        for f in sorted(glob.glob(cell["pattern"])):
            d = np.load(f)
            r = analyse(d["Zs"], d["Zj"])
            r["file"] = f
            rows.append(r); allrows.append(r)
            print(f"{f:<32}{r['sd_ratio']:>16.4f}{r['edge_r2']:>9.3f}{r['cos_gR_gC']:>14.4f}"
                  f"{r['sep_gR']:>12.4g}{r['sep_gC']:>12.4g}{r['gR_over_gC']:>10.3g}"
                  f"{r['rnd_p95']:>9.2f}")
        if rows:
            out[cell["label"]] = dict(n=len(rows), rows=rows,
                                      median_sd_ratio=float(np.median([r["sd_ratio"] for r in rows])),
                                      median_cos=float(np.median([r["cos_gR_gC"] for r in rows])),
                                      median_sep_gR=float(np.median([r["sep_gR"] for r in rows])),
                                      median_sep_gC=float(np.median([r["sep_gC"] for r in rows])),
                                      median_gR_over_gC=float(np.median([r["gR_over_gC"] for r in rows])))

    # Does the recompute/clamped distinction appear exactly where the edge is strong?
    sd_r = np.array([r["sd_ratio"] for r in allrows])
    sep_r = np.array([r["gR_over_gC"] for r in allrows])
    cos_r = np.array([r["cos_gR_gC"] for r in allrows])
    ok = np.isfinite(sd_r) & np.isfinite(sep_r) & (sep_r > 0)
    from scipy.stats import spearmanr
    rho_sep, p_sep = spearmanr(sd_r[ok], sep_r[ok])
    rho_cos, p_cos = spearmanr(sd_r[ok], cos_r[ok])
    # How much of the headline blindness is just a flat edge?  If sep(g_R) falls as the realised
    # edge strengthens, then the impressive ratios come from site pairs where A barely drives B.
    # This is the tautology concern in measurable form and must be reported, not avoided.
    sepR = np.array([r["sep_gR"] for r in allrows])
    rho_blind, p_blind = spearmanr(sd_r, sepR)
    strong = sd_r >= 0.10
    print("\n" + "=" * 114)
    print("THEOREM 1, EMPIRICALLY: the recompute and clamped generators separate where the edge is strong")
    print("=" * 114)
    print(f"  n captures = {len(allrows)};  strong-edge (sd ratio >= 0.10): {int(strong.sum())}")
    print(f"  Spearman(edge strength, g_R BLINDNESS ratio) = {rho_blind:+.3f}  (p={p_blind:.2g})"
          f"   <-- the headline falls as the edge strengthens")
    print(f"  Spearman(edge strength, g_R/g_C separation) = {rho_sep:+.3f}  (p={p_sep:.2g})")
    print(f"  Spearman(edge strength, |cos(g_R,g_C)|)     = {rho_cos:+.3f}  (p={p_cos:.2g})")
    for nm, msk in (("weak edge  (sd ratio < 0.10)", ~strong), ("STRONG edge (sd ratio >= 0.10)", strong)):
        if msk.sum():
            print(f"  {nm:<32} median |cos(g_R,g_C)| = {np.median(cos_r[msk]):.4f}   "
                  f"median g_R/g_C = {np.median(sep_r[msk]):.3g}   "
                  f"median sep g_C = {np.median([r['sep_gC'] for r, k in zip(allrows, msk) if k]):.3g}")
    out["_summary"] = dict(n=len(allrows), n_strong=int(strong.sum()),
                           rho_blind=float(rho_blind), p_blind=float(p_blind),
                           median_sepR_strong=float(np.median(sepR[strong])),
                           median_sepR_weak=float(np.median(sepR[~strong])),
                           strong_gt10=int(np.sum(sepR[strong] > 10)),
                           spearman_sep_rho=float(rho_sep), spearman_sep_p=float(p_sep),
                           spearman_cos_rho=float(rho_cos), spearman_cos_p=float(p_cos),
                           weak_median_cos=float(np.median(cos_r[~strong])) if (~strong).sum() else None,
                           strong_median_cos=float(np.median(cos_r[strong])) if strong.sum() else None,
                           weak_median_gR_over_gC=float(np.median(sep_r[~strong])) if (~strong).sum() else None,
                           strong_median_gR_over_gC=float(np.median(sep_r[strong])) if strong.sum() else None,
                           strong_median_sep_gC=float(np.median([r["sep_gC"] for r, k in zip(allrows, strong) if k])) if strong.sum() else None)
    json.dump(out, open("edge_strength.json", "w"), indent=1)
    print("\nsaved -> edge_strength.json")
