"""THE COMPARISON PROPOSITION 1 IS ACTUALLY ABOUT  (added 2026-08-27).

Section 6 compares the single-site family against the JOINT family and finds an order: joint
dominates.  That is expected, not surprising -- joint patching reaches a product grid that
essentially contains the single-site cross, so the reachable sets are nested and Loewner
comparability is the only possible outcome.  Testing incomparability there was the wrong test.

Proposition 1 is about two families NEITHER of which contains the other:

    recompute  R = {(a, h(a))} u {(a_0, b)}      patch A, let B recompute
    clamped    C = {(a, b_0)}  u {(a_0, b)}      patch A, HOLD B at its clean value

They share the patch-B arm and differ only on the patch-A arm.  Neither reachable set contains
the other exactly when the realised edge has m != 0, and then
    g_R = (a-a_0)(b - m a - c)  is blind to R and VISIBLE to C
    g_C = (a-a_0)(b - b_0)      is blind to C and VISIBLE to R
which is mutual non-domination -- the property that makes N(.,F) a preorder rather than an order.

So the prediction is sharp and falsifiable, and it comes with its own scope condition: the effect
must APPEAR where the measured edge slope is large and VANISH where it is near zero, since at
m = 0 the two families coincide.  Both directions are tested here.
"""
import argparse, glob, json
import numpy as np
from invariant import one
from nulls import loo_envelope, escape_binom_p, holm
from edge_strength import analyse as edge_analyse, generators
from subspace_test import prep
from cells import MODEL_LABELS

# A pre-stated VALIDITY SCREEN on the chart, applied before any outcome is looked at.
# matched.edge_chart picks the downstream coordinate as the leading generalized eigenvector of
# (Cov(B | arm A), Cov(B | arm B)).  Both covariances are built from at most 16 donor directions
# in a 768- (GPT-2) or 3584-dimensional (Qwen) space, so the DENOMINATOR is rank-deficient and a
# direction can be found in its null space, where the ratio is set by the ridge rather than by the
# data.  The symptom is unmistakable: sd(b | arm A) / sd(b | arm B) explodes, i.e. patching B --
# the site itself -- moves b less than patching A does, which cannot be true of a valid chart.
# We therefore declare a chart invalid when that ratio exceeds SD_RATIO_MAX and report the count.
SD_RATIO_MAX = 50.0


def cells_present(*need):
    """Only captures that carry EVERY array this comparison needs, in the requested chart."""
    out = {}
    for f in sorted(glob.glob("*_matched_*.npz")):
        have = np.load(f).files
        if any(k not in have for k in need):
            continue
        task, _, tag, lA, lB, seed = f[:-4].split("_")
        out.setdefault(f"{MODEL_LABELS.get(tag, tag)} {task}", []).append(f)
    return out


def named_directions(Zs, Zc, seed=0):
    """Held-out response of each family's OWN theory-named generator under the other family."""
    gR, gC, diag = generators(Zs, Zc)          # both generators, in the shared whitened basis
    Rtr, Rte, Ctr, Cte = prep(Zs, Zc, seed=seed)
    q = lambda M, v: float(np.mean((M @ v) ** 2))
    return dict(
        # g_R should be small under R and large under C
        gR_on_R=q(Rte, gR), gR_on_C=q(Cte, gR), gR_ratio=q(Cte, gR) / max(q(Rte, gR), 1e-300),
        # g_C should be small under C and large under R
        gC_on_C=q(Cte, gC), gC_on_R=q(Rte, gC), gC_ratio=q(Rte, gC) / max(q(Cte, gC), 1e-300),
        **{k: diag[k] for k in ("slope_whitened", "sd_ratio", "edge_r2", "cos_gR_gC")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-donors", action="store_true",
                    help="evaluate only on states from donors the edge chart was NOT fitted on. "
                         "matched.edge_chart fits on the first half of the patch-A arm, which is "
                         "donors 0..15 (rows are donor-major, 7 coefficients each); this "
                         "restricts the comparison to donors 16..31 in both arms, so the chart "
                         "never saw the states it is scored on. The chart uses no readout in "
                         "either case; this closes the transductive objection as well.")
    ap.add_argument("--chart", default="u", choices=["u", "edge", "edge_x"],
                    help="u = unembedding direction (the paper's current chart); "
                         "edge = the readout-free edge-aligned chart of matched.edge_chart")
    A = ap.parse_args()
    # edge_x: the CROSS-SEED control -- states projected onto a chart fitted on a DIFFERENT
    # seed of the same layer pair, i.e. a different prompt and a disjoint donor set. This is the
    # control that separates chart selection from the effect; the donor holdout does not, because
    # the chart is still fitted on this capture's own prompt.
    KS, KC = {"edge": ("Zse", "Zce"), "edge_x": ("Zse_x", "Zce_x")}.get(A.chart, ("Zs", "Zc"))
    found = cells_present(KS, KC)
    N_AL_, N_FIT = 7, 16                      # coefficients per donor; donors used for the fit

    def holdout(Z):
        """Keep only rows from donors the chart was not fitted on, in both arms."""
        n = len(Z) // 2
        keep = np.array([r for r in range(n) if r // N_AL_ >= N_FIT])
        return np.vstack([Z[keep], Z[n + keep]])
    if not found:
        raise SystemExit("no matched captures with a clamped family -- re-run matched.py")

    res, rows_all, dropped = {}, [], []
    for lab, files in sorted(found.items()):
        rows, nd = [], []
        for f in files:
            d = np.load(f)
            Zr_, Zc_ = d[KS], d[KC]
            if A.holdout_donors:
                Zr_, Zc_ = holdout(Zr_), holdout(Zc_)
            x = named_directions(Zr_, Zc_); x["file"] = f
            if A.chart.startswith("edge") and x["sd_ratio"] > SD_RATIO_MAX:
                dropped.append((f, x["sd_ratio"]))      # chart selection failed; see SD_RATIO_MAX
                continue
            rows.append(one(Zr_, Zc_))                  # R vs C, identical machinery
            nd.append(x); rows_all.append(x)
        if not rows:
            continue
        spec = np.array([r["log_spectrum"] for r in rows])
        nulls = [r["null_log_spectrum"] for r in rows]
        below = above = 0
        for i in range(len(rows)):
            lo, hi = loo_envelope(nulls, i)
            below += spec[i].min() < lo
            above += spec[i].max() > hi
        n = len(rows)
        res[lab] = dict(n=n, below=int(below), above=int(above),
                        p_below=escape_binom_p(below, n), p_above=escape_binom_p(above, n),
                        median_gR_ratio=float(np.median([x["gR_ratio"] for x in nd])),
                        median_gC_ratio=float(np.median([x["gC_ratio"] for x in nd])),
                        median_cos=float(np.median([x["cos_gR_gC"] for x in nd])),
                        rows=nd)

    keys = list(res)
    for pk, qk in (("p_below", "q_below"), ("p_above", "q_above")):
        for k, q in zip(keys, holm([res[k][pk] for k in keys])):
            res[k][qk] = float(q)

    if dropped:
        print(f"CHART VALIDITY SCREEN: {len(dropped)} capture(s) excluded, "
              f"sd(b|armA)/sd(b|armB) > {SD_RATIO_MAX:g} (the pencil's denominator collapsed):")
        for f, r in dropped:
            print(f"    {f:<40} sd ratio {r:.3g}")
        print()
    print("=" * 104)
    print(f"RECOMPUTE vs CLAMPED  [chart: {A.chart}"
          f"{', donors held out from the chart fit' if A.holdout_donors else ''}]"
          f"  -- incomparability is PREDICTED here")
    print("=" * 104)
    print(f"{'cell':<22}{'n':>4}{'R-blind':>10}{'q':>9}{'C-blind':>10}{'q':>9}"
          f"{'g_R vis under C':>17}{'g_C vis under R':>17}  verdict")
    print("-" * 104)
    for k in keys:
        r = res[k]
        lo_ok, hi_ok = r["q_below"] < 0.05, r["q_above"] < 0.05
        v = ("MUTUAL NON-DOMINATION" if lo_ok and hi_ok else
             "R-blind only" if lo_ok else "C-blind only" if hi_ok else "not established")
        print(f"{k:<22}{r['n']:>4}{r['below']:>7}/{r['n']:<2}{r['q_below']:>9.1e}"
              f"{r['above']:>7}/{r['n']:<2}{r['q_above']:>9.1e}"
              f"{r['median_gR_ratio']:>17.4g}{r['median_gC_ratio']:>17.4g}  {v}")

    tb = sum(res[k]["below"] for k in keys); ta = sum(res[k]["above"] for k in keys)
    tn = sum(res[k]["n"] for k in keys)
    both_cells = sum((res[k]["q_below"] < 0.05) and (res[k]["q_above"] < 0.05) for k in keys)
    print(f"\n  pooled: R-blind {tb}/{tn}, C-blind {ta}/{tn}; "
          f"cells with mutual non-domination {both_cells}/{len(keys)}")

    # the scope condition: the effect must track the measured edge slope
    from scipy.stats import spearmanr
    m = np.array([abs(x["slope_whitened"]) for x in rows_all])
    gr = np.array([x["gR_ratio"] for x in rows_all])
    gc = np.array([x["gC_ratio"] for x in rows_all])
    cos = np.array([x["cos_gR_gC"] for x in rows_all])
    print(f"\n  scope condition (Proposition 1 needs m != 0), n={len(m)} captures:")
    print(f"    Spearman(|m|, g_C visible under R) = {spearmanr(m, gc)[0]:+.3f} "
          f"(p={spearmanr(m, gc)[1]:.2g})")
    print(f"    Spearman(|m|, g_R visible under C) = {spearmanr(m, gr)[0]:+.3f} "
          f"(p={spearmanr(m, gr)[1]:.2g})")
    print(f"    Spearman(|m|, |cos(g_R,g_C)|)      = {spearmanr(m, cos)[0]:+.3f} "
          f"(p={spearmanr(m, cos)[1]:.2g})")
    strong = m >= np.median(m)
    print(f"    strong-edge half: median g_C-under-R {np.median(gc[strong]):.3g}, "
          f"g_R-under-C {np.median(gr[strong]):.3g}")
    print(f"    weak-edge   half: median g_C-under-R {np.median(gc[~strong]):.3g}, "
          f"g_R-under-C {np.median(gr[~strong]):.3g}")
    res["_screen"] = dict(n_dropped=len(dropped), sd_ratio_max=SD_RATIO_MAX,
                          dropped=[f for f, _ in dropped])
    res["_chart"] = dict(
        median_abs_m=float(np.median(m)),
        median_cos=float(np.median(cos)),
        median_r2=float(np.median([x["edge_r2"] for x in rows_all])),
        median_sd_ratio=float(np.median([x["sd_ratio"] for x in rows_all])),
        median_gR_under_C=float(np.median(gr)), median_gC_under_R=float(np.median(gc)))
    print(f"\n  chart diagnostics (median over {len(m)} captures): |m| = {np.median(m):.3f}, "
          f"R2(b~a) = {res['_chart']['median_r2']:.3f}, "
          f"sd ratio = {res['_chart']['median_sd_ratio']:.3f}, "
          f"|cos(g_R,g_C)| = {np.median(cos):.4f}")
    res["_scope"] = dict(n=len(m),
                         rho_m_gC=float(spearmanr(m, gc)[0]), p_m_gC=float(spearmanr(m, gc)[1]),
                         rho_m_gR=float(spearmanr(m, gr)[0]), p_m_gR=float(spearmanr(m, gr)[1]),
                         rho_m_cos=float(spearmanr(m, cos)[0]),
                         strong_gC=float(np.median(gc[strong])), weak_gC=float(np.median(gc[~strong])),
                         strong_gR=float(np.median(gr[strong])), weak_gR=float(np.median(gr[~strong])),
                         pooled_below=int(tb), pooled_above=int(ta), pooled_n=int(tn),
                         cells_mutual=int(both_cells), n_cells=len(keys))
    fn = f"recompute_vs_clamped_{A.chart}{'_holdout' if A.holdout_donors else ''}.json"
    json.dump(res, open(fn, "w"), indent=1, default=float)
    print(f"\nsaved -> {fn}")
