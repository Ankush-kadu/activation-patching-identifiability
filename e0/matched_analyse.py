"""Analyse the support-matched captures, side by side with the frozen alpha=1 protocol.

Each `{task}_matched_{model}_{lA}_{lB}_{seed}.npz` holds three families collected in one pass on
identical prompts, donors and weights:

    Zs   single-site, both arms, alpha in linspace(-0.5, 1.5, 7)
    Zjm  joint, SAME alpha grid          <- matched supports (the corrected protocol)
    Zj   joint, raw donors (alpha = 1)   <- the protocol used in the submitted draft

so the effect of the support gap is measured within-trial rather than across runs.  Four
protocols are reported per cell (see PROTOCOLS below), and for each of them whether a trial's
reverse (J-blind) escape is predicted by the measured strength of the realised A->B edge --
Proposition 1's non-degeneracy hypothesis.

The `symmetric` protocol exists to answer the one serious objection to the others: matching the
grids ENLARGES the joint family (it now reaches the extrapolated coefficients too), so a sceptic
can say the reverse direction disappeared only because the joint family got stronger.  Restricting
BOTH families to the interior coefficients tests exactly that, and it does not resurrect the claim.
"""
import glob, json
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from invariant import run
from edge_strength import analyse as edge_analyse
from cells import restrict_alpha, MODEL_LABELS, INTERIOR
from nulls import holm

# (name, joint key, restrict S to interior alphas, restrict the JOINT family too)
#   matched    S full          vs Jm on the same grid      <- best matched by the measured box ratio
#   frozen     S full          vs J at alpha = 1           <- the protocol of the earlier draft
#   interior   S interior      vs J at alpha = 1           <- conservative check on frozen captures
#   symmetric  S interior      vs Jm interior              <- answers "matching only helped because
#                                                             it ENLARGED the joint family"
PROTOCOLS = [("matched", "Zjm", False, False), ("frozen", "Zj", False, False),
             ("interior", "Zj", True, False), ("symmetric", "Zjm", True, True)]


def cells_present():
    out = {}
    for f in sorted(glob.glob("*_matched_*.npz")):
        task, _, tag, lA, lB, seed = f[:-4].split("_")
        out.setdefault((task, tag), []).append(f)
    return out


if __name__ == "__main__":
    found = cells_present()
    if not found:
        raise SystemExit("no *_matched_*.npz captures found — run matched.py first")

    res, edges = {}, {}
    for (task, tag), files in sorted(found.items()):
        label = f"{MODEL_LABELS.get(tag, tag)} {task}"
        res[label] = {}
        for pname, jkey, restrict, restrict_j in PROTOCOLS:
            pairs, box = [], []
            for f in files:
                d = np.load(f)
                Zs = restrict_alpha(d["Zs"]) if restrict else d["Zs"]
                Zj = d[jkey]
                if restrict_j:
                    keep = np.isin(d["alphaJA"], INTERIOR) & np.isin(d["alphaJB"], INTERIOR)
                    Zj = Zj[keep]
                    if len(Zj) < 30:
                        continue
                pairs.append((f, Zs, Zj))
                box.append([np.ptp(Zs[:, k]) / max(np.ptp(Zj[:, k]), 1e-30) for k in (0, 1)])
            r = run(None, f"{label}  [{pname}]", pairs=pairs)
            r["box_ratio"] = float(np.median(box))
            res[label][pname] = r
        for f in files:
            d = np.load(f)
            edges[f] = edge_analyse(d["Zs"], d["Zjm"])

    # Holm across cells, within each protocol
    for pname, *_ in PROTOCOLS:
        keys = [k for k in res if res[k].get(pname)]
        for pk, qk in (("p_below", "q_below"), ("p_above", "q_above")):
            for k, q in zip(keys, holm([res[k][pname][pk] for k in keys])):
                res[k][pname][qk] = float(q)

    print("\n" + "=" * 108)
    print("SUPPORT MATCHING: what the joint family's alpha grid does to each claim")
    print("=" * 108)
    print(f"{'cell':<24}{'protocol':<10}{'n':>4}{'S-blind':>10}{'q':>10}{'J-blind':>10}{'q':>10}"
          f"{'S/J box':>10}")
    print("-" * 108)
    for k in res:
        for pname, jkey, restrict, restrict_j in PROTOCOLS:
            r = res[k].get(pname)
            if not r:
                continue
            box = r["box_ratio"]
            print(f"{k if pname=='matched' else '':<24}{pname:<10}{r['n']:>4}"
                  f"{r['below']:>7}/{r['n']:<2}{r['q_below']:>10.2g}"
                  f"{r['above']:>7}/{r['n']:<2}{r['q_above']:>10.2g}{box:>10.2f}")

    # ---- does the measured edge strength predict the reverse (J-blind) escape? ----
    print("\n" + "=" * 108)
    print("THEOREM 1's HYPOTHESIS, TESTED: the reverse blind direction appears where the edge is strong")
    print("=" * 108)
    summary = {}
    for pname, *_ in PROTOCOLS:
        sd, ab = [], []
        for k in res:
            r = res[k].get(pname)
            if not r:
                continue
            for t in r["per_trial"]:
                if t["file"] in edges:
                    sd.append(edges[t["file"]]["sd_ratio"]); ab.append(float(t["above"]))
        sd, ab = np.array(sd), np.array(ab)
        if len(sd) < 4 or ab.sum() in (0, len(ab)):
            print(f"  {pname:<10} n={len(sd)}  escapes={int(ab.sum())}/{len(ab)}  (degenerate)")
            summary[pname] = dict(n=len(sd), escapes=int(ab.sum()))
            continue
        rho, p = spearmanr(sd, ab)
        u = mannwhitneyu(sd[ab == 1], sd[ab == 0], alternative="greater")
        print(f"  {pname:<10} n={len(sd)}  J-blind escapes {int(ab.sum())}/{len(ab)}   "
              f"Spearman(edge, escape) = {rho:+.3f} (p={p:.2g})   "
              f"median edge  escaped {np.median(sd[ab==1]):.3f}  vs  not {np.median(sd[ab==0]):.3f}"
              f"  (MWU p={u.pvalue:.2g})")
        summary[pname] = dict(n=len(sd), escapes=int(ab.sum()), spearman_rho=float(rho),
                              spearman_p=float(p), mwu_p=float(u.pvalue),
                              median_edge_escaped=float(np.median(sd[ab == 1])),
                              median_edge_not=float(np.median(sd[ab == 0])))

    json.dump({"cells": res, "edges": edges, "edge_vs_escape": summary},
              open("matched_invariant.json", "w"), indent=1, default=float)
    print("\nsaved -> matched_invariant.json")
