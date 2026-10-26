"""THE IDENTIFICATION GAP AS A FUNCTION OF THE REALISED EDGE  (added 2026-08-28).

Section 7 measures, in charts fixed a priori from parameters, how much information each
intervention family carries about the interaction between two heads.  Three head groups are run
with the identical procedure:

    IOI        S-Inhibition -> Name Mover     (Wang et al.: query-mediated)
    induction  previous-token -> induction    (Elhage/Olsson: query-mediated)
    control    early heads -> Name Mover      (no claimed mechanism)

The framework predicts the gap is set by the REACHABLE GEOMETRY, not by which heads they are: a
flatter realised edge makes the single-site design more degenerate, so the gap should grow as the
edge shrinks -- across all three groups alike.  That is a quantitative, falsifiable prediction and
this script tests it.
"""
import json
import numpy as np
from scipy.stats import spearmanr

GROUPS = {"ioi": "ioi_query_chart.json",
          "induction": "ioi_query_chart_induction.json",
          "control": "ioi_query_chart_control.json"}

if __name__ == "__main__":
    rows, per = [], {}
    for name, f in GROUPS.items():
        R = json.load(open(f))["rows"]
        rows += R
        med = lambda k: float(np.median([r[k] for r in R]))
        per[name] = dict(n=len(R), edge=med("sd_ratio"), edge_r2=med("edge_r2"),
                         info_ratio=med("info_ratio"), se_ratio=med("se_ratio"),
                         info_S=med("info_min_S"), info_J=med("info_min_J"),
                         rej_J=int(sum(r["wald_p_J"] < 0.05 for r in R)),
                         rej_S=int(sum(r["wald_p_S"] < 0.05 for r in R)),
                         # |c|/se under JOINT evidence: is the interaction the joint family
                         # measures actually nonzero?  ~1 means "precisely measured, and zero".
                         t_J=float(np.median([abs(r["coef_J"]) / max(r["se_J"], 1e-300)
                                              for r in R])))
    sd = np.array([r["sd_ratio"] for r in rows])
    try:
        nullsz = json.load(open("additive_null.json"))
    except FileNotFoundError:
        nullsz = {}
    out = dict(groups=per, n=len(rows), additive_null=nullsz)
    print(f"{'head group':<12}{'n':>4}{'edge':>10}{'info gap':>12}{'se ratio':>11}"
          f"{'joint rej':>11}{'single rej':>12}")
    print("-" * 72)
    for k, v in per.items():
        print(f"{k:<12}{v['n']:>4}{v['edge']:>10.4f}{v['info_ratio']:>12.2e}"
              f"{v['se_ratio']:>11.1f}{v['rej_J']:>8}/{v['n']:<3}{v['rej_S']:>9}/{v['n']:<3}")
    print(f"\npooled over {len(rows)} configurations, three independent head groups:")
    for key, lab in (("info_ratio", "information gap J/S"), ("se_ratio", "standard-error ratio")):
        rho, p = spearmanr(sd, [r[key] for r in rows])
        out[f"rho_{key}"], out[f"p_{key}"] = float(rho), float(p)
        print(f"  Spearman(realised edge, {lab:<22}) = {rho:+.3f}   p = {p:.2g}")
    rho, p = spearmanr(sd, [float(r["wald_p_S"] < 0.05) for r in rows])
    out["rho_detect"], out["p_detect"] = float(rho), float(p)
    print(f"  Spearman(realised edge, {'single-site detects':<22}) = {rho:+.3f}   p = {p:.2g}")
    json.dump(out, open("circuits_summary.json", "w"), indent=1, default=float)
    print("\nsaved -> circuits_summary.json")
