"""PREREGISTERED FRESH-TEMPLATE REPLICATION: registered analysis (prereg/PREREG_transport_replication.md, v3 FINAL).

Implements, and only implements, the registered analysis:
  capture   RMS_T (logits) over the 256 joint states of the executed four-point contrast
            T_C = Y(u,v) - Y(u,0) - Y(0,v) + Y(0,0) (fourpoint_modelfree.quartets, donor-id dose rule of the query-chart
            collector); rel = 100 RMS_T / swing (swing = ptp of the B-arm single-site readouts); V2 floor = RMS of
            (re-executed T_C - T_C); resolved iff RMS_T > 10 x floor.
  prompt    M_p (M^rel_p) = median over screened published configurations; paired cells (prompt, Name Mover):
            d = log(median published RMS_T) - log(median control RMS_T); d_p = median over the prompt's cells.
  template  V_t, V^rel_t, d_t = medians over the template's analysable prompts.
  criteria  H1_abs: >= 7/8 templates with V_t < 0.10 logits.  H1_rel: >= 7/8 with V^rel_t < 1 (%).
            H2: both families resolved under V2 (>= 90% of analysed captures resolved) AND >= 7/8 templates with d_t > 0
            AND median d_t > 0; published resolved but controls not -> FLOOR-LIMITED outcome (no ratio, no
            above-control claim); published not resolved -> H2 FAILS.  Non-analysable templates count as not meeting.
  gates     V1 (pilot_report.json), V3 identity (every capture), V4 (>= 50% published screened in and >= 5 templates),
            V5 (swing >= 0.5 logits; otherwise the capture is excluded).
  secondary S1 without T7; S2 prompt level; S3 no screen; S4 per-template table; S5 median |T_C|; S6 held-out
            single-site cubic vs zero; S7 working-model envelope coverage; S8 single-site/joint Wald rejections and
            c_S vs c_J; S9 donor reuse.  All descriptive; none changes a decision.

    python3 analyse_transport_replication.py            # -> <OUT>/analysis/analysis.json
    python3 analyse_transport_replication.py --selftest # core T_C path vs the frozen exploratory artifact (CPU)
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fourpoint_modelfree as FM          # noqa: E402  quartets(), fitted_contrasts()
import attribution_adequacy as AA         # noqa: E402  envelopes()
import detection_covariance as DC         # noqa: E402  single_site(), joint(), chart(), _fwl()

OUT = "/data/ai_models/explainability/replication_transport"
POOL = next((p for p in (os.path.join(HERE, "..", "prereg", "prompt_pool.json"),
                         os.path.join(HERE, "..", "..", "prereg", "prompt_pool.json")) if os.path.exists(p)), None)
MARGIN_ABS, MARGIN_REL, SCREEN, SWING_MIN = 0.10, 1.0, 0.15, 0.5
RESOLVE_X, RESOLVE_FRAC, NEED = 10.0, 0.90, 7
V3_TOL = 1e-12
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))


class _D(dict):
    """dict with the npz interface quartets() uses (d[key], d.files)."""
    @property
    def files(self):
        return list(self.keys())


def capture(d, rr=True):
    T, corners, a0b0, ja, jb, audit = FM.quartets(d)
    n = len(d["Zs"]) // 2
    swing = float(np.ptp(d["Ys"][n:]))
    out = dict(rms_T=rms(T), med_abs_T=float(np.median(np.abs(T))), swing=swing,
               rel=100.0 * rms(T) / swing if swing > 0 else float("inf"), std_Yjm=float(np.std(d["Yjm"])),
               audit={k: float(v) for k, v in audit.items()})
    out["v3_pass"] = all(out["audit"][k] <= V3_TOL for k in ("joint_a_vs_clamped", "joint_b_vs_nm_arm",
                                                             "clamped_b_spread", "nm_arm_a_spread"))
    if rr:
        R = _D({k: d["rr_" + k] for k in ("Zs", "Ys", "Zc", "Yc", "Zjm", "Yjm", "clean")})
        Tr, *_ = FM.quartets(R)
        out["floor"] = rms(Tr - T)
        out["resolved"] = bool(out["rms_T"] > RESOLVE_X * out["floor"])
    return out, T, corners, a0b0


def secondary_capture(d, T, corners, a0b0):
    """S6 (held-out single-site cubic vs zero), S7 (envelope coverage), S8 (Wald rejections; c_S vs c_J)."""
    s = {}
    fc, j_te, _ = FM.fitted_contrasts(d, corners, a0b0, 0)
    pred = fc[(3, "S")]
    s["S6_S3_rmse_heldout"] = rms(pred[j_te] - T[j_te]); s["S6_zero_rms_heldout"] = rms(T[j_te])
    s["S6_S3_worse_than_zero"] = bool(s["S6_S3_rmse_heldout"] > s["S6_zero_rms_heldout"])
    env, _, _ = AA.envelopes(d, 3)
    for fam, (centre, rad) in env.items():
        s[f"S7_{fam}_upper_all"] = bool(np.all(np.abs(T) <= np.abs(centre) + rad + 1e-12))
        s[f"S7_{fam}_interval_all"] = bool(np.all(np.abs(T - centre) <= rad + 1e-12))
    Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
    W = DC.chart(np.vstack([Zs, Zj]))
    cS, pS = DC.single_site(Zs, Ys, W)
    cJ, pJ, _ = DC.joint(Zj, Yj, W, (len(Zs) // 2) // 7)
    s["S8_pS"], s["S8_pJ"] = pS, pJ
    _, RS, eS, GS, XS = DC._fwl(Zs, Ys, W); _, RJ, eJ, GJ, XJ = DC._fwl(Zj, Yj, W)
    VS = float(eS @ eS) / (len(Zs) - XS.shape[1]) * np.linalg.inv(GS)
    VJ = float(eJ @ eJ) / (len(Zj) - XJ.shape[1]) * np.linalg.inv(GJ)
    dc = cS - cJ
    s["S8_cS_vs_cJ_p"] = float(chi2.sf(float(dc @ np.linalg.solve(VS + VJ, dc)), len(dc)))
    return s


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def template_values(rows, pool, templates, use_screen=True):
    """V_t, V^rel_t, d_t, med|T|_t per template from capture rows (registered aggregation)."""
    ok = lambda r: r["analysed"] if use_screen else r["analysed_noscreen"]
    out = {}
    for T in pool["admitted"]:
        k = T["k"]
        if k not in templates:
            continue
        prompts = {}
        for pr in T["prompts"]:
            pub = [r for r in rows if r["template"] == k and r["seed"] == pr["seed"] and r["group"] == "published" and ok(r)]
            ctl = [r for r in rows if r["template"] == k and r["seed"] == pr["seed"] and r["group"] == "control" and ok(r)]
            if not pub:
                continue
            cells = []
            for nm in {tuple(r["target"]) for r in pub}:
                P = [r["rms_T"] for r in pub if tuple(r["target"]) == nm]
                C = [r["rms_T"] for r in ctl if tuple(r["target"]) == nm]
                if P and C and np.median(C) > 0:
                    cells.append(float(np.log(np.median(P)) - np.log(np.median(C))))
            prompts[pr["seed"]] = dict(M=med([r["rms_T"] for r in pub]), Mrel=med([r["rel"] for r in pub]),
                                       Mabs=med([r["med_abs_T"] for r in pub]), d=med(cells) if cells else None,
                                       n_pub=len(pub), n_ctl=len(ctl))
        out[k] = dict(analysable=bool(prompts), prompts=prompts,
                      V=med([p["M"] for p in prompts.values()]), Vrel=med([p["Mrel"] for p in prompts.values()]),
                      Vabs=med([p["Mabs"] for p in prompts.values()]),
                      d=med([p["d"] for p in prompts.values() if p["d"] is not None]))
    return out


def criteria(tv, n_total):
    vals = list(tv.values())
    k_abs = sum(1 for t in vals if t["V"] is not None and t["V"] < MARGIN_ABS)
    k_rel = sum(1 for t in vals if t["Vrel"] is not None and t["Vrel"] < MARGIN_REL)
    k_pos = sum(1 for t in vals if t["d"] is not None and t["d"] > 0)
    md = med([t["d"] for t in vals])
    need = n_total - (8 - NEED)          # 7 of 8; 6 of 7 when a registered subset of 7 is summarised (S1)
    return dict(n_templates=n_total, k_below_abs=k_abs, k_below_rel=k_rel, k_d_positive=k_pos, median_d=md,
                ratio_exp_median_d=float(np.exp(md)) if md is not None else None,
                H1_abs=k_abs >= need, H1_rel=k_rel >= need, H2_counts=(k_pos >= need and md is not None and md > 0))


def analyse():
    pool = json.load(open(POOL))
    rep = json.load(open(os.path.join(OUT, "pilot", "pilot_report.json")))
    man = json.load(open(os.path.join(OUT, "fresh", "manifest.json")))
    rows = []
    for m in man["rows"]:
        r = dict(m)
        r["analysed"] = r["analysed_noscreen"] = False
        if m["error"] is None:
            d = np.load(os.path.join(OUT, "fresh", m["file"]))
            c, T, corners, a0b0 = capture(d)
            r.update(c)
            r["v5_pass"] = bool(c["swing"] >= SWING_MIN)
            r["screened_in"] = bool(c["std_Yjm"] >= SCREEN)
            r["analysed"] = r["screened_in"] and r["v5_pass"]
            r["analysed_noscreen"] = r["v5_pass"]
            r["secondary"] = secondary_capture(d, T, corners, a0b0) if r["analysed"] else None
        rows.append(r)
    ok = [r for r in rows if r["error"] is None]
    pub_all = [r for r in rows if r["group"] == "published"]
    fams = {g: [r for r in rows if r["group"] == g and r["analysed"]] for g in ("published", "control")}
    v2 = {g: dict(n=len(v), resolved=sum(r["resolved"] for r in v),
                  frac=(sum(r["resolved"] for r in v) / len(v)) if v else 0.0,
                  median_floor=med([r["floor"] for r in v]), median_rms=med([r["rms_T"] for r in v]))
          for g, v in fams.items()}
    for g in v2:
        v2[g]["family_resolved"] = bool(v2[g]["n"] > 0 and v2[g]["frac"] >= RESOLVE_FRAC)
    tv = template_values(rows, pool, {T["k"] for T in pool["admitted"]})
    n_analysable_t = sum(1 for t in tv.values() if t["analysable"])
    gates = dict(V1=bool(rep.get("v1_pass") and rep.get("rerun_equivalent")),
                 V3=all(r["v3_pass"] for r in ok),
                 V4=bool(sum(r["analysed"] for r in pub_all) >= 0.5 * len(pub_all) and n_analysable_t >= 5),
                 V5_excluded=[r["file"] for r in ok if not r["v5_pass"]],
                 errors=[(r["file"], r["error"]) for r in rows if r["error"]],
                 screened_out=[(r["file"], r["std_Yjm"]) for r in ok if not r["screened_in"]])
    C = criteria(tv, 8)
    if not v2["published"]["family_resolved"]:
        h2 = "FAIL: published deviations not numerically resolved under V2"
    elif not v2["control"]["family_resolved"]:
        h2 = ("FLOOR-LIMITED: published deviations resolved; control deviations at the numerical floor "
              "(no calibrated published/control ratio; no formal above-control result)")
    elif C["H2_counts"]:
        h2 = "PASS: published above control in >= 7/8 templates and median log-ratio > 0; both families resolved"
    else:
        h2 = "FAIL: the 7/8 count or the positive-median condition is not met"
    valid = gates["V1"] and gates["V3"] and gates["V4"]
    upgrade = bool(valid and C["H1_abs"] and C["H1_rel"] and h2.startswith("PASS"))
    t7 = {k: v for k, v in tv.items() if k != 7}
    sec = dict(
        S1_without_T7=criteria(t7, 7),
        S2_prompt_level=[dict(template=k, seed=s, **p) for k, t in tv.items() for s, p in t["prompts"].items()],
        S3_no_screen=criteria(template_values(rows, pool, {T["k"] for T in pool["admitted"]}, use_screen=False), 8),
        S5_template_median_abs_T={k: t["Vabs"] for k, t in tv.items()},
        S6_published_S3_worse_than_zero=f"{sum(r['secondary']['S6_S3_worse_than_zero'] for r in fams['published'])}/{len(fams['published'])}",
        S7_published_upper_all={f: f"{sum(r['secondary'][f'S7_{f}_upper_all'] for r in fams['published'])}/{len(fams['published'])}" for f in ("S", "J")},
        S7_published_interval_all={f: f"{sum(r['secondary'][f'S7_{f}_interval_all'] for r in fams['published'])}/{len(fams['published'])}" for f in ("S", "J")},
        S8={g: dict(rej_S={e: sum(r["secondary"]["S8_pS"][e] < 0.05 for r in v) for e in ("homo", "hc0", "hc3", "arm", "fgls", "cr1")},
                    rej_J={e: sum(r["secondary"]["S8_pJ"][e] < 0.05 for r in v) for e in ("homo", "hc0", "cr2way")},
                    cS_vs_cJ_inconsistent=sum(r["secondary"]["S8_cS_vs_cJ_p"] < 0.05 for r in v), n=len(v))
            for g, v in fams.items()},
        S9_donor_reuse={T["k"]: [sum(1 for pair in pr["donor_pairs"]
                                     if any(tuple(pair) in {tuple(x) for x in q["donor_pairs"]}
                                            for q in T["prompts"] if q is not pr))
                                 for pr in T["prompts"]] for T in pool["admitted"]})
    res = dict(prereg="v3 FINAL", gates=gates, V2=v2, templates={k: dict(V=t["V"], Vrel=t["Vrel"], d=t["d"], analysable=t["analysable"])
                                                                   for k, t in tv.items()},
               criteria=C, H2=h2, validity_pass=valid, upgrade_to_preregistered_result=upgrade,
               medians=dict(V=med([t["V"] for t in tv.values()]), Vrel=med([t["Vrel"] for t in tv.values()]),
                            d=C["median_d"]),
               secondary=sec, rows=[{k: v for k, v in r.items() if k != "secondary"} for r in rows])
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "analysis", "analysis.json"), "w"), indent=1, default=float)
    print(f"[ANALYSIS] gates V1 {gates['V1']} V3 {gates['V3']} V4 {gates['V4']}; errors {len(gates['errors'])}; "
          f"screened out {len(gates['screened_out'])}; V5 excluded {len(gates['V5_excluded'])}")
    for k, t in sorted(tv.items()):
        print(f"[ANALYSIS] T{k}: V {t['V']}  Vrel {t['Vrel']}  d {t['d']}  analysable {t['analysable']}")
    print(f"[ANALYSIS] H1_abs {C['H1_abs']} ({C['k_below_abs']}/8)  H1_rel {C['H1_rel']} ({C['k_below_rel']}/8)  "
          f"d>0 {C['k_d_positive']}/8 median d {C['median_d']}")
    print(f"[ANALYSIS] V2 published {v2['published']}  control {v2['control']}")
    print(f"[ANALYSIS] H2: {h2}")
    print(f"[ANALYSIS] upgrade to preregistered result: {upgrade}")


def selftest():
    """Core path on the stored exploratory captures (rerun absent): RMS_T must equal attribution_adequacy.json."""
    A = json.load(open(os.path.join(HERE, "attribution_adequacy.json")))
    ref = {r["file"]: r for r in A["groups"]["published"]["rows"]}
    worst = 0.0
    for f in sorted(glob.glob(os.path.join(HERE, "qchart_*.npz"))):
        c, T, corners, a0b0 = capture(np.load(f), rr=False)
        r = ref[os.path.basename(f)]
        worst = max(worst, abs(c["rms_T"] - r["rms_T"]), abs(c["swing"] - r["swing"]))
        assert c["v3_pass"], (f, c["audit"])
    d = _D({k: v for k, v in np.load(sorted(glob.glob(os.path.join(HERE, "qchart_*.npz")))[0]).items()})
    for k in ("Zs", "Ys", "Zc", "Yc", "Zjm", "Yjm", "clean"):
        d["rr_" + k] = d[k]
    c0, *_ = capture(d)
    assert c0["floor"] == 0.0 and c0["resolved"], c0
    print(f"SELFTEST OK: 31 captures, max |diff| vs attribution_adequacy {worst:.2e}; V3 pass; zero floor for identical rerun")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    selftest() if A.selftest else analyse()
