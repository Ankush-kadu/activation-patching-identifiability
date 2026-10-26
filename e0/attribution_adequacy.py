"""POST-REVIEW ADEQUACY OF THE IOI ATTRIBUTION ANALYSIS (arXiv version of the paper).  EXPLORATORY.

STATUS.  Exploratory and post hoc: every capture analysed here was inspected before this file was written
(three base prompts; 31 published S-Inhibition -> Name Mover configurations, 28 controls, 18 induction
pairs).  Nothing here is confirmatory.  It writes a NEW artifact, attribution_adequacy.json, and changes no
existing artifact.

WHAT IT ANSWERS.  The attribution analysis reports, per capture, the median over the 256 support-matched
joint states of a 95% Wald worst case for the four-point transport contrast, under a cubic working model
fitted either to single-site (S) or to joint (J) interventions (misattribution_fourpoint.py).  Reviewers
asked whether that working model describes the network.  The executed four-point contrast
T_C = Y(u,v) - Y(u,0) - Y(0,v) + Y(0,0) is available at every joint state (fourpoint_modelfree.quartets),
and the fitted model's contrast at the same corners is exactly D c (the additive terms cancel), so the
envelope can be checked state by state against the network itself.  This file computes:

  1. reproduction: per-capture med_S / med_J equal misattribution_fourpoint.json to 1e-12 (asserted);
  2. coverage of the executed T_C by each envelope, degrees 2/3/4, under two criteria:
       upper-bound  |T_C| <= |D c| + radius      (the criterion of the synthetic self-test)
       interval     |T_C - D c| <= radius        (the Wald region contains the executed contrast)
     reported per state (share covered) and per capture (all 256 states covered);
  3. absolute sizes in logits: envelope median, its centre |D c| and radius, and the executed T_C
     (RMS and median |T_C|), plus each relative to the B-arm swing ptp(Ys[B arm]);
  4. the swing against a typical per-donor B effect at alpha ~ 1 (mean of the 5/6 and 7/6 doses);
  5. covariance sensitivity of the degree-3 envelope: HC0 and donor-clustered (CR1) for S, HC0 and
     two-way donor-clustered (V_up + V_down, as detection_covariance.joint) for J;
  6. per-prompt (seed) medians, and a paired published-vs-control comparison of RMS T_C within each
     (Name Mover head, seed) cell, where published and control configurations share the B arm;
  7. between-prompt dispersion of log RMS T_C (input to a replication sample-size calculation);
  8. the QC summary block of ioi_query_chart.json rebuilt from its rows (no shipped script writes it);
  9. single-site and joint interaction-test rejections for the induction group under the covariance
     estimators of detection_covariance.py (that analysis covered only the IOI and control groups).

NOT MEASURABLE HERE.  The float32 numerical floor of T_C: the four corners were executed in different
batches and never re-executed, so no floor estimate exists (recorded as null).

    python3 attribution_adequacy.py      # -> attribution_adequacy.json
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import misattribution as M                    # noqa: E402
import misattribution_fourpoint as F          # noqa: E402
import fourpoint_modelfree as FM              # noqa: E402
import response_diagnostics as RD             # noqa: E402
import detection_covariance as DC             # noqa: E402
from cells import N_AL, ALPHAS                # noqa: E402

STATUS = ("EXPLORATORY: post-review analysis on previously inspected captures (three base prompts); "
          "no confirmatory claim")
DEGREES = (2, 3, 4)
TOL = 1e-12
GROUPS = {"published": "qchart_*.npz", "controls": "qctrl_*.npz", "induction": "qind_*.npz"}
K_5_6, K_7_6 = ALPHAS.index(5 / 6), ALPHAS.index(7 / 6)


def label(f):
    """(source head tag, Name Mover / target tag, seed) from names such as qchart_810_100_2.npz."""
    _, a, b, seed = os.path.basename(f)[:-4].split("_")
    return a, b, int(seed)


def chart(Zs, Zj):
    allZ = np.vstack([Zs, Zj])
    mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    return lambda Z: (Z - mu) / sd


def envelopes(d, deg):
    """Signed centre D c and radius per joint state, for S and J, under the degree-deg class
    (exactly misattribution_fourpoint.one's construction)."""
    Zs, Ys, Zj, Yj, Zc = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"], d["Zc"]
    W = chart(Zs, Zj)
    n = len(Zs) // 2
    a0, b0 = float(np.median(Zs[n:, 0])), float(np.median(Zc[:n, 1]))
    out = {}
    with RD.working_class(deg):
        D = F.four_point(W, Zj, a0, b0)
        for fam, (Z, Y) in (("S", (Zs, Ys)), ("J", (Zj, Yj))):
            c, V = M.fit(Z, Y, W)
            centre = D @ c
            rad = np.sqrt(np.maximum(M.CRIT * np.einsum("ij,jk,ik->i", D, V, D), 0.0))
            out[fam] = (centre, rad)
    return out, (a0, b0), D


def cov_sensitivity(d, D, W):
    """Degree-3 envelope under alternative covariances (medians over states, logits)."""
    Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
    res = {}
    n = len(Zs); half = n // 2
    c, R, e, G, X = DC._fwl(Zs, Ys, W)
    p = X.shape[1]
    donor = (np.arange(n) % half) // N_AL
    ng = len(np.unique(donor)); fac = ng / (ng - 1) * (n - 1) / (n - p)
    Vs = {"homo": (float(e @ e) / (n - p)) * np.linalg.inv(G),
          "hc0": DC._sandwich(R, G, (R * (e ** 2)[:, None]).T @ R),
          "cr1_donor": fac * DC._sandwich(R, G, DC._cluster_meat(R, e, donor))}
    for k, V in Vs.items():
        res[f"S_{k}"] = float(np.median(np.abs(D @ c) + np.sqrt(np.maximum(M.CRIT * np.einsum("ij,jk,ik->i", D, V, D), 0))))
    n_donor = half // N_AL
    cJ, oJ, V2 = DC.joint(Zj, Yj, W, n_donor)
    cJ2, RJ, eJ, GJ, XJ = DC._fwl(Zj, Yj, W)
    VJ = {"homo": (float(eJ @ eJ) / (len(Zj) - XJ.shape[1])) * np.linalg.inv(GJ),
          "hc0": DC._sandwich(RJ, GJ, (RJ * (eJ ** 2)[:, None]).T @ RJ), "cr2way": V2}
    for k, V in VJ.items():
        res[f"J_{k}"] = float(np.median(np.abs(D @ cJ) + np.sqrt(np.maximum(M.CRIT * np.einsum("ij,jk,ik->i", D, V, D), 0))))
    return res


def one(f, group, ref_rows):
    d = np.load(f)
    Zs, Ys = d["Zs"], d["Ys"]
    n = len(Zs) // 2
    swing = float(np.ptp(Ys[n:]))
    T, corners, a0b0_q, ja, jb, audit = FM.quartets(d)
    si, nm, seed = label(f)
    r = dict(file=os.path.basename(f), group=group, source=si, target=nm, seed=seed, swing=swing,
             rms_T=float(np.sqrt(np.mean(T ** 2))), med_abs_T=float(np.median(np.abs(T))),
             max_abs_T=float(np.max(np.abs(T))), audit=audit)
    r["rms_T_pct_swing"] = 100 * r["rms_T"] / swing
    r["med_abs_T_pct_swing"] = 100 * r["med_abs_T"] / swing
    # typical per-donor B effect at alpha ~ 1: mean of the 5/6 and 7/6 doses on the B arm, minus clean
    y0 = float(d["clean"])
    n_don = n // N_AL
    eff = np.array([0.5 * (Ys[n + j * N_AL + K_5_6] + Ys[n + j * N_AL + K_7_6]) - y0 for j in range(n_don)])
    r["mean_abs_effect_alpha1"] = float(np.mean(np.abs(eff)))
    r["swing_over_effect"] = swing / r["mean_abs_effect_alpha1"] if r["mean_abs_effect_alpha1"] > 0 else None
    for deg in DEGREES:
        env, a0b0, D = envelopes(d, deg)
        assert np.allclose(a0b0, a0b0_q, rtol=0, atol=1e-12), (f, a0b0, a0b0_q)
        for fam, (centre, rad) in env.items():
            w = np.abs(centre) + rad
            key = f"d{deg}_{fam}"
            r[key] = dict(
                env_med=float(np.median(w)), env_med_pct_swing=100 * float(np.median(w)) / swing,
                centre_med=float(np.median(np.abs(centre))), radius_med=float(np.median(rad)),
                cover_upper_share=float(np.mean(np.abs(T) <= w + TOL)),
                cover_interval_share=float(np.mean(np.abs(T - centre) <= rad + TOL)))
            r[key]["cover_upper_all"] = bool(r[key]["cover_upper_share"] == 1.0)
            r[key]["cover_interval_all"] = bool(r[key]["cover_interval_share"] == 1.0)
        if deg == 3:
            if ref_rows is not None:
                ref = ref_rows[os.path.basename(f)]
                for fam in ("S", "J"):
                    got = r[f"d3_{fam}"]["env_med"] / swing
                    assert abs(got - ref[f"med_{fam}"]) <= 1e-12 * max(1.0, abs(ref[f"med_{fam}"])), \
                        (f, fam, got, ref[f"med_{fam}"])
                r["reproduces_misattribution_fourpoint"] = True
            W = chart(Zs, d["Zjm"])
            r["cov_sensitivity_d3"] = cov_sensitivity(d, D, W)
    return r


def qs(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return dict(n=int(len(x)), median=float(np.median(x)), q10=float(np.percentile(x, 10)),
                q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                q90=float(np.percentile(x, 90)), min=float(np.min(x)), max=float(np.max(x)))


def summarise(rows):
    g = lambda k: [r[k] for r in rows]
    s = dict(n=len(rows), rms_T=qs(g("rms_T")), rms_T_pct_swing=qs(g("rms_T_pct_swing")),
             med_abs_T=qs(g("med_abs_T")), med_abs_T_pct_swing=qs(g("med_abs_T_pct_swing")),
             max_abs_T=qs(g("max_abs_T")), swing=qs(g("swing")),
             mean_abs_effect_alpha1=qs(g("mean_abs_effect_alpha1")), swing_over_effect=qs(g("swing_over_effect")),
             swing_over_effect_ratio_of_medians=float(np.median(g("swing")) / np.median(g("mean_abs_effect_alpha1"))))
    for deg in DEGREES:
        for fam in ("S", "J"):
            k = f"d{deg}_{fam}"
            s[k] = dict(env_med=qs([r[k]["env_med"] for r in rows]),
                        env_med_pct_swing=qs([r[k]["env_med_pct_swing"] for r in rows]),
                        centre_med=qs([r[k]["centre_med"] for r in rows]),
                        radius_med=qs([r[k]["radius_med"] for r in rows]),
                        cover_upper_share=qs([r[k]["cover_upper_share"] for r in rows]),
                        cover_interval_share=qs([r[k]["cover_interval_share"] for r in rows]),
                        cover_upper_all=int(sum(r[k]["cover_upper_all"] for r in rows)),
                        cover_interval_all=int(sum(r[k]["cover_interval_all"] for r in rows)))
    cs = [r["cov_sensitivity_d3"] for r in rows]
    s["cov_sensitivity_d3_pct_swing"] = {k: float(np.median([100 * c[k] / r["swing"] for c, r in zip(cs, rows)]))
                                         for k in cs[0]}
    by_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        rr = [r for r in rows if r["seed"] == seed]
        by_seed[str(seed)] = dict(n=len(rr),
                                  env_S_pct_swing=float(np.median([r["d3_S"]["env_med_pct_swing"] for r in rr])),
                                  env_J_pct_swing=float(np.median([r["d3_J"]["env_med_pct_swing"] for r in rr])),
                                  rms_T=float(np.median([r["rms_T"] for r in rr])),
                                  rms_T_pct_swing=float(np.median([r["rms_T_pct_swing"] for r in rr])),
                                  env_S_over_swing_gt1=int(sum(r["d3_S"]["env_med_pct_swing"] > 100 for r in rr)))
    s["by_seed"] = by_seed
    return s


def paired(pub, ctl):
    """Within each (target head, seed) cell: median RMS T_C of published vs control configurations."""
    cells = {}
    for key in sorted({(r["target"], r["seed"]) for r in pub} & {(r["target"], r["seed"]) for r in ctl}):
        P = [r for r in pub if (r["target"], r["seed"]) == key]
        C = [r for r in ctl if (r["target"], r["seed"]) == key]
        sw = {round(r["swing"], 12) for r in P + C}
        cells[f"{key[0]}_{key[1]}"] = dict(n_pub=len(P), n_ctl=len(C),
                                           rms_T_pub=float(np.median([r["rms_T"] for r in P])),
                                           rms_T_ctl=float(np.median([r["rms_T"] for r in C])),
                                           shared_swing=len(sw) == 1)
        cells[f"{key[0]}_{key[1]}"]["ratio"] = cells[f"{key[0]}_{key[1]}"]["rms_T_pub"] / cells[f"{key[0]}_{key[1]}"]["rms_T_ctl"]
    ratios = [c["ratio"] for c in cells.values()]
    return dict(cells=cells, n_cells=len(cells), ratio=qs(ratios), cells_pub_above_ctl=int(sum(x > 1 for x in ratios)),
                all_cells_share_swing=bool(all(c["shared_swing"] for c in cells.values())))


def dispersion(pub):
    """Between- and within-prompt spread of log RMS T_C (published), for a replication sample size."""
    lr = {s: np.log([r["rms_T"] for r in pub if r["seed"] == s]) for s in sorted({r["seed"] for r in pub})}
    seed_medians = np.array([np.median(v) for v in lr.values()])
    within = np.concatenate([v - np.mean(v) for v in lr.values()])
    return dict(n_prompts=len(lr), per_prompt_median_log_rms_T={str(s): float(np.median(v)) for s, v in lr.items()},
                between_prompt_sd_of_medians=float(np.std(seed_medians, ddof=1)),
                within_prompt_sd=float(np.sqrt(np.sum(within ** 2) / (len(within) - len(lr)))))


def qc_summary_rebuild():
    Q = json.load(open(os.path.join(HERE, "ioi_query_chart.json")))
    rows, stored = Q["rows"], Q["summary"]
    med = lambda k: float(np.median([r[k] for r in rows]))
    rebuilt = dict(n=len(rows), info_S=med("info_min_S"), info_J=med("info_min_J"), info_ratio=med("info_ratio"),
                   se_ratio=med("se_ratio"), wald_rej_J=int(sum(r["wald_p_J"] < 0.05 for r in rows)),
                   wald_rej_S=int(sum(r["wald_p_S"] < 0.05 for r in rows)), signflip_S=med("signflip_S"),
                   signflip_J=med("signflip_J"), edge_r2=med("edge_r2"), sd_ratio=med("sd_ratio"))
    match = {k: (bool(np.isclose(rebuilt[k], stored[k], rtol=1e-12, atol=0)) if k in stored else None) for k in rebuilt}
    return dict(rebuilt=rebuilt, stored=stored, all_match=all(v for v in match.values() if v is not None), match=match)


def induction_detection():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, GROUPS["induction"]))):
        d = np.load(f); Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
        W = DC.chart(np.vstack([Zs, Zj]))
        _, pS = DC.single_site(Zs, Ys, W)
        _, pJ, _ = DC.joint(Zj, Yj, W, (len(Zs) // 2) // N_AL)
        rows.append(dict(file=os.path.basename(f), p_S=pS, p_J=pJ))
    cnt = lambda fam, est: int(sum(r[f"p_{fam}"][est] < DC.ALPHA for r in rows))
    return dict(n=len(rows), rej_S={e: cnt("S", e) for e in ("homo", "hc0", "hc3", "arm", "fgls", "cr1")},
                rej_J={e: cnt("J", e) for e in ("homo", "hc0", "cr2way")}, rows=rows)


def main():
    ref = json.load(open(os.path.join(HERE, "misattribution_fourpoint.json")))
    ref_rows = {os.path.basename(r["file"]): r for r in ref["rows"]}
    out = dict(status=STATUS, degrees=list(DEGREES), numerical_floor=None,
               numerical_floor_note="not measurable from existing captures (corners never re-executed)",
               groups={})
    R = {}
    for group in ("published", "controls"):
        files = sorted(glob.glob(os.path.join(HERE, GROUPS[group])))
        R[group] = [one(f, group, ref_rows if group == "published" else None) for f in files]
        out["groups"][group] = dict(summary=summarise(R[group]), rows=R[group])
    assert len(R["published"]) == ref["n"] == 31 and all(r.get("reproduces_misattribution_fourpoint") for r in R["published"])
    ind = [one(f, "induction", None) for f in sorted(glob.glob(os.path.join(HERE, GROUPS["induction"])))]
    out["groups"]["induction"] = dict(summary=dict(n=len(ind), rms_T=qs([r["rms_T"] for r in ind]),
                                                   rms_T_pct_swing=qs([r["rms_T_pct_swing"] for r in ind]),
                                                   d3_S_env_pct_swing=qs([r["d3_S"]["env_med_pct_swing"] for r in ind]),
                                                   d3_J_env_pct_swing=qs([r["d3_J"]["env_med_pct_swing"] for r in ind])),
                                      rows=ind)
    out["paired_published_vs_control"] = paired(R["published"], R["controls"])
    out["dispersion_published"] = dispersion(R["published"])
    out["qc_summary"] = qc_summary_rebuild()
    out["induction_detection"] = induction_detection()
    # sanity: the IOI/control robust counts must reproduce detection_covariance.json
    dcj = json.load(open(os.path.join(HERE, "detection_covariance.json")))
    out["checks"] = dict(
        reproduces_misattribution_fourpoint_rows=True,
        detection_covariance_ioi_rej_S=dcj["ioi"]["summary"]["rej_S"],
        detection_covariance_control_rej_S=dcj["control"]["summary"]["rej_S"],
        qc_summary_all_match=out["qc_summary"]["all_match"])
    json.dump(out, open(os.path.join(HERE, "attribution_adequacy.json"), "w"), indent=1, default=float)
    P = out["groups"]["published"]["summary"]
    print(f"[ADEQ] published n={P['n']}  rms T_C median {P['rms_T']['median']:.4g} logits "
          f"({P['rms_T_pct_swing']['median']:.3g}% swing); controls {out['groups']['controls']['summary']['rms_T']['median']:.3g}")
    for deg in DEGREES:
        for fam in ("S", "J"):
            k = P[f"d{deg}_{fam}"]
            print(f"[ADEQ] d={deg} {fam}: env {k['env_med']['median']:.4g} logits ({k['env_med_pct_swing']['median']:.3g}% swing); "
                  f"all-state coverage upper {k['cover_upper_all']}/{P['n']} interval {k['cover_interval_all']}/{P['n']}; "
                  f"per-state share upper {k['cover_upper_share']['median']:.3f} interval {k['cover_interval_share']['median']:.3f}")
    print(f"[ADEQ] paired cells: {out['paired_published_vs_control']['cells_pub_above_ctl']}/"
          f"{out['paired_published_vs_control']['n_cells']} published>control; ratio median "
          f"{out['paired_published_vs_control']['ratio']['median']:.3g}")
    print(f"[ADEQ] QC summary rebuilt matches stored: {out['qc_summary']['all_match']}; "
          f"induction rej_S {out['induction_detection']['rej_S']} rej_J {out['induction_detection']['rej_J']}")
    print("saved -> attribution_adequacy.json")


if __name__ == "__main__":
    main()
