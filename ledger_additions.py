"""Ledger keys added for the arXiv version of Paper A.  Called at the end of ledger_paperA.build().

Rules: this module only ADDS keys; it never changes the value or macro of a key the frozen ledger already defines
(extend() asserts the macros).  Existing keys may receive documentation or provenance fixes only (note, unit,
aggregation, artifact list), each marked as such below.  Every value is derived here from a hashed artifact in e0/,
exactly like the frozen ledger's own entries.
"""
import numpy as np


def extend(add, load_json, L):
    before = {k: v["macro"] for k, v in L.items()}

    # ---- one-at-a-time (clamped) baseline: the full 2x2 set partition, per protocol ----
    # ofat_baseline.json -> pooled.{row,donor}: gR_gt10, gC_gt10, gR_gt10_gC_le10, gC_gt10_gR_le10, n.
    OB = load_json("ofat_baseline.json")
    src = ["ofat_baseline.json", "ofat_baseline.py"]
    for proto, tag in (("row", ""), ("donor", "Donor")):
        p = OB["pooled"][proto]
        n, r_only, c_only = p["n"], p["gR_gt10_gC_le10"], p["gC_gt10_gR_le10"]
        both = p["gR_gt10"] - r_only
        neither = n - both - r_only - c_only
        assert both + c_only == p["gC_gt10"] and neither >= 0, (proto, p)
        note = f"{proto} protocol; recompute generator g_R vs clamped generator g_C, each >10x"
        add(f"ofat.{proto}.both", f"Ofat{tag}Both", f"{both}/{n}", src, "capture", "count", "both >10x; " + note)
        add(f"ofat.{proto}.onlyR", f"Ofat{tag}OnlyRecompute", f"{r_only}/{n}", src, "capture", "count",
            "g_R >10x and g_C <=10x; " + note)
        add(f"ofat.{proto}.onlyC", f"Ofat{tag}OnlyClampedCount", f"{c_only}/{n}", src, "capture", "count",
            "g_C >10x and g_R <=10x; " + note)
        add(f"ofat.{proto}.neither", f"Ofat{tag}Neither", f"{neither}/{n}", src, "capture", "count",
            "neither >10x; " + note)
        add(f"ofat.{proto}.net", f"Ofat{tag}NetDifference", f"{p['gR_gt10'] - p['gC_gt10']}", src, "capture",
            "difference of counts", "g_R count minus g_C count (= only-recompute minus only-clamped); " + note)

    # ---- post-review adequacy of the IOI attribution analysis (EXPLORATORY; attribution_adequacy.py) ----
    AQ = load_json("attribution_adequacy.json")
    assert AQ["status"].startswith("EXPLORATORY")
    src = ["attribution_adequacy.json", "attribution_adequacy.py", "qchart_*.npz", "qctrl_*.npz"]
    P, C = AQ["groups"]["published"]["summary"], AQ["groups"]["controls"]["summary"]
    n = P["n"]
    ex = "EXPLORATORY, post-review, three base prompts"
    num = {2: "Two", 3: "", 4: "Four"}
    for deg in (2, 3, 4):
        for fam in ("S", "J"):
            k = P[f"d{deg}_{fam}"]
            add(f"adq.d{deg}.{fam}.upper_all", f"AdqCoverUpper{fam}{num[deg]}", f"{k['cover_upper_all']}/{n}", src,
                "capture", "count", f"all 256 joint states satisfy |T_C| <= |Dc| + radius; degree {deg}; {ex}")
            add(f"adq.d{deg}.{fam}.interval_all", f"AdqCoverInterval{fam}{num[deg]}", f"{k['cover_interval_all']}/{n}",
                src, "capture", "count", f"all 256 joint states satisfy |T_C - Dc| <= radius; degree {deg}; {ex}")
    for fam in ("S", "J"):
        k = P[f"d3_{fam}"]
        add(f"adq.{fam}.upper_share", f"AdqShareUpper{fam}", 100 * k["cover_upper_share"]["median"], src,
            "capture", "median of per-capture percent", f"percent of joint states covered, upper-bound criterion; {ex}")
        add(f"adq.{fam}.interval_share", f"AdqShareInterval{fam}", 100 * k["cover_interval_share"]["median"], src,
            "capture", "median of per-capture percent", f"percent of joint states covered, interval criterion; {ex}")
        for q, tag in (("median", ""), ("q25", "Lo"), ("q75", "Hi")):
            add(f"adq.{fam}.env.{q}", f"AdqEnv{fam}{tag}", k["env_med"][q], src, "capture", f"{q} over captures",
                f"median over joint states of the degree-3 95% Wald worst case, logits; {ex}")
        add(f"adq.{fam}.centre", f"AdqCentre{fam}", k["centre_med"]["median"], src, "capture", "median",
            f"median over states of |Dc| (point prediction of T_C), logits; {ex}")
        add(f"adq.{fam}.radius", f"AdqRadius{fam}", k["radius_med"]["median"], src, "capture", "median",
            f"median over states of the Wald radius, logits; {ex}")
    for q, tag in (("median", ""), ("q25", "Lo"), ("q75", "Hi")):
        add(f"adq.rmsT.{q}", f"AdqRmsT{tag}", P["rms_T"][q], src, "capture", f"{q} over captures",
            f"RMS of the executed four-point contrast over the 256 joint states, logits; {ex}")
        add(f"adq.rmsTpct.{q}", f"AdqRmsTPct{tag}", P["rms_T_pct_swing"][q], src, "capture", f"{q} over captures",
            f"same, percent of the B-arm swing; {ex}")
    add("adq.medabsTpct", "AdqMedAbsTPct", P["med_abs_T_pct_swing"]["median"], src, "capture", "median",
        f"median over states of |T_C|, percent of the B-arm swing; {ex}")
    add("adq.ctrl.rmsT", "AdqCtrlRmsT", C["rms_T"]["median"], src, "capture", "median",
        f"control heads, RMS executed contrast, logits; numerical floor not measured; {ex}")
    add("adq.pub_over_ctrl", "AdqPubOverCtrl", P["rms_T"]["median"] / C["rms_T"]["median"], src, "capture",
        "ratio of medians", f"descriptive; {ex}")
    PC = AQ["paired_published_vs_control"]
    add("adq.paired.cells", "AdqPairedCells", f"{PC['cells_pub_above_ctl']}/{PC['n_cells']}", src,
        "(Name Mover head, seed) cell", "count", f"published median RMS T_C above control median; {ex}")
    add("adq.paired.ratio.med", "AdqPairedRatio", PC["ratio"]["median"], src, "(Name Mover head, seed) cell",
        "median", f"published/control RMS T_C within cell; {ex}")
    add("adq.paired.ratio.min", "AdqPairedRatioMin", PC["ratio"]["min"], src, "(Name Mover head, seed) cell", "min", ex)
    add("adq.paired.ratio.max", "AdqPairedRatioMax", PC["ratio"]["max"], src, "(Name Mover head, seed) cell", "max", ex)
    add("adq.swing_over_effect", "AdqSwingOverEffect", P["swing_over_effect"]["median"], src, "capture",
        "median of per-capture ratios", f"B-arm swing / mean |per-donor B effect at alpha~1|; {ex}")
    add("adq.swing_over_effect.rom", "AdqSwingOverEffectRoM", P["swing_over_effect_ratio_of_medians"], src,
        "capture", "ratio of medians", ex)
    add("adq.effect_alpha1", "AdqEffectAlphaOne", P["mean_abs_effect_alpha1"]["median"], src, "capture", "median",
        f"mean over donors of |B effect at alpha~1|, logits; {ex}")
    CV = P["cov_sensitivity_d3_pct_swing"]
    for key, mac in (("S_hc0", "AdqEnvSHCzero"), ("S_cr1_donor", "AdqEnvSCluster"), ("J_hc0", "AdqEnvJHCzero"),
                     ("J_cr2way", "AdqEnvJTwoWay")):
        add(f"adq.cov.{key}", mac, CV[key], src, "capture", "median of per-capture percent",
            f"degree-3 envelope, percent of swing, covariance {key}; {ex}")
    words = {"0": "Zero", "1": "One", "2": "Two"}
    for s_, v in P["by_seed"].items():
        add(f"adq.seed{s_}.envS", f"AdqSeed{words[s_]}EnvS", v["env_S_pct_swing"], src, "capture", "median within prompt",
            f"single-site envelope, percent of swing; {ex}")
        add(f"adq.seed{s_}.overone", f"AdqSeed{words[s_]}OverOne", f"{v['env_S_over_swing_gt1']}/{v['n']}", src,
            "capture", "count", f"single-site envelope above the swing; {ex}")
    DP = AQ["dispersion_published"]
    add("adq.disp.between", "AdqSdBetweenPrompt", DP["between_prompt_sd_of_medians"], src, "prompt", "sd",
        f"SD over prompts of the per-prompt median log RMS T_C; {ex}")
    add("adq.disp.within", "AdqSdWithinPrompt", DP["within_prompt_sd"], src, "capture", "pooled sd",
        f"pooled within-prompt SD of log RMS T_C; {ex}")
    ID = AQ["induction_detection"]
    rob = [ID["rej_S"][e] for e in ("hc0", "hc3", "arm", "fgls", "cr1")]
    add("adq.ind.rejS.homo", "AdqIndRejSHomo", f"{ID['rej_S']['homo']}/{ID['n']}", ["attribution_adequacy.json", "qind_*.npz"],
        "configuration", "count", f"induction group, pooled homoskedastic single-site Wald; {ex}")
    add("adq.ind.rejS.lo", "AdqIndRejSLo", f"{min(rob)}", ["attribution_adequacy.json"], "configuration", "min over robust estimators", ex)
    add("adq.ind.rejS.hi", "AdqIndRejSHi", f"{max(rob)}", ["attribution_adequacy.json"], "configuration", "max over robust estimators", ex)
    add("adq.ind.rejJ.2way", "AdqIndRejJTwo", f"{ID['rej_J']['cr2way']}/{ID['n']}", ["attribution_adequacy.json"],
        "configuration", "count", f"induction group, joint Wald, two-way donor clustering; {ex}")
    assert AQ["qc_summary"]["all_match"], "QC summary rebuilt from rows does not match the stored block"

    # ---- unit-of-analysis corrections (heads, not configurations) and detection diagnostics ----
    PS = load_json("pseudorep.json")
    psrc = ["pseudorep.json", "pseudorep.py"]
    le, sp = PS["head_level"]["log_edge"], PS["spearman"]
    add("head.edge.exactp", "HeadEdgeExactP", le["mwu_exact_heads_p_greater"], psrc, "source head",
        "exact one-sided Mann-Whitney p", "published vs control source heads (4 vs 4); replaces the configuration-level p")
    add("head.rho", "HeadRho", sp["rho"], psrc, "head", "Spearman", "head as the unit; replaces the configuration-level rho")
    add("head.rho.p", "HeadRhoP", sp["p_less"], psrc, "head", "exact permutation p (one-sided)", "")
    add("head.rho.n", "HeadRhoN", f"{sp['n']}", psrc, "head", "count", "")
    add("head.rho.p2", "HeadRhoPTwo", sp["p_two_sided"], psrc, "head", "exact permutation p (two-sided)",
        "comparable in sidedness to the configuration-level scipy Spearman p it replaces")
    DS = load_json("detection_consistency.json")
    dsrc = ["detection_consistency.json", "detection_consistency.py"]
    gi, gc = DS["groups"]["ioi"], DS["groups"]["control"]
    add("det.armvar.ioi", "DetArmVarIOI", gi["median_resid_var_armA_over_armB"], dsrc, "configuration", "median",
        "single-site residual variance, arm A / arm B, published heads")
    add("det.armvar.ctrl", "DetArmVarCtrl", gc["median_resid_var_armA_over_armB"], dsrc, "configuration", "median",
        "same, control heads")
    add("det.inconsistent", "DetInconsistent", f"{gi['inconsistent_among_rej_S']}/{gi['rej_S']}", dsrc, "configuration",
        "count", "single-site rejections whose c_S is inconsistent with c_J (Wald, V_S + V_J)")
    add("det.inconsistent.all", "DetInconsistentAll", f"{gi['inconsistent_all']}/{gi['n']}", dsrc, "configuration", "count", "")

    # ---- audit quantities quoted by the corrections ledger (audit_extras.py; DESCRIPTIVE) ----
    AX = load_json("audit_extras.json")
    xsrc = ["audit_extras.json", "audit_extras.py"]
    k_, n_ = AX["donor_duplicate_captures"]
    add("audit.donor_dup", "AuditDonorDupCaptures", f"{k_}/{n_}", xsrc, "capture", "count",
        "captures with an identical recorded patch-A trajectory across the donor-index partition; donor-prompt identity "
        "not checked")
    fb = AX["fig1b"]
    add("audit.fig1b.single", "FigOneBRatio", fb["single"], xsrc, "capture", "value", "Figure 1(b) title, one capture")
    add("audit.fig1b.median", "FigOneBCaptureMedian", fb["median"], xsrc, "capture", "median",
        "upstream-coordinate range ratio, median over GPT-2 IOI matched captures")
    add("audit.fig1b.median_distinct", "FigOneBDistinctMedian", fb["median_distinct"], xsrc, "capture", "median",
        f"same over the {fb['n_distinct']} captures with distinct upstream coordinates")
    qi = AX["qc_info"]
    xs_, xt_ = AX["cross_seed_same_prompt"]
    add("audit.xseed.same_prompt", "AuditXseedSamePrompt", f"{xs_}/{xt_}", xsrc, "capture", "count",
        "cross-seed chart control rows whose base prompt equals the chart source's")
    add("audit.qc.plugin", "QCInfoRatioPlugin", qi["plugin_median_of_ratios"], xsrc, "configuration", "median",
        "plug-in homoskedastic Fisher information ratio per observation, J over S")
    add("audit.qc.design_rom", "QCInfoRatioRoM", qi["design_ratio_of_medians"], xsrc, "configuration", "ratio of medians",
        "design-Gram information, ratio of medians")
    add("audit.qc.own_whitening", "QCInfoRatioOwnChart", qi["design_own_family_whitening_median"], xsrc, "configuration",
        "median", "design-Gram information ratio with each family whitened by its own mean and sd (standardisation dependence)")

    # ---- preregistered fresh-template replication (prereg v3 FINAL; registered analysis output) ----
    RA = load_json("replication_analysis.json")
    assert RA["prereg"] == "v3 FINAL"
    rsrc = ["replication_analysis.json"]
    tv, cr, v2, sec = RA["templates"], RA["criteria"], RA["V2"], RA["secondary"]
    Vs = [t["V"] for t in tv.values()]; Vr = [t["Vrel"] for t in tv.values()]
    add("rep.h1abs", "RepHAbsCount", f"{cr['k_below_abs']}/8", rsrc, "template", "count", "template values < 0.10 logits (H1_abs needs 7/8)")
    add("rep.h1rel", "RepHRelCount", f"{cr['k_below_rel']}/8", rsrc, "template", "count", "template values < 1% of swing (H1_rel needs 7/8)")
    add("rep.dpos", "RepDPositive", f"{cr['k_d_positive']}/8", rsrc, "template", "count", "templates with published > control (descriptive under the floor rule)")
    add("rep.meddratio", "RepMedianRatio", cr["ratio_exp_median_d"], rsrc, "template", "exp(median log ratio)",
        "published/control, descriptive only: controls floor-limited, no calibrated ratio claimed")
    add("rep.V.min", "RepVMin", min(Vs), rsrc, "template", "min", "template RMS T_C, logits")
    add("rep.V.max", "RepVMax", max(Vs), rsrc, "template", "max", "")
    add("rep.V.med", "RepVMed", RA["medians"]["V"], rsrc, "template", "median", "")
    add("rep.Vrel.min", "RepVrelMin", min(Vr), rsrc, "template", "min", "percent of the B-arm swing")
    add("rep.Vrel.max", "RepVrelMax", max(Vr), rsrc, "template", "max", "")
    add("rep.Vrel.med", "RepVrelMed", RA["medians"]["Vrel"], rsrc, "template", "median", "")
    add("rep.pub.resolved", "RepPubResolved", f"{v2['published']['resolved']}/{v2['published']['n']}", rsrc, "capture", "count",
        "published captures with RMS T_C > 10x their batch-composition floor")
    add("rep.ctl.resolved", "RepCtlResolved", f"{v2['control']['resolved']}/{v2['control']['n']}", rsrc, "capture", "count",
        "control captures resolved (the registered family threshold is 90%)")
    add("rep.floor.med", "RepFloorMed", v2["published"]["median_floor"], rsrc, "capture", "median", "V2 floor, logits (published)")
    add("rep.ctl.rms.med", "RepCtlRmsMed", v2["control"]["median_rms"], rsrc, "capture", "median", "control RMS T_C, logits")
    add("rep.pub.rms.med", "RepPubRmsMed", v2["published"]["median_rms"], rsrc, "capture", "median", "published RMS T_C, logits")
    rows = [r for r in RA["rows"] if r["group"] == "published" and r.get("analysed")]
    rel = np.array([r["rel"] for r in rows])
    add("rep.pub.rel.p90", "RepPubRelPninety", float(np.percentile(rel, 90)), rsrc, "capture", "90th percentile",
        "per-capture published RMS T_C as percent of swing")
    add("rep.pub.rel.max", "RepPubRelMax", float(rel.max()), rsrc, "capture", "max", "")
    add("rep.n", "RepCaptures", f"{len(RA['rows'])}", rsrc, "capture", "count", "")
    add("rep.screened", "RepScreenedOut", f"{len(RA['gates']['screened_out'])}", rsrc, "capture", "count", "std(Y_joint) < 0.15")
    add("rep.errors", "RepErrors", f"{len(RA['gates']['errors'])}", rsrc, "capture", "count", "")
    add("rep.s6", "RepSSixWorse", sec["S6_published_S3_worse_than_zero"], rsrc, "capture", "count",
        "single-site cubic predicts T_C worse than zero (held-out)")
    add("rep.s7.S", "RepSSevenS", sec["S7_published_upper_all"]["S"], rsrc, "capture", "count", "single-site envelope covers T_C at every joint state")
    add("rep.s7.J", "RepSSevenJ", sec["S7_published_upper_all"]["J"], rsrc, "capture", "count", "joint envelope covers T_C at every joint state")
    rc = sec["S8"]["control"]["rej_S"]; rp = sec["S8"]["published"]["rej_S"]
    rob = ("hc0", "hc3", "arm", "fgls", "cr1")
    add("rep.s8.ctl.lo", "RepCtlRobLo", f"{min(rc[e] for e in rob)}", rsrc, "configuration", "min over robust estimators", "")
    add("rep.s8.ctl.hi", "RepCtlRobHi", f"{max(rc[e] for e in rob)}", rsrc, "configuration", "max over robust estimators", "")
    add("rep.s8.ctl.n", "RepCtlN", f"{sec['S8']['control']['n']}", rsrc, "configuration", "count", "")
    add("rep.s8.ctl.homo", "RepCtlHomo", f"{rc['homo']}/{sec['S8']['control']['n']}", rsrc, "configuration", "count", "pooled covariance")
    add("rep.s8.pub.homo", "RepPubHomo", f"{rp['homo']}/{sec['S8']['published']['n']}", rsrc, "configuration", "count", "pooled covariance")
    add("rep.s8.incons", "RepInconsistent", f"{sec['S8']['published']['cS_vs_cJ_inconsistent']}/{sec['S8']['published']['n']}", rsrc,
        "configuration", "count", "c_S inconsistent with c_J")

    # ---- counts and one derived ratio used in the arXiv text ----
    prompts = sorted({(r["template"], r["seed"]) for r in RA["rows"]})
    add("rep.templates", "RepTemplates", f"{cr['n_templates']}", rsrc, "template", "count", "fixed evaluation set")
    add("rep.prompts", "RepPrompts", f"{len(prompts)}", rsrc, "prompt", "count", "fresh prompts (template, seed)")
    HO = load_json("positive_control_heldout.json")
    add("predictM.cells", "PredCells", f"{len(HO['cells'])}", ["positive_control_heldout.json"], "cell", "count",
        "model x task cells behind Table 1")
    add("adq.envS_over_rmsT", "AdqEnvSOverRmsT", P["d3_S"]["env_med"]["median"] / P["rms_T"]["median"],
        ["attribution_adequacy.json", "attribution_adequacy.py"],
        "capture", "ratio of medians", f"single-site envelope / executed RMS T_C; {ex}")

    # ---- review round (corrections C38-C45): quantities defined in review_extras.py ----
    RX = load_json("review_extras.json")
    assert RX["status"].startswith("DESCRIPTIVE")
    xsrc = ["review_extras.json", "review_extras.py"]
    for lab, pre in (("GPT-2", "FSGPT"), ("Qwen3.5-9B", "FSQwen")):
        add(f"fs.{lab}.gap_bottom", pre + "EigengapBottomRel", RX["dk_gap"][lab]["J_bottom_rel_median"], xsrc, "capture",
            "median", "joint family: (lambda_{k+1} - lambda_k)/lambda_max at the BOTTOM-k (k=3) boundary; the accepted "
            "value was taken at the top-k boundary (C38)")
    ec = RX["edge_cos"]
    add("edge.rho_cos_abs_slope", "EdgeCosRhoSlope", ec["rho_cos_abs_slope"], xsrc, "capture", "Spearman",
        "|cos(g_R, g_C)| (Euclidean, fixed whitened column-scaled basis) vs absolute whitened slope")
    rs = RX["reverse_survivors"]
    add("edge.matched.survivors_weak", "ReverseSurvivorsWeak", f"{rs['n_below_0p10']}/{rs['n_survivors']}", xsrc, "capture",
        "count", "matched-protocol reverse-end escapes with sd(b|arm A)/sd(b|arm B) < 0.10")
    tl = RX["tl_calibration"]
    add("mech.tl_calibration_overlap", "TLCalibOverlap", f"{tl['calibration_rows_in_test_half']}/{tl['calibration_rows']}",
        xsrc, "row", "count", "surrogate 'tuned lens' calibration rows that fall in the first file's scored half")
    # replication: control-family floor and the eight registered template values (nested medians)
    add("rep.floor.med.ctl", "RepFloorMedCtl", RA["V2"]["control"]["median_floor"], rsrc, "capture", "median",
        "V2 floor, logits (control)")
    words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]
    for t in range(1, 9):
        tv_ = RA["templates"][str(t)]
        add(f"rep.T{t}.V", f"RepVT{words[t-1]}", tv_["V"], rsrc, "template", "median of prompt medians",
            f"template T{t}: median over its three prompts of the median RMS T_C over screened published configurations, logits")
        add(f"rep.T{t}.Vrel", f"RepVrelT{words[t-1]}", tv_["Vrel"], rsrc, "template", "median of prompt medians",
            f"template T{t}: the same nested median of per-capture RMS T_C as percent of the B-arm swing")

    # ---- the executed contrast against a typical per-donor effect (exploratory; attribution_adequacy rows) ----
    prow = AQ["groups"]["published"]["rows"]
    add("adq.rmsT_pct_effect", "AdqRmsTPctEffect",
        float(np.median([100 * r["rms_T"] / r["mean_abs_effect_alpha1"] for r in prow])),
        ["attribution_adequacy.json", "attribution_adequacy.py"], "capture", "median of per-capture ratios",
        "RMS T_C as percent of the capture's mean |per-donor B effect at alpha~1|; EXPLORATORY, post-review, three base prompts")
    add("adq.maxT_pct_effect", "AdqMaxTPctEffect",
        float(np.median([100 * r["max_abs_T"] / r["mean_abs_effect_alpha1"] for r in prow])),
        ["attribution_adequacy.json", "attribution_adequacy.py"], "capture", "median of per-capture ratios",
        "largest single |T_C| over the joint states as percent of the capture's mean |per-donor B effect at alpha~1|; EXPLORATORY")

    # ---- representation table (failed hypothesis (4); corrections C54, C55): degree-specific same-family nulls ----
    # mech_classes.py rebuilds the null at every degree; the frozen ledger carries only the degree-1 null (MC*Null).
    MC = load_json("mech_classes.json")
    MS, MB2, MB1 = MC["summary"], MC["raw"]["B2"], MC["raw"]["B1"]
    msrc = ["mech_classes.json", "mech_classes.py"]
    reps = (("raw", "MCRaw"), ("logitlens", "MCLL"), ("tunedlens", "MCTL"), ("randproj", "MCRand"))
    p05 = "median across captures of per-capture fifth percentiles over five same-family resamples"
    for rep, pre in reps:
        add(f"mech.{rep}.null2", f"{pre}NullTwo", MS[f"{rep}/d2"]["null_vS"], msrc, "capture", p05,
            "DEGREE-2 same-family null (rebuilt at degree 2)")
    sci = lambda x: (lambda m, e: rf"{m}\times10^{{{int(e)}}}")(*f"{x:.2e}".split("e"))
    for dg, word in ((1, "One"), (2, "Two")):
        assert all(MS[f"{p}/d{dg}"]["n"] == 12 for p, _ in reps)
        r = [MS[f"{p}/d{dg}"]["null_vS"] / MS[f"{p}/d{dg}"]["vS"] for p, _ in reps]
        for f_, tag in ((min, "Lo"), (max, "Hi")):
            v = f_(r)
            add(f"mech.nullratio.d{dg}.{tag.lower()}", f"MCNullRatio{word}{tag}", sci(v) if dg == 2 else v, msrc,
                "representation", f"{f_.__name__} over the four representations of (capture-median null / capture-median share)",
                f"degree-{dg} same-family null (capture median of per-capture 5th percentiles) over capture-median v_S; "
                "a ratio of capture medians, not a median of within-capture ratios; DESCRIPTIVE", raw=v if dg == 2 else None)
    # degree-2 share below the random-chart-of-raw control, per capture; reported for every row, since the PCA chart
    # of raw activations shows the same ordering (the count is not lens-specific)
    assert set(MB1) == set(MB2) and len(MB1) == 12
    rp2 = {c: MB1[c]["randproj"]["2"]["vS_on_Sblind"] for c in MB1}
    for rep, key, mac in (("logitlens", "ll", "MCLLBelowRandTwo"), ("raw", "raw", "MCRawBelowRandTwo"),
                          ("tunedlens", "tl", "MCTLBelowRandTwo")):
        k_ = sum(MB2[c][rep]["2"]["vS_on_Sblind"] < rp2[c] for c in MB1)
        add(f"mech.{key}_below_rand.d2", mac, f"{k_}/{len(MB1)}", msrc, "capture", "count",
            f"degree-2 vS_on_Sblind of the {rep} row below the random-chart-of-raw control (randproj) in the same "
            "capture; raw activations with no lens show the same ordering, so the count is not lens-specific; "
            "descriptive, one model-by-task unit (GPT-2 IOI)")
    # the accepted rationale for 'no separation' (C55): spread between rows vs the shipped refit check
    for dg, word in ((1, "One"), (2, "Two")):
        add(f"mech.rand_over_ll.d{dg}", f"MCRandOverLLDeg{word}", MS[f"randproj/d{dg}"]["vS"] / MS[f"logitlens/d{dg}"]["vS"],
            msrc, "representation", "ratio of capture medians",
            f"degree-{dg} capture-median share, random projection over logit lens: a direct cross-representation "
            "magnitude comparison, which the registered comparison rule does not use as evidence; DESCRIPTIVE (C55)")
    B0 = MC["raw"]["B0"]
    add("mech.b0.d2_ratio", "MCBZeroReparamDegTwo",
        B0["reparameterized"]["2"]["vS_on_Sblind"] / B0["base"]["2"]["vS_on_Sblind"], msrc, "capture", "ratio",
        "degree-2 v_S after / before the shipped B0 reparameterisation; single capture; column-scaled (non-orthogonal) "
        "Q followed by chart refit; not a same-class numerical benchmark (C55)")

    # ---- provenance fix (value and macro unchanged): predict.total is the sum of the per-cell training-only counts
    # in edge_strength_matched.json, not an output of positive_control.py ----
    PT = load_json("edge_strength_matched.json")["predict_unmatched"]["total"]["corrected"]
    assert L["predict.total"]["value"] == f"{PT[0]}/{PT[1]}", (L["predict.total"]["value"], PT)
    add("predict.total", "PredTotalCount", L["predict.total"]["value"],
        ["edge_strength_matched.json", "edge_strength_matched.py"], "capture", "count",
        "theory-named direction above 10x, training-only estimation, unmatched joint family (raw donors), pooled over "
        "every model x task cell (sum of predict_unmatched.cells[*].corrected_gt10)")

    # ---- documentation-only note fixes (values, macros and artifacts unchanged) ----
    fisher = "residualised cross-block design Gram (smallest eigenvalue of R^T R / n per observation; not Fisher information)"
    for k, v in L.items():
        n0 = v.get("note", "")
        if "partial Fisher information" in n0:
            v["note"] = n0.replace("smallest eigenvalue of the partial Fisher information", "smallest eigenvalue of the " + fisher) \
                          .replace("partial Fisher information", fisher)
    L["mis.S"]["note"] = ("median over captures of the median pointwise degree-3 95% working-model Wald envelope, as percent of "
                          "that capture's B-arm swing: an identification width under the working model, not an observed error")
    L["mis.over1"]["note"] = "captures whose single-site working-model envelope exceeds the whole B-arm swing"
    for lab in ("GPT-2", "Qwen3.5-9B"):
        L[f"fs.{lab}.gap"]["note"] = "joint family, TOP-k boundary (w[p-k]-w[p-k-1]); not the low-energy subspace's gap (C38)"
    for k, v in L.items():
        v["note"] = v.get("note", "").replace("theory-named", "named (recompute) generator g_R")
    L["reparam.invariance"]["aggregation"] = "max over cells of capture medians of the per-capture max relative eigenvalue change"
    L["reparam.invariance"]["note"] = "UNREGULARISED generalized eigenvalues under phi -> A phi (C52)"
    if "ksite.joint" in L:
        L["ksite.joint"]["note"] = "joint design of 4000 independent uniform points at k=5, degree 3 (not a product grid; C49)"
    # representation table: the frozen null keys are the DEGREE-1 null; the degree-3 omission (C54)
    for rep, _ in reps:
        L[f"mech.{rep}.null1"].update(note="DEGREE-1 same-family null", unit="capture", aggregation=p05)
        for dg in (1, 2):
            L[f"mech.{rep}/d{dg}"]["note"] = (
                "v_S on the S-blind direction; degree 3 NOT reported: below numerical resolution (under orthogonal "
                "reparameterisations of one capture degrees 1-2 are invariant to numerical precision and degree 3 is "
                "not; e0/b0_orthogonal.json). mech_classes.json raw.B0 is a column-scaled, non-orthogonal "
                "reparameterisation with chart refit, not an orthogonal check")
    # edge-strength statistics (C44): the x variable is the training-row ratio s, with arm B as denominator
    s_tr = "training-row edge strength s = sd(b|arm A)/sd(b|arm B)"
    for k, n_ in (("edge.spearman_cos_rho", s_tr + " vs |cos(g_R, g_C)| (Euclidean, in each capture's training-fitted "
                                            "whitened, column-scaled coefficient basis); not the slope; corrected row protocol (C44)"),
                  ("edge.spearman_sep_rho", s_tr + " vs sep(g_R)/sep(g_C), the within-capture ratio of held-out J/S energy "
                                            "ratios r(g_R)/r(g_C); corrected row protocol (C44)"),
                  ("edge.nstrong", "captures with " + s_tr + " >= 0.10 (denominator arm B, not the joint family); "
                                   "corrected row protocol (C44)"),
                  ("pc.zero", "zero-noise Euclidean projection residual of the least-energy pencil direction onto "
                              "span{g, a g, b g} in the fixed scaled basis: subspace membership, not generator recovery (C41)"),
                  ("ofat.rho", "row-protocol captures: |training-whitened edge slope| vs r(g_R)/r(g_C)")):
        L[k]["note"] = n_
    for k, w in (("reparam.angle_before", "before"), ("reparam.angle_after", "recomputed after")):
        L[k].update(unit="cell", aggregation="median of within-cell medians of per-capture median angles",
                    note=f"principal angle between single-site and joint low-energy training-Gram eigenspaces, {w} "
                         "phi -> A phi; earlier unmatched captures; coordinate-dependent (C34)")
    for dg in ("deg2", "deg3"):
        for suf in ("below", "above", "logr"):
            L[f"three.{dg}.{suf}"]["unit"] = "capture"
        L[f"three.{dg}.logr"]["note"] = ("minus the median over captures of the minimum held-out log10(S/J) energy ratio "
                                         "over all frozen training eigenvectors (C45)")
    # joint-family Wald covariance, identification-table statistics and the replication maximum
    msum = ("marginal-sum donor covariance V_up + V_down (intersection term and cross-role donor dependence omitted; "
            "calibration unknown)")
    for k, grp in (("det.ioi.j2", "published heads"), ("det.ctl.j2", "control heads")):
        if k in L:
            L[k]["note"] = f"joint-family Wald rejections of zero interaction with the {msum}, {grp}"
    if "det.power" in L:
        L["det.power"]["note"] = ("minimum nominal plug-in power of the joint Wald test at the single-site estimate, over "
                                  f"published captures rejected by the pooled single-site test, with the {msum}")
    if "adq.ind.rejJ.2way" in L:
        L["adq.ind.rejJ.2way"]["note"] = f"induction group, joint Wald with the {msum}; EXPLORATORY, post-review, three base prompts"
    chart = "common per-coordinate standardisation over all single-site and joint rows"
    for k, v in L.items():
        if not k.startswith(("id.", "idUnmatched.")):
            continue
        if k.endswith(".se"):
            v["note"] = ("ratio of Euclidean norms of the marginal-SE vectors of the three cross coefficients, single-site / "
                         f"joint (homoskedastic OLS covariance after partialling out the additive block, per family); {chart}")
        elif k.endswith((".flip", ".flipj")):
            fam = "single-site" if k.endswith(".flip") else "joint"
            v["note"] = (f"fraction of 40 random complementary row-half splits of the {fam} rows on which the fitted ab "
                         f"coefficient changes sign; {chart}; descriptive, parameterisation-dependent")
    if "rep.pub.rel.max" in L:
        L["rep.pub.rel.max"]["note"] = ("maximum over analysed published replication captures of the per-capture RMS T_C as "
                                        "percent of the B-arm swing; an RMS over the tested joint grid, not a pointwise maximum")

    # ---- guard: additions never overwrite ----
    for k, m in before.items():
        assert L[k]["macro"] == m, f"ledger_additions changed an existing key: {k}"
    macros = [v["macro"] for v in L.values()]
    assert len(macros) == len(set(macros)), "duplicate macro name after additions"
