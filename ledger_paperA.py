"""CLAIM-TO-ARTIFACT LEDGER (Paper-A-only frozen copy).  Every manuscript number is generated here, never typed.

PROVENANCE.  Frozen copy of the project ledger generator (repository commit ed345ae, the project's claims.py)
for the arXiv version of "What Can Activation Patching Actually Establish?".  Documented changes, all
outside the derivations: (1) the default data/code directory is ./e0 next to this file; (2) every
artifact is listed in claims_paperA.json (no truncated lists); (3) outputs are claims_paperA.json and
claims_paperA.tex next to this file; (4) the header comment of the .tex names this file.

WHY THIS EXISTS.  Bug #10: the figure "106x/49x" reached the abstract and could not be
reproduced from the frozen code and data; an entire ridge column had been imported from a
different experiment.  The freeze (FREEZE.md + git) stops ARTIFACTS drifting but does nothing
about prose that drifted before the freeze.  So no quantitative claim may be transcribed by
hand into the abstract, body, tables, captions, or appendix.  Each is a LaTeX macro emitted
from a re-derivation run against a hashed artifact.

USAGE
    python claims.py            # re-derive everything -> claims.json + claims.tex
    python claims.py --verify   # re-derive and DIFF against claims.json; nonzero exit on drift

Each claim records: the artifact it comes from, that artifact's sha256, the statistical unit,
the aggregation, and the derived value.

THE UNIT IS A CAPTURE, NOT A SITE PAIR.  One capture is a (layer pair, seed) run: the seed also
selects the prompt and the donor draw, so two captures on the same layer pair are not repeats of
one measurement, and two captures within a cell still share weights and task.  Counts of the form
"k/n" are therefore over captures.  Calling them site pairs, as an earlier version of this ledger
did, understates the replication structure and was corrected on 2026-08-27.  Prefer deriving from RAW data (npz) over a summary
JSON wherever the derivation is cheap -- a summary JSON is itself a transcription step.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

E0 = os.environ.get("E0_DIR",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "e0"))
sys.path.insert(0, E0)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def files(pat):
    return sorted(glob.glob(os.path.join(E0, pat)))


def load_json(name):
    with open(os.path.join(E0, name)) as f:
        return json.load(f)


# ------------------------------------------------------------- derivations ---
def d_interaction_ratio(pattern):
    """Median PER-TRIAL ratio of interaction-coefficient magnitude, single vs joint.

    Re-derived from RAW npz via the frozen mechanism_case.case(), not from any summary.
    Per-trial median is used deliberately: a ratio of two independently taken medians is
    the less honest statistic and is what bug #10 propagated.
    """
    from mechanism_case import case
    rs = []
    for f in files(pattern):
        d = np.load(f)
        if "Ys" not in d:
            continue
        r = case(d["Zs"], d["Ys"], d["Zj"], d["Yj"])
        rs.append(r["cross_mass_from_single"] / max(r["cross_mass_from_joint"], 1e-12))
    return float(np.median(rs)), len(rs), [os.path.basename(f) for f in files(pattern)]


def d_reciprocal(pattern):
    """Median held-out S-blind and J-blind ratios, and the both>10x count. From RAW npz."""
    from subspace_test import analyse
    sS, sJ = [], []
    for f in files(pattern):
        r = analyse(np.load(f)["Zs"], np.load(f)["Zj"])
        sS.append(r["sep_Sblind"]); sJ.append(r["sep_Jblind"])
    sS, sJ = np.array(sS), np.array(sJ)
    return (float(np.median(sS)), float(np.median(sJ)),
            int(np.sum((sS > 10) & (sJ > 10))), len(sS))


def d_ioi_published():
    """S-Inhib -> Name Mover ONLY, with the pre-specified std(Y_joint)>=0.15 screen."""
    d = [x for x in load_json("ioi_circuit.json")
         if tuple(x["pair"]) == ("S-Inhib", "NameMover") and x["std_Yj"] >= 0.15]
    r = np.array([x["ratio"] for x in d]); rr = np.array([x["ridge_ratio"] for x in d])
    return float(np.median(r)), float(np.median(rr)), float(r.min()), len(d)


def d_nullcal(model_key):
    """Same-family null p95 vs cross-family minimum -- the primary calibration."""
    c = load_json("controls_subspace.json")[model_key]
    return float(c["c4_same_p95"]), float(c["c4_same_median"])


def d_task(task, model):
    j = load_json(f"nonioi_{task}_{model}.json")
    return (int(j["n_screened_out"]), int(j["n_kept"]), int(j.get("both", 0)),
            float(j.get("median_S", float("nan"))), float(j.get("median_J", float("nan"))))


def d_invariant(key):
    """Invariant energy shares and null-envelope escape counts (coordinate-free)."""
    r = load_json("invariant.json")[key]
    return (r["vS_on_Sblind"], r["vJ_on_Jblind"], r["both_ends"], r["n"],
            r["null_lo"], r["null_hi"], r["cov_reg_invariance"], r["euc_reg_invariance"])


def d_reach():
    """Day-1 exact-witness experiment: recompute vs clamped, per norm."""
    r = load_json("reachability.json")
    rec = [v for k, v in r.items() if k.endswith("/recompute")]
    cla = [v for k, v in r.items() if k.endswith("/clamped")]
    return (max(v["raw_gap"] for v in rec), max(v["auc"] for v in rec),
            min(v["auc"] for v in cla), min(v["raw_gap"] for v in cla),
            float(np.median([v["erank"] for v in rec])),
            float(np.median([v["erank"] for v in cla])))


def d_operator():
    """Track A: operator-slice theorem, d* per operator + the dichotomy."""
    r = json.load(open(os.path.join(E0, "operator_degree.json")))
    by = {x["operator"]: x for x in r}
    return by


def d_mechclass():
    """Track B: degree ladder per representation, and the null verdict."""
    return json.load(open(os.path.join(E0, "mech_classes.json")))["summary"]


def d_finite():
    """Appendix: finite-sample bounds and their measured vacuity."""
    r = load_json("finite_sample.json")
    out = {}
    for lab, rows in r.items():
        dk = float(np.median([x["J_davis_kahan_bound"] for x in rows]))
        gap = float(np.median([x["J_eigengap_rel"] for x in rows]))
        rate = float(np.median([x["S_rate_slope"] for x in rows]))
        ang = float(np.median([x["prop5"]["angle_Sblind_half1_vs_half2"] for x in rows]))
        out[lab] = (dk, gap, rate, ang)
    return out


def d_coord():
    """Coordinate robustness of the reciprocal result (4 charts)."""
    import numpy as _np
    r = json.load(open(os.path.join(E0, "coord_robust.json")))
    out = {}
    for k in ("u", "pc1", "diff", "pc2d"):
        vals = _np.array([[x[k][0], x[k][1]] for x in r])
        out[k] = (float(_np.median(vals[:, 0])), float(_np.median(vals[:, 1])),
                  int(_np.sum((vals[:, 0] > 10) & (vals[:, 1] > 10))), len(vals))
    return out


def d_basis(model_key):
    """Basis / mechanism-class robustness."""
    b = load_json("controls_subspace.json")[model_key]["basis"]
    return {k: (v[0], v[1], v[2], v[3]) for k, v in b.items()}


def d_poscontrol():
    """Positive control + theory-direction random null, for the prediction section."""
    import numpy as _np
    r = json.load(open(os.path.join(E0, "positive_control.json")))
    pa = r["part_a"]
    zero = next(x["resid"] for x in pa if x["noise"] == 0.0)
    worst = max(x["resid"] for x in pa)
    p95 = [v["rnd_sep_p95_median"] for k, v in r.items() if k != "part_a"]
    rnd_p1 = float(min(x["rnd_p1"] for x in pa))     # conservative: smallest over the noise sweep
    return zero, worst, float(min(p95)), float(max(p95)), rnd_p1


def d_ioi_rmse():
    """Held-out single-site RMSE agreement between additive and interactive fits, IOI heads."""
    import numpy as _np
    d = [x for x in load_json("ioi_circuit.json")
         if tuple(x["pair"]) == ("S-Inhib", "NameMover") and x["std_Yj"] >= 0.15]
    rel = [abs(x["rmse_full"] - x["rmse_add"]) / max(x["rmse_add"], 1e-12) for x in d]
    return 100.0 * float(_np.median(rel))


def d_path():
    """Path-patching family vs its matched same-family null."""
    r = load_json("path_controls.json")
    return (r["cross_P"], r["cross_J"], r["same_median"], r["same_p95"],
            r["ci_J"][0], r["ci_J"][1])


def d_comm():
    """Mediator-interaction magnitude on the IOI configurations."""
    import numpy as _np
    r = load_json("comm_compare.json")
    v = _np.abs([x["int_over_sd"] for x in r])
    return float(v.min()), float(v.max()), len(r)


def d_reparam():
    r = load_json("reparam.json")
    inv = max(v["unreg_max_rel_diff"] for v in r.values())
    ridge = min(v["ridge_0.01_max_rel_diff"] for v in r.values())
    ang_b = float(np.median([v["angles_before_med"] for v in r.values()]))
    ang_a = float(np.median([v["angles_after_med"] for v in r.values()]))
    scale = float(np.median([v["rel_resid_Gj_minus_cGs"] for v in r.values()]))
    return inv, ridge, ang_b, ang_a, scale


def d_matched():
    """Support-matched vs frozen vs interior protocols, per model x task cell."""
    return load_json("matched_invariant.json")


def d_edge():
    """Edge strength (Theorem 1's non-degeneracy hypothesis) measured on every capture."""
    # corrected 2026-09-24: support-matched captures, training-only estimation (edge_strength_matched.py);
    # the legacy unmatched, all-rows values (edge_strength.json) are superseded and kept only as a record
    return load_json("edge_strength_matched.json")["corrected_row"]["summary"]


def d_ident(matched=True):
    """Identifiability of the interaction coefficients: partial Fisher information and se."""
    return load_json("identifiability_matched.json" if matched else "identifiability.json")


def d_predict_matched(pattern, jkey="Zjm"):
    """Theory-named direction against the SUPPORT-MATCHED joint family.

    The frozen captures hold only the raw-donor joint family, which Sec. 6 shows is confounded
    with the single-site family's interpolation grid.  Table 1 must not rest on that pairing.
    """
    from positive_control import part_b
    sp = []
    for f in files(pattern):
        d = np.load(f)
        if jkey not in d.files:
            continue
        sp.append(part_b(d["Zs"], d[jkey])["sep_pred"])
    sp = np.array(sp)
    return (float(np.median(sp)), int(np.sum(sp > 10)), len(sp)) if len(sp) else (float("nan"), 0, 0)


def d_predict(pattern):
    """Theory-named direction: median held-out J/S ratio and >10x count. From RAW npz."""
    from positive_control import part_b
    sp, res = [], []
    for f in files(pattern):
        d = np.load(f)
        r = part_b(d["Zs"], d["Zj"])
        sp.append(r["sep_pred"]); res.append(r["resid_discovered_in_predicted"])
    sp = np.array(sp)
    return float(np.median(sp)), int(np.sum(sp > 10)), len(sp), float(np.median(res))


# ------------------------------------------------------------------ format ---
def fmt(x, sig=3):
    """LaTeX-ready number: plain below 1e4, scientific above."""
    if not np.isfinite(x):
        return r"\textsc{nan}"
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        m, e = f"{x:.{sig-1}e}".split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.{sig}g}"


def build():
    L = {}

    def add(mid, macro, value, artifacts, unit, agg, note="", raw=None):
        arts = []
        for a in artifacts:
            p = os.path.join(E0, a)
            for q in ([p] if os.path.exists(p) else files(a)):
                arts.append({"file": os.path.relpath(q, os.path.join(E0, "..")),
                             "sha256": sha(q)})
        L[mid] = {"macro": macro, "value": value if raw is None else raw,
                  "latex": fmt(value) if isinstance(value, float) else str(value),
                  "unit": unit, "aggregation": agg, "note": note,
                  "n_artifacts": len(arts),
                  "artifacts": arts}

    # ---- consequence: interaction coefficient ----
    v, n, _ = d_interaction_ratio("adj_raw_*.npz")
    add("gpt2.interaction_ratio", "GPTInteractionRatio", v, ["adj_raw_*.npz", "mechanism_case.py"],
        "capture", "median of per-trial ratios",
        f"n={n}; replaces the withdrawn 106x (bug #10)")
    v, n, _ = d_interaction_ratio("rec_raw_*.npz")
    add("qwen.interaction_ratio", "QwenInteractionRatio", v, ["rec_raw_*.npz", "mechanism_case.py"],
        "capture", "median of per-trial ratios", f"n={n}; replaces the withdrawn 49x")

    # ---- reciprocal blindness ----
    for pat, pre, lab in (("adj_raw_*.npz", "GPT", "GPT-2"), ("rec_raw_*.npz", "Qwen", "Qwen3.5")):
        s, j, b, n = d_reciprocal(pat)
        add(f"{lab}.sblind", pre + "Sblind", s, [pat, "subspace_test.py"], "capture", "median",
            "held-out, frozen direction")
        add(f"{lab}.jblind", pre + "Jblind", j, [pat, "subspace_test.py"], "capture", "median",
            "held-out, frozen direction")
        add(f"{lab}.both", pre + "Both", f"{b}/{n}", [pat], "capture", "count",
            "captures = layer pair x seed; NOT independent replications")

    # ---- published IOI edge ----
    r, rr, mn, n = d_ioi_published()
    add("ioi.published_raw", "IOIPublishedRawMedian", r, ["ioi_circuit.json"], "trial", "median",
        "S-Inhib->NameMover only, screened")
    add("ioi.published_ridge", "IOIPublishedRidgeMedian", rr, ["ioi_circuit.json"], "trial",
        "median", "ridge lambda=1e-2")
    add("ioi.published_min", "IOIPublishedMinRatio", mn, ["ioi_circuit.json"], "trial", "min", "")
    add("ioi.published_n", "IOIPublishedN", f"{n}", ["ioi_circuit.json"], "trial", "count",
        "of 9; one failed the std>=0.15 screen")

    # ---- null calibration (primary evidence, per review) ----
    for key, pre in (("GPT-2", "GPT"), ("Qwen3.5", "Qwen")):
        p95, med = d_nullcal(key)
        add(f"{key}.null_p95", pre + "NullPNinetyFive", p95, ["controls_subspace.json"],
            "resample", "p95", "same-family null")
        add(f"{key}.null_med", pre + "NullMedian", med, ["controls_subspace.json"],
            "resample", "median", "same-family null")

    # ---- cross-task ----
    for task, model, pre in (("induction", "gpt2", "IndGPT"), ("capital", "gpt2", "CapGPT"),
                             ("greater", "gpt2", "GtGPT"), ("agreement", "gpt2", "AgrGPT"),
                             ("induction", "Qwen3.5-9B-Base", "IndQwen"),
                             ("capital", "Qwen3.5-9B-Base", "CapQwen")):
        try:
            so, kept, both, ms, mj = d_task(task, model)
        except FileNotFoundError:
            continue
        add(f"{task}.{model}.both", pre + "Both", f"{both}/{kept}",
            [f"nonioi_{task}_{model}.json"], "capture", "count", f"screened out {so}")
        add(f"{task}.{model}.sblind", pre + "Sblind", ms, [f"nonioi_{task}_{model}.json"],
            "capture", "median", "")
        add(f"{task}.{model}.jblind", pre + "Jblind", mj, [f"nonioi_{task}_{model}.json"],
            "capture", "median", "")

    # ---- theory-predicted direction (one cell per model x task; cells.py) ----
    from cells import CELLS
    pred_tot = [0, 0]
    for c in CELLS:
        pat = c["pattern"]
        pre = "Pred" + ("GPTIOI" if c["pre"] == "GPT" else "QwenIOI" if c["pre"] == "Qwen" else c["pre"])
        if not files(pat):
            continue
        PU = load_json("edge_strength_matched.json")["predict_unmatched"]["cells"][pre]   # training-only refit (corrected)
        v, k, n = PU["corrected_median"], PU["corrected_gt10"], PU["n"]
        add(f"predict.{pre}", pre + "Ratio", v, ["edge_strength_matched.json", "edge_strength_matched.py"], "capture",
            "median", "direction named from geometry alone, no Y used; unmatched joint family, training-only estimation")
        add(f"predict.{pre}.n", pre + "Count", f"{k}/{n}", ["edge_strength_matched.json"], "capture", "count", "")
        pred_tot[0] += k; pred_tot[1] += n
    add("predict.total", "PredTotalCount", f"{pred_tot[0]}/{pred_tot[1]}",
        ["positive_control.py"], "capture", "count",
        "theory-named direction above 10x, pooled over every model x task cell")

    # ---- theory-named direction, SUPPORT-MATCHED joint family (primary for Table 1) ----
    mpat = {"PredGPTIOI": "ioi_matched_gpt2_*.npz", "PredQwenIOI": "ioi_matched_qwen35_*.npz",
            "PredIndGPT": "induction_matched_gpt2_*.npz",
            "PredIndQwen": "induction_matched_qwen35_*.npz",
            "PredCapGPT": "capital_matched_gpt2_*.npz",
            "PredCapQwen": "capital_matched_qwen35_*.npz",
            "PredGtGPT": "greater_matched_gpt2_*.npz",
            "PredAgrGPT": "agreement_matched_gpt2_*.npz",
            # THIRD model family: Qwen3-8B is dense standard attention, unlike Qwen3.5-9B-Base's
            # hybrid Gated-DeltaNet/attention stack, so the two Qwen cells are not one datapoint.
            "PredQIOI": "ioi_matched_qwen3_*.npz",
            "PredQInd": "induction_matched_qwen3_*.npz"}
    # 2026-09-23: Table 1 reads positive_control_heldout.json, which implements the stated protocol
    # (edge, whitening and column scaling fitted on TRAINING rows only). positive_control.py::part_b
    # fitted them on all rows before splitting; it is kept, unread, as the historical record.
    HO = load_json("positive_control_heldout.json")
    hsrc = ["positive_control_heldout.json", "positive_control_heldout.py"]
    mtot, dtot, p95s = [0, 0], [0, 0], []
    for pre, pat in mpat.items():
        cell = HO["cells"].get(pre)
        if not cell:
            continue
        r, dn = cell["summary"]["row"], cell["summary"]["donor"]
        add(f"predictM.{pre}", pre + "MRatio", r["median"], hsrc, "capture",
            "median", "theory-named direction, held-out rows, support-matched joint family")
        add(f"predictM.{pre}.n", pre + "MCount", f"{r['gt10']}/{r['n']}", hsrc, "capture", "count", "")
        mtot[0] += r["gt10"]; mtot[1] += r["n"]; dtot[0] += dn["gt10"]; dtot[1] += dn["n"]
        p95s.append(r["rnd_p95_median"])
    if mtot[1]:
        add("predictM.total", "PredMTotalCount", f"{mtot[0]}/{mtot[1]}", hsrc, "capture", "count",
            "pooled over every model x task cell, held-out rows, support-matched protocol")
        add("predictM.donor", "DonorNamedCount", f"{dtot[0]}/{dtot[1]}", hsrc, "capture", "count",
            "theory-named direction above 10x with a common donor partition; joint rows scored only if both donors are held out")
        add("predictM.nullLo", "HeldoutNullLo", min(p95s), hsrc, "cell", "min over cells",
            "95th percentile of random same-class directions, same captures and split")
        add("predictM.nullHi", "HeldoutNullHi", max(p95s), hsrc, "cell", "max over cells", "")
        beat = [sum(r_["row"]["ratio"] > r_["row"]["rnd_p95"] for r_ in HO["cells"][pre]["rows"]) for pre in mpat if pre in HO["cells"]]
        nall = [len(HO["cells"][pre]["rows"]) for pre in mpat if pre in HO["cells"]]
        add("predictM.beatnull", "HeldoutBeatsNull", f"{sum(beat)}/{sum(nall)}", hsrc, "capture", "count",
            "captures whose named-generator ratio exceeds that capture's own random-direction 95th percentile, held-out rows")
        PL = HO["prompt_level"]["row"]
        add("predictM.prompts", "HeldoutPrompts", f"{PL['prompts']}", hsrc, "(cell, prompt) group", "count", "")
        add("predictM.promptStrict", "HeldoutPromptStrict", f"{PL['strict']}/{PL['prompts']}", hsrc,
            "(cell, prompt) group", "count", "groups in which EVERY capture exceeds 10x")
        add("predictM.promptBootLo", "HeldoutPromptBootLo", 100 * PL["boot_lo"], hsrc, "prompt", "percent",
            "2.5th percentile of the pooled fraction above 10x, bootstrap over (cell, prompt) groups")
        add("predictM.n", "MatchedN", f"{mtot[1]}", ["matched_invariant.json"], "capture",
            "count", "support-matched captures; distinct from the 82 frozen raw captures")

    # ---- ROBUSTNESS: prompts, not captures, as the unit ----
    try:
        PC = load_json("prompt_cluster.json")
    except FileNotFoundError:
        PC = None
    if PC:
        src = ["prompt_cluster.json", "prompt_cluster.py"]
        add("pc.prompts", "PCPrompts", f"{PC['prompts']}", src, "prompt", "count",
            "distinct (cell, prompt) groups behind the matched captures")
        add("pc.esc", "PCPromptEscapes", f"{PC['prompt_escapes']}/{PC['prompts']}", src, "prompt",
            "count", "prompt escapes only if EVERY capture on it escapes")
        add("pc.p", "PCPromptP", PC["p_prompt"], src, "prompt", "binomial p", "")
        add("pc.boot", "PCBootLo", PC["boot_lo"], src, "prompt",
            "2.5 pct of a cluster bootstrap", "escape fraction resampling prompts")

    # ---- what the unidentified interaction costs a per-component number ----
    try:
        MA = load_json("misattribution_fourpoint.json")   # 2026-09-23: the four-point contrast misattribution.py documents but did not compute
    except FileNotFoundError:
        MA = None
    if MA:
        src = ["misattribution_fourpoint.json", "misattribution_fourpoint.py"]
        add("mis.n", "MisN", f"{MA['n']}", src, "capture", "count", "")
        add("mis.S", "MisS", 100 * MA["med_S"], src, "capture", "median",
            "percent of the reported per-component number that single-site evidence cannot exclude")
        add("mis.J", "MisJ", 100 * MA["med_J"], src, "capture", "median", "same, joint evidence")
        add("mis.ratio", "MisRatio", MA["ratio"], src, "capture", "ratio of medians", "")
        add("mis.worstS", "MisWorstS", 100 * MA["max_S"], src, "capture",
            "median over captures of the per-capture maximum",
            "worst case over the reachable states, single-site evidence")
        add("mis.worstJ", "MisWorstJ", 100 * MA["max_J"], src, "capture",
            "median over captures of the per-capture maximum",
            "worst case over the reachable states, joint evidence")
        add("mis.over1", "MisOverOne", f"{MA['over_one_S']}/{MA['n']}", src, "capture", "count",
            "captures where the un-excluded misattribution exceeds the reported number entirely")
        add("mis.tenthS", "MisOverTenthS", f"{MA['over_tenth_S']}/{MA['n']}", src, "capture",
            "count", "single-site licence exceeds 10% of the reported number")
        add("mis.tenthJ", "MisOverTenthJ", f"{MA['over_tenth_J']}/{MA['n']}", src, "capture",
            "count", "joint licence exceeds 10%")
        add("mis.maxJ", "MisMaxJ", 100 * MA["max_J_any"], src, "capture", "maximum",
            "largest joint licence over any capture")
        add("mis.cov", "MisCoverage", 100 * MA["coverage"], src, "synthetic trial", "percent",
            "realised coverage of the 95% ellipsoid bound against a known interaction")
    try:   # SUPERSEDED two-term calculation (h(a,b)-h(a0,b)); kept only so the manuscript can disclose the change
        MO = load_json("misattribution.json")
        add("mis.S.twoterm", "MisSTwoTerm", 100 * MO["med_S"], ["misattribution.json", "misattribution.py"], "capture", "median",
            "SUPERSEDED: the earlier calculation that omitted the two baseline-B terms of the four-point contrast")
    except (FileNotFoundError, KeyError):
        pass

    # ---- ROBUSTNESS: the same claim under a DONOR-level holdout ----
    # subspace_test.prep splits by row, and every donor contributes N_AL interpolation
    # coefficients, so a donor appears on both sides of the split.  The paper's null is built by
    # the identical procedure and so absorbs that leakage, but the stricter test is worth running.
    try:
        DS = load_json("donor_split.json")
    except FileNotFoundError:
        DS = None
    if DS:
        de, dn = sum(v["escapes"] for v in DS.values()), sum(v["n"] for v in DS.values())
        add("donor.total", "DonorTotalCount", f"{de}/{dn}", ["donor_split.json"], "capture",
            "count", "escapes under a donor-level holdout with a donor-level same-family null")
        add("donor.nalpha", "NAlpha", f"{__import__('cells').N_AL}", ["cells.py"],
            "grid point", "count", "interpolation coefficients per donor")
        add("donor.cells", "DonorCells", f"{len(DS)}", ["donor_split.json"], "cell", "count", "")
        add("donor.pmax", "DonorPMax", max(v["p"] for v in DS.values()),
            ["donor_split.json"], "cell", "binomial p", "worst cell")

    # ---- Track A: operator-slice resolving theorem ----
    ops = d_operator()
    for op, macro in (("affine", "DstarAffine"), ("rmsnorm", "DstarRMSNorm"),
                      ("layernorm", "DstarLayerNorm")):
        if op in ops and ops[op].get("deg_P") is not None:
            add(f"opdeg.{op}", macro, f"{ops[op]['d_star']}",
                ["operator_degree.json", "operator_degree.py"], "operator",
                "exact (population)", f"deg P = {ops[op]['deg_P']}")
            add(f"opdeg.{op}.degP", macro + "DegP", f"{ops[op]['deg_P']}",
                ["operator_degree.json"], "operator", "exact", "")
    for op, macro in (("softmax", "DstarSoftmax"), ("gated_linear", "DstarGatedLinear")):
        if op in ops:
            add(f"opdeg.{op}", macro, r"\infty", ["operator_degree.json"], "operator",
                "exact (population)", "transcendental h => I(C) = (0)")

    # ---- Track B: representation-induced classes (pre-registered comparison NULL) ----
    try:
        mc = d_mechclass()
        for rep, pre in (("raw", "MCRaw"), ("logitlens", "MCLL"),
                         ("tunedlens", "MCTL"), ("randproj", "MCRand")):
            for dg, word in ((1, "One"), (2, "Two")):
                key = f"{rep}/d{dg}"
                if key in mc:
                    add(f"mech.{key}", f"{pre}Deg{word}", mc[key]["vS"],
                        ["mech_classes.json", "mech_classes.py"], "capture", "median",
                        "v_S on the S-blind direction; deg 3 NOT reported (below numerical "
                        "resolution -- orthogonal reparam moves it 7x)")
            k1 = f"{rep}/d1"
            if k1 in mc:
                add(f"mech.{rep}.null1", f"{pre}Null", mc[k1]["null_vS"],
                    ["mech_classes.json"], "resample", "median p05", "same-family null")
    except FileNotFoundError:
        pass

    # ---- Day-1 exact witness (restored 2026-08-19) ----
    gap, auc_r, auc_c, gap_c, er_r, er_c = d_reach()
    add("reach.gap_recompute", "ExactGapRecompute", gap, ["reachability.json", "reachability.py"],
        "planted pair", "max over 3 norms",
        "max |Delta Y| over the COMPLETE single-site family; 0 = bit-identical evidence")
    add("reach.auc_recompute", "ExactAUCRecompute", auc_r, ["reachability.json"], "planted pair",
        "max over 3 norms", "strongest classifier (GBT+MLP), chance = 0.5")
    add("reach.auc_clamped", "ExactAUCClamped", auc_c, ["reachability.json"], "planted pair",
        "min over 3 norms", "same classifier, clamped semantics")
    add("reach.gap_clamped", "ExactGapClamped", gap_c, ["reachability.json"], "planted pair",
        "min over 3 norms", "")
    add("reach.erank_recompute", "ExactERankRecompute", er_r, ["reachability.json"],
        "planted pair", "median", "effective rank of the reachable set")
    add("reach.erank_clamped", "ExactERankClamped", er_c, ["reachability.json"],
        "planted pair", "median", "")

    # ---- prediction section: positive control + random null ----
    try:
        z, w, lo, hi, rp1 = d_poscontrol()
        add("pc.rndp1", "PCRandPOne", rp1, ["positive_control.json"], "noise level",
            "min over noise levels", "random-direction 1st percentile of the subspace residual")
        add("pc.zero", "PCResidZero", z, ["positive_control.json"], "noise level", "value",
            "planted-generator recovery residual at zero noise")
        add("pc.worst", "PCResidWorst", w, ["positive_control.json"], "noise level", "max",
            "worst residual across the noise sweep")
        add("pc.rndlo", "PCRandLo", lo, ["positive_control.json"], "capture", "min over data",
            "random-direction p95")
        add("pc.rndhi", "PCRandHi", hi, ["positive_control.json"], "capture", "max over data",
            "random-direction p95")
    except (FileNotFoundError, KeyError, StopIteration):
        pass
    try:
        add("ioi.rmse_pct", "IOIRMSEAgreePct", d_ioi_rmse(), ["ioi_circuit.json"], "trial",
            "median", "percent RMSE gap between additive and interactive fits, held-out single-site")
    except (FileNotFoundError, KeyError):
        pass

    # ---- appendix: path patching + mediator interaction ----
    try:
        pP, pJ, sm, sp, cl, ch = d_path()
        add("path.P", "PathPblind", pP, ["path_controls.json"], "config", "median", "")
        add("path.J", "PathJblind", pJ, ["path_controls.json"], "config", "median", "")
        add("path.null_med", "PathNullMed", sm, ["path_controls.json"], "resample", "median", "")
        add("path.null_p95", "PathNullPNinetyFive", sp, ["path_controls.json"], "resample",
            "p95", "matched same-family null")
        add("path.ci_lo", "PathJCILo", cl, ["path_controls.json"], "config", "CI lower", "")
        add("path.ci_hi", "PathJCIHi", ch, ["path_controls.json"], "config", "CI upper", "")
    except (FileNotFoundError, KeyError):
        pass
    try:
        lo, hi, n = d_comm()
        add("comm.lo", "CommIntLo", lo, ["comm_compare.json"], "config", "min",
            "|INT|/sd over IOI configurations")
        add("comm.hi", "CommIntHi", hi, ["comm_compare.json"], "config", "max", "")
        add("comm.n", "CommN", f"{n}", ["comm_compare.json"], "config", "count", "")
    except (FileNotFoundError, KeyError):
        pass

    # ---- appendix: coordinate robustness ----
    try:
        for k, pre in (("u", "CoordU"), ("pc1", "CoordPCone"),
                       ("diff", "CoordDiff"), ("pc2d", "CoordPCtwod")):
            sS, sJ, both, n = d_coord()[k]
            add(f"coord.{k}.S", pre + "Sblind", sS, ["coord_robust.json"], "capture",
                "median", "")
            add(f"coord.{k}.J", pre + "Jblind", sJ, ["coord_robust.json"], "capture",
                "median", "")
            add(f"coord.{k}.both", pre + "Both", f"{both}/{n}", ["coord_robust.json"],
                "capture", "count", "")
    except (FileNotFoundError, KeyError):
        pass

    # ---- appendix: basis / class robustness ----
    try:
        for mk, mpre in (("GPT-2", "BGPT"), ("Qwen3.5", "BQwen")):
            for bn, bpre in (("poly", "Poly"), ("taylor", "Taylor"),
                             ("rbf", "Rbf"), ("fourier", "Fourier")):
                sS, sJ, both, n = d_basis(mk)[bn]
                add(f"basis.{mk}.{bn}", mpre + bpre + "Both", f"{both}/{n}",
                    ["controls_subspace.json"], "capture", "count", "")
                add(f"basis.{mk}.{bn}.S", mpre + bpre + "Sblind", sS,
                    ["controls_subspace.json"], "capture", "median", "")
    except (FileNotFoundError, KeyError):
        pass

    # ---- appendix: finite-sample ----
    try:
        fs = d_finite()
        for lab, pre in (("GPT-2", "FSGPT"), ("Qwen3.5-9B", "FSQwen")):
            if lab in fs:
                dk, gap, rate, ang = fs[lab]
                add(f"fs.{lab}.dk", pre + "DavisKahan", dk, ["finite_sample.json"],
                    "capture", "median", "vacuous when > sqrt(k) = 1.73")
                add(f"fs.{lab}.gap", pre + "EigengapRel", gap, ["finite_sample.json"],
                    "capture", "median", "relative eigengap at the blind-subspace boundary")
                add(f"fs.{lab}.rate", pre + "RateSlope", rate, ["finite_sample.json"],
                    "capture", "median", "log-log concentration slope; theory -0.5")
                add(f"fs.{lab}.ang", pre + "DirReproDeg", ang, ["finite_sample.json"],
                    "capture", "median", "S-blind direction agreement across disjoint halves")
    except (FileNotFoundError, KeyError):
        pass

    # ---- coordinate-invariant reporting (one cell per model x task; cells.py) ----
    for c in CELLS:
        key, pre = c["label"], c["pre"]
        try:
            r = load_json("invariant.json")[key]
        except (FileNotFoundError, KeyError):
            continue
        vs, vj, both, n, lo, hi = (r["vS_on_Sblind"], r["vJ_on_Jblind"], r["both_ends"],
                                   r["n"], r["null_lo"], r["null_hi"])
        add(f"inv.{key}.vS", pre + "VisSblind", 100.0 * vs, ["invariant.json", "invariant.py"],
            "capture", "median", "percent of POOLED energy under S, for the S-blind dir")
        add(f"inv.{key}.vJ", pre + "VisJblind", 100.0 * vj, ["invariant.json"], "capture",
            "median", "percent of pooled energy under J, for the J-blind dir")
        add(f"inv.{key}.both", pre + "BothEnds", f"{both}/{n}", ["invariant.json"], "capture",
            "count", "profile escapes the EXTREME-VALUE same-family null envelope at BOTH ends")
        add(f"inv.{key}.below", pre + "BelowEnv", f"{r['below']}/{n}", ["invariant.json"],
            "capture", "count", "S-blind end escapes the extreme-value envelope")
        add(f"inv.{key}.above", pre + "AboveEnv", f"{r['above']}/{n}", ["invariant.json"],
            "capture", "count", "J-blind end escapes the extreme-value envelope")
        add(f"inv.{key}.both_pw", pre + "BothEndsPointwise", f"{r['both_ends_pointwise']}/{n}",
            ["invariant.json"], "capture", "count",
            "pre-2026-08-27 pointwise envelope; anti-conservative, kept for the record")
        add(f"inv.{key}.env", pre + "NullEnvelope",
            f"[{lo:+.2f}, {hi:+.2f}]", ["invariant.json"], "resample",
            "2.5 pct of null minima / 97.5 pct of null maxima", "log10 held-out response ratio",
            raw=[lo, hi])
        add(f"inv.{key}.qS", pre + "QS", r["q_below"], ["invariant.json", "nulls.py"],
            "cell", "Holm-adjusted binomial p", "S-blind end, extreme-value envelope")
        add(f"inv.{key}.qJ", pre + "QJ", r["q_above"], ["invariant.json", "nulls.py"],
            "cell", "Holm-adjusted binomial p", "J-blind end, extreme-value envelope")
    vs, vj, both, n, lo, hi, cov, euc = d_invariant("Qwen3.5 IOI")
    add("inv.cov_reg", "CovRegInvariance", cov, ["invariant.json"], "capture", "median",
        "covariant regularizer G + lam*G_ref IS invariant")
    add("inv.euc_reg", "EucRegInvariance", euc, ["invariant.json"], "capture", "median",
        "Euclidean G + lam*I is NOT invariant")

    # ---- support-matched protocol (PRIMARY) vs the frozen one ----
    try:
        M = d_matched()
        short = {"GPT-2 ioi": "MGPTIOI", "GPT-2 induction": "MGPTInd", "GPT-2 capital": "MGPTCap",
                 "GPT-2 greater": "MGPTGt", "GPT-2 agreement": "MGPTAgr",
                 "Qwen3.5 ioi": "MQwenIOI", "Qwen3.5 induction": "MQwenInd",
                 "Qwen3.5 capital": "MQwenCap",
                 # third model family; omitting these made the appendix's "every cell" false
                 "Qwen3-8B ioi": "MQIOI", "Qwen3-8B induction": "MQInd"}
        for cell, pre in short.items():
            if cell not in M["cells"]:
                continue
            for pname, psuf in (("matched", "Matched"), ("frozen", "Frozen"),
                                ("interior", "Interior"), ("symmetric", "Symmetric")):
                r = M["cells"][cell].get(pname)
                if not r:
                    continue
                add(f"m.{cell}.{pname}.S", pre + psuf + "Sblind", f"{r['below']}/{r['n']}",
                    ["matched_invariant.json", "matched.py", "matched_analyse.py"], "capture",
                    "count", f"{pname} protocol; escapes the extreme-value same-family envelope")
                add(f"m.{cell}.{pname}.J", pre + psuf + "Jblind", f"{r['above']}/{r['n']}",
                    ["matched_invariant.json"], "capture", "count", f"{pname} protocol")
                add(f"m.{cell}.{pname}.qS", pre + psuf + "QS", r["q_below"],
                    ["matched_invariant.json"], "cell", "Holm-adjusted binomial p", "")
                add(f"m.{cell}.{pname}.qJ", pre + psuf + "QJ", r["q_above"],
                    ["matched_invariant.json"], "cell", "Holm-adjusted binomial p", "")
                add(f"m.{cell}.{pname}.box", pre + psuf + "Box", r["box_ratio"],
                    ["matched_invariant.json"], "capture", "median",
                    "single-site range / joint range in the chart; 1.0 = matched supports")
        for pname, psuf in (("matched", "Matched"), ("frozen", "Frozen"),
                            ("interior", "Interior"), ("symmetric", "Symmetric")):
            e = M["edge_vs_escape"].get(pname)
            if not e or "spearman_rho" not in e:
                continue
            add(f"m.edge.{pname}.n", "Edge" + psuf + "Escapes", f"{e['escapes']}/{e['n']}",
                ["matched_invariant.json"], "capture", "count", "reverse-blind escapes")
            add(f"m.edge.{pname}.rho", "Edge" + psuf + "Rho", e["spearman_rho"],
                ["matched_invariant.json"], "capture", "Spearman",
                "edge strength vs reverse-blind escape")
            add(f"m.edge.{pname}.p", "Edge" + psuf + "P", e["spearman_p"],
                ["matched_invariant.json"], "capture", "Spearman p", "")
            add(f"m.edge.{pname}.mwu", "Edge" + psuf + "MWU", e["mwu_p"],
                ["matched_invariant.json"], "capture", "Mann-Whitney p", "")
            add(f"m.edge.{pname}.med1", "Edge" + psuf + "MedEscaped", e["median_edge_escaped"],
                ["matched_invariant.json"], "capture", "median", "edge strength, escaped")
            add(f"m.edge.{pname}.med0", "Edge" + psuf + "MedNot", e["median_edge_not"],
                ["matched_invariant.json"], "capture", "median", "edge strength, not escaped")
    except (FileNotFoundError, KeyError):
        pass

    # ---- recompute vs clamped: the comparison Proposition 1 is actually about ----
    rc_pre = {"GPT-2 ioi": "RCGPTIOI", "GPT-2 induction": "RCGPTInd", "GPT-2 capital": "RCGPTCap",
              "GPT-2 greater": "RCGPTGt", "GPT-2 agreement": "RCGPTAgr",
              "Qwen3.5 ioi": "RCQwenIOI", "Qwen3.5 induction": "RCQwenInd",
              "Qwen3.5 capital": "RCQwenCap"}
    for chart, csuf in (("edge", ""), ("u", "Uchart"), ("edge_holdout", "Holdout"),
                        ("edge_x", "Xseed")):
        try:
            R = load_json(f"recompute_vs_clamped_{chart}.json")
        except FileNotFoundError:
            continue
        src = [f"recompute_vs_clamped_{chart}.json", "recompute_vs_clamped.py"]
        for cell, pre in rc_pre.items():
            if cell not in R:
                continue
            r = R[cell]
            p_ = pre + csuf
            add(f"rc{csuf}.{cell}.R", p_ + "Rblind", f"{r['below']}/{r['n']}", src, "capture",
                "count", "recompute-blind direction escapes the same-family envelope")
            add(f"rc{csuf}.{cell}.C", p_ + "Cblind", f"{r['above']}/{r['n']}", src, "capture",
                "count", "clamped-blind direction escapes")
            add(f"rc{csuf}.{cell}.qR", p_ + "QR", r["q_below"], src, "cell",
                "Holm-adjusted binomial p", "")
            add(f"rc{csuf}.{cell}.qC", p_ + "QC", r["q_above"], src, "cell",
                "Holm-adjusted binomial p", "")
            add(f"rc{csuf}.{cell}.gR", p_ + "GRvisC", r["median_gR_ratio"], src, "capture",
                "median", "g_R response under clamped / under recompute evidence")
            add(f"rc{csuf}.{cell}.gC", p_ + "GCvisR", r["median_gC_ratio"], src, "capture",
                "median", "g_C response under recompute / under clamped evidence")
        scr = R.get("_screen")
        if scr:
            add(f"rc{csuf}.dropped", "RC" + csuf + "Dropped", f"{scr['n_dropped']}",
                src, "capture", "count",
                "captures excluded by the pre-stated chart validity screen")
        sc, ch = R.get("_scope"), R.get("_chart")
        if sc:
            add(f"rc{csuf}.mutual", "RC" + csuf + "Mutual",
                f"{sc['cells_mutual']}/{sc['n_cells']}", src, "cell", "count",
                "cells showing mutual non-domination")
            add(f"rc{csuf}.pooledR", "RC" + csuf + "PooledR",
                f"{sc['pooled_below']}/{sc['pooled_n']}", src, "capture", "count", "")
            add(f"rc{csuf}.pooledC", "RC" + csuf + "PooledC",
                f"{sc['pooled_above']}/{sc['pooled_n']}", src, "capture", "count", "")
        if ch:
            for k, mac in (("median_abs_m", "M"), ("median_cos", "Cos"), ("median_r2", "RTwo"),
                           ("median_sd_ratio", "SdRatio")):
                add(f"rc{csuf}.chart.{k}", "RC" + csuf + "Chart" + mac, ch[k], src, "capture",
                    "median", "chart diagnostic")

    # ---- the published mechanism's own chart (weights only, no fitting) ----
    try:
        Q = load_json("ioi_query_chart.json")
        qs, qc = Q["summary"], Q["calibration"]
        src = ["ioi_query_chart.json", "ioi_query_chart.py"]
        for k, mac, agg, note in (
                ("n", "QCN", "count", "screened configurations of published S-Inhib -> Name Mover heads"),
                ("info_ratio", "QCInfoRatio", "median",
                 "joint / single-site partial Fisher information for the interaction"),
                ("se_ratio", "QCSeRatio", "median", "se under single-site / under joint evidence"),
                ("info_S", "QCInfoS", "median", ""), ("info_J", "QCInfoJ", "median", ""),
                ("edge_r2", "QCEdgeRTwo", "median", "realised edge in the mechanism's own chart"),
                ("signflip_S", "QCFlipS", "median", ""), ("signflip_J", "QCFlipJ", "median", "")):
            add(f"qc.{k}", mac, qs[k], src, "configuration", agg, note)
        add("qc.rejJ", "QCRejJ", f"{qs['wald_rej_J']}/{qs['n']}", src, "configuration", "count",
            "joint evidence rejects zero interaction (Wald chi2_3, 0.05)")
        add("qc.rejS", "QCRejS", f"{qs['wald_rej_S']}/{qs['n']}", src, "configuration", "count",
            "single-site evidence rejects; reported because it does NOT favour our reading")
        add("qc.calS", "QCCalS", 100.0 * qc["rate_S"], src, "shuffle", "rejection rate",
            "single-site Wald rejection rate on SHUFFLED readouts; nominal 5")
        add("qc.calJ", "QCCalJ", 100.0 * qc["rate_J"], src, "shuffle", "rejection rate",
            "the same under joint evidence")
        # NEGATIVE CONTROL: the identical procedure on head pairs with no claimed mechanism
        try:
            K = load_json("ioi_query_chart_control.json")["rows"]
            csrc = ["ioi_query_chart_control.json", "ioi_query_chart.py"]
            med = lambda R, k: float(np.median([r[k] for r in R]))
            add("qc.ctrl.n", "QCCtrlN", f"{len(K)}", csrc, "configuration", "count",
                "control configurations: early heads with no claimed role in IOI")
            add("qc.ctrl.sd", "QCCtrlEdge", med(K, "sd_ratio"), csrc, "configuration", "median",
                "realised edge for control head pairs")
            add("qc.ctrl.info", "QCCtrlInfoRatio", med(K, "info_ratio"), csrc, "configuration",
                "median", "information gap for control pairs -- LARGER, as the framework predicts")
            add("qc.ctrl.rejJ", "QCCtrlRejJ",
                f"{sum(r['wald_p_J'] < 0.05 for r in K)}/{len(K)}", csrc, "configuration",
                "count", "joint rejects, control")
            add("qc.ctrl.rejS", "QCCtrlRejS",
                f"{sum(r['wald_p_S'] < 0.05 for r in K)}/{len(K)}", csrc, "configuration",
                "count", "single-site rejects, control")
            from scipy.stats import mannwhitneyu as _mwu
            P = load_json("ioi_query_chart.json")["rows"]
            add("qc.ctrl.edgeratio", "QCEdgeVsCtrl",
                med(P, "sd_ratio") / max(med(K, "sd_ratio"), 1e-300), csrc + ["ioi_query_chart.json"],
                "configuration", "ratio of medians",
                "published-head edge relative to control heads")
            add("qc.ctrl.p", "QCEdgeVsCtrlP",
                float(_mwu([r["sd_ratio"] for r in P], [r["sd_ratio"] for r in K],
                           alternative="greater").pvalue),
                csrc + ["ioi_query_chart.json"], "configuration", "Mann-Whitney p", "")
        except (FileNotFoundError, KeyError):
            pass
    except (FileNotFoundError, KeyError):
        pass

    # ---- 2026-09-23 post-review reanalyses (each re-derived by an independent checker) ----
    try:
        DC = load_json("detection_covariance.json")
        dsrc = ["detection_covariance.json", "detection_covariance.py"]
        rob = ("hc0", "hc3", "arm", "fgls", "cr1")
        io, ct = DC["ioi"]["summary"], DC["control"]["summary"]
        add("det.ioi.roblo", "DetIOIRobLo", f"{min(io['rej_S'][e] for e in rob)}", dsrc, "configuration", "min over 5 robust covariances", "single-site rejections, IOI")
        add("det.ioi.robhi", "DetIOIRobHi", f"{max(io['rej_S'][e] for e in rob)}", dsrc, "configuration", "max over 5 robust covariances", "")
        add("det.ctl.roblo", "DetCtrlRobLo", f"{min(ct['rej_S'][e] for e in rob)}", dsrc, "configuration", "min over 5 robust covariances", "single-site rejections, controls")
        add("det.ctl.robhi", "DetCtrlRobHi", f"{max(ct['rej_S'][e] for e in rob)}", dsrc, "configuration", "max over 5 robust covariances", "")
        add("det.power", "DetPowerMin", io["min_power_J2way_at_cS_among_rej_S"], dsrc, "configuration", "min over single-site rejections",
            "power of the joint Wald test at the single-site estimate, conservative two-way donor-cluster covariance")
        add("det.ioi.j2", "DetIOIJTwo", f"{io['rej_J']['cr2way']}/{io['n']}", dsrc, "configuration", "count",
            "joint-family rejections of zero interaction, conservative two-way donor-cluster covariance, published heads")
        add("det.ctl.j2", "DetCtrlJTwo", f"{ct['rej_J']['cr2way']}/{ct['n']}", dsrc, "configuration", "count", "same, control heads")
    except (FileNotFoundError, KeyError):
        pass
    try:
        RD = load_json("response_diagnostics.json")
        rsrc = ["response_diagnostics.json", "response_diagnostics.py"]
        sens = RD["heldout_partition_sensitivity"]["3"]
        add("resp.ss", "RespSSR", sens["S->S"]["r2_median_over_seeds"], rsrc, "capture", "median over 10 donor partitions of the capture median", "held-out R^2, single-site fit on single-site states")
        add("resp.jj", "RespJJR", sens["J->J"]["r2_median_over_seeds"], rsrc, "capture", "same", "held-out R^2, joint fit on joint states")
        add("resp.sj", "RespSJR", sens["S->J"]["r2_median_over_seeds"], rsrc, "capture", "same", "held-out R^2, single-site fit on joint states")
        AS = RD["heldout_additive_partition_sensitivity"]
        asj = [AS[d]["S->J"]["r2_median_over_seeds"] for d in ("2", "3", "4")]
        add("resp.addsj.lo", "RespAddSJLo", f"{min(asj):.2f}", rsrc, "capture", "min over degrees 2-4 of the median over 10 partitions", "held-out joint R^2 of the ADDITIVE block fitted on single-site states", raw=min(asj))
        add("resp.addsj.hi", "RespAddSJHi", f"{max(asj):.2f}", rsrc, "capture", "max over degrees 2-4, same", "same", raw=max(asj))
        add("resp.addjj", "RespAddJJR", AS["3"]["J->J"]["r2_median_over_seeds"], rsrc, "capture", "median over 10 partitions", "held-out joint R^2 of the additive block fitted on joint states, degree 3")
        beats = sum(AS[d]["J->J"]["r2_median_over_seeds"] >= RD["heldout_partition_sensitivity"][d]["J->J"]["r2_median_over_seeds"] for d in ("2", "3", "4"))
        add("resp.addjj.beats", "RespAddJJBeats", f"{beats}/3", rsrc, "degree", "count over degrees 2-4", "degrees at which the additive joint fit predicts held-out joint readouts at least as well as the full class")
        for fam, tag in (("S->S", "SS"), ("S->J", "SJ"), ("J->J", "JJ"), ("J->S", "JS")):
            add(f"resp.add3.{tag}", f"RespAddThree{tag}", f"{AS['3'][fam]['r2_median_over_seeds']:.2f}", rsrc, "capture",
                "median over 10 partitions of the capture median", f"additive-only refit, degree 3, {fam}", raw=AS['3'][fam]['r2_median_over_seeds'])
            MSL = load_json("model_selection_local.json")      # ridge control with family-local, fold-local whitening (the fix)
            v_ = MSL["ridge_fix_published"]["3"][fam]["local"]
            add(f"resp.rdg3.{tag}", f"RespRidgeThree{tag}", f"{v_:.2f}", ["model_selection_local.json", "model_selection_local.py"],
                "capture", "median over 10 partitions of the capture median", f"CV ridge on the interaction block, family-local fold-local whitening, degree 3, {fam}", raw=v_)
            add(f"resp.full3.{tag}", f"RespFullThree{tag}", f"{RD['heldout_partition_sensitivity']['3'][fam]['r2_median_over_seeds']:.2f}", rsrc,
                "capture", "median over 10 partitions of the capture median", f"full cubic, {fam}", raw=RD['heldout_partition_sensitivity']['3'][fam]['r2_median_over_seeds'])
        RG = RD["heldout_ridge_cross_partition_sensitivity"]
        rsj = [RG[d]["r2"]["S->J"]["r2_median_over_seeds"] for d in ("2", "3", "4")]
        add("resp.ridgesj.max", "RespRidgeSJMax", f"{max(rsj):.2f}", rsrc, "capture", "max over degrees 2-4 of the median over 10 partitions",
            "held-out joint R^2 of the single-site fit with a CV-chosen ridge penalty on the interaction block", raw=max(rsj))
        add("resp.ridgelam.s", "RespRidgeLamS", f"{RG['3']['lambda_median']['S']:.1f}", rsrc, "capture x partition", "median", "CV-chosen interaction penalty, single-site fits, degree 3", raw=RG["3"]["lambda_median"]["S"])
        add("resp.ridgelam.j", "RespRidgeLamJ", f"{RG['3']['lambda_median']['J']:.1e}".replace("e+0", "\\times10^{") + "}", rsrc, "capture x partition", "median",
            "CV-chosen interaction penalty, joint fits, degree 3", raw=RG["3"]["lambda_median"]["J"])
        add("resp.sjneg.min", "RespSJNegMin", min(RD["heldout_r2_negative_by_seed"]["3"]["S->J"]), rsrc, "capture", "min over 10 partitions",
            "captures with negative held-out joint R^2 for the single-site cubic fit, fewest over partitions")
        neg = RD["heldout"]["3"]["S->J"]["r2_negative"]
        add("resp.sjneg", "RespSJNeg", f"{neg}/{RD['n_captures']}", rsrc, "capture", "count", "captures where the single-site fit's held-out joint R^2 < 0")
        env = RD["envelope"]
        add("env.s2", "EnvSDtwo", env["2"]["med_S_pct"], rsrc, "capture", "median", "single-site transport envelope, degree 2, % of B-arm swing")
        add("env.s4", "EnvSDfour", env["4"]["med_S_pct"], rsrc, "capture", "median", "same, degree 4")
        add("env.rmin", "EnvRatioMin", min(env[d]["ratio"] for d in ("2", "3", "4")), rsrc, "capture", "min over degrees 2-4 of the ratio of medians", "")
        add("env.ord", "EnvOrderedAll", "yes" if all(env[d]["ordered"] == env[d]["n"] for d in ("2", "3", "4")) else "no", rsrc, "capture", "all degrees", "single-site envelope larger in every capture at every degree")
    except (FileNotFoundError, KeyError):
        pass
    try:   # simplicity-aware model selection within the single-site family (exploratory; e0/model_selection_local.py)
        MS = load_json("model_selection_local.json")
        ssrc = ["model_selection_local.json", "model_selection_local.py"]
        s3 = MS["summary"]["published"]["3"]; n3 = int(s3["n"]["seed0"])
        g_ = lambda k: s3[k]["seed0"]
        add("ms.onese", "MsOneSEA", f"{int(g_('select.one_se.A'))}/{n3}", ssrc, "capture", "count, seed-0 partition", "one-SE rule selects the additive-only model")
        add("ms.paired", "MsPairedA", f"{int(g_('select.paired.A'))}/{n3}", ssrc, "capture", "count, seed-0 partition", "paired donor-level rule selects the additive-only model")
        add("ms.mincv", "MsMinCvInter", f"{int(g_('select.min_cv.R') + g_('select.min_cv.F'))}/{n3}", ssrc, "capture", "count, seed-0 partition",
            "minimum-CV selection picks a model with interaction terms (ridge or full)")
        add("ms.resF", "MsResolvedF", f"{int(g_('resolved_F'))}/{n3}", ssrc, "capture", "count", "paired F-vs-A difference resolved in favour of F")
        add("ms.gainse", "MsGainSE", f"{g_('cv_gain_mincv_vs_A_over_se.median'):.2f}", ssrc, "capture", "median", "min-CV gain over additive, in donor-level SEs", raw=g_("cv_gain_mincv_vs_A_over_se.median"))
        for r_, t_ in (("min_cv", "MinCv"), ("one_se", "OneSE"), ("paired", "Paired")):
            add(f"ms.{r_}.rj", f"Ms{t_}RJ", f"{s3[r_ + '.R2_J']['seeds1_9_median']:.2f}", ssrc, "capture", "median over partitions 1-9 of the capture median",
                f"held-out joint R^2 of the {r_} selection", raw=s3[r_ + ".R2_J"]["seeds1_9_median"])
            add(f"ms.{r_}.t", f"Ms{t_}T", f"{s3[r_ + '.RMSE_T']['seeds1_9_median']:.3f}", ssrc, "capture", "same",
                f"held-out contrast RMSE of the {r_} selection, logits", raw=s3[r_ + ".RMSE_T"]["seeds1_9_median"])
    except (FileNotFoundError, KeyError):
        pass
    try:   # model-free controlled four-point transport contrast (exploratory; e0/fourpoint_modelfree.py)
        MF = load_json("fourpoint_modelfree.json")
        msrc = ["fourpoint_modelfree.json", "fourpoint_modelfree.py"]
        P_, C_ = MF["groups"]["published"]["summary"], MF["groups"]["controls"]["summary"]
        n_ = P_["n"]
        add("mf.margin", "MfMargin", f"{MF['margin_logits']:.2f}", msrc, "-", "declared", "RMS margin, logits", raw=MF["margin_logits"])
        add("mf.rms.med", "MfRmsMed", f"{P_['rms_T_grid']['median']:.3f}", msrc, "capture", "median over captures",
            "RMS of the measured controlled four-point contrast over the recorded joint grid, logits", raw=P_["rms_T_grid"]["median"])
        add("mf.rms.max", "MfRmsMax", f"{P_['rms_T_grid']['max']:.3f}", msrc, "capture", "max over captures", "same", raw=P_["rms_T_grid"]["max"])
        add("mf.rms.pct", "MfRmsPct", f"{P_['rms_T_grid_pct_swing']['median']:.2f}", msrc, "capture", "median over captures",
            "same, % of the B-arm readout swing", raw=P_["rms_T_grid_pct_swing"]["median"])
        add("mf.maxabs", "MfMaxAbsMax", f"{P_['max_abs_T_grid']['max']:.2f}", msrc, "capture x row", "max", "largest single measured |contrast|, logits", raw=P_["max_abs_T_grid"]["max"])
        add("mf.below", "MfBelow", f"{P_['grid_below_margin']}/{n_}", msrc, "capture", "count", "captures whose grid RMS is below the margin")
        add("mf.upper", "MfUpperBelow", f"{P_['upper95_below_margin']}/{n_}", msrc, "capture", "count",
            "captures whose nominal donor-conditional 95% upper bound (held-out donors) is below the margin")
        add("mf.zero", "MfZero", f"{P_['rms_T_heldout']['median']:.3f}", msrc, "capture", "median", "held-out RMS error of predicting zero, logits", raw=P_["rms_T_heldout"]["median"])
        for d_ in ("2", "3", "4"):
            w_ = {"2": "Two", "3": "Three", "4": "Four"}[d_]
            v_ = P_[f"rmse_heldout_d{d_}_S"]["median"]
            add(f"mf.s{d_}", f"MfS{w_}", f"{v_:.2f}" if v_ < 100 else f"{v_:.0f}", msrc, "capture", "median",
                f"held-out RMS error of the single-site-fitted degree-{d_} model's contrast, logits", raw=v_)
            v_ = P_[f"rmse_heldout_d{d_}_J"]["median"]
            add(f"mf.j{d_}", f"MfJ{w_}", f"{v_:.3f}", msrc, "capture", "median", f"same, joint-fitted degree-{d_} model", raw=v_)
        add("mf.sworse", "MfSWorse", f"{P_['S_worse_than_zero']}/{n_}", msrc, "capture", "count",
            "captures where the single-site cubic's held-out contrast error exceeds zero's (95% interval of the difference above 0)")
        add("mf.jworse", "MfJWorse", f"{P_['J_worse_than_zero']}/{n_}", msrc, "capture", "count", "same, joint cubic")
        add("mf.ctrl.rms", "MfCtrlRmsMed", f"{C_['rms_T_grid']['median']:.4f}", msrc, "capture", "median", "controls: grid RMS of the measured contrast, logits", raw=C_["rms_T_grid"]["median"])
        add("mf.ctrl.below", "MfCtrlBelow", f"{C_['grid_below_margin']}/{C_['n']}", msrc, "capture", "count", "controls below the margin")
        ps_ = P_["partition_sensitivity"]
        add("mf.part.s", "MfPartS", f"{ps_['S']['median']:.1f}", msrc, "capture", "median over captures of the median over partitions 1-9", "single-site cubic contrast error", raw=ps_["S"]["median"])
        add("mf.part.j", "MfPartJ", f"{ps_['J']['median']:.3f}", msrc, "capture", "same", "joint cubic contrast error", raw=ps_["J"]["median"])
        add("mf.part.t", "MfPartT", f"{ps_['rms_T']['median']:.3f}", msrc, "capture", "same", "zero predictor", raw=ps_["rms_T"]["median"])
        # the denominator of every "% of the B-arm swing": a measured readout range, not an estimated effect
        sw_, thr_key = P_["swing"], "swing_below_0p5"
        thr_ = float(thr_key.rsplit("_", 1)[1].replace("p", "."))   # threshold as fixed in fourpoint_modelfree.py
        add("mf.swing.min", "MfSwingMin", f"{sw_['min']:.2f}", msrc, "capture", "min over captures",
            "B-arm readout swing ptp(Ys[B arm]), the denominator of every % of the swing, logits", raw=sw_["min"])
        add("mf.swing.med", "MfSwingMed", f"{sw_['median']:.2f}", msrc, "capture", "median over captures", "same", raw=sw_["median"])
        add("mf.swing.thr", "MfSwingThr", f"{thr_:g}", msrc, "-", "declared", f"threshold behind {thr_key}, logits", raw=thr_)
        add("mf.swing.below", "MfSwingBelow", f"{P_[thr_key]}/{n_}", msrc, "capture", "count",
            "captures whose B-arm swing is below that threshold")
        clean_ = {round(float(np.load(f)["clean"]), 4) for f in files("qchart_*.npz")}
        add("mf.prompts", "MfBasePrompts", f"{len(clean_)}", ["qchart_*.npz"], "capture", "count of distinct clean readouts",
            "base prompts shared by the published captures (the seed fixes the prompt; the clean run does not depend on the head pair)")
    except (FileNotFoundError, KeyError):
        pass
    try:
        PCM = load_json("prompt_cluster_matched.json")
        psrc = ["prompt_cluster_matched.json", "prompt_cluster_matched.py", "matched_invariant.json"]
        m_, f_, s_ = PCM["matched"], PCM["frozen"], PCM["symmetric"]
        add("pcm.s.all", "PcMatchedSAll", f"{m_['below']['prompts_all_escape']}/{m_['below']['prompts']}", psrc, "prompt", "count",
            "matched protocol, S-blind end: prompts on which every capture escapes")
        add("pcm.s.maj", "PcMatchedSMaj", f"{m_['below']['prompts_majority_escape']}/{m_['below']['prompts']}", psrc, "prompt", "count",
            "matched protocol, S-blind end: prompts on which a majority of captures escape")
        e = int(np.floor(np.log10(m_['below']['p_prompt_all'])))
        add("pcm.s.p", "PcMatchedSP", f"<10^{{{e + 1}}}", psrc, "prompt", "one-sided binomial vs 2.5%",
            "strict prompt count, matched protocol, S-blind end", raw=m_['below']['p_prompt_all'])
        add("pcm.j.maj", "PcMatchedJMaj", f"{m_['above']['prompts_majority_escape']}/{m_['above']['prompts']}", psrc, "prompt", "count",
            "matched protocol, reverse end: prompts on which a majority of captures escape")
        add("pcf.j.maj", "PcFrozenJMaj", f"{f_['above']['prompts_majority_escape']}/{f_['above']['prompts']}", psrc, "prompt", "count",
            "unmatched (joint at 1) protocol, reverse end: prompts on which a majority of captures escape")
        add("pcs.j.maj", "PcSymJMaj", f"{s_['above']['prompts_majority_escape']}/{s_['above']['prompts']}", psrc, "prompt", "count",
            "symmetric protocol, reverse end: prompts on which a majority of captures escape")
        ppc = list(m_['below']['prompts_per_cell'].values())
        add("pcm.ppc", "PcPromptsPerCell", f"{min(ppc)}--{max(ppc)}", psrc, "prompt", "min-max over cells", "prompts per model x task cell")
    except (FileNotFoundError, KeyError):
        pass
    try:
        OB = load_json("ofat_baseline.json")
        osrc = ["ofat_baseline.json", "ofat_baseline.py"]
        pr, pdn = OB["pooled"]["row"], OB["pooled"]["donor"]
        add("ofat.row", "OfatRowCount", f"{pr['gC_gt10']}/{pr['n']}", osrc, "capture", "count", "clamped generator above 10x, held-out rows")
        add("ofat.donor", "OfatDonorCount", f"{pdn['gC_gt10']}/{pdn['n']}", osrc, "capture", "count", "same, strict donor split")
        add("ofat.inc", "OfatIncrement", f"{pr['gR_gt10_gC_le10']}", osrc, "capture", "count", "captures only the recompute generator clears")
        add("ofat.onlyc", "OfatOnlyClamped", f"{pr['gC_gt10_gR_le10']}", osrc, "capture", "count", "captures only the clamped generator clears")
        for k_, m_ in (("gC_beats_euclid", "OfatGCBeatsE"), ("gC_beats_inv", "OfatGCBeatsI"), ("gR_beats_inv", "OfatGRBeatsI")):
            add(f"ofat.{k_}", m_, f"{pr[k_]}/{pr['n']}", osrc, "capture", "count", f"{k_}: beats the random-direction null p95")
        add("ofat.ratio", "OfatRatioMedian", pr["gR_over_gC_median"], osrc, "capture", "median", "recompute / clamped held-out ratio")
        add("ofat.rho", "OfatSpearman", pr["spearman_abs_m_whitened_vs_gR_over_gC"][0], osrc, "capture", "Spearman", "realised edge slope vs recompute/clamped advantage")
        for pre, cell in OB["cells"].items():
            add(f"ofat.{pre}", pre + "MOfat", f"{cell['summary']['row']['gC_gt10']}/{cell['summary']['row']['n']}" if 'summary' in cell else f"{cell['row']['gC_gt10']}/{cell['row']['n']}",
                osrc, "capture", "count", "clamped generator above 10x, held-out rows")
        add("ofat.prompt", "OfatPromptStrict", f"{OB['prompt_level']['row']['gC_strict']}/{OB['prompt_level']['row']['prompts']}", osrc, "(cell, prompt) group", "count", "clamped generator, every capture of the prompt above 10x")
    except (FileNotFoundError, KeyError):
        pass
    try:
        PS = load_json("pseudorep.json")
        psrc = ["pseudorep.json", "pseudorep.py"]
        hl = PS["head_level"]
        add("head.edge", "HeadEdgeRatio", 10 ** hl["log_edge"]["head_mean_diff"], psrc, "source head", "10^(mean log10 difference)", "IOI vs control realised edge, head as the unit")
        rs = list(hl["rej_S"]["head_values"]["ioi"].values())
        add("head.rejlo", "HeadRejIOILo", 100 * min(rs), psrc, "source head", "min over IOI heads", "single-site rejection rate, %")
        add("head.rejhi", "HeadRejIOIHi", 100 * max(rs), psrc, "source head", "max over IOI heads", "")
        add("head.sep", "HeadCompleteSep", "yes" if hl["log_edge"]["complete_separation"] and hl["rej_S"]["complete_separation"] else "no", psrc, "source head", "", "")
    except (FileNotFoundError, KeyError):
        pass

    # ---- the identification gap as a function of the realised edge (three head groups) ----
    try:
        CS = load_json("circuits_summary.json")
        src = ["circuits_summary.json", "circuits_summary.py", "ioi_query_chart.py"]
        for g, pre in (("ioi", "GrpIOI"), ("induction", "GrpInd"), ("control", "GrpCtrl")):
            v = CS["groups"][g]
            add(f"grp.{g}.n", pre + "N", f"{v['n']}", src, "configuration", "count", "")
            add(f"grp.{g}.edge", pre + "Edge", v["edge"], src, "configuration", "median",
                "realised edge sd(b|arm A)/sd(b|arm B) in the circuit's own chart")
            add(f"grp.{g}.info", pre + "InfoRatio", v["info_ratio"], src, "configuration",
                "median", "joint / single-site partial Fisher information")
            add(f"grp.{g}.se", pre + "SeRatio", v["se_ratio"], src, "configuration", "median", "")
            add(f"grp.{g}.infoJ", pre + "InfoJ", v["info_J"], src, "configuration", "median",
                "joint-family partial information; near-constant across groups")
            add(f"grp.{g}.rejJ", pre + "RejJ", f"{v['rej_J']}/{v['n']}", src, "configuration",
                "count", "joint rejects zero interaction")
            add(f"grp.{g}.rejS", pre + "RejS", f"{v['rej_S']}/{v['n']}", src, "configuration",
                "count", "single-site rejects")
            add(f"grp.{g}.tJ", pre + "TJ", v["t_J"], src, "configuration", "median",
                "|c|/se of the interaction under JOINT evidence; ~1 = measured, and zero")
            nz = CS.get("additive_null", {}).get(g)
            if nz:
                add(f"grp.{g}.nullS", pre + "NullSizeS", 100 * nz["size_S"], src,
                    "bootstrap draw", "percent",
                    "size of the single-site Wald test under the ADDITIVE null (nominal 5%)")
                add(f"grp.{g}.nullJ", pre + "NullSizeJ", 100 * nz["size_J"], src,
                    "bootstrap draw", "percent", "same, joint evidence")
        add("grp.n", "GrpN", f"{CS['n']}", src, "configuration", "count",
            "pooled over three independent head groups")
        add("grp.rho", "GrpRho", CS["rho_info_ratio"], src, "configuration", "Spearman",
            "realised edge vs the identification gap")
        add("grp.p", "GrpP", CS["p_info_ratio"], src, "configuration", "Spearman p", "")
        add("grp.rhose", "GrpRhoSe", CS["rho_se_ratio"], src, "configuration", "Spearman", "")
        add("grp.rhodet", "GrpRhoDetect", CS["rho_detect"], src, "configuration", "Spearman",
            "realised edge vs whether single-site evidence detects the interaction")
        add("grp.pdet", "GrpPDetect", CS["p_detect"], src, "configuration", "Spearman p", "")
    except (FileNotFoundError, KeyError):
        pass

    # ---- the k-site bound: evidence grows linearly, the class polynomially ----
    try:
        K = load_json("ksite.json")["summary"]
        for kk in (2, 3, 4, 5):
            add(f"ksite.blind{kk}", f"KBlind{'Two Three Four Five'.split()[kk-2]}",
                100.0 * K[f"blind_frac_k{kk}_d3"], ["ksite.json", "ksite.py"], "site count",
                "exact (population)",
                "percent of the degree-3 class invisible to single-site evidence at k sites")
        add("ksite.joint", "KJointBlind", f"{K['joint_blind_k5_d3']}", ["ksite.json"],
            "site count", "exact", "blind dimension of the joint product grid at k=5, degree 3")
        try:
            CE = load_json("ksite_chain.json")["summary"]["counterexample"]
            add("ksite.ce.k", "KCEk", f"{CE['k']}", ["ksite_chain.json", "ksite.py"],
                "configuration", "exact", "counterexample to the PER-EDGE form of the bound")
            add("ksite.ce.e", "KCEe", f"{CE['e']}", ["ksite_chain.json"], "configuration", "exact", "")
            add("ksite.ce.d", "KCEd", f"{CE['d']}", ["ksite_chain.json"], "configuration", "exact", "")
            add("ksite.ce.bound", "KCEBound", f"{1 + CE['d'] * CE['k'] * CE['e']}", ["ksite_chain.json"], "configuration", "exact", "Prop. 6 bound 1 + d*sum(e_i) evaluated with the per-edge degree (the naive sum k(de+1) recorded in the artifact omits the k-1 agreement constraints)")
            add("ksite.ce.rank", "KCERank", f"{CE['true_rank']}", ["ksite_chain.json"],
                "configuration", "exact", "the actual rank on a chained circuit")
            KX = load_json("ksite_exact.json")["summary"]
            add("ksite.ce.xrank", "KCERankExact", f"{KX['counterexample']['exact_rank_shared_edges']}", ["ksite_exact.json", "ksite_exact.py"],
                "configuration", "exact rank over F_p, generic coefficients, edges shared across arms", "the chain counterexample's exact rank")
            add("ksite.ce.xrankpa", "KCERankExactPerArm", f"{KX['counterexample']['exact_rank_per_arm_edges']}", ["ksite_exact.json"],
                "configuration", "exact rank, per-arm edges as in ksite.py", "")
            add("ksite.ce.cb", "KCEComposedBound", f"{KX['counterexample']['composed_bound']}", ["ksite_exact.json"], "configuration", "exact", "")
            add("ksite.exact.holds", "KBoundHoldsExact", "yes" if KX["all_hold"] else "NO", ["ksite_exact.json"], "configuration",
                "every k<=5, e<=2, d<=6 configuration, three constructions", "Proposition 6 bound verified with exact ranks")
            add("ksite.exact.viol", "KPerEdgeViolations", f"{KX['n_per_edge_violations']}", ["ksite_exact.json"], "configuration", "count",
                "chain configurations in which the per-edge form of the bound fails")
        except (FileNotFoundError, KeyError, TypeError):
            pass
        add("ksite.holds", "KBoundHolds", "yes" if K["all_hold"] else "NO", ["ksite.json"],
            "configuration", "all of 40", "rank bound verified numerically")
    except (FileNotFoundError, KeyError):
        pass

    # ---- three sites in a trained model ----
    try:
        T3 = load_json("threesite.json")["summary"]
        for dg, word in (("deg2", "Two"), ("deg3", "Three")):
            if dg not in T3:
                continue
            t = T3[dg]
            add(f"three.{dg}.below", f"ThreeDeg{word}Sblind", f"{t['below']}/{t['n']}",
                ["threesite.json", "threesite.py"], "site triple", "count",
                "escapes the extreme-value same-family envelope at the S-blind end")
            add(f"three.{dg}.above", f"ThreeDeg{word}Reverse", f"{t['above']}/{t['n']}",
                ["threesite.json"], "site triple", "count", "reverse end")
            add(f"three.{dg}.p", f"ThreeDeg{word}P", t["p_below"], ["threesite.json"],
                "cell", "binomial p", "")
            add(f"three.{dg}.dim", f"ThreeDeg{word}Dim", f"{t['dim_class']}",
                ["threesite.json"], "class", "exact", "dimension of the degree-d class at k=3")
            add(f"three.{dg}.logr", f"ThreeDeg{word}LogRatio", -t["median_min_logr"],
                ["threesite.json"], "site triple", "median",
                "log10 of how much less visible the least-visible direction is under single-site")
    except (FileNotFoundError, KeyError):
        pass

    # ---- Theorem 1's non-degeneracy hypothesis, measured ----
    try:
        E = d_edge()
        for k, macro, unit, agg, note in (
                ("spearman_cos_rho", "EdgeCosRho", "capture", "Spearman",
                 "measured edge slope vs |cos(g_R, g_C)|"),
                ("spearman_cos_p", "EdgeCosP", "capture", "Spearman p", ""),
                ("spearman_sep_rho", "EdgeSepRho", "capture", "Spearman",
                 "edge slope vs sep(g_R)/sep(g_C)"),
                ("spearman_sep_p", "EdgeSepP", "capture", "Spearman p", ""),
                ("weak_median_cos", "EdgeWeakCos", "capture", "median", "weak edge"),
                ("strong_median_cos", "EdgeStrongCos", "capture", "median", "strong edge"),
                ("strong_median_gR_over_gC", "EdgeStrongRatio", "capture", "median",
                 "how much more blind g_R is than g_C under recompute evidence"),
                ("strong_median_sep_gC", "EdgeStrongSepGC", "capture", "median",
                 "the clamped generator's separation where the edge is strong")):
            if E.get(k) is not None:
                add(f"edge.{k}", macro, E[k], ["edge_strength_matched.json", "edge_strength_matched.py"],
                    unit, agg, note)
        add("edge.n", "EdgeN", f"{E['n']}", ["edge_strength_matched.json"], "capture", "count",
            "support-matched captures, training-only estimation")
        for k, macro, note in (("weak_median_sep_gC", "EdgeWeakSepGC", "clamped generator separation, flat edges"),
                               ("strong_median_rnd_p95", "EdgeStrongRndP", "random-direction 95th percentile, strong edges")):
            add(f"edge.{k}", macro, E[k], ["edge_strength_matched.json"], "capture", "median", note)
        add("edge.strong_gc_beats", "EdgeStrongGCBeats", f"{E['strong_gC_beats_p95']}/{E['n_strong']}", ["edge_strength_matched.json"],
            "capture", "count", "strong-edge captures where the clamped generator beats its null p95")
        for k, macro, note in (
                ("rho_blind", "EdgeBlindRho",
                 "the theory-named blindness ratio FALLS as the realised edge strengthens"),
                ("p_blind", "EdgeBlindP", ""),
                ("median_sepR_strong", "EdgeBlindStrong", "median ratio, strong-edge captures"),
                ("median_sepR_weak", "EdgeBlindWeak", "median ratio, weak-edge captures")):
            if E.get(k) is not None:
                add(f"edge.{k}", macro, E[k], ["edge_strength_matched.json"], "capture",
                    "Spearman" if "rho" in k or "_p" in k else "median", note)
        if E.get("strong_gt10") is not None:
            add("edge.strong_gt10", "EdgeStrongGtTen", f"{E['strong_gt10']}/{E['n_strong']}",
                ["edge_strength_matched.json"], "capture", "count",
                "strong-edge captures still above the descriptive 10x criterion")
        add("edge.nstrong", "EdgeNStrong", f"{E['n_strong']}", ["edge_strength_matched.json"],
            "capture", "count", "sd(b|arm A)/sd(b|joint) >= 0.10")
        RDV = load_json("ratio_drivers.json")["results"]["row"]["all"]
        rsrc_ = ["ratio_drivers.json", "ratio_drivers.py"]
        for key, t_ in (("sd_ratio_joint", "Sd"), ("edge_r2", "Rtwo")):
            sp_ = RDV[key]["spearman_with_g_ratio"]
            add(f"rd.{key}.rho", f"Rd{t_}Rho", f"{sp_['rho']:.2f}", rsrc_, "capture", "Spearman with the g ratio", key, raw=sp_["rho"])
            add(f"rd.{key}.lo", f"Rd{t_}Lo", f"{sp_['ci95'][0]:.2f}", rsrc_, "(cell, prompt) group", "95% bootstrap lower, 38 groups", key, raw=sp_["ci95"][0])
            add(f"rd.{key}.hi", f"Rd{t_}Hi", f"{sp_['ci95'][1]:.2f}", rsrc_, "(cell, prompt) group", "95% bootstrap upper", key, raw=sp_["ci95"][1])
            add(f"rd.{key}.med", f"Rd{t_}Med", f"{RDV[key]['median']:.2f}", rsrc_, "capture", "median", key, raw=RDV[key]["median"])
    except (FileNotFoundError, KeyError):
        pass

    # ---- identifiability of the interaction coefficients (replaces the 716x ratios) ----
    # PRIMARY is the support-matched protocol, so Sec. 7 and Sec. 6 rest on the same captures.
    ipre = {"GPT-2 IOI": "IdGPTIOI", "Qwen3.5 IOI": "IdQwenIOI",
            "GPT-2 induction": "IdGPTInd", "Qwen3.5 induction": "IdQwenInd",
            "GPT-2 capital": "IdGPTCap", "Qwen3.5 capital": "IdQwenCap",
            "GPT-2 greater": "IdGPTGt", "GPT-2 agreement": "IdGPTAgr"}
    for matched, suffix, src in ((True, "", "identifiability_matched.json"),
                                 (False, "Unmatched", "identifiability.json")):
        try:
            I = d_ident(matched)
        except FileNotFoundError:
            continue
        se_all, ratio_all = [], []
        for cell, pre in ipre.items():
            if cell not in I:
                continue
            r = I[cell]
            p = pre + suffix
            add(f"id{suffix}.{cell}.se", p + "SeRatio", r["se_ratio"],
                [src, "identifiability.py"], "capture", "median",
                "se of the interaction coefficient under single-site / under joint evidence")
            add(f"id{suffix}.{cell}.infoS", p + "InfoS", r["info_min_S"], [src],
                "capture", "median",
                "smallest eigenvalue of the partial Fisher information under single-site")
            add(f"id{suffix}.{cell}.infoJ", p + "InfoJ", r["info_min_J"], [src],
                "capture", "median", "same, under joint evidence")
            # |c|/se is a ratio of vector norms with a null expectation near 0.8, NOT a
            # t-statistic; the correct object is the joint Wald test on the 3-coefficient block.
            add(f"id{suffix}.{cell}.waldJ", p + "WaldJ",
                f"{r['wald_rej_J']}/{r['n']}", [src], "capture", "count",
                "captures where JOINT evidence rejects zero interaction (Wald chi2_3, 0.05)")
            add(f"id{suffix}.{cell}.waldS", p + "WaldS",
                f"{r['wald_rej_S']}/{r['n']}", [src], "capture", "count",
                "the same under single-site evidence; reported because it does NOT favour us")
            add(f"id{suffix}.{cell}.flip", p + "FlipS", r["signflip_S"], [src], "capture",
                "median",
                "fraction of random half-splits where the single-site estimate flips sign")
            add(f"id{suffix}.{cell}.flipj", p + "FlipJ", r["signflip_J"], [src], "capture",
                "median", "the same quantity under JOINT evidence, reported for symmetry")
            se_all.append(r["se_ratio"])
            # median-of-ratios (the per-capture quantity), not a ratio of two medians
            ratio_all.append(r["info_ratio"])
        if se_all:
            for nm, vals, agg in (("SeMin", min(se_all), "min over cells"),
                                  ("SeMax", max(se_all), "max over cells"),
                                  ("InfoRatioMin", min(ratio_all), "min over cells"),
                                  ("InfoRatioMax", max(ratio_all), "max over cells")):
                add(f"id{suffix}.{nm}", "Id" + suffix + nm, vals, [src], "cell", agg, "")

    # ---- reparameterization invariance ----
    inv, ridge, ab, aa, scale = d_reparam()
    add("reparam.invariance", "ReparamInvariance", inv, ["reparam.json", "reparam.py"],
        "capture", "max over trials and models",
        "unregularized generalized eigenvalues under phi -> A phi")
    add("reparam.ridge_shift", "ReparamRidgeShift", ridge, ["reparam.json"], "capture",
        "min over models", "ridge lambda=1e-2 is NOT invariant")
    add("reparam.angle_before", "AngleBefore", ab, ["reparam.json"], "capture", "median", "")
    add("reparam.angle_after", "AngleAfter", aa, ["reparam.json"], "capture", "median",
        "same subspaces after reparameterization -- angles are coordinate-dependent")
    add("reparam.scale_resid", "ScaleResidual", scale, ["reparam.json"], "capture", "median",
        "relative residual of G_J - c G_S; large => not a global scale difference")
    # ---- arXiv additions for Paper A (new keys only; nothing above is changed) ----
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location(
        "ledger_additions", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger_additions.py"))
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    _m.extend(add, load_json, L)
    return L


def emit_tex(L, path):
    bad = [c["macro"] for c in L.values() if not c["macro"].isalpha()]
    if bad:
        raise SystemExit(f"INVALID LaTeX macro name(s) (letters only): {bad}")
    lines = ["% GENERATED BY ledger_paperA.py -- DO NOT EDIT BY HAND.",
             "% Every manuscript number is a macro from a hashed frozen artifact.",
             "% Regenerate: python ledger_paperA.py ; verify: python ledger_paperA.py --verify", ""]
    for mid, c in sorted(L.items(), key=lambda kv: kv[1]["macro"]):
        lines.append(rf"\newcommand{{\{c['macro']}}}{{{c['latex']}\xspace}}"
                     f"  % {mid}: {c['aggregation']} over {c['unit']}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    A = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    jpath, tpath = os.path.join(here, "claims_paperA.json"), os.path.join(here, "claims_paperA.tex")

    try:
        fresh = build()
    except KeyError as e:
        # ROOT-CAUSE GUARD (2026-08-28).  A bare KeyError here means an artifact on disk predates
        # a change to the analysis that produces it -- e.g. identifiability.json regenerated with
        # --matched but not without, so a field the ledger expects is absent.  The failure is
        # correct (the build must not proceed on a stale artifact) but the message was useless.
        sys.exit(f"LEDGER FAILED: artifact field {e} is missing.\n"
                 f"  An artifact in {E0} is older than the analysis that reads it.\n"
                 f"  Regenerate everything in dependency order with ./rebuild_all.sh, which runs\n"
                 f"  every variant of every analysis, then retry.")
    if A.verify:
        if not os.path.exists(jpath):
            print("FAIL: no claims.json to verify against"); sys.exit(2)
        old = json.load(open(jpath))
        bad = []
        for k, v in fresh.items():
            if k not in old:
                bad.append(f"  NEW  {k}"); continue
            if old[k]["latex"] != v["latex"]:
                bad.append(f"  DRIFT {k}: ledger={old[k]['latex']} rederived={v['latex']}")
            for a, b in zip(old[k]["artifacts"], v["artifacts"]):
                if a.get("sha256") != b.get("sha256"):
                    bad.append(f"  ARTIFACT CHANGED {k}: {a.get('file')}")
        for k in old:
            if k not in fresh:
                bad.append(f"  MISSING {k}")
        if bad:
            print(f"VERIFY FAILED ({len(bad)} problems)"); print("\n".join(bad)); sys.exit(1)
        print(f"VERIFY OK — {len(fresh)} claims re-derived, all match, all artifacts unchanged")
        sys.exit(0)

    json.dump(fresh, open(jpath, "w"), indent=1)
    emit_tex(fresh, tpath)
    print(f"{len(fresh)} claims -> claims.json + claims.tex")
    for mid, c in sorted(fresh.items(), key=lambda kv: kv[1]["macro"]):
        print(f"  \\{c['macro']:<26} = {c['latex']:<22} [{c['aggregation']} / {c['unit']}]"
              f"{'  ' + c['note'] if c['note'] else ''}")
