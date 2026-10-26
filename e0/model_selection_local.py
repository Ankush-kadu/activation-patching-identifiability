"""IOI attribution analysis, simplicity-aware model selection on the single-site family, with family-local, fold-local preprocessing.

WHY THIS FILE EXISTS (2026-09-24).  The IOI attribution analysis (adequacy tables) reports that a cubic response
model fitted to single-site training rows of the 31 published GPT-2 IOI captures fails on held-out joint rows, that an
additive-only refit transfers, and that a ridge on the interaction block with lambda chosen by donor-grouped CV "does not
transfer".  An external review found two problems with that ridge control:
  (1) response_diagnostics.heldout whitens with POOLED single-site + joint training states (response_diagnostics.py:188-202),
      so the ridge penalty -- which is not whitening-invariant -- is set in coordinates the joint rows helped define, and
      the standardiser is not refitted inside the CV folds.  It is not a strictly single-family pipeline.
  (2) choosing the CV-minimising lambda on a penalty grid is not the same as comparing a discrete additive-only candidate
      under a rule that prefers the simpler model when the data do not resolve the difference.
It asked what explicitly simplicity-aware selection does on the existing single-site data, with donor-level validation
differences, a rule that prefers the simpler model when those differences are unresolved, and preprocessing fitted on the
relevant training family only and refitted within validation folds.  This file answers that, and re-runs the paper's ridge
control with the preprocessing fixed so the paper can say whether the fix changes anything.

STATUS.  EXPLORATORY, post hoc, on captures that have been inspected many times before (qchart_* published configurations,
qctrl_* controls).  The rules, grid and interval conventions below were chosen after those data were seen (the rules follow
the reviewer's request and textbook conventions; nothing was tuned on the outputs).  Nothing here is confirmatory.

DATA AND SPLITS.  Per capture: single-site rows Zs/Ys (2 arms x 32 donors x 7 alphas; donor = (row mod 224) // 7), support-
matched joint rows Zjm/Yjm, and the executed quartet corners (Zc/Yc clamped arm, clean).  Outer split: the donor partition
response_diagnostics.partition(seed) -- 16 training donors, 16 held out; seed 0 is primary, seeds 1-9 are sensitivity.

CANDIDATES (degree d in 2, 3, 4; primary 3; ADD_d / CROSS_d exactly as response_diagnostics.poly_class):
    A          OLS on ADD_d                                  (additive-only; response_diagnostics.ols_model)
    R(lambda)  ADD_d + CROSS_d with a ridge penalty on CROSS_d only (response_diagnostics.ridge_cross), lambda over
               response_diagnostics.LAMBDAS = 10^-6 .. 10^4 (21 values); the penalty acts in the fold-local whitened coordinates
    F          OLS on ADD_d + CROSS_d                        (full class, lstsq; equals the paper's FWL working_model up to
               rounding -- the prediction gap is recorded)
  Complexity order, simplest first:  A, R(10^4), R(10^3.5), ..., R(10^-6), F.

PREPROCESSING.  Whitening = per-coordinate (Z - mean) / sd, as the paper.  Outer refit: mean/sd of the training donors'
  SINGLE-SITE rows only.  Inner fold: mean/sd of that fold's inner-training single-site rows only, refitted in every fold.
  A and F are closed under per-coordinate affine maps, so the whitening changes them only by rounding; it matters for R.
  No joint row enters any fit, whitening or selection of A, R, F or the rules.

INNER CV (single-site training rows only).  4 folds grouped by donor (16 training donors -> 4 per fold), fold assignment
  array_split(default_rng(0).permutation(donors), 4) exactly as response_diagnostics.ridge_cv.  Every training donor is
  validated once, on its 14 rows (2 arms x 7 alphas).
    CV MSE(k)          mean over all training rows of candidate k's out-of-fold squared error (per-fold SSEs are stored)
    donor MSE_i(k)     the same over donor i's 14 rows
    paired difference  Delta = mean_i [MSE_i(F) - MSE_i(A)]  (= CV MSE(F) - CV MSE(A), equal rows per donor)
    SE                 DONOR-LEVEL: sd_i(.) / sqrt(16) over the 16 validation donors (fold-level SEs over the 4 folds are
                       stored too, not used).  Out-of-fold errors share training data across folds, so every SE and interval
                       here is nominal.
  "Relative" CV differences are divided by the variance of the training single-site readouts, var(Ys[train donors]).

SELECTION RULES.
    (i)   min-CV    argmin CV MSE over {A, R(lambda), F}; exact ties go to the simpler candidate.
    (ii)  1-SE      the simplest candidate whose CV MSE <= min CV MSE + donor-level SE of the minimising candidate's CV MSE.
    (iii) paired    F if the 95% t-interval (15 df) of Delta lies entirely below 0 (F resolved better); otherwise A.
                    A 2,000-draw donor-bootstrap percentile interval of Delta is stored as a check, not used.

EVALUATION.  Each rule's selection, and A and F directly, is refit on all single-site training rows (training-only single-site
  whitening) and scored on
    R2_S     held-out single-site rows (held-out donors)
    R2_J     held-out joint rows (both donors held out); R^2 = response_diagnostics.score (1 - SSE / sum (y - mean y_test)^2)
    RMSE_T   sqrt(mean (T_C - T_hat)^2) over the held-out joint rows, logits.  T_C is the MEASURED controlled four-point
             contrast from the executed quartets and T_hat the same contrast of the fitted model at the executed corner
             coordinates (fourpoint_modelfree.quartets / model_contrast).  "zero" = rms(T_C), the additivity prediction.

THE PAPER'S RIDGE CONTROL, FIXED.  The ridge row of the adequacy table is R(lambda*) with lambda* the CV-minimising lambda over the
  ridge grid alone (ties -> smallest lambda, as ridge_cv), fitted within each family.  Re-run here with family-local,
  fold-local whitening for both fitted families (single-site: groups = donor, whitening on single-site rows; joint: groups =
  upstream donor, whitening on joint training rows only).  The pooled-whitening original is re-run by calling
  response_diagnostics.ridge_cv itself; its capture-medians must reproduce response_diagnostics.json (asserted at d = 3 on the
  published captures before anything is written; recorded at d = 2, 4).  Controls have no reference in that JSON.

AGGREGATION.  Published (31 qchart_*) and controls (28 qctrl_*) separately.  Per partition seed: counts and capture-medians.
  Across partitions: seed 0, and the median over seeds 1-9 of the per-seed capture-medians [min, max] ("median of medians");
  the ridge comparison also gives the median over seeds 0-9, which is what response_diagnostics.json reports.  Captures share
  base prompts and donors, so these are distributions of capture-level values with no pooled significance claim.

    python3 model_selection_local.py      # -> model_selection_local.json   (CPU only)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import t as student_t

from cells import N_AL
from fourpoint_modelfree import model_contrast, quartets, rms
from response_diagnostics import (DEGREES, LAMBDAS, SEEDS, ols_model, partition, ridge_cross, ridge_cv, score,
                                  working_class, working_model)

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY_DEGREE = 3
K_FOLDS = 4
BOOT = 2000
WORKERS = int(os.environ.get("MSL_WORKERS", "10"))
GROUPS = (("published", "qchart_*.npz"), ("controls", "qctrl_*.npz"))
PAIRS = ("S->S", "S->J", "J->S", "J->J")

LAM_DESC = np.asarray(LAMBDAS)[::-1]
NAMES = ["A"] + [f"R({lam:.3g})" for lam in LAM_DESC] + ["F"]          # complexity order, simplest first
KIND = ["A"] + ["R"] * len(LAM_DESC) + ["F"]
IA, IF = 0, len(NAMES) - 1
IR = np.arange(1, IF)                                                    # ridge candidates, descending lambda
RULES = ("min_cv", "one_se", "paired")
MODELS = RULES + ("A", "F", "ridge_local", "ridge_pooled")              # everything scored on the held-out rows


def lam_of(k):
    return float(LAM_DESC[k - 1]) if KIND[k] == "R" else None


def whitener(Z):
    mu, sd = Z.mean(0), Z.std(0) + 1e-12                                 # the paper's whitening formula
    return lambda X: (X - mu) / sd


def fit_candidate(k, Z, Y, W, add, cross):
    if k == IA:
        return ols_model(Z, Y, W, add)
    if k == IF:
        return ols_model(Z, Y, W, add + cross)
    return ridge_cross(Z, Y, W, add, cross, lam_of(k))


def inner_cv(Z, Y, groups, add, cross, ks):
    """Out-of-fold squared errors of candidates ks; whitening refitted on each fold's inner-training rows of this family."""
    folds = np.array_split(np.random.default_rng(0).permutation(np.unique(groups)), K_FOLDS)   # as ridge_cv
    sq = np.full((len(NAMES), len(Y)), np.nan)
    fold_of = np.full(len(Y), -1)
    for f, fo in enumerate(folds):
        te = np.isin(groups, fo)
        fold_of[te] = f
        W = whitener(Z[~te])
        for k in ks:
            sq[k, te] = (Y[te] - fit_candidate(k, Z[~te], Y[~te], W, add, cross)(Z[te])) ** 2
    assert (fold_of >= 0).all()
    return sq, fold_of


def ridge_argmin(cv):
    """The paper's ridge choice: argmin CV over the ridge grid in ascending lambda (np.argmin -> smallest lambda on ties)."""
    asc = IR[::-1]
    return int(asc[int(np.argmin(cv[asc]))])


def select(sq, groups, fold_of):
    cv = sq.mean(1)
    donors = np.unique(groups)
    n = len(donors)
    dmse = np.array([sq[:, groups == g].mean(1) for g in donors]).T                  # K x donors
    se = dmse.std(1, ddof=1) / np.sqrt(n)
    fmse = np.array([sq[:, fold_of == f].mean(1) for f in range(K_FOLDS)]).T         # K x folds
    fse = fmse.std(1, ddof=1) / np.sqrt(K_FOLDS)
    kmin = int(np.flatnonzero(cv == cv.min())[0])                                      # ties -> simpler
    k1se = int(np.flatnonzero(cv <= cv[kmin] + se[kmin])[0])
    diff = dmse[IF] - dmse[IA]
    m, s = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(n))
    tq = float(student_t.ppf(0.975, n - 1))
    ci = [m - tq * s, m + tq * s]
    bm = diff[np.random.default_rng(0).integers(0, n, (BOOT, n))].mean(1)
    ci_b = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    return dict(cv=cv, se=se, fse=fse, fold_sse=np.array([sq[:, fold_of == f].sum(1) for f in range(K_FOLDS)]).T,
                donor_diff=diff, delta=m, delta_se=s, ci_t=ci, ci_boot=ci_b,
                gain_min_vs_A=float(cv[IA] - cv[kmin]), gain_min_vs_A_over_se=float((cv[IA] - cv[kmin]) / se[kmin]),
                sel=dict(min_cv=kmin, one_se=k1se, paired=IF if ci[1] < 0 else IA),
                k_ridge=ridge_argmin(cv))


def run(task):
    """One capture x one partition seed, all degrees."""
    group, f, seed = task
    raw = np.load(os.path.join(HERE, f))
    d = {k: raw[k] for k in ("Zs", "Ys", "Zc", "Yc", "Zjm", "Yjm", "clean")}
    Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
    T, corners, a0b0, ja, _, audit = quartets(d)
    assert audit["joint_a_vs_clamped"] < 1e-9 and audit["joint_b_vs_nm_arm"] < 1e-9, (f, audit)
    s_tr, j_tr, j_te, _ = partition(len(Zs), seed)
    donor = (np.arange(len(Zs)) % (len(Zs) // 2)) // N_AL
    gS, gJ = donor[s_tr], ja[j_tr]
    Zt, Yt = Zs[s_tr], Ys[s_tr]
    Zjt, Yjt = Zj[j_tr], Yj[j_tr]
    held = {"S": (Zs[~s_tr], Ys[~s_tr]), "J": (Zj[j_te], Yj[j_te])}
    Tte = T[j_te]
    cte = {k: v[j_te] for k, v in corners.items()}
    var_S = float(np.var(Yt))
    sd_te = {k: float(np.std(v[1])) for k, v in held.items()}
    Wp = whitener(np.vstack([Zt, Zjt]))                            # the paper's POOLED whitening (reproduction only)

    def metrics(pred):
        return [score(held["S"][1], pred(held["S"][0]))[1], score(held["J"][1], pred(held["J"][0]))[1],
                rms(Tte - model_contrast(pred, cte, a0b0))]

    out = dict(file=f, group=group, seed=seed, var_S_train=var_S, rms_T_heldout=rms(Tte), deg={})
    for deg in DEGREES:
        with working_class(deg) as (add, cross):
            # ---- selection on single-site training rows, fold-local single-site whitening
            sq, fold_of = inner_cv(Zt, Yt, gS, add, cross, range(len(NAMES)))
            sel = select(sq, gS, fold_of)
            W = whitener(Zt)                                           # outer refit: training single-site rows only
            preds = [fit_candidate(k, Zt, Yt, W, add, cross) for k in range(len(NAMES))]
            outer = [metrics(p) for p in preds]
            kr = sel["k_ridge"]
            # ---- the paper's ridge, joint family, fixed: joint-only, fold-local whitening
            sqJ, _ = inner_cv(Zjt, Yjt, gJ, add, cross, IR)
            kJ = ridge_argmin(sqJ.mean(1))
            pJ = fit_candidate(kJ, Zjt, Yjt, whitener(Zjt), add, cross)
            local = {"S->S": outer[kr][0], "S->J": outer[kr][1], "J->S": score(held["S"][1], pJ(held["S"][0]))[1],
                     "J->J": score(held["J"][1], pJ(held["J"][0]))[1]}
            # ---- the paper's ridge exactly as response_diagnostics.heldout (pooled whitening) -- reproduction
            pS0, lS0 = ridge_cv(Zt, Yt, Wp, add, cross, gS)
            pJ0, lJ0 = ridge_cv(Zjt, Yjt, Wp, add, cross, gJ)
            pooled = {f"{ft}->{fe}": score(held[fe][1], p(held[fe][0]))[1]
                      for ft, p in (("S", pS0), ("J", pJ0)) for fe in ("S", "J")}
            # ---- checks: A and F are whitening-invariant, so pooled-whitening refits must agree up to rounding
            gapA = max(float(np.max(np.abs(preds[IA](z) - ols_model(Zt, Yt, Wp, add)(z))) / sd_te[k])
                       for k, (z, _) in held.items())
            try:
                wm = working_model(Zt, Yt, Wp, add, cross)
                gapF = max(float(np.max(np.abs(preds[IF](z) - wm(z))) / sd_te[k]) for k, (z, _) in held.items())
            except np.linalg.LinAlgError:
                gapF = float("nan")
            contrast_A = float(np.max(np.abs(model_contrast(preds[IA], corners, a0b0))))
        ev = {r: outer[sel["sel"][r]] for r in RULES}
        ev.update(A=outer[IA], F=outer[IF], ridge_local=outer[kr], ridge_pooled=metrics(pS0))
        out["deg"][str(deg)] = dict(
            sel={r: NAMES[k] for r, k in sel["sel"].items()}, sel_kind={r: KIND[k] for r, k in sel["sel"].items()},
            sel_lambda={r: lam_of(k) for r, k in sel["sel"].items()},
            delta_FA=sel["delta"], delta_FA_rel=sel["delta"] / var_S, delta_FA_se=sel["delta_se"],
            delta_FA_ci95_t=sel["ci_t"], delta_FA_ci95_boot=sel["ci_boot"],
            cv_gain_mincv_vs_A_rel=sel["gain_min_vs_A"] / var_S, cv_gain_mincv_vs_A_over_se=sel["gain_min_vs_A_over_se"],
            resolved_F=bool(sel["ci_t"][1] < 0), resolved_A=bool(sel["ci_t"][0] > 0),
            resolved_F_boot=bool(sel["ci_boot"][1] < 0),
            eval={m: dict(R2_S=v[0], R2_J=v[1], RMSE_T=v[2]) for m, v in ev.items()},
            ridge_local=dict(lambda_S=lam_of(kr), lambda_J=lam_of(kJ), r2=local),
            ridge_pooled=dict(lambda_S=lS0, lambda_J=lJ0, r2=pooled),
            checks=dict(A_pooled_vs_local_whitening_gap_over_sd=gapA, F_vs_paper_working_model_pooled_gap_over_sd=gapF,
                        A_contrast_max_abs=contrast_A),
            detail=dict(cv_mse=sel["cv"].tolist(), cv_se_donor=sel["se"].tolist(), cv_se_fold=sel["fse"].tolist(),
                        fold_sse=sel["fold_sse"].tolist(), donor_diff_FA=sel["donor_diff"].tolist(),
                        outer_R2_S_R2_J_RMSE_T=outer))
    return out


# ------------------------------------------------------------------ summaries

def med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def summarise(rows, deg):
    """Flat dict of counts and capture-medians for one group x seed x degree."""
    x = [r["deg"][str(deg)] for r in rows]
    s = {"n": len(x)}
    for rule in RULES:
        for kind in ("A", "R", "F"):
            s[f"select.{rule}.{kind}"] = sum(r["sel_kind"][rule] == kind for r in x)
        s[f"select.{rule}.R_at_lambda_max"] = sum(r["sel_lambda"][rule] == float(LAMBDAS[-1]) for r in x)
    s["delta_FA_rel.median"] = med([r["delta_FA_rel"] for r in x])
    s["delta_FA_rel.q25"] = float(np.percentile([r["delta_FA_rel"] for r in x], 25))
    s["delta_FA_rel.q75"] = float(np.percentile([r["delta_FA_rel"] for r in x], 75))
    s["delta_FA_negative"] = sum(r["delta_FA"] < 0 for r in x)
    s["cv_gain_mincv_vs_A_rel.median"] = med([r["cv_gain_mincv_vs_A_rel"] for r in x])
    s["cv_gain_mincv_vs_A_over_se.median"] = med([r["cv_gain_mincv_vs_A_over_se"] for r in x])
    s["resolved_F"] = sum(r["resolved_F"] for r in x)
    s["resolved_A"] = sum(r["resolved_A"] for r in x)
    s["resolved_F_boot"] = sum(r["resolved_F_boot"] for r in x)
    s["zero.RMSE_T"] = med([r["rms_T_heldout"] for r in rows])
    for m in MODELS:
        for k in ("R2_S", "R2_J", "RMSE_T"):
            s[f"{m}.{k}"] = med([r["eval"][m][k] for r in x])
        s[f"{m}.R2_J_negative"] = sum(r["eval"][m]["R2_J"] < 0 for r in x)
    for tag in ("ridge_local", "ridge_pooled"):
        for p in PAIRS:
            s[f"{tag}.r2.{p}"] = med([r[tag]["r2"][p] for r in x])
        for fam in ("S", "J"):
            lam = np.array([r[tag][f"lambda_{fam}"] for r in x])
            s[f"{tag}.lambda_{fam}.median"] = float(np.median(lam))
            s[f"{tag}.lambda_{fam}.share_at_max"] = float(np.mean(lam == LAMBDAS[-1]))
    return s


def across_seeds(per_seed):
    """per_seed: {seed: flat summary}.  seed 0, median over seeds 1-9 [min, max], median over seeds 0-9."""
    out = {}
    for key in per_seed[0]:
        v = np.array([per_seed[s][key] for s in SEEDS], float)
        rest = v[1:]
        out[key] = dict(seed0=float(v[0]), seeds1_9_median=float(np.nanmedian(rest)),
                        seeds1_9_range=[float(np.nanmin(rest)), float(np.nanmax(rest))],
                        seeds0_9_median=float(np.nanmedian(v)))
    return out


def main():
    os.chdir(HERE)
    t0 = time.time()
    files = {g: sorted(glob.glob(p)) for g, p in GROUPS}
    assert len(files["published"]) == 31 and len(files["controls"]) == 28, {g: len(v) for g, v in files.items()}
    tasks = [(g, f, s) for g, fs in files.items() for s in SEEDS for f in fs]
    with Pool(WORKERS) as pool:
        results = pool.map(run, tasks, chunksize=1)
    by = {(r["group"], r["seed"]): [] for r in results}
    for r in results:
        by[(r["group"], r["seed"])].append(r)

    # ---- reproduction of response_diagnostics.json's ridge (pooled whitening), BEFORE anything is written
    ref = json.load(open("response_diagnostics.json"))["heldout_ridge_cross_partition_sensitivity"]
    repro = {}
    for deg in DEGREES:
        gaps = [abs(summarise(by[("published", s)], deg)[f"ridge_pooled.r2.{p}"] - ref[str(deg)]["r2"][p]["r2_median_by_seed"][s])
                for s in SEEDS for p in PAIRS]
        repro[str(deg)] = float(max(gaps))
    assert repro[str(PRIMARY_DEGREE)] < 1e-6, f"pooled ridge does not reproduce response_diagnostics.json: {repro}"

    summary, per_seed_all = {}, {}
    for g in files:
        summary[g], per_seed_all[g] = {}, {}
        for deg in DEGREES:
            ps = {s: summarise(by[(g, s)], deg) for s in SEEDS}
            per_seed_all[g][str(deg)] = {str(s): v for s, v in ps.items()}
            summary[g][str(deg)] = across_seeds(ps)
    ridge_fix = {deg: {p: dict(pooled_response_diagnostics_json=ref[deg]["r2"][p]["r2_median_over_seeds"],
                               pooled_rerun=summary["published"][deg][f"ridge_pooled.r2.{p}"]["seeds0_9_median"],
                               local=summary["published"][deg][f"ridge_local.r2.{p}"]["seeds0_9_median"],
                               local_range=[min(per_seed_all["published"][deg][str(s)][f"ridge_local.r2.{p}"] for s in SEEDS),
                                            max(per_seed_all["published"][deg][str(s)][f"ridge_local.r2.{p}"] for s in SEEDS)])
                     for p in PAIRS} for deg in map(str, DEGREES)}

    # ---- A and F under single-site whitening vs the published seed-0 numbers (whitening-invariant classes)
    rd_rows = {r["file"]: r for r in json.load(open("response_diagnostics.json"))["rows"]}
    fm = {c["file"]: c for grp in json.load(open("fourpoint_modelfree.json"))["groups"].values() for c in grp["captures"]}
    seed0 = by[("published", 0)] + by[("controls", 0)]
    checks = dict(
        ridge_pooled_reproduction_max_abs_diff_capture_median_r2=repro,
        F_R2_J_seed0_max_abs_diff_vs_response_diagnostics_rows={
            str(deg): float(max(abs(r["deg"][str(deg)]["eval"]["F"]["R2_J"] - rd_rows[r["file"]]["heldout"][str(deg)]["S->J"][1])
                                / max(1.0, abs(rd_rows[r["file"]]["heldout"][str(deg)]["S->J"][1]))
                                for r in by[("published", 0)])) for deg in DEGREES},
        F_RMSE_T_seed0_max_rel_diff_vs_fourpoint_modelfree={
            g: {str(deg): float(max(abs(r["deg"][str(deg)]["eval"]["F"]["RMSE_T"] - fm[r["file"]]["rmse_heldout"][f"d{deg}_S"])
                                    / fm[r["file"]]["rmse_heldout"][f"d{deg}_S"] for r in by[(g, 0)])) for deg in DEGREES}
            for g in files},
        F_numerics_note="F is whitening-invariant in exact arithmetic; where its joint predictions are off by 1e3-1e7 logits "
                        "(controls, d >= 3) rounding in a near-singular design makes lstsq/single-site-whitening and "
                        "FWL/pooled-whitening disagree in those astronomically wrong values; no conclusion depends on it",
        A_R2_J_seed0_capture_median_vs_response_diagnostics={
            str(deg): [summary["published"][str(deg)]["A.R2_J"]["seed0"],
                       json.load(open("response_diagnostics.json"))["heldout_additive"][str(deg)]["S->J"]["r2"]["median"]]
            for deg in DEGREES},
        zero_RMSE_T_seed0_max_abs_diff_vs_fourpoint_modelfree=float(max(abs(r["rms_T_heldout"] - fm[r["file"]]["rms_T_heldout"])
                                                                       for r in seed0)),
        A_contrast_max_abs_over_all=float(max(r["deg"][str(d)]["checks"]["A_contrast_max_abs"] for r in results for d in DEGREES)),
        A_whitening_gap_over_sd_max={str(d): float(max(r["deg"][str(d)]["checks"]["A_pooled_vs_local_whitening_gap_over_sd"]
                                                       for r in results)) for d in DEGREES},
        F_vs_working_model_gap_over_sd_max={
            g: {str(d): float(np.nanmax([r["deg"][str(d)]["checks"]["F_vs_paper_working_model_pooled_gap_over_sd"]
                                         for r in results if r["group"] == g])) for d in DEGREES} for g in files})

    res = dict(
        status="EXPLORATORY, post hoc: previously inspected captures; rules chosen after the data were seen (reviewer request, "
               "textbook conventions); nothing tuned on these outputs; nothing confirmatory",
        primary_degree=PRIMARY_DEGREE, degrees=list(DEGREES), seeds=list(SEEDS), k_folds=K_FOLDS,
        candidates=NAMES, candidate_kind=KIND, lambdas=[float(v) for v in LAMBDAS],
        rules=dict(min_cv="argmin CV MSE over {A, R(lambda), F}; exact ties -> simpler",
                   one_se="simplest candidate (A < R(largest lambda) < ... < R(smallest) < F) with CV MSE <= min + donor-level "
                          "SE of the minimising candidate's CV MSE",
                   paired="F iff the 95% t-interval (df = n_donors - 1) of mean donor-level [MSE(F) - MSE(A)] lies below 0; "
                          "else A"),
        estimands=dict(
            se="donor-level: sd over the 16 validation donors / sqrt(16) (fold-level SEs stored in rows, unused); nominal",
            delta_FA_rel="(CV MSE(F) - CV MSE(A)) / var(training single-site readouts); negative favours F",
            R2_S="held-out single-site rows (held-out donors), response_diagnostics.score",
            R2_J="held-out joint rows with both donors held out, response_diagnostics.score",
            RMSE_T="sqrt(mean (T_C - T_hat)^2) over held-out joint rows, logits; T_C measured from executed quartets "
                   "(fourpoint_modelfree.quartets), T_hat = model_contrast of the refit model; zero.RMSE_T = rms(T_C)",
            ridge_local="the paper's ridge control (argmin CV over the ridge grid alone) with family-local, fold-local "
                        "whitening; ridge_pooled = response_diagnostics.ridge_cv with the paper's pooled whitening",
            aggregation="per seed: counts / medians over captures; across seeds: seed0, median over seeds 1-9 [min, max], "
                        "median over seeds 0-9"),
        ridge_fix_published=ridge_fix,
        summary=summary, per_seed=per_seed_all, checks=checks,
        rows_seed0={g: by[(g, 0)] for g in files},
        runtime_s=None)
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open("model_selection_local.json", "w"), indent=1, default=float)

    # ---- print
    print(f"model_selection_local: {len(tasks)} capture x seed tasks, {res['runtime_s']} s")
    print("checks:", json.dumps({k: v for k, v in checks.items()}, default=float))
    for g in files:
        for deg in DEGREES:
            S = summary[g][str(deg)]
            v = lambda k: S[k]["seed0"]
            w = lambda k: S[k]["seeds1_9_median"]
            print(f"\n[{g}] d={deg}  n={int(v('n'))}   seed0 | seeds1-9 median-of-medians")
            for rule in RULES:
                print(f"  {rule:>7}: A/R/F = {int(v(f'select.{rule}.A'))}/{int(v(f'select.{rule}.R'))}/{int(v(f'select.{rule}.F'))}"
                      f" (R at lam_max {int(v(f'select.{rule}.R_at_lambda_max'))})"
                      f"  | seeds1-9 med A/R/F {w(f'select.{rule}.A'):.0f}/{w(f'select.{rule}.R'):.0f}/{w(f'select.{rule}.F'):.0f}")
            print(f"  delta(F-A)/var median {v('delta_FA_rel.median'):.3g} [q25 {v('delta_FA_rel.q25'):.3g}, q75 {v('delta_FA_rel.q75'):.3g}]"
                  f" | resolved F {int(v('resolved_F'))}, resolved A {int(v('resolved_A'))} | seeds1-9 {w('delta_FA_rel.median'):.3g},"
                  f" resolved F {w('resolved_F'):.0f}")
            print(f"  min-CV gain over A: {v('cv_gain_mincv_vs_A_rel.median'):.3g} of var, {v('cv_gain_mincv_vs_A_over_se.median'):.3g} SE"
                  f" | seeds1-9 {w('cv_gain_mincv_vs_A_rel.median'):.3g}, {w('cv_gain_mincv_vs_A_over_se.median'):.3g} SE")
            print(f"  {'model':>12} {'R2_S':>8} {'R2_J':>12} {'RMSE_T':>9} {'R2_J<0':>6} || {'R2_J s1-9':>12} {'RMSE_T s1-9':>11}"
                  f"   (zero RMSE_T {v('zero.RMSE_T'):.4f} | {w('zero.RMSE_T'):.4f})")
            for m in MODELS:
                print(f"  {m:>12} {v(f'{m}.R2_S'):>8.3f} {v(f'{m}.R2_J'):>12.4g} {v(f'{m}.RMSE_T'):>9.4f}"
                      f" {int(v(f'{m}.R2_J_negative')):>6} || {w(f'{m}.R2_J'):>12.4g} {w(f'{m}.RMSE_T'):>11.4f}")
    print("\nridge control, published, median over seeds 0-9 (pooled JSON | pooled rerun | family-local, fold-local):")
    for deg, rr in ridge_fix.items():
        print(f"  d={deg} " + "  ".join(f"{p} {x['pooled_response_diagnostics_json']:.3f} | {x['pooled_rerun']:.3f} | {x['local']:.3f}"
                                         f" [{x['local_range'][0]:.3f},{x['local_range'][1]:.3f}]" for p, x in rr.items()))
    print("saved -> model_selection_local.json")


if __name__ == "__main__":
    main()
