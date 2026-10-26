"""IOI attribution analysis, working-model diagnostics: held-out readout fit and polynomial-degree sensitivity.

WHY THIS FILE EXISTS (2026-09-23).  The IOI attribution analysis bounds the four-point transport contrast
(misattribution_fourpoint.py) under a cubic working response model and its nominal 95% Wald region.
Two reviewers read that as a model-conditional sensitivity calculation and asked (i) whether the
working model predicts held-out readouts at all, and (ii) how the result depends on the polynomial
degree.  This file answers both for the 31 GPT-2 IOI S-Inhibition -> Name Mover captures
(qchart_*.npz).  It does not choose a degree; it reports 2, 3 and 4.

THE CLASSES.  identifiability.py writes the degree-3 working model as [ADD | CROSS]: ADD is the
intercept plus the pure powers of each coordinate, CROSS is every monomial with both coordinates
present.  The same split at degree d (whitened coordinates a, b):

    ADD_d   = 1, a, b, a^2, b^2, ..., a^d, b^d            (2d + 1 columns)
    CROSS_d = a^i b^j  with  i >= 1, j >= 1, i + j <= d    (d(d - 1)/2 columns)

    d = 2   ADD  1, a, b, a^2, b^2                          CROSS  ab
    d = 3   identical to identifiability.ADD / CROSS (asserted below)
    d = 4   ADD  1, a, b, a^2, b^2, a^3, b^3, a^4, b^4      CROSS  ab, a^2 b, a b^2, a^3 b, a^2 b^2, a b^3

ADD_d + CROSS_d is the full bivariate polynomial of total degree <= d.  The Wald region for the
cross block is the ellipsoid at chi2_{0.95, |CROSS_d|}, i.e. 1, 3 and 6 degrees of freedom.

REUSE.  misattribution.fit / worst_case, identifiability.partial_info and
misattribution_fourpoint.four_point / one are fixed to the degree-3 class through their modules'
globals (ADD, CROSS, and misattribution.CRIT).  working_class(d) rebinds exactly those globals for
the duration of a block and restores them afterwards, so every degree runs the paper's own code
path.  At d = 3 the rebinding is the identity (asserted), and the degree-3 envelope is checked row
by row against misattribution_fourpoint.json (read only).

PART 1: HELD-OUT READOUT FIT (donor partition, training-only fitting).
  Donor structure, as in positive_control_heldout.donor_protocol: single-site rows are
  2 arms x 32 donors x 7 interpolation coefficients (row index within an arm // 7 is the donor);
  joint row k crosses upstream donor ids[k // 16] with downstream donor ids[k % 16], ids = 0, 2, ..., 30.
  That mapping is verified against the data here (the joint a-value must equal the single-site
  arm-A row of its upstream donor, and the joint b-value the arm-B row of its downstream donor, at the
  alpha indices ioi_query_chart.collect uses).  Half the donors are training donors (donor_protocol's
  seed-0 permutation).  Everything fitted -- the whitening (mean/sd over training single-site rows plus
  training joint rows) and the polynomial coefficients -- uses training rows only: single-site rows of
  training donors, and joint rows whose BOTH donors are training donors.  The scored rows are
  single-site rows of held-out donors and joint rows whose BOTH donors are held out.  The working model
  is fitted separately to single-site and to joint training rows (the paper's estimator:
  misattribution.fit for the cross block, least squares for the additive block given it -- together the
  OLS fit of the full class), and each fit predicts both held-out families.
  Partition sensitivity: the same summary for donor_protocol-style partitions with seeds 0..9.
  Additive reference (added 2026-09-23 after a review re-derived it): the additive block ADD_d alone,
  fitted by least squares on the same training rows with the same whitening, scored on the same held-out
  rows.  It isolates what the interaction block contributes to (or costs) out-of-family prediction.
  Regularised reference (same date): the full class with a ridge penalty on the interaction block only
  (lambda -> 0 is the full fit, lambda -> inf the additive one), lambda chosen by 4-fold CV over training
  donors inside the fitted family (single-site: donor; joint: upstream donor), then refit on all training
  rows and scored on the same held-out rows.  Negative-R^2 counts are also recorded for every partition.

PART 2: ENVELOPE BY DEGREE (full data, exactly misattribution_fourpoint.one).
  Per capture: the four-point contrast D over the joint (support-matched) states, the 95% Wald
  worst case of |D c| per state, its median (and max) over states, divided by the B-arm readout swing
  ptp(Ys[B arm]); with c, V fitted on the single-site family (S) and on the joint family (J).

    python3 response_diagnostics.py      # -> response_diagnostics.json
"""
import glob
import json
import os
from contextlib import contextmanager

import numpy as np
from scipy.stats import chi2

import identifiability as I
import misattribution as M
import misattribution_fourpoint as F
from cells import N_AL
from identifiability import dm

HERE = os.path.dirname(os.path.abspath(__file__))
DEGREES = (2, 3, 4)
SEEDS = tuple(range(10))                 # seed 0 is donor_protocol's partition and the primary one
PAIRS = ("S->S", "S->J", "J->S", "J->J")  # (family fitted on) -> (held-out family scored)


def poly_class(d):
    """[ADD | CROSS] at degree d, built exactly as identifiability.py builds degree 3."""
    add = [(0, 0)] + [e for k in range(1, d + 1) for e in ((k, 0), (0, k))]
    full = [(i, t - i) for t in range(d + 1) for i in range(t + 1)]
    return add, [e for e in full if e not in add]


assert poly_class(3) == (I.ADD, I.CROSS), "degree-3 class must be identifiability's"
assert np.isclose(M.CRIT, chi2.ppf(0.95, 3))
_MODS = (I, M, F)


@contextmanager
def working_class(d):
    """Rebind the class globals the reused functions read, for the duration of the block."""
    add, cross = poly_class(d)
    saved = {(m, k): getattr(m, k) for m in _MODS for k in ("ADD", "CROSS")}
    saved[(M, "CRIT")] = M.CRIT
    try:
        for m in _MODS:
            m.ADD, m.CROSS = add, cross
        M.CRIT = float(chi2.ppf(0.95, len(cross)))
        yield add, cross
    finally:
        for (m, k), v in saved.items():
            setattr(m, k, v)


# ------------------------------------------------------------------ donor structure

def joint_donors(n_rows):
    """(upstream donor, downstream donor) of every support-matched joint row, as donor_protocol."""
    n_don = (n_rows // 2) // N_AL
    assert n_don * N_AL == n_rows // 2
    ids = np.arange(0, n_don, 2)
    return n_don, np.repeat(ids, len(ids)), np.tile(ids, len(ids))


def joint_map_error(Zs, Zj):
    """Max |joint coordinate - the single-site coordinate its donors predict| (0 if the map is right).
    Alpha indices as ioi_query_chart.collect: ((i + j) % N_AL, (3 i + j) % N_AL) for pair (i, j)."""
    n = len(Zs) // 2
    _, ja, jb = joint_donors(len(Zs))
    ka, kb = (ja + jb) % N_AL, (3 * ja + jb) % N_AL
    return float(max(np.abs(Zj[:, 0] - Zs[ja * N_AL + ka, 0]).max(),
                     np.abs(Zj[:, 1] - Zs[n + jb * N_AL + kb, 1]).max()))


def partition(n_rows, seed):
    """donor_protocol's split: masks for training/held-out single-site rows and joint rows."""
    n_don, ja, jb = joint_donors(n_rows)
    donor = (np.arange(n_rows) % (n_rows // 2)) // N_AL
    train = np.random.default_rng(seed).permutation(n_don)[: n_don // 2]
    s_tr = np.isin(donor, train)
    j_tr = np.isin(ja, train) & np.isin(jb, train)
    j_te = ~np.isin(ja, train) & ~np.isin(jb, train)
    return s_tr, j_tr, j_te, train


# ------------------------------------------------------------------ part 1

def working_model(Z, Y, W, add, cross):
    """Point fit by the paper's estimator: cross block from misattribution.fit (FWL); additive block by
    least squares given it.  Together these are the OLS coefficients of the full class."""
    c, _ = M.fit(Z, Y, W)
    beta, *_ = np.linalg.lstsq(dm(W(Z), add), Y - dm(W(Z), cross) @ c, rcond=None)
    return lambda Zn: dm(W(Zn), add) @ beta + dm(W(Zn), cross) @ c


def ols_model(Z, Y, W, cols):
    """Plain lstsq on the full design -- a numerical cross-check of working_model only."""
    coef, *_ = np.linalg.lstsq(dm(W(Z), cols), Y, rcond=None)
    return lambda Zn: dm(W(Zn), cols) @ coef


LAMBDAS = np.logspace(-6, 4, 21)


def ridge_cross(Z, Y, W, add, cross, lam):
    """min ||Y - A b - C c||^2 + lam ||c||^2 (penalty on the interaction block only), by augmentation."""
    A, C = dm(W(Z), add), dm(W(Z), cross)
    X = np.vstack([np.hstack([A, C]), np.hstack([np.zeros((len(cross), len(add))), np.sqrt(lam) * np.eye(len(cross))])])
    coef, *_ = np.linalg.lstsq(X, np.r_[Y, np.zeros(len(cross))], rcond=None)
    b, c = coef[:len(add)], coef[len(add):]
    return lambda Zn: dm(W(Zn), add) @ b + dm(W(Zn), cross) @ c


def ridge_cv(Z, Y, W, add, cross, groups, k=4):
    """lambda by k-fold CV over groups (donors) of the training rows; refit on all of them."""
    g = np.unique(groups); folds = np.array_split(np.random.default_rng(0).permutation(g), k)
    err = []
    for lam in LAMBDAS:
        e = 0.0
        for fo in folds:
            te = np.isin(groups, fo)
            e += float(np.sum((Y[te] - ridge_cross(Z[~te], Y[~te], W, add, cross, lam)(Z[te])) ** 2))
        err.append(e)
    lam = float(LAMBDAS[int(np.argmin(err))])
    return ridge_cross(Z, Y, W, add, cross, lam), lam


def score(y, yhat):
    """RMSE (readout units) and R^2 = 1 - SSE / sum (y - mean(y_test))^2 on the scored rows."""
    e = y - yhat
    return float(np.sqrt(np.mean(e ** 2))), float(1.0 - e @ e / np.sum((y - y.mean()) ** 2))


def heldout(d, Zs, Ys, Zj, Yj, seed):
    s_tr, j_tr, j_te, _ = partition(len(Zs), seed)
    trZ = np.vstack([Zs[s_tr], Zj[j_tr]])
    mu, sd = trZ.mean(0), trZ.std(0) + 1e-12                        # TRAIN-only whitening
    W = lambda Z: (Z - mu) / sd
    fam_tr = {"S": (Zs[s_tr], Ys[s_tr]), "J": (Zj[j_tr], Yj[j_tr])}
    _, ja, _ = joint_donors(len(Zs))
    donor = (np.arange(len(Zs)) % (len(Zs) // 2)) // N_AL
    grp_tr = {"S": donor[s_tr], "J": ja[j_tr]}                      # CV groups: donor / upstream donor
    fam_te = {"S": (Zs[~s_tr], Ys[~s_tr]), "J": (Zj[j_te], Yj[j_te])}
    out, out_add, out_ridge, lams, gap_route, gap_white = {}, {}, {}, {}, 0.0, 0.0
    with working_class(d) as (add, cross):
        for ft, (Zt, Yt) in fam_tr.items():
            pred_add = ols_model(Zt, Yt, W, add)                      # additive block only
            pred_rdg, lams[ft] = ridge_cv(Zt, Yt, W, add, cross, grp_tr[ft])
            for fe, (Ze, Ye) in fam_te.items():
                out_add[f"{ft}->{fe}"] = score(Ye, pred_add(Ze))
                out_ridge[f"{ft}->{fe}"] = score(Ye, pred_rdg(Ze))
            try:
                pred = working_model(Zt, Yt, W, add, cross)
            except np.linalg.LinAlgError:
                for fe in fam_te:
                    out[f"{ft}->{fe}"] = (float("nan"), float("nan"))
                continue
            pred_ols = ols_model(Zt, Yt, W, add + cross)
            # whitening from this family's training rows alone: the full class is closed under
            # per-coordinate affine maps, so predictions should not move (numerical check)
            mu2, sd2 = Zt.mean(0), Zt.std(0) + 1e-12
            pred_w2 = working_model(Zt, Yt, lambda Z: (Z - mu2) / sd2, add, cross)
            for fe, (Ze, Ye) in fam_te.items():
                yh = pred(Ze)
                out[f"{ft}->{fe}"] = score(Ye, yh)
                gap_route = max(gap_route, float(np.max(np.abs(yh - pred_ols(Ze))) / np.std(Ye)))
                gap_white = max(gap_white, float(np.max(np.abs(yh - pred_w2(Ze))) / np.std(Ye)))
    sizes = [int(s_tr.sum()), int((~s_tr).sum()), int(j_tr.sum()), int(j_te.sum())]
    sd_te = {"S": float(np.std(fam_te["S"][1])), "J": float(np.std(fam_te["J"][1]))}
    return out, out_add, out_ridge, lams, sizes, sd_te, gap_route, gap_white


# ------------------------------------------------------------------ part 2

def envelope(d, f):
    """misattribution_fourpoint.one under the degree-d class, plus the partial-information condition
    numbers (identifiability.partial_info, same full-data whitening as one())."""
    with working_class(d):
        r = F.one(f)
        if r is None:
            return None
        dd = np.load(f)
        allZ = np.vstack([dd["Zs"], dd["Zjm"]])
        mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
        W = lambda Z: (Z - mu) / sd
        for tag, Z in (("S", dd["Zs"]), ("J", dd["Zjm"])):
            ev = np.linalg.eigvalsh(I.partial_info(Z, W)[0])
            r[f"log10cond_{tag}"] = float(np.log10(ev.max() / max(ev.min(), 1e-300)))
    return r


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(median=float(np.median(x)), q25=float(np.percentile(x, 25)),
                q75=float(np.percentile(x, 75)), n=int(len(x)))


def main():
    os.chdir(HERE)
    files = sorted(glob.glob("qchart_*.npz"))
    if not files:
        raise SystemExit("no qchart_*.npz captures found")
    data = {f: {k: np.load(f)[k] for k in ("Zs", "Ys", "Zjm", "Yjm")} for f in files}
    map_err = max(joint_map_error(v["Zs"], v["Zjm"]) for v in data.values())
    assert map_err < 1e-9, f"joint rows do not map to their donors as donor_protocol states ({map_err})"

    # ---- part 1: held-out readout fit
    rows = {f: {"file": f, "reported": float(np.ptp(v["Ys"][len(v["Ys"]) // 2:])), "heldout": {},
                "envelope": {}} for f, v in data.items()}
    heldout_summary, sens, checks_route, checks_white, sizes_by_seed = {}, {}, {}, {}, {}
    add_summary, add_sens, rdg_sens, neg_by_seed = {}, {}, {}, {}
    sd_te_all = {"S": [], "J": []}
    for d in DEGREES:
        per_seed = {p: {"rmse": [], "r2": []} for p in PAIRS}
        add_seed = {p: [] for p in PAIRS}
        rdg_seed = {p: [] for p in PAIRS}
        lam_all = {"S": [], "J": []}
        neg_by_seed[str(d)] = {p: [] for p in PAIRS}
        gr = gw = 0.0
        for seed in SEEDS:
            acc = {p: {"rmse": [], "r2": [], "pct": []} for p in PAIRS}
            acc_add = {p: {"r2": [], "pct": []} for p in PAIRS}
            acc_rdg = {p: [] for p in PAIRS}
            for f, v in data.items():
                out, out_add, out_rdg, lams, sizes, sd_te, g1, g2 = heldout(d, v["Zs"], v["Ys"], v["Zjm"], v["Yjm"], seed)
                for ft in lam_all:
                    lam_all[ft].append(lams[ft])
                for p in PAIRS:
                    acc_rdg[p].append(out_rdg[p][1])
                    acc_add[p]["r2"].append(out_add[p][1])
                    acc_add[p]["pct"].append(100.0 * out_add[p][0] / rows[f]["reported"])
                gr, gw = max(gr, g1), max(gw, g2)
                sizes_by_seed[seed] = sizes
                for p in PAIRS:
                    acc[p]["rmse"].append(out[p][0]); acc[p]["r2"].append(out[p][1])
                    acc[p]["pct"].append(100.0 * out[p][0] / rows[f]["reported"])
                if seed == 0:
                    rows[f]["heldout"][str(d)] = {p: [out[p][0], out[p][1]] for p in PAIRS}
                    if d == DEGREES[0]:
                        for k in sd_te_all:
                            sd_te_all[k].append(sd_te[k])
            for p in PAIRS:
                per_seed[p]["rmse"].append(float(np.nanmedian(acc[p]["rmse"])))
                per_seed[p]["r2"].append(float(np.nanmedian(acc[p]["r2"])))
                add_seed[p].append(float(np.nanmedian(acc_add[p]["r2"])))
                rdg_seed[p].append(float(np.nanmedian(acc_rdg[p])))
                neg_by_seed[str(d)][p].append(int(np.sum(np.array(acc[p]["r2"]) < 0)))
            if seed == 0:
                add_summary[str(d)] = {p: dict(r2=qs(acc_add[p]["r2"]), rmse_pct_swing=qs(acc_add[p]["pct"]),
                                               r2_negative=int(np.sum(np.array(acc_add[p]["r2"]) < 0)))
                                       for p in PAIRS}
            if seed == 0:
                heldout_summary[str(d)] = {p: dict(rmse=qs(acc[p]["rmse"]), r2=qs(acc[p]["r2"]),
                                                   rmse_pct_swing=qs(acc[p]["pct"]),
                                                   r2_negative=int(np.sum(np.array(acc[p]["r2"]) < 0)))
                                           for p in PAIRS}
        sens[str(d)] = {p: dict(rmse_median_by_seed=per_seed[p]["rmse"], r2_median_by_seed=per_seed[p]["r2"],
                                rmse_median_over_seeds=float(np.median(per_seed[p]["rmse"])),
                                r2_median_over_seeds=float(np.median(per_seed[p]["r2"])),
                                rmse_median_range=[min(per_seed[p]["rmse"]), max(per_seed[p]["rmse"])],
                                r2_median_range=[min(per_seed[p]["r2"]), max(per_seed[p]["r2"])])
                        for p in PAIRS}
        rdg_sens[str(d)] = dict(
            r2={p: dict(r2_median_by_seed=rdg_seed[p], r2_median_over_seeds=float(np.median(rdg_seed[p])),
                        r2_median_range=[min(rdg_seed[p]), max(rdg_seed[p])]) for p in PAIRS},
            lambda_median={ft: float(np.median(v)) for ft, v in lam_all.items()},
            lambda_share_at_max={ft: float(np.mean(np.array(v) == LAMBDAS[-1])) for ft, v in lam_all.items()},
            lambda_share_at_min={ft: float(np.mean(np.array(v) == LAMBDAS[0])) for ft, v in lam_all.items()})
        add_sens[str(d)] = {p: dict(r2_median_by_seed=add_seed[p], r2_median_over_seeds=float(np.median(add_seed[p])),
                                    r2_median_range=[min(add_seed[p]), max(add_seed[p])]) for p in PAIRS}
        checks_route[str(d)], checks_white[str(d)] = gr, gw

    # ---- part 2: envelope by degree, full data
    env_summary, dropped = {}, {}
    for d in DEGREES:
        rs = []
        for f in files:
            r = envelope(d, f)
            if r is None:
                continue
            rs.append(r)
            rows[f]["envelope"][str(d)] = dict(med_S=r["med_S"], med_J=r["med_J"], max_S=r["max_S"],
                                               max_J=r["max_J"], log10cond_S=r["log10cond_S"],
                                               log10cond_J=r["log10cond_J"])
        dropped[str(d)] = len(files) - len(rs)
        g = lambda k: np.array([r[k] for r in rs])
        env_summary[str(d)] = dict(
            n=len(rs), n_cross=len(poly_class(d)[1]), crit=float(chi2.ppf(0.95, len(poly_class(d)[1]))),
            med_S_pct=100 * float(np.median(g("med_S"))), med_J_pct=100 * float(np.median(g("med_J"))),
            max_S_pct=100 * float(np.median(g("max_S"))), max_J_pct=100 * float(np.median(g("max_J"))),
            ratio=float(np.median(g("med_S")) / max(np.median(g("med_J")), 1e-300)),
            ordered=int(np.sum(g("med_S") > g("med_J"))),
            captures_med_S_over_100pct=int(np.sum(g("med_S") > 1.0)),
            log10cond_S_median=float(np.median(g("log10cond_S"))),
            log10cond_J_median=float(np.median(g("log10cond_J"))))

    # ---- degree 3 must reproduce the ledger's misattribution_fourpoint.json row by row
    ledger = {r["file"]: r for r in json.load(open("misattribution_fourpoint.json"))["rows"]}
    rel = [abs(rows[f]["envelope"]["3"][k] - ledger[f][k]) / max(abs(ledger[f][k]), 1e-300)
           for f in files if "3" in rows[f]["envelope"] and f in ledger
           for k in ("med_S", "med_J", "max_S", "max_J")]
    n_matched = sum(1 for f in files if "3" in rows[f]["envelope"] and f in ledger)

    s_tr, j_tr, j_te, train = partition(len(data[files[0]]["Zs"]), 0)
    res = dict(
        n_captures=len(files),
        classes={str(d): dict(ADD=poly_class(d)[0], CROSS=poly_class(d)[1],
                              wald_crit=float(chi2.ppf(0.95, len(poly_class(d)[1])))) for d in DEGREES},
        partition=dict(seed=0, rule="positive_control_heldout.donor_protocol", train_donors=sorted(map(int, train)),
                       sizes_S_train_S_test_J_train_J_test=sizes_by_seed[0],
                       sizes_by_seed={str(s): v for s, v in sizes_by_seed.items()},
                       heldout_readout_sd_median=dict(S=float(np.median(sd_te_all["S"])),
                                                      J=float(np.median(sd_te_all["J"])))),
        heldout=heldout_summary,
        heldout_partition_sensitivity=sens,
        heldout_additive=add_summary,
        heldout_additive_partition_sensitivity=add_sens,
        heldout_ridge_cross_partition_sensitivity=rdg_sens,
        heldout_r2_negative_by_seed=neg_by_seed,
        envelope=env_summary,
        checks=dict(joint_donor_map_max_abs_err=map_err,
                    deg3_envelope_max_rel_diff_vs_misattribution_fourpoint_json=float(max(rel)) if rel else None,
                    deg3_envelope_captures_matched=n_matched,
                    fwl_vs_lstsq_max_pred_gap_over_test_sd=checks_route,
                    whitening_invariance_max_pred_gap_over_test_sd=checks_white,
                    envelope_captures_dropped=dropped),
        estimands=dict(
            unit="capture (one qchart_*.npz: one S-Inhibition -> Name Mover head pair x prompt seed); 31 captures",
            readout="logit(IO) - logit(S) at the final position (ioi_query_chart.collect)",
            rmse="sqrt(mean over held-out rows of (Y - Yhat)^2), readout units; held-out rows = single-site rows of"
                 " held-out donors (S) or joint rows with both donors held out (J); seed-0 donor partition",
            r2="1 - SSE / sum over the same held-out rows of (Y - mean of those held-out Y)^2",
            rmse_pct_swing="100 * rmse / ptp(B-arm single-site readout, full data) -- a reporting unit only, "
                           "not used in any fit",
            heldout_aggregation="median (and q25/q75) over captures of the per-capture value; pair 'X->Y' = fitted"
                                " on training rows of family X, scored on held-out rows of family Y",
            partition_sensitivity="the capture-median of rmse / r2 for each of seeds 0..9 (donor_protocol-style"
                                  " partitions), and its min-max across seeds",
            med_S_pct="median over captures of [median over joint states of the 95% Wald worst case of |D c|, c and"
                      " its covariance fitted on ALL single-site rows] / ptp(B-arm readout), x100",
            med_J_pct="as med_S_pct with c fitted on all joint (support-matched) rows",
            max_S_pct="median over captures of [max over joint states of the same worst case] / swing, x100",
            ratio="median(med_S) / median(med_J) over captures",
            ordered="captures with per-capture med_S > med_J",
            log10cond="log10 condition number of the FWL partial information for the cross block"
                      " (identifiability.partial_info), full-data whitening; median over captures"),
        rows=list(rows.values()))
    json.dump(res, open("response_diagnostics.json", "w"), indent=1, default=float)

    # ---- print
    print(f"n = {len(files)} captures | donor partition seed 0: train donors {sorted(map(int, train))}")
    print(f"rows  S train/test {sizes_by_seed[0][0]}/{sizes_by_seed[0][1]}   J train/test "
          f"{sizes_by_seed[0][2]}/{sizes_by_seed[0][3]} | held-out readout sd (median) "
          f"S {res['partition']['heldout_readout_sd_median']['S']:.3f}  J {res['partition']['heldout_readout_sd_median']['J']:.3f}")
    print("\nPART 1  held-out readout fit.  Left: seed-0 partition, median over captures.  Right: the capture-median"
          " for each of seeds 0..9 -> its median over seeds [min, max]")
    print(f"{'deg':>3} {'fit->test':>9} {'RMSE':>10} {'RMSE %swing':>12} {'R^2':>12} {'R^2<0':>6}"
          f" {'RMSE seeds med [min,max]':>34} {'R^2 seeds med [min,max]':>40}")
    for d in DEGREES:
        for p in PAIRS:
            h, s = heldout_summary[str(d)][p], sens[str(d)][p]
            print(f"{d:>3} {p:>9} {h['rmse']['median']:>10.4f} {h['rmse_pct_swing']['median']:>11.2f}%"
                  f" {h['r2']['median']:>12.4f} {h['r2_negative']:>3}/{h['r2']['n']:<2}"
                  f"  {s['rmse_median_over_seeds']:>9.4f} [{s['rmse_median_range'][0]:>8.4f},{s['rmse_median_range'][1]:>9.4f}]"
                  f"  {s['r2_median_over_seeds']:>12.4f} [{s['r2_median_range'][0]:>12.4f},{s['r2_median_range'][1]:>10.4f}]")
    print("J train/test rows by seed:", {s: v[2:] for s, v in sizes_by_seed.items()})
    print("\nADDITIVE block only, same partitions: R^2 capture-median, median over seeds [min, max]")
    for d in DEGREES:
        print(f"{d:>3} " + "  ".join(f"{p} {add_sens[str(d)][p]['r2_median_over_seeds']:.3f} "
                                     f"[{add_sens[str(d)][p]['r2_median_range'][0]:.3f},{add_sens[str(d)][p]['r2_median_range'][1]:.3f}]"
                                     for p in PAIRS))
    print("\nRIDGE on the interaction block, lambda by donor-grouped CV inside the fitted family: R^2 median over seeds [min, max]")
    for d in DEGREES:
        r = rdg_sens[str(d)]
        print(f"{d:>3} " + "  ".join(f"{p} {r['r2'][p]['r2_median_over_seeds']:.3f} "
                                     f"[{r['r2'][p]['r2_median_range'][0]:.3f},{r['r2'][p]['r2_median_range'][1]:.3f}]" for p in PAIRS)
              + f"  | lambda median S {r['lambda_median']['S']:.2g} J {r['lambda_median']['J']:.2g}"
              + f"  at max S {r['lambda_share_at_max']['S']:.2f} J {r['lambda_share_at_max']['J']:.2f}")
    print("negative-R^2 counts by seed:", {d: {p: v for p, v in x.items() if p == 'S->J'} for d, x in neg_by_seed.items()})
    print("\nPART 2  four-point envelope by degree (full data; median over captures; % of B-arm swing)")
    print(f"{'deg':>3} {'|CROSS|':>7} {'n':>3} {'med S':>10} {'med J':>9} {'ratio':>9} {'S>J':>6}"
          f" {'worst S':>11} {'worst J':>9} {'S>100%':>7} {'log10cond S':>12} {'J':>6}")
    for d in DEGREES:
        e = env_summary[str(d)]
        print(f"{d:>3} {e['n_cross']:>7} {e['n']:>3} {e['med_S_pct']:>9.2f}% {e['med_J_pct']:>8.3f}%"
              f" {e['ratio']:>9.1f} {e['ordered']:>3}/{e['n']:<2} {e['max_S_pct']:>10.1f}% {e['max_J_pct']:>8.2f}%"
              f" {e['captures_med_S_over_100pct']:>7} {e['log10cond_S_median']:>12.1f} {e['log10cond_J_median']:>6.1f}")
    c = res["checks"]
    print(f"\nchecks: joint-donor map max err {c['joint_donor_map_max_abs_err']:.1e} | degree-3 envelope vs "
          f"misattribution_fourpoint.json max rel diff "
          f"{c['deg3_envelope_max_rel_diff_vs_misattribution_fourpoint_json']:.1e} over {n_matched} captures")
    print("        FWL-vs-lstsq max held-out prediction gap / test sd:",
          {k: f"{v:.1e}" for k, v in checks_route.items()})
    print("        whitening-invariance max prediction gap / test sd:",
          {k: f"{v:.1e}" for k, v in checks_white.items()}, "| envelope drops", dropped)
    print("saved -> response_diagnostics.json")


if __name__ == "__main__":
    main()
