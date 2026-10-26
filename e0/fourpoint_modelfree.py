"""IOI attribution analysis, model-free: the MEASURED controlled four-point transport contrast on the existing IOI captures.

WHY THIS FILE EXISTS (2026-09-24).  Two workshop reviewers and an external review asked
whether the response model behind the IOI attribution analysis describes the network, and pointed out that held-out readout R^2 does not
validate the particular contrast the section is about.  This file measures that contrast directly, with no response
model, and scores the fitted models against it.

STATUS.  EXPLORATORY, on captures that have been inspected before.  A scoping prototype (session scratch, r2_scope/proto.py)
computed capture medians of |C| / swing on these files before this file existed; the margin below was fixed after that
prototype and before any RMS summary in raw logits was computed.  Nothing here is confirmatory.

THE CONTRAST.  For a joint row with upstream descriptor u = (donor ja, dose ka) and downstream descriptor v = (jb, kb):

    T_C(u, v) = Y(u, v) - Y(u, 0) - Y(0, v) + Y(0, 0)

where every term is an EXECUTED readout on the same base prompt (ioi_query_chart.collect):
    Y(u, v) = Yjm[r]                      both heads forced: S <- Sclean + AL[ka](dS[ja] - Sclean), NM <- Nclean + AL[kb](dN[jb] - Nclean)
    Y(u, 0) = Yc[ja * N_AL + ka]          the CLAMPED arm: the same S vector, NM forced to its clean vector
    Y(0, v) = Ys[n + jb * N_AL + kb]      the NM arm: the same NM vector, S upstream and therefore at its clean value
    Y(0, 0) = clean
An A-only row with B recomputing (Ys[:n]) is NOT a valid Y(u, 0) and is never used.  The identity of the injected vectors is
fixed by construction in ioi_query_chart.collect; here it is checked on the projected coordinates (max error asserted 0).

ESTIMANDS (per capture; raw logit-difference units first, then / the B-arm swing ptp(Ys[B arm]) as in the IOI attribution analysis).
    RMS_T      = sqrt(mean T_C^2) over the recorded joint grid (all 256 rows) and over the held-out rows (both donors held
                 out under the seed-0 donor partition of response_diagnostics / donor_protocol).  On the recorded grid this is a
                 deterministic finite-grid fact.
    RMSE_pred  = sqrt(mean (T_C - T_hat)^2) on the held-out rows, for T_hat from: zero (= RMS_T), the working model fitted on
                 single-site TRAINING rows, and the working model fitted on joint TRAINING rows (degree 3 primary; 2 and 4
                 reported, never selected).  T_hat is the same four-point contrast of the model's predictions at the four
                 executed corner coordinates.  The additive-only model's contrast is identically zero, so it is not a
                 separate baseline.
    signed mean of T_C (secondary: opposite signs can cancel).
MARGIN.  0.10 logits RMS (an effect-size convention taken from a planned confirmatory design; borrowing it does not make this confirmatory).
    "RMS_T < 0.10 on the recorded grid" is a finite-grid statement.  "Supported below 0.10 for the donor population" needs
    the nominal donor-conditional 95% upper bound below the margin.
INTERVALS.  10,000 donor resamples per capture, conditional on the captured prompt and head pair and holding the fitted
    predictors fixed: the SAME donor identities supply both axes (dS[v] and dN[v] come from one donor prompt v), so one
    resample of donor identities is drawn and every pair (i, j) of resampled identities is kept (Cartesian structure carried
    consistently).  Grid rows and the four terms are never resampled independently.  Nominal, approximate.
ACROSS CAPTURES.  The 31 configurations share three base prompts (one per seed) and their donors; they are summarised as a
    distribution of capture-level estimates (median, quartiles, max, counts), with no pooled significance claim.

    python3 fourpoint_modelfree.py      # -> fourpoint_modelfree.json
"""
import glob
import json
import os

import numpy as np

import misattribution as M
from cells import N_AL
from identifiability import dm
from response_diagnostics import DEGREES, SEEDS, joint_donors, partition, working_class, working_model

HERE = os.path.dirname(os.path.abspath(__file__))
MARGIN = 0.10          # logits, RMS
B = 10_000
PRIMARY_DEGREE = 3


def quartets(d):
    """Executed readouts and corner coordinates of every joint row's controlled quartet."""
    Zs, Ys, Zc, Yc, Zj, Yj = (d[k] for k in ("Zs", "Ys", "Zc", "Yc", "Zjm", "Yjm"))
    y00 = float(d["clean"])
    n = len(Zs) // 2
    _, ja, jb = joint_donors(len(Zs))
    ka, kb = (ja + jb) % N_AL, (3 * ja + jb) % N_AL
    iu, iv = ja * N_AL + ka, n + jb * N_AL + kb
    Yu0, Y0v = Yc[iu], Ys[iv]
    T = Yj - Yu0 - Y0v + y00
    corners = dict(uv=Zj, u0=Zc[iu], v0=Zs[iv])                     # (a, b) coordinates of the executed corners
    a0 = float(np.median(Zs[n:, 0])); b0 = float(np.median(Zc[:n, 1]))
    audit = dict(joint_a_vs_clamped=float(np.abs(Zj[:, 0] - Zc[iu, 0]).max()),
                 joint_b_vs_nm_arm=float(np.abs(Zj[:, 1] - Zs[iv, 1]).max()),
                 clamped_b_spread=float(np.ptp(Zc[:n, 1])), nm_arm_a_spread=float(np.ptp(Zs[n:, 0])))
    return T, corners, (a0, b0), ja, jb, audit


def model_contrast(pred, corners, a0b0):
    """Four-point contrast of a fitted predictor at the executed corner coordinates."""
    a0, b0 = a0b0
    c00 = np.array([[a0, b0]])
    return pred(corners["uv"]) - pred(corners["u0"]) - pred(corners["v0"]) + pred(c00)[0]


def fitted_contrasts(d, corners, a0b0, seed):
    """Working models fitted on TRAINING rows only (train-only whitening, as response_diagnostics.heldout)."""
    Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
    s_tr, j_tr, j_te, train = partition(len(Zs), seed)
    trZ = np.vstack([Zs[s_tr], Zj[j_tr]])
    mu, sd = trZ.mean(0), trZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    out = {}
    for deg in DEGREES:
        with working_class(deg) as (add, cross):
            for fam, (Zt, Yt) in (("S", (Zs[s_tr], Ys[s_tr])), ("J", (Zj[j_tr], Yj[j_tr]))):
                try:
                    out[(deg, fam)] = model_contrast(working_model(Zt, Yt, W, add, cross), corners, a0b0)
                except np.linalg.LinAlgError:
                    out[(deg, fam)] = np.full(len(Zj), np.nan)
    return out, j_te, train


rms = lambda x: float(np.sqrt(np.mean(np.square(x))))


def boot(T, preds, ja, jb, keep_ids, rng):
    """Donor-identity resamples over keep_ids; every (i, j) pair of resampled identities is kept."""
    ids = np.array(sorted(keep_ids))
    pos = {int(v): k for k, v in enumerate(ids)}
    sel = np.isin(ja, ids) & np.isin(jb, ids)
    grid = -np.ones((len(ids), len(ids)), int)
    for r in np.where(sel)[0]:
        grid[pos[int(ja[r])], pos[int(jb[r])]] = r
    assert (grid >= 0).all()
    stats = {"T": [], **{k: [] for k in preds}, **{f"{k}-zero": [] for k in preds}}
    for _ in range(B):
        s = rng.integers(0, len(ids), len(ids))
        rows = grid[np.ix_(s, s)].ravel()
        t = T[rows]; stats["T"].append(rms(t))
        for k, p in preds.items():
            e = rms(t - p[rows]); stats[k].append(e); stats[f"{k}-zero"].append(e - rms(t))
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {k: q(np.array(v)) for k, v in stats.items()}


def one(f, rng):
    d = {k: np.load(f)[k] for k in ("Zs", "Ys", "Zc", "Yc", "Zjm", "Yjm", "clean")}
    T, corners, a0b0, ja, jb, audit = quartets(d)
    assert audit["joint_a_vs_clamped"] < 1e-9 and audit["joint_b_vs_nm_arm"] < 1e-9, (f, audit)
    n = len(d["Ys"]) // 2
    swing = float(np.ptp(d["Ys"][n:]))
    fits, j_te, train = fitted_contrasts(d, corners, a0b0, 0)
    held = sorted(set(int(v) for v in ja[j_te]))
    prim = {"S": fits[(PRIMARY_DEGREE, "S")], "J": fits[(PRIMARY_DEGREE, "J")]}
    r = dict(file=os.path.basename(f), seed=int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]),     # qchart_<si><nm>_100_<seed>
             swing=swing, audit=audit, n_grid=int(len(T)), n_heldout=int(j_te.sum()),
             rms_T_grid=rms(T), rms_T_heldout=rms(T[j_te]), mean_T_grid=float(T.mean()), max_abs_T_grid=float(np.abs(T).max()),
             rmse_heldout={f"d{deg}_{fam}": rms(T[j_te] - v[j_te]) for (deg, fam), v in fits.items()},
             ci_heldout=boot(T, prim, ja, jb, held, rng), ci_grid_rms_T=boot(T, {}, ja, jb, sorted(set(int(v) for v in ja)), rng)["T"])
    # the other nine donor partitions: held-out RMS_T and primary-degree prediction errors (sensitivity, not replications)
    sens = []
    for s in SEEDS[1:]:
        fs, jt, _ = fitted_contrasts(d, corners, a0b0, s)
        sens.append(dict(seed=s, rms_T=rms(T[jt]), S=rms(T[jt] - fs[(PRIMARY_DEGREE, "S")][jt]), J=rms(T[jt] - fs[(PRIMARY_DEGREE, "J")][jt])))
    r["partition_sensitivity"] = sens
    return r


def summarise(R):
    g = lambda k: np.array([x[k] for x in R])
    q = lambda v: dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                       max=float(np.max(v)), min=float(np.min(v)))
    sw = g("swing")
    out = dict(n=len(R), swing=q(sw), swing_below_0p5=int((sw < 0.5).sum()),
               rms_T_grid=q(g("rms_T_grid")), rms_T_heldout=q(g("rms_T_heldout")),
               rms_T_grid_pct_swing=q(100 * g("rms_T_grid") / sw),
               grid_below_margin=int((g("rms_T_grid") < MARGIN).sum()),
               upper95_below_margin=int(sum(x["ci_heldout"]["T"][1] < MARGIN for x in R)),
               upper95_grid_below_margin=int(sum(x["ci_grid_rms_T"][1] < MARGIN for x in R)),
               mean_T_grid=q(g("mean_T_grid")), max_abs_T_grid=q(g("max_abs_T_grid")))
    for deg in DEGREES:
        for fam in ("S", "J"):
            v = np.array([x["rmse_heldout"][f"d{deg}_{fam}"] for x in R])
            out[f"rmse_heldout_d{deg}_{fam}"] = q(v)
            out[f"rmse_heldout_d{deg}_{fam}_pct_swing"] = q(100 * v / sw)
    out["S_worse_than_zero"] = int(sum(x["ci_heldout"]["S-zero"][0] > 0 for x in R))
    out["J_worse_than_zero"] = int(sum(x["ci_heldout"]["J-zero"][0] > 0 for x in R))
    out["partition_sensitivity"] = {k: q(np.array([np.median([s[k] for s in x["partition_sensitivity"]]) for x in R]))
                                    for k in ("rms_T", "S", "J")}
    return out


def main():
    os.chdir(HERE)
    rng = np.random.default_rng(20260924)
    res = dict(margin_logits=MARGIN, bootstrap_draws=B, primary_degree=PRIMARY_DEGREE,
               status="EXPLORATORY: previously inspected captures; a scoping prototype computed normalised medians of |T_C| "
                      "before the margin was fixed", groups={})
    for tag, pat in (("published", "qchart_*.npz"), ("controls", "qctrl_*.npz")):
        R = [one(f, rng) for f in sorted(glob.glob(pat))]
        res["groups"][tag] = dict(summary=summarise(R), captures=R)
    json.dump(res, open(os.path.join(HERE, "fourpoint_modelfree.json"), "w"), indent=1, default=float)
    for tag in ("published", "controls"):
        s = res["groups"][tag]["summary"]
        print(f"{tag}: n={s['n']}  RMS_T grid median {s['rms_T_grid']['median']:.4f} logits (max {s['rms_T_grid']['max']:.4f}); "
              f"{s['rms_T_grid_pct_swing']['median']:.3f}% of swing; grid<0.10 in {s['grid_below_margin']}; "
              f"held-out upper95<0.10 in {s['upper95_below_margin']}")
        print(f"   held-out RMSE d3: zero {s['rms_T_heldout']['median']:.4f}  S-fit {s['rmse_heldout_d3_S']['median']:.4f}  "
              f"J-fit {s['rmse_heldout_d3_J']['median']:.4f}  | S worse than zero (CI) {s['S_worse_than_zero']}, J {s['J_worse_than_zero']}")


if __name__ == "__main__":
    main()
