"""What drives the Table 1 named-generator ratio?  (exploratory, added 2026-09-24; review item)

The paper read the held-out ratio of g = (a - a0)(b - m a - c) as "largely measuring how nearly affine
the realised edge is".  A review recomputed, on the 185 support-matched captures, a Spearman of
-0.88 with a B-spread ratio and -0.05 with the edge R^2.  This script re-derives both on the SAME
captures and the SAME corrected pipeline Table 1 reads (positive_control_heldout: edge, whitening and
column scaling fitted on TRAINING rows only), with the prompt as the resampling unit.

REUSED, NOT COPIED.  Captures: the ledger's `mpat` literal (ledger_paperA.py).  Splits and the g / g_C
scores: ofat_baseline.row_split / donor_split / score, which rebuild positive_control_heldout's index
sets and assert g's ratio against positive_control_heldout._score.  Here every g ratio is additionally
checked against positive_control_heldout.json (per capture) and claims_paperA.json (the
Table 1 macros: per-cell medians and counts, PredMTotalCount, DonorNamedCount, HeldoutBeatsNull).

PER-CAPTURE DIAGNOSTICS (all from intervention STATES, no readout).  "Patch-A arm" = the single-site
arm in which A is forced from donors and B recomputes; it is identified exactly as _fit_edge does it
(the arm with the larger sd of a among the rows used).  "Joint" = the support-matched joint family
Zjm that Table 1 scores.  Primary versions use the TRAINING rows of the protocol's split, i.e. the
rows the frozen direction was fitted on; `*_all` versions use every row of the capture (the
reviewer's recomputation, reported for reconciliation).
  edge_r2            R^2 of the least-squares line b ~ a on the patch-A arm (the fit that gives m, c)
  sd_ratio_joint     sd(b | patch-A arm) / sd(b | joint family)                  <- item 6's quantity
  sd_ratio_armB      sd(b | patch-A arm) / sd(b | patch-B arm)  (edge_strength.py's "edge strength")
  resid_ratio_joint  sd(b - m a - c | patch-A arm) / sd(b | joint) = sd_ratio_joint * sqrt(1 - R^2):
                     the factor by which g's single-site energy on that arm is small
  abs_m_whitened     |m| in train-whitened units (ofat_baseline.score)

    python3 ratio_drivers.py      # -> ratio_drivers.json   (CPU, ~1 min)
"""
import sys
sys.dont_write_bytecode = True                  # leave e0/__pycache__ untouched
import ast, glob, json, os, time
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ofat_baseline import row_split, donor_split, score, THR   # noqa: E402
from prompt_cluster import prompt_of_capture                     # noqa: E402

LEDGER = os.path.join(HERE, "..", "claims_paperA.json")
N_BOOT, SEED = 2000, 0
PROTOS = dict(row=row_split, donor=donor_split)      # row = Table 1; donor = DonorNamedCount


def mpat():
    src = open(os.path.join(HERE, "..", "ledger_paperA.py")).read()
    s = src.index("mpat = {"); e = src.index("}", s) + 1
    return ast.literal_eval(src[s + len("mpat = "):e])


def arm_a_mask(Zs, arm, rows):
    """Boolean mask over `rows` selecting the patch-A arm, identified as positive_control_heldout._fit_edge
    identifies it (larger sd of a) from those rows only."""
    a = Zs[rows, 0]; r_arm = arm[rows]
    h1, h2 = a[r_arm == 0], a[r_arm == 1]
    lab = 1 if np.std(h1) < np.std(h2) else 0
    return r_arm == lab


def edge_diag(Zs, Zj, arm, s_rows, j_rows):
    A = Zs[s_rows][arm_a_mask(Zs, arm, s_rows)]
    B = Zs[s_rows][~arm_a_mask(Zs, arm, s_rows)]
    r = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
    r2 = float(r ** 2) if np.isfinite(r) else 0.0
    sdA, sdB, sdJ = float(np.std(A[:, 1])), float(np.std(B[:, 1])), float(np.std(Zj[j_rows, 1]))
    m, c = np.polyfit(A[:, 0], A[:, 1], 1)
    sdRes = float(np.std(A[:, 1] - m * A[:, 0] - c))
    return dict(edge_r2=r2, sd_ratio_joint=sdA / sdJ, sd_ratio_armB=sdA / sdB,
                resid_ratio_joint=sdRes / sdJ, sd_b_armA=sdA, sd_b_armB=sdB, sd_b_joint=sdJ)


def cluster_boot_spearman(x, y, groups, n_boot=N_BOOT, seed=SEED):
    """Spearman with a percentile CI from resampling (cell, prompt) groups with replacement."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y)[0]); p = float(spearmanr(x, y)[1])
    keys = sorted(set(groups)); idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[keys[k]] for k in rng.integers(len(keys), size=len(keys))])
        if np.ptp(x[pick]) == 0 or np.ptp(y[pick]) == 0:
            continue
        bs.append(spearmanr(x[pick], y[pick])[0])
    return dict(rho=rho, p_capture_level=p, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(x)), groups=len(keys), n_boot=len(bs))


def summarise(rows, proto, mask, label):
    g = np.array([r[proto]["gR"] for r in rows])[mask]
    grp = [r["group"] for r, k in zip(rows, mask) if k]
    out = dict(subset=label, n=int(mask.sum()), groups=len(set(grp)), median_g_ratio=float(np.median(g)))
    for key in ("sd_ratio_joint", "edge_r2", "resid_ratio_joint", "sd_ratio_armB", "abs_m_whitened",
                "sd_ratio_joint_all", "edge_r2_all"):
        v = np.array([r[proto][key] for r in rows])[mask]
        out[key] = dict(median=float(np.median(v)), iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                        spearman_with_g_ratio=cluster_boot_spearman(v, g, grp))
    e = np.array([r[proto]["edge_r2"] for r in rows])[mask]
    out["median_nonaffine_share_1_minus_r2"] = float(np.median(1 - e))
    return out


def main():
    t0 = time.time()
    os.chdir(HERE)
    HO = json.load(open("positive_control_heldout.json"))["cells"]
    L = json.load(open(LEDGER))
    P = prompt_of_capture()
    rows, cellrows = [], defaultdict(list)
    for pre, pat in mpat().items():
        ledger = {r["file"]: r for r in HO[pre]["rows"]}
        for f in sorted(glob.glob(pat)):
            d = np.load(f)
            if "Zjm" not in d.files:
                continue
            Zs, Zj, Zc, arm = d["Zs"], d["Zjm"], d["Zc"], d["arm"]
            r = dict(file=f, cell=pre, group=f"{pre}|{P[f]}")
            for proto, split in PROTOS.items():
                idx = split(len(Zs), len(Zj))
                sc = score(Zs, Zj, Zc, arm, idx)                     # asserts g vs _score
                assert np.isclose(sc["gR"], ledger[f][proto]["ratio"], rtol=1e-9), (f, proto)
                assert np.isclose(sc["euclid_p95"], ledger[f][proto]["rnd_p95"], rtol=1e-9), (f, proto)
                s_tr, _, j_tr, _ = idx
                dg = edge_diag(Zs, Zj, arm, s_tr, j_tr)
                da = edge_diag(Zs, Zj, arm, np.arange(len(Zs)), np.arange(len(Zj)))
                r[proto] = dict(gR=sc["gR"], gC=sc["gC"], euclid_p95=sc["euclid_p95"], inv_p95=sc["inv_p95"],
                                abs_m_whitened=abs(sc["m_whitened"]), **dg,
                                sd_ratio_joint_all=da["sd_ratio_joint"], edge_r2_all=da["edge_r2"],
                                sd_ratio_armB_all=da["sd_ratio_armB"])
            rows.append(r); cellrows[pre].append(r)

    # ---------------- verification against the ledger Table 1 reads ----------------
    checks = []
    for pre, rs in cellrows.items():
        g = np.array([r["row"]["gR"] for r in rs])
        med, cnt = float(np.median(g)), f"{int((g > THR).sum())}/{len(g)}"
        assert np.isclose(med, L[f"predictM.{pre}"]["value"], rtol=1e-9), pre
        assert cnt == L[f"predictM.{pre}.n"]["value"], pre
        checks.append(dict(cell=pre, median=med, ledger_latex=L[f"predictM.{pre}"]["latex"], count=cnt))
    tot = lambda proto: f"{sum(r[proto]['gR'] > THR for r in rows)}/{len(rows)}"
    assert tot("row") == L["predictM.total"]["value"] == "171/185"
    assert tot("donor") == L["predictM.donor"]["value"]
    beat = f"{sum(r['row']['gR'] > r['row']['euclid_p95'] for r in rows)}/{len(rows)}"
    assert beat == L["predictM.beatnull"]["value"]
    assert len({r["group"] for r in rows}) == int(L["predictM.prompts"]["value"]) == 38
    verification = dict(ok=True, per_cell=checks, PredMTotalCount=tot("row"), DonorNamedCount=tot("donor"),
                        HeldoutBeatsNull=beat, prompt_groups=38,
                        note="every per-capture g ratio equals positive_control_heldout.json (rtol 1e-9); "
                             "cell medians/counts equal claims_paperA.json predictM.* values")

    # ---------------- analysis ----------------
    res = {}
    for proto in PROTOS:
        g = np.array([r[proto]["gR"] for r in rows])
        p95 = np.array([r[proto]["euclid_p95"] for r in rows])
        subsets = dict(all=np.ones(len(rows), bool), escape_gt10=g > THR, beats_null=g > p95)
        res[proto] = {k: summarise(rows, proto, m, k) for k, m in subsets.items()}
        # within-cell Spearman (removes between-cell differences in edge and ratio level)
        wc = {}
        for pre, rs in cellrows.items():
            gg = [r[proto]["gR"] for r in rs]
            wc[pre] = dict(n=len(rs),
                           sd_ratio_joint=float(spearmanr([r[proto]["sd_ratio_joint"] for r in rs], gg)[0]),
                           edge_r2=float(spearmanr([r[proto]["edge_r2"] for r in rs], gg)[0]))
        res[proto]["within_cell_spearman"] = wc
        res[proto]["within_cell_spearman_median"] = dict(
            sd_ratio_joint=float(np.median([v["sd_ratio_joint"] for v in wc.values()])),
            edge_r2=float(np.median([v["edge_r2"] for v in wc.values()])))

    meta = dict(
        captures="ledger mpat (ledger_paperA.py), files carrying Zjm: 185 support-matched captures, 10 cells, 38 (cell, prompt) groups",
        g_ratio="Table 1 statistic: held-out mean (phi.g)^2 over test Zjm rows / over test Zs rows; edge, whitening, "
                "scaling fitted on training rows (positive_control_heldout); 'row' = Table 1 split, 'donor' = common "
                "donor partition, joint rows scored only if both donors held out",
        definitions={"patch-A arm": "single-site arm with A forced from donors and B recomputed; identified as the arm "
                                     "with larger sd(a) among the rows used (as _fit_edge)",
                     "sd_ratio_joint": "sd(b | patch-A arm, training S rows) / sd(b | support-matched joint Zjm, training J rows)",
                     "sd_ratio_joint_all": "same, all rows of the capture (reviewer's recomputation)",
                     "edge_r2": "R^2 of OLS b ~ a on patch-A arm training rows (the fit giving m, c)",
                     "resid_ratio_joint": "sd(b - m a - c | patch-A arm, training) / sd(b | joint, training)",
                     "sd_ratio_armB": "sd(b | patch-A arm) / sd(b | patch-B arm), training rows (edge_strength.py's edge strength)"},
        bootstrap=f"{N_BOOT} resamples of the 38 (cell, prompt) groups with replacement, percentile 95% CI, seed {SEED}; "
                  "p_capture_level treats captures as independent and is descriptive only",
        escape_subsets="escape_gt10: g ratio > 10 (Table 1 count, 171 row / 173 donor); beats_null: g ratio > the capture's "
                       "Euclidean random-direction p95 (HeldoutBeatsNull, 180 row)",
        runtime_s=None)
    per_capture = [dict(file=r["file"], cell=r["cell"], group=r["group"], **{p: r[p] for p in PROTOS}) for r in rows]
    meta["runtime_s"] = round(time.time() - t0, 1)
    json.dump(dict(meta=meta, verification=verification, results=res, captures=per_capture),
              open("ratio_drivers.json", "w"), indent=1, default=float)

    for proto in PROTOS:
        print(f"\n=== {proto} protocol ===")
        for sub in ("all", "escape_gt10", "beats_null"):
            s = res[proto][sub]
            print(f"  [{sub}] n={s['n']} groups={s['groups']} median g={s['median_g_ratio']:.1f}  "
                  f"median 1-R2={s['median_nonaffine_share_1_minus_r2']:.3f}")
            for k in ("sd_ratio_joint", "sd_ratio_joint_all", "resid_ratio_joint", "sd_ratio_armB", "edge_r2",
                      "edge_r2_all", "abs_m_whitened"):
                v = s[k]; sp = v["spearman_with_g_ratio"]
                print(f"    {k:20s} median {v['median']:.3f}  rho(g) {sp['rho']:+.3f} "
                      f"[{sp['ci95'][0]:+.3f}, {sp['ci95'][1]:+.3f}]")
        print("  within-cell Spearman medians:", res[proto]["within_cell_spearman_median"])
    print(f"\nverification OK; runtime {meta['runtime_s']} s; saved -> ratio_drivers.json")


if __name__ == "__main__":
    main()
