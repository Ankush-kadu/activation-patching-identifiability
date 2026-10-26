"""Classical one-factor-at-a-time (clamped) baseline for Table 1  (added 2026-09-23).

WHY.  Table 1 reports that the theory-named recompute generator g_R = (a - a0)(b - m a - c) has a
held-out joint/single-site squared-response ratio above 10x in most support-matched captures
(positive_control_heldout.py).  A reviewer pointed out that the classical clamped contrast
g_C = (a - a0)(b - b0), which needs only the clean point (a0, b0) and no edge fit, already clears
10x in 149/185 captures under the OLD all-rows (leaky) protocol.  This file asks the same question
under the two CORRECTED protocols, so that the fraction of the Table 1 effect that needs
"the network completes the design" (the edge m, c) can be read off directly.

WHAT IS HELD FIXED.  Same captures (the ledger's `mpat` dict in ledger_paperA.py, files carrying
Zjm), same row and donor splits as positive_control_heldout.row_protocol / donor_protocol (the
index sets are rebuilt here line for line and VERIFIED: g_R's ratio and its Euclidean-null p95
recomputed on them must equal positive_control_heldout.json), same train-only whitening and column
scaling, same polynomial basis (degree 3 in (a, b)), same statistic.

b0.  Estimated from the RECORDED CLEAN STATE, not from the edge: in every matched capture the
clamped family Zc has, on its patch-A arm, B forced to its clean activation, so Zc[arm == 0, 1] is
B_clean . u (constant to machine precision in all 185 captures).  Only rows of that arm whose
index is in the S-TRAINING set are read.  This involves no readout and no test row.  The edge-based
alternative b0 = m a0 + c (edge_strength.generators' definition, here with (a0, m, c) fitted on
training rows) is reported alongside as a sensitivity column.  a0 is the same training-row value
g_R uses (_fit_edge: mean of a on the patch-B arm, which is A_clean . u exactly).

STATISTIC (per capture, per protocol).  ratio(g) = mean_{test J rows} (phi(z) . g)^2 /
mean_{test S rows} (phi(z) . g)^2, phi the train-scaled degree-3 design.  Unit: capture.

NULLS (per capture, per protocol; 200 draws each, evaluated on the same test rows):
  euclid  g = u / |u| in the train-scaled coefficient basis (identical draws to
          positive_control_heldout._score, so its p95 must reproduce the ledger's)
  inv     g = G_ref^{-1/2} u,  G_ref = (G_S + G_J) / 2 from TRAINING rows (same u draws).
          The distribution of the J/S ratio under this null does not depend on the coefficient
          basis (any invertible reparametrisation of the polynomial space leaves it unchanged),
          unlike the Euclidean null, which depends on the column-scaling choice.

    python3 ofat_baseline.py      # -> ofat_baseline.json
"""
import sys
sys.dont_write_bytecode = True                  # leave e0/__pycache__ untouched
import ast, glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from positive_control_heldout import _fit_edge, _score, N_COEF
from positive_control import g_affine
from levelC2 import design
from subspace_test import DEG, gram, prep
from edge_strength import generators
from prompt_cluster import prompt_of_capture

N_NULL, THR, SEED = 200, 10.0, 0


# ------------------------------------------------------------------ splits ---
def row_split(n_s, n_j, seed=SEED):
    """Index sets of positive_control_heldout.row_protocol (same rng calls, same order)."""
    rng = np.random.default_rng(seed)
    i = rng.permutation(n_s)
    j = rng.permutation(n_j)
    return i[: n_s // 2], i[n_s // 2:], j[: n_j // 2], j[n_j // 2:]


def donor_split(n_s, n_j, seed=SEED):
    """Index sets of positive_control_heldout.donor_protocol (boolean masks -> sorted indices)."""
    half = n_s // 2
    n_don = half // N_COEF
    assert n_don * N_COEF == half, (n_s, N_COEF)
    donor = (np.arange(n_s) % half) // N_COEF
    ids = np.arange(0, n_don, 2)
    ja, jb = np.repeat(ids, len(ids)), np.tile(ids, len(ids))
    assert len(ja) == n_j, (len(ja), n_j)
    train = np.random.default_rng(seed).permutation(n_don)[: n_don // 2]
    s_tr = np.isin(donor, train)
    j_tr = np.isin(ja, train) & np.isin(jb, train)
    j_te = ~np.isin(ja, train) & ~np.isin(jb, train)
    return (np.flatnonzero(s_tr), np.flatnonzero(~s_tr),
            np.flatnonzero(j_tr), np.flatnonzero(j_te))


# ------------------------------------------------------------------- score ---
def score(Zs, Zj, Zc, arm, idx, seed=SEED):
    s_tr, s_te, j_tr, j_te = idx
    # g_R through the ledger's own function on these index sets (used to verify the split)
    ref = _score(Zs[s_tr], arm[s_tr], Zj[j_tr], Zs[s_te], Zj[j_te], seed)

    # every fitted quantity below uses training rows only, exactly as _score does
    a0, m, c = _fit_edge(Zs[s_tr], arm[s_tr])
    trZ = np.vstack([Zs[s_tr], Zj[j_tr]])
    mu, sd = trZ.mean(0), trZ.std(0) + 1e-12
    Wn = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(Wn(trZ), DEG), axis=0); cs[cs == 0] = 1.0
    Str, Jtr = design(Wn(Zs[s_tr]), DEG) / cs, design(Wn(Zj[j_tr]), DEG) / cs
    Ste, Jte = design(Wn(Zs[s_te]), DEG) / cs, design(Wn(Zj[j_te]), DEG) / cs
    q = lambda M, v: float(np.mean((M @ v) ** 2))
    ratio = lambda v: q(Jte, v) / max(q(Ste, v), 1e-300)
    unit = lambda v: v / np.linalg.norm(v)

    a0w = (a0 - mu[0]) / sd[0]
    gR = unit(g_affine(a0w, m * sd[0] / sd[1], (m * mu[0] + c - mu[1]) / sd[1]) * cs)
    clamp_tr = s_tr[arm[s_tr] == 0]                  # clamped patch-A arm, TRAINING rows only
    b_cl = Zc[clamp_tr, 1]
    b0_clean, b0_edge = float(np.mean(b_cl)), float(m * a0 + c)
    gC = unit(g_affine(a0w, 0.0, (b0_clean - mu[1]) / sd[1]) * cs)       # (a - a0)(b - b0)
    gCe = unit(g_affine(a0w, 0.0, (b0_edge - mu[1]) / sd[1]) * cs)

    rng = np.random.default_rng(seed)                # Euclidean null: _score's draws
    euc = []
    for _ in range(N_NULL):
        g = rng.normal(size=len(gR)); g /= np.linalg.norm(g)
        euc.append(ratio(g))
    Gref = 0.5 * (gram(Str) + gram(Jtr))             # invariance-respecting null, TRAIN Grams
    w, V = np.linalg.eigh(Gref)
    assert w.min() > 1e-12 * w.max(), w
    Rinv = (V / np.sqrt(w)) @ V.T                    # symmetric G_ref^{-1/2}
    rng = np.random.default_rng(seed)
    inv = [ratio(Rinv @ rng.normal(size=len(gR))) for _ in range(N_NULL)]
    euc, inv = np.array(euc), np.array(inv)

    out = dict(gR=ratio(gR), gC=ratio(gC), gC_edge_b0=ratio(gCe),
               euclid_p95=float(np.percentile(euc, 95)), inv_p95=float(np.percentile(inv, 95)),
               euclid_frac_gt10=float(np.mean(euc > THR)), inv_frac_gt10=float(np.mean(inv > THR)),
               a0=a0, m=m, c=c, m_whitened=float(m * sd[0] / sd[1]),
               b0_clean=b0_clean, b0_edge=b0_edge,
               b0_gap_whitened=float((b0_edge - b0_clean) / sd[1]),
               b_clamped_ptp=float(np.ptp(b_cl)), cos_gR_gC=float(abs(gR @ gC)),
               Gref_cond=float(w.max() / w.min()),
               sizes=[len(s_tr), len(s_te), len(j_tr), len(j_te)])
    out["gR_over_gC"] = out["gR"] / max(out["gC"], 1e-300)
    assert np.isclose(out["gR"], ref["ratio"], rtol=1e-10, atol=0), (out["gR"], ref["ratio"])
    assert np.isclose(out["euclid_p95"], ref["rnd_p95"], rtol=1e-10, atol=0)
    return out


def old_leaky_gC(Zs, Zj):
    """Reviewer's reconciliation number: edge_strength.generators + subspace_test.prep, i.e. edge,
    whitening and scaling fitted on ALL rows (train and test) and b0 = m a0 + c.  Not a result."""
    _, Ste, _, Jte = prep(Zs, Zj, seed=SEED)
    gR, gC, _ = generators(Zs, Zj)
    q = lambda M, v: float(np.mean((M @ v) ** 2))
    return q(Jte, gC) / max(q(Ste, gC), 1e-300), q(Jte, gR) / max(q(Ste, gR), 1e-300)


# -------------------------------------------------------------------- main ---
def main():
    os.chdir(HERE)                                   # mpat globs / ledger file keys are e0-relative
    src = open(os.path.join("..", "ledger_paperA.py")).read()
    start = src.index("mpat = {"); end = src.index("}", start) + 1
    mpat = ast.literal_eval(src[start + len("mpat = "):end])
    HO = json.load(open("positive_control_heldout.json"))["cells"]
    protos = dict(row=row_split, donor=donor_split)

    cells, old = {}, dict(gC_gt10=0, gR_gt10=0, n=0)
    for pre, pat in mpat.items():
        ledger = {r["file"]: r for r in HO[pre]["rows"]} if pre in HO else {}
        rows = []
        for f in sorted(glob.glob(pat)):
            d = np.load(f)                               # numeric arrays only; no pickle needed
            if "Zjm" not in d.files:
                continue
            Zs, Zj, Zc, arm = d["Zs"], d["Zjm"], d["Zc"], d["arm"]
            half = len(Zs) // 2
            assert np.array_equal(arm, np.r_[np.zeros(half, int), np.ones(len(Zs) - half, int)]), f
            r = dict(file=f)
            for proto, split in protos.items():
                r[proto] = score(Zs, Zj, Zc, arm, split(len(Zs), len(Zj)))
                L = ledger[f][proto]                     # the Table 1 number for this capture
                assert np.isclose(r[proto]["gR"], L["ratio"], rtol=1e-9), (f, proto)
                assert np.isclose(r[proto]["euclid_p95"], L["rnd_p95"], rtol=1e-9), (f, proto)
                assert r[proto]["sizes"] == L["sizes"], (f, proto)
            oc, orr = old_leaky_gC(Zs, Zj)
            old["gC_gt10"] += int(oc > THR); old["gR_gt10"] += int(orr > THR); old["n"] += 1
            rows.append(r)
        if not rows:
            continue
        summ = {}
        for proto in protos:
            col = lambda k: np.array([r[proto][k] for r in rows])
            gR, gC, gCe = col("gR"), col("gC"), col("gC_edge_b0")
            ep, ip = col("euclid_p95"), col("inv_p95")
            summ[proto] = dict(
                n=len(rows),
                gR_gt10=int((gR > THR).sum()), gR_median=float(np.median(gR)),
                gC_gt10=int((gC > THR).sum()), gC_median=float(np.median(gC)),
                gC_edge_b0_gt10=int((gCe > THR).sum()), gC_edge_b0_median=float(np.median(gCe)),
                gR_over_gC_median=float(np.median(gR / gC)),
                gR_gt10_gC_le10=int(((gR > THR) & (gC <= THR)).sum()),
                gC_gt10_gR_le10=int(((gC > THR) & (gR <= THR)).sum()),
                euclid_p95_median=float(np.median(ep)), inv_p95_median=float(np.median(ip)),
                euclid_frac_gt10_median=float(np.median(col("euclid_frac_gt10"))),
                inv_frac_gt10_median=float(np.median(col("inv_frac_gt10"))),
                gR_beats_euclid=int((gR > ep).sum()), gR_beats_inv=int((gR > ip).sum()),
                gC_beats_euclid=int((gC > ep).sum()), gC_beats_inv=int((gC > ip).sum()),
                cos_gR_gC_median=float(np.median(col("cos_gR_gC"))),
                abs_m_whitened_median=float(np.median(np.abs(col("m_whitened")))))
        cells[pre] = dict(pattern=pat, summary=summ, rows=rows)

    # ---- pooled over cells (unit: capture) ----
    from scipy.stats import spearmanr
    pooled = {}
    for proto in protos:
        allr = [r[proto] for v in cells.values() for r in v["rows"]]
        g = lambda k: np.array([x[k] for x in allr])
        gR, gC = g("gR"), g("gC")
        am = np.abs(g("m_whitened")); only_R = (gR > THR) & (gC <= THR)
        rho, p = spearmanr(am, gR / gC)
        pooled[proto] = dict(
            spearman_abs_m_whitened_vs_gR_over_gC=[float(rho), float(p)],
            abs_m_whitened_median_all=float(np.median(am)),
            abs_m_whitened_median_gR_only=float(np.median(am[only_R])) if only_R.any() else None,
            n=len(allr), gR_gt10=int((gR > THR).sum()), gC_gt10=int((gC > THR).sum()),
            gC_edge_b0_gt10=int((g("gC_edge_b0") > THR).sum()),
            gR_gt10_gC_le10=int(((gR > THR) & (gC <= THR)).sum()),
            gC_gt10_gR_le10=int(((gC > THR) & (gR <= THR)).sum()),
            gR_over_gC_median=float(np.median(gR / gC)),
            gR_over_gC_gt1=int((gR > gC).sum()),
            gR_beats_euclid=int((gR > g("euclid_p95")).sum()), gR_beats_inv=int((gR > g("inv_p95")).sum()),
            gC_beats_euclid=int((gC > g("euclid_p95")).sum()), gC_beats_inv=int((gC > g("inv_p95")).sum()),
            euclid_p95_median=float(np.median(g("euclid_p95"))), inv_p95_median=float(np.median(g("inv_p95"))),
            b_clamped_ptp_max=float(g("b_clamped_ptp").max()),
            b0_gap_whitened_median_abs=float(np.median(np.abs(g("b0_gap_whitened")))),
            Gref_cond_max=float(g("Gref_cond").max()))

    # ---- the prompt as the unit, same rule as positive_control_heldout (strict: every capture) ----
    P = prompt_of_capture()
    prompt_level = {}
    for proto in protos:
        grp = defaultdict(lambda: dict(gR=[], gC=[]))
        for pre, v in cells.items():
            for r in v["rows"]:
                k = (pre, P.get(r["file"], r["file"]))
                grp[k]["gR"].append(r[proto]["gR"] > THR); grp[k]["gC"].append(r[proto]["gC"] > THR)
        prompt_level[proto] = dict(prompts=len(grp), gR_strict=int(sum(all(h["gR"]) for h in grp.values())),
                                   gC_strict=int(sum(all(h["gC"]) for h in grp.values())))

    meta = dict(
        b0_source="recorded clean state: Zc[arm==0,1] (B forced to its clean activation on the clamped "
                  "patch-A arm), mean over S-TRAINING rows of that arm; constant to machine precision",
        a0_source="positive_control_heldout._fit_edge on S-training rows (mean a on the patch-B arm)",
        statistic="held-out mean (phi.g)^2 over test joint (Zjm) rows / same over test single-site (Zs) rows",
        unit="capture (one *_matched_*.npz file)", threshold=THR, n_null=N_NULL, seed=SEED,
        nulls=dict(euclid="unit Gaussian in the train-scaled coefficient basis (draws identical to "
                          "positive_control_heldout._score)",
                   inv="G_ref^{-1/2} u, u~N(0,I), G_ref=(G_S+G_J)/2 from TRAINING rows (same u draws)"),
        old_leaky_reconciliation=dict(old, note="edge_strength.generators + subspace_test.prep on (Zs, Zjm): "
                                               "all-rows fit, b0 = m a0 + c; reproduces the reviewer's number"))
    json.dump(dict(meta=meta, pooled=pooled, prompt_level=prompt_level, cells=cells),
              open("ofat_baseline.json", "w"), indent=1, default=float)

    # ---- print ----
    for proto in protos:
        print(f"\n=== {proto} protocol (train-only edge, whitening, scaling; b0 from recorded clean state) ===")
        print(f"{'cell':13s} {'n':>3} {'gR>10':>6} {'gC>10':>6} {'gCe>10':>7} {'med gR':>9} {'med gC':>9} "
              f"{'med gR/gC':>9} {'euc p95':>8} {'inv p95':>8} {'gC>euc':>6} {'gC>inv':>6} {'gR>inv':>6} {'|cos|':>6}")
        for pre, v in cells.items():
            s = v["summary"][proto]
            print(f"{pre:13s} {s['n']:>3} {s['gR_gt10']:>6} {s['gC_gt10']:>6} {s['gC_edge_b0_gt10']:>7} "
                  f"{s['gR_median']:>9.1f} {s['gC_median']:>9.1f} {s['gR_over_gC_median']:>9.3g} "
                  f"{s['euclid_p95_median']:>8.2f} {s['inv_p95_median']:>8.2f} {s['gC_beats_euclid']:>6} "
                  f"{s['gC_beats_inv']:>6} {s['gR_beats_inv']:>6} {s['cos_gR_gC_median']:>6.3f}")
        p = pooled[proto]
        print(f"POOLED  gR>10 {p['gR_gt10']}/{p['n']}  gC>10 {p['gC_gt10']}/{p['n']}  "
              f"gC(b0=m a0+c)>10 {p['gC_edge_b0_gt10']}/{p['n']}  | gR>10 & gC<=10: {p['gR_gt10_gC_le10']}  "
              f"gC>10 & gR<=10: {p['gC_gt10_gR_le10']}  | median gR/gC {p['gR_over_gC_median']:.3g}  "
              f"gR>gC in {p['gR_over_gC_gt1']}/{p['n']}")
        print(f"        null p95 median: euclid {p['euclid_p95_median']:.2f}  inv {p['inv_p95_median']:.2f}  | "
              f"beats euclid p95: gR {p['gR_beats_euclid']} gC {p['gC_beats_euclid']}  "
              f"beats inv p95: gR {p['gR_beats_inv']} gC {p['gC_beats_inv']}")
        print(f"        edge: Spearman(|m_whitened|, gR/gC) = {p['spearman_abs_m_whitened_vs_gR_over_gC'][0]:+.3f} "
              f"(p={p['spearman_abs_m_whitened_vs_gR_over_gC'][1]:.2g});  median |m_whitened| all "
              f"{p['abs_m_whitened_median_all']:.3g}, in the gR>10 & gC<=10 captures {p['abs_m_whitened_median_gR_only']:.3g}")
        pl = prompt_level[proto]
        print(f"        prompt level (every capture >10x): gR {pl['gR_strict']}/{pl['prompts']}  "
              f"gC {pl['gC_strict']}/{pl['prompts']}")
    print(f"\nreconciliation, OLD leaky protocol: gC>10 {old['gC_gt10']}/{old['n']}  gR>10 {old['gR_gt10']}/{old['n']}")
    print(f"b on clamped arm: max ptp {pooled['row']['b_clamped_ptp_max']:.3g};  "
          f"median |b0_edge - b0_clean| (train-whitened sd units) row {pooled['row']['b0_gap_whitened_median_abs']:.3g}"
          f" donor {pooled['donor']['b0_gap_whitened_median_abs']:.3g};  max cond(G_ref) "
          f"{max(pooled['row']['Gref_cond_max'], pooled['donor']['Gref_cond_max']):.3g}")
    print("saved -> ofat_baseline.json")


if __name__ == "__main__":
    main()
