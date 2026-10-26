"""edge_strength.py, re-run on the 185 support-matched captures with the corrected pipeline
(exploratory, added 2026-09-24; review item).

edge_strength.py produced the appendix's "realised edge, measured" paragraph (EdgeN 82, EdgeNStrong 37,
EdgeCosRho -0.841, EdgeWeakCos 1, EdgeStrongCos 0.992, EdgeStrongSepGC 56.1, EdgeStrongRatio 1.54) and
the limitations-appendix bullet (EdgeBlindRho -0.936, EdgeBlindWeak 4972, EdgeBlindStrong 130,
EdgeStrongGtTen 34/37).  It globs the UNMATCHED *_raw_*.npz captures (8 cells, no Qwen3-8B), scores the
raw alpha=1 joint family, and its generators() fits a0, m, c, whitening and column scaling on ALL rows
(train and test), with b0 = m a0 + c.  The Reproducibility statement says that pipeline was replaced.

This script computes every one of those summaries four ways, so each legacy number has a counterpart:
  legacy_raw        edge_strength.analyse on the raw captures (recomputed; must equal edge_strength.json)
  legacy_matched    edge_strength.analyse on the 185 matched captures (Zs, Zjm): isolates the DATA change
  corrected_row     185 matched captures, Table 1's row split, edge/whitening/scaling on TRAINING rows only
  corrected_donor   same, common donor partition; joint rows scored only if BOTH donors are held out
In the corrected variants g_R is the Table 1 generator (ratio asserted equal to
positive_control_heldout.json), g_C is Table 1's clamped contrast with b0 from the recorded clean state
(ofat_baseline.score), and g_C with b0 = m a0 + c (edge_strength's definition, train-fitted) is kept as a
sensitivity column.  "Edge strength" is edge_strength's sd(b | patch-A arm) / sd(b | patch-B arm),
computed on the protocol's TRAINING single-site rows; "strong" is its fixed threshold >= 0.10.
Random-direction null: Euclidean p95 (ofat_baseline / positive_control_heldout draws, 200) and the
basis-invariant G_ref^{-1/2} null p95.

Spearman CIs resample the 38 (cell, prompt) groups (2000 draws); capture-level p-values are descriptive.

    python3 edge_strength_matched.py     # -> edge_strength_matched.json   (CPU, ~2 min)
"""
import sys
sys.dont_write_bytecode = True                  # leave e0/__pycache__ untouched
import glob, json, os, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ofat_baseline import row_split, donor_split, score, THR          # noqa: E402
from positive_control_heldout import _fit_edge                         # noqa: E402
from positive_control import g_affine                                  # noqa: E402
from levelC2 import design                                             # noqa: E402
from subspace_test import DEG                                          # noqa: E402
from edge_strength import analyse as legacy_analyse                    # noqa: E402
from cells import CELLS                                                # noqa: E402
from prompt_cluster import prompt_of_capture                           # noqa: E402
from ratio_drivers import mpat, edge_diag, cluster_boot_spearman      # noqa: E402

STRONG = 0.10          # edge_strength.py's threshold on sd(b|arm A)/sd(b|arm B)


def cosines(Zs, Zj, Zc, arm, idx):
    """g_R, g_C (clean b0), g_C (edge b0) unit vectors exactly as ofat_baseline.score builds them (train-only
    fit), returned as held-out ratios and pairwise |cos|; ratios are asserted against score() by the caller."""
    s_tr, s_te, j_tr, j_te = idx
    a0, m, c = _fit_edge(Zs[s_tr], arm[s_tr])
    trZ = np.vstack([Zs[s_tr], Zj[j_tr]])
    mu, sd = trZ.mean(0), trZ.std(0) + 1e-12
    Wn = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(Wn(trZ), DEG), axis=0); cs[cs == 0] = 1.0
    Ste, Jte = design(Wn(Zs[s_te]), DEG) / cs, design(Wn(Zj[j_te]), DEG) / cs
    q = lambda M, v: float(np.mean((M @ v) ** 2))
    ratio = lambda v: q(Jte, v) / max(q(Ste, v), 1e-300)
    unit = lambda v: v / np.linalg.norm(v)
    a0w = (a0 - mu[0]) / sd[0]
    gR = unit(g_affine(a0w, m * sd[0] / sd[1], (m * mu[0] + c - mu[1]) / sd[1]) * cs)
    b0c = float(np.mean(Zc[s_tr[arm[s_tr] == 0], 1]))
    gC = unit(g_affine(a0w, 0.0, (b0c - mu[1]) / sd[1]) * cs)
    gCe = unit(g_affine(a0w, 0.0, (m * a0 + c - mu[1]) / sd[1]) * cs)
    return dict(gR=ratio(gR), gC=ratio(gC), gCe=ratio(gCe),
                cos_gR_gC=float(abs(gR @ gC)), cos_gR_gCe=float(abs(gR @ gCe)))


def summary(rows, gc_key="sep_gC", cos_key="cos", p95_key="rnd_p95", with_ci=True):
    """edge_strength.py's _summary keys, plus the numbers the appendix prose leans on."""
    sd_r = np.array([r["sd_ratio"] for r in rows]); sepR = np.array([r["sep_gR"] for r in rows])
    sepC = np.array([r[gc_key] for r in rows]); cos = np.array([r[cos_key] for r in rows])
    p95 = np.array([r[p95_key] for r in rows]); rat = sepR / np.maximum(sepC, 1e-300)
    strong = sd_r >= STRONG; weak = ~strong
    ok = np.isfinite(sd_r) & np.isfinite(rat) & (rat > 0)
    med = lambda v, m: float(np.median(v[m])) if m.any() else None
    rb, pb = spearmanr(sd_r, sepR); rs, ps = spearmanr(sd_r[ok], rat[ok]); rc, pc = spearmanr(sd_r[ok], cos[ok])
    out = dict(n=len(rows), n_strong=int(strong.sum()),
               rho_blind=float(rb), p_blind=float(pb),
               median_sepR_strong=med(sepR, strong), median_sepR_weak=med(sepR, weak),
               strong_gt10=int((sepR[strong] > THR).sum()),
               spearman_sep_rho=float(rs), spearman_sep_p=float(ps),
               spearman_cos_rho=float(rc), spearman_cos_p=float(pc),
               weak_median_cos=med(cos, weak), strong_median_cos=med(cos, strong),
               weak_median_gR_over_gC=med(rat, weak), strong_median_gR_over_gC=med(rat, strong),
               strong_median_sep_gC=med(sepC, strong),
               # added: what the appendix prose needs to be checked against
               weak_median_sep_gC=med(sepC, weak),
               strong_median_rnd_p95=med(p95, strong), weak_median_rnd_p95=med(p95, weak),
               strong_gC_gt10=int((sepC[strong] > THR).sum()), weak_gt10=int((sepR[weak] > THR).sum()),
               strong_gR_gt_gC=int((rat[strong] > 1).sum()),
               strong_gR_beats_p95=int((sepR[strong] > p95[strong]).sum()),
               strong_gC_beats_p95=int((sepC[strong] > p95[strong]).sum()),
               median_sd_ratio=float(np.median(sd_r)),
               all_median_gR_over_gC=float(np.median(rat)), all_median_cos=float(np.median(cos)))
    if "abs_m_w" in rows[0]:
        am = np.array([r["abs_m_w"] for r in rows])
        out["spearman_absm_cos_rho"] = float(spearmanr(am, cos)[0])
        out["spearman_absm_sepR_rho"] = float(spearmanr(am, sepR)[0])
    if with_ci and "group" in rows[0]:
        grp = [r["group"] for r in rows]
        out["ci"] = dict(rho_blind=cluster_boot_spearman(sd_r, sepR, grp),
                         spearman_sep=cluster_boot_spearman(sd_r[ok], rat[ok], [g for g, k in zip(grp, ok) if k]),
                         spearman_cos=cluster_boot_spearman(sd_r[ok], cos[ok], [g for g, k in zip(grp, ok) if k]))
        if "abs_m_w" in rows[0]:
            out["ci"]["absm_cos"] = cluster_boot_spearman(am, cos, grp)
    return out


def main():
    t0 = time.time()
    os.chdir(HERE)
    HO = json.load(open("positive_control_heldout.json"))["cells"]
    P = prompt_of_capture()

    # ---- legacy_raw: recompute and check against the artifact the ledger reads ----
    raw = []
    for cell in CELLS:
        for f in sorted(glob.glob(cell["pattern"])):
            d = np.load(f)
            raw.append(dict(legacy_analyse(d["Zs"], d["Zj"]), file=f, cell=cell["label"]))
    for r in raw:
        r["cos"] = r["cos_gR_gC"]
    leg_raw = summary(raw, with_ci=False)
    E = json.load(open("edge_strength.json"))["_summary"]
    for k, v in E.items():
        if v is not None:
            assert np.isclose(leg_raw[k], v, rtol=1e-9), (k, leg_raw[k], v)

    # ---- matched captures: legacy pipeline and the two corrected protocols ----
    lm, cr, cd = [], [], []
    for pre, pat in mpat().items():
        ledger = {r["file"]: r for r in HO[pre]["rows"]}
        for f in sorted(glob.glob(pat)):
            d = np.load(f)
            if "Zjm" not in d.files:
                continue
            Zs, Zj, Zc, arm = d["Zs"], d["Zjm"], d["Zc"], d["arm"]
            grp = f"{pre}|{P[f]}"
            L = legacy_analyse(Zs, Zj)
            lm.append(dict(L, cos=L["cos_gR_gC"], file=f, cell=pre, group=grp))
            for proto, split, dest in (("row", row_split, cr), ("donor", donor_split, cd)):
                idx = split(len(Zs), len(Zj))
                sc = score(Zs, Zj, Zc, arm, idx)
                cz = cosines(Zs, Zj, Zc, arm, idx)
                assert np.isclose(sc["gR"], ledger[f][proto]["ratio"], rtol=1e-9), (f, proto)
                for a, b in (("gR", "gR"), ("gC", "gC"), ("gCe", "gC_edge_b0"), ("cos_gR_gC", "cos_gR_gC")):
                    assert np.isclose(cz[a], sc[b], rtol=1e-9), (f, proto, a)
                ed = edge_diag(Zs, Zj, arm, idx[0], idx[2])
                dest.append(dict(file=f, cell=pre, group=grp, sd_ratio=ed["sd_ratio_armB"], edge_r2=ed["edge_r2"],
                                 abs_m_w=abs(sc["m_whitened"]), sep_gR=sc["gR"], sep_gC=sc["gC"],
                                 sep_gC_edge_b0=sc["gC_edge_b0"], cos=cz["cos_gR_gC"],
                                 cos_edge_b0=cz["cos_gR_gCe"], rnd_p95=sc["euclid_p95"], inv_p95=sc["inv_p95"]))
    assert len(lm) == len(cr) == len(cd) == 185

    out = dict(
        legacy_raw=dict(summary=leg_raw, note="edge_strength.analyse on CELLS raw captures; equals edge_strength.json _summary"),
        legacy_matched=dict(summary=summary(lm),
                            note="legacy pipeline (all-rows fit, b0 = m a0 + c, 400 Euclidean draws) on the 185 matched (Zs, Zjm)"))
    for name, rows in (("corrected_row", cr), ("corrected_donor", cd)):
        out[name] = dict(
            summary=summary(rows),                                                        # PRIMARY: g_C, clean b0
            summary_gC_edge_b0=summary(rows, gc_key="sep_gC_edge_b0", cos_key="cos_edge_b0", with_ci=False),
            summary_inv_null=summary(rows, p95_key="inv_p95", with_ci=False))
    # the strong/weak split under the corrected definition, per cell
    out["corrected_row"]["strong_by_cell"] = {pre: f"{sum(r['sd_ratio'] >= STRONG for r in cr if r['cell'] == pre)}/"
                                                   f"{sum(r['cell'] == pre for r in cr)}" for pre in mpat()}
    # ---- The appendix table tab:predict-unmatched (Pred*Ratio / Pred*Count) is ALSO legacy: the ledger's
    # d_predict runs positive_control.part_b (all-rows fit) on the raw captures.  Counterpart: the same raw
    # captures and raw joint family, scored with positive_control_heldout.row_protocol (training-only fit).
    from positive_control import part_b
    from positive_control_heldout import row_protocol
    LG = json.load(open(os.path.join(HERE, "..", "claims_paperA.json")))
    pu, tot = {}, dict(legacy=[0, 0], corrected=[0, 0])
    for cell in CELLS:
        pre = "Pred" + ("GPTIOI" if cell["pre"] == "GPT" else "QwenIOI" if cell["pre"] == "Qwen" else cell["pre"])
        old, new = [], []
        for f in sorted(glob.glob(cell["pattern"])):
            d = np.load(f)
            old.append(part_b(d["Zs"], d["Zj"])["sep_pred"]); new.append(row_protocol(d["Zs"], d["Zj"])["ratio"])
        old, new = np.array(old), np.array(new)
        assert np.isclose(np.median(old), LG[f"predict.{pre}"]["value"], rtol=1e-9), pre
        assert f"{int((old > THR).sum())}/{len(old)}" == LG[f"predict.{pre}.n"]["value"], pre
        pu[pre] = dict(n=len(old), legacy_median=float(np.median(old)), legacy_gt10=int((old > THR).sum()),
                       corrected_median=float(np.median(new)), corrected_gt10=int((new > THR).sum()))
        tot["legacy"][0] += int((old > THR).sum()); tot["corrected"][0] += int((new > THR).sum())
        tot["legacy"][1] += len(old); tot["corrected"][1] += len(new)
    assert f"{tot['legacy'][0]}/{tot['legacy'][1]}" == LG["predict.total"]["value"]
    out["predict_unmatched"] = dict(cells=pu, total=tot,
                                    note="legacy = positive_control.part_b on raw captures (reproduces ledger predict.*); "
                                         "corrected = positive_control_heldout.row_protocol on the same (Zs, Zj)")

    out["meta"] = dict(
        strong_definition=f"sd(b | patch-A arm)/sd(b | patch-B arm) >= {STRONG}; corrected variants compute it on the "
                          "protocol's training single-site rows, legacy variants on all rows",
        gC_primary="(a - a0)(b - b0), b0 = recorded clean B (Zc patch-A arm, training rows) = Table 1 clamped column",
        gC_edge_b0="(a - a0)(b - m a0 - c), train-fitted: edge_strength.py's definition",
        null="rnd_p95: Euclidean random same-class directions (200 draws corrected, 400 legacy); summary_inv_null uses the "
             "basis-invariant G_ref^{-1/2} null",
        ci="2000 resamples of the 38 (cell, prompt) groups; percentile 95%",
        runtime_s=round(time.time() - t0, 1))
    out["captures"] = dict(legacy_matched=lm, corrected_row=cr, corrected_donor=cd)
    json.dump(out, open("edge_strength_matched.json", "w"), indent=1, default=float)

    keys = ("n", "n_strong", "rho_blind", "median_sepR_weak", "median_sepR_strong", "strong_gt10",
            "spearman_cos_rho", "weak_median_cos", "strong_median_cos", "strong_median_sep_gC", "weak_median_sep_gC",
            "strong_median_gR_over_gC", "weak_median_gR_over_gC", "spearman_sep_rho", "strong_median_rnd_p95",
            "strong_gR_gt_gC", "strong_gC_gt10", "strong_gR_beats_p95", "strong_gC_beats_p95", "median_sd_ratio",
            "spearman_absm_cos_rho")
    cols = [("legacy_raw", out["legacy_raw"]["summary"]), ("legacy_matched", out["legacy_matched"]["summary"]),
            ("corr_row", out["corrected_row"]["summary"]), ("corr_row_edgeb0", out["corrected_row"]["summary_gC_edge_b0"]),
            ("corr_donor", out["corrected_donor"]["summary"])]
    print(f"{'key':26s}" + "".join(f"{c:>16s}" for c, _ in cols))
    for k in keys:
        cells = []
        for _, s in cols:
            v = s.get(k)
            cells.append(f"{v:>16.4g}" if isinstance(v, float) else f"{str(v):>16s}")
        print(f"{k:26s}" + "".join(cells))
    for name in ("legacy_matched", "corrected_row", "corrected_donor"):
        ci = out[name]["summary"]["ci"]
        print(name, {k: [round(v["rho"], 3)] + [round(x, 3) for x in v["ci95"]] for k, v in ci.items()})
    print("strong by cell (row):", out["corrected_row"]["strong_by_cell"])
    print("tab:predict-unmatched  legacy -> corrected (median, >10x):")
    for pre, v in pu.items():
        print(f"  {pre:12s} {v['legacy_median']:>9.1f} {v['legacy_gt10']:>2}/{v['n']:<2} -> "
              f"{v['corrected_median']:>9.1f} {v['corrected_gt10']:>2}/{v['n']:<2}")
    print("  total", tot)
    print(f"legacy_raw reproduces edge_strength.json; runtime {out['meta']['runtime_s']} s; saved -> edge_strength_matched.json")


if __name__ == "__main__":
    main()
