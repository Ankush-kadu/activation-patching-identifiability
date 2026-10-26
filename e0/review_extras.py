"""Quantities introduced by the section-by-section review of the arXiv manuscript (corrections C38-C45).
DESCRIPTIVE; previously inspected data; nothing here is new evidence for a claim of the paper.

  dk_gap          finite_sample.py compares BOTTOM-k eigenspaces (bottom_k) but takes its eigengap at the TOP-k
                  boundary: with ascending eigenvalues w and p=10, k=3 it uses w[p-k]-w[p-k-1] = w[7]-w[6].
                  Here the bottom-k boundary gap w[k]-w[k-1] is computed on the same features (prep_full, gram),
                  relative to ||G||, for both families; medians over the files finite_sample.py uses.
  edge_cos        Spearman correlation of the Euclidean coefficient cosine |cos(g_R, g_C)| with the edge-strength
                  ratio sd(b|arm A)/sd(b|arm B) and with the absolute whitened slope, over the corrected-row
                  captures of edge_strength_matched.json.
  reverse_survivors  support-matched captures whose reverse end escapes, and how many of them have
                  sd(b|arm A)/sd(b|arm B) < 0.10 (matched_invariant.json).
  chart_cells     cells of the cross-seed chart control with both nominal Holm q < 0.05; the own-chart cell with
                  the largest recompute-weak escape fraction; the own-chart ratio screen (recompute_vs_clamped_*.json).
  tl_calibration  the 'tuned lens' of mech_classes.py is a least-squares map fitted on the first 224 single-site
                  rows (the patch-A arm) of the first file; how many of those rows fall in that file's scored
                  (test) half under run_file's seed-0 split.
  taylor_dup      whether controls_subspace.json's Taylor row equals its polynomial row exactly (it is built with a
                  zero centre, i.e. the same design matrix).

    python3 review_extras.py      # -> review_extras.json
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finite_sample import prep_full, gram      # noqa: E402

STATUS = "DESCRIPTIVE: audit quantities for the corrections ledger (review round); previously inspected captures"
K = 3


def dk_gap():
    out = {}
    for pat, lab in (("adj_raw_*.npz", "GPT-2"), ("rec_raw_*.npz", "Qwen3.5-9B")):
        rows = {"S": {"bottom": [], "top": []}, "J": {"bottom": [], "top": []}}
        files = sorted(glob.glob(os.path.join(HERE, pat)))
        for f in files:
            d = np.load(f)
            Ms, Mj = prep_full(d["Zs"], d["Zj"])
            for tag, M in (("S", Ms), ("J", Mj)):
                w = np.linalg.eigvalsh(gram(M)); p = len(w); nG = w[-1]
                rows[tag]["bottom"].append(float((w[K] - w[K - 1]) / nG))
                rows[tag]["top"].append(float((w[p - K] - w[p - K - 1]) / nG))
        out[lab] = {"n": len(files), **{f"{t}_{e}_rel_median": float(np.median(rows[t][e])) for t in rows for e in rows[t]}}
    return out


def edge_cos():
    E = json.load(open(os.path.join(HERE, "edge_strength_matched.json")))["captures"]["corrected_row"]
    cos = np.array([r["cos"] for r in E]); sd = np.array([r["sd_ratio"] for r in E]); m = np.array([r["abs_m_w"] for r in E])
    return dict(n=len(E), rho_cos_sd_ratio=float(spearmanr(sd, cos)[0]), rho_cos_abs_slope=float(spearmanr(m, cos)[0]),
                n_strong_sd_ratio_armB=int((sd >= 0.10).sum()))


def reverse_survivors():
    M = json.load(open(os.path.join(HERE, "matched_invariant.json")))
    surv = [t["file"] for c in M["cells"].values() if c.get("matched") for t in c["matched"]["per_trial"] if t["above"]]
    below = [f for f in surv if M["edges"][f]["sd_ratio"] < 0.10]
    return dict(n_survivors=len(surv), n_below_0p10=len(below), below=below)


def chart_cells():
    X = json.load(open(os.path.join(HERE, "recompute_vs_clamped_edge_x.json")))
    O = json.load(open(os.path.join(HERE, "recompute_vs_clamped_edge.json")))
    both = sorted(k for k, v in X.items() if not k.startswith("_") and v["q_below"] < 0.05 and v["q_above"] < 0.05)
    frac = {k: v["below"] / v["n"] for k, v in O.items() if not k.startswith("_")}
    return dict(xseed_cells_both_sig=[len(both), X["_scope"]["n_cells"]], xseed_cells_both_sig_names=both,
                own_chart_max_fraction_cell=max(frac, key=frac.get), own_chart_fractions=frac,
                own_chart_ratio_screen_dropped=O["_screen"]["n_dropped"], own_chart_sd_ratio_max=O["_screen"]["sd_ratio_max"],
                xseed_ratio_screen_dropped=X["_screen"]["n_dropped"])


def tl_calibration():
    f = sorted(glob.glob(os.path.join(HERE, "actsg_*_*_*.npz")))[0]
    n = len(np.load(f)["A_single"])
    calib = set(range(n // 2))
    i = np.random.default_rng(0).permutation(n)
    test = set(i[n // 2:].tolist())
    return dict(file=os.path.basename(f), n_single=n, calibration_rows=len(calib), calibration_rows_in_test_half=len(calib & test))


def taylor_dup():
    C = json.load(open(os.path.join(HERE, "controls_subspace.json")))
    return {m: bool(C[m]["basis"]["poly"] == C[m]["basis"]["taylor"]) for m in C}


def main():
    out = dict(status=STATUS, dk_gap=dk_gap(), edge_cos=edge_cos(), reverse_survivors=reverse_survivors(),
               chart_cells=chart_cells(), tl_calibration=tl_calibration(), taylor_dup=taylor_dup())
    json.dump(out, open(os.path.join(HERE, "review_extras.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "status"}, indent=1))


if __name__ == "__main__":
    main()
