"""Audit quantities introduced by the corrections ledger (arXiv version).  DESCRIPTIVE; previously inspected data.

Each quantity here is quoted by a correction entry and needs a defining computation in the snapshot:

  donor_duplicate_captures   matched captures in which, under the donor protocol's common partition
                             T = default_rng(0).permutation(32)[:16], some held-out donor's patch-A-arm trajectory
                             (Zs rows d*7..d*7+6, both coordinates) is bit-identical to some training donor's
                             (duplicate donor prompts): the donor holdout holds out donor INDICES, not content.
  fig1b_ratio                ptp(Zs[:,0]) / ptp(Zj[:,0]) for ioi_matched_gpt2_5_10_0.npz (Figure 1(b)'s title);
                             its median over the ioi_matched_gpt2 captures, and over captures with distinct
                             upstream coordinates (captures sharing upstream layer and seed are byte-identical).
  qc_info_plugin             per published query-chart capture, lambda_min(R'R/(n s2)) for J over S, where R is the
                             cross block residualised on the additive block (identifiability's chart: pooled
                             whitening of Zs and Zjm) and s2 = RSS/(n-10) of the full cubic fit; median over captures.
                             Also the ratio of medians of the design Gram lambda_min(R'R/n), and the same design
                             ratio with each family whitened by its own mean and sd (standardisation dependence).
  cross_seed_same_prompt     rows of the cross-seed chart control whose base prompt equals the chart source's.

    python3 audit_extras.py      # -> audit_extras.json
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from identifiability import ADD, CROSS, dm      # noqa: E402

STATUS = "DESCRIPTIVE: audit quantities for the corrections ledger; previously inspected captures"
MATCHED = ["ioi_matched_gpt2_", "ioi_matched_qwen35_", "induction_matched_gpt2_", "induction_matched_qwen35_",
           "capital_matched_gpt2_", "capital_matched_qwen35_", "greater_matched_gpt2_", "agreement_matched_gpt2_",
           "ioi_matched_qwen3_", "induction_matched_qwen3_"]


def matched_files():
    out = []
    for p in MATCHED:
        for f in sorted(glob.glob(os.path.join(HERE, p + "*.npz"))):
            rest = os.path.basename(f)[len(p):-4].split("_")
            if len(rest) == 3 and all(x.isdigit() for x in rest) and "Zjm" in np.load(f).files:
                out.append(f)
    return out


def donor_duplicates(files):
    T = set(np.random.default_rng(0).permutation(32)[:16].tolist())
    hits = []
    for f in files:
        Zs = np.load(f)["Zs"]
        traj = {d: Zs[d * 7:(d + 1) * 7] for d in range(32)}
        dup = any(np.array_equal(traj[h], traj[t]) for h in range(32) if h not in T for t in T)
        hits.append(bool(dup))
    return int(sum(hits)), len(hits)


def fig1b():
    r = lambda d: float(np.ptp(d["Zs"][:, 0]) / max(np.ptp(d["Zj"][:, 0]), 1e-30))
    files = sorted(glob.glob(os.path.join(HERE, "ioi_matched_gpt2_*.npz")))
    vals, seen, distinct = {}, {}, []
    for f in files:
        d = np.load(f)
        vals[os.path.basename(f)] = r(d)
        key = (d["Zs"][:224, 0].tobytes(), d["Zj"][:, 0].tobytes())
        if key not in seen:
            seen[key] = True; distinct.append(vals[os.path.basename(f)])
    return dict(single=vals["ioi_matched_gpt2_5_10_0.npz"], median=float(np.median(list(vals.values()))),
                n=len(vals), median_distinct=float(np.median(distinct)), n_distinct=len(distinct))


def cross_resid(Z, W):
    Ma, Mc = dm(W(Z), ADD), dm(W(Z), CROSS)
    coef, *_ = np.linalg.lstsq(Ma, Mc, rcond=None)
    return Mc - Ma @ coef


def s2_full(Z, Y, W):
    X = np.hstack([dm(W(Z), ADD), dm(W(Z), CROSS)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ beta
    return float(e @ e) / (len(Y) - X.shape[1])


def chart(Z):
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    return lambda Q: (Q - mu) / sd


def qc_info():
    design, plugin, own, lS, lJ = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(HERE, "qchart_*.npz"))):
        d = np.load(f); Zs, Ys, Zj, Yj = d["Zs"], d["Ys"], d["Zjm"], d["Yjm"]
        W = chart(np.vstack([Zs, Zj]))
        RS, RJ = cross_resid(Zs, W), cross_resid(Zj, W)
        eS = np.linalg.eigvalsh(RS.T @ RS / len(Zs)).min(); eJ = np.linalg.eigvalsh(RJ.T @ RJ / len(Zj)).min()
        design.append(eJ / eS); lS.append(eS); lJ.append(eJ)
        plugin.append((eJ / s2_full(Zj, Yj, W)) / (eS / s2_full(Zs, Ys, W)))
        WS, WJ = chart(Zs), chart(Zj)
        RS2, RJ2 = cross_resid(Zs, WS), cross_resid(Zj, WJ)
        own.append(np.linalg.eigvalsh(RJ2.T @ RJ2 / len(Zj)).min() / np.linalg.eigvalsh(RS2.T @ RS2 / len(Zs)).min())
    return dict(n=len(design), design_median_of_ratios=float(np.median(design)),
                design_ratio_of_medians=float(np.median(lJ) / np.median(lS)),
                plugin_median_of_ratios=float(np.median(plugin)),
                design_own_family_whitening_median=float(np.median(own)))


def cross_seed_same_prompt():
    """Rows of the cross-seed chart control (recompute_vs_clamped_edge_x.json) whose base prompt equals that of the
    capture supplying the chart (the first seed of the same layer pair, matched.py), from matched_*.json metadata."""
    prompt = {}
    for f in glob.glob(os.path.join(HERE, "matched_*.json")):
        for r in json.load(open(f)).get("rows", []):
            if "file" in r and "prompt" in r:
                prompt[r["file"]] = r["prompt"]
    X = json.load(open(os.path.join(HERE, "recompute_vs_clamped_edge_x.json")))
    same = total = 0
    for cell, v in X.items():
        if cell.startswith("_"):
            continue
        for r in v["rows"]:
            f = r["file"]; stem, seed = f[:-4].rsplit("_", 1)
            seeds = sorted(int(g[len(stem) + 1:-4]) for g in (os.path.basename(x) for x in glob.glob(os.path.join(HERE, stem + "_*.npz")))
                           if g[len(stem) + 1:-4].isdigit())
            src = f"{stem}_{seeds[0]}.npz"
            assert src != f and "pA" in np.load(os.path.join(HERE, src)).files, (f, src)
            total += 1; same += int(prompt[f] == prompt[src])
    return [same, total]


def main():
    files = matched_files()
    k, n = donor_duplicates(files)
    out = dict(status=STATUS, n_matched=len(files), donor_duplicate_captures=[k, n], fig1b=fig1b(), qc_info=qc_info(),
               cross_seed_same_prompt=cross_seed_same_prompt())
    json.dump(out, open(os.path.join(HERE, "audit_extras.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "status"}, indent=1))


if __name__ == "__main__":
    main()
