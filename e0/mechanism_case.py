"""THE "SO WHAT?" EXPERIMENT — does the blind subspace change a MECHANISTIC conclusion?

M2 must not be an arbitrary vector in feature space. It must be an interpretable
mechanistic claim a human would actually make. The cleanest such pair:

  M1  ADDITIVE / SEPARABLE : "components A and B contribute independently
                             to the logit difference"
                             span {1, a, b, a^2, b^2, a^3, b^3}   -- NO cross terms

  M2  INTERACTIVE          : "B's contribution DEPENDS ON the state of A"
                             full degree-3 span, adds {ab, a^2 b, a b^2}

M1 is exactly the assumption behind every per-component activation-patching
number ("head X contributes +0.5 to the logit difference").

PREDICTION FROM THE THEORY
On the patch-A arm b is pinned at b0, so ab -> a*b0, which is LINEAR in a and
therefore confounded with the additive term. Same on the patch-B arm. So the
single-site family cannot determine whether the two components interact. The
joint family moves a and b independently and identifies it.

If confirmed: conventional single-site patching evidence cannot license the
additivity assumption it routinely relies on.
"""
import json, glob, numpy as np

ADD = [(0,0),(1,0),(0,1),(2,0),(0,2),(3,0),(0,3)]                  # no cross terms
FULL = [(i, t-i) for t in range(4) for i in range(t+1)]            # all degree<=3
CROSS = [e for e in FULL if e not in ADD]                          # ab, a^2b, ab^2


def dm(Z, exps):
    return np.array([[a**i * b**j for (i, j) in exps] for a, b in Z], float)


def fit(M, y):
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    return c


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def case(Zs, Ys, Zj, Yj, seed=0):
    rng = np.random.default_rng(seed)
    allZ = np.vstack([Zs, Zj]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    i = rng.permutation(len(Zs)); tr, te = i[:len(i)//2], i[len(i)//2:]

    Sa_tr, Sa_te = dm(W(Zs[tr]), ADD), dm(W(Zs[te]), ADD)
    Sf_tr, Sf_te = dm(W(Zs[tr]), FULL), dm(W(Zs[te]), FULL)
    Ja, Jf = dm(W(Zj), ADD), dm(W(Zj), FULL)

    cA = fit(Sa_tr, Ys[tr])          # M1 additive, fitted on single-site TRAIN
    cI = fit(Sf_tr, Ys[tr])          # M2 interactive, fitted on single-site TRAIN

    # --- do they fit the CONVENTIONAL evidence equally well? (held-out) ---
    eA_s, eI_s = rmse(Sa_te @ cA, Ys[te]), rmse(Sf_te @ cI, Ys[te])

    # --- do they make DIFFERENT predictions under the diagnostic family? ---
    pA_j, pI_j = Ja @ cA, Jf @ cI
    gap = rmse(pA_j, pI_j)

    # --- which one does the MODEL actually agree with? ---
    eA_j, eI_j = rmse(pA_j, Yj), rmse(pI_j, Yj)

    # --- how much interaction mass did the single-site fit even see? ---
    cross_mass = float(np.linalg.norm(cI[[FULL.index(e) for e in CROSS]]))
    # ...vs the interaction actually present, measured on the JOINT family
    cJ = fit(Jf, Yj)
    cross_mass_true = float(np.linalg.norm(cJ[[FULL.index(e) for e in CROSS]]))

    return dict(
        single_rmse_additive=eA_s, single_rmse_interactive=eI_s,
        single_rmse_ratio=eI_s / max(eA_s, 1e-12),
        joint_pred_gap=gap, joint_rmse_additive=eA_j, joint_rmse_interactive=eI_j,
        std_Yj=float(np.std(Yj)), std_Ys=float(np.std(Ys)),
        cross_mass_from_single=cross_mass, cross_mass_from_joint=cross_mass_true,
        # verdict: indistinguishable on conventional evidence AND different on joint
        indistinguishable=bool(abs(eI_s - eA_s) / max(eA_s, 1e-12) < 0.10),
        diverges=bool(gap > 0.5 * np.std(Yj)))


def run(pattern, label):
    print("=" * 108)
    print(f"{label}   M1 = ADDITIVE (independent contributions)   M2 = INTERACTIVE (B depends on A)")
    print("=" * 108)
    print(f"{'trial':>12} | {'CONVENTIONAL EVIDENCE (held-out single-site)':^40} | "
          f"{'DIAGNOSTIC (joint)':^34}")
    print(f"{'':>12} | {'RMSE M1':>11} {'RMSE M2':>11} {'ratio':>7} {'same?':>7} | "
          f"{'M1-M2 gap':>10} {'M1 err':>9} {'M2 err':>9}  which survives")
    print("-" * 108)
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        if "Ys" not in d:
            continue
        r = case(d["Zs"], d["Ys"], d["Zj"], d["Yj"])
        rows.append(r)
        win = ("INTERACTIVE" if r["joint_rmse_interactive"] < 0.8 * r["joint_rmse_additive"]
               else ("ADDITIVE" if r["joint_rmse_additive"] < 0.8 * r["joint_rmse_interactive"]
                     else "tie"))
        t = f.split("raw_")[1].replace(".npz", "")
        print(f"{t:>12} | {r['single_rmse_additive']:>11.4f} {r['single_rmse_interactive']:>11.4f} "
              f"{r['single_rmse_ratio']:>7.3f} {'YES' if r['indistinguishable'] else 'no':>7} | "
              f"{r['joint_pred_gap']:>10.3f} {r['joint_rmse_additive']:>9.3f} "
              f"{r['joint_rmse_interactive']:>9.3f}  {win}")
    if not rows:
        return None
    ind = sum(r["indistinguishable"] for r in rows)
    div = sum(r["diverges"] for r in rows)
    both = sum(r["indistinguishable"] and r["diverges"] for r in rows)
    print(f"\n  indistinguishable on conventional evidence (<10% RMSE diff): {ind}/{len(rows)}")
    print(f"  predictions diverge under diagnostic (> 0.5 sd)            : {div}/{len(rows)}")
    print(f"  ** BOTH (ambiguous then resolved)                          : {both}/{len(rows)} **")
    cm_s = np.median([r["cross_mass_from_single"] for r in rows])
    cm_j = np.median([r["cross_mass_from_joint"] for r in rows])
    print(f"\n  interaction-coefficient magnitude recovered from SINGLE-site evidence: {cm_s:.4f}")
    print(f"  interaction-coefficient magnitude present per JOINT evidence         : {cm_j:.4f}")
    print(f"  -> single-site evidence recovers {100*cm_s/max(cm_j,1e-12):.0f}% of the true interaction mass")
    return dict(n=len(rows), indistinguishable=ind, diverges=div, both=both,
                cross_from_single=float(cm_s), cross_from_joint=float(cm_j))


if __name__ == "__main__":
    out = {}
    out["gpt2"] = run("adj_raw_*.npz", "GPT-2")
    print()
    out["qwen"] = run("rec_raw_*.npz", "Qwen3.5-9B (ACTIVE pairs)")
    json.dump(out, open("mechanism_case.json", "w"), indent=1)
    print("\n" + "=" * 108)
    print("If a case is BOTH indistinguishable on conventional evidence AND divergent under")
    print("the diagnostic family, then a standard activation-patching experiment could not")
    print("have licensed the additivity assumption it implicitly relies on.")
