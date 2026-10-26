"""C4 RANDOMIZED-FAMILY NULL for the non-IOI tasks.

A separation number means nothing without the null.  The single cleanest control from
`controls_subspace.py` is C4: split the SINGLE-site family in half at random and run the
IDENTICAL generalized-eigenproblem procedure.  If the procedure manufactures separations
from finite-sample covariance noise, Sep(S1,S2) will be as large as Sep(S,J).

Reused verbatim from controls_subspace -- no new code path, no new parameters.
Also reports the C2 bootstrap CI so the non-IOI claim is stated as an interval.
"""
import glob, json
import numpy as np
from controls_subspace import recip, c4_random_family_null, c2_bootstrap, BASES

from cells import CELLS                          # 2026-08-27: one cell per model x task


def load(pattern):
    return [(f, np.load(f)) for f in sorted(glob.glob(pattern))]


if __name__ == "__main__":
    out = {}
    for c in [c for c in CELLS if c["nonioi"] is not None]:
        task = c["label"]
        data = load(c["pattern"])
        if not data:
            continue
        print("=" * 96)
        print(f"{task:>20}   n_trials={len(data)}")
        print("=" * 96)

        same, cross = c4_random_family_null(data)
        ratio = np.median(cross) / max(np.median(same), 1e-12)
        clean = cross.min() > np.percentile(same, 95)
        print(f"  C4 same-family  S1 vs S2 : median {np.median(same):>12.2f}  "
              f"p95 {np.percentile(same,95):>12.2f}  max {same.max():>12.2f}  (n={len(same)})")
        print(f"  C4 CROSS-family S  vs J  : median {np.median(cross):>12.4g}  "
              f"min {cross.min():>12.4g}                        (n={len(cross)})")
        print(f"  cross/same median ratio  : {ratio:.4g}")
        print(f"  min(cross) > p95(same)   : {clean}   <-- the control that matters")

        bt = c2_bootstrap(data)
        for k, (lo, md, hi) in bt.items():
            print(f"  bootstrap {k:>8}: median {md:>12.4g}   95% CI [{lo:.4g}, {hi:.4g}]")

        print(f"  basis robustness (both>10x):", end=" ")
        bas = {}
        for bn in ("poly", "rbf", "fourier"):
            v = np.array([recip(d["Zs"], d["Zj"], bn) for _, d in data])
            v = v[np.isfinite(v[:, 0])]
            bas[bn] = int(np.sum((v[:, 0] > 10) & (v[:, 1] > 10)))
            print(f"{bn} {bas[bn]}/{len(v)}", end="   ")
        print("\n")

        out[task] = dict(n=len(data), same_median=float(np.median(same)),
                         same_p95=float(np.percentile(same, 95)),
                         same_max=float(same.max()),
                         cross_median=float(np.median(cross)),
                         cross_min=float(cross.min()),
                         ratio=float(ratio), clean_separation=bool(clean),
                         bootstrap={k: list(v) for k, v in bt.items()}, basis=bas)
    json.dump(out, open("nonioi_controls.json", "w"), indent=1)
    print("saved -> nonioi_controls.json")
