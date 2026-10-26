"""STRESS TEST: is the theory-named direction held out at the DONOR level, or only by row?

Section 6 of the paper shows that a chart chosen by optimising a statistic on the states it will
be scored on is a selection effect, and that this needs no readout to occur.  The same objection
must be put to the paper's own headline: the S-blind direction is fitted on a TRAIN half and
scored on a TEST half, but `subspace_test.prep` splits by ROW, and every donor contributes N_AL
interpolation coefficients.  The same donor therefore appears on both sides of the split, so
"held out" controls for noise but not for the direction being specific to that donor set.

This script re-runs the escape analysis with the split taken at the level of DONORS instead:

    single-site   rows are donor-major within each arm, so donor = (row within arm) // N_AL
    joint         pairs = [(i, j) for i in range(0, n, STEP) for j in range(0, n, STEP)], so the
                  A-donor of row r is r // 16

Train donors and test donors are disjoint.  If the escape counts survive this, the direction
generalises to interventions built from donors it never saw; if they collapse, the headline is
the Section 6 failure mode wearing a different hat and the paper must say so.
"""
import glob
import json

import numpy as np
from scipy.linalg import eigh

from cells import N_AL
from levelC2 import design
from subspace_test import DEG, gram, q

N_JA = 16          # A-donor blocks in the matched joint grid (32 donors, stride 2)


def donor_ids(n_rows, kind):
    """Donor label for every row of a capture."""
    if kind == "S":
        per_arm = n_rows // 2
        within = np.concatenate([np.arange(per_arm), np.arange(per_arm)])
        arm = np.concatenate([np.zeros(per_arm, int), np.ones(per_arm, int)])
        # a donor is identified by (arm, donor index within arm)
        return arm * 10_000 + within // N_AL
    return np.arange(n_rows) // N_JA


def split_by_donor(M, ids, rng):
    uniq = rng.permutation(np.unique(ids))
    tr = set(uniq[: len(uniq) // 2].tolist())
    m = np.array([i in tr for i in ids])
    return M[m], M[~m]


def analyse_donor(Zs, Zj, seed=0, delta=1e-10):
    rng = np.random.default_rng(seed)
    allZ = np.vstack([Zs, Zj])
    mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(W(allZ), DEG), axis=0)
    cs[cs == 0] = 1.0
    Ms, Mj = design(W(Zs), DEG) / cs, design(W(Zj), DEG) / cs
    Str, Ste = split_by_donor(Ms, donor_ids(len(Ms), "S"), rng)
    Jtr, Jte = split_by_donor(Mj, donor_ids(len(Mj), "J"), rng)
    if min(len(Str), len(Ste), len(Jtr), len(Jte)) < 20:
        return None
    Gs, Gj = gram(Str), gram(Jtr)
    I = np.eye(Gs.shape[0])
    lam, V = eigh(Gs + delta * I, Gj + delta * I)
    hS = V[:, 0] / np.linalg.norm(V[:, 0])
    hJ = V[:, -1] / np.linalg.norm(V[:, -1])
    return dict(sep_Sblind=q(Jte, hS) / max(q(Ste, hS), 1e-300),
                sep_Jblind=q(Ste, hJ) / max(q(Jte, hJ), 1e-300))


def quarter(M, ids, rng):
    """Four DISJOINT donor groups: (A_train, A_test, B_train, B_test).

    A first version of this drew two INDEPENDENT halves, so the two 'families' shared ~38% of
    their training rows, the null separation sat at ~0.9, and every capture escaped it. That is
    not a null, it is a tautology. These four groups are disjoint by construction.
    """
    uniq = rng.permutation(np.unique(ids))
    k = len(uniq) // 4
    if k == 0:
        return None
    g = [set(uniq[i * k:(i + 1) * k].tolist()) for i in range(4)]
    return [M[np.array([i in s for i in ids])] for s in g]


def envelope(Ms, reps=200, seed=0):
    """Same-family null: two DISJOINT donor halves of the single-site family stand in for the
    two families, with the direction named and scored exactly as in the real statistic."""
    rng = np.random.default_rng(seed + 991)
    ids = donor_ids(len(Ms), "S")
    out = []
    for _ in range(reps):
        parts = quarter(Ms, ids, rng)
        if parts is None:
            break
        Atr, Ate, Btr, Bte = parts
        if min(map(len, parts)) < 20:
            break
        Ga, Gb = gram(Atr), gram(Btr)
        I = np.eye(Ga.shape[0])
        try:
            _, V = eigh(Ga + 1e-10 * I, Gb + 1e-10 * I)
        except np.linalg.LinAlgError:
            continue
        h = V[:, 0] / np.linalg.norm(V[:, 0])
        out.append(q(Bte, h) / max(q(Ate, h), 1e-300))
    return float(np.percentile(out, 97.5)) if out else float("inf")


if __name__ == "__main__":
    from collections import defaultdict
    from scipy.stats import binomtest

    cells = defaultdict(list)
    for f in sorted(glob.glob("*_matched_*.npz")):
        task, _, tag, _, _, _ = f[:-4].split("_")
        cells[(task, tag)].append(f)

    print(f"{'cell':>24}{'n':>4}{'ROW-split':>12}{'DONOR-split':>13}{'binom p':>11}")
    res = {}
    for (task, tag), files in sorted(cells.items()):
        esc_d = n = 0
        for f in files:
            d = np.load(f)          # no allow_pickle: only numeric arrays are read below
            Zs, Zj = d["Zs"], d["Zjm"]
            r = analyse_donor(Zs, Zj)
            if r is None:
                continue
            allZ = np.vstack([Zs, Zj])
            mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
            W = lambda Z: (Z - mu) / sd
            cs = np.linalg.norm(design(W(allZ), DEG), axis=0)
            cs[cs == 0] = 1.0
            env = envelope(design(W(Zs), DEG) / cs)
            n += 1
            esc_d += r["sep_Sblind"] > env
        if not n:
            continue
        p = binomtest(esc_d, n, 0.025, alternative="greater").pvalue
        res[f"{tag} {task}"] = dict(n=n, escapes=esc_d, p=float(p))
        print(f"{tag + ' ' + task:>24}{n:>4}{'—':>12}{str(esc_d) + '/' + str(n):>13}{p:>11.2g}")
    json.dump(res, open("donor_split.json", "w"), indent=1)
    tot_e = sum(v["escapes"] for v in res.values())
    tot_n = sum(v["n"] for v in res.values())
    print(f"\n  TOTAL under a donor-level holdout: {tot_e}/{tot_n} captures escape "
          f"their own same-family extreme-value envelope")
    print("saved -> donor_split.json")
