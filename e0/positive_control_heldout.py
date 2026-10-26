"""Held-out test of the theory-named blind direction, implemented as the paper describes it.

WHY THIS FILE EXISTS (2026-09-23).  An independent pre-submission review, re-running the supplement,
found that positive_control.py::part_b does not implement the protocol the paper states.  It fits the
edge (a0, m, c) on ALL single-site states, and prep() whitens and column-scales on train AND test
states, before splitting into train and test; the direction is then scored on the test half.  No
behavioural readout is used at any point, so the leak is of design geometry, not of responses, but
"named, frozen, then evaluated on held-out interventions" was not what the code did.
positive_control.py is left untouched as the historical record; this file implements the stated
protocol and is what the ledger now reads.

Two held-out protocols, both for the SAME named generator g = (a - a0)(b - m a - c):

  row    the train/test row split of prep() (same seed), but the edge, the whitening and the column
         scaling are all estimated from TRAINING rows only.
  donor  a common donor partition.  Single-site rows are (arm, donor, coefficient) with 7 interpolation
         coefficients per donor per arm; the support-matched joint family crosses every second donor
         upstream with every second donor downstream.  Half the donors are drawn for training; the
         edge, whitening and scaling are fitted on training-donor single-site rows (plus joint rows
         whose BOTH donors are training donors); the score uses single-site rows of held-out donors
         and joint rows whose BOTH donors are held out.  The direction never sees a held-out donor.

For each, the statistic is the paper's: held-out joint energy over held-out single-site energy of the
direction, and the null is 200 random directions in the same polynomial class under the same split
(95th percentile), exactly as in part_b.

    python3 positive_control_heldout.py      # -> positive_control_heldout.json
"""
import glob, json
import numpy as np
from levelC2 import design
from subspace_test import DEG
from positive_control import g_affine

N_COEF = 7          # interpolation coefficients per donor per arm in the single-site family


def _fit_edge(Zs_rows, arm_of_rows):
    """a0 from the patch-B arm (A pinned), m, c from the patch-A arm (B recomputes).  Arms are known by
    design (first half of Zs is one arm, second half the other); which is which is read from the
    spread of a, as part_b does, using training rows only."""
    h1, h2 = Zs_rows[arm_of_rows == 0], Zs_rows[arm_of_rows == 1]
    armA, armB = (h2, h1) if np.std(h1[:, 0]) < np.std(h2[:, 0]) else (h1, h2)
    a0 = float(np.mean(armB[:, 0]))
    m, c = np.polyfit(armA[:, 0], armA[:, 1], 1)
    return a0, float(m), float(c)


def _score(Zs_tr, arm_tr, Zj_tr, Zs_te, Zj_te, seed):
    a0, m, c = _fit_edge(Zs_tr, arm_tr)
    trZ = np.vstack([Zs_tr, Zj_tr])
    mu, sd = trZ.mean(0), trZ.std(0) + 1e-12                     # TRAIN-only whitening
    Wn = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(Wn(trZ), DEG), axis=0); cs[cs == 0] = 1.0   # TRAIN-only scaling
    a0w, mw, cw = (a0 - mu[0]) / sd[0], m * sd[0] / sd[1], (m * mu[0] + c - mu[1]) / sd[1]
    gp = g_affine(a0w, mw, cw) * cs; gp /= np.linalg.norm(gp)
    Ste, Jte = design(Wn(Zs_te), DEG) / cs, design(Wn(Zj_te), DEG) / cs
    q = lambda M, v: float(np.mean((M @ v) ** 2))
    rng = np.random.default_rng(seed)
    rnd = []
    for _ in range(200):
        g = rng.normal(size=len(gp)); g /= np.linalg.norm(g)
        rnd.append(q(Jte, g) / max(q(Ste, g), 1e-300))
    return dict(ratio=q(Jte, gp) / max(q(Ste, gp), 1e-300), rnd_p95=float(np.percentile(rnd, 95)),
                a0=a0, m=m, c=c)


def row_protocol(Zs, Zj, seed=0):
    rng = np.random.default_rng(seed)                             # same draws as subspace_test.prep
    n, half = len(Zs), len(Zs) // 2
    i = rng.permutation(n); s_tr, s_te = i[: n // 2], i[n // 2:]
    j = rng.permutation(len(Zj)); j_tr, j_te = j[: len(Zj) // 2], j[len(Zj) // 2:]
    arm = np.r_[np.zeros(half, int), np.ones(n - half, int)]
    out = _score(Zs[s_tr], arm[s_tr], Zj[j_tr], Zs[s_te], Zj[j_te], seed)
    out["sizes"] = [len(s_tr), len(s_te), len(j_tr), len(j_te)]
    return out


def donor_protocol(Zs, Zj, seed=0):
    n_rows, half = len(Zs), len(Zs) // 2
    n_don = half // N_COEF                                        # donors per arm (32 in the captures)
    assert n_don * N_COEF == half, (n_rows, N_COEF)
    donor = (np.arange(n_rows) % half) // N_COEF                  # donor index within its arm
    arm = np.r_[np.zeros(half, int), np.ones(n_rows - half, int)]
    ids = np.arange(0, n_don, 2)                                  # support-matched joint grid
    ja, jb = np.repeat(ids, len(ids)), np.tile(ids, len(ids))
    assert len(ja) == len(Zj), (len(ja), len(Zj))
    order = np.random.default_rng(seed).permutation(n_don); train = order[: n_don // 2]
    s_tr = np.isin(donor, train)
    j_tr = np.isin(ja, train) & np.isin(jb, train)
    j_te = ~np.isin(ja, train) & ~np.isin(jb, train)              # BOTH endpoints unseen
    out = _score(Zs[s_tr], arm[s_tr], Zj[j_tr], Zs[~s_tr], Zj[j_te], seed)
    out["sizes"] = [int(s_tr.sum()), int((~s_tr).sum()), int(j_tr.sum()), int(j_te.sum())]
    return out


def main():
    # The support-matched captures the headline table reads are listed once, in the ledger's `mpat`
    # dict literal; read that literal rather than duplicating the list here.
    import ast, os
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ledger_paperA.py")).read()
    start = src.index("mpat = {"); end = src.index("}", start) + 1
    mpat = ast.literal_eval(src[start + len("mpat = "):end])
    res = {}
    for pre, pat in mpat.items():
        rows = []
        for f in sorted(glob.glob(pat)):
            d = np.load(f)
            if "Zjm" not in d.files:
                continue
            r = dict(file=f, row=row_protocol(d["Zs"], d["Zjm"]), donor=donor_protocol(d["Zs"], d["Zjm"]))
            rows.append(r)
        if not rows:
            continue
        summ = {}
        for proto in ("row", "donor"):
            rat = np.array([r[proto]["ratio"] for r in rows]); p95 = np.array([r[proto]["rnd_p95"] for r in rows])
            summ[proto] = dict(n=len(rows), gt10=int((rat > 10).sum()), median=float(np.median(rat)),
                               rnd_p95_median=float(np.median(p95)), beats_null=int((rat > p95).sum()))
        res[pre] = dict(pattern=pat, summary=summ, rows=rows)
    tot = {p: [sum(v["summary"][p]["gt10"] for v in res.values()), sum(v["summary"][p]["n"] for v in res.values())]
           for p in ("row", "donor")}

    # ---- the prompt, not the capture, as the unit (same rule as prompt_cluster.py, but for THIS
    # direction and THIS statistic; prompt_cluster.py scores the optimised eigenvector) ----
    from collections import defaultdict
    from prompt_cluster import prompt_of_capture
    P = prompt_of_capture()
    prompt_level = {}
    for proto in ("row", "donor"):
        groups = defaultdict(list)
        for pre, v in res.items():
            for r in v["rows"]:
                groups[(pre, P.get(r["file"], r["file"]))].append(bool(r[proto]["ratio"] > 10))
        keys = list(groups)
        strict = sum(all(h) for h in groups.values())          # a prompt counts only if EVERY capture on it does
        rng = np.random.default_rng(0)
        frac = [np.mean([h for i in rng.choice(len(keys), len(keys), replace=True) for h in groups[keys[i]]])
                for _ in range(2000)]
        prompt_level[proto] = dict(prompts=len(keys), strict=int(strict),
                                   boot_lo=float(np.percentile(frac, 2.5)), boot_median=float(np.median(frac)))
    json.dump(dict(cells=res, total=tot, prompt_level=prompt_level),
              open("positive_control_heldout.json", "w"), indent=1, default=float)
    for proto, v in prompt_level.items():
        print(f"prompt level ({proto}): {v['strict']}/{v['prompts']} prompts with every capture above 10x; "
              f"cluster bootstrap 2.5th pct of the pooled fraction {v['boot_lo']:.3f}")
    for pre, v in res.items():
        s = v["summary"]
        print(f"{pre:13s} row {s['row']['gt10']:>2}/{s['row']['n']:<2} med {s['row']['median']:>8.1f} "
              f"null-p95 {s['row']['rnd_p95_median']:>6.2f} | donor {s['donor']['gt10']:>2}/{s['donor']['n']:<2} "
              f"med {s['donor']['median']:>8.1f} null-p95 {s['donor']['rnd_p95_median']:>6.2f}")
    print("TOTAL >10x   row", tot["row"], "  donor", tot["donor"])


if __name__ == "__main__":
    main()
