"""COORDINATE-INVARIANT REPORTING: visibility, regularization, and the full held-out profile.

reparam.py showed that two things we had been reporting are NOT invariant under an invertible
re-coordinatization phi -> A phi of a fixed mechanism class F:
    (a) ridge  G + lam*I           -- because I is tied to Euclidean coefficient coordinates
    (b) Euclidean principal angles -- and, for the same reason,
    (c) any "absolute visibility"  h^T G_Q h reported under ||h||_2 = 1.

FIX 1 -- INVARIANT VISIBILITY.  Normalize in FUNCTION space using the pooled Gram
G_pool = G_S + G_J, and report the energy SHARE
    v_S(h) = h^T G_S h / h^T G_pool h ,      v_J(h) = h^T G_J h / h^T G_pool h ,   v_S + v_J = 1 .
Under phi -> A phi with h -> A^{-T} h every quadratic form is unchanged, so v_S and v_J are
invariant.  (v_S = lam/(1+lam) for the generalized eigenvalue lam, a monotone reparameterization
of the invariant object.)

FIX 2 -- INVARIANT REGULARIZATION.  Replace G_Q + lam*I by G_Q + lam*G_ref, G_ref=(G_S+G_J)/2,
which transforms covariantly, so the generalized spectrum is invariant.

FIX 3 -- THE WHOLE PROFILE, NOT ONE DIRECTION.  Freeze every train eigenvector and re-score its
response ratio on the untouched complement; compare against a SAME-FAMILY NULL that passes
through the identical procedure.

FIX 4 (2026-08-27) -- A CORRECT NULL FOR AN EXTREME.  The previous envelope was the 2.5-97.5
percentile of null profile VALUES pooled over eigen-indices, but the test statistic is the
profile's MIN / MAX.  The primary envelope is now the extreme-value one, built LEAVE-ONE-TRIAL-OUT
so no trial is scored against a threshold its own nulls helped set, and each cell carries a
one-sided binomial p-value against the envelope's 2.5% nominal per-end escape rate, Holm-adjusted
across cells.  The pointwise envelope is kept as `*_pointwise` for the record.

FIX 5 (2026-08-27) -- ONE CELL PER MODEL x TASK.  Cells come from cells.CELLS; task captures
are model-tagged, so no cell pools two models.
"""
import glob, json
import numpy as np
from scipy.linalg import eigh
from subspace_test import prep, gram
from reparam import rand_invertible
from cells import CELLS, restrict_alpha
from nulls import extreme_envelope, loo_envelope, escape_binom_p, holm, NOMINAL

LAM = 1e-6
NULL_REPS = 20


def spectrum(Gs, Gj, lam=LAM):
    """Generalized eigenvalues of (G_S, G_J) with COVARIANT regularization."""
    Gref = 0.5 * (Gs + Gj)
    return eigh(Gs + lam * Gref, Gj + lam * Gref)


def vis(Ms, Mj, h):
    """Invariant energy shares (v_S, v_J) on the given samples; they sum to 1."""
    s = float(np.mean((Ms @ h) ** 2))
    j = float(np.mean((Mj @ h) ** 2))
    t = s + j
    return (s / t, j / t) if t > 0 else (np.nan, np.nan)


def heldout_profile(Vmat, Ms, Mj):
    """Held-out log10 response ratio of every frozen train eigenvector, sorted."""
    out = []
    for k in range(Vmat.shape[1]):
        v = Vmat[:, k] / np.linalg.norm(Vmat[:, k])
        num = float(np.mean((Ms @ v) ** 2)); den = float(np.mean((Mj @ v) ** 2))
        out.append(np.log10(max(num, 1e-300) / max(den, 1e-300)))
    return sorted(out)


def one(Zs, Zj, seed=0, null_reps=NULL_REPS):
    Str, Ste, Jtr, Jte = prep(Zs, Zj, seed=seed)
    Gs, Gj = gram(Str), gram(Jtr)
    lam, V = spectrum(Gs, Gj)

    hS = V[:, 0] / np.linalg.norm(V[:, 0])      # most S-blind / J-visible
    hJ = V[:, -1] / np.linalg.norm(V[:, -1])    # most J-blind / S-visible
    vS_s, vS_j = vis(Ste, Jte, hS)              # HELD-OUT, frozen direction
    vJ_s, vJ_j = vis(Ste, Jte, hJ)
    ho = heldout_profile(V, Ste, Jte)

    # ---- same-family null: split S in half, identical train-freeze-score procedure ----
    null = []
    rng = np.random.default_rng(seed)
    allS = np.vstack([Str, Ste])
    for r in range(null_reps):
        i = rng.permutation(len(allS))
        a, b = allS[i[: len(i) // 2]], allS[i[len(i) // 2:]]
        na, nb = len(a) // 2, len(b) // 2
        try:
            _, V2 = spectrum(gram(a[:na]), gram(b[:nb]))
            null.append(heldout_profile(V2, a[na:], b[nb:]))
        except np.linalg.LinAlgError:
            pass

    # ---- invariance check of the COVARIANT regularizer vs the Euclidean one ----
    A = rand_invertible(Gs.shape[0], np.random.default_rng(seed))
    l_cov, _ = spectrum(A @ Gs @ A.T, A @ Gj @ A.T)
    n = Gs.shape[0]
    l_euc = eigh(A @ Gs @ A.T + LAM * np.eye(n), A @ Gj @ A.T + LAM * np.eye(n),
                 eigvals_only=True)
    l_euc0 = eigh(Gs + LAM * np.eye(n), Gj + LAM * np.eye(n), eigvals_only=True)
    rel = lambda x, y: float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-300)))

    return dict(
        log_spectrum=[float(x) for x in ho],                 # HELD-OUT (primary)
        log_spectrum_insample=[float(x) for x in np.log10(np.maximum(lam, 1e-300))],
        null_log_spectrum=[[float(x) for x in row] for row in null],
        vS_on_Sblind=vS_s, vJ_on_Sblind=vS_j,
        vS_on_Jblind=vJ_s, vJ_on_Jblind=vJ_j,
        cov_reg_invariance=rel(l_cov, lam),
        euc_reg_invariance=rel(l_euc, l_euc0))


def run(pattern, label, files=None, alpha_restrict=False, pairs=None):
    """pairs: optional list of (name, Zs, Zj); otherwise load Zs/Zj from the globbed files."""
    if pairs is None:
        files = sorted(glob.glob(pattern)) if files is None else files
        if not files:
            return None
        prep_S = restrict_alpha if alpha_restrict else (lambda Z: Z)
        pairs = [(f, prep_S(np.load(f)["Zs"]), np.load(f)["Zj"]) for f in files]
    if not pairs:
        return None
    files = [p[0] for p in pairs]
    rows = [one(Zs, Zj) for _, Zs, Zj in pairs]
    nulls_per_trial = [r["null_log_spectrum"] for r in rows]
    spec = np.array([r["log_spectrum"] for r in rows])

    # pointwise envelope (the pre-2026-08-27 statistic; kept for the record)
    allnull = np.concatenate([np.array(p) for ps in nulls_per_trial for p in ps])
    lo_pw, hi_pw = np.percentile(allnull, [2.5, 97.5])
    below_pw = int(np.sum(spec.min(1) < lo_pw)); above_pw = int(np.sum(spec.max(1) > hi_pw))
    both_pw = int(np.sum((spec.min(1) < lo_pw) & (spec.max(1) > hi_pw)))

    # extreme-value envelope, LEAVE-ONE-TRIAL-OUT (PRIMARY)
    below = above = both = 0
    per_trial = []
    for i in range(len(rows)):
        lo_i, hi_i = loo_envelope(nulls_per_trial, i)
        mn, mx = float(spec[i].min()), float(spec[i].max())
        b_, a_ = mn < lo_i, mx > hi_i
        below += b_; above += a_; both += (b_ and a_)
        per_trial.append(dict(file=files[i].split("/")[-1], lo=lo_i, hi=hi_i,
                              min=mn, max=mx, below=bool(b_), above=bool(a_)))
    n = len(rows)
    p_below, p_above = escape_binom_p(below, n), escape_binom_p(above, n)
    # pooled envelope, for the figure band only
    lo, hi = extreme_envelope([p for ps in nulls_per_trial for p in ps])

    print("=" * 96)
    print(f"{label}   n={n}")
    print("=" * 96)
    print(f"  INVARIANT VISIBILITY on HELD-OUT data (energy shares, sum to 1)")
    for nm, a, b in (("S-blind dir", "vS_on_Sblind", "vJ_on_Sblind"),
                     ("J-blind dir", "vS_on_Jblind", "vJ_on_Jblind")):
        vs = np.median([r[a] for r in rows]); vj = np.median([r[b] for r in rows])
        print(f"    {nm}:  v_S = {vs:.6f}  ({100*vs:.4f}% of pooled energy)   "
              f"v_J = {vj:.6f}  ({100*vj:.4f}%)")
    print(f"\n  HELD-OUT PROFILE vs SAME-FAMILY NULL  (log10 response ratio)")
    print(f"    extreme-value envelope, leave-one-out (PRIMARY), pooled [{lo:+.2f}, {hi:+.2f}]:")
    print(f"        S-blind end {below}/{n} (binom p={p_below:.2g})   "
          f"J-blind end {above}/{n} (binom p={p_above:.2g})   BOTH {both}/{n}")
    print(f"    pointwise envelope (pre-fix, anti-conservative) [{lo_pw:+.2f}, {hi_pw:+.2f}]: "
          f"below {below_pw}  above {above_pw}  both {both_pw}")
    print(f"\n  REGULARIZER INVARIANCE under phi -> A phi (max relative spectrum change)")
    print(f"    covariant  G + lam*G_ref : {np.median([r['cov_reg_invariance'] for r in rows]):.3e}")
    print(f"    Euclidean  G + lam*I     : {np.median([r['euc_reg_invariance'] for r in rows]):.3e}")
    return dict(n=n, files=[f.split("/")[-1] for f in files],
                null_lo=float(lo), null_hi=float(hi),
                both_ends=both, below=below, above=above,
                p_below=p_below, p_above=p_above, per_trial=per_trial,
                null_lo_pointwise=float(lo_pw), null_hi_pointwise=float(hi_pw),
                both_ends_pointwise=both_pw, below_pointwise=below_pw, above_pointwise=above_pw,
                vS_on_Sblind=float(np.median([r["vS_on_Sblind"] for r in rows])),
                vJ_on_Sblind=float(np.median([r["vJ_on_Sblind"] for r in rows])),
                vS_on_Jblind=float(np.median([r["vS_on_Jblind"] for r in rows])),
                vJ_on_Jblind=float(np.median([r["vJ_on_Jblind"] for r in rows])),
                cov_reg_invariance=float(np.median([r["cov_reg_invariance"] for r in rows])),
                euc_reg_invariance=float(np.median([r["euc_reg_invariance"] for r in rows])),
                spectra=spec.tolist())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="invariant.json")
    ap.add_argument("--alpha-restrict", action="store_true",
                    help="keep only single-site states with alpha in [0,1] (support check)")
    A = ap.parse_args()
    out = {}
    for c in CELLS:
        r = run(c["pattern"], c["label"], alpha_restrict=A.alpha_restrict)
        if r:
            out[c["label"]] = r
        print()

    # Holm across cells: the escape counts are a family of 2 x n_cells tests
    keys = list(out)
    for end, pk, ak in (("low", "p_below", "q_below"), ("high", "p_above", "q_above")):
        q = holm([out[k][pk] for k in keys])
        for k, v in zip(keys, q):
            out[k][ak] = float(v)
    print("=" * 96)
    print(f"{'cell':<22}{'n':>4}{'S-blind':>10}{'q':>10}{'J-blind':>10}{'q':>10}   verdict")
    print("-" * 96)
    for k in keys:
        r = out[k]
        lo_ok, hi_ok = r["q_below"] < 0.05, r["q_above"] < 0.05
        verdict = ("mutual non-domination" if lo_ok and hi_ok else
                   "S-blind only" if lo_ok else "J-blind only" if hi_ok else "not established")
        print(f"{k:<22}{r['n']:>4}{r['below']:>7}/{r['n']:<2}{r['q_below']:>10.2g}"
              f"{r['above']:>7}/{r['n']:<2}{r['q_above']:>10.2g}   {verdict}")
    json.dump(out, open(A.out, "w"), indent=1)
    print(f"\nsaved -> {A.out}")
