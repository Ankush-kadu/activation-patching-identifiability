"""Exact ranks for Proposition 6's numerical check (the float64 check in ksite.py underestimates them).

WHY THIS FILE EXISTS (2026-09-23).  ksite.py measures rank(Res_Q) by an SVD of a float64 monomial
design matrix.  With composing polynomial edges the single-site arms are polynomial curves of degree
up to d * e^(k-1) in the swept value (degree 96 at k=5, e=2, d=6), where float64 cannot resolve the
monomial columns: a review found that the recorded "true rank" of the chain counterexample (50) is a
numerical rank, and that the exact rank is 59 for edges shared across arms and 61 for the per-arm
edges ksite.py draws.  An underestimated rank would also let "rank <= bound" pass when it should not.

This file recomputes every configuration of ksite.py exactly: ranks over the prime field
F_p, p = 2^31 - 1, with generic random edge coefficients and points (the rank over Q is at least the
rank mod p, and equals it for generic choices; three independent draws must agree).  Each arm gets
more points than its univariate degree, so the rank is that of the arms themselves, not of a sample.

  constructions  star (every downstream coordinate a degree-e function of the patched one; ksite.py
                 without --chain), chain with per-arm edges (ksite.py --chain), and chain with edges
                 SHARED across arms (the network case: edge j-1 -> j is the same whichever site is
                 patched)
  checks         rank <= 1 + d * sum_i e_i with e_i the composed degree (Proposition 6), and the
                 per-edge form 1 + d*k*e, which the chain counterexample violates

    python3 ksite_exact.py      # -> ksite_exact.json
"""
import json
import os
import random

import numpy as np

from ksite import monomials

HERE = os.path.dirname(os.path.abspath(__file__))
P = 2 ** 31 - 1


def rank_mod_p(M):
    """Rank of an int64 matrix over F_p by vectorised Gaussian elimination (products stay < 2^62)."""
    M = M.copy() % P
    r, (n, m) = 0, M.shape
    for c in range(m):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        i = r + piv[0]
        M[[r, i]] = M[[i, r]]
        inv = pow(int(M[r, c]), P - 2, P)
        M[r] = (M[r] * inv) % P
        f = M[:, c].copy(); f[r] = 0
        nz = np.nonzero(f)[0]
        if len(nz):
            M[nz] = (M[nz] - (f[nz, None] * M[r][None, :]) % P) % P
        r += 1
        if r == n:
            break
    return r


def composed(k, i, e, chain):
    return e ** max(k - 1 - i, 0) if chain else e


def states(k, e, d, construction, seed):
    rnd = random.Random(seed)
    clean = [rnd.randrange(P) for _ in range(k)]
    shared = [[rnd.randrange(1, P) for _ in range(e + 1)] for _ in range(k)]
    per_arm = [[[rnd.randrange(1, P) for _ in range(e + 1)] for _ in range(k)] for _ in range(k)]
    chain = construction != "star"
    rows = []
    for i in range(k):
        n_pts = d * composed(k, i, e, chain) + 3          # more points than the arm's univariate degree
        for _ in range(n_pts):
            z = clean[:]
            z[i] = rnd.randrange(P)
            for j in range(i + 1, k):
                c = shared[j] if construction == "chain_shared" else per_arm[i][j]
                src, base = (z[j - 1], clean[j - 1]) if chain else (z[i], clean[i])
                dv = (src - base) % P
                z[j] = (clean[j] + sum(c[q] * pow(dv, q, P) for q in range(1, e + 1))) % P
            rows.append(z)
    return rows


def design(rows, mons):
    Z = np.array(rows, dtype=object)
    out = np.empty((len(rows), len(mons)), dtype=np.int64)
    for a, m in enumerate(mons):
        col = np.ones(len(rows), dtype=object)
        for t, p_ in enumerate(m):
            if p_:
                col = col * np.array([pow(int(x), p_, P) for x in Z[:, t]], dtype=object) % P
        out[:, a] = np.array(col, dtype=np.int64)
    return out


def main():
    os.chdir(HERE)
    rows_out, ok_all = [], True
    for construction in ("star", "chain_per_arm", "chain_shared"):
        chain = construction != "star"
        for k in (2, 3, 4, 5):
            for e in (1, 2):
                for d in (2, 3, 4, 5, 6):
                    mons = monomials(k, d)
                    ranks = {rank_mod_p(design(states(k, e, d, construction, s), mons)) for s in range(3)}
                    assert len(ranks) == 1, (construction, k, e, d, ranks)
                    r = ranks.pop()
                    bound = 1 + d * sum(composed(k, i, e, chain) for i in range(k))
                    per_edge = 1 + d * k * e
                    ok = r <= bound
                    ok_all &= ok
                    rows_out.append(dict(construction=construction, k=k, e=e, d=d, dim_class=len(mons), rank=r,
                                         bound=bound, per_edge_bound=per_edge, holds=bool(ok),
                                         per_edge_violated=bool(r > per_edge), attained=bool(r == bound)))
                    print(f"{construction:14s} k={k} e={e} d={d} dim={len(mons):4d} rank={r:4d} bound={bound:4d}"
                          f" per-edge={per_edge:4d} {'OK' if ok else 'VIOLATED'}"
                          f"{'  per-edge form violated' if r > per_edge else ''}")
    ce = {c: next(o["rank"] for o in rows_out if o["construction"] == c and (o["k"], o["e"], o["d"]) == (4, 2, 5))
          for c in ("chain_per_arm", "chain_shared")}
    blind = {kk: next(1 - o["rank"] / o["dim_class"] for o in rows_out
                      if o["construction"] == "star" and (o["k"], o["e"], o["d"]) == (kk, 1, 3)) for kk in (2, 3, 4, 5)}
    summary = dict(all_hold=bool(ok_all), prime=P, counterexample=dict(k=4, e=2, d=5, per_edge_bound=41,
                   exact_rank_shared_edges=ce["chain_shared"], exact_rank_per_arm_edges=ce["chain_per_arm"],
                   composed_bound=1 + 5 * (8 + 4 + 2 + 1)),
                   blind_frac_affine_d3={str(k): v for k, v in blind.items()},
                   n_per_edge_violations=sum(o["per_edge_violated"] for o in rows_out if o["construction"] != "star"))
    json.dump(dict(rows=rows_out, summary=summary), open("ksite_exact.json", "w"), indent=1)
    print("all hold:", ok_all, "| counterexample exact ranks:", ce, "| affine d=3 blind fractions:", blind)


if __name__ == "__main__":
    main()
