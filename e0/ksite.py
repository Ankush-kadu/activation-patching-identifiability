"""THE k-SITE BOUND: single-site evidence grows linearly, the mechanism class grows polynomially.

Added 2026-08-27.  Every experiment in this paper is a two-site slice, and the obvious question is
what happens with more sites.  That answer does not need a new experiment, only a rank count.

Under patch-and-recompute with k sites patched one at a time, the reachable set is a union of k
CURVES: on arm i the patched coordinate is swept, coordinates upstream of i stay at their clean
values, and coordinates downstream of i are whatever the network recomputes.  Write e_i for the
largest total degree of the map from the coordinate PATCHED on arm i to any coordinate downstream
of it.  Restricting a degree-d hypothesis to arm i is then a UNIVARIATE polynomial of degree at
most d*e_i, so arm i imposes at most d*e_i + 1 independent constraints.  With e = max_i e_i,

CAREFUL -- e IS THE COMPOSED DEGREE, NOT THE PER-EDGE DEGREE.  In a chain A -> B -> C with each
edge of degree p, patching A gives b of degree p but c of degree p^2, so e = p^2, not p.  An
earlier version of this file wrote the bound with the per-edge degree and "verified" it on a
construction in which every downstream coordinate was a direct degree-p function of the patched
one -- i.e. on exactly the case where the two agree.  That check was circular, and the bound as
then stated is FALSE for chains: at k=4, per-edge degree 2, d=5 the per-edge bound is 44 and
the measured rank exceeds it (see ksite_chain.json).  Both constructions are tested below.

For AFFINE edges (p = 1) composition is affine, so e = 1 for chains as well, and the bound is
k(d+1).  Every fraction quoted in the paper is that affine case.  Hence

    rank(Res_Q|P_<=d)  <=  1 + d * sum_i e_i   (<= 1 + k d e),   dim P_<=d = C(d + k, k)
    dim N(Q, P_<=d)    >=  C(d + k, k) - (1 + d * sum_i e_i)

THE ARMS SHARE A POINT.  Every arm passes through the clean configuration, because patching a
site with its own clean value is a no-op.  So the k restrictions are not independent: they must
AGREE at that point, which removes (k-1) degrees of freedom from the naive sum sum_i(d e_i + 1).
That is why the bound is 1 + d*sum_i(e_i) and not k(d e + 1).

The evidence grows LINEARLY in d and k.  For a FIXED degree d and growing circuit size k the
class grows like k^d / d!  (C(d+k,k) is symmetric in d and k; d^k/k! is the other regime, fixed k
and growing d).  Either way the blind fraction does not shrink as one studies bigger circuits --
it grows.  This script verifies the bound by
computing the actual numerical rank of the evidence map on synthetic union-of-curves reachable
sets, and contrasts it with a joint (product-grid) family.
"""
import json
import numpy as np
from itertools import combinations_with_replacement
from math import comb

RNG = np.random.default_rng(0)


def monomials(k, d):
    """Exponent tuples of all monomials of total degree <= d in k variables."""
    out = []
    for t in range(d + 1):
        for c in combinations_with_replacement(range(k), t):
            e = [0] * k
            for i in c:
                e[i] += 1
            out.append(tuple(e))
    return out


def single_site_states(k, e, n=400, rng=RNG, chain=False):
    """Union of the k recompute arms: sweep site i, pin upstream, recompute downstream.

    chain=False: every downstream coordinate is a degree-e function of the PATCHED coordinate.
    chain=True : each edge is degree e and they COMPOSE, so coordinate j has degree e^(j-i) in the
                 patched coordinate -- the realistic case for a multi-site circuit, and the one in
                 which the per-edge form of the bound fails.
    """
    clean = rng.normal(size=k)
    coef = [[rng.normal(size=e + 1) for _ in range(k)] for _ in range(k)]
    rows = []
    for i in range(k):
        for tv in np.linspace(-2, 2, n):
            z = clean.copy()
            z[i] = tv
            for j in range(i + 1, k):
                # THE RECOMPUTE MAP MUST FIX THE CLEAN POINT (corrected 2026-08-29).  In a network
                # an alpha=0 patch is a no-op: feeding a site its own clean value leaves every
                # downstream coordinate at ITS clean value, so all k arms meet at the clean
                # configuration.  An earlier version drew each downstream coordinate as a free
                # polynomial of the swept value, which does NOT return clean at the clean input,
                # so the arms never intersected.  That inflated the rank by (k-1) -- the degrees of
                # freedom the shared point removes -- and understated the blind fraction for k>=3
                # (45.0/57.1/66.1% where the truth is 50.0/62.9/71.4%).  Writing the map in
                # DEVIATION form makes it vanish at the clean input by construction.
                src = z[j - 1] if chain else tv
                base = clean[j - 1] if chain else clean[i]
                z[j] = clean[j] + sum(coef[i][j][p] * (src - base) ** p for p in range(1, e + 1))
            rows.append(z)
    return np.array(rows)


def composed_degree(k, i, e, chain):
    """The degree e_i of the map from the coordinate patched on arm i to the furthest downstream."""
    return e ** max(k - 1 - i, 0) if chain else e


def joint_states(k, n=4000, rng=RNG):
    return rng.uniform(-2, 2, size=(n, k))


def design(Z, mons):
    return np.stack([np.prod(Z ** np.array(m), axis=1) for m in mons], 1)


def rank_of(Z, mons, tol=1e-9):
    M = design(Z, mons)
    cs = np.linalg.norm(M, axis=0); cs[cs == 0] = 1.0
    s = np.linalg.svd(M / cs, compute_uv=False)
    return int(np.sum(s > tol * s[0]))


if __name__ == "__main__":
    import sys
    CHAIN = "--chain" in sys.argv        # test the composing (realistic) construction too
    OUT = "ksite_chain.json" if CHAIN else "ksite.json"
    out, ok_all = [], True
    print(f"{'k':>3}{'e':>3}{'d':>3}{'dim P<=d':>10}{'rank Res':>10}{'k(de+1)':>10}"
          f"{'dim N':>8}{'lower bd':>10}{'blind %':>9}  bound holds")
    print("-" * 82)
    for k in (2, 3, 4, 5):
        for e in (1, 2):
            for d in (2, 3, 4, 5, 6):
                mons = monomials(k, d)
                P = len(mons)
                assert P == comb(d + k, k)
                r = rank_of(single_site_states(k, e, chain=CHAIN), mons)
                # sharp form: the k arms meet at the clean point, so the naive per-arm sum
                # sum_i(d*e_i + 1) overcounts by the (k-1) agreement constraints there.
                bound = 1 + sum(d * composed_degree(k, i, e, CHAIN) for i in range(k))
                dimN, lower = P - r, max(P - bound, 0)
                ok = (r <= bound) and (dimN >= lower)
                ok_all &= ok
                print(f"{k:>3}{e:>3}{d:>3}{P:>10}{r:>10}{bound:>10}{dimN:>8}{lower:>10}"
                      f"{100*dimN/P:>8.1f}%  {'yes' if ok else 'NO'}")
                out.append(dict(k=k, e=e, d=d, family="single", dim_class=P, rank=r,
                                bound=bound, dim_blind=dimN, lower_bound=lower,
                                blind_frac=dimN / P, holds=bool(ok)))
    print("\njoint family (product grid), degree 3, same classes:")
    print(f"{'k':>3}{'dim P<=3':>10}{'rank Res':>10}{'dim N':>8}")
    for k in (2, 3, 4, 5):
        mons = monomials(k, 3)
        r = rank_of(joint_states(k), mons)
        print(f"{k:>3}{len(mons):>10}{r:>10}{len(mons)-r:>8}")
        out.append(dict(k=k, e=None, d=3, family="joint", dim_class=len(mons), rank=r,
                        dim_blind=len(mons) - r, blind_frac=(len(mons) - r) / len(mons)))

    # Record the counterexample that shows the PER-EDGE form of the bound is false, so the
    # correction is documented by an artifact rather than by a hand-typed number in the appendix.
    ce = None
    for o in out:
        if o["family"] == "single" and o.get("e") == 2 and o["k"] == 4 and o["d"] == 5:
            ce = dict(k=4, e=2, d=5, per_edge_bound=4 * (5 * 2 + 1), true_rank=o["rank"])
    pick = lambda kk: next(o["blind_frac"] for o in out
                           if o["family"] == "single" and o["k"] == kk and o["d"] == 3
                           and o["e"] == 1)
    summary = dict(all_hold=bool(ok_all), chain=bool(CHAIN), counterexample=ce, blind_frac_k2_d3=float(pick(2)),
                   blind_frac_k3_d3=float(pick(3)), blind_frac_k4_d3=float(pick(4)),
                   blind_frac_k5_d3=float(pick(5)),
                   joint_blind_k5_d3=int(next(o["dim_blind"] for o in out
                                              if o["family"] == "joint" and o["k"] == 5)))
    json.dump({"rows": out, "summary": summary}, open(OUT, "w"), indent=1)
    print(f"\nbound holds in every configuration: {ok_all}")
    print(f"blind fraction of the degree-3 class under single-site evidence:  "
          f"k=2 {100*summary['blind_frac_k2_d3']:.0f}%   k=3 {100*summary['blind_frac_k3_d3']:.0f}%   "
          f"k=4 {100*summary['blind_frac_k4_d3']:.0f}%   k=5 {100*summary['blind_frac_k5_d3']:.0f}%")
    print(f"saved -> {OUT}")
