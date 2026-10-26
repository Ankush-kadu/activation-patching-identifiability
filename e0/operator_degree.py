"""TRACK A — OPERATOR-SLICE RESOLVING THEOREM.

Turns the isolated observation "RMSNorm moves d* from 2 to 5" into a statement about the
algebraic complexity an architectural operator induces in the reachable set.

--------------------------------------------------------------------------------------------
SETTING (controlled operator slice, two sites a -> b, patch-and-recompute semantics)
    pinned branch    H = { a = a0 }        (patch the downstream site; upstream stays clean)
    recompute branch C = { b = h(a) }      (patch the upstream site; downstream RECOMPUTES)
    reachable set    R = H u C
h is the edge map the architectural operator implements.

THEOREM A (operator-slice resolving theorem).  Let L = a - a0, and suppose the recompute branch
is Zariski-dense in an irreducible affine curve V(P) for a polynomial P with gcd(L,P) = 1.
Then, in k[a,b] (a UFD),
        I(R) = I(H) ^ I(C) = (L) ^ (P) = (L * P),
and therefore the first degree at which an exact blind mechanism difference exists is
        d* = deg(L*P) = 1 + deg P .

POPULATION, NOT GRID.  Every d* here refers to the ideal of the EXACT operator-induced
reachable variety.  Any finite point set admits high-degree vanishing polynomials, so sampled
grids can only ever CORROBORATE, never establish, d*.  Grids below are verification only.

COROLLARIES (instantiations, not separate results)
    affine    h(a) = m a + c              -> P = b - m a - c,  deg 1  -> d* = 2
    RMSNorm   h(a) = beta + c a / sqrt(kappa a^2 + delta)
                                          -> P = (b-beta)^2 (kappa a^2 + delta) - c^2 a^2,
                                             deg 4                    -> d* = 5
    LayerNorm same algebraic shape        -> deg 4                    -> d* = 5

THE DICHOTOMY (this is the part that is not degree bookkeeping).
If, ON THE ANALYSED SLICE, h is TRANSCENDENTAL over k(a) -- as for the softmax/logistic edge and
the sigmoid-gated linear attention edge studied here -- then no nonzero polynomial vanishes on
the recompute branch, so I(C) = (0), hence I(R) = I(H) ^ (0) = (0) and
        d* = infinity  for the polynomial mechanism hierarchy ON THAT SLICE.
SCOPE.  This is a statement about the specified operator-induced reachable geometry in the
chosen polynomial mechanism coordinates.  It is NOT the claim that a whole transformer has no
exact blind mechanisms -- a full network may carry exact constraints for unrelated reasons.
Algebraic slices admit exact blind mechanisms; transcendental slices do not.
"""
import json
import numpy as np
import sympy as sp

a, b, s = sp.symbols("a b s", real=True)


# --------------------------------------------------------- symbolic derivation ---
def minimal_relation(kind, params):
    """Return (P, note) where P is the minimal polynomial relation of the recompute branch,
    or (None, note) when the branch is provably not algebraic."""
    if kind == "affine":
        m, c = params["m"], params["c"]
        return sp.expand(b - m * a - c), "polynomial edge; no elimination needed"

    if kind in ("rmsnorm", "layernorm"):
        # h(a) = beta + c a / sqrt(kappa a^2 + delta)   (delta > 0)
        beta, c, kappa, delta = (params[k] for k in ("beta", "c", "kappa", "delta"))
        # introduce s = sqrt(kappa a^2 + delta), eliminate by resultant
        f = sp.expand((b - beta) * s - c * a)          # (b-beta) s = c a
        g = sp.expand(s ** 2 - (kappa * a ** 2 + delta))
        P = sp.expand(sp.resultant(f, g, s))
        return sp.expand(sp.simplify(P)), "radical cleared by resultant in s"

    if kind in ("softmax", "gated_linear"):
        return None, ("h is transcendental over k(a): its graph lies on no algebraic curve, "
                      "so I(C) = (0)")
    raise ValueError(kind)


def analyse_operator(kind, params, a0):
    P, note = minimal_relation(kind, params)
    L = sp.expand(a - a0)
    out = {"operator": kind, "note": note}

    if P is None:
        out.update(deg_P=None, d_star=None, d_star_label="infinity",
                   irreducible=None, coprime=None,
                   ideal="(0)  -- exact blind space is trivial at every degree")
        return out

    Pp = sp.Poly(P, a, b)
    facs = sp.factor_list(P)
    n_nonconst = sum(mult for f, mult in facs[1])
    irred = (len(facs[1]) == 1 and facs[1][0][1] == 1)
    g = sp.gcd(sp.Poly(L, a, b), Pp)
    coprime = sp.Poly(g, a, b).total_degree() == 0

    out.update(P=sp.sstr(sp.expand(P)), deg_P=int(Pp.total_degree()),
               d_star=int(Pp.total_degree()) + 1, d_star_label=str(Pp.total_degree() + 1),
               irreducible=bool(irred), n_factors=int(n_nonconst), coprime=bool(coprime),
               generator=sp.sstr(sp.expand(L * P)),
               ideal="(L*P)")
    return out


# ------------------------------------------------------ numerical corroboration ---
def edge_fn(kind, params):
    if kind == "affine":
        return lambda x: params["m"] * x + params["c"]
    if kind in ("rmsnorm", "layernorm"):
        return lambda x: params["beta"] + params["c"] * x / np.sqrt(
            params["kappa"] * x ** 2 + params["delta"])
    if kind == "softmax":
        return lambda x: params["v"] / (1.0 + np.exp(-params["g"] * x))
    if kind == "gated_linear":
        return lambda x: (params["w"] * x) / (1.0 + np.exp(-params["g"] * x))
    raise ValueError(kind)


def exps(d):
    return [(i, t - i) for t in range(d + 1) for i in range(t + 1)]


def dmat(Z, d):
    return np.array([[x ** i * y ** j for (i, j) in exps(d)] for x, y in Z], float)


def rel_residual(kind, params, a0, d, span, npts=241):
    """Relative max |P(z)| on the reachable set for the best degree-d candidate."""
    h = edge_fn(kind, params)
    grid = np.linspace(-span, span, npts)
    R = np.array([(x, h(x)) for x in grid] + [(a0, y) for y in grid], float)
    M = dmat(R, d)
    cs = np.linalg.norm(M, axis=0); cs[cs == 0] = 1.0
    Vt = np.linalg.svd(M / cs, full_matrices=True)[2]
    v = Vt[-1] / cs
    v = v / np.linalg.norm(v)
    return float(np.max(np.abs(M @ v)) / max(np.max(np.abs(M) @ np.abs(v)), 1e-300))


def numeric_dstar(kind, params, a0, dmax=7, spans=(2.0, 8.0, 32.0), tol=1e-12):
    """DOMAIN-INVARIANCE criterion (corrected 2026-08-19).

    A fixed singular-value threshold cannot distinguish an EXACT algebraic relation from a very
    good POLYNOMIAL APPROXIMATION of a smooth transcendental curve on a bounded interval -- the
    first version of this function reported d*=5 for softmax for exactly that reason.  The
    distinguishing property is that an exact relation vanishes on the WHOLE curve, so its
    residual is INVARIANT to widening the sampling interval, whereas an approximation degrades.
    Measured: algebraic operators stay ~1e-16 across spans 2..32; transcendental ones climb by
    ~6 orders of magnitude.
    """
    out = {"per_degree": {}}
    found = None
    for d in range(1, dmax + 1):
        rs = [rel_residual(kind, params, a0, d, sp_) for sp_ in spans]
        out["per_degree"][d] = [float(x) for x in rs]
        if found is None and max(rs) < tol:
            found = d
    # DECAY-PATTERN corroboration.  An exact relation shows a CATASTROPHIC single-degree drop
    # to machine precision; a polynomial approximation of a smooth transcendental curve decays
    # geometrically.  Numerics can never CERTIFY non-existence -- the transcendence proof does
    # that -- so this only corroborates.
    sudden = None
    for d in range(2, dmax + 1):
        prev = max(out["per_degree"][d - 1]); cur = max(out["per_degree"][d])
        if prev > 1e-6 and cur < 1e-13 and cur / max(prev, 1e-300) < 1e-6:
            sudden = d
            break
    out["sudden_drop_degree"] = sudden
    out["signature"] = "algebraic (catastrophic drop)" if sudden else \
                       "no catastrophic drop -> consistent with geometric approximation"
    out["d_star_numeric"] = sudden
    out["widest_residual_at_dmax"] = float(max(
        rel_residual(kind, params, a0, dmax, sp_) for sp_ in spans))
    return out


# ------------------------------------------------------------------------ main ---
OPS = [
    ("affine", dict(m=sp.Rational(17, 10), c=sp.Rational(2, 5)),
     dict(m=1.7, c=0.4)),
    ("rmsnorm", dict(beta=sp.Rational(-1, 5), c=2, kappa=1, delta=sp.Rational(1, 100)),
     dict(beta=-0.2, c=2.0, kappa=1.0, delta=0.01)),
    ("layernorm", dict(beta=sp.Rational(1, 10), c=sp.Rational(3, 2),
                       kappa=sp.Rational(15, 16), delta=sp.Rational(1, 100)),
     dict(beta=0.1, c=1.5, kappa=0.9375, delta=0.01)),
    ("softmax", None, dict(v=3.0, g=1.4)),
    ("gated_linear", None, dict(w=1.3, g=1.1)),
]
A0 = sp.Rational(3, 10)
A0F = 0.3


if __name__ == "__main__":
    print("=" * 100)
    print("THEOREM A — operator-slice resolving theorem:  d* = 1 + deg P")
    print("  (population statement about the exact reachable variety; grids corroborate only)")
    print("=" * 100)
    print(f"  {'operator':<14}{'deg P':>7}{'irred':>7}{'coprime':>9}{'d* (theory)':>13}"
          f"  numeric d*   resid@d=7 (widest span)")
    print("-" * 100)

    RES = []
    for kind, spar, npar in OPS:
        sym = analyse_operator(kind, spar, A0) if spar is not None else \
            analyse_operator(kind, None, A0)
        num = numeric_dstar(kind, npar, A0F)
        sym["numeric"] = num
        sym["domain_invariant"] = num["d_star_numeric"] is not None
        RES.append(sym)
        dp = "-" if sym["deg_P"] is None else sym["deg_P"]
        ir = "-" if sym["irreducible"] is None else ("yes" if sym["irreducible"] else "NO")
        cp = "-" if sym["coprime"] is None else ("yes" if sym["coprime"] else "NO")
        nd = num["d_star_numeric"] if num["d_star_numeric"] else "none<=7"
        print(f"  {kind:<14}{str(dp):>7}{ir:>7}{cp:>9}{sym['d_star_label']:>13}"
              f"   {str(nd):>10}   {num['widest_residual_at_dmax']:.1e}")

    print("\n  minimal relations P (recompute branch):")
    for r in RES:
        if r.get("P"):
            print(f"    {r['operator']:<12} P = {r['P']}")
        else:
            print(f"    {r['operator']:<12} {r['note']}")

    print("\n" + "=" * 100)
    print("THE DICHOTOMY")
    print("=" * 100)
    alg = [r for r in RES if r["deg_P"] is not None]
    tra = [r for r in RES if r["deg_P"] is None]
    print(f"  ALGEBRAIC operators ({', '.join(r['operator'] for r in alg)}):")
    print(f"    exact blind mechanisms EXIST, first at degree d* = 1 + deg P")
    print(f"    -> affine d*={alg[0]['d_star']}, normalization d*={alg[1]['d_star']}: an "
          f"architectural operator that raises the algebraic")
    print(f"       complexity of the reachable set RAISES the onset of exact blindness.")
    print(f"  TRANSCENDENTAL operators ({', '.join(r['operator'] for r in tra)}):")
    print(f"    I(C) = (0), so d* = infinity: NO exact blind mechanism at any finite degree.")
    print(f"    The polynomial mechanism class is exactly identifiable under this family;")
    print(f"    only the approximate Gram analysis applies to such operators.")

    ok_theory = all(r["numeric"]["d_star_numeric"] == r["d_star"] for r in alg)
    ok_inf = all(r["numeric"]["d_star_numeric"] is None for r in tra)
    print(f"\n  numeric d* matches theory on algebraic operators   : {ok_theory}")
    print(f"  no catastrophic drop for transcendental            : {ok_inf}")
    print("\n  residual decay by degree (max over spans 2/8/32):")
    for r in RES:
        pd = r["numeric"]["per_degree"]
        row = "  ".join(f"d{d}:{max(pd[d]):.0e}" for d in sorted(pd))
        print(f"    {r['operator']:<14} {row}   [{r['numeric']['signature']}]")
    print("\n  NOTE: numerics cannot certify NON-existence of a polynomial relation.")
    print("        For softmax / gated linear attention the transcendence of h over k(a) is the")
    print("        proof; the absence of a catastrophic drop merely corroborates it.")
    print(f"\n  SCOPE (do not overstate).  The statement is about the ANALYSED OPERATOR SLICES,")
    print(f"  not about whole networks: for the softmax and gated-linear slices studied here the")
    print(f"  recomputation graph is transcendental over the chosen polynomial mechanism")
    print(f"  coordinates, so the polynomial hierarchy has NO finite-degree exact blind direction")
    print(f"  on those slices.  A full network may still carry exact constraints for unrelated")
    print(f"  reasons.  What this does give is the natural bridge to the trained-model sections:")
    print(f"  exact polynomial blindness disappears in the transcendental regime, yet FINITE")
    print(f"  evidence can remain severely ill-conditioned -- which is what the Gram/resolving-")
    print(f"  spectrum analysis measures.")

    json.dump(RES, open("operator_degree.json", "w"), indent=1, default=str)
    print("\nsaved -> operator_degree.json")
