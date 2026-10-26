"""Does constrained reachability determine mechanism identifiability?

Reframed Track C hypothesis (sharper than "joint patches recover interactions"):

    normalization -> reachable-set compression -> larger mechanism-equivalence
    class -> loss of identifiability under single-site patching

Three things are measured per condition, not one:
  1. REACHABILITY   what joint (a,b) configurations patch-and-recompute can produce
  2. IDENTIFIABILITY whether ANY statistic over the complete single-site family
                     separates the mechanisms (strongest classifier we can throw)
  3. AUGMENTATION   whether the theory-predicted joint intervention breaks it

Norm variants matter for the headline model: Qwen3.5-9B uses RMSNorm, NOT
LayerNorm, so the LayerNorm finding does not transfer for free. Both are
approximately invariant to positive radial scaling, but only approximately --
real implementations carry eps and learned affine parameters. Swept explicitly.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

LOG = "[E0/reach]"
D_MODEL = 16
I_A, I_B = 0, 1


# ---------------------------------------------------------------- norms ----

class NoNorm(nn.Module):
    def forward(self, x):
        return x


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * gamma  -- what Qwen/Llama actually use."""

    def __init__(self, d, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d)) if affine else None

    def forward(self, x):
        r = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return r * self.gamma if self.gamma is not None else r


def make_norm(kind, eps=1e-5, affine=True):
    if kind == "none":
        return NoNorm()
    if kind == "layernorm":
        return nn.LayerNorm(D_MODEL, eps=eps, elementwise_affine=affine)
    if kind == "rmsnorm":
        return RMSNorm(D_MODEL, eps=eps, affine=affine)
    raise ValueError(kind)


# ------------------------------------------------------------ mechanism ----

class Block(nn.Module):
    """A writes a*e_A; the A->B edge reads the NORMED residual and writes b*e_B."""

    def __init__(self, norm_kind, slope=1.0, bias=0.0, eps=1e-5, affine=True):
        super().__init__()
        self.norm = make_norm(norm_kind, eps, affine)
        self.slope, self.bias = slope, bias

    def edge(self, a):
        """Realised b = h(a). This is the network's OWN edge, incl. the norm."""
        a_t = torch.as_tensor(float(a), dtype=torch.float32)
        resid = torch.zeros(D_MODEL)
        resid[I_A] = a_t
        with torch.no_grad():
            r = self.norm(resid)
        return float(self.slope * r[I_A] + self.bias)


def responses(block, gamma, a0, values, semantics, use_f2):
    """Output vector Y over the ENTIRE single-site intervention family.

    f1 = a + b
    f2 = a + b + gamma * (b - h(a)) * (a - a0)      h = block's REALISED edge
    """
    b0 = block.edge(a0)
    out = []

    def y(a, b):
        base = a + b
        return base + gamma * (b - block.edge(a)) * (a - a0) if use_f2 else base

    for v in values:                       # patch A, B free
        b = block.edge(v) if semantics == "recompute" else b0
        out.append(y(v, b))
    for v in values:                       # patch B, A at baseline
        out.append(y(a0, v))
    return np.array(out, dtype=np.float64)


def reachable_set(block, a0, values, semantics):
    b0 = block.edge(a0)
    pts = [(v, block.edge(v) if semantics == "recompute" else b0) for v in values]
    pts += [(a0, v) for v in values]
    return np.array(pts)


def effective_rank(pts):
    """Participation ratio of PCA spectrum: (sum l)^2 / sum l^2. 1.0 = a line."""
    X = pts - pts.mean(0)
    if np.allclose(X, 0):
        return 0.0
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ------------------------------------------------------------ experiment ----

def run_condition(norm_kind, semantics, n_models=200, seed=0, eps=1e-5, affine=True):
    """Generate n_models planted pairs; measure reachability + identifiability."""
    rng = np.random.default_rng(seed)
    values = np.linspace(-3, 3, 41)

    X, y_lab, b_spans, eranks = [], [], [], []
    for i in range(n_models):
        slope = float(rng.uniform(0.5, 2.0)) * (1 if rng.random() < .5 else -1)
        bias = float(rng.uniform(-1, 1))
        gamma = float(rng.uniform(0.5, 2.0))
        a0 = float(rng.uniform(-0.5, 0.5))
        blk = Block(norm_kind, slope, bias, eps, affine)

        for use_f2 in (False, True):
            X.append(responses(blk, gamma, a0, values, semantics, use_f2))
            y_lab.append(int(use_f2))

        R = reachable_set(blk, a0, values, semantics)
        a_only = R[:len(values)]
        b_spans.append(a_only[:, 1].max() - a_only[:, 1].min())
        eranks.append(effective_rank(R))

    X = np.array(X); y_lab = np.array(y_lab)
    # normalise each response vector so the classifier cannot cheat on scale
    mu, sd = X.mean(1, keepdims=True), X.std(1, keepdims=True) + 1e-12
    Xn = (X - mu) / sd

    Xtr, Xte, ytr, yte = train_test_split(Xn, y_lab, test_size=.4,
                                          random_state=seed, stratify=y_lab)
    aucs = {}
    for name, clf in [("GBT", GradientBoostingClassifier(random_state=seed)),
                      ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1500,
                                            random_state=seed))]:
        clf.fit(Xtr, ytr)
        aucs[name] = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    raw_gap = np.abs(X[y_lab == 0] - X[y_lab == 1]).max()
    return dict(b_span=float(np.mean(b_spans)), erank=float(np.mean(eranks)),
                auc=max(aucs.values()), aucs=aucs, raw_gap=float(raw_gap))


def scale_invariance_sweep(norm_kind, eps=1e-5, affine=True):
    """Explicit magnitude AND sign sweep -- do NOT assume exact scale invariance."""
    blk = Block(norm_kind, 1.0, 0.0, eps, affine)
    return [(a, blk.edge(a)) for a in
            (-100, -10, -3, -1, -0.1, -0.01, 0.01, 0.1, 1, 3, 10, 100)]


if __name__ == "__main__":
    print("=" * 84)
    print("SCALE-INVARIANCE SWEEP (magnitude x sign) -- is the edge really ~sign(a)?")
    print("=" * 84)
    for nk in ("none", "layernorm", "rmsnorm"):
        pts = scale_invariance_sweep(nk)
        print(f"\n{LOG} norm={nk}")
        print("   a:  " + " ".join(f"{a:>9.2f}" for a, _ in pts))
        print("   b:  " + " ".join(f"{b:>9.4f}" for _, b in pts))

    print("\n" + "=" * 84)
    print("REACHABILITY x IDENTIFIABILITY")
    print("=" * 84)
    print(f"\n{'norm':<12}{'semantics':<12}{'b-span':>10}{'eff.rank':>10}"
          f"{'max|dY|':>12}{'AUC':>8}   verdict")
    print("-" * 84)
    rows = []
    for nk in ("none", "layernorm", "rmsnorm"):
        for sem in ("recompute", "clamped"):
            r = run_condition(nk, sem)
            v = "IDENTIFIABLE" if r["auc"] >= .80 else "indistinguishable"
            rows.append((nk, sem, r))
            print(f"{nk:<12}{sem:<12}{r['b_span']:>10.4f}{r['erank']:>10.4f}"
                  f"{r['raw_gap']:>12.4f}{r['auc']:>8.3f}   {v}")

    json.dump({f"{nk}/{sem}": {k: (float(v) if isinstance(v, (int, float)) else v)
                               for k, v in r.items() if k != "aucs"}
               for nk, sem, r in rows},
              open("reachability.json", "w"), indent=1)
    print("\nsaved -> reachability.json")

    print("\n" + "=" * 84)
    print("GATE (Tier-1, theorem-backed pair)")
    print("=" * 84)
    ok = True
    for nk, sem, r in rows:
        if sem == "recompute" and r["auc"] >= .80:
            print(f"  KILL: {nk}/recompute single-site AUC={r['auc']:.3f} >= 0.80"); ok = False
        if sem == "clamped" and r["auc"] < .95:
            print(f"  KILL: {nk}/clamped single-site AUC={r['auc']:.3f} < 0.95"); ok = False
    print("  ALL TIER-1 CRITERIA PASS" if ok else "  >>> GATE FIRED <<<")
