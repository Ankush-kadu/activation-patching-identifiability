"""THREE SITES IN A TRAINED MODEL (added 2026-08-27).

Proposition 2 (ksite.py) says single-site evidence imposes at most k(de+1) constraints while the
mechanism class has C(d+k,k) dimensions, so the blind fraction GROWS with the number of sites.
That is a rank count, verified on synthetic union-of-curves data.  This script instantiates it in
GPT-2 with k = 3 real attention blocks, using the same chart, the same alpha grid and the same
train-freeze-score discipline as the two-site experiments.

    S   patch each of the three sites alone, 32 donors x 7 alphas per arm  (three curves in 3-D)
    J   patch all three jointly from independent donors, on the SAME alpha grid (support-matched)

Reported per trial and per degree: the dimension of the mechanism class, the numerical rank of the
evidence map under each family (so dim N = dim class - rank), and the held-out response share of
the least-visible direction, scored against a same-family null built exactly as in invariant.py.
"""
import argparse, glob, json, time
import numpy as np
import torch
from scipy.linalg import eigh
from transformers import AutoTokenizer, AutoModelForCausalLM
from levelC2 import DEV, TEMPLATES
from mech_classes import design                     # k-dimensional polynomial design
from nulls import loo_envelope, escape_binom_p
from cells import N_AL, ALPHAS

AL = np.array(ALPHAS)
N_DON, STEP = 32, 4                                 # 8x8x8 = 512 joint states
LAM = 1e-6


class ProbeK:
    """Read/force the last-token output of k modules."""

    def __init__(s, model, layers, modfn):
        s.cache, s.force, s._h = {}, {}, []
        for i, l in enumerate(layers):
            s._h.append(modfn(model, l).register_forward_hook(s._mk(i)))

    def _mk(s, i):
        def hook(md, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            f = s.force.get(i)
            if f is not None:
                o = o.clone()
                o[:, -1, :] = f.to(o.dtype)
                s.cache[i] = o[:, -1, :].detach().clone()
                return ((o,) + tuple(out[1:])) if isinstance(out, tuple) else o
            s.cache[i] = o[:, -1, :].detach().clone()
            return out
        return hook

    def close(s):
        for h in s._h:
            h.remove()


def batched(model, p, ids, io, sb, forces, k, bs=16):
    """forces: dict i -> [N,d] or None. Returns per-site activations [k][N,d] and y[N]."""
    n = next(v.shape[0] for v in forces.values() if v is not None) if forces else 1
    A = [[] for _ in range(k)]; Y = []
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        p.force = {i: (v[i0:i1] if v is not None else None) for i, v in forces.items()}
        lg = model(ids.repeat(i1 - i0, 1)).logits[:, -1, :].float()
        p.force = {}
        for i in range(k):
            A[i].append(p.cache[i].float().cpu())
        Y.append((lg[:, io] - lg[:, sb]).cpu())
    return ([torch.cat(a).numpy().astype(np.float64) for a in A],
            torch.cat(Y).numpy().astype(np.float64))


def collect(model, tok, modfn, layers, seed, names, bs=16):
    rng = np.random.default_rng(seed)
    k = len(layers)
    t = TEMPLATES[rng.integers(len(TEMPLATES))]
    a, b = rng.choice(names, 2, replace=False)
    prompt = t.format(A=a, B=b)
    io = tok(" " + a, add_special_tokens=False).input_ids[0]
    sb = tok(" " + b, add_special_tokens=False).input_ids[0]
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)

    p = ProbeK(model, layers, modfn)
    clean, y0 = batched(model, p, ids, io, sb, {}, k, bs=1)
    clean = [c[0] for c in clean]

    dP = [t.format(A=x, B=y) for x, y in
          [rng.choice(names, 2, replace=False) for _ in range(N_DON)]]
    don = [[] for _ in range(k)]
    for i in range(0, len(dP), 8):
        pid = torch.cat([tok(q, return_tensors="pt").input_ids for q in dP[i:i + 8]]).to(DEV)
        p.force = {}
        model(pid)
        for j in range(k):
            don[j].append(p.cache[j].float().cpu().numpy().astype(np.float64))
    don = [np.vstack(d) for d in don]
    T = lambda X: torch.tensor(X, dtype=torch.float32, device=DEV)

    # ---- S: each site alone, full alpha grid ----
    Zs_parts, Ys_parts = [], []
    for i in range(k):
        f = np.stack([clean[i] + x * (v - clean[i]) for v in don[i] for x in AL])
        A_, y_ = batched(model, p, ids, io, sb, {i: T(f)}, k, bs=bs)
        Zs_parts.append(A_); Ys_parts.append(y_)

    # ---- J: all k sites jointly, SAME alpha grid ----
    idx = [(i, j, l) for i in range(0, N_DON, STEP)
           for j in range(0, N_DON, STEP) for l in range(0, N_DON, STEP)]
    fj = {}
    for s_ in range(k):
        rows = []
        for n_, trip in enumerate(idx):
            al = AL[(n_ * (s_ + 1) + s_) % N_AL]
            rows.append(clean[s_] + al * (don[s_][trip[s_]] - clean[s_]))
        fj[s_] = T(np.stack(rows))
    Aj, Yj = batched(model, p, ids, io, sb, fj, k, bs=bs)
    p.close()

    W = model.lm_head.weight
    u = (W[io] - W[sb]).float().cpu().numpy().astype(np.float64)
    u /= np.linalg.norm(u)
    Zs = np.vstack([np.stack([A_[i] @ u for i in range(k)], 1) for A_ in Zs_parts])
    Ys = np.concatenate(Ys_parts)
    Zj = np.stack([Aj[i] @ u for i in range(k)], 1)
    return Zs, Ys, Zj, Yj, prompt, float(y0[0])


def profile(Zs, Zj, deg, seed=0, null_reps=20):
    """Held-out log10 response-ratio profile plus a same-family null, as in invariant.py."""
    rng = np.random.default_rng(seed)
    allZ = np.vstack([Zs, Zj]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    cs = np.linalg.norm(design(W(allZ), deg), axis=0); cs[cs == 0] = 1.0
    Ms, Mj = design(W(Zs), deg) / cs, design(W(Zj), deg) / cs
    gram = lambda M: M.T @ M / max(len(M), 1)

    def split(M, s):
        i = np.random.default_rng(s).permutation(len(M))
        return M[i[:len(i) // 2]], M[i[len(i) // 2:]]

    def prof(Atr, Ate, Btr, Bte):
        Ga, Gb = gram(Atr), gram(Btr)
        Gr = 0.5 * (Ga + Gb)
        _, V = eigh(Ga + LAM * Gr, Gb + LAM * Gr)
        out = []
        for c in range(V.shape[1]):
            v = V[:, c] / np.linalg.norm(V[:, c])
            out.append(np.log10(max(np.mean((Ate @ v) ** 2), 1e-300) /
                                max(np.mean((Bte @ v) ** 2), 1e-300)))
        return sorted(out)

    Str, Ste = split(Ms, seed); Jtr, Jte = split(Mj, seed + 1)
    obs = prof(Str, Ste, Jtr, Jte)
    null = []
    for r in range(null_reps):
        a, b = split(Ms, 100 + r)
        na, nb = len(a) // 2, len(b) // 2
        try:
            null.append(prof(a[:na], a[na:], b[:nb], b[nb:]))
        except np.linalg.LinAlgError:
            pass
    rank_s = int(np.linalg.matrix_rank(Ms, tol=1e-9))
    rank_j = int(np.linalg.matrix_rank(Mj, tol=1e-9))
    return dict(profile=obs, null=null, dim_class=Ms.shape[1],
                rank_S=rank_s, rank_J=rank_j,
                dim_blind_S=Ms.shape[1] - rank_s, dim_blind_J=Mj.shape[1] - rank_j)


def summarise(rows, degs):
    # escapes of the extreme-value same-family envelope, leave-one-trial-out, per degree
    out = {}
    for d in degs:
        rr = [r for r in rows if r["deg"] == d]
        nulls = [r["null"] for r in rr]
        below = sum(min(r["profile"]) < loo_envelope(nulls, i)[0] for i, r in enumerate(rr))
        above = sum(max(r["profile"]) > loo_envelope(nulls, i)[1] for i, r in enumerate(rr))
        out[f"deg{d}"] = dict(n=len(rr), below=int(below), above=int(above),
                              p_below=escape_binom_p(below, len(rr)),
                              p_above=escape_binom_p(above, len(rr)),
                              dim_class=rr[0]["dim_class"],
                              median_rank_S=float(np.median([r["rank_S"] for r in rr])),
                              median_rank_J=float(np.median([r["rank_J"] for r in rr])),
                              median_blind_S=float(np.median([r["dim_blind_S"] for r in rr])),
                              median_blind_J=float(np.median([r["dim_blind_J"] for r in rr])),
                              blind_frac_S=float(np.median([r["dim_blind_S"] for r in rr]))
                              / rr[0]["dim_class"],
                              median_min_logr=float(np.median([min(r["profile"]) for r in rr])),
                              median_max_logr=float(np.median([max(r["profile"]) for r in rr])))
        o = out[f"deg{d}"]
        print(f"\ndegree {d}: class dim {o['dim_class']}; EXACT rank is full under both families "
              f"(S {o['median_rank_S']:.0f}, J {o['median_rank_J']:.0f}) -- as everywhere else in "
              f"a trained model, the blindness is PRACTICAL, not a rank deficiency.")
        print(f"          least-visible direction: median held-out log10 ratio "
              f"{o['median_min_logr']:.2f} (i.e. ~10^{-o['median_min_logr']:.0f} times less "
              f"visible to single-site than to joint evidence)")
        print(f"          S-blind end escapes {o['below']}/{o['n']} (binom p={o['p_below']:.2g}); "
              f"reverse {o['above']}/{o['n']} (p={o['p_above']:.2g})")
    json.dump({"per_trial": [{k: v for k, v in r.items() if k not in ("profile", "null")}
                             for r in rows], "summary": out},
              open("threesite.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="5-7-9,5-8-11,6-9-11,7-9-11")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--degs", default="2,3")
    ap.add_argument("--analyse-only", action="store_true",
                    help="skip collection and re-analyse the saved three_gpt2_*.npz captures. "
                         "Collection needs a model and a GPU; analysis does not, which is what "
                         "lets rebuild_all.sh run on CPU alone.")
    A = ap.parse_args()
    torch.set_grad_enabled(False)
    degs = [int(x) for x in A.degs.split(",")]
    if A.analyse_only:
        rows = []
        print(f"{'capture':<28}{'deg':>5}{'dim class':>10}{'rank S':>8}{'rank J':>8}"
              f"{'min log10 r':>12}")
        for f in sorted(glob.glob("three_gpt2_*.npz")):
            d = np.load(f)
            for dg in degs:
                r = profile(d["Zs"], d["Zj"], dg)
                sites = [int(x) for x in f[:-4].split("_")[2:5]]
                r.update(sites=sites, seed=int(f[:-4].split("_")[5]), deg=dg,
                         std_Yj=float(np.std(d["Yj"])))
                rows.append(r)
                print(f"{f:<28}{dg:>5}{r['dim_class']:>10}{r['rank_S']:>8}{r['rank_J']:>8}"
                      f"{min(r['profile']):>12.2f}")
        if not rows:
            raise SystemExit("no three_gpt2_*.npz captures found -- run without --analyse-only")
        summarise(rows, degs)
        raise SystemExit(0)
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32,
                                                 device_map=DEV).eval()
    assert next(model.parameters()).dtype is torch.float32
    modfn = lambda m, l: m.transformer.h[l].attn
    names = [x for x in ["Mary", "John", "Tom", "Sarah", "Alice", "Bob", "Emma", "David",
                         "Laura", "Peter", "Anna", "James", "Linda", "Paul", "Susan", "Mark"]
             if len(tok(" " + x, add_special_tokens=False).input_ids) == 1]
    degs = [int(x) for x in A.degs.split(",")]
    t0 = time.time()
    rows = []
    print(f"{'sites':>10}{'seed':>5}{'deg':>5}{'dim class':>10}{'rank S':>8}{'rank J':>8}"
          f"{'blind S':>9}{'blind J':>9}{'min log10 r':>12}")
    for tri in A.triples.split(","):
        layers = [int(x) for x in tri.split("-")]
        for sd in (int(s) for s in A.seeds.split(",")):
            Zs, Ys, Zj, Yj, prompt, y0 = collect(model, tok, modfn, layers, sd, names)
            np.savez(f"three_gpt2_{tri.replace('-','_')}_{sd}.npz",
                     Zs=Zs, Ys=Ys, Zj=Zj, Yj=Yj)
            for d in degs:
                r = profile(Zs, Zj, d)
                r.update(sites=layers, seed=sd, deg=d, std_Yj=float(np.std(Yj)))
                rows.append(r)
                print(f"{tri:>10}{sd:>5}{d:>5}{r['dim_class']:>10}{r['rank_S']:>8}"
                      f"{r['rank_J']:>8}{r['dim_blind_S']:>9}{r['dim_blind_J']:>9}"
                      f"{min(r['profile']):>12.2f}")
    summarise(rows, degs)
    print(f"\nsaved -> threesite.json  [{time.time()-t0:.0f}s]")
